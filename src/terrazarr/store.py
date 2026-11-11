# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 MindEarth
"""Store helpers: obstore-backed zarr stores for local and s3:// paths, icechunk repositories,
and the CRS setup.

Credentials for s3:// paths come from the environment:
AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL (optional), AWS_REGION (optional).
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import rioxarray  # noqa: F401  # registers the .rio accessor
import xarray as xr
import zarr


def _s3_credentials() -> dict:
    return {
        "access_key": os.environ.get("AWS_ACCESS_KEY_ID", ""),
        "secret_key": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
        "endpoint": os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("MINIO_ENDPOINT"),
        "region": os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"),
    }


def get_obstore(path: str):
    """Return an obstore S3Store for an s3:// path (bucket + prefix)."""
    from obstore.store import S3Store

    parsed = urlparse(path)
    creds = _s3_credentials()
    kwargs = {
        "prefix": parsed.path.lstrip("/"),
        "virtual_hosted_style_request": False,
        "client_options": {"allow_http": True},
        "retry_config": {"max_retries": 25, "max_attempts": 25},
    }
    if creds["access_key"]:
        kwargs["access_key_id"] = creds["access_key"]
        kwargs["secret_access_key"] = creds["secret_key"]
    if creds["endpoint"]:
        kwargs["endpoint"] = creds["endpoint"]
    if creds["region"]:
        kwargs["region"] = creds["region"]
    return S3Store(parsed.netloc, **kwargs)


def get_zarr_store(input_path: str, s3_profile: str | None = None):
    """Zarr store for a local path or an s3:// URL (obstore-backed)."""
    if input_path.startswith("s3://"):
        return zarr.storage.ObjectStore(get_obstore(input_path))
    return zarr.storage.LocalStore(input_path)


# ---------------------------------------------------------------------------
# Icechunk (optional: pip install terrazarr[icechunk])
# ---------------------------------------------------------------------------


def _import_icechunk():
    try:
        import icechunk
    except ImportError as e:
        raise ImportError("Icechunk support needs the icechunk package: pip install 'terrazarr[icechunk]'") from e
    return icechunk


def is_icechunk(obj) -> bool:
    """True for an icechunk Repository, Session or store, without importing icechunk."""
    return type(obj).__module__.split(".")[0] == "icechunk"


def icechunk_storage(path: str):
    """Icechunk storage for a local path or an s3:// URL, credentials as for :func:`get_obstore`."""
    icechunk = _import_icechunk()
    if not path.startswith("s3://"):
        return icechunk.local_filesystem_storage(path)
    parsed = urlparse(path)
    creds = _s3_credentials()
    kwargs: dict = {"bucket": parsed.netloc, "prefix": parsed.path.strip("/") or None, "allow_http": True}
    if creds["access_key"]:
        kwargs["access_key_id"] = creds["access_key"]
        kwargs["secret_access_key"] = creds["secret_key"]
        kwargs["session_token"] = os.environ.get("AWS_SESSION_TOKEN") or None
    else:
        kwargs["from_env"] = True
    if creds["endpoint"]:
        # MinIO and other S3-compatible endpoints, path-style like get_obstore
        kwargs["endpoint_url"] = creds["endpoint"]
        kwargs["force_path_style"] = True
    if creds["region"]:
        kwargs["region"] = creds["region"]
    return icechunk.s3_storage(**kwargs)


def open_icechunk_repo(path: str, create: bool = False):
    """The icechunk Repository at a local path or s3:// URL, created if absent with ``create``."""
    icechunk = _import_icechunk()
    storage = icechunk_storage(path)
    if create:
        return icechunk.Repository.open_or_create(storage)
    return icechunk.Repository.open(storage)


def _with_split_manifests(repo, window_shards: int):
    """
    ``repo`` re-opened with manifests split every ``2 * window_shards`` chunks (shards, on a
    sharded array) per dim. A commit rewrites the manifests of the chunks it changed; unsplit,
    that is the whole array's, so commit time grows with the raster (22 ms per commit at 16k
    shards, 6 ms split). Windows start on multiples of ``window_shards``, so a level-0 window
    and its fused level-1 region each fall in one split. Applies to the writes of the returned
    Repository object, whatever configuration the repository was created with.
    """
    icechunk = _import_icechunk()
    config = repo.config
    config.manifest = icechunk.ManifestConfig(
        splitting=icechunk.ManifestSplittingConfig.from_dict(
            {icechunk.ManifestSplitCondition.AnyArray(): {icechunk.ManifestSplitDimCondition.Any(): 2 * window_shards}}
        )
    )
    return repo.reopen(config=config)


def icechunk_read_store(repo, ref: str = "main"):
    """
    Read-only store of a repository at ``ref``: a branch, a tag or a snapshot id. A read-only
    store pickles to dask workers as it is.
    """
    if ref in repo.list_branches():
        return repo.readonly_session(branch=ref).store
    if ref in repo.list_tags():
        return repo.readonly_session(tag=ref).store
    try:
        return repo.readonly_session(snapshot_id=ref).store
    except Exception as e:
        raise ValueError(f"{ref!r} is not a branch, a tag or a snapshot id of the repository") from e


def _identity(obj):
    return obj


@dataclass
class _Branch:
    """The state the views of one :class:`IcechunkBranchStore` share."""

    repo: Any
    name: str
    session: Any
    committed: tuple[str, Any] | None = None  # (snapshot id, its read-only store)


class IcechunkBranchStore(zarr.storage.WrapperStore):
    """
    A zarr store over a branch of an icechunk repository, written in commits.

    Reads and writes in this process go to the current writable session of the branch, so
    one store object serves the whole pipeline although every :meth:`commit` replaces the
    session. Pickled (to a dask worker) it becomes the read-only store of the last commit:
    workers read what was committed and cannot write through it. Distributed writes go
    through :meth:`fork` and :meth:`merge` instead.
    """

    def __init__(self, repo, branch: str = "main") -> None:
        if branch not in repo.list_branches():
            repo.create_branch(branch, repo.lookup_branch("main"))
        self._branch = _Branch(repo, branch, repo.writable_session(branch))
        self._read_only_view = False

    @property
    def branch(self) -> str:
        return self._branch.name

    @property
    def session(self):
        """The current writable session of the branch."""
        return self._branch.session

    @property
    def _store(self):  # WrapperStore delegates every call to this
        store = self.session.store
        return store.with_read_only(True) if self._read_only_view else store

    def with_read_only(self, read_only: bool = False) -> IcechunkBranchStore:
        # zarr opens a group or array with mode="r" through a read-only view of the store; the
        # view shares the branch, so it follows the session too. Not through __init__, which
        # would open a new session.
        view = object.__new__(IcechunkBranchStore)
        view._branch, view._read_only_view = self._branch, read_only
        return view

    def _with_store(self, store):
        # WrapperStore wraps the result of the inner store's __enter__ in a new wrapper; this
        # one has no store of its own to swap, the session is the inner store
        return self

    @property
    def supports_consolidated_metadata(self) -> bool:
        return False

    def commit(self, message: str) -> str | None:
        """Commit the changes made so far, if any, and open a new session; the snapshot id or None."""
        if not self.session.has_uncommitted_changes:
            return None
        snapshot = self.session.commit(f"terrazarr: {message}")
        self._branch.session = self._branch.repo.writable_session(self._branch.name)
        return snapshot

    def fork(self, message: str):
        """Commit the pending changes (a fork needs none) and fork the session for the workers."""
        self.commit(message)
        return self.session.fork()

    def merge(self, forks: Iterable) -> None:
        """Merge the fork sessions the workers returned into the session."""
        forks = list(forks)
        if forks:
            self.session.merge(*forks)

    def committed_store(self):
        """Read-only store at the snapshot the current session is based on (the last commit)."""
        snapshot = self.session.snapshot_id
        if self._branch.committed is None or self._branch.committed[0] != snapshot:
            self._branch.committed = (snapshot, self._branch.repo.readonly_session(snapshot_id=snapshot).store)
        return self._branch.committed[1]

    def __reduce__(self):
        # Pickled as the read-only store of the last commit, which unpickles as it is. Changes not
        # committed yet are not on the other side: what workers read must be committed first, and
        # they write through forks (a write through this copy fails, the store is read-only).
        # Pending changes are legitimate here: icechunk 2's to_icechunk keeps the metadata it
        # writes in the session while the workers read the levels below.
        return (_identity, (self.committed_store(),))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, IcechunkBranchStore) and other._branch is self._branch

    def __hash__(self) -> int:
        return id(self._branch)

    def __str__(self) -> str:
        return f"icechunk://{self.branch}@{self.session.snapshot_id}"

    def __repr__(self) -> str:
        return f"IcechunkBranchStore({self})"


def get_output_store(output, branch: str = "main", s3_profile: str | None = None, window_shards: int = 32):
    """
    The store :func:`terrazarr.to_geozarr` writes to: a zarr store for a local path or s3:// URL,
    an :class:`IcechunkBranchStore` on ``branch`` for an icechunk Repository, with manifests
    split for windows of ``window_shards``² shards (see :func:`_with_split_manifests`).
    """
    if not is_icechunk(output):
        return get_zarr_store(output, s3_profile)
    if not hasattr(output, "writable_session"):
        raise TypeError(
            f"An icechunk output is written as the Repository, not a {type(output).__name__}: "
            "terrazarr commits as it goes and opens its own sessions on `branch`"
        )
    return IcechunkBranchStore(_with_split_manifests(output, window_shards), branch)


def get_storage_options(path: str, s3_profile: str | None = None) -> dict:
    """fsspec storage options for xarray's path-based zarr writes (baseline only)."""
    if not path.startswith("s3://"):
        return {}
    creds = _s3_credentials()
    opts = {"key": creds["access_key"], "secret": creds["secret_key"]}
    if creds["endpoint"]:
        opts["client_kwargs"] = {"endpoint_url": creds["endpoint"]}
    return {"storage_options": opts}


def set_spatial_info(
    ds: xr.Dataset, input_crs: str = "EPSG:4326", x_dim: str = "x", y_dim: str = "y"
) -> xr.Dataset:
    """Ensure the dataset has spatial dims and a CRS (rioxarray)."""
    ds = ds.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim)
    if not ds.rio.crs:
        if "spatial_ref" in ds.data_vars:
            ds = ds.assign_coords({"spatial_ref": ds["spatial_ref"]})
        else:
            ds = ds.rio.write_crs(input_crs)
    for var in ds.data_vars:
        ds[var].encoding["grid_mapping"] = "spatial_ref"
    return ds
