# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mindearth
"""Store helpers vendored from me_geotools.utils.{zarr,s3}, without private dependencies.

Credentials for s3:// paths come from the environment:
AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL (optional), AWS_REGION (optional).
"""

from __future__ import annotations

import os
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
