# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 MindEarth
# Derived from eopf-geozarr (Development Seed for ESA), Copyright 2025 European Space Agency (ESA),
# Apache-2.0: https://github.com/EOPF-Explorer/data-model
"""
GeoZarr-spec 0.4 compliant conversion tools for EOPF datasets.

This module provides functions to convert EOPF datasets to GeoZarr-spec 0.4 compliant format
while maintaining native projections and using /2 downsampling logic.

Key compliance features:
- _ARRAY_DIMENSIONS attributes on all arrays
- CF standard names for all variables
- grid_mapping attributes referencing CF grid_mapping variables
- GeoTransform attributes in grid_mapping variables
- Native CRS preservation (no TMS reprojection)
- Proper multiscales metadata structure

Scaling design (see CHANGELOG.md):
- one zarr store object is used for every read, write and attribute update
- level N+1 is built from level N re-opened from the store, one dask block per shard
- the dask block is derived from the shard (or shard_size), never from the tile width
- shards never exceed one slice on non-spatial dims, so task memory is bounded by shard_size²
- level pixel size is exactly 2**level times the native pixel size
"""

import dataclasses
import os
import time
import uuid
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any

import dask
import dask.array as da
import numpy as np
import xarray as xr
import zarr
from affine import Affine
from pyproj import CRS
from rasterio.warp import calculate_default_transform
from zarr.codecs import BloscCodec
from zarr.core.sync import sync
from zarr.storage import StoreLike
from zarr.storage._common import make_store_path

from terrazarr import utils
from terrazarr.store import get_zarr_store, set_spatial_info
from terrazarr.types import (
    OverviewLevelJSON,
    StandardXCoordAttrsJSON,
    StandardYCoordAttrsJSON,
    TileMatrixJSON,
    TileMatrixLimitJSON,
    TileMatrixSetJSON,
    XarrayEncodingJSON,
)

# Official GeoZarr convention registrations (https://geozarr.org/conventions.html)
GEOZARR_CONVENTIONS: list[dict[str, str]] = [
    {
        "schema_url": "https://raw.githubusercontent.com/zarr-conventions/multiscales/refs/tags/v1/schema.json",
        "spec_url": "https://github.com/zarr-conventions/multiscales/blob/v1/README.md",
        "uuid": "d35379db-88df-4056-af3a-620245f8e347",
        "name": "multiscales",
        "description": "Multiscale layout of zarr datasets",
    },
    {
        "schema_url": "https://raw.githubusercontent.com/zarr-conventions/geo-proj/refs/tags/v1/schema.json",
        "spec_url": "https://github.com/zarr-conventions/geo-proj/blob/v1/README.md",
        "uuid": "f17cb550-5864-4468-aeb7-f3180cfb622f",
        "name": "proj",
        "description": "Coordinate reference system information for geospatial data",
    },
    {
        "schema_url": "https://raw.githubusercontent.com/zarr-conventions/spatial/refs/tags/v1/schema.json",
        "spec_url": "https://github.com/zarr-conventions/spatial/blob/v1/README.md",
        "uuid": "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4",
        "name": "spatial",
        "description": "Spatial coordinate information",
    },
]

SPATIAL_DIMS = ("y", "x")

FUSE_LEVEL_1 = os.environ.get("TERRAZARR_FUSE_LEVEL_1", "1") != "0"
"""Reduce level 1 from the level-0 blocks in the same compute as the level-0 write."""

MIN_BLOCKS_FROM_STORE = int(os.environ.get("TERRAZARR_MIN_BLOCKS_FROM_STORE", "16"))
# Level 0 (and the fused level 1) is written in windows of this many dask blocks per axis, one
# dask compute each, so the graph held by the client and scheduler is bounded by the window and
# not by the raster (about 40 KB per block task). Must be even so that a window of level-0 shards
# reduces onto whole level-1 shards.
WINDOW_SHARDS = int(os.environ.get("TERRAZARR_WINDOW_SHARDS", "32"))
# A non-spatial dim (bands, time) of at most this many slices that the source holds in a single
# block is kept whole in the dask block: one read, one transpose and one task write all slices,
# instead of a dask rechunk that splits the block slice by slice.
MAX_LEAD_PER_TASK = int(os.environ.get("TERRAZARR_MAX_LEAD_PER_TASK", "16"))
# Level-0 group attributes that record the windows written and the completion of the level.
WINDOWS_DONE_ATTR = "terrazarr:windows_done"
LEVEL_COMPLETE_ATTR = "terrazarr:level0_complete"
"""Below this many output shards, an overview is reduced from the in-memory parent blocks
(one task per parent shard) rather than one task per output shard, to keep parallelism."""


# ---------------------------------------------------------------------------
# Store-path helpers (one store object, group paths relative to its root)
# ---------------------------------------------------------------------------


def _group_path(group_name: str | None) -> str:
    """Normalize a group name ('/', '/a/b', 'a/b') to a store-relative path ('' for root)."""
    return (group_name or "").strip("/")


def _join_path(*parts: str | int) -> str:
    return "/".join(str(p).strip("/") for p in parts if str(p).strip("/"))


def _group_or_none(path: str) -> str | None:
    return path or None


def _node_exists(store: StoreLike, path: str) -> bool:
    """True if a zarr v3 node (group or array) exists at *path* in *store*."""
    key = _join_path(path, "zarr.json") if path else "zarr.json"
    try:
        return bool(sync(store.exists(key)))
    except Exception:
        return False


def _delete_prefix(store: StoreLike, path: str) -> None:
    """Remove every key under *path* (an array or group) from the store."""
    if not path:
        return
    try:
        sync(store.delete_dir(path))
    except FileNotFoundError:
        pass
    except Exception as cleanup_error:
        print(f"    ⚠️ Failed to remove {path}: {cleanup_error}")


def _dask_chunks_for(
    enc: Mapping[str, Any],
    dims: Sequence[Hashable],
    shard_size: int,
    source_chunks: Sequence[Sequence[int]] | None = None,
) -> dict[Hashable, int]:
    """
    Dask block for a variable, derived from its zarr encoding.

    Sharded: one block per shard, which zarr needs to write a shard in a single put
    instead of a read-modify-write. Unsharded: shard_size on y/x (a multiple of the
    zarr chunk, so no rechunk is needed to align). A non-spatial dim is one slice per
    block, unless ``source_chunks`` shows the source holds all its slices (at most
    ``MAX_LEAD_PER_TASK``) in one block, e.g. a band-last (y, x, band) image: then the
    block keeps them, so the source block is read and transposed once and one task
    writes every slice, instead of dask splitting it slice by slice.
    """
    shards = enc.get("shards")
    if shards:
        out = dict(zip(dims, shards))
    else:
        out = {d: (shard_size if d in SPATIAL_DIMS else 1) for d in dims}
    if source_chunks is not None:
        for d, c in zip(dims, source_chunks):
            if d not in SPATIAL_DIMS and len(c) == 1 and c[0] <= MAX_LEAD_PER_TASK:
                out[d] = c[0]
    return out


def _pin_grid_mapping(ds: xr.Dataset) -> xr.Dataset:
    """
    Make the CF ``grid_mapping`` reference survive the write.

    rioxarray keeps ``grid_mapping`` in ``encoding``; an explicit ``encoding=`` passed to
    ``to_zarr`` replaces a variable's own encoding, so the attribute would be lost and the
    written level could not be decoded with a CRS. Move it to ``attrs`` for every data
    variable, and drop it from ``encoding`` so xarray does not see it twice.
    """
    for var in ds.data_vars:
        v = ds[var]
        name = v.encoding.get("grid_mapping") or v.attrs.get("grid_mapping")
        if name and name in ds:
            v.attrs["grid_mapping"] = name
        v.encoding.pop("grid_mapping", None)
    return ds


DEFAULT_COMPRESSOR = object()
"""Sentinel: Blosc zstd level 3 with byte shuffle. Pass ``None`` for no compression."""


def make_compressor(name: str = "zstd", clevel: int = 3) -> BloscCodec | None:
    """Blosc codec for the data variables, or None for uncompressed."""
    if name in {"none", "None", ""}:
        return None
    return BloscCodec(cname=name, clevel=clevel, shuffle="shuffle", blocksize=0)


@dataclasses.dataclass
class _PyramidGeometry:
    data_vars: list[Hashable]
    native_width: int
    native_height: int
    native_crs: Any
    native_bounds: tuple[float, float, float, float]
    native_px: tuple[float, float]
    overview_levels: list[OverviewLevelJSON]


def _chunks_along(size: int, chunk: int) -> tuple[int, ...]:
    full, rest = divmod(size, chunk)
    return (chunk,) * full + ((rest,) if rest else ())


_ARRAY_CACHE: dict[tuple[str, str, str], zarr.Array] = {}


def _open_array_cached(store: StoreLike, path: str, run_token: str) -> zarr.Array:
    """Open a written level array once per process and pipeline run (one metadata GET)."""
    key = (run_token, repr(store), path)
    arr = _ARRAY_CACHE.get(key)
    if arr is None:
        if len(_ARRAY_CACHE) > 256:
            _ARRAY_CACHE.clear()
        arr = zarr.open_array(store, path=path, mode="r", zarr_format=3)
        _ARRAY_CACHE[key] = arr
    return arr


def _read_reduce_block(
    block_id: tuple[int, ...] | None = None,
    *,
    store: StoreLike,
    path: str,
    run_token: str,
    level_shape: tuple[int, ...],
    shard_size: int,
    fy: int,
    fx: int,
    method: str,
    nodata_value: float | None,
    out_dtype: Any,
) -> np.ndarray:
    """
    One overview shard from its parent shards, read from the store one at a time.

    Task memory is one parent shard plus the output block, whatever the level size, and a
    parent shard that holds no valid pixel costs one GET and a scan.
    """
    assert block_id is not None
    arr = _open_array_cached(store, path, run_token)
    nlead = len(level_shape) - 2
    by, bx = block_id[-2:]
    y0, x0 = by * shard_size, bx * shard_size
    th = min(shard_size, level_shape[-2] - y0)
    tw = min(shard_size, level_shape[-1] - x0)
    out = np.empty((1,) * nlead + (th, tw), dtype=out_dtype)
    lead_index = tuple(int(i) for i in block_id[:nlead])
    step_y, step_x = max(1, shard_size // fy), max(1, shard_size // fx)
    for qy in range(0, th, step_y):
        oh = min(step_y, th - qy)
        for qx in range(0, tw, step_x):
            ow = min(step_x, tw - qx)
            py0, px0 = (y0 + qy) * fy, (x0 + qx) * fx
            sub = arr[lead_index + (slice(py0, py0 + oh * fy), slice(px0, px0 + ow * fx))]
            out[(0,) * nlead + (slice(qy, qy + oh), slice(qx, qx + ow))] = utils.reduce_block(
                sub, fy, fx, method, nodata_value, out_dtype
            )
    return out


def _all_fill(block: np.ndarray, fill_value: Any) -> bool:
    """True if every element of ``block`` equals the array's fill value (NaN-aware)."""
    if fill_value is None:
        return False
    if isinstance(fill_value, float) and np.isnan(fill_value):
        return bool(np.isnan(block).all())
    return bool((block == fill_value).all())


def _block_slices(block_info: Any, offset: Sequence[int]) -> tuple[slice, ...]:
    """Slices of a map_blocks block in the target array, shifted by the window ``offset``."""
    loc = block_info[0]["array-location"]
    return tuple(slice(a + o, b + o) for (a, b), o in zip(loc, offset))


def _write_block(
    block: np.ndarray, arr: zarr.Array, offset: Sequence[int], block_info: Any = None
) -> np.ndarray:
    """Write one level-0 block; an all-fill block is skipped (zarr would store nothing)."""
    block = np.ascontiguousarray(block)
    if not _all_fill(block, arr.fill_value):
        arr[_block_slices(block_info, offset)] = block
    return np.ones((1,) * block.ndim, dtype=bool)


def _write_and_reduce(
    block: np.ndarray,
    arr: zarr.Array,
    fy: int,
    fx: int,
    method: str,
    nodata_value: float | None,
    out_dtype: Any,
    offset: Sequence[int] = (),
    block_info: Any = None,
) -> np.ndarray:
    """
    Write one level-0 block and return its reduction for level 1: one read per block.

    The block is made contiguous once (a band-last source arrives as a strided transpose),
    and an all-fill block skips the zarr write altogether: zarr stores nothing for it but
    would still walk every inner chunk of the shard to find that out.
    """
    block = np.ascontiguousarray(block)
    if not _all_fill(block, arr.fill_value):
        arr[_block_slices(block_info, offset)] = block
    return utils.reduce_block(block, fy, fx, method, nodata_value, out_dtype)


def _fused_write_reduce_array(
    data: da.Array,
    store: StoreLike,
    array_path: str,
    target_height: int,
    target_width: int,
    method: str,
    nodata_value: float | None,
    offset: Sequence[int] | None = None,
    arr: zarr.Array | None = None,
) -> da.Array:
    """
    Lazy level-1 array whose tasks also write the level-0 blocks of ``data`` to the array
    already created at ``array_path``. Dask fuses a block read into its first consumer, so
    two consumers of the input would read it twice; this keeps one read per block.
    ``data`` may be a window of the level-0 array starting at ``offset``.
    """
    if arr is None:
        arr = zarr.open_array(store, path=array_path, mode="r+", zarr_format=3)
    src_h, src_w = data.shape[-2:]
    fy, fx = src_h // target_height, src_w // target_width
    out_chunks = (
        *data.chunks[:-2],
        tuple(c // fy for c in data.chunks[-2]),
        tuple(c // fx for c in data.chunks[-1]),
    )
    out = da.map_blocks(
        _write_and_reduce,
        data,
        arr=arr,
        fy=fy,
        fx=fx,
        method=method,
        nodata_value=nodata_value,
        out_dtype=data.dtype,
        offset=tuple(offset) if offset is not None else (0,) * data.ndim,
        dtype=data.dtype,
        chunks=out_chunks,
        meta=np.empty((0,) * data.ndim, dtype=data.dtype),
    )
    if out.shape[-2] != target_height or out.shape[-1] != target_width:
        out = out[..., :target_height, :target_width]
    return out


def _level0_windows(
    chunks_y: Sequence[int], chunks_x: Sequence[int], window_shards: int
) -> list[tuple[tuple[int, int], slice, slice]]:
    """
    Windows of at most ``window_shards`` dask blocks per axis over a block grid:
    ``((wy, wx), y_slice, x_slice)`` in pixel coordinates, row-major.
    """
    oy = np.concatenate([[0], np.cumsum(chunks_y)])
    ox = np.concatenate([[0], np.cumsum(chunks_x)])
    out = []
    for wy, i0 in enumerate(range(0, len(chunks_y), window_shards)):
        i1 = min(i0 + window_shards, len(chunks_y))
        for wx, j0 in enumerate(range(0, len(chunks_x), window_shards)):
            j1 = min(j0 + window_shards, len(chunks_x))
            out.append(((wy, wx), slice(int(oy[i0]), int(oy[i1])), slice(int(ox[j0]), int(ox[j1]))))
    return out


def _read_window_marker(store: StoreLike, group_path: str) -> tuple[set[tuple[int, int]], bool, int | None]:
    """Windows already written to the level-0 group, its completion flag and the window size used."""
    try:
        attrs = zarr.open_group(store, path=_group_or_none(group_path), mode="r", use_consolidated=False).attrs
        done = {tuple(w) for w in attrs.get(WINDOWS_DONE_ATTR, {}).get("windows", [])}
        size = attrs.get(WINDOWS_DONE_ATTR, {}).get("window_shards")
        return done, bool(attrs.get(LEVEL_COMPLETE_ATTR, False)), size
    except Exception:
        return set(), False, None


def _write_window_marker(
    store: StoreLike, group_path: str, done: set[tuple[int, int]], window_shards: int, complete: bool
) -> None:
    g = zarr.open_group(store, path=_group_or_none(group_path), mode="r+", use_consolidated=False)
    g.attrs.update(
        {
            WINDOWS_DONE_ATTR: {"window_shards": window_shards, "windows": sorted(list(w) for w in done)},
            LEVEL_COMPLETE_ATTR: complete,
        }
    )


def _overview_arrays_from_store(
    store: StoreLike,
    parent_path: str,
    parent_ds: xr.Dataset,
    data_vars: Sequence[Hashable],
    height: int,
    width: int,
    shard_size: int,
    method: str,
    nodata_value: float | None,
    run_token: str,
) -> dict[Hashable, da.Array]:
    """Lazy overview arrays, one task per output shard (see :func:`_read_reduce_block`)."""
    arrays: dict[Hashable, da.Array] = {}
    for var in data_vars:
        pv = parent_ds[var]
        lead_dims = [d for d in pv.dims if d not in SPATIAL_DIMS]
        pv = pv.transpose(*lead_dims, *SPATIAL_DIMS)
        ph, pw = pv.shape[-2:]
        lead_shape = tuple(int(n) for n in pv.shape[:-2])
        chunks = (
            *[(1,) * n for n in lead_shape],
            _chunks_along(height, shard_size),
            _chunks_along(width, shard_size),
        )
        arrays[var] = da.map_blocks(
            _read_reduce_block,
            chunks=chunks,
            dtype=pv.dtype,
            store=store,
            path=_join_path(parent_path, str(var)),
            run_token=run_token,
            level_shape=(*lead_shape, height, width),
            shard_size=shard_size,
            fy=ph // height,
            fx=pw // width,
            method=method,
            nodata_value=nodata_value,
            out_dtype=pv.dtype,
        )
    return arrays


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def to_geozarr(
    data: xr.DataArray | xr.Dataset | xr.DataTree,
    output: str,
    *,
    groups: Iterable[str] | None = None,
    chunk_size: int = 256,
    shard_size: int = 4096,
    sharding: bool = True,
    method: str = "mean",
    nodata: float | None = None,
    min_dimension: int | None = None,
    compressor: Any = DEFAULT_COMPRESSOR,
    window_shards: int | None = None,
    max_retries: int = 3,
    crs_groups: Iterable[str] | None = None,
    gcp_group: str | None = None,
    store: StoreLike | None = None,
    s3_profile: str | None = None,
) -> xr.DataTree:
    """
    Write an xarray object as a GeoZarr multiscale pyramid.

    The counterpart of ``rioxarray``'s ``to_raster`` for GeoZarr: ``data`` becomes level 0 of
    a zarr v3 store and every further level is its ``2**L`` reduction, with CRS, transform
    and multiscales metadata on each level. ``data`` must have ``y``/``x`` dimensions and a
    CRS readable by rioxarray (``ds.rio.crs``); a dask-backed object is written lazily, in
    windows, with bounded memory whatever its size. Also available as the accessor
    ``ds.terrazarr.to_geozarr(output, ...)``.

    Parameters
    ----------
    data : xarray.DataArray, xarray.Dataset or xarray.DataTree
        A DataArray is written as the variable ``data`` (or its name); a Dataset as the root
        group; a DataTree group by group (see ``groups``).
    output : str
        Output store: a local path or an ``s3://`` URL (credentials from the environment).
    groups : iterable of str, optional
        For a DataTree, the groups to write as pyramids; default the root when it holds data
        variables, otherwise every group that does.
    chunk_size : int, default 256
        Zarr chunk on y/x, the tile a client fetches.
    shard_size : int, default 4096
        Zarr shard on y/x with ``sharding``, i.e. how many chunks travel in one object, and the
        dask block on y/x in every case; a multiple of ``chunk_size``.
    sharding : bool, default True
        Write sharded arrays (one object per shard) instead of one object per chunk.
    method : {"mean", "min", "max", "median", "nearest"}, default "mean"
        Reduction from one level to the next.
    nodata : float, optional
        Numeric nodata value, excluded from the reduction; NaN is always nodata. A block with
        less than 30 % valid pixels becomes nodata.
    min_dimension : int, optional
        Stop adding levels when both dimensions are below this; default ``chunk_size``.
    compressor : zarr codec, optional
        Codec for every level's data variables, see :func:`make_compressor`; default Blosc
        zstd level 3 with byte shuffle, ``None`` for no compression.
    window_shards : int, optional
        Level 0 is written in windows of this many blocks per axis, one dask compute each
        (default 32). Bounds the memory of the process that holds the dask graph.
    max_retries : int, default 3
        Attempts per window, and per band in the fallback path.
    crs_groups, gcp_group : optional
        Groups that need CRS information on a best-effort basis, and the group holding
        ground control points (Sentinel-1 inputs).
    store : zarr store, optional
        Store to write to, built from ``output`` when omitted.
    s3_profile : str, optional
        Kept for compatibility; credentials come from the environment.

    Returns
    -------
    xarray.DataTree
        The written pyramid, one child per level.
    """
    if isinstance(data, xr.DataArray):
        data = data.to_dataset(name=data.name or "data")
    if isinstance(data, xr.Dataset):
        data = xr.DataTree(data)
        groups = ["/"]
    if groups is None:
        groups = ["/"] if data.data_vars else [p for p, node in data.subtree_with_keys if node.data_vars]
        groups = ["/" + g.lstrip("./") if g not in (".", "/") else "/" for g in groups]
    dt_input, output_path, enable_sharding, nodata_value = data, output, sharding, nodata
    if min_dimension is None:
        min_dimension = chunk_size
    if chunk_size <= 0 or shard_size <= 0:
        raise ValueError("chunk_size and shard_size must be positive")
    if shard_size % chunk_size:
        raise ValueError(
            f"shard_size ({shard_size}) must be a multiple of chunk_size ({chunk_size})"
        )

    dt = dt_input.copy()
    if compressor is DEFAULT_COMPRESSOR:
        compressor = make_compressor()
    if store is None:
        store = get_zarr_store(output_path, s3_profile)

    if enable_sharding:
        print("🔧 Zarr sharding enabled for spatial dimensions")

    # Get the measurements datasets prepared for GeoZarr compliance
    geozarr_groups = setup_datatree_metadata_geozarr_spec_compliant(
        dt, groups, gcp_group
    )

    # Create the GeoZarr compliant store through iterative processing
    dt_geozarr = iterative_copy(
        dt,
        geozarr_groups,
        store,
        compressor,
        shard_size,
        min_dimension,
        chunk_size,
        max_retries,
        crs_groups,
        gcp_group,
        enable_sharding,
        method=method,
        nodata_value=nodata_value,
        window_shards=window_shards,
    )

    # Consolidate metadata at the root level AFTER all groups are written
    print("Consolidating metadata at root level for consistent zarr access...")
    consolidate_metadata(store)
    print("✅ Root level metadata consolidation completed")

    return dt_geozarr


def setup_datatree_metadata_geozarr_spec_compliant(
    dt: xr.DataTree, groups: Iterable[str], gcp_group: str | None = None
) -> dict[str, xr.Dataset]:
    """
    Set up GeoZarr-spec compliant CF standard names and CRS information.

    Parameters
    ----------
    dt : xr.DataTree
        The data tree containing the datasets to process
    groups : list[str]
        List of group names to process as Geozarr datasets

    Returns
    -------
    dict[str, xr.Dataset]
        Dictionary of datasets with GeoZarr compliance applied
    """
    geozarr_groups: dict[str, xr.Dataset] = {}
    grid_mapping_var_name = "spatial_ref"
    epsg_CPM_260 = dt.attrs.get("other_metadata", {}).get("horizontal_CRS_code", None)
    if epsg_CPM_260 is not None:
        epsg_CPM_260 = epsg_CPM_260.split(":")[-1]

    for key in groups:
        node = dt if key == "/" else dt[key]
        if not node.data_vars:
            continue

        print(f"Processing group for GeoZarr compliance: {key}")
        ds = node.to_dataset().copy()

        # Process all variables in the group
        for var_name in ds.data_vars:
            print(f"  Processing variable / band: {var_name}")

            # Set CF standard name and _ARRAY_DIMENSIONS
            if _is_sentinel1(dt):
                ds[var_name].attrs[
                    "standard_name"
                ] = "surface_backwards_scattering_coefficient_of_radar_wave"
                ds[var_name].attrs["units"] = "1"

            if hasattr(ds[var_name], "dims"):
                ds[var_name].attrs["_ARRAY_DIMENSIONS"] = list(ds[var_name].dims)
            ds[var_name].attrs["grid_mapping"] = grid_mapping_var_name

            # Set CRS if available
            if "proj:epsg" in ds[var_name].attrs:
                epsg = ds[var_name].attrs["proj:epsg"]
                print(f"    Setting CRS for {var_name} to EPSG:{epsg}")
                ds = ds.rio.write_crs(f"epsg:{epsg}")
            elif epsg_CPM_260:
                print(
                    f"    Setting CRS for {var_name} to EPSG:{epsg_CPM_260} (CPM 2.6.0 default)"
                )
                ds = ds.rio.write_crs(f"epsg:{epsg_CPM_260}")

        # Add _ARRAY_DIMENSIONS to coordinate variables
        _add_coordinate_metadata(ds)

        # Set up spatial_ref variable with GeoZarr required attributes
        _setup_grid_mapping(ds, grid_mapping_var_name)

        geozarr_groups[key] = ds

    return geozarr_groups


def iterative_copy(
    dt_input: xr.DataTree,
    geozarr_groups: dict[str, xr.Dataset],
    store: StoreLike,
    compressor: Any,
    shard_size: int = 4096,
    min_dimension: int = 256,
    chunk_size: int = 256,
    max_retries: int = 3,
    crs_groups: Iterable[str] | None = None,
    gcp_group: str | None = None,
    enable_sharding: bool = False,
    method: str = "mean",
    nodata_value: float | None = None,
    window_shards: int | None = None,
) -> xr.DataTree:
    """
    Iteratively copy groups from original DataTree to GeoZarr DataTree.

    Parameters
    ----------
    dt_input : xarray.DataTree
        Input DataTree to copy from
    geozarr_groups : dict[str, xr.Dataset]
        Dictionary of GeoZarr groups to process
    store : zarr store
        Output store
    compressor : Any
        Compressor to use for encoding
    shard_size, min_dimension, chunk_size, max_retries, crs_groups, gcp_group,
    enable_sharding, method, nodata_value
        See :func:`to_geozarr`

    Returns
    -------
    xarray.DataTree
        Updated GeoZarr DataTree with copied groups and variables including multiscale children
    """
    # Create result DataTree and initialize the root group of the store
    dt_result = xr.DataTree()
    zarr.open_group(store, mode="a", zarr_format=3)

    written_groups: set[str] = set()
    reference_crs = None

    common_kwargs = dict(
        shard_size=shard_size,
        compressor=compressor,
        max_retries=max_retries,
        min_dimension=min_dimension,
        chunk_size=chunk_size,
        gcp_group=gcp_group,
        enable_sharding=enable_sharding,
        method=method,
        nodata_value=nodata_value,
        window_shards=window_shards,
    )

    # Process all groups in the tree using iterative approach
    for relative_path, node in dt_input.subtree_with_keys:
        if relative_path == ".":
            if "/" in geozarr_groups:
                print("Processing '/' as GeoZarr root group")
                write_geozarr_group(
                    dt_input, dt_result, "/", geozarr_groups["/"], store, **common_kwargs
                )
                written_groups.add("/")
            continue
        if relative_path.endswith("_VH") or relative_path.endswith("_VV"):
            # skip sentinel-1 top-level polarization groups
            continue

        current_group_path = "/" + relative_path
        print(f"Processing group '{current_group_path}' in iterative copy")

        if current_group_path in geozarr_groups:
            print(f"Processing '{current_group_path}' as GeoZarr group")
            write_geozarr_group(
                dt_input,
                dt_result,
                current_group_path,
                geozarr_groups[current_group_path],
                store,
                **common_kwargs,
            )
            written_groups.add(current_group_path)
            continue

        # Get dataset from the node
        ds = node.to_dataset().drop_encoding()

        # Add CRS information if needed
        if crs_groups and current_group_path in crs_groups:
            print(f"Adding CRS information for group '{current_group_path}'")
            if reference_crs is None:
                reference_crs = _find_reference_crs(geozarr_groups)
            ds = prepare_dataset_with_crs_info(ds, reference_crs=reference_crs)

        # Process groups with data variables
        if node.data_vars:
            print(
                f"Writing group '{current_group_path}' with data variables to GeoZarr DataTree"
            )
            encoding = _create_encoding(ds, compressor, shard_size)
            _pin_grid_mapping(ds)
            ds.to_zarr(
                store,
                group=_group_or_none(_group_path(current_group_path)),
                mode="w",
                consolidated=False,
                zarr_format=3,
                encoding=encoding,
            )
            dt_result[relative_path] = xr.DataTree(ds)

        written_groups.add(current_group_path)

    return dt_result


def prepare_dataset_with_crs_info(
    ds: xr.Dataset, reference_crs: str | None = None
) -> xr.Dataset:
    """
    Prepare a dataset with CRS information without writing it to disk.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to prepare with CRS information
    reference_crs : str, optional
        Reference CRS to use (e.g., "epsg:4326")

    Returns
    -------
    xr.Dataset
        Dataset with CRS information added
    """
    ds = ds.copy()

    # Set up coordinate variables with proper attributes
    _add_coordinate_metadata(ds)

    # Add CRS information if we have spatial coordinates and a reference CRS
    if "x" in ds.coords and "y" in ds.coords and reference_crs:
        print(f"  Adding CRS information: {reference_crs}")
        ds = ds.rio.write_crs(reference_crs)
        ds.attrs["grid_mapping"] = "spatial_ref"

        # Ensure spatial_ref variable has proper attributes
        if "spatial_ref" in ds:
            _add_geotransform(ds, "spatial_ref")

    # Set up data variables with proper attributes
    for var_name in ds.data_vars:
        if "_ARRAY_DIMENSIONS" not in ds[var_name].attrs and hasattr(
            ds[var_name], "dims"
        ):
            ds[var_name].attrs["_ARRAY_DIMENSIONS"] = list(ds[var_name].dims)

        # Add grid_mapping reference if spatial coordinates are present
        if "x" in ds[var_name].coords and "y" in ds[var_name].coords and reference_crs:
            ds[var_name].attrs["grid_mapping"] = "spatial_ref"
            ds[var_name].attrs["proj:epsg"] = reference_crs.split(":")[-1]
            if "spatial_ref" in ds and "GeoTransform" in ds["spatial_ref"].attrs:
                ds[var_name].attrs["proj:transform"] = ds["spatial_ref"].attrs[
                    "GeoTransform"
                ]

    return ds


def write_geozarr_group(
    dt_input: xr.DataTree,
    dt_result: xr.DataTree,
    group_name: str,
    ds: xr.Dataset,
    store: StoreLike,
    shard_size: int = 4096,
    compressor: Any = None,
    max_retries: int = 3,
    min_dimension: int = 256,
    chunk_size: int = 256,
    gcp_group: str | None = None,
    enable_sharding: bool = False,
    method: str = "mean",
    nodata_value: float | None = None,
    window_shards: int | None = None,
) -> xr.DataTree:
    """
    Write a group to a GeoZarr dataset with multiscales support.

    Level 0 is written band by band (resumable), then every overview level is derived
    from the previous one re-opened from the store. A failure in the pyramid is raised,
    never swallowed, so a partial store is never reported as success.

    Returns
    -------
    xarray.DataTree
        The written GeoZarr DataTree with multiscale groups as children
    """
    print(f"\n=== Processing {group_name} with GeoZarr-spec compliance ===")

    # Normalize to north-up (descending y) convention.
    ds = _normalize_north_up(ds)

    for var in ds.data_vars:
        if set(SPATIAL_DIMS).issubset(ds[var].dims):
            leading_dims = [d for d in ds[var].dims if d not in SPATIAL_DIMS]
            ds[var] = ds[var].transpose(*leading_dims, "y", "x")
            ds[var].attrs["_ARRAY_DIMENSIONS"] = list(ds[var].dims)

    # Create a new container for the group
    dt = xr.DataTree()
    group_path = _group_path(group_name)
    if group_path:
        dt_result[group_path] = dt
    dt.attrs = ds.attrs.copy()

    # Create encoding for all variables
    encoding = _create_geozarr_encoding(
        ds, compressor, chunk_size, shard_size, enable_sharding
    )

    _data_vars_to_check = [
        var for var in ds.data_vars if not utils.is_grid_mapping_variable(ds, var)
    ]

    # One dask block per shard for every data variable: the level-0 write and the fused
    # level-1 reduction below share these blocks.
    for var in _data_vars_to_check:
        if isinstance(ds[var].data, da.Array):
            ds[var] = ds[var].chunk(
                _dask_chunks_for(
                    encoding.get(var, {}), ds[var].dims, shard_size, ds[var].data.chunks
                )
            )

    geometry = _pyramid_geometry(ds, group_name, min_dimension, chunk_size, None)

    # Write native data in the group 0 (overview level 0)
    level0_path = _join_path(group_path, 0)
    existing_native_dataset = _load_existing_dataset(store, level0_path, shard_size)

    if (
        _is_level_complete(existing_native_dataset, _data_vars_to_check)
        and _is_level_valid(existing_native_dataset)
        and _read_window_marker(store, level0_path)[1]
    ):
        print(
            f"Level 0 already exists and is complete at {level0_path}, "
            "loading from store and skipping write..."
        )
        ds = _normalize_north_up(existing_native_dataset)
    else:
        # Level 1 is reduced from the level-0 blocks while they are in memory, so the
        # largest level is read and decoded once instead of twice.
        fused_level = None
        if FUSE_LEVEL_1 and geometry is not None and len(geometry.overview_levels) > 1:
            # Level 0 is incomplete, so a level 1 left by an interrupted run is only as
            # complete as level 0's windows: the windowed writer keeps its regions of the
            # windows done and writes the rest, whatever the group looks like.
            level1_path = _join_path(group_path, 1)
            if True:
                l1 = geometry.overview_levels[1]
                l1_h, l1_w = int(l1["height"]), int(l1["width"])
                g = geometry

                def _build_level1(
                    arrays: Mapping[Hashable, da.Array], ds=ds, g=g
                ) -> tuple[str, xr.Dataset, dict[Hashable, XarrayEncodingJSON]]:
                    ov = create_overview_dataset_all_vars(
                        ds,
                        1,
                        l1_w,
                        l1_h,
                        g.native_crs,
                        g.native_bounds,
                        g.data_vars,
                        None,
                        enable_sharding,
                        method=method,
                        nodata_value=nodata_value,
                        native_px=g.native_px,
                        data_arrays=arrays,
                    )
                    ov, enc1 = _prepare_overview_for_write(
                        ov, 1, shard_size, compressor, chunk_size, enable_sharding
                    )
                    return level1_path, ov, enc1

                fused_level = (
                    level1_path,
                    {"height": l1_h, "width": l1_w, "method": method, "nodata_value": nodata_value},
                    _build_level1,
                )
        success, ds = write_dataset_band_by_band_with_validation(
            ds,
            existing_native_dataset,
            store,
            encoding,
            max_retries,
            level0_path,
            shard_size,
            False,
            fused_level=fused_level,
            window_shards=window_shards,
        )
        if not success:
            raise RuntimeError(f"Failed to write all bands for {group_name}")

    # Validate level 0 orientation and rasterio consistency before building the pyramid.
    _validate_pyramid_level(ds, f"{group_name}/0")

    # Sentinel-1: data is reprojected upstream, GCPs are not needed for the pyramid
    if _is_sentinel1(dt_input):
        assert gcp_group is not None, "GCP group required for processing Sentinel-1"
    ds_gcp = None

    print(f"Creating GeoZarr-spec compliant multiscales for {group_name}")
    try:
        create_geozarr_compliant_multiscales(
            ds=ds,
            store=store,
            group_name=group_name,
            min_dimension=min_dimension,
            chunk_size=chunk_size,
            shard_size=shard_size,
            ds_gcp=ds_gcp,
            enable_sharding=enable_sharding,
            method=method,
            nodata_value=nodata_value,
            compressor=compressor,
            geometry=geometry,
        )
    except Exception as e:
        print(f"❌ Failed to create multiscales for {group_name}: {e}")
        raise

    # Consolidate metadata for the group
    print(f"  Consolidating metadata for group {group_name}...")
    consolidate_metadata(store, path=_group_or_none(group_path))
    print("  ✅ Metadata consolidated")

    return dt


def _pyramid_geometry(
    ds: xr.Dataset,
    group_name: str,
    min_dimension: int,
    chunk_size: int,
    ds_gcp: xr.Dataset | None,
) -> _PyramidGeometry | None:
    """Native extent, pixel size and the overview levels of a group; None without data."""
    # Get spatial information from the first data variable
    data_vars = [
        var for var in ds.data_vars if not utils.is_grid_mapping_variable(ds, var)
    ]
    if not data_vars:
        return None

    first_var = data_vars[0]
    native_height, native_width = ds[first_var].shape[-2:]
    native_crs = ds.rio.crs

    if ds_gcp is None:
        native_bounds = ds.rio.bounds(recalc=True)
    else:
        if "azimuth_time" in ds.dims and "ground_range" in ds.dims:
            ds.rio.set_spatial_dims(
                x_dim="ground_range", y_dim="azimuth_time", inplace=True
            )
        try:
            if ds.rio.get_gcps() is not None:
                transform, width, height = calculate_default_transform(
                    ds.rio.crs,
                    CRS.from_epsg(4326),
                    ds.rio.width,
                    ds.rio.height,
                    gcps=ds.rio.get_gcps(),
                )
                native_bounds = (
                    transform[2],
                    transform[5] + height * transform[4],
                    transform[2] + width * transform[0],
                    transform[5],
                )
            else:
                native_bounds = ds.rio.bounds(recalc=True)
        except Exception as e:
            print(f"Error computing native bounds: {e}")
            native_bounds = (
                ds_gcp["longitude"].values.min(),
                ds_gcp["latitude"].values.min(),
                ds_gcp["longitude"].values.max(),
                ds_gcp["latitude"].values.max(),
            )

    print(f"Creating GeoZarr-compliant multiscales for {group_name}")
    print(f"Native resolution: {native_width} x {native_height}")
    print(f"Native CRS: {native_crs}")

    left, bottom, right, top = native_bounds
    native_px_w = (right - left) / native_width
    native_px_h = (top - bottom) / native_height
    native_px = (native_px_w, native_px_h)

    # Calculate overview levels
    overview_levels = calculate_overview_levels(
        native_width, native_height, min_dimension, chunk_size
    )

    return _PyramidGeometry(
        data_vars=data_vars,
        native_width=native_width,
        native_height=native_height,
        native_crs=native_crs,
        native_bounds=native_bounds,
        native_px=native_px,
        overview_levels=overview_levels,
    )


def _prepare_overview_for_write(
    overview_ds: xr.Dataset,
    level: int,
    shard_size: int,
    compressor: Any,
    chunk_size: int,
    enable_sharding: bool,
) -> tuple[xr.Dataset, dict[Hashable, XarrayEncodingJSON]]:
    """One block per shard, encoding, north-up check, transform and grid_mapping pinned."""
    chunks: dict[Hashable, int] = {"x": shard_size, "y": shard_size}
    chunks.update({dim: 1 for dim in overview_ds.dims if dim not in SPATIAL_DIMS})
    overview_ds = overview_ds.chunk(chunks)

    encoding = _create_geozarr_encoding(
        overview_ds, compressor, chunk_size, shard_size, enable_sharding
    )

    # Ensure north-up / west-east convention before writing and update the
    # GeoTransform in spatial_ref so rasterio sees consistent bounds+transform.
    if "y" in overview_ds.coords and len(overview_ds.coords["y"]) > 1:
        if float(overview_ds.coords["y"].values[0]) < float(
            overview_ds.coords["y"].values[-1]
        ):
            overview_ds = overview_ds.isel(y=slice(None, None, -1))
    if "spatial_ref" in overview_ds:
        _ov_tf = overview_ds.rio.transform(recalc=True)
        overview_ds.rio.write_transform(_ov_tf, grid_mapping_name="spatial_ref", inplace=True)
    _validate_pyramid_level(overview_ds, level)
    _pin_grid_mapping(overview_ds)
    return overview_ds, encoding


def create_geozarr_compliant_multiscales(
    ds: xr.Dataset,
    store: StoreLike,
    group_name: str,
    min_dimension: int = 256,
    chunk_size: int = 256,
    shard_size: int = 4096,
    ds_gcp: xr.Dataset | None = None,
    enable_sharding: bool = False,
    method: str = "mean",
    nodata_value: float | None = None,
    compressor: Any = DEFAULT_COMPRESSOR,
    geometry: _PyramidGeometry | None = None,
) -> dict[str, Any]:
    """
    Create GeoZarr-spec compliant multiscales following the specification exactly.

    Every level L has a pixel size of exactly 2**L native pixels anchored at the
    top-left corner (COG convention). Its extent is therefore shorter at the right
    and bottom by the rows/columns trimmed by the /2 reduction, which is what the
    reduced data actually covers.

    Returns
    -------
    dict
        Dictionary with overview levels information
    """
    if compressor is DEFAULT_COMPRESSOR:
        compressor = make_compressor()
    group_path = _group_path(group_name)
    run_token = uuid.uuid4().hex

    g = geometry
    if g is None:
        g = _pyramid_geometry(ds, group_name, min_dimension, chunk_size, ds_gcp)
    if g is None:
        return {}
    data_vars = g.data_vars
    native_crs, native_bounds, native_px = g.native_crs, g.native_bounds, g.native_px
    overview_levels = g.overview_levels
    left, bottom, right, top = native_bounds
    native_px_w, native_px_h = native_px

    print(f"Total overview levels: {len(overview_levels)}")
    for ol in overview_levels:
        print(
            f"Overview level {ol['level']}: {ol['width']} x {ol['height']} "
            f"(scale factor: {ol['scale_factor']})"
        )

    # Create native CRS tile matrix set and limits
    tile_matrix_set = create_native_crs_tile_matrix_set(
        native_crs, native_bounds, overview_levels, None, native_px=native_px
    )
    tile_matrix_limits = _create_tile_matrix_limits(overview_levels, chunk_size)

    # Build GeoZarr official multiscales layout (https://github.com/zarr-conventions/multiscales)
    # Always north-up (descending y, e < 0) so that rasterio.windows.from_bounds()
    # never raises "Bounds and transform are inconsistent".
    layout: list[dict[str, Any]] = []
    for ol in overview_levels:
        lv = int(ol["level"])
        w, h = ol["width"], ol["height"]
        px_w, px_h = _level_pixel_size(native_px, lv)
        entry: dict[str, Any] = {
            "asset": str(lv),
            "transform": {
                "scale": [1.0, 1.0] if lv == 0 else [2.0, 2.0],
                "translation": [0.0, 0.0],
            },
            "spatial:shape": [h, w],
            "spatial:transform": [px_w, 0.0, left, 0.0, -px_h, top],
            "spatial:bbox": [left, top - h * px_h, left + w * px_w, top],
        }
        if lv > 0:
            entry["derived_from"] = str(lv - 1)
        layout.append(entry)

    # CRS attributes for geo-proj convention (at least one of proj:code/proj:wkt2/proj:projjson required)
    epsg_code = native_crs.to_epsg() if native_crs else None
    proj_attrs: dict[str, Any] = {}
    if epsg_code:
        proj_attrs["proj:code"] = f"EPSG:{epsg_code}"
    if native_crs:
        proj_attrs["proj:wkt2"] = native_crs.to_wkt()

    # spatial: convention attributes at group level (overview-0 transform as baseline)
    spatial_attrs: dict[str, Any] = {
        "spatial:dimensions": ["y", "x"],
        "spatial:bbox": [left, bottom, right, top],
        "spatial:transform": [native_px_w, 0.0, left, 0.0, -native_px_h, top],
        "spatial:registration": "pixel",
    }

    # Add multiscales metadata to the group through the zarr attrs API
    group_attrs: dict[str, Any] = {
        "zarr_conventions": GEOZARR_CONVENTIONS,
        # multiscales convention: only "layout" and "resampling_method" are spec-defined keys
        "multiscales": {
            "layout": layout,
            "resampling_method": method,
            "tile_matrix_set": tile_matrix_set,
            "tile_matrix_limits": tile_matrix_limits,
        },
        **proj_attrs,
        **spatial_attrs,
        # OGC TileMatrixSet info kept as top-level group attributes for OGC API – Tiles clients
        "tile_matrix_set": tile_matrix_set,
        "tile_matrix_limits": tile_matrix_limits,
    }
    zarr_group = zarr.open_group(
        store, path=_group_or_none(group_path), mode="r+", zarr_format=3, use_consolidated=False
    )
    zarr_group.attrs.update(group_attrs)
    print(f"Added GeoZarr-spec compliant multiscales metadata to {group_name}")

    # Create overview levels as children groups
    timing_data = []
    overview_datasets = {}

    # Read level 0 from the store: one dask block per shard, no link to the input graph.
    level_0_path = _join_path(group_path, 0)
    previous_level_ds = _load_existing_dataset(store, level_0_path, shard_size)
    previous_level_path: str | None = level_0_path
    if previous_level_ds is None:
        previous_level_ds = ds
        previous_level_path = None
    previous_level_ds = _normalize_north_up(previous_level_ds)
    _validate_pyramid_level(previous_level_ds, f"{group_name}/0")

    for overview in overview_levels:
        level = int(overview["level"])

        # Skip level 0 - native resolution is already in group 0
        if level == 0:
            print("Skipping level 0 - native resolution is already in group 0")
            continue

        # Check if this overview level already exists in the store
        level_path = _join_path(group_path, level)
        _existing_level_ds = _load_existing_dataset(store, level_path, shard_size)
        if _is_level_complete(_existing_level_ds, data_vars) and _is_level_valid(
            _existing_level_ds
        ):
            print(
                f"Overview level {level} already exists and is complete, "
                "loading from store and skipping creation..."
            )
            overview_datasets[level] = _existing_level_ds
            previous_level_ds = _normalize_north_up(_existing_level_ds)
            previous_level_path = level_path
            continue

        width = overview["width"]
        height = overview["height"]
        scale_factor = overview["scale_factor"]

        print(f"\nCreating overview level {level} (1:{scale_factor} scale)...")
        print(f"Target dimensions: {width} x {height}")
        print(f"  Using pyramid approach: creating level {level} from level {level - 1}")

        if ds_gcp is not None:
            ds_gcp_overview = utils.compute_overview_gcps(
                ds_gcp, scale_factor, width, height
            )
        else:
            ds_gcp_overview = None

        # Overview data: one task per output shard that reads its parent shards from the
        # store one at a time (no 2x2 block merge), or from the in-memory parent when the
        # previous level is not on the store.
        data_arrays = None
        n_blocks = -(-height // shard_size) * -(-width // shard_size)
        if previous_level_path is not None and n_blocks >= MIN_BLOCKS_FROM_STORE:
            data_arrays = _overview_arrays_from_store(
                store,
                previous_level_path,
                previous_level_ds,
                data_vars,
                height,
                width,
                shard_size,
                method,
                nodata_value,
                run_token,
            )
        overview_ds = create_overview_dataset_all_vars(
            previous_level_ds,
            level,
            width,
            height,
            native_crs,
            native_bounds,
            data_vars,
            ds_gcp_overview,
            enable_sharding,
            method=method,
            nodata_value=nodata_value,
            native_px=native_px,
            data_arrays=data_arrays,
        )
        overview_ds, encoding = _prepare_overview_for_write(
            overview_ds, level, shard_size, compressor, chunk_size, enable_sharding
        )

        # Write overview level: one dask block per shard, one compute per level
        start_time = time.time()
        print(f"Writing overview level {level} at {level_path}")
        overview_ds.to_zarr(
            store,
            group=level_path,
            mode="w",
            consolidated=False,
            zarr_format=3,
            encoding=encoding,
            align_chunks=not enable_sharding,
        )

        overview_datasets[level] = overview_ds
        proc_time = time.time() - start_time

        timing_data.append(
            {
                "level": level,
                "time": proc_time,
                "pixels": width * height,
                "width": width,
                "height": height,
                "scale_factor": scale_factor,
            }
        )
        print(f"Level {level}: Successfully created in {proc_time:.2f}s")

        consolidate_metadata(store, path=level_path)
        print(f"  ✅ Metadata consolidated for overview level {level}")

        # Re-open the just-written level to break the lazy chain back to level 0.
        previous_level_ds = _load_existing_dataset(store, level_path, shard_size)
        if previous_level_ds is None:
            print(
                f"  ⚠️ Could not reload level {level} from the store, using in-memory dataset"
            )
            previous_level_ds = overview_ds
            previous_level_path = None
        else:
            previous_level_ds = _normalize_north_up(previous_level_ds)
            previous_level_path = level_path

    print(
        f"\n✅ Created {len(overview_levels)} GeoZarr-compliant overview levels using pyramid approach"
    )

    return {
        "overview_datasets": overview_datasets,
        "levels": overview_levels,
        "timing": timing_data,
        "tile_matrix_set": tile_matrix_set,
        "tile_matrix_limits": tile_matrix_limits,
    }


def _level_pixel_size(native_px: tuple[float, float], level: int) -> tuple[float, float]:
    """Pixel size (w, h magnitudes) of overview *level*: exactly 2**level native pixels."""
    return native_px[0] * 2**level, native_px[1] * 2**level


def calculate_overview_levels(
    native_width: int,
    native_height: int,
    min_dimension: int = 256,
    chunk_size: int = 256,
) -> list[OverviewLevelJSON]:
    """
    Calculate overview levels following COG /2 downsampling logic.

    Returns
    -------
    list
        List of overview level dictionaries
    """
    overview_levels: list[OverviewLevelJSON] = []
    level = 0
    current_width = native_width
    current_height = native_height

    while min(current_width, current_height) >= min_dimension:
        # Calculate zoom level for TMS compatibility
        zoom_for_width = max(0, int(np.ceil(np.log2(current_width / chunk_size))))
        zoom_for_height = max(0, int(np.ceil(np.log2(current_height / chunk_size))))
        zoom = max(zoom_for_width, zoom_for_height)

        overview_level: dict[str, Any] = {
            "level": level,
            "zoom": zoom,
            "width": current_width,
            "height": current_height,
            "scale_factor": 2**level,
        }
        overview_levels.append(overview_level)  # type: ignore[arg-type]

        level += 1
        current_width = native_width // (2**level)
        current_height = native_height // (2**level)

    return overview_levels


def create_native_crs_tile_matrix_set(
    native_crs: Any,
    native_bounds: tuple[float, float, float, float],
    overview_levels: Iterable[OverviewLevelJSON],
    group_prefix: str | None = "",
    native_px: tuple[float, float] | None = None,
) -> TileMatrixSetJSON:
    """
    Create a custom Tile Matrix Set for the native CRS following GeoZarr spec.

    When ``native_px`` is given, the cell size of level L is exactly 2**L times the
    native cell size; otherwise it is derived from the bounds and the level shape.

    Returns
    -------
    dict
        Tile Matrix Set definition following OGC standard
    """
    left, bottom, right, top = native_bounds
    tile_matrices: list[TileMatrixJSON] = []

    for overview in overview_levels:
        level = int(overview["level"])
        width = overview["width"]
        height = overview["height"]

        if native_px is not None:
            px_w, px_h = _level_pixel_size(native_px, level)
        else:
            px_w = (right - left) / width
            px_h = (top - bottom) / height
        cell_size = max(px_w, px_h)

        # Calculate scale denominator
        scale_denominator = cell_size * 3779.5275

        # Calculate matrix dimensions
        chunk_size = overview["chunks"][1][0] if "chunks" in overview else 256
        tile_height = overview["chunks"][0][0] if "chunks" in overview else 256
        matrix_width = int(np.ceil(width / chunk_size))
        matrix_height = int(np.ceil(height / tile_height))

        matrix_id = f"{group_prefix}/{level}" if group_prefix else str(level)

        tile_matrices.append(
            {
                "id": matrix_id,
                "scaleDenominator": scale_denominator,
                "cellSize": cell_size,
                "pointOfOrigin": [left, top],
                "tileWidth": chunk_size,
                "tileHeight": tile_height,
                "matrixWidth": matrix_width,
                "matrixHeight": matrix_height,
            }
        )

    # Create the complete Tile Matrix Set
    epsg_code = native_crs.to_epsg() if native_crs else None
    crs_uri = (
        f"http://www.opengis.net/def/crs/EPSG/0/{epsg_code}"
        if epsg_code
        else (native_crs.to_wkt() if native_crs else "")
    )

    return {
        "id": f"Native_CRS_{epsg_code if epsg_code else 'Custom'}",
        "title": f"Native CRS Tile Matrix Set ({native_crs})",
        "crs": crs_uri,
        "supportedCRS": crs_uri,
        "orderedAxes": ["X", "Y"],
        "tileMatrices": tile_matrices,
    }


def create_overview_dataset_all_vars(
    ds: xr.Dataset,
    level: int,
    width: int,
    height: int,
    native_crs: Any,
    native_bounds: tuple[float, float, float, float],
    data_vars: Sequence[Hashable],
    ds_gcp: xr.Dataset | None = None,
    enable_sharding: bool = False,
    method: str = "mean",
    nodata_value: float | None = None,
    native_px: tuple[float, float] | None = None,
    data_arrays: Mapping[Hashable, da.Array] | None = None,
) -> xr.Dataset:
    """
    Create an overview dataset containing all variables for a specific level (lazy).

    ``data_arrays`` supplies ready-made lazy arrays per variable (see
    :func:`_overview_arrays_from_store`); otherwise the reduction runs on the dask blocks
    of ``ds``.

    The pixel size of the level is exactly 2**level native pixels (when ``native_px``
    is given), anchored at the top-left corner of the native bounds.

    Returns
    -------
    xarray.Dataset
        Overview dataset with all variables
    """
    left, bottom, right, top = native_bounds

    # Always north-up (descending y, pixel_size_y < 0). The source dataset is
    # guaranteed to have been flipped to descending y before reaching here.
    if native_px is not None:
        pixel_size_x, pixel_size_y_mag = _level_pixel_size(native_px, level)
    else:
        pixel_size_x = (right - left) / width
        pixel_size_y_mag = (top - bottom) / height
    y_origin, pixel_size_y = top, -pixel_size_y_mag

    overview_transform = Affine(pixel_size_x, 0.0, left, 0.0, pixel_size_y, y_origin)

    # Pixel-centre coordinates using the closed-form formula, always float64.
    x_coords = np.asarray(left + (np.arange(width) + 0.5) * pixel_size_x, dtype=np.float64)
    y_coords = np.asarray(
        top - (np.arange(height) + 0.5) * pixel_size_y_mag, dtype=np.float64
    )

    # Guard: overviews must always be north-up (descending y, dy < 0).
    if len(y_coords) > 1:
        assert (
            y_coords[0] > y_coords[-1]
        ), f"Overview level {level} y-order is not descending (north-up)"
        assert (
            float(y_coords[1] - y_coords[0]) < 0
        ), f"Overview level {level} transform y sign is not negative (north-up)"

    # Check if we're dealing with geographic coordinates (EPSG:4326)
    if native_crs and native_crs.to_epsg() == 4326:
        overview_coords = {
            "x": (["x"], x_coords, _get_lon_coord_attrs()),
            "y": (["y"], y_coords, _get_lat_coord_attrs()),
        }
    else:
        overview_coords = {
            "x": (["x"], x_coords, _get_x_coord_attrs()),
            "y": (["y"], y_coords, _get_y_coord_attrs()),
        }

    for dim in ds.dims:
        if dim not in SPATIAL_DIMS and dim in ds.coords:
            overview_coords[dim] = ds.coords[dim]

    spatial_dims = list(SPATIAL_DIMS)

    # Find the grid_mapping variable name
    grid_mapping_var_name = _find_grid_mapping_var_name(ds, data_vars)

    # Downsample all data variables
    overview_data_vars = {}
    for var in data_vars:
        print(f"  Downsampling {var}...")

        source_data = ds[var]
        data_dtype = source_data.dtype
        if source_data.ndim > 2:
            lead = [d for d in source_data.dims if d not in spatial_dims]
            source_data = source_data.transpose(*lead, *spatial_dims)
            dims = [*lead, *spatial_dims]
        else:
            dims = spatial_dims

        if data_arrays is not None and var in data_arrays:
            downsampled_data = data_arrays[var]
        else:
            data = source_data.data
            if not isinstance(data, da.Array):
                data = da.from_array(data, chunks="auto")
            if data.ndim > 2:
                # one slice per block on non-spatial dims: task memory stays shard_size²
                data = data.rechunk({i: 1 for i in range(data.ndim - 2)})
            downsampled_data = utils.downsample_2d_array(
                data,
                height,
                width,
                method=method,
                nodata_value=nodata_value,
                out_dtype=data_dtype,
            )

        attrs = {
            "_ARRAY_DIMENSIONS": dims,
            "grid_mapping": grid_mapping_var_name,
        }
        if "standard_name" in ds[var].attrs:
            attrs["standard_name"] = ds[var].attrs["standard_name"]

        overview_data_vars[var] = (dims, downsampled_data, attrs)

    # Create overview dataset
    overview_ds = xr.Dataset(overview_data_vars, coords=overview_coords)

    # Set CRS using rioxarray first
    overview_ds.rio.write_crs(native_crs, inplace=True)
    overview_ds.attrs["grid_mapping"] = grid_mapping_var_name

    # Add grid_mapping variable after setting CRS
    _add_grid_mapping_variable(
        overview_ds, ds, grid_mapping_var_name, overview_transform, native_crs
    )

    return overview_ds


def _write_level0_windows(
    ds: xr.Dataset,
    store: StoreLike,
    group_path: str,
    variables: Sequence[Hashable],
    fused_level: tuple[str, dict[str, Any], Any] | None,
    window_shards: int,
    max_retries: int,
    done: set[tuple[int, int]],
) -> None:
    """
    Write the dask-backed level-0 variables (arrays already created at ``group_path``) in
    windows of ``window_shards``² blocks, one dask compute per window and all variables
    together; with ``fused_level`` every task also reduces its block and the window
    stores the reduction into the level-1 array (created here from the builder), in
    shard-aligned regions. Windows in ``done`` are skipped; each finished window is
    added to the marker attributes, the completion flag last.
    """
    arrays = {v: zarr.open_array(store, path=_join_path(group_path, str(v)), mode="r+", zarr_format=3) for v in variables}
    l1_arrays: dict[Hashable, zarr.Array] = {}
    fy = fx = 1
    l1_h = l1_w = 0
    method = nodata_value = None
    if fused_level is not None:
        fused_path, spec, builder = fused_level
        l1_h, l1_w, method, nodata_value = spec["height"], spec["width"], spec["method"], spec["nodata_value"]
        # Level-1 metadata and coordinates: the builder needs lazy arrays of the right
        # shape; the delayed chunk writes it returns are dropped, the windows do them.
        lazy = {
            v: _fused_write_reduce_array(ds[v].data, store, _join_path(group_path, str(v)), l1_h, l1_w, method, nodata_value, arr=arrays[v])
            for v in variables
        }
        fused_path, fused_ds, fused_enc = builder(lazy)
        # On resume the level-1 arrays of the interrupted run hold the regions of the
        # windows done and are kept as they are (xarray's append mode would recreate
        # them). If any is missing the level is rebuilt from scratch: every window again.
        l1_present = done and all(_node_exists(store, _join_path(fused_path, str(v))) for v in variables)
        if not l1_present:
            if done:
                print("  Level-1 arrays of the interrupted run are incomplete: rewriting every window")
                done.clear()
            fused_ds.to_zarr(
                store,
                group=fused_path,
                mode="w",
                consolidated=False,
                zarr_format=3,
                encoding=fused_enc,
                align_chunks=not any((e or {}).get("shards") for e in fused_enc.values()),
                compute=False,
            )
        l1_arrays = {v: zarr.open_array(store, path=_join_path(fused_path, str(v)), mode="r+", zarr_format=3) for v in variables}
        first = ds[variables[0]].data
        fy, fx = first.shape[-2] // l1_h, first.shape[-1] // l1_w
        print(f"  Fusing overview level {fused_path} into the level-0 windows")

    first = ds[variables[0]].data
    windows = _level0_windows(first.chunks[-2], first.chunks[-1], window_shards)
    todo = [w for w in windows if w[0] not in done]
    print(f"  Writing level 0 in {len(windows)} windows of {window_shards}² blocks ({len(todo)} to do, {len(done)} done)")
    for n, (key, ys, xs) in enumerate(todo, 1):
        delayed = []
        for v in variables:
            data = ds[v].data
            win = data[..., ys, xs]
            offset = (0,) * (data.ndim - 2) + (ys.start, xs.start)
            if fused_level is None:
                delayed.append(
                    da.map_blocks(
                        _write_block, win, arr=arrays[v], offset=offset, dtype=bool,
                        chunks=tuple((1,) * len(c) for c in win.chunks), meta=np.empty((0,) * win.ndim, dtype=bool),
                    )
                )
                continue
            h1 = min((ys.stop - ys.start) // fy, l1_h - ys.start // fy)
            w1 = min((xs.stop - xs.start) // fx, l1_w - xs.start // fx)
            red = _fused_write_reduce_array(win, store, "", h1, w1, method, nodata_value, offset=offset, arr=arrays[v])
            l1 = l1_arrays[v]
            unit = (l1.shards or l1.chunks)[-2:]
            if any(c % u for c, u in zip(red.chunksize[-2:], unit)):
                red = red.rechunk({red.ndim - 2: unit[0], red.ndim - 1: unit[1]})
            region = (slice(None),) * (data.ndim - 2) + (slice(ys.start // fy, ys.start // fy + h1), slice(xs.start // fx, xs.start // fx + w1))
            delayed.append(da.store(red, l1, regions=region, lock=False, compute=False))
        for attempt in range(max_retries):
            try:
                dask.compute(*delayed)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"    ⚠️  Window {key} attempt {attempt + 1} failed: {e}; retrying")
                    time.sleep(2)
                else:
                    raise RuntimeError(f"window {key} failed after {max_retries} attempts: {e}") from e
        done.add(key)
        _write_window_marker(store, group_path, done, window_shards, complete=len(done) == len(windows))
        if n % 10 == 0 or n == len(todo):
            print(f"    window {n}/{len(todo)} done")


def write_dataset_band_by_band_with_validation(
    ds: xr.Dataset,
    existing_dataset: xr.Dataset | None,
    store: StoreLike,
    encoding: dict[Hashable, XarrayEncodingJSON],
    max_retries: int,
    group_name: str,
    shard_size: int = 4096,
    force_overwrite: bool = False,
    fused_level: tuple[str, dict[str, Any], Any] | None = None,
    window_shards: int | None = None,
) -> tuple[bool, xr.Dataset]:
    """
    Write dataset band by band with individual band validation.

    Bands that already exist and validate are skipped when the level carries the
    completion marker. Otherwise the remaining bands are written in windows of
    ``window_shards``² dask blocks, one dask compute per window and all bands together,
    so the graph is bounded by the window whatever the size of the raster. Every window
    is retried up to ``max_retries`` times; the windows done are recorded in the group's
    attributes, and a level whose marker lists some windows resumes from the missing ones.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to write
    existing_dataset : xarray.Dataset, optional
        Existing dataset at the target group, if any
    store : zarr store
        Output store
    encoding : dict
        Encoding configuration for variables
    max_retries : int
        Maximum number of attempts per band in the fallback path
    group_name : str
        Store-relative path of the level group (e.g. "0" or "measurements/0")
    shard_size : int
        Dask block size on y/x for unsharded variables
    force_overwrite : bool, default False
        Force overwrite existing bands even if they're valid
    fused_level : (path, spec, builder), optional
        Level 1, written in the same computes as the bands: every band task writes its
        level-0 block and returns the reduced block, which the window then stores into
        the level-1 array; ``builder(arrays)`` turns lazy reduced arrays into
        ``(path, dataset, encoding)``, used once to create the level-1 metadata.
    window_shards : int, optional
        Dask blocks per axis and per window (default ``WINDOW_SHARDS``, even).

    Returns
    -------
    tuple[bool, xarray.Dataset]
        (True if all bands were written successfully, updated dataset)
    """
    group_path = _group_path(group_name)
    print(
        f"Writing GeoZarr-spec compliant base resolution for {group_path or '/'} band by band with validation"
    )

    data_vars = [
        var for var in ds.data_vars if not utils.is_grid_mapping_variable(ds, var)
    ]

    skipped_vars: list[Hashable] = []
    failed_vars: list[Hashable] = []
    to_write: list[Hashable] = []

    store_exists = existing_dataset is not None and len(existing_dataset.data_vars) > 0
    if window_shards is None:
        window_shards = WINDOW_SHARDS
    if window_shards < 2 or window_shards % 2:
        raise ValueError(f"window_shards must be even and >= 2, got {window_shards}")

    done, complete, done_size = _read_window_marker(store, group_path)
    resume = bool(done) and not complete and done_size == window_shards and not force_overwrite
    if not resume:
        done = set()
    resumed_vars: list[Hashable] = []   # arrays kept from the interrupted run: metadata exists

    for var in data_vars:
        if not force_overwrite and store_exists:
            if utils.validate_existing_band_data(existing_dataset, var, ds):
                if complete:
                    ds[var] = existing_dataset[var]  # type: ignore[index]
                    print(f"  ✅ Band {var} already exists and is valid, skipping")
                    skipped_vars.append(var)
                    continue
                if resume:
                    print(f"  ⏩ Band {var} exists, level 0 incomplete: resuming from {len(done)} windows done")
                    to_write.append(var)
                    resumed_vars.append(var)
                    continue
            if var in existing_dataset:  # type: ignore[operator]
                print(f"    🧹 Removing invalid existing variable {var}...")
                _delete_prefix(store, _join_path(group_path, str(var)))
        to_write.append(var)

    coords_on_store = existing_dataset is not None and "x" in existing_dataset.coords

    def _write_var(var: Hashable, compute: bool, with_coords: bool) -> Any:
        single_var_ds = ds[[var]]
        enc = single_var_ds[var].encoding
        var_encoding: dict[Hashable, Any] = {}
        if var in encoding:
            var_encoding[var] = encoding[var]
        if with_coords:
            for coord in single_var_ds.coords:
                if coord in encoding:
                    var_encoding[coord] = encoding[coord]
        var_enc = encoding.get(var, {})
        source_chunks = ds[var].data.chunks if isinstance(ds[var].data, da.Array) else None
        single_var_ds[var] = single_var_ds[var].chunk(
            _dask_chunks_for(var_enc, single_var_ds[var].dims, shard_size, source_chunks)
        )
        # Drop on-disk chunk hints inherited from the source; keep the CRS reference.
        single_var_ds[var].encoding = {k: v for k, v in enc.items() if k == "grid_mapping"}
        _pin_grid_mapping(single_var_ds)
        return single_var_ds.to_zarr(
            store,
            group=_group_or_none(group_path),
            mode="a",
            consolidated=False,
            zarr_format=3,
            encoding=var_encoding,
            align_chunks=var_enc.get("shards") is None,
            compute=compute,
        )

    written: list[Hashable] = []
    if to_write:
        lazy_vars = [v for v in to_write if isinstance(ds[v].data, da.Array)]
        eager_vars = [v for v in to_write if v not in lazy_vars]
        try:
            # Metadata (and numpy coordinates) are written eagerly by each call; the
            # chunk writes are done window by window below.
            for i, var in enumerate(to_write):
                if var in resumed_vars and var in lazy_vars:
                    continue
                _write_var(var, compute=(var in eager_vars), with_coords=(i == 0 and not coords_on_store))
                coords_on_store = True
            written.extend(eager_vars)
            if lazy_vars:
                _write_level0_windows(
                    ds, store, group_path, lazy_vars, fused_level, window_shards, max_retries, done
                )
                written.extend(lazy_vars)
            for var in written:
                print(f"    ✅ Successfully wrote {var}")
            if fused_level is not None and lazy_vars:
                consolidate_metadata(store, path=fused_level[0])
                print(f"    ✅ Successfully wrote fused level {fused_level[0]}")
        except Exception as e:
            print(f"    ❌ Level-0 write failed: {e}")
            failed_vars = [v for v in to_write if v not in written]

    successful_vars = skipped_vars + written

    # Consolidate metadata
    consolidate_metadata(store, path=_group_or_none(group_path))
    print(f"  ✅ Metadata consolidated for {len(successful_vars)} variables")

    if failed_vars:
        print(
            f"❌ Failed to write {len(failed_vars)} variables for {group_path or '/'}: {failed_vars}"
        )
        print(f"✅ Successfully wrote {len(written)} new variables")
        print(f"⏭️  Skipped {len(skipped_vars)} existing valid variables: {skipped_vars}")
        return False, ds

    print(
        f"✅ Successfully processed all {len(successful_vars)} variables for {group_path or '/'}"
    )
    if skipped_vars:
        print(f"   - Wrote {len(written)} new variables")
        print(f"   - Skipped {len(skipped_vars)} existing valid variables")
    return True, ds


def consolidate_metadata(
    store: StoreLike,
    path: str | None = None,
    zarr_format: zarr.core.common.ZarrFormat | None = None,
) -> zarr.Group:
    """
    Consolidate metadata of all nodes in a hierarchy.

    Parameters
    ----------
    store : StoreLike
        The store-like object whose metadata to consolidate
    path : str, optional
        Path to a group in the store to consolidate at
    zarr_format : {2, 3, None}, optional
        The zarr format of the hierarchy

    Returns
    -------
    zarr.Group
        The group with consolidated metadata
    """
    return zarr.Group(
        sync(async_consolidate_metadata(store, path=path, zarr_format=zarr_format))
    )


async def async_consolidate_metadata(
    store: StoreLike,
    path: str | None = None,
    zarr_format: zarr.core.common.ZarrFormat | None = None,
) -> zarr.core.group.AsyncGroup:
    """Consolidate metadata of all nodes in a hierarchy asynchronously."""
    store_path = await make_store_path(store, path=path)

    if not store_path.store.supports_consolidated_metadata:
        store_name = type(store_path.store).__name__
        raise TypeError(
            f"The Zarr Store in use ({store_name}) doesn't support consolidated metadata",
        )

    group = await zarr.core.group.AsyncGroup.open(
        store_path, zarr_format=zarr_format, use_consolidated=False
    )
    group.store_path.store._check_writable()

    members_metadata = {
        k: v.metadata
        async for k, v in group.members(
            max_depth=None, use_consolidated_for_children=False
        )
    }

    zarr.core.group.ConsolidatedMetadata._flat_to_nested(members_metadata)

    consolidated_metadata = zarr.core.group.ConsolidatedMetadata(
        metadata=members_metadata
    )
    metadata = dataclasses.replace(
        group.metadata, consolidated_metadata=consolidated_metadata
    )
    group = dataclasses.replace(
        group,
        metadata=metadata,
    )

    await group._save_metadata()
    return group


def _normalize_north_up(ds: xr.Dataset) -> xr.Dataset:
    """Normalize coordinates and synchronize the stored GeoTransform."""

    if ds.sizes.get("x", 0) > 1 and float(ds.x[0]) > float(ds.x[-1]):
        ds = ds.isel(x=slice(None, None, -1))

    if ds.sizes.get("y", 0) > 1 and float(ds.y[0]) < float(ds.y[-1]):
        ds = ds.isel(y=slice(None, None, -1))

    transform = ds.rio.transform(recalc=True)

    return ds.rio.write_transform(
        transform,
        grid_mapping_name=ds.rio.grid_mapping,
        inplace=False,
    )


# Helper functions
def _add_coordinate_metadata(ds: xr.Dataset) -> None:
    """Add proper metadata to coordinate variables."""
    is_geographic = bool(ds.rio.crs and ds.rio.crs.to_epsg() == 4326)
    for coord_name in ds.coords:
        if coord_name == "x":
            ds[coord_name].attrs.update(
                _get_lon_coord_attrs() if is_geographic else _get_x_coord_attrs()
            )
        elif coord_name == "y":
            ds[coord_name].attrs.update(
                _get_lat_coord_attrs() if is_geographic else _get_y_coord_attrs()
            )
        elif coord_name == "time":
            ds[coord_name].attrs.update(
                {"_ARRAY_DIMENSIONS": ["time"], "standard_name": "time"}
            )
        elif coord_name == "angle":
            ds[coord_name].attrs.update(
                {
                    "_ARRAY_DIMENSIONS": ["angle"],
                    "standard_name": "angle",
                    "long_name": "angle coordinate",
                }
            )
        elif coord_name == "band":
            ds[coord_name].attrs.update(
                {
                    "_ARRAY_DIMENSIONS": ["band"],
                    "standard_name": "band",
                    "long_name": "spectral band identifier",
                }
            )
        elif coord_name == "detector":
            ds[coord_name].attrs.update(
                {
                    "_ARRAY_DIMENSIONS": ["detector"],
                    "standard_name": "detector",
                    "long_name": "detector identifier",
                }
            )
        else:
            if "_ARRAY_DIMENSIONS" not in ds[coord_name].attrs:
                ds[coord_name].attrs["_ARRAY_DIMENSIONS"] = [coord_name]


def _setup_grid_mapping(ds: xr.Dataset, grid_mapping_var_name: str) -> None:
    """Set up spatial_ref variable with GeoZarr required attributes."""

    if ds.rio.crs and "spatial_ref" in ds:
        ds["spatial_ref"].attrs["_ARRAY_DIMENSIONS"] = []
        if ds.rio.transform():
            transform_gdal = ds.rio.transform().to_gdal()
            transform_str = " ".join([str(i) for i in transform_gdal])
            ds["spatial_ref"].attrs["GeoTransform"] = transform_str

    ds.attrs["grid_mapping"] = grid_mapping_var_name
    for band in ds.data_vars:
        if band != "spatial_ref":
            ds[band].attrs["grid_mapping"] = grid_mapping_var_name


def _add_geotransform(ds: xr.Dataset, grid_mapping_var: str) -> None:
    """Add GeoTransform to grid_mapping variable."""
    ds[grid_mapping_var].attrs["_ARRAY_DIMENSIONS"] = []

    if len(ds.coords["x"]) > 1 and len(ds.coords["y"]) > 1:
        x_coords = ds.coords["x"].values
        y_coords = ds.coords["y"].values

        pixel_size_x = float(x_coords[1] - x_coords[0])
        pixel_size_y = float(y_coords[1] - y_coords[0])

        x_origin = float(x_coords[0]) - pixel_size_x / 2
        y_origin = float(y_coords[0]) - pixel_size_y / 2

        transform_str = f"{x_origin} {pixel_size_x} 0.0 {y_origin} 0.0 {pixel_size_y}"
        ds[grid_mapping_var].attrs["GeoTransform"] = transform_str


def _find_reference_crs(geozarr_groups: Mapping[str, xr.Dataset]) -> str | None:
    """Find the reference CRS in the geozarr groups."""
    for group in geozarr_groups.values():
        if group.rio.crs:
            crs_string: str = group.rio.crs.to_string()
            return crs_string
    return None


def _create_encoding(
    ds: xr.Dataset, compressor: Any, shard_size: int
) -> dict[Hashable, XarrayEncodingJSON]:
    """Create encoding for (non-GeoZarr) dataset variables."""
    encoding: dict[Hashable, XarrayEncodingJSON] = {}
    chunking: tuple[int, ...]
    for var in ds.data_vars:
        if hasattr(ds[var].data, "chunks"):
            current_chunks = ds[var].chunks
            chunking = tuple(
                (current_chunks[i][0] if len(current_chunks[i]) > 0 else ds[var].shape[i])
                for i in range(len(current_chunks))
            )
        else:
            data_shape = ds[var].shape
            if len(data_shape) >= 2:
                chunk_y = min(shard_size, data_shape[-2])
                chunk_x = min(shard_size, data_shape[-1])
                if len(data_shape) == 3:
                    chunking = (1, chunk_y, chunk_x)
                else:
                    chunking = (chunk_y, chunk_x)
            elif len(data_shape) == 1:
                chunking = (min(shard_size, data_shape[-1]),)
            else:
                chunking = ()

        enc: XarrayEncodingJSON = {
            "compressors": [compressor] if compressor is not None else None
        }
        if chunking:
            enc["chunks"] = chunking
        encoding[var] = enc

    for coord in ds.coords:
        encoding[coord] = {"compressors": None}

    return encoding


def _create_geozarr_encoding(
    ds: xr.Dataset,
    compressor: Any,
    chunk_size: int,
    shard_size: int,
    enable_sharding: bool = False,
) -> dict[Hashable, XarrayEncodingJSON]:
    """
    Zarr encoding for a GeoZarr level.

    Chunks are ``chunk_size`` on y/x and 1 on any other dim, clipped to the array shape.
    Shards (optional) are ``shard_size`` on y/x and 1 on any other dim, always a
    multiple of the chunk and never larger than the array, so task memory stays
    bounded by shard_size² regardless of the time/band extent.
    """
    encoding: dict[Hashable, XarrayEncodingJSON] = {}

    for var in ds.data_vars:
        if utils.is_grid_mapping_variable(ds, var):
            encoding[var] = {"compressors": None}
            continue

        shape = ds[var].shape
        dims = ds[var].dims
        ndim = len(shape)
        if ndim == 0:
            encoding[var] = {"compressors": None}
            continue

        base_chunks = [chunk_size if d in SPATIAL_DIMS else 1 for d in dims]
        targets = [shard_size if d in SPATIAL_DIMS else 1 for d in dims]

        chunks = []
        for axis in range(ndim):
            c = base_chunks[axis]
            if c > shape[axis]:
                print(
                    f"ℹ️  Adjusting chunk dim for var={var}, axis={axis}: "
                    f"{c} → {shape[axis]} (dataset smaller)"
                )
                c = shape[axis]
            chunks.append(max(1, c))

        shards = None
        if enable_sharding:
            shards = tuple(
                c * max(1, min(t, s) // c) for c, t, s in zip(chunks, targets, shape)
            )

        encoding[var] = {
            "chunks": tuple(chunks),
            "compressors": compressor,
            "shards": shards,
        }

    # coordinates: no compression, no sharding
    for coord in ds.coords:
        encoding[coord] = {"compressors": None}

    return encoding


def _load_existing_dataset(
    store: StoreLike, path: str, shard_size: int | None = None
) -> xr.Dataset | None:
    """
    Open a level group from the store, if it exists.

    With ``shard_size`` the dask block is that size on y/x (one block per shard)
    and one slice on any other dim; otherwise dask's "auto" chunking is used.
    """
    try:
        if not _node_exists(store, path):
            return None
        chunks: Any = "auto"
        if shard_size is not None:
            chunks = {"y": shard_size, "x": shard_size}
        ds = xr.open_dataset(
            store,
            group=_group_or_none(path),
            zarr_format=3,
            engine="zarr",
            chunks=chunks,
            decode_coords="all",
            consolidated=False,
        )
        return set_spatial_info(ds)
    except Exception as e:
        print(f"Warning: Could not open existing dataset at {path}: {e}")
    return None


def _create_tile_matrix_limits(
    overview_levels: Iterable[OverviewLevelJSON], chunk_size: int
) -> dict[str, TileMatrixLimitJSON]:
    """Create tile matrix limits for overview levels."""
    tile_matrix_limits: dict[str, TileMatrixLimitJSON] = {}
    for ol in overview_levels:
        level_str = str(ol["level"])
        max_tile_col = int(np.ceil(ol["width"] / chunk_size)) - 1
        max_tile_row = int(np.ceil(ol["height"] / chunk_size)) - 1

        tile_matrix_limits[level_str] = {
            "tileMatrix": level_str,
            "minTileCol": 0,
            "maxTileCol": max_tile_col,
            "minTileRow": 0,
            "maxTileRow": max_tile_row,
        }

    return tile_matrix_limits


def _get_x_coord_attrs() -> StandardXCoordAttrsJSON:
    """Get standard attributes for x coordinate."""
    return {
        "units": "m",
        "long_name": "x coordinate of projection",
        "standard_name": "projection_x_coordinate",
        "_ARRAY_DIMENSIONS": ["x"],
    }


def _get_y_coord_attrs() -> StandardYCoordAttrsJSON:
    """Get standard attributes for y coordinate."""
    return {
        "units": "m",
        "long_name": "y coordinate of projection",
        "standard_name": "projection_y_coordinate",
        "_ARRAY_DIMENSIONS": ["y"],
    }


def _get_lon_coord_attrs():
    """Get standard attributes for longitude coordinate."""
    return {
        "units": "degrees_east",
        "long_name": "longitude",
        "standard_name": "longitude",
        "_ARRAY_DIMENSIONS": ["x"],
    }


def _get_lat_coord_attrs():
    """Get standard attributes for latitude coordinate."""
    return {
        "units": "degrees_north",
        "long_name": "latitude",
        "standard_name": "latitude",
        "_ARRAY_DIMENSIONS": ["y"],
    }


def _find_grid_mapping_var_name(ds: xr.Dataset, data_vars: Sequence[Hashable]) -> str:
    """Find the grid_mapping variable name from the dataset."""
    grid_mapping_var_name = ds.attrs.get("grid_mapping", None)
    if not grid_mapping_var_name and data_vars:
        first_var = data_vars[0]
        if first_var in ds and "grid_mapping" in ds[first_var].attrs:
            grid_mapping_var_name = ds[first_var].attrs["grid_mapping"]

    if not grid_mapping_var_name:
        grid_mapping_var_name = "spatial_ref"

    return str(grid_mapping_var_name)


def _add_grid_mapping_variable(
    overview_ds: xr.Dataset,
    ds: xr.Dataset,
    grid_mapping_var_name: str,
    overview_transform: Any,
    native_crs: Any,
) -> None:
    """Add grid_mapping variable to overview dataset."""

    base_attrs: dict[str, Any] = {
        "_ARRAY_DIMENSIONS": [],
    }

    if overview_transform is not None:
        transform_gdal = overview_transform.to_gdal()
        transform_str = " ".join([str(i) for i in transform_gdal])
        base_attrs["GeoTransform"] = transform_str

    if grid_mapping_var_name in ds:
        grid_mapping_attrs = ds[grid_mapping_var_name].attrs.copy()
        grid_mapping_attrs.update(base_attrs)

        overview_ds.coords[grid_mapping_var_name] = xr.DataArray(
            data=ds[grid_mapping_var_name].values,
            attrs=grid_mapping_attrs,
        )
    else:
        print(f"  Creating new grid_mapping variable '{grid_mapping_var_name}'")

        grid_mapping_attrs = base_attrs.copy()

        if native_crs:
            grid_mapping_attrs["spatial_ref"] = native_crs.to_wkt()
            grid_mapping_attrs["crs_wkt"] = native_crs.to_wkt()

        overview_ds.coords[grid_mapping_var_name] = xr.DataArray(
            data=np.array(b"", dtype="S1"),
            attrs=grid_mapping_attrs,
        )

    # Ensure all data variables have the grid_mapping attribute
    for var_name in overview_ds.data_vars:
        if not utils.is_grid_mapping_variable(overview_ds, var_name):
            if "grid_mapping" not in overview_ds[var_name].attrs:
                overview_ds[var_name].attrs["grid_mapping"] = grid_mapping_var_name
                print(f"  Added grid_mapping attribute to {var_name}")


def _is_level_valid(ds: xr.Dataset | None) -> bool:
    if ds is None:
        return False

    try:
        if "x" not in ds.coords or "y" not in ds.coords:
            return True

        if not ds.x.to_index().is_monotonic_increasing:
            return False

        if not ds.y.to_index().is_monotonic_decreasing:
            return False

        stored = ds.rio.transform(recalc=False)
        calculated = ds.rio.transform(recalc=True)
        bounds = ds.rio.bounds(recalc=True)

        if not np.allclose(
            stored.to_gdal(),
            calculated.to_gdal(),
            rtol=1e-9,
            atol=1e-9,
        ):
            return False

        from rasterio.windows import from_bounds

        from_bounds(*bounds, transform=stored)
        return stored.a > 0 and stored.e < 0

    except Exception:
        return False


def _validate_pyramid_level(ds: xr.Dataset, label: Any) -> None:
    """Assert north-up / west-east convention and rasterio consistency for a pyramid level."""
    from rasterio.windows import from_bounds as _rasterio_from_bounds

    bounds = ds.rio.bounds(recalc=True)
    stored_transform = ds.rio.transform(recalc=False)
    calculated_transform = ds.rio.transform(recalc=True)

    assert np.allclose(
        stored_transform.to_gdal(),
        calculated_transform.to_gdal(),
        rtol=1e-9,
        atol=1e-9,
    ), f"{label}: stored GeoTransform differs from coordinates"

    assert stored_transform.a > 0
    assert stored_transform.e < 0

    _rasterio_from_bounds(*bounds, transform=stored_transform)


def _is_level_complete(
    level_ds: xr.Dataset | None, data_vars: Sequence[Hashable]
) -> bool:
    """True if *level_ds* is not None and every variable in *data_vars* is present."""
    if level_ds is None:
        return False
    return all(var in level_ds.data_vars for var in data_vars)


def _is_sentinel1(dt: xr.DataTree) -> bool:
    """Return True if the input DataTree represents a Sentinel-1 product."""
    stac_props = dt.attrs.get("stac_discovery", {}).get("properties", {})
    return bool(stac_props.get("product:type", "not-a-product").startswith("S01"))
