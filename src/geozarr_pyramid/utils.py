"""Utility functions for GeoZarr conversion."""

import warnings
from typing import Any

import dask.array as da
import numpy as np
import rioxarray  # noqa: F401  # Import to enable .rio accessor
import xarray as xr

VALID_FRACTION = 0.3
"""Minimum fraction of valid source pixels for an output pixel to be kept."""


def result_dtype(dtype: Any, method: str) -> np.dtype:
    """Output dtype of :func:`reduce_block` before any final cast."""
    dtype = np.dtype(dtype)
    is_integer = np.issubdtype(dtype, np.integer)
    if method == "mean":
        return np.dtype("float64")
    if method in {"min", "max", "nearest"}:
        return dtype
    if method == "median":
        if (is_integer and dtype.itemsize <= 2) or dtype == np.float32:
            return np.dtype("float32")
        return np.dtype("float64")
    raise ValueError(f"Unknown method: {method}")


def _fill_value(dtype: np.dtype, fill: float) -> Any:
    if np.issubdtype(dtype, np.integer) and (fill is None or np.isnan(fill)):
        return 0
    return dtype.type(fill)


def reduce_block(
    block: np.ndarray,
    fy: int,
    fx: int,
    method: str = "mean",
    nodata_value: float | None = None,
    out_dtype: Any = None,
) -> np.ndarray:
    """
    Reduce the last two axes of a numpy block by integer factors ``fy`` and ``fx``.

    Row/column remainders are trimmed. NaN is always invalid; a numeric ``nodata_value``
    is invalid too. An output pixel whose valid fraction is below ``VALID_FRACTION`` is
    nodata. A block that holds no valid pixel at all is answered without a reduction,
    which is what makes sparse rasters cheap. ``mean`` is computed in float64, ``min`` and
    ``max`` in the source dtype, ``median`` in float32 for integers up to 16 bits and
    float64 otherwise; ``nearest`` takes the pixel at the block centre. The result is
    cast to ``out_dtype`` when given, rounding to nearest for integer targets.
    """
    block = np.asarray(block)
    if block.ndim < 2:
        raise ValueError(f"Expected at least 2 dims, got shape {block.shape}")
    lead = block.shape[:-2]
    h, w = block.shape[-2:]
    th, tw = h // fy, w // fx
    res_dtype = result_dtype(block.dtype, method)
    out_dt = np.dtype(out_dtype) if out_dtype is not None else res_dtype
    if th == 0 or tw == 0:
        return np.empty(lead + (th, tw), dtype=out_dt)

    is_integer = np.issubdtype(block.dtype, np.integer)
    has_numeric_nodata = nodata_value is not None and not np.isnan(nodata_value)
    fill = nodata_value if has_numeric_nodata else np.nan

    trimmed = block[..., : th * fy, : tw * fx]

    if method == "nearest":
        return trimmed[..., fy // 2 :: fy, fx // 2 :: fx].astype(out_dt, copy=False)

    # Fast path: nothing valid in the block (NaN != nodata is True, so a NaN-only block
    # with a numeric nodata still takes the slow path and is handled there).
    if has_numeric_nodata:
        empty = not bool((trimmed != nodata_value).any())
    elif not is_integer:
        empty = bool(np.isnan(trimmed).all())
    else:
        empty = False
    if empty:
        return np.full(lead + (th, tw), _fill_value(out_dt, fill), dtype=out_dt)

    blocks = trimmed.reshape(*lead, th, fy, tw, fx)
    axes = (-3, -1)
    n = fy * fx

    isnan = None
    if is_integer:
        valid = (blocks != nodata_value) if has_numeric_nodata else None
    else:
        isnan = np.isnan(blocks)
        valid = ~isnan
        if has_numeric_nodata:
            valid &= blocks != nodata_value

    if valid is None:
        cnt = None
        keep = None
    else:
        cnt = valid.sum(axis=axes, dtype=np.int32)
        keep = cnt >= VALID_FRACTION * n

    if method == "mean":
        if valid is None:
            result = blocks.sum(axis=axes, dtype=np.float64) / n
        elif has_numeric_nodata and nodata_value == 0 and (isnan is None or not isnan.any()):
            # nodata 0 adds nothing to the sum: no masked copy of the block
            result = blocks.sum(axis=axes, dtype=np.float64) / np.maximum(cnt, 1)
        else:
            total = np.where(valid, blocks, 0).sum(axis=axes, dtype=np.float64)
            result = total / np.maximum(cnt, 1)
    elif method in {"min", "max"}:
        if is_integer:
            if valid is None:
                masked = blocks
            else:
                info = np.iinfo(blocks.dtype)
                extreme = info.max if method == "min" else info.min
                masked = np.where(valid, blocks, blocks.dtype.type(extreme))
            result = masked.min(axis=axes) if method == "min" else masked.max(axis=axes)
        else:
            masked = np.where(valid, blocks, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                reducer = np.nanmin if method == "min" else np.nanmax
                result = reducer(masked, axis=axes)
    elif method == "median":
        work = res_dtype
        if valid is None:
            result = np.median(blocks.astype(work, copy=False), axis=axes)
        else:
            masked = np.where(valid, blocks.astype(work, copy=False), work.type(np.nan))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                result = np.nanmedian(masked, axis=axes)
    else:
        raise ValueError(f"Unknown method: {method}")

    if keep is not None:
        result = np.where(keep, result, fill)
    if np.issubdtype(out_dt, np.integer) and np.issubdtype(result.dtype, np.floating):
        result = np.rint(result)
    return result.astype(out_dt, copy=False)


def _align_chunks(arr: da.Array, fy: int, fx: int) -> da.Array:
    """Rechunk the last two axes so every chunk boundary is a multiple of the factor."""
    new_chunks = {}
    for axis, f in ((arr.ndim - 2, fy), (arr.ndim - 1, fx)):
        chunks = arr.chunks[axis]
        if f == 1 or all(c % f == 0 for c in chunks[:-1]):
            continue
        total = sum(chunks)
        bounds = sorted({(b - b % f) for b in np.cumsum(chunks)[:-1]} - {0})
        edges = [0, *bounds, total]
        new_chunks[axis] = tuple(b - a for a, b in zip(edges[:-1], edges[1:]) if b > a)
    return arr.rechunk(new_chunks) if new_chunks else arr


def downsample_2d_array(
    source_data: da.Array,
    target_height: int,
    target_width: int,
    nodata_value: float | None = None,
    method: str = "mean",
    out_dtype: Any = None,
) -> da.Array:
    """
    Downsample the last two axes of a Dask array to an exact output shape, lazily.

    One :func:`reduce_block` task per source block, so with source chunks that are
    multiples of the reduction factor there is no rechunk. Leading axes are kept as is
    (their chunking should be 1 for bounded task memory). See :func:`reduce_block` for
    dtypes and the nodata rules; ``out_dtype`` folds the final cast into the task.
    """
    if source_data.ndim < 2:
        raise ValueError(f"Expected at least a 2-D array, got shape {source_data.shape}")
    source_height, source_width = source_data.shape[-2:]
    if not 0 < target_height <= source_height:
        raise ValueError("target_height must be between 1 and source height")
    if not 0 < target_width <= source_width:
        raise ValueError("target_width must be between 1 and source width")
    if method not in {"mean", "min", "max", "median", "nearest"}:
        raise ValueError(f"Unknown method: {method}")
    if not isinstance(source_data, da.Array):
        source_data = da.from_array(source_data, chunks="auto")

    fy = source_height // target_height
    fx = source_width // target_width
    src = _align_chunks(source_data, fy, fx)
    dtype = np.dtype(out_dtype) if out_dtype is not None else result_dtype(src.dtype, method)
    out_chunks = (
        *src.chunks[:-2],
        tuple(c // fy for c in src.chunks[-2]),
        tuple(c // fx for c in src.chunks[-1]),
    )
    out = da.map_blocks(
        reduce_block,
        src,
        fy=fy,
        fx=fx,
        method=method,
        nodata_value=nodata_value,
        out_dtype=dtype,
        dtype=dtype,
        chunks=out_chunks,
    )
    if out.shape[-2] != target_height or out.shape[-1] != target_width:
        out = out[..., :target_height, :target_width]
    return out


def is_grid_mapping_variable(ds: xr.Dataset, var_name: str) -> bool:
    """
    Check if a variable is a grid_mapping variable by looking for references to it.

    Returns
    -------
    bool
        True if this variable is referenced as a grid_mapping
    """
    for data_var in ds.data_vars:
        if data_var != var_name and "grid_mapping" in ds[data_var].attrs:
            if ds[data_var].attrs["grid_mapping"] == var_name:
                return True
    return False


def calculate_aligned_chunk_size(dimension_size: int, target_chunk_size: int) -> int:
    """
    Calculate a chunk size that divides evenly into the dimension size.

    Returns
    -------
    int
        Aligned chunk size that divides evenly into dimension_size
    """
    if target_chunk_size >= dimension_size:
        return dimension_size

    for chunk_size in range(target_chunk_size, int(target_chunk_size * 0.51), -1):
        if dimension_size % chunk_size == 0:
            return chunk_size

    return min(target_chunk_size, dimension_size)


def validate_existing_band_data(
    existing_group: xr.Dataset, var_name: str, reference_ds: xr.Dataset
) -> bool:
    """
    Validate that a specific band exists in the dataset and matches the reference.

    Checks what the writer guarantees: presence, shape, dtype, the dims attribute,
    the CRS, and that one inner chunk is readable. Pixel values are not judged:
    a NaN at the sampled position is legitimate data.

    Returns
    -------
    bool
        True if the variable exists and is valid, False otherwise
    """
    try:
        if (
            var_name not in existing_group.data_vars
            and var_name not in existing_group.coords
        ):
            return False

        if var_name in reference_ds.data_vars:
            if reference_ds[var_name].shape != existing_group[var_name].shape:
                return False
            if reference_ds[var_name].dtype != existing_group[var_name].dtype:
                return False

        is_data_var = var_name in reference_ds.data_vars and not is_grid_mapping_variable(
            reference_ds, var_name
        )
        if is_data_var and "_ARRAY_DIMENSIONS" not in existing_group[var_name].attrs:
            return False

        if existing_group.rio.crs != reference_ds.rio.crs:
            return False

        if var_name in existing_group.data_vars and not is_grid_mapping_variable(
            existing_group, var_name
        ):
            arr = existing_group[var_name]
            if arr.size == 0:
                return False
            # Readability probe: dask fuses the slice, so one inner chunk is fetched.
            arr.isel({dim: 0 for dim in arr.dims}).values

        return True

    except Exception as e:
        print(f"Error validating variable {var_name}: {e}")
        return False


def all_chunks_written(store, path: str) -> bool:
    """
    Every chunk (or shard) of the array at *path* exists in the store.

    Costs one listing of the array's chunk keys; on sharded stores that is one key per
    shard. True proves the array complete; False does not prove it incomplete, because
    zarr does not store chunks that equal the fill value (``write_empty_chunks`` is off),
    so a sparse raster never reaches the full count.
    """
    import zarr

    z = zarr.open_array(store, path=path, mode="r")
    return z.nchunks_initialized == z.nchunks


def compute_overview_gcps(
    ds_gcp: xr.Dataset, scale_factor: float, width: int, height: int
) -> xr.Dataset:
    """Compute new GCPs for a given overview from the original GCPs."""
    ds_gcp_overview = (
        ds_gcp.assign_coords(
            line=np.round(ds_gcp.line / scale_factor).astype(np.int64),
            pixel=np.round(ds_gcp.pixel / scale_factor).astype(np.int64),
        )
        .pipe(lambda ds: ds.groupby(["line", "pixel"]))
        .mean()
        .rename_dims(line="azimuth_time", pixel="ground_range")
    )

    return ds_gcp_overview
