"""Utility functions for GeoZarr conversion."""

import dask.array as da
import numpy as np
import rioxarray  # noqa: F401  # Import to enable .rio accessor
import xarray as xr

VALID_FRACTION = 0.3
"""Minimum fraction of valid source pixels for an output pixel to be kept."""


def downsample_2d_array(
    source_data: da.Array,
    target_height: int,
    target_width: int,
    nodata_value: float | None = None,
    method: str = "mean",
) -> da.Array:
    """
    Downsample a 2-D Dask array to an exact output shape.

    The reduction is block-wise: with source chunks that are multiples of the
    block size it introduces no rechunk. Min and max are computed in the source
    dtype; median promotes to float32 for integer types up to 16 bits, float64
    otherwise; mean returns float64.
    """

    if source_data.ndim != 2:
        raise ValueError(f"Expected a 2-D array, got shape {source_data.shape}")

    source_height, source_width = source_data.shape

    if not 0 < target_height <= source_height:
        raise ValueError("target_height must be between 1 and source height")
    if not 0 < target_width <= source_width:
        raise ValueError("target_width must be between 1 and source width")

    if method == "nearest":
        # Pixel-centre mapping; preserves sampled NaN and numeric nodata values.
        y = np.minimum(
            ((np.arange(target_height) + 0.5) * source_height / target_height).astype(
                np.int64
            ),
            source_height - 1,
        )
        x = np.minimum(
            ((np.arange(target_width) + 0.5) * source_width / target_width).astype(
                np.int64
            ),
            source_width - 1,
        )

        return da.take(da.take(source_data, y, axis=0), x, axis=1)

    if method not in {"mean", "min", "max", "median"}:
        raise ValueError(f"Unknown method: {method}")

    block_height = source_height // target_height
    block_width = source_width // target_width

    # Exact output shape using integer blocks; bottom/right remainders are
    # discarded. The level transform accounts for this (constant pixel size).
    trimmed = source_data[
        : target_height * block_height,
        : target_width * block_width,
    ]

    blocks = trimmed.reshape(
        target_height,
        block_height,
        target_width,
        block_width,
    )
    reduction_axes = (1, 3)

    is_integer = np.issubdtype(blocks.dtype, np.integer)
    has_numeric_nodata = nodata_value is not None and not np.isnan(nodata_value)

    # NaN is always invalid, including when a numeric nodata value is configured.
    if is_integer:
        valid = da.ones(blocks.shape, dtype=bool, chunks=blocks.chunks)
    else:
        valid = ~da.isnan(blocks)
    if has_numeric_nodata:
        valid = valid & (blocks != nodata_value)

    valid_count = valid.sum(axis=reduction_axes)
    required_count = VALID_FRACTION * block_height * block_width
    keep = valid_count >= required_count

    if method == "mean":
        total = da.where(valid, blocks, 0).sum(axis=reduction_axes)
        result = total / da.maximum(valid_count, 1)
    elif method in {"min", "max"}:
        if is_integer:
            info = np.iinfo(blocks.dtype)
            fill = info.max if method == "min" else info.min
            masked = da.where(valid, blocks, blocks.dtype.type(fill))  # stays in dtype
            result = (
                masked.min(axis=reduction_axes)
                if method == "min"
                else masked.max(axis=reduction_axes)
            )
        else:
            masked = da.where(valid, blocks, np.nan)
            reducer = da.nanmin if method == "min" else da.nanmax
            result = reducer(masked, axis=reduction_axes)
    else:  # median
        work_dtype = (
            np.float32
            if (is_integer and blocks.dtype.itemsize <= 2) or blocks.dtype == np.float32
            else np.float64
        )
        masked = da.where(valid, blocks.astype(work_dtype), work_dtype(np.nan))
        result = da.nanmedian(masked, axis=reduction_axes)

    output_nodata = nodata_value if has_numeric_nodata else np.nan

    return da.where(keep, result, output_nodata)


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
    Strict completeness check: every chunk (or shard) of the array at *path* exists.

    Costs one listing of the array's chunk keys; on sharded stores that is one
    key per shard.
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
