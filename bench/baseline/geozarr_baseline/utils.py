"""Utility functions for GeoZarr conversion."""

import dask.array as da
import numpy as np
import rasterio  # noqa: F401  # Import to enable .rio accessor
import xarray as xr
import warnings


def downsample_2d_array(
    source_data: da.Array,
    target_height: int,
    target_width: int,
    nodata_value: float | None = None,
    method: str = "mean",
) -> da.Array:
    """Downsample a 2-D Dask array to an exact output shape."""

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

    # ponytail: exact output shape using integer blocks; bottom/right remainders
    # are discarded. Use area resampling if those pixels must contribute.
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

    # NaN is always invalid, including when a numeric nodata value is configured.
    valid = ~da.isnan(blocks)
    if nodata_value is not None and not np.isnan(nodata_value):
        valid &= blocks != nodata_value

    valid_count = valid.sum(axis=reduction_axes)
    required_count = 0.3 * block_height * block_width
    keep = valid_count >= required_count

    if method == "mean":
        total = da.where(valid, blocks, 0).sum(axis=reduction_axes)
        result = total / da.maximum(valid_count, 1)
    else:
        masked = da.where(valid, blocks, np.nan)

        reducer = {
            "min": da.nanmin,
            "max": da.nanmax,
            "median": da.nanmedian,
        }[method]

        result = reducer(masked, axis=reduction_axes)

    output_nodata = (
        nodata_value
        if nodata_value is not None and not np.isnan(nodata_value)
        else np.nan
    )

    return da.where(keep, result, output_nodata)


def is_grid_mapping_variable(ds: xr.Dataset, var_name: str) -> bool:
    """
    Check if a variable is a grid_mapping variable by looking for references to it.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to check
    var_name : str
        Variable name to check

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

    This ensures that Zarr chunks align properly with the data dimensions,
    preventing chunk overlap issues when writing with Dask.

    Parameters
    ----------
    dimension_size : int
        Size of the dimension to chunk
    target_chunk_size : int
        Desired chunk size

    Returns
    -------
    int
        Aligned chunk size that divides evenly into dimension_size
    """
    if target_chunk_size >= dimension_size:
        return dimension_size

    # Find the largest divisor of dimension_size that is <= target_chunk_size
    for chunk_size in range(target_chunk_size, int(target_chunk_size * 0.51), -1):
        if dimension_size % chunk_size == 0:
            return chunk_size

    # If no divisor is found, return the closest value to target_chunk_size
    return min(target_chunk_size, dimension_size)


def validate_existing_band_data(
    existing_group: xr.Dataset, var_name: str, reference_ds: xr.Dataset
) -> bool:
    """
    Validate that a specific band exists and is complete in the dataset.

    Parameters
    ----------
    existing_group : xarray.Dataset
        Existing dataset to validate
    var_name : str
        Name of the variable to validate
    reference_ds : xarray.Dataset
        Reference dataset structure for comparison

    Returns
    -------
    bool
        True if the variable exists and is valid, False otherwise
    """
    try:
        # Check if the variable exists
        if (
            var_name not in existing_group.data_vars
            and var_name not in existing_group.coords
        ):
            return False

        # Check shape matches
        if var_name in reference_ds.data_vars:
            expected_shape = reference_ds[var_name].shape
            existing_shape = existing_group[var_name].shape

            if expected_shape != existing_shape:
                return False

        # Check required attributes for data variables
        if var_name in reference_ds.data_vars and not is_grid_mapping_variable(
            reference_ds, var_name
        ):
            required_attrs = ["_ARRAY_DIMENSIONS", "standard_name"]
            for attr in required_attrs:
                if attr not in existing_group[var_name].attrs:
                    return False

        # Check rio CRS
        if existing_group.rio.crs != reference_ds.rio.crs:
            return False

        # Basic data integrity check for data variables
        if var_name in existing_group.data_vars and not is_grid_mapping_variable(
            existing_group, var_name
        ):
            try:
                # Just check if we can access the array metadata without reading data
                array_info = existing_group[var_name]
                if array_info.size == 0:
                    return False
                # read a piece of data to ensure it's valid
                test = array_info.isel(
                    {dim: 0 for dim in array_info.dims}
                ).values.mean()
                if np.isnan(test):
                    return False
            except Exception as e:
                print(f"Error validating variable {var_name}: {e}")
                return False

        return True

    except Exception:
        return False


def compute_overview_gcps(
    ds_gcp: xr.Dataset, scale_factor: float, width: int, height: int
) -> xr.Dataset:
    """Compute new GCPs for a given overview from the original GCPs.

    Parameters
    ----------
    ds_gcp : xr.Dataset
        the original GCPs
    scale_factor : float
        Overview's scale factor
    width : int
        Overview's width
    height : int
        Overview's height

    Returns
    -------
    ds_gcp_overview : xr.Dataset
        A new dataset where GCPs line and pixel coordinates are updated
        for the overview, and where duplicate line/pixel GCPs are
        merged together by averaging their latitude, longitude and height.

    """
    ds_gcp_overview = (
        # compute the new decimated line/pixel coordinates
        # TODO: trim line values with height and pixel values with width?
        ds_gcp.assign_coords(
            line=np.round(ds_gcp.line / scale_factor).astype(np.int64),
            pixel=np.round(ds_gcp.pixel / scale_factor).astype(np.int64),
        )
        # find duplicate line/pixel GCPs
        # and compute average for latitude, longitude and height
        .pipe(lambda ds: ds.groupby(["line", "pixel"])).mean()
        # re-assign original dimensions
        .rename_dims(line="azimuth_time", pixel="ground_range")
    )

    return ds_gcp_overview
