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
"""

import dataclasses
import itertools
import os
import time
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any

import dask.array as da
import numpy as np
import xarray as xr
import zarr
from pyproj import CRS
from rasterio.warp import calculate_default_transform
from zarr.codecs import BloscCodec
from zarr.core.sync import sync
from zarr.storage import StoreLike
from zarr.storage._common import make_store_path
from geozarr_baseline.store import get_storage_options
from geozarr_baseline.store import get_zarr_store, set_spatial_info
from geozarr_baseline import fs_utils, utils
from geozarr_baseline.types import (
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


def create_geozarr_dataset(
    dt_input: xr.DataTree,
    groups: Iterable[str],
    output_path: str,
    spatial_chunk: int = 4096,
    min_dimension: int = 256,
    tile_width: int = 256,
    max_retries: int = 3,
    crs_groups: Iterable[str] | None = None,
    gcp_group: str | None = None,
    enable_sharding: bool = False,
    method: str = "mean",
    nodata_value: float | None = None,
) -> xr.DataTree:
    """
    Create a GeoZarr-spec 0.4 compliant dataset from EOPF data.

    Parameters
    ----------
    dt_input : xr.DataTree
        Input EOPF DataTree
    groups : list[str]
        List of group names to process as Geozarr datasets.
    output_path : str
        Output path for the Zarr store
    spatial_chunk : int, default 4096
        Spatial chunk size for encoding
    min_dimension : int, default 256
        Minimum dimension for overview levels
    tile_width : int, default 256
        Tile width for TMS compatibility
    max_retries : int, default 3
        Maximum number of retries for network operations
    crs_groups : Iterabl[str], optional
        Iterable of group names that need CRS information added on best-effort basis
    gcp_group : str, optional
        Group name where GCPs (Ground Control Points) are located.
    enable_sharding : bool, default False
        Enable zarr sharding for spatial dimensions of each variable

    Returns
    -------
    xr.DataTree
        DataTree containing the GeoZarr compliant data
    """
    dt = dt_input.copy()
    compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle", blocksize=0)

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
        output_path,
        compressor,
        spatial_chunk,
        min_dimension,
        tile_width,
        max_retries,
        crs_groups,
        gcp_group,
        enable_sharding,
        method=method,
        nodata_value=nodata_value,
    )

    # Consolidate metadata at the root level AFTER all groups are written
    print("Consolidating metadata at root level for consistent zarr access...")
    try:
        zarr_group = fs_utils.open_zarr_group(output_path, mode="r+")
        consolidate_metadata(zarr_group.store)
        print("✅ Root level metadata consolidation completed")
    except Exception as e:
        print(f"⚠️ Warning: Root level consolidation failed: {e}")

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

        if gcp_group is not None:
            ds_gcp = dt[gcp_group].to_dataset()
        else:
            ds_gcp = None

        # Process all variables in the group
        for var_name in ds.data_vars:
            print(f"  Processing variable / band: {var_name}")

            # Set CF standard name and _ARRAY_DIMENSIONS
            if _is_sentinel1(dt):
                ds[var_name].attrs[
                    "standard_name"
                ] = "surface_backwards_scattering_coefficient_of_radar_wave"
                ds[var_name].attrs["units"] = "1"
            # else:  # Default to optical data standard name
            #     ds[var_name].attrs["standard_name"] = "built_area_fraction"

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
    output_path: str,
    compressor: Any,
    spatial_chunk: int = 4096,
    min_dimension: int = 256,
    tile_width: int = 256,
    max_retries: int = 3,
    crs_groups: Iterable[str] | None = None,
    gcp_group: str | None = None,
    enable_sharding: bool = False,
    method: str = "mean",
    nodata_value: float | None = None,
) -> xr.DataTree:
    """
    Iteratively copy groups from original DataTree to GeoZarr DataTree.

    Parameters
    ----------
    dt_input : xarray.DataTree
        Input DataTree to copy from
    geozarr_groups : dict[str, xr.Dataset]
        Dictionary of GeoZarr groups to process
    output_path : str
        Output path for the Zarr store
    compressor : Any
        Compressor to use for encoding
    spatial_chunk : int, default 4096
        Spatial chunk size for encoding
    min_dimension : int, default 256
        Minimum dimension for overview levels
    tile_width : int, default 256
        Tile width for TMS compatibility
    max_retries : int, default 3
        Maximum number of retries for network operations
    crs_groups : Iterable[str], optional
        Iterable of group names that need CRS information added on best-effort basis
    gcp_group : str, optional
        Group name where GCPs (Ground Control Points) are located

    Returns
    -------
    xarray.DataTree
        Updated GeoZarr DataTree with copied groups and variables including multiscale children
    """
    # Create result DataTree and initialize storage
    dt_result = xr.DataTree()
    storage_options = get_storage_options(output_path)
    dt_result.to_zarr(
        output_path,
        mode="a",
        consolidated=False,
        compute=True,
        **storage_options,
    )

    written_groups: set[str] = set()
    reference_crs = None

    # Process all groups in the tree using iterative approach
    for relative_path, node in dt_input.subtree_with_keys:
        if relative_path == ".":
            if "/" in geozarr_groups:
                print("Processing '/' as GeoZarr root group")
                write_geozarr_group(
                    dt_input,
                    dt_result,
                    "/",
                    geozarr_groups["/"],
                    output_path,
                    spatial_chunk=spatial_chunk,
                    compressor=compressor,
                    max_retries=max_retries,
                    min_dimension=min_dimension,
                    tile_width=tile_width,
                    gcp_group=gcp_group,
                    enable_sharding=enable_sharding,
                    method=method,
                    nodata_value=nodata_value,
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
                output_path,
                spatial_chunk=spatial_chunk,
                compressor=compressor,
                max_retries=max_retries,
                min_dimension=min_dimension,
                tile_width=tile_width,
                gcp_group=gcp_group,
                enable_sharding=enable_sharding,
                method=method,
                nodata_value=nodata_value,
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

            # Set up encoding
            encoding = _create_encoding(ds, compressor, spatial_chunk)

            # Write the dataset
            group_param = current_group_path.lstrip("/") if current_group_path else None
            ds.to_zarr(
                output_path,
                group=group_param,
                mode="w",
                consolidated=False,
                zarr_format=3,
                encoding=encoding,
                **storage_options,
            )

            dt_result[relative_path] = xr.DataTree(ds)

        written_groups.add(current_group_path)

    return dt_result if isinstance(dt_result, xr.DataTree) else xr.DataTree(dt_result)


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
    output_path: str,
    spatial_chunk: int = 4096,
    compressor: Any = None,
    max_retries: int = 3,
    min_dimension: int = 256,
    tile_width: int = 256,
    gcp_group: str | None = None,
    enable_sharding: bool = False,
    method: str = "mean",
    nodata_value: float | None = None,
) -> xr.DataTree:
    """
    Write a group to a GeoZarr dataset with multiscales support.

    Parameters
    ----------
    dt_input : xr.DataTree
        The original DataTree
    dt_result : xr.DataTree
        Result DataTree to update
    group_name : str
        Name of the group to write
    ds : xarray.Dataset
        Dataset to write
    output_path : str
        Output path for the GeoZarr dataset
    spatial_chunk : int, default 4096
        Spatial chunk size
    compressor : Any, optional
        Compressor to use for encoding
    max_retries : int, default 3
        Maximum number of retries for writing
    min_dimension : int, default 256
        Minimum dimension for overview levels
    tile_width : int, default 256
        Tile width for TMS compatibility
    gcp_group : str, optional
        Group name where GCPs (Ground Control Points) are located
        in the input DataTree (ignored if ``dt_input`` does not
        correspond to a Sentinel-1 product)

    Returns
    -------
    xarray.DataTree
        The written GeoZarr DataTree with multiscale groups as children
    """
    print(f"\n=== Processing {group_name} with GeoZarr-spec compliance ===")

    # Normalize to north-up (descending y) convention.
    ds = _normalize_north_up(ds)

    for var in ds.data_vars:
        if {"y", "x"}.issubset(ds[var].dims):
            leading_dims = [d for d in ds[var].dims if d not in {"y", "x"}]
            ds[var] = ds[var].transpose(*leading_dims, "y", "x")
            ds[var].attrs["_ARRAY_DIMENSIONS"] = list(ds[var].dims)

    # Create a new container for the group
    dt = xr.DataTree()
    group_key = group_name.lstrip("/")
    if group_key:
        dt_result[group_key] = dt
    dt.attrs = ds.attrs.copy()

    # Create encoding for all variables
    encoding = _create_geozarr_encoding(
        ds, compressor, tile_width, spatial_chunk, enable_sharding
    )

    # Write native data in the group 0 (overview level 0)
    native_dataset_group_name = f"{group_name}/0"
    native_dataset_path = f"{output_path}/{native_dataset_group_name.lstrip('/')}"

    # Check for existing dataset
    existing_native_dataset = _load_existing_dataset(native_dataset_path)

    # Get data variables to check (excluding grid_mapping variables)
    _data_vars_to_check = [
        var for var in ds.data_vars if not utils.is_grid_mapping_variable(ds, var)
    ]

    if _is_level_complete(
        existing_native_dataset, _data_vars_to_check
    ) and _is_level_valid(existing_native_dataset):
        print(
            f"Level 0 already exists and is complete at {native_dataset_path}, "
            "loading from disk and skipping write..."
        )
        ds = _normalize_north_up(existing_native_dataset)
    else:
        # Write native data band by band
        success, ds = write_dataset_band_by_band_with_validation(
            ds,
            existing_native_dataset,
            output_path,
            encoding,
            max_retries,
            native_dataset_group_name,
            False,
        )
        if not success:
            raise RuntimeError(f"Failed to write all bands for {group_name}")

    # Validate level 0 orientation and rasterio consistency before building the pyramid.
    _validate_pyramid_level(ds, f"{group_name}/0")

    # Create GeoZarr-spec compliant multiscales
    if _is_sentinel1(dt_input):
        assert gcp_group is not None, "GCP group required for processing Sentinel-1"
        ds_gcp = dt_input[gcp_group].to_dataset()
        # For Sentinel-1, ds_gcp is set to None since data is now reprojected and doesn't need GCP handling
        ds_gcp = None
    else:
        ds_gcp = None

    try:
        print(f"Creating GeoZarr-spec compliant multiscales for {group_name}")
        create_geozarr_compliant_multiscales(
            ds=ds,
            output_path=output_path,
            group_name=group_name,
            min_dimension=min_dimension,
            tile_width=tile_width,
            spatial_chunk=spatial_chunk,
            ds_gcp=ds_gcp,
            enable_sharding=enable_sharding,
            method=method,
            nodata_value=nodata_value,
        )
    except Exception as e:
        print(
            f"Warning: Failed to create GeoZarr-spec compliant multiscales for {group_name}: {e}"
        )
        print("Continuing with next group...")

    # Consolidate metadata
    print(f"  Consolidating metadata for group {group_name}...")
    _group_suffix = group_name.lstrip("/")
    group_path = fs_utils.normalize_path(
        f"{output_path}/{_group_suffix}" if _group_suffix else output_path
    )
    zarr_group = fs_utils.open_zarr_group(group_path, mode="r+")
    consolidate_metadata(zarr_group.store)
    print("  ✅ Metadata consolidated")

    return dt


def create_geozarr_compliant_multiscales(
    ds: xr.Dataset,
    output_path: str,
    group_name: str,
    min_dimension: int = 256,
    tile_width: int = 256,
    spatial_chunk: int = 4096,
    ds_gcp: xr.Dataset | None = None,
    enable_sharding: bool = False,
    method: str = "mean",
    nodata_value: float | None = None,
) -> dict[str, Any]:
    """
    Create GeoZarr-spec compliant multiscales following the specification exactly.

    Parameters
    ----------
    ds : xarray.Dataset
        Source dataset with all variables
    output_path : str
        Output path for the Zarr store
    group_name : str
        Name of the resolution group
    min_dimension : int, default 256
        Minimum dimension for overview levels
    tile_width : int, default 256
        Tile width for TMS compatibility
    spatial_chunk : int, default 4096
        Spatial chunk size for encoding
    ds_gcp : xr.Dataset, optional
        Source dataset with Sentinel-1 ground control points
        at native resolution

    Returns
    -------
    dict
        Dictionary with overview levels information
    """
    compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")

    # Get spatial information from the first data variable
    data_vars = [
        var for var in ds.data_vars if not utils.is_grid_mapping_variable(ds, var)
    ]
    if not data_vars:
        return {}

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
            # TODO: check GCP bounds vs. raster data bounds?
            # Below we compute GCP bbox and assume that it roughly corresponds
            # to the data bounds, which might be too crude / wrong approximation.
            # Alternatively we could check GCPs' line/pixel values and adjust
            # the bounds if we know approx the resolution.
            native_bounds = (
                ds_gcp["longitude"].values.min(),
                ds_gcp["latitude"].values.min(),
                ds_gcp["longitude"].values.max(),
                ds_gcp["latitude"].values.max(),
            )

    print(f"Creating GeoZarr-compliant multiscales for {group_name}")
    print(f"Native resolution: {native_width} x {native_height}")
    print(f"Native CRS: {native_crs}")

    # Calculate overview levels
    overview_levels = calculate_overview_levels(
        native_width, native_height, min_dimension, tile_width
    )

    print(f"Total overview levels: {len(overview_levels)}")
    for ol in overview_levels:
        print(
            f"Overview level {ol['level']}: {ol['width']} x {ol['height']} "
            f"(scale factor: {ol['scale_factor']})"
        )

    # Create native CRS tile matrix set
    tile_matrix_set = create_native_crs_tile_matrix_set(
        native_crs, native_bounds, overview_levels, None
    )

    # Create tile matrix limits
    tile_matrix_limits = _create_tile_matrix_limits(overview_levels, tile_width)

    # Build GeoZarr official multiscales layout (https://github.com/zarr-conventions/multiscales)
    left, bottom, right, top = native_bounds
    # Always use north-up (descending y, e < 0) convention so that
    # rasterio.windows.from_bounds() never raises "Bounds and transform
    # are inconsistent" (its check: (bottom-top)/transform.e > 0 requires e < 0).
    y_origin = top
    layout: list[dict[str, Any]] = []
    for ol in overview_levels:
        lv = ol["level"]
        w, h = ol["width"], ol["height"]
        # Affine transform in Rasterio/Affine order [a, b, c, d, e, f]
        # a=pixel_width, b=row_rot(0), c=x_origin, d=col_rot(0), e=pixel_height(<0 north-up), f=y_origin
        pix_w = (right - left) / w
        pix_h_mag = (top - bottom) / h
        pix_h = -pix_h_mag  # north-up: negative pixel height
        lv_spatial_transform = [pix_w, 0.0, left, 0.0, pix_h, y_origin]
        entry: dict[str, Any] = {
            "asset": str(lv),
            "transform": {
                "scale": [1.0, 1.0] if lv == 0 else [2.0, 2.0],
                "translation": [0.0, 0.0],
            },
            "spatial:shape": [h, w],
            "spatial:transform": lv_spatial_transform,
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
    native_pix_w = (right - left) / native_width
    native_pix_h = -((top - bottom) / native_height)  # north-up: negative
    spatial_attrs: dict[str, Any] = {
        "spatial:dimensions": ["y", "x"],
        "spatial:bbox": [left, bottom, right, top],
        "spatial:transform": [native_pix_w, 0.0, left, 0.0, native_pix_h, y_origin],
        "spatial:registration": "pixel",
    }

    # Add multiscales metadata to the group
    zarr_json_path = fs_utils.normalize_path(f"{output_path}/{group_name}/zarr.json")
    zarr_json = fs_utils.read_json_metadata(zarr_json_path)
    zarr_json_attributes = zarr_json.get("attributes", {})
    zarr_json_attributes["zarr_conventions"] = GEOZARR_CONVENTIONS
    # multiscales convention: only "layout" and "resampling_method" are spec-defined keys
    # (https://github.com/zarr-conventions/multiscales)
    zarr_json_attributes["multiscales"] = {
        "layout": layout,
        "resampling_method": method,
        # Keep OGC TileMatrixSet info as additional properties (useful for OGC clients)
        "tile_matrix_set": tile_matrix_set,
        "tile_matrix_limits": tile_matrix_limits,
    }
    zarr_json_attributes.update(proj_attrs)
    zarr_json_attributes.update(spatial_attrs)
    # OGC TileMatrixSet info kept as top-level group attributes (not inside multiscales)
    # for OGC API – Tiles client interoperability
    zarr_json_attributes["tile_matrix_set"] = tile_matrix_set
    zarr_json_attributes["tile_matrix_limits"] = tile_matrix_limits
    zarr_json["attributes"] = zarr_json_attributes
    fs_utils.write_json_metadata(zarr_json_path, zarr_json)

    print(f"Added GeoZarr-spec compliant multiscales metadata to {group_name}")

    # Create overview levels as children groups
    timing_data = []
    overview_datasets = {}

    # Read level 0 from disk to avoid keeping the original lazy dataset in memory.
    # Each subsequent level will also be read from disk after writing.
    level_0_path = fs_utils.normalize_path(f"{output_path}/{group_name}/0")
    previous_level_ds = _load_existing_dataset(level_0_path)
    if previous_level_ds is None:
        previous_level_ds = ds
    # Level 0 may have been written before the north-up fix; normalize so
    # that downsample_2d_array receives rows in the correct (descending y) order.
    previous_level_ds = _normalize_north_up(previous_level_ds)
    _validate_pyramid_level(previous_level_ds, f"{group_name}/0")

    for overview in overview_levels:
        level = overview["level"]
        if isinstance(level, str):
            level = int(level)

        # Skip level 0 - native resolution is already in group 0
        if level == 0:
            print("Skipping level 0 - native resolution is already in group 0")
            continue

        # Check if this overview level already exists on disk
        _existing_level_path = fs_utils.normalize_path(
            f"{output_path}/{group_name}/{level}"
        )
        _existing_level_ds = _load_existing_dataset(_existing_level_path)
        if _is_level_complete(_existing_level_ds, data_vars) and _is_level_valid(
            _existing_level_ds
        ):
            print(
                f"Overview level {level} already exists and is complete, "
                "loading from disk and skipping creation..."
            )
            overview_datasets[level] = _existing_level_ds
            previous_level_ds = _normalize_north_up(_existing_level_ds)
            continue

        width = overview["width"]
        height = overview["height"]
        scale_factor = overview["scale_factor"]

        print(f"\nCreating overview level {level} (1:{scale_factor} scale)...")
        print(f"Target dimensions: {width} x {height}")
        print(
            f"  Using pyramid approach: creating level {level} from level {level - 1}"
        )

        if ds_gcp is not None:
            ds_gcp_overview = utils.compute_overview_gcps(
                ds_gcp, scale_factor, width, height
            )
        else:
            ds_gcp_overview = None

        # Create overview dataset
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
        )

        # with 2x downsampling from previous level, the chunk sizes are halved
        chunks = {"x": spatial_chunk, "y": spatial_chunk}

        if enable_sharding:
            # One Dask block per complete Zarr shard.
            chunks.update(
                {
                    dim: -1
                    for dim in overview_ds.dims
                    if dim not in {"x", "y"}
                }
            )

        overview_ds = overview_ds.chunk(chunks)

        # Create encoding for this overview level
        encoding = _create_geozarr_encoding(
            overview_ds, compressor, tile_width, spatial_chunk, enable_sharding
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
            overview_ds.rio.write_transform(
                _ov_tf, grid_mapping_name="spatial_ref", inplace=True
            )
        _validate_pyramid_level(overview_ds, level)

        # Write overview level
        overview_path = fs_utils.normalize_path(f"{output_path}/{group_name}/{level}")
        start_time = time.time()

        storage_options = get_storage_options(overview_path)
        print(f"Writing overview level {level} at {overview_path}")

        # Ensure the directory exists for local paths
        if not fs_utils.is_s3_path(overview_path):
            os.makedirs(os.path.dirname(overview_path), exist_ok=True)

        # Write the overview dataset
        overview_group = f"{group_name}/{level}"
        # When sharding enabled, let Dask rechunk to shard boundaries
        align_chunks_flag = True if not enable_sharding else False
        overview_ds.to_zarr(
            output_path,
            group=overview_group,
            mode="w",
            consolidated=False,
            zarr_format=3,
            encoding=encoding,
            align_chunks=align_chunks_flag,
            **storage_options,
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

        # Consolidate metadata
        group_path = fs_utils.normalize_path(
            f"{output_path}/{overview_group.lstrip('/')}"
        )
        zarr_group = fs_utils.open_zarr_group(group_path, mode="r+")
        consolidate_metadata(zarr_group.store)
        print(f"  ✅ Metadata consolidated for overview level {level}")

        # Read the just-written level back from disk to break the lazy chain
        # back to the original dataset, keeping memory usage constant.
        previous_level_ds = _load_existing_dataset(overview_path)
        if previous_level_ds is None:
            print(
                f"  ⚠️ Could not reload level {level} from disk, using in-memory dataset"
            )
            previous_level_ds = overview_ds
        else:
            previous_level_ds = _normalize_north_up(previous_level_ds)

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


def calculate_overview_levels(
    native_width: int,
    native_height: int,
    min_dimension: int = 256,
    tile_width: int = 256,
) -> list[OverviewLevelJSON]:
    """
    Calculate overview levels following COG /2 downsampling logic.

    Parameters
    ----------
    native_width : int
        Width of the native resolution data
    native_height : int
        Height of the native resolution data
    min_dimension : int, default 256
        Stop creating overviews when dimension is smaller than this
    tile_width : int, default 256
        Tile width for TMS compatibility calculations

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
        zoom_for_width = max(0, int(np.ceil(np.log2(current_width / tile_width))))
        zoom_for_height = max(0, int(np.ceil(np.log2(current_height / tile_width))))
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
) -> TileMatrixSetJSON:
    """
    Create a custom Tile Matrix Set for the native CRS following GeoZarr spec.

    Parameters
    ----------
    native_crs : rasterio.crs.CRS
        Native CRS of the data
    native_bounds : tuple
        Native bounds (left, bottom, right, top)
    overview_levels : Iterable[OverViewLevelJSON]
        Iterable of overview level dictionaries
    group_prefix : str, optional
        Group prefix for the tile matrix IDs

    Returns
    -------
    dict
        Tile Matrix Set definition following OGC standard
    """
    left, bottom, right, top = native_bounds
    tile_matrices: list[TileMatrixJSON] = []

    for overview in overview_levels:
        level = overview["level"]
        width = overview["width"]
        height = overview["height"]

        # Calculate cell size
        cell_size_x = (right - left) / width
        cell_size_y = (top - bottom) / height
        cell_size = max(cell_size_x, cell_size_y)

        # Calculate scale denominator
        scale_denominator = cell_size * 3779.5275

        # Calculate matrix dimensions
        tile_width = overview["chunks"][1][0] if "chunks" in overview else 256
        tile_height = overview["chunks"][0][0] if "chunks" in overview else 256
        matrix_width = int(np.ceil(width / tile_width))
        matrix_height = int(np.ceil(height / tile_height))

        matrix_id = f"{group_prefix}/{level}" if group_prefix else str(level)

        tile_matrices.append(
            {
                "id": matrix_id,
                "scaleDenominator": scale_denominator,
                "cellSize": cell_size,
                "pointOfOrigin": [left, top],
                "tileWidth": tile_width,
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
) -> xr.Dataset:
    """
    Create an overview dataset containing all variables for a specific level.

    Parameters
    ----------
    ds : xarray.Dataset
        Source dataset
    level : int
        Overview level number
    width : int
        Width of this overview level
    height : int
        Height of this overview level
    native_crs : rasterio.crs.CRS
        Native CRS of the data
    native_bounds : tuple
        Native bounds (left, bottom, right, top)
    data_vars : Sequence[Hashable]
        Sequence of data variable names to include
    ds_gcp : xr.Dataset, optional
        Source dataset with Sentinel-1 ground control points
        at native resolution

    Returns
    -------
    xarray.Dataset
        Overview dataset with all variables
    """
    from affine import Affine

    left, bottom, right, top = native_bounds

    # Always use north-up (descending y, pixel_size_y < 0) so that rasterio
    # spatial operations (clip, reproject, from_bounds) never raise
    # "Bounds and transform are inconsistent". The source dataset is
    # guaranteed to have been flipped to descending y before reaching here
    # (see write_geozarr_group normalization step).
    pixel_size_x = (right - left) / width
    pixel_size_y_mag = (top - bottom) / height
    y_origin, pixel_size_y = top, -pixel_size_y_mag

    overview_transform = Affine(pixel_size_x, 0.0, left, 0.0, pixel_size_y, y_origin)

    # Pixel-centre coordinates using the closed-form formula, always float64.
    # Casting to the parent coordinate dtype could reduce precision and cause
    # rasterio to detect inconsistent bounds/transform at higher zoom levels.
    x_coords = left + (np.arange(width) + 0.5) * pixel_size_x
    y_coords = top - (np.arange(height) + 0.5) * pixel_size_y_mag

    x_coords = np.asarray(x_coords, dtype=np.float64)
    y_coords = np.asarray(y_coords, dtype=np.float64)

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
        lon_attrs = _get_lon_coord_attrs()
        lat_attrs = _get_lat_coord_attrs()
        overview_coords = {
            "x": (["x"], x_coords, lon_attrs),
            "y": (["y"], y_coords, lat_attrs),
        }

    else:
        x_attrs = _get_x_coord_attrs()
        y_attrs = _get_y_coord_attrs()

        overview_coords = {
            "x": (["x"], x_coords, x_attrs),
            "y": (["y"], y_coords, y_attrs),
        }

    for dim in ds.dims:
        if dim not in {"y", "x"} and dim in ds.coords:
            overview_coords[dim] = ds.coords[dim]

    spatial_dims = ["y", "x"]

    # Find the grid_mapping variable name
    grid_mapping_var_name = _find_grid_mapping_var_name(ds, data_vars)

    # Downsample all data variables
    overview_data_vars = {}
    for var in data_vars:
        print(f"  Downsampling {var}...")

        source_data = ds[var]  # .values
        data = source_data.data
        data_dtype = data.dtype

        if not isinstance(data, da.Array):
            data = da.from_array(data, chunks=data.chunks)

        # Create downsampled data
        if source_data.ndim == 3:
            non_spatial_dim = next(dim for dim in source_data.dims if dim not in spatial_dims)
            dims = [non_spatial_dim, *spatial_dims]

            # Normalize (y, x, band) → (band, y, x), lazily
            data_3d = source_data.transpose(*dims).data

            downsampled_data = da.stack(
                [
                    utils.downsample_2d_array(
                        data_3d[i],
                        height,
                        width,
                        method=method,
                        nodata_value=nodata_value,
                    )
                    for i in range(data_3d.shape[0])
                ],
                axis=0,
            )
        else:
            downsampled_data = utils.downsample_2d_array(
                data, height, width, method=method, nodata_value=nodata_value
            )
            dims = spatial_dims

        attrs = {
            "_ARRAY_DIMENSIONS": dims,
            "grid_mapping": grid_mapping_var_name,
        }
        if "standard_name" in ds[var].attrs:
            attrs["standard_name"] = ds[var].attrs["standard_name"]

        overview_data_vars[var] = (dims, downsampled_data.astype(data_dtype), attrs)

    # Create overview dataset
    overview_ds = xr.Dataset(overview_data_vars, coords=overview_coords)

    # Set CRS using rioxarray first
    overview_ds.rio.write_crs(native_crs, inplace=True)
    overview_ds.attrs["grid_mapping"] = grid_mapping_var_name

    # Add grid_mapping variable after setting CRS
    # TODO: refactor? grid mapping attributes and variables are handled
    # below and above in different function bodies in a confusing way.
    # ds.rio.write_crs may conflict with manual metadata handling
    # (i.e., rioxarray writes grid_mapping attributes to Xarray encoding, not attrs)
    # --
    _add_grid_mapping_variable(
        overview_ds, ds, grid_mapping_var_name, overview_transform, native_crs
    )

    return overview_ds


def write_dataset_band_by_band_with_validation(
    ds: xr.Dataset,
    existing_dataset: xr.Dataset | None,
    output_path: str,
    encoding: dict[Hashable, XarrayEncodingJSON],
    max_retries: int,
    group_name: str,
    force_overwrite: bool = False,
) -> tuple[bool, xr.Dataset]:
    """
    Write dataset band by band with individual band validation.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to write
    existing_dataset : xarray.Dataset, optional
        Existing dataset on the target Zarr store
    output_path : str
        Path to the output Zarr store
    encoding : dict
        Encoding configuration for variables
    max_retries : int
        Maximum number of retries for each band
    group_name : str
        Name of the group (for logging)
    force_overwrite : bool, default False
        Force overwrite existing bands even if they're valid

    Returns
    -------
    tuple[bool, xarray.Dataset]
        (True if all bands were written successfully, updated dataset)
    """
    print(
        f"Writing GeoZarr-spec compliant base resolution for {group_name} band by band with validation"
    )

    # Get data variables
    data_vars = [
        var for var in ds.data_vars if not utils.is_grid_mapping_variable(ds, var)
    ]

    successful_vars = []
    failed_vars = []
    skipped_vars = []

    store_exists = existing_dataset is not None and len(existing_dataset.data_vars) > 0

    store_storage_options = get_storage_options(output_path)
    fs = fs_utils.get_filesystem(output_path)

    def cleanup_prefix(prefix: str) -> None:
        key = prefix.lstrip("/")
        base_path = output_path.rstrip("/")
        if fs_utils.is_s3_path(base_path):
            target_path = fs_utils.normalize_path(f"{base_path}/{key}")
        else:
            target_path = os.path.join(base_path, key)
        try:
            fs.rm(target_path, recursive=True)
        except FileNotFoundError:
            pass
        except Exception as cleanup_error:
            print(f"    ⚠️ Failed to remove {target_path}: {cleanup_error}")

    # Write data variables one by one with validation
    for var in data_vars:
        # Check if this variable already exists and is valid
        if not force_overwrite and store_exists:
            if utils.validate_existing_band_data(existing_dataset, var, ds):
                ds.drop_vars(str(var))
                ds[var] = existing_dataset[var]  # type: ignore
                print(f"  ✅ Band {var} already exists and is valid, skipping")
                skipped_vars.append(var)
                successful_vars.append(var)
                continue
            # Remove invalid existing variable using filesystem-agnostic method
            print(f"    🧹 Removing invalid existing variable {var}...")
            cleanup_prefix(f"{group_name.lstrip('/')}/{var}")

        print(f"  Writing data variable {var}...")

        # Create a single-variable dataset with its coordinates
        single_var_ds = ds[[var]]

        # Create encoding for this variable only
        var_encoding = {}
        if var in encoding:
            var_encoding[var] = encoding[var]

        # Add coordinate encoding if not already present
        for coord in single_var_ds.coords:
            if coord in encoding and (
                existing_dataset is None or coord not in existing_dataset.coords
            ):
                var_encoding[coord] = encoding[coord]

        # Try to write this variable with retries
        success = False
        for attempt in range(max_retries):
            try:
                # Ensure the dataset is properly chunked to align with encoding
                if (
                    var in var_encoding
                    and "shards" in var_encoding[var]
                    and var_encoding[var]["shards"] is not None
                ):
                    # For sharded variables, use the shards dimensions
                    shard_dims = var_encoding[var].get("shards", None)
                    if shard_dims is not None:
                        var_dims = single_var_ds[var].dims
                        chunk_dict = {}
                        for i, dim in enumerate(var_dims):
                            if i < len(shard_dims):
                                chunk_dict[dim] = shard_dims[i]
                        single_var_ds[var] = single_var_ds[var].chunk(chunk_dict)
                elif var in var_encoding and "chunks" in var_encoding[var]:
                    target_chunks = var_encoding[var]["chunks"]
                    # Create chunk dict using the actual dimensions of the variable
                    var_dims = single_var_ds[var].dims
                    chunk_dict = {}
                    for i, dim in enumerate(var_dims):
                        if i < len(target_chunks):
                            chunk_dict[dim] = target_chunks[i]
                    # Rechunk the dataset to match the target chunks
                    single_var_ds[var] = single_var_ds[var].chunk(chunk_dict)
                else:
                    single_var_ds[var] = single_var_ds[var].chunk()

                single_var_ds.to_zarr(
                    output_path,
                    group=group_name,
                    mode="a",
                    consolidated=False,
                    zarr_format=3,
                    encoding=var_encoding,
                    **store_storage_options,
                )

                print(f"    ✅ Successfully wrote {var}")
                successful_vars.append(var)
                success = True
                if existing_dataset is None:
                    group_path = fs_utils.normalize_path(
                        f"{output_path}/{group_name.lstrip('/')}"
                    )
                    existing_dataset = xr.open_dataset(
                        group_path,
                        mode="r",
                        engine="zarr",
                        decode_coords="all",
                        chunks="auto",
                        **store_storage_options,
                    )
                break

            except Exception as e:
                # Delete the started data array to avoid conflict on next attempt
                for written_var in var_encoding.keys():
                    target_components = [group_name.lstrip("/"), str(written_var)]
                    target_prefix = "/".join(
                        component for component in target_components if component
                    )
                    cleanup_prefix(target_prefix)
                if attempt < max_retries - 1:
                    print(
                        f"    ⚠️  Attempt {attempt + 1} failed for {var}: {e}, retrying in 2 seconds..."
                    )
                    time.sleep(2)
                else:
                    print(
                        f"    ❌ Failed to write {var} after {max_retries} attempts: {e}"
                    )
                    failed_vars.append(var)
                    break

        if not success:
            print(f"  Failed to write data variable {var}")

    # Consolidate metadata
    group_path = fs_utils.normalize_path(f"{output_path}/{group_name.lstrip('/')}")
    zarr_group = fs_utils.open_zarr_group(group_path, mode="r+")
    consolidate_metadata(zarr_group.store)

    print(f"  ✅ Metadata consolidated for {len(successful_vars)} variables")

    # Report results
    if failed_vars:
        print(
            f"❌ Failed to write {len(failed_vars)} variables for {group_name}: {failed_vars}"
        )
        print(
            f"✅ Successfully wrote {len(successful_vars) - len(skipped_vars)} new variables"
        )
        print(
            f"⏭️  Skipped {len(skipped_vars)} existing valid variables: {skipped_vars}"
        )
        return False, ds
    else:
        print(
            f"✅ Successfully processed all {len(successful_vars)} variables for {group_name}"
        )
        if skipped_vars:
            print(
                f"   - Wrote {len(successful_vars) - len(skipped_vars)} new variables"
            )
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
    """
    Consolidate metadata of all nodes in a hierarchy asynchronously.

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
    zarr.core.group.AsyncGroup
        The group with consolidated metadata
    """
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
    for coord_name in ds.coords:
        if coord_name == "x":
            # Check if this is geographic coordinates (EPSG:4326)
            if ds.rio.crs and ds.rio.crs.to_epsg() == 4326:
                ds[coord_name].attrs.update(
                    {
                        "_ARRAY_DIMENSIONS": ["x"],
                        "standard_name": "longitude",
                        "units": "degrees_east",
                        "long_name": "longitude",
                    }
                )
            else:
                ds[coord_name].attrs.update(
                    {
                        "_ARRAY_DIMENSIONS": ["x"],
                        "standard_name": "projection_x_coordinate",
                        "units": "m",
                        "long_name": "x coordinate of projection",
                    }
                )
        elif coord_name == "y":
            # Check if this is geographic coordinates (EPSG:4326)
            if ds.rio.crs and ds.rio.crs.to_epsg() == 4326:
                ds[coord_name].attrs.update(
                    {
                        "_ARRAY_DIMENSIONS": ["y"],
                        "standard_name": "latitude",
                        "units": "degrees_north",
                        "long_name": "latitude",
                    }
                )
            else:
                ds[coord_name].attrs.update(
                    {
                        "_ARRAY_DIMENSIONS": ["y"],
                        "standard_name": "projection_y_coordinate",
                        "units": "m",
                        "long_name": "y coordinate of projection",
                    }
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
            # Generic coordinate
            if "_ARRAY_DIMENSIONS" not in ds[coord_name].attrs:
                ds[coord_name].attrs["_ARRAY_DIMENSIONS"] = [coord_name]


def _setup_grid_mapping(ds: xr.Dataset, grid_mapping_var_name: str) -> None:
    """Set up spatial_ref variable with GeoZarr required attributes."""

    # Use standard CRS and transform if available
    if ds.rio.crs and "spatial_ref" in ds:
        ds["spatial_ref"].attrs["_ARRAY_DIMENSIONS"] = []
        if ds.rio.transform():
            transform_gdal = ds.rio.transform().to_gdal()
            transform_str = " ".join([str(i) for i in transform_gdal])
            ds["spatial_ref"].attrs["GeoTransform"] = transform_str

    # Update all data variables to reference the grid_mapping
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
    ds: xr.Dataset, compressor: Any, spatial_chunk: int, tile_width: int
) -> dict[Hashable, XarrayEncodingJSON]:
    """Create encoding for dataset variables."""
    encoding: dict[Hashable, XarrayEncodingJSON] = {}
    chunking: tuple[int, ...]
    for var in ds.data_vars:
        if hasattr(ds[var].data, "chunks"):
            current_chunks = ds[var].chunks
            if len(current_chunks) >= 2:
                chunking = tuple(
                    (
                        current_chunks[i][0]
                        if len(current_chunks[i]) > 0
                        else ds[var].shape[i]
                    )
                    for i in range(len(current_chunks))
                )
            else:
                chunking = (
                    (
                        current_chunks[0][0]
                        if len(current_chunks[0]) > 0
                        else ds[var].shape[0]
                    ),
                )
        else:
            data_shape = ds[var].shape
            if len(data_shape) >= 2:
                chunk_y = min(spatial_chunk, data_shape[-2])
                chunk_x = min(spatial_chunk, data_shape[-1])
                if len(data_shape) == 3:
                    chunking = (1, chunk_y, chunk_x)
                else:
                    chunking = (chunk_y, chunk_x)
            else:
                chunking = (min(spatial_chunk, data_shape[-1]),)

        encoding[var] = {"compressors": [compressor], "chunks": chunking}

    # Add coordinate encoding
    for coord in ds.coords:
        encoding[coord] = {"compressors": None}

    return encoding


def _create_geozarr_encoding(
    ds: xr.Dataset,
    compressor: Any,
    tile_width: int,
    spatial_chunk: int,
    enable_sharding: bool = False,
) -> dict[Hashable, XarrayEncodingJSON]:
    encoding: dict[Hashable, XarrayEncodingJSON] = {}

    for var in ds.data_vars:
        if utils.is_grid_mapping_variable(ds, var):
            encoding[var] = {"compressors": None}
            continue

        shape = ds[var].shape
        ndim = len(shape)

        # --------------------------------------------
        # 1) BASE CHUNKS = tile_width
        # --------------------------------------------
        if ndim == 3:
            chunks = [1, tile_width, tile_width]
        elif ndim == 2:
            chunks = [tile_width, tile_width]
        else:
            chunks = [tile_width]

        # --------------------------------------------
        # 🆕 NEW SAFETY FIX:
        # shrink chunk dims if larger than ds dims
        # --------------------------------------------
        for axis in range(ndim):
            if chunks[axis] > shape[axis]:
                print(
                    f"ℹ️  Adjusting chunk dim for var={var}, axis={axis}: "
                    f"{chunks[axis]} → {shape[axis]} (dataset smaller)"
                )
                chunks[axis] = shape[axis]

        # --------------------------------------------
        # 2) SHARDS (optional)
        # --------------------------------------------
        shards = None
        if enable_sharding:
            # skip sharding if spatial_chunk > dims
            if ndim == 3:
                shards = (shape[0], spatial_chunk, spatial_chunk)
            elif ndim == 2:
                shards = (spatial_chunk, spatial_chunk)
            else:  # 1D
                shards = (spatial_chunk,)

        # --------------------------------------------
        # 3) final encoding
        # --------------------------------------------
        encoding[var] = {
            "chunks": tuple(chunks),
            "compressors": compressor,
            "shards": shards,
        }

    # coordinates: no compression, no sharding
    for coord in ds.coords:
        encoding[coord] = {"compressors": None}

    return encoding


def _load_existing_dataset(path: str) -> xr.Dataset | None:
    """Load existing dataset if it exists."""
    try:
        if fs_utils.path_exists(path):
            ds = xr.open_dataset(
                get_zarr_store(path),
                zarr_format=3,
                engine="zarr",
                chunks="auto",
                decode_coords="all",
                # **storage_options,
            )
            return set_spatial_info(ds)
    except Exception as e:
        print(f"Warning: Could not open existing dataset at {path}: {e}")
    return None


def _create_tile_matrix_limits(
    overview_levels: Iterable[OverviewLevelJSON], tile_width: int
) -> dict[str, TileMatrixLimitJSON]:
    """Create tile matrix limits for overview levels."""
    tile_matrix_limits: dict[str, TileMatrixLimitJSON] = {}
    for ol in overview_levels:
        level_str = str(ol["level"])
        max_tile_col = int(np.ceil(ol["width"] / tile_width)) - 1
        max_tile_row = int(np.ceil(ol["height"] / tile_width)) - 1

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
            if native_crs.to_epsg():
                grid_mapping_attrs["spatial_ref"] = native_crs.to_wkt()
                grid_mapping_attrs["crs_wkt"] = native_crs.to_wkt()
            else:
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
    """
    Check whether a level dataset contains all required data variables.

    Parameters
    ----------
    level_ds : xr.Dataset or None
        Dataset loaded from disk, or None if the path does not exist.
    data_vars : Sequence[Hashable]
        Variable names that must all be present for the level to be
        considered complete.

    Returns
    -------
    bool
        True if *level_ds* is not None and every variable in *data_vars*
        is present in its ``data_vars``.
    """
    if level_ds is None:
        return False
    return all(var in level_ds.data_vars for var in data_vars)


def _is_sentinel1(dt: xr.DataTree) -> bool:
    """Return True if the input DataTree represents a Sentinel-1 product."""
    stac_props = dt.attrs.get("stac_discovery", {}).get("properties", {})
    if stac_props.get("product:type", "not-a-product").startswith("S01"):
        return True
    else:
        return False
