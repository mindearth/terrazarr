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

Scaling design (see CHANGES.md):
- one zarr store object is used for every read, write and attribute update
- level N+1 is built from level N re-opened from the store, one dask block per shard
- the dask block is derived from the shard (or spatial_chunk), never from the tile width
- shards never exceed one slice on non-spatial dims, so task memory is bounded by spatial_chunk²
- level pixel size is exactly 2**level times the native pixel size
"""

import dataclasses
import time
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

from geozarr_pyramid import utils
from geozarr_pyramid.store import get_zarr_store, set_spatial_info
from geozarr_pyramid.types import (
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
    enc: Mapping[str, Any], dims: Sequence[Hashable], spatial_chunk: int
) -> dict[Hashable, int]:
    """
    Dask block for a variable, derived from its zarr encoding.

    Sharded: one block per shard, which zarr needs to write a shard in a single put
    instead of a read-modify-write. Unsharded: spatial_chunk on y/x (a multiple of the
    zarr chunk, so no rechunk is needed to align) and one slice on any other dim.
    """
    shards = enc.get("shards")
    if shards:
        return dict(zip(dims, shards))
    return {d: (spatial_chunk if d in SPATIAL_DIMS else 1) for d in dims}


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


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


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
    s3_profile: str | None = None,
    store: StoreLike | None = None,
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
        Output path for the Zarr store (local path or s3:// URL)
    spatial_chunk : int, default 4096
        Shard size on y/x when sharding is enabled, and the dask block size on y/x
        in every case. Must be a multiple of ``tile_width``.
    min_dimension : int, default 256
        Minimum dimension for overview levels
    tile_width : int, default 256
        Zarr chunk size on y/x (the tile served to clients)
    max_retries : int, default 3
        Maximum number of retries for network operations
    crs_groups : Iterable[str], optional
        Iterable of group names that need CRS information added on best-effort basis
    gcp_group : str, optional
        Group name where GCPs (Ground Control Points) are located.
    enable_sharding : bool, default False
        Enable zarr sharding for spatial dimensions of each variable
    method : str, default "mean"
        Resampling method: mean, min, max, median or nearest
    nodata_value : float, optional
        Nodata value; NaN is always treated as nodata
    s3_profile : str, optional
        Kept for CLI compatibility; credentials come from the environment
    store : zarr store, optional
        Store to write to. Built from ``output_path`` when omitted.

    Returns
    -------
    xr.DataTree
        DataTree containing the GeoZarr compliant data
    """
    if tile_width <= 0 or spatial_chunk <= 0:
        raise ValueError("tile_width and spatial_chunk must be positive")
    if spatial_chunk % tile_width:
        raise ValueError(
            f"spatial_chunk ({spatial_chunk}) must be a multiple of tile_width ({tile_width})"
        )

    dt = dt_input.copy()
    compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle", blocksize=0)
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
    store : zarr store
        Output store
    compressor : Any
        Compressor to use for encoding
    spatial_chunk, min_dimension, tile_width, max_retries, crs_groups, gcp_group,
    enable_sharding, method, nodata_value
        See :func:`create_geozarr_dataset`

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
            encoding = _create_encoding(ds, compressor, spatial_chunk)
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
        ds, compressor, tile_width, spatial_chunk, enable_sharding
    )

    # Write native data in the group 0 (overview level 0)
    level0_path = _join_path(group_path, 0)
    existing_native_dataset = _load_existing_dataset(store, level0_path, spatial_chunk)

    _data_vars_to_check = [
        var for var in ds.data_vars if not utils.is_grid_mapping_variable(ds, var)
    ]

    if _is_level_complete(
        existing_native_dataset, _data_vars_to_check
    ) and _is_level_valid(existing_native_dataset):
        print(
            f"Level 0 already exists and is complete at {level0_path}, "
            "loading from store and skipping write..."
        )
        ds = _normalize_north_up(existing_native_dataset)
    else:
        success, ds = write_dataset_band_by_band_with_validation(
            ds,
            existing_native_dataset,
            store,
            encoding,
            max_retries,
            level0_path,
            spatial_chunk,
            False,
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
            tile_width=tile_width,
            spatial_chunk=spatial_chunk,
            ds_gcp=ds_gcp,
            enable_sharding=enable_sharding,
            method=method,
            nodata_value=nodata_value,
        )
    except Exception as e:
        print(f"❌ Failed to create multiscales for {group_name}: {e}")
        raise

    # Consolidate metadata for the group
    print(f"  Consolidating metadata for group {group_name}...")
    consolidate_metadata(store, path=_group_or_none(group_path))
    print("  ✅ Metadata consolidated")

    return dt


def create_geozarr_compliant_multiscales(
    ds: xr.Dataset,
    store: StoreLike,
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

    Every level L has a pixel size of exactly 2**L native pixels anchored at the
    top-left corner (COG convention). Its extent is therefore shorter at the right
    and bottom by the rows/columns trimmed by the /2 reduction, which is what the
    reduced data actually covers.

    Returns
    -------
    dict
        Dictionary with overview levels information
    """
    compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")
    group_path = _group_path(group_name)

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
        native_width, native_height, min_dimension, tile_width
    )

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
    tile_matrix_limits = _create_tile_matrix_limits(overview_levels, tile_width)

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
    previous_level_ds = _load_existing_dataset(store, level_0_path, spatial_chunk)
    if previous_level_ds is None:
        previous_level_ds = ds
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
        _existing_level_ds = _load_existing_dataset(store, level_path, spatial_chunk)
        if _is_level_complete(_existing_level_ds, data_vars) and _is_level_valid(
            _existing_level_ds
        ):
            print(
                f"Overview level {level} already exists and is complete, "
                "loading from store and skipping creation..."
            )
            overview_datasets[level] = _existing_level_ds
            previous_level_ds = _normalize_north_up(_existing_level_ds)
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

        # Create overview dataset (lazy)
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
        )

        # Downsampled blocks are half a shard; merge 2x2 of them into one block per shard.
        # Non-spatial dims: one slice per block, so task memory is bounded by spatial_chunk².
        chunks: dict[Hashable, int] = {"x": spatial_chunk, "y": spatial_chunk}
        chunks.update({dim: 1 for dim in overview_ds.dims if dim not in SPATIAL_DIMS})
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
        _pin_grid_mapping(overview_ds)

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
        previous_level_ds = _load_existing_dataset(store, level_path, spatial_chunk)
        if previous_level_ds is None:
            print(
                f"  ⚠️ Could not reload level {level} from the store, using in-memory dataset"
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


def _level_pixel_size(native_px: tuple[float, float], level: int) -> tuple[float, float]:
    """Pixel size (w, h magnitudes) of overview *level*: exactly 2**level native pixels."""
    return native_px[0] * 2**level, native_px[1] * 2**level


def calculate_overview_levels(
    native_width: int,
    native_height: int,
    min_dimension: int = 256,
    tile_width: int = 256,
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
    native_px: tuple[float, float] | None = None,
) -> xr.Dataset:
    """
    Create an overview dataset containing all variables for a specific level (lazy).

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
        data = source_data.data
        data_dtype = data.dtype

        if not isinstance(data, da.Array):
            data = da.from_array(data, chunks="auto")

        if source_data.ndim == 3:
            non_spatial_dim = next(
                dim for dim in source_data.dims if dim not in spatial_dims
            )
            dims = [non_spatial_dim, *spatial_dims]

            # Normalize (y, x, band) → (band, y, x), lazily
            data_3d = source_data.transpose(*dims).data
            if not isinstance(data_3d, da.Array):
                data_3d = da.from_array(data_3d, chunks="auto")

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

        # Integer outputs: round to nearest instead of truncating toward zero.
        if np.issubdtype(data_dtype, np.integer) and np.issubdtype(
            downsampled_data.dtype, np.floating
        ):
            downsampled_data = da.rint(downsampled_data)

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
    _add_grid_mapping_variable(
        overview_ds, ds, grid_mapping_var_name, overview_transform, native_crs
    )

    return overview_ds


def write_dataset_band_by_band_with_validation(
    ds: xr.Dataset,
    existing_dataset: xr.Dataset | None,
    store: StoreLike,
    encoding: dict[Hashable, XarrayEncodingJSON],
    max_retries: int,
    group_name: str,
    spatial_chunk: int = 4096,
    force_overwrite: bool = False,
) -> tuple[bool, xr.Dataset]:
    """
    Write dataset band by band with individual band validation.

    Bands that already exist and validate are skipped. All remaining bands are written
    in a single dask compute; if that batch fails, each band that did not land is
    retried individually with cleanup between attempts.

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
    spatial_chunk : int
        Dask block size on y/x for unsharded variables
    force_overwrite : bool, default False
        Force overwrite existing bands even if they're valid

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

    for var in data_vars:
        if not force_overwrite and store_exists:
            if utils.validate_existing_band_data(existing_dataset, var, ds):
                ds[var] = existing_dataset[var]  # type: ignore[index]
                print(f"  ✅ Band {var} already exists and is valid, skipping")
                skipped_vars.append(var)
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
        single_var_ds[var] = single_var_ds[var].chunk(
            _dask_chunks_for(var_enc, single_var_ds[var].dims, spatial_chunk)
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
        print(f"  Writing {len(to_write)} data variable(s) in one compute: {to_write}")
        try:
            # Metadata (and numpy coordinates) are written eagerly by each call; the
            # chunk writes of every variable are computed together.
            delayed = []
            for i, var in enumerate(to_write):
                delayed.append(
                    _write_var(var, compute=False, with_coords=(i == 0 and not coords_on_store))
                )
                coords_on_store = True
            dask.compute(*delayed)
            written = list(to_write)
            for var in written:
                print(f"    ✅ Successfully wrote {var}")
        except Exception as e:
            print(f"    ⚠️ Batch write failed: {e}; re-validating per variable")
            reloaded = _load_existing_dataset(store, group_path, spatial_chunk)
            written = [
                v
                for v in to_write
                if reloaded is not None
                and utils.validate_existing_band_data(reloaded, v, ds)
            ]
            for var in [v for v in to_write if v not in written]:
                success = False
                for attempt in range(max_retries):
                    _delete_prefix(store, _join_path(group_path, str(var)))
                    try:
                        _write_var(var, compute=True, with_coords=False)
                        print(f"    ✅ Successfully wrote {var} (attempt {attempt + 1})")
                        written.append(var)
                        success = True
                        break
                    except Exception as e2:
                        if attempt < max_retries - 1:
                            print(
                                f"    ⚠️  Attempt {attempt + 1} failed for {var}: {e2}, retrying in 2 seconds..."
                            )
                            time.sleep(2)
                        else:
                            print(
                                f"    ❌ Failed to write {var} after {max_retries} attempts: {e2}"
                            )
                if not success:
                    failed_vars.append(var)

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
    ds: xr.Dataset, compressor: Any, spatial_chunk: int
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
                chunk_y = min(spatial_chunk, data_shape[-2])
                chunk_x = min(spatial_chunk, data_shape[-1])
                if len(data_shape) == 3:
                    chunking = (1, chunk_y, chunk_x)
                else:
                    chunking = (chunk_y, chunk_x)
            elif len(data_shape) == 1:
                chunking = (min(spatial_chunk, data_shape[-1]),)
            else:
                chunking = ()

        enc: XarrayEncodingJSON = {"compressors": [compressor]}
        if chunking:
            enc["chunks"] = chunking
        encoding[var] = enc

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
    """
    Zarr encoding for a GeoZarr level.

    Chunks are ``tile_width`` on y/x and 1 on any other dim, clipped to the array shape.
    Shards (optional) are ``spatial_chunk`` on y/x and 1 on any other dim, always a
    multiple of the chunk and never larger than the array, so task memory stays
    bounded by spatial_chunk² regardless of the time/band extent.
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

        base_chunks = [tile_width if d in SPATIAL_DIMS else 1 for d in dims]
        targets = [spatial_chunk if d in SPATIAL_DIMS else 1 for d in dims]

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
    store: StoreLike, path: str, spatial_chunk: int | None = None
) -> xr.Dataset | None:
    """
    Open a level group from the store, if it exists.

    With ``spatial_chunk`` the dask block is that size on y/x (one block per shard)
    and one slice on any other dim; otherwise dask's "auto" chunking is used.
    """
    try:
        if not _node_exists(store, path):
            return None
        chunks: Any = "auto"
        if spatial_chunk is not None:
            chunks = {"y": spatial_chunk, "x": spatial_chunk}
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
