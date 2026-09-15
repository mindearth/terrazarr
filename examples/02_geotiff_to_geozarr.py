# SPDX-License-Identifier: Apache-2.0
"""A GeoZarr pyramid from a GeoTIFF on local disk.

    python examples/02_geotiff_to_geozarr.py input.tif out.zarr [--workers 4]

A GeoTIFF is opened with rioxarray in shard-sized dask blocks; a COG (tiled) is read block by
block, a striped file needs GDAL's block cache to hold a block-row of strips (GDAL_CACHEMAX).
Sample input: [DATA-4] (a small public raster; any GeoTIFF with a CRS works).
"""
from __future__ import annotations

import argparse

import rioxarray
import xarray as xr
from dask.distributed import Client

from geozarr_pyramid.geozarr import create_geozarr_dataset, make_compressor
from geozarr_pyramid.store import set_spatial_info


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("input"); p.add_argument("output")
    p.add_argument("--chunk-size", type=int, default=256); p.add_argument("--shard-size", type=int, default=4096)
    p.add_argument("--method", default="mean"); p.add_argument("--nodata", type=float, default=None)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()

    client = Client(n_workers=a.workers, threads_per_worker=1, dashboard_address=":0", silence_logs=50)
    try:
        da = rioxarray.open_rasterio(a.input, chunks={"band": 1, "y": a.shard_size, "x": a.shard_size}, lock=False, masked=False)
        if da.sizes["band"] == 1:
            da = da.squeeze("band", drop=True)
        ds = da.to_dataset(name="data")
        ds["data"].attrs.pop("_FillValue", None); ds["data"].encoding.pop("_FillValue", None)
        ds = set_spatial_info(ds)
        create_geozarr_dataset(
            xr.DataTree(ds), groups=["/"], output_path=a.output,
            chunk_size=a.chunk_size, shard_size=a.shard_size, enable_sharding=True,
            min_dimension=a.chunk_size, method=a.method, nodata_value=a.nodata,
            compressor=make_compressor("zstd", 3),
        )
    finally:
        client.close()
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
