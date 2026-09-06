"""
Extract a window of a (striped) GeoTIFF into a zarr v3 input store, reading full-width
row strips in parallel so an untiled TIFF is decoded once.

    python bench/tif_to_zarr.py --src s3://bucket/x.tif --dst s3://bucket/x.zarr \
        --row0 74000 --col0 78000 --rows 32768 --cols 32768 --strip 2048 --chunk 2048
"""

from __future__ import annotations

import argparse
import os
import time

import dask
import dask.array as da
import numpy as np
import rasterio
import rioxarray  # noqa: F401
import xarray as xr
from rasterio.windows import Window
from zarr.codecs import BloscCodec

from geozarr_pyramid.store import get_zarr_store

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
os.environ.setdefault("GDAL_HTTP_MULTIRANGE", "YES")
os.environ.setdefault("CPL_VSIL_CURL_CHUNK_SIZE", str(16 * 1024 * 1024))
os.environ.setdefault("CPL_VSIL_CURL_CACHE_SIZE", str(512 * 1024 * 1024))
os.environ.setdefault("GDAL_CACHEMAX", "1024")


def read_strip(src: str, row0: int, nrows: int, col0: int, ncols: int, dtype: str) -> np.ndarray:
    with rasterio.open(src) as r:
        return r.read(1, window=Window(col0, row0, ncols, nrows)).astype(dtype, copy=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--row0", type=int, default=0)
    p.add_argument("--col0", type=int, default=0)
    p.add_argument("--rows", type=int, default=None, help="default: to the bottom edge")
    p.add_argument("--cols", type=int, default=None, help="default: to the right edge")
    p.add_argument("--strip", type=int, default=2048, help="rows per read")
    p.add_argument("--chunk", type=int, default=2048, help="zarr chunk on y/x")
    p.add_argument("--dtype", default=None, help="cast on read (default: source dtype)")
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()

    with rasterio.open(a.src) as r:
        H, W = r.height, r.width
        crs, tf = r.crs, r.transform
        dtype = a.dtype or r.dtypes[0]
        nodata = r.nodata
    rows = a.rows or (H - a.row0)
    cols = a.cols or (W - a.col0)
    assert a.row0 + rows <= H and a.col0 + cols <= W

    strips = []
    for r0 in range(a.row0, a.row0 + rows, a.strip):
        n = min(a.strip, a.row0 + rows - r0)
        strips.append(da.from_delayed(dask.delayed(read_strip)(a.src, r0, n, a.col0, cols, dtype), shape=(n, cols), dtype=dtype))
    data = da.concatenate(strips, axis=0).rechunk((a.chunk, a.chunk))

    res_x, res_y = tf.a, -tf.e
    x = tf.c + (a.col0 + np.arange(cols) + 0.5) * res_x
    y = tf.f - (a.row0 + np.arange(rows) + 0.5) * res_y
    ds = xr.Dataset({"data": (("y", "x"), data)}, coords={"y": y, "x": x})
    ds = ds.rio.write_crs(crs)
    # an explicit encoding= replaces the variable encoding where rioxarray keeps grid_mapping
    ds["data"].attrs["grid_mapping"] = ds["data"].encoding.pop("grid_mapping", "spatial_ref")
    if nodata is not None:
        ds["data"].attrs["_FillValue"] = nodata
    ds.attrs["source"] = a.src
    ds.attrs["window"] = [a.row0, a.col0, rows, cols]

    print(f"reading {rows}x{cols} {dtype} from {a.src} in {len(strips)} strips; writing {a.dst}", flush=True)
    t0 = time.time()
    with dask.config.set(scheduler="threads", num_workers=a.threads):
        ds.to_zarr(
            get_zarr_store(a.dst), mode="w", zarr_format=3, consolidated=True,
            encoding={"data": {"chunks": (a.chunk, a.chunk), "compressors": BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")}},
        )
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
