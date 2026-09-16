# SPDX-License-Identifier: Apache-2.0
"""A large sparse raster from an S3 zarr store to an S3 GeoZarr pyramid, resumable.

    export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_ENDPOINT_URL=[DATA-3]
    python examples/03_s3_to_s3.py s3://bucket/input.zarr s3://bucket/pyramid.zarr --workers 8

The settings are the ones measured for a 2 M × 2 M four-band orthophoto (docs/benchmarks.md,
"Scaling"): 8 single-threaded workers, chunk 512 for imagery, shard 4096, level 0 in windows of
32² blocks so the main process stays under 1 GB. Run it detached (systemd-run, nohup, a job
scheduler); if it is interrupted, run it again with the same arguments and it finishes from the
windows already written.
"""
from __future__ import annotations

import argparse

import xarray as xr
from dask.distributed import Client

from terrazarr.geozarr import create_geozarr_dataset, make_compressor
from terrazarr.store import get_zarr_store, set_spatial_info


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("input"); p.add_argument("output")
    p.add_argument("--chunk-size", type=int, default=512); p.add_argument("--shard-size", type=int, default=4096)
    p.add_argument("--method", default="mean"); p.add_argument("--nodata", type=float, default=0)
    p.add_argument("--workers", type=int, default=8); p.add_argument("--memory-limit", default="6GB")
    p.add_argument("--window-shards", type=int, default=32)
    a = p.parse_args()

    client = Client(n_workers=a.workers, threads_per_worker=1, memory_limit=a.memory_limit, dashboard_address=":0")
    print("dashboard:", client.dashboard_link)
    try:
        ds = xr.open_dataset(get_zarr_store(a.input), engine="zarr", chunks={"y": a.shard_size, "x": a.shard_size}, consolidated=False)
        ds = set_spatial_info(ds)
        create_geozarr_dataset(
            xr.DataTree(ds), groups=["/"], output_path=a.output,
            chunk_size=a.chunk_size, shard_size=a.shard_size, enable_sharding=True,
            min_dimension=a.chunk_size, method=a.method, nodata_value=a.nodata,
            compressor=make_compressor("zstd", 3), window_shards=a.window_shards,
        )
    finally:
        client.close()
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
