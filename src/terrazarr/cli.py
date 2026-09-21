# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 MindEarth
"""Command line entry point: build a GeoZarr pyramid from a zarr input."""

from __future__ import annotations

import argparse
import logging
import os

import xarray as xr

from terrazarr.geozarr import make_compressor, to_geozarr
from terrazarr.store import get_zarr_store, set_spatial_info

log = logging.getLogger(__name__)


def get_dask_client(
    n_workers: int = 4,
    threads_per_worker: int | None = None,
    memory_limit: str | None = "auto",
):
    """
    Return a dask.distributed.Client.

    Connects through dask-gateway when DASK_GATEWAY_URL and DASK_GATEWAY_PASSWORD
    are set, otherwise starts a local cluster. Threads per worker and the memory
    limit are the two levers that trade concurrency against peak memory; the
    reduction holds roughly one shard per thread.
    """
    gateway_url = os.environ.get("DASK_GATEWAY_URL")
    gateway_password = os.environ.get("DASK_GATEWAY_PASSWORD")

    if gateway_url and gateway_password:
        from dask_gateway import Gateway
        from dask_gateway.auth import BasicAuth

        gateway = Gateway(gateway_url, auth=BasicAuth(username="anyuser", password=gateway_password))
        options = gateway.cluster_options()
        if threads_per_worker and "worker_cores" in options:
            options["worker_cores"] = threads_per_worker
        if memory_limit and memory_limit != "auto" and "worker_memory" in options:
            options["worker_memory"] = memory_limit
        cluster = gateway.new_cluster(options)
        cluster.scale(n_workers)
        client = cluster.get_client()
        client._me_gateway_cluster = cluster
        client.wait_for_workers(n_workers)
        return client

    from dask.distributed import Client

    return Client(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create GeoZarr dataset from Zarr input.")
    parser.add_argument("--input", required=True, help="Input Zarr path (local or s3://...)")
    parser.add_argument("--output", required=True, help="Output GeoZarr path (local or s3://...)")
    parser.add_argument("--chunk-size", type=int, default=256, help="Zarr chunk size on y/x")
    parser.add_argument(
        "--shard-size", type=int, default=4096,
        help="Zarr shard size on y/x with --sharding, and the dask block size on y/x in every case; "
        "a multiple of --chunk-size",
    )
    parser.add_argument(
        "--method", default="mean", choices=["mean", "min", "max", "median", "nearest"],
        help="Resampling method for pyramid levels",
    )
    parser.add_argument("--sharding", action="store_true", default=False, help="Enable zarr sharding")
    parser.add_argument("--nodata", type=float, default=None, help="Nodata value (NaN is always nodata)")
    parser.add_argument("--s3-profile", default=None, help="Kept for compatibility; credentials come from the environment")
    parser.add_argument(
        "--compressor", default="zstd", choices=["zstd", "lz4", "lz4hc", "blosclz", "zlib", "none"],
        help="Blosc codec for the data variables of every level (byte shuffle), or none",
    )
    parser.add_argument("--clevel", type=int, default=3, help="Blosc compression level")
    parser.add_argument(
        "--workers", type=int, default=min(8, os.cpu_count() or 4),
        help="Dask worker processes. Zarr serialises chunk assembly on one event loop per "
        "process, so several single-threaded workers beat one multi-threaded worker",
    )
    parser.add_argument(
        "--threads-per-worker", type=int, default=1,
        help="Threads per Dask worker; task memory is workers × threads × shard-size² × itemsize × 2",
    )
    parser.add_argument("--memory-limit", default="auto", help='Memory limit per worker, e.g. "12GB"')
    parser.add_argument(
        "--window-shards", type=int, default=None,
        help="Level 0 is written in windows of this many blocks per axis, one dask compute each "
        "(default 32, even): the graph in the main process is bounded by the window, about "
        "40 KB per block task, instead of growing with the raster",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    client = get_dask_client(
        n_workers=args.workers,
        threads_per_worker=args.threads_per_worker,
        memory_limit=args.memory_limit,
    )
    log.info("Dask dashboard: %s", client.dashboard_link)

    ds = xr.open_dataset(
        get_zarr_store(args.input, args.s3_profile),
        engine="zarr",
        chunks={"y": args.shard_size, "x": args.shard_size},
    )
    ds = set_spatial_info(ds)

    try:
        to_geozarr(
            ds,
            args.output,
            chunk_size=args.chunk_size,
            shard_size=args.shard_size,
            sharding=args.sharding,
            method=args.method,
            nodata=args.nodata,
            min_dimension=args.chunk_size,
            compressor=make_compressor(args.compressor, args.clevel),
            window_shards=args.window_shards,
            s3_profile=args.s3_profile,
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()
