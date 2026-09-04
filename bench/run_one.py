"""
Run one implementation (baseline or optimized) on an input store and print JSON metrics.

Runs in-process (threaded dask workers) so store traffic and peak RSS of the
whole pipeline are captured by one process. Launch one process per run.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent


def count_files(path: str) -> int:
    n = 0
    for _root, _dirs, files in os.walk(path):
        n += len(files)
    return n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--impl", choices=["baseline", "optimized"], required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--chunk-size", type=int, default=4096)
    p.add_argument("--tile-width", type=int, default=256)
    p.add_argument("--method", default="mean")
    p.add_argument("--nodata", type=float, default=None)
    p.add_argument("--sharding", action="store_true")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if args.impl == "baseline":
        sys.path.insert(0, str(HERE / "baseline"))
        from geozarr_baseline.geozarr import create_geozarr_dataset
        from geozarr_baseline.store import get_zarr_store, set_spatial_info
    else:
        from geozarr_pyramid.geozarr import create_geozarr_dataset
        from geozarr_pyramid.store import get_zarr_store, set_spatial_info

    import shutil
    import xarray as xr
    import zarr
    from dask.distributed import Client, get_task_stream

    # --- instrument every LocalStore in the process (reads and writes, both impls)
    counters: Counter = Counter()
    LocalStore = zarr.storage.LocalStore
    _get, _set, _delete = LocalStore.get, LocalStore.set, LocalStore.delete

    async def get(self, key, prototype, byte_range=None):
        counters["get_chunk" if "/c/" in key or key.startswith("c/") else "get_meta"] += 1
        return await _get(self, key, prototype, byte_range)

    async def set_(self, key, value):
        counters["set_chunk" if "/c/" in key or key.startswith("c/") else "set_meta"] += 1
        return await _set(self, key, value)

    async def delete(self, key):
        counters["delete"] += 1
        return await _delete(self, key)

    LocalStore.get, LocalStore.set, LocalStore.delete = get, set_, delete

    client = Client(processes=False, n_workers=1, threads_per_worker=args.threads, dashboard_address=":0", silence_logs=50)

    if args.quiet:
        sys.stdout = open(os.devnull, "w")
    shutil.rmtree(args.output, ignore_errors=True)

    ds = xr.open_dataset(
        get_zarr_store(args.input), engine="zarr",
        chunks={"y": args.chunk_size, "x": args.chunk_size}, consolidated=False,
    )
    ds = set_spatial_info(ds)
    dt = xr.DataTree(ds)

    t0 = time.perf_counter()
    err = None
    task_stream = get_task_stream(client=client)
    task_stream.__enter__()
    try:
        create_geozarr_dataset(
            dt, groups=["/"], output_path=args.output,
            spatial_chunk=args.chunk_size, min_dimension=args.tile_width, tile_width=args.tile_width,
            max_retries=1, enable_sharding=args.sharding, method=args.method, nodata_value=args.nodata,
        )
    except Exception as e:  # report, don't hide
        err = f"{type(e).__name__}: {e}"
    wall = time.perf_counter() - t0
    task_stream.__exit__(None, None, None)
    tasks_executed = len(task_stream.data)
    client.close()

    sys.stdout = sys.__stdout__
    print(json.dumps({
        "impl": args.impl,
        "ok": err is None,
        "error": err,
        "wall_s": round(wall, 2),
        "tasks_executed": tasks_executed,
        "store_get_chunk": counters["get_chunk"],
        "store_set_chunk": counters["set_chunk"],
        "store_get_meta": counters["get_meta"],
        "store_set_meta": counters["set_meta"],
        "store_delete": counters["delete"],
        "files_on_disk": count_files(args.output),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    }))


if __name__ == "__main__":
    main()
