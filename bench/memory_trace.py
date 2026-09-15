"""CLI-equivalent run of the pipeline on a store with 8 workers, sampling the RSS of the main process
(dask client + scheduler) and of the worker processes every 10 s into a log.

    source bench/s3env.sh; .venv/bin/python bench/memory_trace.py out.zarr trace.log [s3://test/agea4.zarr]

Run it detached with a memory cap, e.g. systemd-run --user --unit x -p MemoryMax=30G ..., and stop
it once the curve is clear: on agea4 the main process reaches 29 GB in 19 min of graph construction.
"""
import os
import subprocess
import sys
import threading
import time

import xarray as xr

from geozarr_pyramid.cli import get_dask_client
from geozarr_pyramid.geozarr import create_geozarr_dataset, make_compressor
from geozarr_pyramid.store import get_zarr_store, set_spatial_info

OUT = sys.argv[1]; LOG = sys.argv[2]; SRC = sys.argv[3] if len(sys.argv) > 3 else "s3://test/agea4.zarr"
def sample(stop):
    me = os.getpid(); t0 = time.time()
    with open(LOG, "a") as f:
        while not stop.is_set():
            ps = subprocess.run(["ps", "-eo", "pid,ppid,rss,args"], capture_output=True, text=True).stdout.splitlines()[1:]
            rows = [l.split(None, 3) for l in ps]
            main = sum(int(r[2]) for r in rows if int(r[0]) == me)
            kids = {int(r[0]) for r in rows if int(r[1]) == me}
            grand = {int(r[0]) for r in rows if int(r[1]) in kids}
            workers = sum(int(r[2]) for r in rows if int(r[0]) in kids | grand)
            f.write(f"{time.time()-t0:7.0f}s main={main/1024/1024:6.2f}GB workers={workers/1024/1024:6.2f}GB n={len(kids|grand)}\n"); f.flush()
            stop.wait(10)


if __name__ == '__main__':
    stop = threading.Event(); threading.Thread(target=sample, args=(stop,), daemon=True).start()
    def mark(msg):
        with open(LOG, "a") as f: f.write(f"### {time.strftime('%H:%M:%S')} {msg}\n")
    mark("client start")
    client = get_dask_client(n_workers=8, threads_per_worker=1, memory_limit="auto")
    mark("open dataset")
    ds = xr.open_dataset(get_zarr_store(SRC), engine="zarr", chunks={"y": 4096, "x": 4096}, consolidated=False)
    ds = set_spatial_info(ds)
    mark(f"dataset open: {dict(ds.sizes)} var chunks {ds['z18'].chunks and [len(c) for c in ds['z18'].chunks]}")
    dt = xr.DataTree(ds)
    mark("create_geozarr_dataset")
    try:
        create_geozarr_dataset(dt, groups=["/"], output_path=OUT, shard_size=4096, min_dimension=256, chunk_size=256, max_retries=1,
                               enable_sharding=True, method="mean", nodata_value=0, compressor=make_compressor("zstd", 3))
        mark("done")
    except Exception as e:
        mark(f"failed: {type(e).__name__}: {str(e)[:300]}")
    finally:
        stop.set(); client.close()
