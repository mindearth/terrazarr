"""Memory and time of the pipeline against raster size, on metadata-only inputs.

    .venv/bin/python bench/scaling.py --sizes 16384,32768,65536,131072 --workers 8 --out bench/out/scaling.jsonl

For each size N a local zarr v3 array of shape (N, N, 4) uint8, dims (y, x, band), chunks
(2048, 2048, 4) in (10240, 10240, 4) shards is created with metadata only: no chunk is stored,
every read returns the fill value, so the pipeline's nodata fast path answers every block and
nothing is written. What remains is the cost that grows with the number of dask blocks: graph
construction in the client, scheduling, and the per-task fixed overheads. Sampled every 2 s:
RSS of the main process (client + scheduler) and of the worker processes.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
import xarray as xr
import zarr


def make_input(path: str, n: int) -> None:
    shutil.rmtree(path, ignore_errors=True)
    root = zarr.open_group(path, mode="w", zarr_format=3)
    root.create_array("z18", shape=(n, n, 4), dtype="uint8", chunks=(2048, 2048, 4), shards=(10240, 10240, 4),
                      fill_value=0, dimension_names=("y", "x", "band"))
    for name, vals, dims in (("y", 5e6 - np.arange(n) * 0.6, ("y",)), ("x", 1e6 + np.arange(n) * 0.6, ("x",)), ("band", np.arange(4), ("band",))):
        a = root.create_array(name, shape=vals.shape, dtype=vals.dtype, chunks=vals.shape, dimension_names=dims)
        a[:] = vals
    sr = root.create_array("spatial_ref", shape=(), dtype="int64")
    sr[()] = 0
    from pyproj import CRS
    wkt = CRS.from_epsg(3857).to_wkt()
    sr.attrs.update({"crs_wkt": wkt, "spatial_ref": wkt, "_ARRAY_DIMENSIONS": []})
    root["z18"].attrs["grid_mapping"] = "spatial_ref"


def sampler(log: list, stop: threading.Event) -> None:
    me = os.getpid(); t0 = time.time()
    while not stop.is_set():
        rows = [l.split(None, 3) for l in subprocess.run(["ps", "-eo", "pid,ppid,rss"], capture_output=True, text=True).stdout.splitlines()[1:]]
        kids = {int(r[0]) for r in rows if int(r[1]) == me}
        grand = {int(r[0]) for r in rows if int(r[1]) in kids}
        main = sum(int(r[2]) for r in rows if int(r[0]) == me)
        workers = sum(int(r[2]) for r in rows if int(r[0]) in kids | grand)
        log.append((round(time.time() - t0, 1), main / 2**20, workers / 2**20))
        stop.wait(2)


def run(n: int, workers: int, chunk: int, workdir: str) -> dict:
    from dask.distributed import Client, get_task_stream

    from geozarr_pyramid.geozarr import create_geozarr_dataset
    from geozarr_pyramid.store import get_zarr_store, set_spatial_info

    inp = os.path.join(workdir, f"scal_in_{n}.zarr"); out = os.path.join(workdir, f"scal_out_{n}.zarr")
    make_input(inp, n); shutil.rmtree(out, ignore_errors=True)
    samples: list = []; stop = threading.Event()
    threading.Thread(target=sampler, args=(samples, stop), daemon=True).start()
    client = Client(processes=workers > 1, n_workers=workers, threads_per_worker=1, dashboard_address=":0", silence_logs=50)
    t0 = time.perf_counter()
    ds = set_spatial_info(xr.open_dataset(get_zarr_store(inp), engine="zarr", chunks={"y": chunk, "x": chunk}, consolidated=False))
    nblocks = int(np.prod([len(c) for c in ds["z18"].chunks]))
    t_open = time.perf_counter() - t0
    ts = get_task_stream(client=client); ts.__enter__()
    sys.stdout = open(os.devnull, "w")
    err = None
    try:
        create_geozarr_dataset(xr.DataTree(ds), groups=["/"], output_path=out, spatial_chunk=chunk, min_dimension=256, tile_width=256,
                               max_retries=1, enable_sharding=True, method="mean", nodata_value=0)
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {str(e)[:200]}"
    sys.stdout = sys.__stdout__
    wall = time.perf_counter() - t0
    ts.__exit__(None, None, None)
    tasks = len(ts.data)
    stop.set(); client.close()
    shutil.rmtree(inp, ignore_errors=True); shutil.rmtree(out, ignore_errors=True)
    return {"n": n, "pixels_per_band": n * n, "bands": 4, "chunk": chunk, "workers": workers, "input_blocks": nblocks,
            "level0_shards_all_bands": 4 * (n // chunk) ** 2, "tasks_executed": tasks, "ok": err is None, "error": err,
            "open_s": round(t_open, 1), "wall_s": round(wall, 1),
            "main_peak_gb": round(max(s[1] for s in samples), 2), "workers_peak_gb": round(max(s[2] for s in samples), 2),
            "samples": samples[:: max(1, len(samples) // 40)]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", default="16384,32768,65536,131072")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--chunk", type=int, default=4096)
    p.add_argument("--workdir", default="bench/data/scaling")
    p.add_argument("--out", default="bench/out/scaling.jsonl")
    a = p.parse_args()
    os.makedirs(a.workdir, exist_ok=True)
    for n in (int(s) for s in a.sizes.split(",")):
        r = run(n, a.workers, a.chunk, a.workdir)
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")
        print({k: v for k, v in r.items() if k != "samples"}, flush=True)


if __name__ == "__main__":
    main()
