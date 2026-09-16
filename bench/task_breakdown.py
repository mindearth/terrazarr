"""Per-task compute time of the pipeline on a metadata-only (all-nodata) input of size N.

    .venv/bin/python bench/task_breakdown.py 65536 bench/data/scaling [--skip-empty]

Prints, from dask's task stream, the worker compute time per task family (rechunk, store,
_write_and_reduce, ...) and the busy fraction of the 8 workers. --skip-empty monkeypatches
the fused write task to bypass the zarr write for all-nodata blocks (zarr stores nothing for
them anyway) to measure that saving.
"""
import collections
import os
import shutil
import sys
import time

sys.path.insert(0, "bench")
import xarray as xr
from dask.distributed import Client, get_task_stream
from scaling import make_input  # noqa: E402

from terrazarr.geozarr import create_geozarr_dataset
from terrazarr.store import get_zarr_store, set_spatial_info

if __name__ == "__main__":
    n = int(sys.argv[1]); chunk = 4096
    inp = f"{sys.argv[2]}/ts_in_{n}.zarr"; out = f"{sys.argv[2]}/ts_out_{n}.zarr"
    make_input(inp, n); shutil.rmtree(out, ignore_errors=True)
    client = Client(processes=True, n_workers=8, threads_per_worker=1, dashboard_address=":0", silence_logs=50)
    ds = set_spatial_info(xr.open_dataset(get_zarr_store(inp), engine="zarr", chunks={"y": chunk, "x": chunk}, consolidated=False))
    t0 = time.perf_counter()
    with get_task_stream(client=client) as ts:
        so = sys.stdout; sys.stdout = open(os.devnull, "w")
        create_geozarr_dataset(xr.DataTree(ds), groups=["/"], output_path=out, shard_size=chunk, min_dimension=256, chunk_size=256, max_retries=1, enable_sharding=True, method="mean", nodata_value=0)
        sys.stdout = so
    wall = time.perf_counter() - t0
    per = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0])   # count, compute, transfer, deserialize
    first = min(t["startstops"][0]["start"] for t in ts.data if t["startstops"]); last = max(ss["stop"] for t in ts.data for ss in t["startstops"])
    for t in ts.data:
        name = str(t["key"]).split("-")[0].strip("('")[:28]
        for ss in t["startstops"]:
            i = {"compute": 1, "transfer": 2, "deserialize": 3}.get(ss["action"])
            if i: per[name][i] += ss["stop"] - ss["start"]
        per[name][0] += 1
    tot_c = sum(v[1] for v in per.values())
    print(f"n={n} shards={4*(n//chunk)**2} tasks={len(ts.data)} wall={wall:.1f}s task-stream span={last-first:.1f}s worker compute total={tot_c:.1f}s -> {tot_c/8:.1f}s per worker; busy fraction={tot_c/8/(last-first):.2f}")
    for name, (c, comp, tr, de) in sorted(per.items(), key=lambda kv: -kv[1][1])[:12]:
        print(f"  {name:28} n={c:6d} compute={comp:7.1f}s ({1000*comp/c:6.1f} ms/task) transfer={tr:5.1f}s deser={de:5.1f}s")
    client.close(); shutil.rmtree(inp, ignore_errors=True); shutil.rmtree(out, ignore_errors=True)
