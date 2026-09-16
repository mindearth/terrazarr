"""A window of a large zarr on S3 (default s3://test/agea4.zarr) through the pipeline with 8 workers, with
per-task timings from the task stream.

    source bench/s3env.sh; .venv/bin/python bench/s3_window.py 32768 out.zarr [y0 x0] [s3://store] [compressor clevel]

The window origin defaults to the data-dense area found by listing agea4's stored shards
(bench/out/agea4_shards.json); pass y0 x0 to choose another.
"""
import collections
import os
import shutil
import sys
import time

import xarray as xr
import zarr
from dask.distributed import Client, get_task_stream

from terrazarr.geozarr import create_geozarr_dataset, make_compressor
from terrazarr.store import get_zarr_store, set_spatial_info

if __name__ == "__main__":
    n = int(sys.argv[1]); out = sys.argv[2]
    y0, x0 = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (10240, 1085440)
    src = sys.argv[5] if len(sys.argv) > 5 else "s3://test/agea4.zarr"
    comp, clevel = (sys.argv[6] if len(sys.argv) > 6 else "zstd"), (int(sys.argv[7]) if len(sys.argv) > 7 else 3)
    y0 -= y0 % 10240; x0 -= x0 % 10240            # shard-aligned in the source
    shutil.rmtree(out, ignore_errors=True)
    client = Client(processes=True, n_workers=8, threads_per_worker=1, dashboard_address=":0", silence_logs=50)
    ds = xr.open_dataset(get_zarr_store(src), engine="zarr", chunks={"y": 4096, "x": 4096}, consolidated=False)
    ds = set_spatial_info(ds.isel(y=slice(y0, y0 + n), x=slice(x0, x0 + n)))
    t0 = time.perf_counter()
    with get_task_stream(client=client) as ts:
        so = sys.stdout; sys.stdout = open(os.devnull, "w")
        create_geozarr_dataset(xr.DataTree(ds), groups=["/"], output_path=out, shard_size=4096, min_dimension=256, chunk_size=256,
                               max_retries=1, enable_sharding=True, method="mean", nodata_value=0, compressor=make_compressor(comp, clevel))
        sys.stdout = so
    wall = time.perf_counter() - t0
    per = collections.defaultdict(lambda: [0, 0.0, 0.0])
    for t in ts.data:
        name = str(t["key"]).split("-")[0].strip("('")[:24]
        for ss in t["startstops"]:
            if ss["action"] == "compute": per[name][1] += ss["stop"] - ss["start"]
            if ss["action"] == "transfer": per[name][2] += ss["stop"] - ss["start"]
        per[name][0] += 1
    tot = sum(v[1] for v in per.values()); shards = 4 * (n // 4096) ** 2
    g = zarr.open_group(out, mode="r", use_consolidated=False); l0 = g["0"]["z18"]
    size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(out) for f in fs)
    print(f"window {n}² {comp}-{clevel}: shards(all bands)={shards} tasks={len(ts.data)} wall={wall:.1f}s worker compute={tot:.1f}s ({tot/shards*1000:.0f} ms per shard) objects={sum(len(f) for _,_,f in os.walk(out))} bytes={size/1e6:.0f} MB l0 shards stored={l0.nchunks_initialized}")
    for name, (c, comp, tr) in sorted(per.items(), key=lambda kv: -kv[1][1])[:8]:
        print(f"  {name:24} n={c:6d} compute={comp:7.1f}s ({1000*comp/c:6.1f} ms/task) transfer={tr:5.1f}s")
    client.close()
