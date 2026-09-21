"""Run the upstream EOPF converter (eopf-geozarr 0.11.0 from its git tag, `.venv-eopf`, Python 3.12) on a zarr input; print JSON metrics.

    .venv-eopf/bin/python bench/tools/run_eopf.py --input in.zarr --output out.zarr --shard-size 4096 --threads 8

The converter needs the data under a child group (a root-only tree writes nothing), so the
input dataset is placed at /measurements; level 0 is the group itself and the overviews are
its children r2, r4, ... (0.11.0 layout). The zarr chunk is the converter's own choice
(no tile-width parameter since 0.8); overviews are computed from the whole previous level
as one numpy array (`ds[var].values`).
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time


def count_files(path: str) -> int:
    return sum(len(f) for _, _, f in os.walk(path))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--shard-size", type=int, default=4096, help="eopf spatial_chunk")
    p.add_argument("--chunk-size", type=int, default=256, help="unused by 0.11.0 (kept for a uniform command line)")
    p.add_argument("--min-dimension", type=int, default=256)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()

    import shutil

    import rioxarray  # noqa: F401
    import xarray as xr
    from dask.distributed import Client
    from eopf_geozarr.conversion.geozarr import create_geozarr_dataset

    shutil.rmtree(a.output, ignore_errors=True)
    client = Client(processes=False, n_workers=1, threads_per_worker=a.threads, dashboard_address=":0", silence_logs=50)
    ds = xr.open_dataset(a.input, engine="zarr", chunks={"y": a.chunk_size, "x": a.chunk_size}, consolidated=False)
    ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y")
    if not ds.rio.crs:
        ds = ds.rio.write_crs("EPSG:4326")
    dt = xr.DataTree.from_dict({"/measurements": ds})
    if a.quiet:
        sys.stdout = open(os.devnull, "w")
        import logging
        logging.disable(logging.CRITICAL)
    t0 = time.perf_counter()
    err = None
    try:
        create_geozarr_dataset(dt, ["/measurements"], a.output, spatial_chunk=a.shard_size, min_dimension=a.min_dimension,
                               max_retries=1, enable_sharding=True)
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    wall = time.perf_counter() - t0
    client.close()
    sys.stdout = sys.__stdout__
    print(json.dumps({"impl": "eopf-geozarr", "ok": err is None, "error": err, "wall_s": round(wall, 2),
                      "objects": count_files(a.output), "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}))


if __name__ == "__main__":
    main()
