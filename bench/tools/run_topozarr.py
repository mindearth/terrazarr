"""Run topozarr (`.venv-topozarr`, Python 3.12) on a zarr input; print JSON metrics.

    .venv-topozarr/bin/python bench/tools/run_topozarr.py --input in.zarr --output out.zarr --levels 8 --workers 8

The source is opened lazily without dask (`chunks=None`), as topozarr's docs recommend for bounded
memory; it streams shard-aligned regions through its own thread pool and Rust kernel and picks
chunk and shard sizes itself (512 KB target chunks, 4 chunks per shard, snapped to the source
chunking). Levels are written as numeric groups with the same variable names as the source.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import time


def count_files(path: str) -> int:
    return sum(len(f) for _, _, f in os.walk(path))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--levels", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--method", default="mean")
    a = p.parse_args()

    import shutil

    import topozarr
    import xarray as xr
    import xproj  # noqa: F401
    import zarr

    shutil.rmtree(a.output, ignore_errors=True)
    ds = xr.open_zarr(a.input, chunks=None, consolidated=False)
    ds = ds.drop_vars("spatial_ref", errors="ignore").proj.assign_crs(spatial_ref="EPSG:4326")
    t0 = time.perf_counter()
    err = None
    stats = None
    try:
        pyr = topozarr.create_pyramid(ds, levels=a.levels, x_dim="x", y_dim="y", method=a.method)
        stats = pyr.write(a.output, max_workers=a.workers, stats=True)
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    wall = time.perf_counter() - t0
    layout = {}
    if err is None:
        g = zarr.open_group(a.output, mode="r", use_consolidated=False)
        for k in sorted(g.group_keys(), key=int):
            arr = g[k]["data"]
            layout[k] = {"shape": arr.shape, "chunks": arr.chunks, "shards": arr.shards}
    print(json.dumps({"impl": "topozarr", "ok": err is None, "error": err, "wall_s": round(wall, 2),
                      "objects": count_files(a.output), "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
                      "layout": layout, "stats": {k: {kk: (round(vv, 1) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk in ("wall_s", "workers", "region_shape", "regions", "skipped")} for k, v in (stats or {}).items()}}))


if __name__ == "__main__":
    main()
