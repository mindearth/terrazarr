"""Compare a pyramid written by any tool with a reference pyramid, level by level, matched by shape.

    python bench/tools/compare_any.py REF.zarr OTHER.zarr [--var data] [--chunk 4096] [--threads 4]

Walks every group of OTHER, finds the arrays named --var (or the single array of a group for
GDAL, whose level-0 array carries the store name), and pairs each with the reference level of the
same shape. Prints max |diff| and the fraction of differing pixels, NaN counted as 0, streamed in
--chunk² dask blocks so any level fits in memory.
"""
from __future__ import annotations

import argparse

import dask
import dask.array as da
import numpy as np
import zarr


def arrays(root: zarr.Group, var: str) -> dict[str, zarr.Array]:
    out = {}
    def walk(g: zarr.Group, prefix: str) -> None:
        keys = list(g.array_keys())
        cands = [k for k in keys if k == var] or [k for k in keys if g[k].ndim == 2 and k not in ("x", "y", "X", "Y", "lat", "lon", "spatial_ref")]
        for k in cands:
            out[f"{prefix}/{k}".lstrip("/")] = g[k]
        for k in g.group_keys():
            walk(g[k], f"{prefix}/{k}")
    walk(root, "")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("ref"); p.add_argument("other")
    p.add_argument("--var", default="data"); p.add_argument("--chunk", type=int, default=4096); p.add_argument("--threads", type=int, default=4)
    p.add_argument("--crop", type=int, default=0, help="also pair a level up to this many pixels larger than the reference on each axis, comparing the top-left overlap (GDAL rounds overview sizes up)")
    a = p.parse_args()
    ref = {v.shape: v for v in arrays(zarr.open_group(a.ref, mode="r", use_consolidated=False), a.var).values()}
    other = arrays(zarr.open_group(a.other, mode="r", use_consolidated=False), a.var)
    print(f"{'other array':>28} {'shape':>16} {'ref?':>5} {'max|diff|':>10} {'frac diff':>10} {'dtype':>8}")
    with dask.config.set(scheduler="threads", num_workers=a.threads):
        for name, arr in sorted(other.items(), key=lambda kv: -kv[1].shape[0]):
            r = ref.get(arr.shape); tag = "yes"
            if r is None and a.crop:
                for sh, cand in ref.items():
                    if all(0 <= o - c <= a.crop for o, c in zip(arr.shape, sh)):
                        r = cand; tag = "crop"; break
            if r is None:
                print(f"{name:>28} {str(arr.shape):>16} {'no':>5} {'-':>10} {'-':>10} {str(arr.dtype):>8}"); continue
            x = da.from_array(arr, chunks=(a.chunk, a.chunk))[: r.shape[0], : r.shape[1]].astype("float64")
            y = da.from_array(r, chunks=(a.chunk, a.chunk)).astype("float64")
            d = da.fabs(da.nan_to_num(x) - da.nan_to_num(y))
            m, f = dask.compute(d.max(), (d > 0).mean())
            print(f"{name:>28} {str(arr.shape):>16} {tag:>5} {float(m):>10.3g} {float(f):>10.4f} {str(arr.dtype):>8}")


if __name__ == "__main__":
    main()
