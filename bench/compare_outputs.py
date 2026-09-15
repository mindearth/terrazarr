"""Compare two pyramid outputs level by level (values and transforms), one shard at a time.

Usage: compare_outputs.py A B [--chunk 4096] [--threads 4]

Every level is streamed through dask in ``chunk``² blocks, so peak memory is about
``threads × chunk² × itemsize × 4`` (two blocks, their masked copies, the diff) whatever the
level size: 4 threads at 4096² float64 is about 2 GB. Loading a level with ``.values`` needs
five full copies of it instead (43 GB for a 32768² float64 level), which is what used to get
this script OOM-killed on the Italy window.
"""

from __future__ import annotations

import argparse

import dask
import dask.array as da
import xarray as xr
import zarr

from geozarr_pyramid.store import get_zarr_store


def _open(path: str, level: int, chunk: int) -> xr.Dataset:
    return xr.open_dataset(
        get_zarr_store(path),
        group=str(level),
        engine="zarr",
        consolidated=False,
        decode_coords="all",
        chunks={"y": chunk, "x": chunk},
    )


def main(a: str, b: str, chunk: int = 4096, threads: int = 4) -> None:
    ra = zarr.open_group(get_zarr_store(a), mode="r", use_consolidated=False)
    levels = sorted(int(k) for k in ra.group_keys())
    print(f"{'level':>5} {'shape':>14} {'max|diff|':>10} {'frac diff':>10} {'px baseline':>12} {'px optimized':>13}")
    with dask.config.set(scheduler="threads", num_workers=threads):
        for lv in levels:
            da_, db_ = _open(a, lv, chunk), _open(b, lv, chunk)
            va, vb = da_["data"].data.astype("float64"), db_["data"].data.astype("float64")
            d = da.fabs(da.nan_to_num(va) - da.nan_to_num(vb))
            dmax, frac = dask.compute(d.max(), (d > 0).mean())
            print(f"{lv:>5} {str(va.shape):>14} {float(dmax):>10.3g} {float(frac):>10.4f} "
                  f"{da_.rio.transform().a:>12.4f} {db_.rio.transform().a:>13.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--chunk", type=int, default=4096, help="dask block edge on y/x (default 4096, one shard)")
    p.add_argument("--threads", type=int, default=4, help="dask threads (default 4)")
    args = p.parse_args()
    main(args.a, args.b, args.chunk, args.threads)
