# SPDX-License-Identifier: Apache-2.0
"""Per-level facts of a pyramid written by any tool: CRS, pixel size against 2**L × native, dtype,
fill value, and whether nodata zeros were averaged into the overviews.

    .venv/bin/python bench/tools/check_levels.py OUT.zarr [--native 8.983152841195227e-05] [--var data]

Levels are found by walking the store's groups (numeric groups, r2/r4/..., ovr_2x/..., nested
under a child group); they are ordered by shape.
"""
from __future__ import annotations

import argparse

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
import zarr


def levels(path: str, var: str):
    root = zarr.open_group(path, mode="r", use_consolidated=False)
    found = []

    def walk(g: zarr.Group, prefix: str) -> None:
        names = [k for k in g.array_keys() if k == var] or [k for k in g.array_keys() if g[k].ndim == 2 and k not in ("x", "y", "X", "Y", "spatial_ref")]
        for k in names:
            found.append((prefix, k, g[k]))
        for k in g.group_keys():
            walk(g[k], f"{prefix}/{k}".strip("/"))

    walk(root, "")
    return sorted(found, key=lambda t: -t[2].shape[-1])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path"); p.add_argument("--native", type=float, default=None, help="native pixel size; default the first level's")
    p.add_argument("--var", default="data")
    a = p.parse_args()
    lv = levels(a.path, a.var)
    native = a.native
    print(f"{'level group':>22} {'shape':>16} {'dtype':>8} {'fill':>6} {'CRS':>9} {'pixel/native':>13} {'exact 2**L':>10} {'zero-frac':>9}")
    for i, (grp, name, arr) in enumerate(lv):
        try:
            ds = xr.open_dataset(a.path, group=grp or None, engine="zarr", consolidated=False, decode_coords="all")
            crs = ds.rio.crs; px = abs(ds.rio.transform().a)
        except Exception:  # noqa: BLE001
            crs, px = None, float("nan")
        if native is None and i == 0:
            native = px
        ratio = px / native if native else float("nan")
        exact = abs(ratio - round(ratio)) < 1e-9 and round(ratio) == 2 ** i
        sample = arr[: min(arr.shape[-2], 4096), : min(arr.shape[-1], 4096)] if arr.ndim == 2 else arr[0, : min(arr.shape[-2], 4096), : min(arr.shape[-1], 4096)]
        zero = float(np.mean(sample == 0)) if sample.size else float("nan")
        fill = arr.fill_value
        print(f"{(grp or '/') + '/' + name:>22} {str(arr.shape):>16} {str(arr.dtype):>8} {str(fill)[:6]:>6} {str(crs)[:9]:>9} {ratio:13.6f} {str(exact):>10} {zero:9.3f}")


if __name__ == "__main__":
    main()
