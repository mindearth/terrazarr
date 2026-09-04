"""Compare baseline and optimized outputs level by level (values and transforms)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

from geozarr_pyramid.store import get_zarr_store


def main(a: str, b: str) -> None:
    ra = zarr.open_group(get_zarr_store(a), mode="r", use_consolidated=False)
    levels = sorted(int(k) for k in ra.group_keys())
    print(f"{'level':>5} {'shape':>14} {'max|diff|':>10} {'frac diff':>10} {'px baseline':>12} {'px optimized':>13}")
    for lv in levels:
        da_ = xr.open_dataset(get_zarr_store(a), group=str(lv), engine="zarr", consolidated=False, decode_coords="all")
        db_ = xr.open_dataset(get_zarr_store(b), group=str(lv), engine="zarr", consolidated=False, decode_coords="all")
        va, vb = da_["data"].values.astype("float64"), db_["data"].values.astype("float64")
        d = np.abs(np.nan_to_num(va) - np.nan_to_num(vb))
        print(f"{lv:>5} {str(va.shape):>14} {d.max():>10.3g} {(d > 0).mean():>10.4f} "
              f"{da_.rio.transform().a:>12.4f} {db_.rio.transform().a:>13.4f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
