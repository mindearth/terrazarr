# SPDX-License-Identifier: Apache-2.0
"""A GeoZarr pyramid from a synthetic raster: no download, under a minute.

    python examples/01_synthetic.py [out_dir]

Writes a 2048² uint8 raster with a nodata border to a zarr store, builds the pyramid with 2
worker processes, then checks level 1 against the nodata-aware mean and prints the layout.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
import zarr
from dask.distributed import Client

from terrazarr import to_geozarr
from terrazarr.store import get_zarr_store, set_spatial_info
from terrazarr.utils import reduce_block


def make_raster(path: Path, n: int = 2048, res: float = 10.0) -> None:
    rng = np.random.default_rng(0)
    data = rng.integers(1, 250, (n, n), dtype="uint8")
    data[: n // 8, :] = 0          # a nodata band at the top: the valid-fraction rule shows there
    data[-512:, -512:] = 0         # an all-nodata block (one 512² shard): skipped, never stored
    ds = xr.Dataset(
        {"data": (("y", "x"), data)},
        coords={"y": 5_000_000.0 - np.arange(n) * res, "x": 1_000_000.0 + np.arange(n) * res},
    ).rio.write_crs("EPSG:3857")
    ds.to_zarr(path, mode="w", zarr_format=3, consolidated=False, encoding={"data": {"chunks": (512, 512)}})


def main(out_dir: str) -> None:
    out_dir = Path(out_dir)
    inp, out = out_dir / "synthetic_in.zarr", out_dir / "synthetic_pyramid.zarr"
    make_raster(inp)

    client = Client(n_workers=2, threads_per_worker=1, dashboard_address=":0", silence_logs=50)
    try:
        ds = xr.open_dataset(get_zarr_store(str(inp)), engine="zarr", chunks={"y": 512, "x": 512}, consolidated=False)
        ds = set_spatial_info(ds)
        to_geozarr(
            ds, str(out),
            chunk_size=128, shard_size=512, sharding=True, min_dimension=128,
            method="mean", nodata=0,
        )
    finally:
        client.close()

    root = zarr.open_group(get_zarr_store(str(out)), mode="r")
    layout = root.attrs["multiscales"]["layout"]
    print(f"{len(layout)} levels:")
    for entry in layout:
        arr = root[entry["asset"]]["data"]
        print(f"  level {entry['asset']}: shape {arr.shape}, chunks {arr.chunks}, shards {arr.shards}, pixel {entry['spatial:transform'][0]} m")

    l0 = root["0"]["data"][:]
    l1 = root["1"]["data"][:]
    expected = reduce_block(l0, 2, 2, "mean", nodata_value=0, out_dtype="uint8")
    assert np.array_equal(l1, expected), "level 1 is not the nodata-aware mean of level 0"
    assert not (out / "0" / "data" / "c" / "3" / "3").exists(), "the all-nodata shard should not be stored"
    print("level 1 equals the nodata-aware mean of level 0; the all-nodata shard was not stored")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="terrazarr-"))
