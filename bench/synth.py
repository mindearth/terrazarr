"""Synthetic input stores for tests and benchmarks (zarr v3, north-up, EPSG:3857)."""

from __future__ import annotations

import shutil
from pathlib import Path

import dask.array as da
import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from zarr.codecs import BloscCodec


def make_input(
    path: str | Path,
    shape: tuple[int, ...] = (2048, 2048),
    dtype: str = "uint8",
    input_chunk: int = 1024,
    nodata: float | None = 0,
    res: float = 10.0,
    origin: tuple[float, float] = (1_000_000.0, 5_000_000.0),
    seed: int = 0,
    nan_corner: bool = False,
) -> str:
    """
    Write a synthetic raster to *path* and return the path.

    shape: (y, x) or (t, y, x). Random values in [1, 250] with a diagonal band of
    nodata and a nodata block in the bottom-right corner, so the valid-fraction rule
    and nodata propagation are exercised.
    """
    path = str(path)
    shutil.rmtree(path, ignore_errors=True)
    h, w = shape[-2:]
    chunks = (1, input_chunk, input_chunk) if len(shape) == 3 else (input_chunk, input_chunk)

    rng = da.random.default_rng(seed)
    if np.issubdtype(np.dtype(dtype), np.integer):
        data = rng.integers(1, 250, size=shape, dtype=dtype, chunks=chunks)
    else:
        data = rng.random(size=shape, chunks=chunks).astype(dtype) * 250 + 1

    yy = da.arange(h, chunks=input_chunk)[:, None]
    xx = da.arange(w, chunks=input_chunk)[None, :]
    band = da.abs(yy - xx) < max(4, h // 64)              # diagonal stripe
    corner = (yy > h * 0.8) & (xx > w * 0.8)               # bottom-right block
    mask = band | corner
    fill = nodata if nodata is not None else np.nan
    data = da.where(mask, np.array(fill, dtype=data.dtype if nodata is not None else "float64"), data).astype(dtype)

    x = origin[0] + (np.arange(w) + 0.5) * res
    y = origin[1] - (np.arange(h) + 0.5) * res
    dims = ("t", "y", "x") if len(shape) == 3 else ("y", "x")
    coords = {"y": y, "x": x}
    if len(shape) == 3:
        coords["t"] = np.arange(shape[0])
    ds = xr.Dataset({"data": (dims, data)}, coords=coords)
    ds = ds.rio.write_crs("EPSG:3857")
    if nan_corner:
        ds["data"] = ds["data"].astype("float32")
        ds["data"][..., 0, 0] = np.nan
    ds.to_zarr(
        path,
        mode="w",
        zarr_format=3,
        consolidated=False,
        encoding={"data": {"chunks": chunks, "compressors": BloscCodec(cname="zstd", clevel=1)}},
    )
    return path
