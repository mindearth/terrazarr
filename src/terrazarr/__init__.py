# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 MindEarth
"""terrazarr: GeoZarr multiscale pyramids for planetary-scale rasters.

    from terrazarr import to_geozarr
    to_geozarr(ds, "out.zarr", chunk_size=256, shard_size=4096, method="mean", nodata=0)

or, on any Dataset or DataArray with a CRS, ``ds.terrazarr.to_geozarr("out.zarr", ...)``.
"""
from __future__ import annotations

import xarray as xr

from .geozarr import make_compressor, to_geozarr

__all__ = ["to_geozarr", "make_compressor"]


@xr.register_dataset_accessor("terrazarr")
class _DatasetAccessor:
    """``ds.terrazarr.to_geozarr(output, **options)``, see :func:`terrazarr.to_geozarr`."""

    def __init__(self, obj: xr.Dataset) -> None:
        self._obj = obj

    def to_geozarr(self, output: str, **options):
        return to_geozarr(self._obj, output, **options)


@xr.register_dataarray_accessor("terrazarr")
class _DataArrayAccessor:
    """``da.terrazarr.to_geozarr(output, **options)``, see :func:`terrazarr.to_geozarr`."""

    def __init__(self, obj: xr.DataArray) -> None:
        self._obj = obj

    def to_geozarr(self, output: str, **options):
        return to_geozarr(self._obj, output, **options)
