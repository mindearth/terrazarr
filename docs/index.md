# terrazarr

GeoZarr multiscale pyramids for planetary-scale rasters: a lazy, windowed and resumable writer
of zarr v3 stores with sharding, for any xarray object with a CRS.
Source, issues and releases: [github.com/mindearth/terrazarr](https://github.com/mindearth/terrazarr).

## Why terrazarr

[eopf-geozarr](https://github.com/EOPF-Explorer/data-model), written by Development Seed for
ESA, turns Sentinel products into GeoZarr multiscale pyramids: native CRS, `/2` overviews, a tile
matrix set, consolidated metadata. It is built for a scene, a few thousand pixels a side, and it
computes every overview level from the whole previous level as one array in memory.

We needed the same output for rasters that are not scenes: a 178 335 × 200 599 float64 layer of
Italy, and a 20 cm orthophoto of the whole country, 2 263 040 × 2 191 360 pixels in four bands.
On those the scene-sized design either ran out of memory or ran for days. terrazarr is the
rewrite of that pipeline for that scale, keeping its metadata layout and its function skeleton:

- **Lazy end to end.** Level 0 is one task per output shard, level 1 is reduced inside the same
  task, every further level reads its parent shards from the store one at a time. Task memory
  is one shard whatever the raster.
- **Windowed and resumable.** Level 0 is written in windows, one dask compute each, so the
  graph in the main process is bounded by the window and not by the raster; each window done
  is recorded, and an interrupted run finishes from the missing ones.
- **Store-aware.** One read per input block, shard-aligned reads and writes, all-nodata blocks
  never touch the store, and a band-last image is read and transposed once per block.
- **Exact.** Level L has exactly `2**L` native pixels per pixel, anchored top-left; nodata is a
  rule (mean of valid pixels, block blanked below 30 % valid), integers round; every level has
  a decodable CRS.

## Install

```bash
pip install terrazarr
```

## Use

`to_geozarr` is the counterpart of rioxarray's `to_raster` for GeoZarr: it takes a DataArray, a
Dataset or a DataTree with a CRS and writes the pyramid.

```python
import xarray as xr
from terrazarr import to_geozarr

ds = xr.open_dataset("in.zarr", engine="zarr", chunks={"y": 4096, "x": 4096})   # lazy, any size
to_geozarr(ds, "out.zarr", chunk_size=256, shard_size=4096, method="mean", nodata=0)
```

or as an accessor, `ds.terrazarr.to_geozarr("out.zarr", ...)`, or from the command line:

```bash
terrazarr --input in.zarr --output out.zarr \
  --chunk-size 256 --shard-size 4096 --sharding \
  --method mean --nodata 0 \
  --workers 8 --threads-per-worker 1 --memory-limit 4GB \
  --compressor zstd --clevel 3
```

Local paths and `s3://` URLs work for both input and output. The [API](api.md) page lists every
option, the output layout and the conventions the store follows; [examples/](https://github.com/mindearth/terrazarr/tree/main/examples)
has a synthetic raster that runs in under a minute, a GeoTIFF to GeoZarr script and an S3 to S3
run for a large sparse raster.

## Results at a glance

On the same inputs and machine, terrazarr against the three other open-source GeoZarr pyramid
writers: [benchmarks](benchmarks.md). The [demo](demo.md) serves the WSF-3D Italy pyramid from a
public bucket through TiTiler.

## Authors and contributors

- Francesco Asaro, MindEarth
- Federico Oldani, MindEarth

## Origin and license

Derived from `eopf_geozarr.conversion.geozarr` of
[EOPF-Explorer/data-model](https://github.com/EOPF-Explorer/data-model), Copyright 2025 European
Space Agency (ESA), written by Development Seed, Apache License 2.0. terrazarr is licensed under
the Apache License 2.0; the changes against the original are listed in the repository's
`CHANGELOG.md`.
