# terrazarr

GeoZarr multiscale pyramids for planetary-scale rasters: a lazy, windowed and resumable writer
of zarr v3 stores with sharding, for any xarray object with a CRS.
Documentation: [mindearth.github.io/terrazarr](https://mindearth.github.io/terrazarr/).

## Why terrazarr

[eopf-geozarr](https://github.com/EOPF-Explorer/data-model), written by Development Seed for
ESA, turns Sentinel products into GeoZarr multiscale pyramids: native CRS, `/2` overviews, a tile
matrix set, consolidated metadata. It is built for a scene, a few thousand pixels a side, and it
computes every overview level from the whole previous level as one array in memory — fine at
scene size, but a design that runs out of memory or runs for days once the raster stops being
scene-sized.

terrazarr is a rewrite of that pipeline for rasters of any size, keeping the same metadata layout
and output shape, so it's a drop-in for anyone with an xarray object and a CRS who has outgrown
the scene-sized assumption. Reach for it if you need:

- **Low, flat memory use.** Peak memory doesn't grow with the raster: task memory is always one
  shard, and the main process' graph is bounded by a window rather than by the whole raster. A
  raster with a million-plus level-0 shard tasks still peaks under a gigabyte, where a writer
  that loads a whole level into memory first needs tens of gigabytes before it reads a pixel.
- **Real scalability.** The same lazy, windowed, shard-aligned execution applies unchanged from
  a single scene to a raster with billions of pixels — no separate code path or reconfiguration
  for "big" inputs, no ceiling where it stops working.
- **Lazy end to end.** Level 0 is one task per output shard, level 1 is reduced inside the same
  task, every further level reads its parent shards from the store one at a time; nothing is
  ever materialised as a whole array in memory.
- **Resumable.** Level 0 is written in windows and each one done is recorded, so an interrupted
  run — a killed job, a preempted spot instance — finishes from the missing windows instead of
  starting over.
- **Store-aware.** One read per input block, shard-aligned reads and writes, all-nodata blocks
  never touch the store, and a band-last image is read and transposed once per block.
- **Exact.** Level L has exactly `2**L` native pixels per pixel, anchored top-left; nodata is a
  rule (mean of valid pixels, block blanked below 30 % valid), integers round; every level has
  a decodable CRS.

## Install

```bash
pip install terrazarr
```

or from a clone, with [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e . --group dev
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

Local paths and `s3://` URLs work for both input and output. The [API](https://mindearth.github.io/terrazarr/api/) page lists every
option, the output layout and the conventions the store follows; [examples/](examples/)
has a synthetic raster that runs in under a minute, a GeoTIFF to GeoZarr script and an S3 to S3
run for a large sparse raster.

## Results at a glance

WSF-3D Italy, a 178 335 × 200 599 float64 raster (286 GB uncompressed) and its 32768² window,
written as a ten-level (eight-level) pyramid with 256² chunks in 4096² shards, same machine
(24 cores, 43 GB), same settings wherever a writer takes them:

**32768x32768 float64**

| writer | wall time | peak memory |
|---|---|---|
| terrazarr, 8 processes | 26 s | 0.5 GB |
| eopf-geozarr 0.11.0 | 460 s | 18.7 GB |
| topozarr 0.1.9 | 33 s | 2.9 GB |
| GDAL 3.13.3 | 34 s | 2.2 GB |

**178 335 × 200 599 float64**

| writer | wall time | peak memory |
|---|---|---|
| terrazarr, 8 processes | 314 s | 0.7 GB |
| eopf-geozarr 0.11.0 | stopped after 42 min, nothing written | 30 GB cap |
| topozarr 0.1.9 | 1006 s | 1.3 GB |
| GDAL 3.13.3 | 655 s | 8.8 GB |

Wall time and peak memory of the whole process tree. terrazarr keeps a numeric nodata out of
the overviews, GDAL does so when the source fill value is numeric, eopf-geozarr and topozarr
average it in. The full comparison, on synthetic rasters as
well, with the writers' characteristics and the correctness checks: [benchmarks](https://mindearth.github.io/terrazarr/benchmarks/). The
[demo](https://mindearth.github.io/terrazarr/demo/) serves the WSF-3D Italy pyramid from a public bucket through TiTiler.

## Authors and contributors

- Francesco Asaro, MindEarth
- Federico Oldani, MindEarth

## Origin and license

Derived from `eopf_geozarr.conversion.geozarr` of
[EOPF-Explorer/data-model](https://github.com/EOPF-Explorer/data-model), Copyright 2025 European
Space Agency (ESA), written by Development Seed, Apache License 2.0. terrazarr is licensed under
the Apache License 2.0; the changes against the original are listed in [CHANGELOG.md](CHANGELOG.md) and
credited in [NOTICE](NOTICE); see [LICENSE](LICENSE) and cite with [CITATION.cff](CITATION.cff).
