# terrazarr

GeoZarr multiscale pyramids for planetary-scale rasters: a lazy, windowed and resumable writer
of zarr v3 stores with sharding, from a zarr or GeoTIFF input on disk or on S3.

## Why this exists

[eopf-geozarr](https://github.com/EOPF-Explorer/data-model), written by Development Seed for
ESA, turns Sentinel products into GeoZarr multiscale pyramids: native CRS, `/2` overviews, a tile
matrix set, consolidated metadata. It is built for a scene, a few thousand pixels a side, and it
holds every overview level in memory as one array while it works.

We needed the same output for rasters that are not scenes: a 178 335 × 200 599 float64 layer of
Italy, and a 20 cm orthophoto of the whole country, 2 263 040 × 2 191 360 pixels in four bands.
On those the scene-sized design either ran out of memory or ran for days. This project is the
rewrite of that pipeline for that scale, keeping its metadata layout and its function skeleton:

- **Lazy end to end.** Level 0 is one task per output shard, level 1 is reduced inside the same
  task, every further level reads its parent shards from the store one at a time. Task memory
  is one shard whatever the raster.
- **Windowed and resumable.** Level 0 is written in windows, one dask compute each, so the
  graph in the main process is bounded by the window (about 40 KB per block task) and not by
  the raster; each window done is recorded, an interrupted run finishes from the missing ones.
- **Store-aware.** One read per input block, shard-aligned reads and writes, all-nodata blocks
  never touch the store, and a band-last image is read and transposed once per block.
- **Exact.** Level L has exactly `2**L` native pixels per pixel, anchored top-left; nodata is a
  rule (mean of valid pixels, block blanked below 30 % valid), integers round; every level has
  a decodable CRS.

The result: full Italy in 6 minutes on 8 processes against 22 for the original, 2178 objects
instead of 430 602 for GDAL's pyramid of the same raster, and a main process that stays under
1 GB on the orthophoto where the one-graph design needed 70 GB. Numbers and methods are in
[docs/benchmarks.md](docs/benchmarks.md).

## Install

```bash
pip install terrazarr
```

or from a clone, with [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e . --group dev
```

Credentials for `s3://` paths come from the environment: `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, `AWS_ENDPOINT_URL` (optional, for MinIO and other S3-compatible
stores), `AWS_REGION` (optional).

## Use

```bash
terrazarr --input in.zarr --output out.zarr \
  --chunk-size 256 --shard-size 4096 --sharding \
  --method mean --nodata 0 \
  --workers 8 --threads-per-worker 1 --memory-limit 4GB \
  --compressor zstd --clevel 3
```

| option | meaning |
|---|---|
| `--chunk-size` | zarr chunk on y/x, the tile a client fetches (256 for analysis, 512 for imagery) |
| `--shard-size` | zarr shard on y/x with `--sharding`, i.e. how many chunks travel in one object, and the dask block in every case; a multiple of the chunk size |
| `--method`, `--nodata` | `mean`, `min`, `max`, `median` or `nearest`; NaN is always nodata, a numeric nodata is excluded from the reduction and a block below 30 % valid becomes nodata |
| `--workers`, `--threads-per-worker` | prefer several single-threaded worker processes: zarr assembles chunks on one event loop per process, so threads queue on it |
| `--window-shards` | level-0 blocks per axis and per window (default 32); the main process holds one window's graph at a time |
| `--compressor`, `--clevel` | blosc codec and level for every level's data |

Memory per worker is about `4 × shard-size² × bands × itemsize` for a band-last input, one shard
and its reduction otherwise. The input is a zarr store with `y`/`x` dims and a CRS, or anything
`xarray` opens the same way; the benchmark harness also reads GeoTIFFs through rioxarray.

From Python:

```python
import xarray as xr
from terrazarr.geozarr import create_geozarr_dataset
from terrazarr.store import get_zarr_store, set_spatial_info

ds = set_spatial_info(xr.open_dataset(get_zarr_store("in.zarr"), engine="zarr", chunks={"y": 4096, "x": 4096}))
create_geozarr_dataset(xr.DataTree(ds), groups=["/"], output_path="out.zarr",
                       chunk_size=256, shard_size=4096, enable_sharding=True, method="mean", nodata_value=0)
```

The output is a GeoZarr store: level groups `0`, `1`, … each with the data variables, `x`, `y`
and `spatial_ref`, a `multiscales` attribute with the layout and a native-CRS tile matrix set
on the root, consolidated metadata at every level. See [examples/](examples/) for runnable
scripts, including a synthetic raster that needs no download.

## Results at a glance

Against the original module on the same inputs and machine (24 cores, 43 GB, 1 Gbit/s to an
S3 store), chunk 256, shard 4096, mean, sharding:

| input | original | this module, 8 processes | speed-up |
|---|---:|---:|---:|
| WSF3Dv3 Italy, 178335×200599 float64, S3 | 1344 s | 461 s | 2.9× |
| WSF3Dv3 Italy, local NVMe | 1287 s | 346 s | 3.7× |
| Italy 32768² window, S3 | 61 s | 33 s | 1.9× |

Store traffic on the 32768² window at shard 8192: 22 240 chunk GETs for the original, 569 here.
Outputs are identical to the original at every level for min, median and float means; integer
means differ by at most 1 where the original truncated.

Against other GeoZarr pyramid writers on full Italy, 10 levels, local NVMe:

| writer | wall | objects | overview values |
|---|---:|---:|---|
| this module, 8 processes | 367 s | 2178 | nodata-aware mean |
| GDAL 3.13.2 (`gdal_translate` + `gdaladdo`) | 700 s | 430 602 | zeros averaged in, levels 1 px larger |
| topozarr 0.1.8 | 1087 s | 41 361 | zeros averaged in |
| eopf-geozarr 0.7.1 | cannot run | | holds each level in memory |

Memory on the orthophoto (1.18 M level-0 shard tasks): the one-graph design needed 45–55 GB in
the main process; windowed, it needs less than 1 GB. Full tables, the writer characteristics,
the input-format study (striped GeoTIFF, COG, zarr) and the scaling law:
[docs/benchmarks.md](docs/benchmarks.md).

## Documentation

- [Demo](https://mindearth.github.io/terrazarr/demo/): the WSF3D Italy pyramid on a map, tiles served from the public bucket by TiTiler
- [docs/benchmarks.md](docs/benchmarks.md): every measurement, how to reproduce it; inputs and references are public (`bench/fetch_data.py`)
- [docs/changes.md](docs/changes.md): the 21 changes against the original, with the finding each one addresses
- [docs/profile.md](docs/profile.md): the py-spy profile that motivated changes 12–17
- [docs/minio.md](docs/minio.md): what an S3-compatible store delivers and what it costs a run
- [docs/testing.md](docs/testing.md): the test suite

## Origin and license

terrazarr is derived from `eopf_geozarr.conversion.geozarr` of
[EOPF-Explorer/data-model](https://github.com/EOPF-Explorer/data-model), Copyright 2025 European
Space Agency (ESA), written by Development Seed, Apache License 2.0. The pipeline, the GeoZarr
metadata layout and several function names are theirs; the changes are listed in
[docs/changes.md](docs/changes.md) and credited in [NOTICE](NOTICE).

This project is licensed under the Apache License 2.0, see [LICENSE](LICENSE). Cite it with
[CITATION.cff](CITATION.cff).
