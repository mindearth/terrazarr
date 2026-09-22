# API

::: terrazarr.to_geozarr

## Accessors

Importing `terrazarr` registers the `terrazarr` accessor on `xarray.Dataset` and
`xarray.DataArray`, so any object with a CRS readable by rioxarray can be written directly:

```python
import terrazarr  # noqa: F401  (registers the accessor)
import xarray as xr

ds = xr.open_dataset("in.zarr", engine="zarr", chunks={"y": 4096, "x": 4096})
ds.terrazarr.to_geozarr("out.zarr", chunk_size=256, shard_size=4096, method="mean", nodata=0)
```

## Command line

```
terrazarr --input IN --output OUT [options]
```

| option | default | meaning |
|---|---|---|
| `--chunk-size` | 256 | zarr chunk on y/x, the tile a client fetches |
| `--shard-size` | 4096 | zarr shard on y/x with `--sharding`, and the dask block in every case; a multiple of the chunk size |
| `--sharding` | off | write sharded arrays, one object per shard |
| `--method` | `mean` | `mean`, `min`, `max`, `median` or `nearest` |
| `--nodata` | none | numeric nodata, excluded from the reduction; NaN is always nodata |
| `--compressor`, `--clevel` | `zstd`, 3 | Blosc codec and level for every level's data; `none` for uncompressed |
| `--workers`, `--threads-per-worker`, `--memory-limit` | 8, 1, `auto` | the dask cluster; several single-threaded processes beat one multi-threaded one, since zarr assembles chunks on one event loop per process |
| `--window-shards` | 32 | level-0 blocks per axis and per window; the main process holds one window's graph at a time |

The input is a zarr store, local or `s3://`, with `y` and `x` dimensions and a CRS; S3
credentials come from `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and, for S3-compatible
stores, `AWS_ENDPOINT_URL`. Environment variables `TERRAZARR_WINDOW_SHARDS`,
`TERRAZARR_FUSE_LEVEL_1` (`0` disables the fused level 1) and
`TERRAZARR_MIN_BLOCKS_FROM_STORE` set the defaults of the corresponding knobs.

## Output layout and conventions

A pyramid is a zarr v3 group with one child group per level, `0` for the native resolution
and `L` for the level with `2**L` native pixels per pixel, each holding the data variables, the
`x` and `y` coordinates and a CF `spatial_ref` grid-mapping variable with `crs_wkt` and a
`GeoTransform`, so every level is georeferenced on its own and opens with rioxarray. The root
group declares the [zarr-conventions](https://github.com/zarr-conventions) it follows and
carries their attributes:

| convention | version | attributes written |
|---|---|---|
| multiscales | v1 | `multiscales.layout`: one entry per level with `asset`, `transform` (scale and translation relative to level 0), `spatial:shape`, `spatial:transform`, `spatial:bbox`; `resampling_method` |
| proj | v1 | `proj:code`, `proj:wkt2` |
| spatial | v1 | `spatial:dimensions`, `spatial:bbox`, `spatial:transform`, `spatial:registration` (`pixel`) |

The root also carries a native-CRS tile matrix set and tile matrix limits in the form
(`tile_matrix_set`, `tile_matrix_limits`), one tile matrix per level with the chunk as tile,
used by the [EOPF GeoZarr data model](https://eopf-explorer.github.io/data-model/). Metadata is consolidated at every level and at the root.
Arrays are zarr v3 with the sharding codec when `sharding` is on (chunks of `chunk_size²`
inside shards of `shard_size²`), Blosc-compressed by default, with the fill value NaN for
floating-point data and the dtype's zero otherwise; a numeric `nodata` is excluded from every
reduction and written back as the value of blocks that fall below 30 % valid pixels.

Geometry is exact: level L has exactly `2**L` native pixels per pixel, anchored on the native
top-left corner, so `spatial:transform` of level L is the native transform scaled by `2**L`, and
levels whose shape is not a multiple of `2**L` are trimmed rather than stretched. Integer
overviews are rounded to nearest; `min` and `max` stay in the source dtype; `median` is computed
in float32 for integers up to 16 bits and float64 otherwise; `nearest` takes the pixel at the
centre of each block.

## Resume

Level 0 (with level 1, which is reduced from the same blocks) is written in windows, and the
level-0 group records every window done in the attribute `terrazarr:windows_done` and sets
`terrazarr:level0_complete` at the end. Running `to_geozarr` again on an interrupted store
finishes the missing windows and does not rewrite the rest; a complete store is left untouched.
Every further level is built from the previous one on the store and skipped when present.
