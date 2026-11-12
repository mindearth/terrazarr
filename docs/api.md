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
| `--chunk-size` | the input's chunk, else 256 on y/x and 1 on another dim | zarr chunk, `N` for y and x or per dim as `y=512,x=512,band=4` (y equal to x); the y/x chunk is the tile a client fetches |
| `--shard-size` | the input's shard, else 4096 on y/x and the chunk on another dim | zarr shard with `--sharding`, and the dask block in every case, keyed like `--chunk-size`; a multiple of the chunk on each dim |
| `--sharding` | off | write sharded arrays, one object per shard |
| `--method` | `mean` | `mean`, `sum`, `min`, `max`, `median` or `nearest`; `sum` adds the valid pixels, keeps the source dtype (integers saturate) and is nodata only where no pixel is valid |
| `--nodata` | none | numeric nodata, excluded from the reduction; NaN is always nodata |
| `--compressor`, `--clevel` | `zstd`, 3 | Blosc codec and level for every level's data; `none` for uncompressed |
| `--workers`, `--threads-per-worker`, `--memory-limit` | 8, 1, `auto` | the dask cluster; several single-threaded processes beat one multi-threaded one, since zarr assembles chunks on one event loop per process |
| `--window-shards` | 32 | blocks (shards) per axis and per window, on every level; the main process holds one window's graph at a time |
| `--resume` | off | check an existing level 0: skip it when its completion marker says complete, write only the windows an interrupted run left out; a level 0 without a marker is rewritten. Without `--resume`, a valid level 0 is used as it is, without checking that it is complete, and an absent or invalid one is written from scratch. An overview level an interrupted run left incomplete is finished from its windows with `--resume` and rewritten without |
| `--input-icechunk`, `--input-ref` | off, `main` | `--input` is an [Icechunk](#icechunk) repository, read at this branch, tag or snapshot id |
| `--output-icechunk`, `--output-branch` | off, `main` | `--output` is an Icechunk repository, created if absent, written to this branch (created from `main` if absent) in commits |

The input is a zarr store, local or `s3://`, with `y` and `x` dimensions and a CRS; S3
credentials come from `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and, for S3-compatible
stores, `AWS_ENDPOINT_URL`. Environment variables `TERRAZARR_WINDOW_SHARDS`,
`TERRAZARR_FUSE_LEVEL_1` (`0` disables the fused level 1) and
`TERRAZARR_MIN_BLOCKS_FROM_STORE` set the defaults of the corresponding knobs.

## Output layout and conventions

A pyramid is a zarr v3 group with one child group per level, `0` for the native resolution
and `L` for the level with `2**L` native pixels per pixel, each holding the data variables, the
`x` and `y` coordinates and a CF `spatial_ref` grid-mapping variable with `crs_wkt` and a
`GeoTransform`, so every level is georeferenced on its own and opens with rioxarray. The pyramid
group (the root for a Dataset, `/rgba` for `groups=["/rgba"]`) declares the [zarr-conventions](https://github.com/zarr-conventions) it follows and
carries their attributes:

| convention | version | attributes written |
|---|---|---|
| multiscales | v1 | `multiscales.layout`: one entry per level with `asset`, `transform` (scale and translation relative to level 0), `spatial:shape`, `spatial:transform`, `spatial:bbox`; `resampling_method` |
| proj | v1 | `proj:code`, `proj:wkt2` |
| spatial | v1 | `spatial:dimensions`, `spatial:bbox`, `spatial:transform`, `spatial:registration` (`pixel`) |

The pyramid group also carries a native-CRS tile matrix set and tile matrix limits
(`tile_matrix_set`, `tile_matrix_limits`, at the top level and inside `multiscales`) in the form
used by the [EOPF GeoZarr data model](https://eopf-explorer.github.io/data-model/): one tile
matrix per level, whose tile is the y/x chunk (`chunk_size`), anchored on the native top-left
corner. They are metadata only and change no pixel. Metadata is consolidated at every level and
at the root, except in an Icechunk repository, which has no consolidated metadata.
Arrays are zarr v3 with the sharding codec when `sharding` is on (chunks and shards of
`chunk_size` and `shard_size`, an int for y/x or a dict by dim such as
`chunk_size={"y": 256, "x": 256, "band": 4}` for `(4, 256, 256)` chunks; a dim left out has the
source's, read from the zarr encoding of its first data variable, else 256 and 4096 on y/x and 1
on another dim; task memory grows with the shard on every dim), Blosc-compressed by default, with the fill value NaN for
floating-point data and the dtype's zero otherwise; a numeric `nodata` is excluded from every
reduction and written back as the value of blocks that fall below 30 % valid pixels.

Geometry is exact: level L has exactly `2**L` native pixels per pixel, anchored on the native
top-left corner, so `spatial:transform` of level L is the native transform scaled by `2**L`, and
levels whose shape is not a multiple of `2**L` are trimmed rather than stretched. Integer
overviews are rounded to nearest; `min` and `max` stay in the source dtype; `median` is computed
in float32 for integers up to 16 bits and float64 otherwise; `nearest` takes the pixel at the
centre of each block.

## Resume

Every level is written in windows of `window_shards`² shards, one dask compute each, so the
graph held by the main process is bounded by the window whatever the size of the raster. Level
1 is reduced from the level-0 blocks inside the level-0 windows; every further level reads its
parent shards from the store. Each level group records every window done in the attribute
`terrazarr:windows_done` and sets `terrazarr:level0_complete` (the name it had when only level 0
had windows) at the end.

Running `to_geozarr(..., resume=True)` (`--resume`) again on an interrupted store finishes the
missing windows of the interrupted level and does not rewrite the rest; a complete store is left
untouched, and a level 0 without a marker cannot be checked and is rewritten. Without `resume`, a
valid level 0 on the store is used as it is, even if an interrupted run left it incomplete, and
an absent or invalid one is written from scratch, and an incomplete overview level is rewritten.
A complete overview level, or one without a marker (written before the overviews had windows),
is skipped.

## Icechunk

[Icechunk](https://icechunk.io) repositories work as input and as output, with the optional
dependency:

```bash
pip install 'terrazarr[icechunk]'
```

It installs Icechunk 2 on Python 3.12 and later, and Icechunk 1 on Python 3.11, where it
keeps xarray below 2026.9: Icechunk 1 does not work with later versions.

To read one, open a read-only session and pass its store; read-only sessions travel to dask
workers as they are:

```python
from terrazarr.store import open_icechunk_repo, icechunk_read_store

store = icechunk_read_store(open_icechunk_repo("s3://bucket/source"), "main")  # branch, tag or snapshot id
ds = xr.open_dataset(store, engine="zarr", chunks={}, consolidated=False)
```

To write one, pass the Repository as `output`, not a session:

```python
repo = open_icechunk_repo("s3://bucket/pyramid", create=True)
to_geozarr(ds, repo, branch="main", chunk_size=256, shard_size=4096)
```

```bash
terrazarr --input s3://bucket/source --input-icechunk \
  --output s3://bucket/pyramid --output-icechunk --output-branch main --sharding
```

terrazarr opens its own sessions on `branch` and commits as it goes: once per window of every
level, together with the window's completion marker. Readers of the
branch see a finished window or none of it, and `resume=True` (`--resume`) continues an
interrupted run from its last commit. Workers write through forks of the session that are
merged back before each commit, and read the last commit.

A session is not enough because a writable Icechunk session can't be sent to dask workers.
Workers need forks, a fork needs a session without uncommitted changes, and a commit makes
the session read-only, so the next window needs a new session from the repository.

terrazarr writes with manifests split every `2 * window_shards` shards per dim (64 by
default), so a commit rewrites only the part of the manifest it touched rather than the whole
array's; this applies to the writes of the run, whatever configuration the repository has.

Another writer committing to the same branch during a run makes the next commit fail with a
conflict: write on a branch of your own and merge it afterwards. Credentials come from the
same environment variables as for `s3://` paths, plus `AWS_SESSION_TOKEN`.
