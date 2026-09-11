# geozarr-pyramid

GeoZarr multiscale pyramid writer (zarr v3, optional sharding), extracted from
`me-geotools/src/me_geotools/zarr_pyramid_v3` and optimized for laziness and scale.

- `src/geozarr_pyramid/` the optimized module (`geozarr.py`, `utils.py`, `store.py`, `cli.py`)
- `bench/baseline/geozarr_baseline/` the original module, verbatim except import paths, kept for comparison
- `bench/` synthetic inputs, a per-run harness and a baseline-vs-optimized driver
- `tests/` unit and end-to-end tests of the optimized module
- `CHANGES.md` what changed and why, with the audit findings each change addresses

## Install

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e . --group dev
```

Credentials for `s3://` paths come from the environment (`AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, `AWS_ENDPOINT_URL`); there is no dependency on private packages.

## Use

```bash
geozarr-pyramid --input in.zarr --output out.zarr \
  --chunk-size 4096 --tile-width 256 --sharding --method mean --nodata 0 \
  --workers 8 --threads-per-worker 1 --memory-limit 4GB --compressor zstd --clevel 3
```

`--tile-width` is the zarr chunk on y/x (the tile served to clients). `--chunk-size` is the
shard on y/x with `--sharding`, and the dask block on y/x in every case; it must be a multiple of
the tile width. Peak memory per task is about one `chunk-size²` block times a small factor
(see `CHANGES.md`, H1), times `--threads-per-worker` per worker.

Prefer several single-threaded worker processes over one multi-threaded worker: zarr assembles
chunks on one asyncio event loop per process, so dask threads queue on it and one 8-thread
process uses under three cores (`bench/results.md`, "Profile"). Level 0 is copied and level 1
reduced from the same blocks in one pass; every further level is built one task per output
shard, reading its parent shards from the store one at a time, so task memory does not grow with
the level. Blocks without a valid pixel are skipped, which makes sparse rasters cheap.

## Results

Baseline is the original `zarr_pyramid_v3` module, run on the same inputs, machine (24 cores,
43 GB) and MinIO with the same settings (chunk 4096 unless stated, tile 256, sharding, 8 threads).
Full tables, methods and the per-change history are in `bench/results.md` and `CHANGES.md`;
the profile that motivated changes 12–17 is `bench/profile.md`.

| input | baseline | optimized, 8 worker processes | speed-up |
|---|---:|---:|---:|
| WSF3Dv3 Italy, 178335×200599 float64, MinIO | 1344 s | 461 s | 2.9× |
| WSF3Dv3 Italy, local NVMe | 1287 s | 346 s | 3.7× |
| Italy 32768² window, MinIO, chunk 4096 | 61 s | 33 s | 1.9× |
| Italy 32768² window, MinIO, chunk 8192 | 71 s | 37 s | 1.9× |

Synthetic scenarios (`bench/compare.py`, one process with 4 threads for both):

| scenario | baseline | optimized | speed-up |
|---|---:|---:|---:|
| s1: uint8 16384², sharded, min | 14.6 s | 8.6 s | 1.7× |
| s2: uint8 16384², unsharded, mean | 32.5 s | 10.0 s | 3.3× |
| s3: uint8 8×8192², sharded, mean | 22.7 s | 12.6 s | 1.8× |
| s4: float32 12288², sharded, median | 18.2 s | 16.6 s | 1.1× |

Memory and store traffic, measured on single-process runs so the counters cover the whole
pipeline:

| input | peak RSS [MB], baseline → optimized | chunk GETs, baseline → optimized |
|---|---:|---:|
| s1 | 3815 → 909 | 1999 → 177 |
| s2 | 2241 → 885 | 5542 → 1508 |
| s3 | 1818 → 962 | 833 → 381 |
| s4 | 5022 → 2550 | 2561 → 417 |
| Italy window, chunk 4096 | 2790 → 2303 | 543 → 391 |
| Italy window, chunk 8192 | 5053 → 7337 | 22240 → 569 |
| full Italy | 3575 → 3953 | 27914 → 25615 |

The baseline's low RSS at chunk 8192 is the flip side of its 22 240 GETs: it reads small pieces
per output tile. The full-Italy wall times are from suite v2 on a quiet machine; the current code
(suite v3, change 17) reads level 0 once, which is where its GET count comes from, and was
re-measured under load with identical output.

Input formats. The pipeline reads GeoTIFFs directly (rioxarray, one GDAL handle per thread,
`/vsis3/` on MinIO), so the same raster was fed in as the striped source (one-row strips), as a
COG (512² tiles, `gdal_translate -of COG`, 40 % of its tiles sparse) and as the 2048-chunked zarr
the extract writes; optimized module, same settings, MinIO (`bench/results.md`, "Input formats"):

| input | 32k window, 8 threads | 32k window, 8 processes | full Italy, 8 threads | full Italy, 8 processes |
|---|---:|---:|---:|---:|
| striped GeoTIFF, read directly | 40 s | 41 s | 1398 s (12 GB GDAL cache, 17.7 GB RSS) | not viable (one strip cache per process) |
| striped GeoTIFF via the extract (259 s) + zarr | | | 1356 s | 719 s |
| COG | 40 s | 32 s | 1104 s | 507 s |
| zarr | 45 s | 36 s | 1097 s | 432 s |

At full scale the chunked formats are equivalent (zarr equal or up to 15 % faster, COG at 1.5–2.5×
the memory); the window's COG edge comes from its 44 % sparse tiles. Reading the striped source
directly costs as much as the extract plus a zarr run, so the extract stays. Note that every
baseline-vs-optimized number above is the pyramid stage alone, fed from the extracted zarr:
end to end from the striped source the optimized pipeline is 719 s against the baseline's
259 + 1344 = 1603 s, 2.2×.

Other writers, same 32768² window, local zarr input, mean, 8 levels (`bench/tools/`,
`bench/results.md` "Other GeoZarr pyramid writers"):

| writer | wall | peak RSS | objects | overview values |
|---|---:|---:|---:|---|
| this module, 8 processes | 19 s | 0.6 GB main | 148 | = baseline |
| this module, 8 threads | 35 s | 2.4 GB | 148 | = baseline |
| topozarr 0.1.8 | 32 s | 3.1 GB | 2108 | zeros averaged in |
| GDAL 3.13.2 (`gdal_translate` + `gdaladdo`) | 38 s | 2.3 GB | 21865 | zeros averaged in |
| eopf-geozarr 0.7.1 (upstream of the baseline) | 78 s | 18.6 GB | 68 | zeros averaged in |

Full Italy, 178335×200599 float64, 10 levels, local NVMe (eopf not run: it holds each whole
level in memory, 286 GB at level 0):

| writer | wall | peak RSS | objects | size | overview values |
|---|---:|---:|---:|---:|---|
| this module, 8 processes | 367 s | 0.7 GB main | 2178 | 2.6 GB | = baseline at all 10 levels |
| GDAL 3.13.2 | 700 s | 9.0 GB | 430 602 | 3.8 GB | 1 px larger per level; 0.9 % of level-1 pixels differ on the overlap, 19.5 % at level 9 |
| topozarr 0.1.8 | 1087 s | 1.3 GB | 41 361 | 2.3 GB | 0.5 % of level-1 pixels differ, 18.5 % at level 9 |

How they work:

| | this module | eopf-geozarr | topozarr | GDAL |
|---|---|---|---|---|
| execution | lazy dask graph, one task per output shard | level 0 lazy (dask); overviews eager numpy, whole level (`ds[var].values`) | own thread pool over shard-aligned regions, Rust kernel, no dask | block-cached single process |
| level 1 | inside the level-0 write task | whole level 0 in numpy | fused into the level-0 copy when levels 1+ fit in RAM, else read back | `gdaladdo` pass over level 0 |
| levels ≥ 2 | 4 parent shards read from the store per task | whole previous level in numpy | previous level read back from the store | each pass from the previous overview |
| memory | one parent shard + output per task × threads | whole level × ~2 | workers × 5 × region | block cache + overview buffers |
| parallelism | dask threads or worker processes | dask threads for level 0 only | `max_workers` threads | codec threads only |
| nodata | mean of valid pixels, block blanked below 30 % valid | zeros averaged in | zeros averaged in, all-fill regions skipped | numeric fill value as nodata; NaN fill: zeros averaged in |
| overview size | trimmed, pixel size exactly `2**L` × native | trimmed | trimmed | rounded up, extent stretched |
| chunk / shard | `--tile-width` / `--chunk-size` / `--sharding` | `spatial_chunk`; one shard per level | chosen by it (512 KB chunks, ≤ 4 per shard) | `BLOCKSIZE`, no shards |
| layout, metadata | `0`, `1`, … groups; GeoZarr multiscales + tile matrix set | `<group>/0`, `1`, …; GeoZarr 0.4 | `0`, `1`, …; zarr-conventions multiscales | root array + `ovr_2x`, …; zarr-conventions multiscales |
| resume | per-band validation, retry, failed batch rewritten | per-band validation, retry | none | none |

Only this module applies the nodata rule; the other three average nodata zeros in and so
differ from the baseline in the same places. Details and the smoke-scale semantics probe are in
`bench/results.md`, "Other GeoZarr pyramid writers".

Output: identical to the baseline at every level for min, median and float means; integer means
differ by at most 1 per level because the baseline truncated (`CHANGES.md`, change 10). The
optimized output additionally carries a decodable CRS on every overview level and exact `2**L`
pixel sizes, which the baseline does not (changes 9 and 11).

### Scaling limits

Memory of the main process grows with the number of level-0 shard tasks, about 36–45 KB per
task (the dask graph and scheduler state), independent of the pixel count: a 2.26 M × 2.19 M
four-band raster at chunk 4096 is 1.18 M tasks and 45–55 GB before any data is read. Until
level 0 is written in windows, keep the shard count of `bands × (H / chunk) × (W / chunk)`
in the low hundreds of thousands, e.g. `--chunk-size 8192` on such an input. Worker memory
scales with the block instead, `chunk² × bands × itemsize × ~4` per worker for a band-last
input. Details and the measured law in `bench/results.md`, "Scaling".

## Test and benchmark

Local, synthetic inputs:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python bench/compare.py              # writes bench/out/results.md
TAG=_v2 bash bench/run_suite.sh                # every benchmark of bench/results.md (about an hour)
bash bench/run_inputs_win32k.sh                # striped GeoTIFF vs COG vs zarr input, 32k window (10 min)
bash bench/run_inputs_full.sh                  # same on full Italy (about 2 hours)
bash bench/tools/run_tools_win32k.sh           # eopf-geozarr, topozarr, GDAL 3.13 and this module on the window
bash bench/tools/run_tools_full.sh             # topozarr, GDAL and this module on full Italy
.venv/bin/python bench/scaling.py --sizes 16384,32768,65536   # main/worker memory vs shard count on metadata-only inputs
.venv/bin/python bench/task_breakdown.py 65536 bench/data/scaling  # per-task compute time from the task stream
.venv/bin/python bench/tiff_tiles.py x.tif     # tiles per level of a (Big)TIFF and how many are sparse
.venv/bin/python bench/compare.py --only s1    # one scenario
.venv/bin/python bench/compare_outputs.py A.zarr B.zarr   # two pyramids level by level, one shard at a time
```

`compare_outputs.py` streams each level in 4096² blocks (`--chunk`, `--threads`), so it needs
about 2 GB whatever the level size. Runs that may exceed the machine's memory are best started in
their own cgroup, `systemd-run --user --scope -p MemoryMax=16G <cmd>`: an out-of-memory kill then
takes only that job, not the terminal pane it runs in.

Each benchmark run is a separate process with an in-process dask cluster, so store traffic,
executed tasks and peak RSS of the whole pipeline are measured for both implementations on
identical inputs.

### Against MinIO

```bash
source bench/s3env.sh                       # exports MinIO credentials from ~/repos/.myenvs
.venv/bin/python bench/tif_to_zarr.py --src s3://bucket/x.tif --dst s3://bucket/x.zarr \
    --row0 74000 --col0 78000 --rows 32768 --cols 32768      # window of a striped GeoTIFF -> zarr input
.venv/bin/python bench/run_one.py --impl optimized --input s3://bucket/x.zarr --output s3://bucket/out.zarr \
    --chunk-size 4096 --tile-width 256 --sharding --method mean --nodata 0 --threads 8
```

The baseline needs `botocore < 1.36` against this MinIO (newer botocore omits the `Content-MD5`
header that MinIO requires on bulk deletes, which s3fs uses; me-geotools pins it for the same
reason). Build it once and run the baseline with that interpreter:

```bash
uv venv --python 3.11 .venv-baseline
uv pip install --python .venv-baseline/bin/python -e . --group dev "s3fs==2024.12.0" "aiobotocore==2.15.2" "botocore<1.36"
.venv-baseline/bin/python bench/run_one.py --impl baseline ...
```

The optimized module does not use s3fs and runs with current botocore. See `bench/results.md`.

#### Extract memory: strip size vs peak RSS

`tif_to_zarr.py` reads full-width row strips in parallel and every strip is one dask task, so
`--threads` strips are decoded at once and each is held until its `--chunk` row is written:

```
peak RSS  ≈  1.5 GB  +  threads × strip × width × itemsize
```

The constant is the GDAL block cache (1 GB), the curl cache (0.5 GB) and the interpreter.
Measured on `WSF3Dv3_Italy.tif` (width 200 599, float64, 8 threads):

| `--strip` | strip size | predicted | measured peak RSS |
|---:|---:|---:|---:|
| 2048 | 3.29 GB | 27.8 GB | 28.4 GB |
| 1024 | 1.64 GB | 14.6 GB | 13.8 GB |

So halving the strip halves the peak, and the peak scales linearly with the raster width and the
thread count. Pick `strip` so that `threads × strip × width × itemsize` is comfortably below the
free memory; the strip count only changes the number of range requests, not the total bytes read.
`bench/run_full_italy.sh` defaults to `STRIP=1024` for this reason. The pyramid step itself is
bounded by `threads × chunk-size² × itemsize × 2` instead (see above), 3.5 GB for the same raster
at chunk 4096.
