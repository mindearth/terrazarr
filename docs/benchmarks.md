# Benchmarks

## Summary

Baseline is the module this project was forked from (eopf-geozarr with the method and nodata options of the internal fork, see "Reproducibility" in the benchmark docs), run on the same inputs, machine (24 cores,
43 GB) and MinIO with the same settings (shard 4096 unless stated, chunk 256, sharding, 8 threads).
The sections below hold the full tables and methods; the profile that motivated changes 12–17 is
[profile.md](profile.md), the MinIO measurements are [minio.md](minio.md), the change history is
[changes.md](changes.md).

| input | baseline | optimized, 8 worker processes | speed-up |
|---|---:|---:|---:|
| WSF3Dv3 Italy, 178335×200599 float64, MinIO | 1344 s | 461 s | 2.9× |
| WSF3Dv3 Italy, local NVMe | 1287 s | 346 s | 3.7× |
| Italy 32768² window, MinIO, shard 4096 | 61 s | 33 s | 1.9× |
| Italy 32768² window, MinIO, shard 8192 | 71 s | 37 s | 1.9× |

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
| Italy window, shard 4096 | 2790 → 2303 | 543 → 391 |
| Italy window, shard 8192 | 5053 → 7337 | 22240 → 569 |
| full Italy | 3575 → 3953 | 27914 → 25615 |

The baseline's low RSS at shard 8192 is the flip side of its 22 240 GETs: it reads small pieces
per output tile. The full-Italy wall times are from suite v2 on a quiet machine; the current code
(suite v3, change 17) reads level 0 once, which is where its GET count comes from, and was
re-measured under load with identical output.

Input formats. The pipeline reads GeoTIFFs directly (rioxarray, one GDAL handle per thread,
`/vsis3/` on MinIO), so the same raster was fed in as the striped source (one-row strips), as a
COG (512² tiles, `gdal_translate -of COG`, 40 % of its tiles sparse) and as the 2048-chunked zarr
the extract writes; optimized module, same settings, MinIO ([benchmarks.md](benchmarks.md), "Input formats"):

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
[benchmarks.md](benchmarks.md) "Other GeoZarr pyramid writers"):

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
| chunk / shard | `--chunk-size` / `--shard-size` / `--sharding` | eopf `spatial_chunk`; one shard per level | chosen by it (512 KB chunks, ≤ 4 per shard) | `BLOCKSIZE`, no shards |
| layout, metadata | `0`, `1`, … groups; GeoZarr multiscales + tile matrix set | `<group>/0`, `1`, …; GeoZarr 0.4 | `0`, `1`, …; zarr-conventions multiscales | root array + `ovr_2x`, …; zarr-conventions multiscales |
| resume | per-band validation, retry, failed batch rewritten | per-band validation, retry | none | none |

Only this module applies the nodata rule; the other three average nodata zeros in and so
differ from the baseline in the same places. Details and the smoke-scale semantics probe are in
[benchmarks.md](benchmarks.md), "Other GeoZarr pyramid writers".

Output: identical to the baseline at every level for min, median and float means; integer means
differ by at most 1 per level because the baseline truncated ([changes.md](changes.md), change 10). The
optimized output additionally carries a decodable CRS on every overview level and exact `2**L`
pixel sizes, which the baseline does not (changes 9 and 11).


### Scaling limits

Measured on a 20 cm orthophoto of Italy (`s3://test/agea4.zarr`: 2 263 040 × 2 191 360 × 4 bands
uint8, band-last, 1.18 TB stored in 5 500 of 47 294 source shards), on which the CLI with 8
worker processes reached 70 GB and took the machine down ([benchmarks.md](benchmarks.md), "Scaling").

Main findings:

- **The memory is the dask graph, not the data.** Level 0 (with the fused level 1) is one dask
  graph; the client and scheduler that hold it live in the main process and need **36–45 KB
  per level-0 shard task**, on top of 0.15 GB. It is linear in the number of tasks and
  independent of the pixel count. The reproduction on the real store went from 0.5 GB to a
  29 GB peak in 19 minutes of graph construction, workers flat at 1.1 GB, before any pixel
  was read.

  | level-0 shard tasks | main process peak | example |
  |---:|---:|---|
  | 1 024 | 0.22 GB | 65536² × 4 bands at shard 4096 |
  | 16 384 | 0.80 GB | 262144² × 4 bands at shard 4096 |
  | 65 536 | 2.56 GB | 524288² × 4 bands at shard 4096 |
  | 1 180 000 | 45–55 GB (extrapolated, 29 GB measured before stopping) | agea4 at shard 4096 |

- **Worker memory scales with the block, not the raster:** about `4 × chunk² × bands × itemsize`
  per worker for a band-last input. Chunk 8192 fits 8 workers on 43 GB (7 GB in total); chunk
  16384 does not (28 GB, the batch write fails).
- **An empty shard is not free.** 0.35 s of worker time each, spent in zarr's sharding codec
  walking 256 inner chunks to store nothing; skipping the write for all-nodata blocks (they
  are never stored anyway) measures 0.13 s. On a raster whose shards are 88 % empty that is
  half the run.
- **A data shard-band costs 2.8 s** at 8 workers, of which 6.1 s per four-band block is the read
  from MinIO; the link delivers 110 MB/s at any concurrency, so 1.18 TB is a 3-hour floor and
  more workers would not read faster. Projected end to end at shard 4096: about 26 hours.
- The band-last layout is handled correctly (level 0 equals the transposed input, level 1 the
  exact per-band mean); it is not the cause.

What was done about it (changes 19–21, branch `windowed-level0`):

1. Level 0 is written in spatial windows, one compute per window (`--window-shards`, default 32
   blocks per axis), so the graph is bounded by the window whatever the raster, with per-window
   resume markers. Levels 2+ already worked from the store.
2. The zarr write is skipped for level-0 blocks equal to the fill value.
3. A band-last source keeps its bands together in one task per block instead of a dask rechunk.

Measured effect: see "Windowed level 0" in [benchmarks.md](benchmarks.md).


## Reproducibility

Every number in this document was measured on one machine: 24 cores, 43 GB RAM, local NVMe,
Ubuntu 22.04 (kernel 6.8), Python 3.11, a MinIO reachable over a 1 Gbit/s link (110 MB/s reads,
22 MB/s writes, see [minio.md](minio.md)). Wall times vary by about ±5 % between repeats and
more when the machine is shared, which the notes say when it happened; store traffic, task
counts and peak memory are stable. Each results file under `bench/out/` names the run; the
suites below write them.

### Data

The inputs, the reference pyramids and the synthetic inputs are public, anonymous reads, under
`s3://me-public-assets/terrazarr/` (eu-central-1), also at `https://me-public-assets.s3.eu-central-1.amazonaws.com/terrazarr/<key>`:

```bash
.venv/bin/python bench/fetch_data.py          # window inputs, window reference, synthetic: about 2 GB
.venv/bin/python bench/fetch_data.py --full   # plus the full-Italy inputs and reference: about 9 GB more
```

| key | what | license |
|---|---|---|
| `inputs/WSF3Dv3_Italy_striped.tif` | the source: 178335 × 200599 float64 GeoTIFF, one-row strips, deflate, 1.88 GB (World Settlement Footprint 3D, Italy) | CC BY 4.0 |
| `inputs/WSF3Dv3_Italy_full.zarr/` | the same as a 2048²-chunked zarr v3 store (`bench/tif_to_zarr.py`), 1.87 GB | CC BY 4.0 |
| `inputs/WSF3Dv3_Italy_cog.tif` | the same as a COG, 512² tiles, 9 overviews, 2.94 GB | CC BY 4.0 |
| `inputs/WSF3Dv3_Italy_win32k*` | the 32768² window (row 73728, col 77824) as zarr, COG and striped GeoTIFF | CC BY 4.0 |
| `reference/italy_full_baseline.zarr/`, `reference/win32k_baseline.zarr/` | the pyramids of the original module, which every comparison uses | CC BY 4.0 |
| `synthetic/s1..s4.zarr`, `synthetic/smoke_u8.zarr` | the synthetic scenarios' inputs (`bench/synth.py` regenerates them) | Apache-2.0 |
| `demo/WSF3Dv3_Italy.zarr/` | the pyramid this module writes from the full-Italy input (chunk 256, shard 4096, mean, nodata 0), behind the [demo](demo.md) | CC BY 4.0 |
| `results/` | the JSON and log files behind every table here | Apache-2.0 |

Not public: the 20 cm orthophoto zarr of the scaling and MinIO studies (2263040 × 2191360 × 4
uint8, band-last, 1.18 TB stored; CC BY 4.0, `[DATA-2-URL]`). The runs that write to an S3
store used a MinIO named by `AWS_ENDPOINT_URL`; bring your own bucket for those and set the
credentials in the environment (`bench/s3env.sh` shows the variables).

Synthetic scenarios (`bench/compare.py`) need nothing external and are what CI runs.

### Commands

Local, synthetic inputs:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python bench/compare.py              # writes bench/out/results.md
TAG=_v2 bash bench/run_suite.sh                # every benchmark of benchmarks.md (about an hour)
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

### Against an S3 store

```bash
source bench/s3env.sh                       # exports MinIO credentials from ~/repos/.myenvs
.venv/bin/python bench/tif_to_zarr.py --src s3://bucket/x.tif --dst s3://bucket/x.zarr \
    --row0 74000 --col0 78000 --rows 32768 --cols 32768      # window of a striped GeoTIFF -> zarr input
.venv/bin/python bench/run_one.py --impl optimized --input s3://bucket/x.zarr --output s3://bucket/out.zarr \
    --shard-size 4096 --chunk-size 256 --sharding --method mean --nodata 0 --threads 8
```

The baseline needs `botocore < 1.36` against this MinIO (newer botocore omits the `Content-MD5`
header that MinIO requires on bulk deletes, which s3fs uses; the upstream pins it for the same
reason). Build it once and run the baseline with that interpreter:

```bash
uv venv --python 3.11 .venv-baseline
uv pip install --python .venv-baseline/bin/python "eopf-geozarr==0.7.1" "dask[distributed]" obstore rioxarray "s3fs==2024.12.0" "aiobotocore==2.15.2" "botocore<1.36"
.venv-baseline/bin/python bench/run_one.py --impl baseline ...
```

The optimized module does not use s3fs and runs with current botocore. See [benchmarks.md](benchmarks.md).

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
bounded by `threads × shard-size² × itemsize × 2` instead (see above), 3.5 GB for the same raster
at shard 4096.

## Results in full


threads per run: 4

## s1: 2D uint8 16384², sharded 4096, min, nodata 0

| metric | baseline | optimized | ratio |
|---|---:|---:|---:|
| wall time [s] | 16.54 | 13.69 | 0.83× |
| peak RSS [MB] | 3564.0 | 1338.3 | 0.38× |
| dask tasks | 112 | 228 | 2.04× |
| chunk GETs | 1999 | 129 | 0.06× |
| chunk PUTs | 39 | 39 | 1.00× |
| metadata GETs | 251 | 256 | 1.02× |
| files on disk | 75 | 75 | 1.00× |

## s2: 2D uint8 16384², unsharded, mean, nodata 0

| metric | baseline | optimized | ratio |
|---|---:|---:|---:|
| wall time [s] | 34.5 | 13.72 | 0.40× |
| peak RSS [MB] | 2340.0 | 1346.4 | 0.58× |
| dask tasks | 8304 | 228 | 0.03× |
| chunk GETs | 5542 | 5540 | 1.00× |
| chunk PUTs | 5233 | 5233 | 1.00× |
| metadata GETs | 251 | 256 | 1.02× |
| files on disk | 5269 | 5269 | 1.00× |

## s3: 3D uint8 8×8192², sharded 4096, mean, nodata 0

| metric | baseline | optimized | ratio |
|---|---:|---:|---:|
| wall time [s] | 30.79 | 20.13 | 0.65× |
| peak RSS [MB] | 1880.2 | 1451.2 | 0.77× |
| dask tasks | 641 | 645 | 1.01× |
| chunk GETs | 833 | 285 | 0.34× |
| chunk PUTs | 27 | 90 | 3.33× |
| metadata GETs | 254 | 261 | 1.03× |
| files on disk | 64 | 127 | 1.98× |

## s4: 2D float32 12288², sharded 4096, median

| metric | baseline | optimized | ratio |
|---|---:|---:|---:|
| wall time [s] | 16.35 | 20.88 | 1.28× |
| peak RSS [MB] | 4928.9 | 3219.2 | 0.65× |
| dask tasks | 167 | 155 | 0.93× |
| chunk GETs | 2561 | 390 | 0.15× |
| chunk PUTs | 29 | 32 | 1.10× |
| metadata GETs | 221 | 223 | 1.01× |
| files on disk | 60 | 63 | 1.05× |

## Notes

- Machine: 24 cores, 43 GB RAM, local NVMe; each run is one process with 4 dask threads (`bench/run_one.py`).
  Wall time varies by about ±5 s between repeats. Two extra repeats of s4 gave baseline 17.3 s / 22.5 s
  against optimized 14.9 s / 18.6 s, so the 1.28× in the table above is within noise; peak RSS and
  chunk GETs are stable across repeats.
- Chunk GETs are the metric that transfers to S3, where every partial-shard read is one range request.
- s3 writes more shard files by design: shards are one time slice each (change 6), which bounds task
  memory by `shard_size²` instead of growing with the time dimension.
- Output comparison (`bench/compare_outputs.py`): s1 (min) and s4 (median) are identical at every level.
  s2 and s3 (mean on uint8) differ by at most 1 per level, compounding to 4 at level 6, because the
  baseline truncated integer means (change 10). All shapes here are powers of two, so pixel sizes agree.

## Against MinIO: WSF3Dv3 Italy window

Input: a 32768×32768 float64 window (row 74000, col 78000) of `WSF3Dv3_Italy.tif`, extracted to a
2048-chunked zarr on MinIO with `bench/tif_to_zarr.py`. Both implementations write to the same
MinIO bucket with `--chunk-size 256 --method mean --nodata 0 --sharding --threads 8`.
The baseline runs in `.venv-baseline` (botocore < 1.36, see README). Every "baseline" number in this file was measured against the internal fork of eopf-geozarr that added `method` and `nodata_value` (the module this project started from); the published harness runs upstream 0.7.1, which always averages, so its overview values differ where nodata is involved (see "Other GeoZarr pyramid writers").

### shard 4096

| metric | baseline | optimized | ratio |
|---|---:|---:|---:|
| wall time [s] | 61.34 | 57.79 | 0.94× |
| peak RSS [MB] | 2789.6 | 2653.0 | 0.95× |
| dask tasks | 821 | 822 | 1.00× |
| chunk GETs | 543 | 455 | 0.84× |
| chunk PUTs | 107 | 107 | 1.00× |
| metadata GETs | 281 | 289 | 1.03× |
| objects | 148 | 148 | 1.00× |

### shard 8192

| metric | baseline | optimized | ratio |
|---|---:|---:|---:|
| wall time [s] | 70.74 | 61.26 | 0.87× |
| peak RSS [MB] | 5053.0 | 8353.3 | 1.65× |
| dask tasks | 742 | 237 | 0.32× |
| chunk GETs | 22240 | 329 | 0.01× |
| chunk PUTs | 44 | 44 | 1.00× |
| metadata GETs | 281 | 289 | 1.03× |
| objects | 85 | 85 | 1.00× |

### Notes

- The first baseline attempt failed at once with `Failed to write all bands`: the baseline never
  stored `grid_mapping`, so its own re-validation of a written level found no CRS (change 11).
  The numbers above are from the rerun with the fix applied to both implementations. Even so the
  baseline output written here has no decodable CRS (`ds.rio.crs is None` at every level); the
  optimized output has one.
- Level-by-level comparison of the chunk-8192 outputs: values are identical at all 8 levels
  (max |diff| 0), pixel sizes agree, shapes are powers of two.
- At shard 8192 the baseline issues 22 240 range requests for a 64-chunk input, i.e. it re-reads
  the input per output tile. The optimized module reads each input chunk about five times in total
  across the whole pyramid.
- Peak RSS at shard 8192 follows the memory model in the README: 8 threads × 8192² × 8 bytes is
  4.3 GB of input blocks in flight, and the observed 8.4 GB is that times a factor of two. The
  baseline uses less memory at 8192 only because it reads small pieces, which is what costs it the
  22 240 GETs. Choose `--shard-size`/`--threads` so that `threads × shard² × itemsize × 2` fits.

## Against MinIO: full WSF3Dv3 Italy

`bench/run_full_italy.sh`: the whole `WSF3Dv3_Italy.tif` (178335×200599 float64, 1.3 % non-zero)
extracted to a 2048-chunked zarr on MinIO, then the optimized pyramid with
`--shard-size 4096 --chunk-size 256 --method mean --nodata 0 --sharding --threads 8`.
Same 24-core / 43 GB machine, MinIO over the local network.

| step | wall time | peak RSS [MB] | objects | size on MinIO [GB] |
|---|---:|---:|---:|---:|
| extract (`tif_to_zarr.py`, 175 strips of 1024 rows) | 259 s | ~14 000 | 5239 | 1.87 |
| pyramid, 10 levels (optimized) | 1276 s (21.3 min) | 3450 | 2178 | 2.69 |
| pyramid, 10 levels (baseline, `IMPL=baseline EXTRACT=0`) | 1344 s (22.4 min) | 3575 | 2166 | |

| metric | baseline | optimized | ratio |
|---|---:|---:|---:|
| wall time [s] | 1343.97 | 1276.04 | 0.95× |
| peak RSS [MB] | 3575.0 | 3450.1 | 0.97× |
| dask tasks | 26955 | 26956 | 1.00× |
| chunk GETs | 27914 | 27771 | 0.99× |
| chunk PUTs | 2115 | 2127 | 1.01× |
| metadata GETs | 341 | 355 | 1.04× |
| objects | 2166 | 2178 | 1.01× |

At shard 4096 the two implementations do almost the same work on this raster: the input chunks
(2048²) fit the output shards (4096²) exactly, so the baseline's per-tile re-reads that dominate
the chunk-8192 window run do not occur. The gains here are in the output, not the runtime.

Baseline vs optimized output (`italy_full_baseline.zarr` vs `italy_full_optimized.zarr`):

- Values are identical at every level (levels 2–9 compared in full, levels 0–1 on four random
  4096² blocks each); origins agree.
- The baseline's overview levels (1–9) have no decodable CRS (change 11); the optimized output has
  one at every level.
- The baseline's pixel size drifts from `2**L` × native from level 3 on, because it derives it from
  the trimmed extent divided by the shape; the drift is 0.2 % at level 9 (0.046087° vs 0.045994°),
  which is 0.4 px of misregistration across the level (change 9).
- The optimized output has 12 more objects: levels 6–9 are written as 2×2 shards sized to the
  level (e.g. 2560×3072 at level 6) where the baseline writes a single 4096² shard.

Checks on the written pyramid:

- 10 levels, 178335×200599 down to 348×391, pixel size exactly `2**L` × 0.0000898°, CRS decodable
  at every level, shards 4096² with 256² inner chunks (smaller shards on the last four levels where
  the level itself is smaller).
- Level 0 equals the input on a 4096² block in the centre of Italy.
- Level 1 equals the 2×2 mean of level 0 on that block with the pyramid's nodata rule: a block
  whose valid fraction is below 30 % (`utils.VALID_FRACTION`, same rule as the baseline) is nodata,
  so a 2×2 block with a single building pixel becomes 0 at level 1. That affects 0.7 % of the
  level-1 pixels in the checked block. A naive `nanmean` check will not match.
- The extract, not the pyramid, is the memory hog: with 2048-row full-width strips (3.3 GB each) and
  8 threads it peaked at 28.4 GB RSS and was restarted with 1024-row strips (`STRIP`), which
  peaked at 13.8 GB. Peak RSS ≈ 1.5 GB + threads × strip × width × itemsize (README, "Extract
  memory").

## Local disk vs MinIO: full WSF3Dv3 Italy

Same input zarr copied to local NVMe (`rclone`, 30 s for 1.87 GB), both implementations reading
and writing locally (`ROOT=bench/data/italy EXTRACT=0 TAG=_local`), run back to back with the
same settings as the MinIO runs (shard 4096, chunk 256, mean, nodata 0, sharding, 8 threads).
The input was in the page cache after the copy, so the local read side is effectively RAM.

| wall time [s] | MinIO | local | local / MinIO |
|---|---:|---:|---:|
| baseline | 1343.97 | 1286.68 | 0.96× |
| optimized | 1276.04 | 1425.04 | 1.12× |

| peak RSS [MB] | MinIO | local |
|---|---:|---:|
| baseline | 3575.0 | 3532.2 |
| optimized | 3450.1 | 3665.1 |

Store traffic (chunk GETs/PUTs, tasks, objects) is identical to the MinIO runs for both.

- Removing MinIO does not make either implementation faster, and the baseline/optimized ordering
  flips between the two stores. The spread between the four runs (1276–1425 s) is run-to-run
  noise on a shared machine, not store latency: on this raster and chunking, the pipeline is not
  I/O-bound.
- It is not fully CPU-bound either. Sampled during the level-0 copy, the process used 1.5–3 cores
  of the 8 dask threads with the input in page cache, which points at serialised work
  (codec / shard assembly under the GIL, per-shard index writes) rather than at the store. That is
  where the next speed-up on a single machine would have to come from; the store optimisations pay
  off when the input chunking does not tile the output shards (see the chunk-8192 window run).
- Local outputs equal the MinIO outputs and each other at every level (levels 2–9 in full,
  levels 0–1 on sampled 4096² blocks); the baseline again has no CRS on levels 1–9.

## Profile of the optimized pipeline (local 32768² Italy window)

Summary; the full report with the frame table, the GIL recording, the kernel micro-benchmark and
what each finding turned into is [profile.md](profile.md).

`py-spy record` over `run_one.py --impl optimized`, shard 4096, 8 threads, input and output on
local NVMe (`bench/data/italy/input/WSF3Dv3_Italy_win32k.zarr`). Wall 47 s at 279 % CPU of the
800 % available; level-0 copy ≈ 18 s, level 1 ≈ 20 s, levels 2–7 ≈ 7.5 s.

| share of samples | where | what |
|---:|---|---|
| 20 % | `numpy sum` in `utils.downsample` | `valid.sum` + `where(valid, blocks, 0).sum` over `axis=(1, 3)` |
| 14 % | zarr `read_batch` / `write_batch` `__setitem__`, `all_equal`, `np.full` | per-chunk copies into the output array, on zarr's single event-loop thread |
| 10 % | `_decode_sync` → `as_numpy_array_wrapper` | blosc decode plus a copy to numpy |
| 8 % | `_encode_sync` | blosc encode |
| 5 % | dask `_concatenate2` | merging 2×2 downsampled blocks into one shard-sized block |
| 12 % | `ThreadPoolExecutor._worker` | dask threads idle, waiting for zarr |

The GIL is held in only 12 % of samples, so the run is not GIL-bound; the dask threads block on
`zarr` `sync()` while one event-loop thread does all the chunk assembly. Splitting the work over
processes removes that serialisation (`run_one.py --workers`):

| dask layout | wall [s] | CPU | peak RSS |
|---|---:|---:|---:|
| 1 process × 8 threads | 48.2 | 277 % | 2.6 GB |
| 4 processes × 2 threads | 29.0 | 808 % | |
| 8 processes × 1 thread | 23.4 | 1051 % | 4.7 GB (sum over the tree) |
| 4 processes × 4 threads | 30.1 | 872 % | |

Outputs are identical to the threaded run at every level. The downsample kernel itself gains
little from re-formulation (a nodata-0 special case without the `where` copy is 14 % faster);
the large kernel win is skipping all-nodata blocks (`any()` on a 4096² block costs 7 ms against
440 ms for the reduction), which applies to ~40 % of the input chunks of this raster.

### Free-threaded Python

Same window, same settings, on a free-threaded CPython 3.13.3 build (`uv venv --python 3.13t
.venv-ft`; numpy 2.5.3, rasterio 1.5.1, pyproj 3.8.0 and numcodecs 0.16.5 built from source for
the `t` ABI, zarr 3.3.0, obstore pinned to 0.10.1 which is the last release with a 3.13t wheel).
`sys._is_gil_enabled()` is False after importing the whole stack.

| interpreter | threads | wall [s] | CPU |
|---|---:|---:|---:|
| 3.11, GIL | 8 | 48.2 | 277 % |
| 3.13t, `PYTHON_GIL=0` | 8 | 45.4 | 322 % |
| 3.13t, `PYTHON_GIL=0` | 16 | 45.8 | 332 % |
| 3.13t, `PYTHON_GIL=1` | 8 | 63.1 | 241 % |
| 3.11, 8 processes | 8 | 23.4 | 1051 % |

Removing the GIL buys 6 % and doubling the threads buys nothing, while the same work in 8
processes halves the wall time. The serialisation is zarr's single asyncio event loop, not the
GIL. A free-threaded build is not worth its dependency cost here (source builds, older obstore).

## Suite v2: after changes 12–16

`TAG=_v2 bash bench/run_suite.sh` on 2026-09-06 (22:20–23:17) at commit `b5333a4`, i.e. with the
per-block reduction, the fused level 1, the shard-aligned overview tasks and the process-based
defaults (changes 12–16). Same machine and settings as the sections above. Two dask layouts per
run: `t8` is one process with 8 threads and carries the store counters; `w8` is 8 single-threaded
worker processes and gives the wall time the CLI defaults now produce (its counters cover the
main process only and are marked `*`). Raw output: `bench/out/suite_v2.log`,
`bench/out/*_v2_*.json`, `bench/out/synthetic_v2.md`.

### Synthetic scenarios (4 threads, baseline vs optimized)

| scenario | baseline wall [s] | optimized wall [s] | ratio | optimized wall before 12–16 [s] | peak RSS baseline → optimized [MB] | chunk GETs baseline → optimized |
|---|---:|---:|---:|---:|---:|---:|
| s1: uint8 16384², sharded, min | 14.55 | 8.62 | 0.59× | 13.69 | 3815 → 909 | 1999 → 177 |
| s2: uint8 16384², unsharded, mean | 32.46 | 9.96 | 0.31× | 13.72 | 2241 → 885 | 5542 → 1508 |
| s3: uint8 8×8192², sharded, mean | 22.66 | 12.58 | 0.56× | 20.13 | 1818 → 962 | 833 → 381 |
| s4: float32 12288², sharded, median | 18.23 | 16.64 | 0.91× | 20.88 | 5022 → 2550 | 2561 → 417 |

Dask tasks per run drop from 155–645 to 44–126 (one task per shard). Output comparison is as in
the first section: s1 and s4 identical at every level, s2 and s3 differ by at most 1 per level
(4 at level 6) from the baseline's truncated integer mean.

### MinIO Italy window

| chunk | layout | wall [s] | peak RSS [MB] | dask tasks | chunk GETs | chunk PUTs | objects |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4096 | t8 | 59.46 | 2678 | 207 | 647 | 107 | 148 |
| 4096 | w8 | 32.68 | 171* | 207 | 22* | 18* | 148 |
| 8192 | t8 | 55.09 | 7337 | 63 | 569 | 44 | 85 |
| 8192 | w8 | 36.93 | 170* | 63 | 22* | 18* | 85 |

Before changes 12–16 (tables above): shard 4096 57.79 s / 822 tasks / 455 GETs, shard 8192
61.26 s / 237 tasks / 329 GETs / 8353 MB.

### Full WSF3Dv3 Italy, MinIO and local disk

| store | layout | wall [s] | peak RSS [MB] | dask tasks | chunk GETs | chunk PUTs | objects |
|---|---|---:|---:|---:|---:|---:|---:|
| MinIO | t8 | 1203.02 (20.1 min) | 4261 | 5708 | 34239 | 2127 | 2178 |
| MinIO | w8 | 460.54 (7.7 min) | 253* | 5708 | 42* | 34* | 2178 |
| local | t8 | 1051.18 (17.5 min) | 3992 | 5708 | 34239 | 2127 | 2178 |
| local | w8 | 346.44 (5.8 min) | 239* | 5708 | 42* | 34* | 2178 |

Before changes 12–16: optimized 1276 s on MinIO and 1425 s locally with 26956 tasks and 27771
GETs; baseline 1344 s and 1287 s. The process layout is 2.9× faster than the baseline on MinIO
and 3.7× locally; the threaded layout gains 6 % and 26 %.

All four outputs were compared with the earlier optimized output (MinIO runs) and with the
baseline output (local runs): levels 2–9 in full, levels 0–1 on three random 4096² blocks each,
max |diff| 0 at all 10 levels.

### Notes

- Threaded runs issue more chunk GETs than before (34239 against 27771 on full Italy, 647 against
  455 on the window) although change 13 was meant to read level 0 once. The fused level-1 graph
  shares no dask keys with the `to_zarr` graph that writes level 0, so dask reads every level-0
  block twice, once per consumer. The wall time still fell because the reduction itself got
  cheaper (change 12). Change 17 fixes the double read, see the next section.
- Per-process counters: the `w8` rows count only the main process, which writes the metadata and
  the small top levels. The number of tasks and objects is layout-independent.
- Peak RSS of the threaded window run at shard 8192 (7.3 GB) is unchanged in kind from the
  earlier 8.4 GB: 8 threads × 8192² × 8 bytes of input in flight. Workers cap this per process.

## Change 17: one read of level 0 (local 32768² Italy window)

`run_one.py --impl optimized`, shard 4096, chunk 256, mean, nodata 0, sharding, 1 process × 8
threads, input and output on local NVMe. The three variants differ only in how level 1 is built.

| level 1 built by | wall [s] | dask tasks | chunk GETs | peak RSS [MB] |
|---|---:|---:|---:|---:|
| re-reading level 0 from the store (`TERRAZARR_FUSE_LEVEL_1=0`) | 43.9 | 125 | 453 | |
| the level-0 dask blocks, second consumer (change 13) | 36.7 | 207 | 645 | |
| the write task itself (change 17) | 28.9 | 140 | 389 | 2312 |

The first two rows are from 2026-09-06, the third from 2026-09-07 with another job holding two
cores, so its wall time is if anything pessimistic. 389 GETs is the 256 input chunks plus the
parent-shard reads of levels 2–7; level 0 is decoded once. The change-17 output equals
`win32k_baseline.zarr` at all 8 levels (`bench/compare_outputs.py`, max |diff| 0, 39 s, 2.1 GB
peak with the blockwise comparison).

## Suite v3: change 17 on the MinIO window and full Italy

Same runs as suite v2 (MinIO window at shard 4096, full Italy on MinIO and on local NVMe, `t8`
and `w8` layouts) with change 17, on 2026-09-07 07:21–08:21 at commit `f849095`. Raw output in
`bench/out/suite_v3.log` and `bench/out/*_v3_*.json`.

| input | layout | wall [s] | peak RSS [MB] | dask tasks | chunk GETs | chunk PUTs | objects |
|---|---|---:|---:|---:|---:|---:|---:|
| window, MinIO | t8 | 49.69 | 2303 | 140 | 391 | 107 | 148 |
| window, MinIO | w8 | 38.79 | 171* | 140 | 22* | 18* | 148 |
| full Italy, MinIO | t8 | 1199.96 (20.0 min) | 3953 | 3549 | 25615 | 2127 | 2178 |
| full Italy, MinIO | w8 | 530.97 (8.8 min) | 244* | 3549 | 42* | 34* | 2178 |
| full Italy, local | t8 | 1292.95 (21.5 min) | 3788 | 3549 | 25615 | 2127 | 2178 |
| full Italy, local | w8 | 476.40 (7.9 min) | 230* | 3549 | 42* | 34* | 2178 |

- Store traffic and task counts are the metrics change 17 targets and they are deterministic:
  full Italy goes from 34239 chunk GETs and 5708 tasks (suite v2) to 25615 and 3549, below the
  27771 GETs of the unfused pipeline and the 27914 of the baseline, because level 0 is now read
  exactly once and level 1 needs no read at all. The window goes from 647 to 391 GETs.
- Wall times are not comparable with suite v2: the machine carried another session's notebook
  runs for the whole hour (load average 17–21 on 24 cores, 3–5 GB of kernels), so the `w8` runs
  came out slower than the 460 s / 346 s of the quiet v2 run, which remain the reference wall
  times. The window under the same load, threaded, still went from 59.5 s to 49.7 s.
- All four outputs were compared with the baseline output on the same store in full at all 10
  levels (`bench/compare_outputs.py --threads 8`, blockwise, 18–22 min and about 4 GB peak per
  pair): max |diff| 0 everywhere, the first time levels 0 and 1 of full Italy were checked
  exhaustively rather than on sampled blocks. Pixel sizes agree except the baseline's known
  level-9 drift (0.046087° against the exact 0.045994°, change 9).

## Input formats: striped GeoTIFF, COG and zarr

The extract step (`tif_to_zarr.py`) exists because the source `WSF3Dv3_Italy.tif` is a striped
GeoTIFF: 178335×200599 float64, deflate, one row per strip, no overviews, 1.88 GB. This section
feeds the pyramid the same data in three formats without an extract:

- **striped**: the source layout, cut with `gdal_translate -srcwin` for the window (one-row
  strips kept: `-co TILED=NO -co BLOCKYSIZE=1`), read through rasterio/GDAL;
- **cog**: `gdal_translate -of COG` with `BLOCKSIZE=512`, `COMPRESS=DEFLATE`, `SPARSE_OK=TRUE`,
  `RESAMPLING=AVERAGE` overviews (window: 218 MB, 20 s to build), read through rasterio/GDAL;
- **zarr**: the 2048²-chunked, unsharded zarr v3 store the extract produces, read through zarr.

`run_one.py` opens a GeoTIFF with rioxarray in `--shard-size` blocks and `lock=False` (one GDAL
handle per dask thread), locally or through `/vsis3/` on MinIO (`bench/run_inputs_win32k.sh`,
`bench/run_inputs_full.sh`). Store counters cover the output only for GeoTIFF inputs. Same
pyramid settings as everywhere else: shard 4096, chunk 256, mean, nodata 0, sharding, 8 threads
(`t8`) or 8 single-threaded worker processes (`w8`).

### 32768² window (2026-09-07 13:56–14:03, quiet machine)

| store | input | layout | wall [s] | CPU | peak RSS [MB] | dask tasks |
|---|---|---|---:|---:|---:|---:|
| local | striped | t8 | 29.1 | 367 % | 10509 † | 204 |
| local | striped | w8 | 19.0 | 1023 % | 1600 | 204 |
| local | cog | t8 | 27.1 | 307 % | 10287 † (3257 with the default cache) | 204 |
| local | cog | w8 | 16.3 | 1026 % | 1559 | 204 |
| local | zarr | t8 | 31.6 | 280 % | 2338 | 140 |
| local | zarr | w8 | 18.3 | 996 % | 553 | 140 |
| MinIO | striped | t8 | 39.8 | 281 % | 10763 † | 204 |
| MinIO | striped | w8 | 40.7 | 579 % | 1693 | 204 |
| MinIO | cog | t8 | 40.1 | 222 % | 10368 † | 204 |
| MinIO | cog | w8 | 32.2 | 587 % | 1530 | 204 |
| MinIO | zarr | t8 | 45.1 | 210 % | 2514 | 140 |
| MinIO | zarr | w8 | 35.8 | 585 % | 590 | 140 |

† GeoTIFF `t8` runs were given `GDAL_CACHEMAX=8192` (MB) so that a block-row of strips stays
cached; the RSS is that allowance being used, not a need. The COG rerun with GDAL's default
cache (5 % of RAM) peaks at 3.3 GB, in line with the zarr input. `w8` runs had 1 GB per process.
Peak RSS of `w8` rows is the main process only.

- Outputs from the COG input equal the baseline window at all 8 levels (`compare_outputs.py`,
  both layouts).
- The COG is the fastest input on both stores, by 5–15 % over zarr; the striped file is not
  slow at this size. A window strip is 32768 × 8 B = 262 KB and a block-row of 4096 strips is
  1 GB, which fits the cache, so the strips are decoded about once per block-row per process.
  With 8 processes on MinIO that redundancy shows: `w8` is no faster than `t8` for the striped
  file (40.7 s against 39.8 s) while the COG gains 20 % and zarr 21 %.
- Why the COG beats the 2048-chunked zarr here: `SPARSE_OK` leaves all-zero tiles unwritten
  and GDAL answers them without I/O or decode; 1806 of the window's 4096 full-resolution tiles
  (44 %) are such tiles (`bench/tiff_tiles.py`). The zarr input stores all 256 of its chunks,
  because its fill value is NaN and zeros are data, so every 2048² block is fetched and
  blosc-decoded before the nodata fast path can skip it. Full-width strips are what the
  extract was written to avoid, and they only cost at full scale, where a block-row is 6.6 GB
  (next table).

### Full WSF3Dv3 Italy (2026-09-07 14:16–15:45)

Full COG: `gdal_translate -of COG` from the striped source on local NVMe, 9.4 min at 355 % CPU,
2.94 GB (source 1.88 GB, zarr input 1.87 GB), 512² tiles, 9 overviews; 54 186 of its 136 808
full-resolution tiles (40 %) are unwritten. The striped and COG files were then uploaded to the
MinIO input prefix next to the zarr.

| store | input | layout | wall [s] | CPU | peak RSS [MB] | dask tasks | load |
|---|---|---|---:|---:|---:|---:|---|
| MinIO | cog | t8 | 1103.7 (18.4 min) | 222 % | 6101 | 5705 | quiet |
| MinIO | zarr | t8 | 1096.6 (18.3 min) | 221 % | 4249 | 3549 | shared † |
| MinIO | cog | w8 | 498.3, rerun 507.1 | 983 % | 1855 | 5705 | shared †, rerun quiet |
| MinIO | zarr | w8 | 701.5, rerun 431.7 | 755 % / 1111 % | 748 | 3549 | shared †, rerun quiet |
| local | cog | w8 | 386.6 | 1327 % | 1764 | 5705 | shared † |
| local | zarr | w8 | 384.7 | 1358 % | 711 | 3549 | shared † |

† From 14:31 another session's notebook kernel held five cores (load average 24–32 on 24 cores,
29 GB used); the `w8` MinIO pair was rerun back to back at 15:29–15:45 on a quiet machine
(load 4–13). Wall times are `/usr/bin/time` of the whole process; `peak RSS` of `w8` rows is the
main process. Store counters cover the output only for GeoTIFF inputs: the threaded COG row counts
16 983 chunk GETs, all parent-shard reads for levels 2–9, and the threaded zarr row's 25 615 is
that plus the 8 624 level-0 chunk requests (5 234 stored chunks and 3 390 misses that return
the fill value), which is the whole read cost of the zarr input in requests.

- At full scale the input format does not decide the wall time. Threaded, COG and zarr land
  within 1 % of each other (1104 s against 1097 s), because one process is bound by zarr's
  event loop on the output side whatever feeds it. With 8 processes the clean pair is
  507 s (COG) against 432 s (zarr) on MinIO and 387 s against 385 s locally: zarr is equal or
  up to 15 % faster. The 20 % COG advantage of the window does not carry over: the window's
  full-width strips and its 44 % sparse tiles favoured GDAL, whereas on full Italy each
  4096² block is 64 tile reads (8 merged range requests) through GDAL against two 2048² chunk
  fetches through obstore, and the zarr input also skips its unwritten chunks (5234 of 8624
  stored: the NaN sea outside the raster is not on disk).
- The COG input costs more memory and tasks: 6.1 GB against 4.2 GB threaded, 1.9 GB against
  0.7 GB per worker, and 5705 tasks against 3549 (rioxarray adds an open-and-read layer per
  block). The GDAL block cache (`GDAL_CACHEMAX`, 2 GB threaded, 1 GB per worker here) is part
  of that RSS.
- The two 701 s / 498 s first attempts of the `w8` pair are what a shared machine and a busy
  MinIO do to an 8-minute run; they are kept in the table as a measure of that noise.
- The striped source at full scale, threaded on MinIO with `GDAL_CACHEMAX=12288` so that a
  block-row of one-row strips (4096 × 200599 × 8 B = 6.6 GB) stays cached (21:51–21:14, idle
  machine):

  | store | input | layout | wall [s] | CPU | peak RSS [MB] | dask tasks |
  |---|---|---|---:|---:|---:|---:|
  | MinIO | striped | t8 | 1397.9 (23.3 min) | 517 % | 17 706 | 5705 |

  It works, at 27 % more wall time than the COG or zarr input in the same layout, four times
  the memory and 2.3× the CPU, which is strips decoded more than once when the 8 threads
  straddle two block-rows. It is also within 3 % of extract plus zarr run (259 s + 1097 s),
  so reading the striped file directly saves nothing over the extract and leaves no chunked
  copy behind. With worker processes it is not an option: each process would need its own
  6.6 GB cache. For the record, no earlier section fed the GeoTIFF to either implementation:
  all baseline-vs-optimized runs read the extracted zarr, and the extract (259 s, 14 GB) is a
  separate, shared step. The first attempt of this run failed after 184 s in the harness's
  output-listing step, a 3-minute MinIO timeout unrelated to the input.

## Other GeoZarr pyramid writers: EOPF, topozarr, GDAL

The same local zarr input through four writers (`bench/tools/`, 2026-09-08, quiet machine):

- **ours**: this module, shard 4096, chunk 256, mean, nodata 0, sharding, 8 threads (`t8`) or 8
  worker processes (`w8`);
- **eopf**: upstream `eopf-geozarr` 0.7.1 (`bench/tools/run_eopf.py`, `.venv-eopf`), the
  package the baseline was forked from: `create_geozarr_dataset(spatial_chunk=4096,
  tile_width=256, min_dimension=256, enable_sharding=True)` with the data under a
  `/measurements` group (a root-only tree writes nothing), 8 dask threads for level 0. Every
  overview is computed from the whole previous level as one numpy array (`ds[var].values`);
- **topozarr**: 0.1.8 on Python 3.12 (`bench/tools/run_topozarr.py`, `.venv-topozarr`):
  `create_pyramid(levels, method="mean").write(max_workers=8)`, source opened lazily without
  dask as its docs advise; its own thread pool, shard-aligned regions and a Rust reduce
  kernel, chunk and shard sizes chosen by it (512 KB chunks, 4 per shard, snapped to the
  source chunking), zstd;
- **gdal**: GDAL 3.13.2 in the official docker image (`bench/tools/run_gdal.sh`; the system GDAL
  is 3.8 and the small image lacks blosc): `gdal_translate -of ZARR -co FORMAT=ZARR_V3
  -co BLOCKSIZE=256,256 -co COMPRESS=ZSTD` then `gdaladdo -r average 2 4 … 128`, which writes
  the overviews as `ovr_2x` … groups with a zarr-conventions `multiscales` attribute.
  `GDAL_NUM_THREADS=8`; no sharding in the classic raster API.

All four write 8 levels, 32768² down to 256². Outputs are compared with the baseline window
by `bench/tools/compare_any.py`, which pairs levels by shape whatever the group layout.

### How each writer works

From the sources (`src/terrazarr`, `eopf_geozarr/conversion/geozarr.py` 0.7.1,
`topozarr/{pyramid,engine,coarsen}.py` 0.1.8, GDAL 3.13.2 Zarr driver and `gdaladdo`).

| aspect | ours | eopf | topozarr | gdal |
|---|---|---|---|---|
| execution | lazy dask graph, one compute per level | level 0 lazy (dask `to_zarr` per band); overviews eager numpy | own thread pool over shard-aligned regions, no dask; source opened lazily | GDAL block-cached streaming, one process |
| level 0 | one task per output shard, block read once | `to_zarr` of the dask array | region copy widened to whole source chunks, each read once | `gdal_translate` block by block, single-threaded apart from the codec |
| level 1 | reduced inside the level-0 write task (change 17) | `ds[var].values` of level 0, numpy reshape-mean | fused into the level-0 copy when levels 1+ fit in half the RAM, else read back from the store | `gdaladdo` pass over level 0 |
| levels ≥ 2 | one task per output shard reading its 4 parent shards from the store (≥ 16 shards), else from memory | whole previous level in numpy | regions of the previous level read back from the store, Rust `block_reduce` | each `gdaladdo` pass from the previous overview |
| peak memory | one parent shard + output per task × threads (2.4 GB threaded, 0.6 GB per worker) | whole level × ~2 (18.6 GB on the window; 286 GB level 0 on full Italy: cannot run) | `workers × 5 × region bytes` (1.2–3.1 GB); pipelined levels held in RAM when they fit | block cache (`GDAL_CACHEMAX`) + overview buffers (2.3 GB window, 9.0 GB full) |
| parallelism | dask threads or worker processes (`--workers`, default 8 × 1) | dask threads for level 0; overviews single-threaded | `max_workers` threads; Rust kernel releases the GIL | `GDAL_NUM_THREADS` for the codec; overview build single-threaded |
| nodata | mean of valid pixels, block blanked below 30 % valid; NaN or numeric nodata | mean over all pixels (nodata only if passed, not exposed by `create_geozarr_dataset`) | mean over all pixels; NaN-aware fill; all-fill regions skipped | numeric zarr `fill_value` honoured as nodata, NaN fill: all pixels averaged |
| overview size | trimmed (`floor`), pixel size exactly `2**L` × native | trimmed | trimmed | rounded up, extent stretched (1 px larger per level) |
| integer output | rounded to nearest | float64 whatever the input | truncated | rounded |
| chunk / shard | `--chunk-size` chunk, `--shard-size` shard, `--sharding`, `--compressor` | `spatial_chunk` chunk; shard = whole level; `tile_width` only in the tile matrix set | chosen by it: 512 KB chunks, ≤ 4 per shard, snapped to the source chunking, zstd; `recommend_encoding` | `BLOCKSIZE` chunk, no shards (classic API), `COMPRESS` |
| level layout | groups `0`, `1`, … with `data` | `<group>/0`, `1`, … | groups `0`, `1`, … | root array + `ovr_2x`, `ovr_4x`, … groups |
| metadata | GeoZarr multiscales + tile matrix set, `spatial:*`, `proj:*`, CRS on every level | GeoZarr 0.4 multiscales + tile matrix set | zarr-conventions multiscales, `proj:*`, `spatial:*` | zarr-conventions multiscales only |
| resume / validation | per-band validation, retry, failed batch rewritten (change 18) | per-band validation and retry | none (`mode="w"` truncates) | none |
| object stores | obstore (`s3://`) | fsspec | any zarr store (obstore, icechunk) | `/vsis3/` |
| input | zarr or GeoTIFF (rioxarray) | zarr DataTree, data under a child group | xarray Dataset with an xproj CRS | anything GDAL reads |

### 32768² window

| writer | wall [s] | CPU | peak RSS [MB] | objects | size | level-0 layout |
|---|---:|---:|---:|---:|---:|---|
| ours, t8 | 34.8 | 285 % | 2417 | 148 | 233 MB | 256² chunks in 4096² shards, blosc-zstd |
| ours, w8 | 18.9 | 1109 % | 561 (main) | 148 | 233 MB | same |
| topozarr | 31.8 | 393 % | 3122 | 2108 | 209 MB | 256² chunks in 1024² shards (128² in 512² from level 2), zstd |
| gdal | 38.2 | container | 2290 (container) | 21865 | 264 MB | 256² chunks, no shards, zstd |
| eopf | 78.0 | 162 % | 18551 | 68 | 273 MB | 4096² chunks in one shard per level (32768²), blosc-zstd |

- **Values.** Ours equals the baseline at every level (max |diff| 0). eopf, gdal and topozarr
  agree with each other and differ from the baseline in the same places: 1.5 % of the level-1
  pixels, up to 128 in value, growing to 42 % of the level-7 pixels. All three average nodata
  zeros in (the input's fill value is NaN, so 0 is data to them), while the baseline and this
  module average valid pixels only and blank a block below 30 % valid. On a raster that is
  98.7 % zeros that is the whole difference; none of the three can express the nodata rule
  (`bench/tools/compare_any.py` output in `bench/out/tools_win32k.log`). At smoke scale with
  a zarr fill value of 0, GDAL did take the fill value as nodata and matched this module to
  within rounding, and topozarr truncated integer means where the others round.
- **eopf** ignored `tile_width` for the chunking (4096² chunks, one shard per level) and
  needed 18.6 GB: the whole 8.6 GB level 0 in memory plus the reduction's temporaries. It
  cannot run on full Italy (level 0 is 286 GB).
- **topozarr** is close to this module's threaded layout on wall time with its own pool,
  and fuses level 1 into the level-0 copy when the upper levels fit in RAM (they do here).
  Per-level stats: level 0 21.9 s, level 1 4.3 s, levels 2–7 4.3 s together.
- **gdal** is single-process; `gdaladdo` runs the 7 overview passes after the translate, and
  the 21 865 unsharded objects are what a 256² chunk pyramid costs without shards.

### Full WSF3Dv3 Italy (local zarr input, 2026-09-08 15:58–17:22)

10 levels, 178335×200599 down to 348×391. eopf not run (whole-level numpy, see above). The
comparisons against `italy_full_baseline.zarr` cover every level in full.

| writer | wall [s] | CPU | peak RSS [MB] | objects | size | overview values vs baseline |
|---|---:|---:|---:|---:|---:|---|
| ours, w8 | 367.2 (6.1 min) | 1401 % | 717 (main) | 2178 | 2.6 GB | identical at all 10 levels |
| gdal | 699.9 (11.7 min) | container | 9044 (container) | 430602 | 3.8 GB | levels one pixel larger; on the overlap 0.9 % of level-1 pixels differ (max 147), 19.5 % at level 9 |
| topozarr | 1087.4 (18.1 min) | 338 % | 1276 | 41361 | 2.3 GB | zeros averaged in: 0.5 % of level-1 pixels differ (max 128), 18.5 % at level 9 |

- **ours** is 1.9× faster than GDAL and 3.0× faster than topozarr at a fraction of the
  memory, with 2178 objects against 430 602 (GDAL, 256² chunks and no shards) and 41 361
  (topozarr, 1024² shards).
- **topozarr** per level: level 0 464 s, level 1 413 s, level 2 116 s; it skipped 3398 of
  8624 regions per level as all-fill. Its level 1 is fused into the level-0 copy only when
  the upper levels fit in RAM, which they do not here (95 GB), so it re-reads the store; the
  level-1 pass then costs almost as much as the copy. At 338 % CPU on 8 workers it is bound
  the same way this module's threaded layout is.
- **gdal**: `gdal_translate` is single-threaded and wrote level 0 in about 8 minutes; the
  nine `gdaladdo` passes took the rest. Its overviews are one pixel larger than the trimmed
  sizes at every level (89168×100300 against 89167×100299), because the generic overview
  builder rounds up where this module, the baseline, topozarr and GDAL's own COG driver (see
  "Input formats": the COG built with 3.8.4 has 100299×89167) round down. Pixel size is
  stretched accordingly, the drift change 9 removed. On the top-left overlap
  (`compare_any.py --crop 1`, `bench/out/tools_full_gdal_compare_crop.log`) 0.93 % of the
  level-1 pixels differ from the baseline, against 0.53 % for topozarr on the same raster:
  both average nodata zeros in, and GDAL's 1.99998 resampling ratio adds a growing
  misregistration on top, up to 19.5 % of the pixels at level 9.

## Scaling: memory and time against raster size (the agea4 case)

`s3://test/agea4.zarr` is a 20 cm orthophoto of Italy: array `z18` of shape (2263040, 2191360, 4)
uint8, dims (y, x, band), chunks (2048, 2048, 4) in (10240, 10240, 4) shards, 19.8 TB
uncompressed; 5 500 of its 47 294 shards are stored (1.18 TB, median 232 MB), the rest is
sea. The CLI with 8 worker processes reached 70 GB on it and took the box down. Measured on
2026-09-11 (`bench/scaling.py`, `bench/memory_trace.py`, `bench/task_breakdown.py`,
`bench/s3_window.py`; raw numbers in `bench/out/scaling.jsonl`, `scaling_chunk.jsonl`,
`agea4_memory_trace.log`).

### The band-last layout is not the problem

A 4096² synthetic (y, x, band) input through `run_one.py` gives level 0 equal to the
transposed input and level 1 equal to the per-band mean of valid pixels, bit for bit. The
pipeline transposes to (band, y, x) and writes one shard per band.

### Memory: the dask graph in the main process

The whole level-0 write (plus the fused level 1) is one dask graph. Its size in the client and
scheduler, which live in the main process, is what grows. On metadata-only stores of the agea4
layout (no chunk stored, every block is nodata, so nothing but the graph costs anything):

| raster | chunk | level-0 shard tasks (4 bands) | tasks executed | main process peak | workers peak (8) | wall |
|---:|---:|---:|---:|---:|---:|---:|
| 16384² | 4096 | 64 | 229 | 0.16 GB | 2.1 GB | 6 s |
| 32768² | 4096 | 256 | 855 | 0.17 GB | 2.3 GB | 13 s |
| 65536² | 4096 | 1 024 | 3 032 | 0.22 GB | 2.4 GB | 50 s |
| 131072² | 4096 | 4 096 | 11 737 | 0.36 GB | 2.6 GB | 198 s |
| 262144² | 4096 | 16 384 | 46 554 | 0.80 GB | 3.0 GB | 846 s |
| 524288² | 4096 | 65 536 | > 100 000 | 2.56 GB | 4.0 GB | 3 056 s |
| 262144² | 8192 | 4 096 | 11 743 | 0.35 GB | 6.9 GB | 718 s |
| 262144² | 16384 | 1 024 | 5 210 | 0.23 GB | 28.2 GB | failed † |

**Main-process memory ≈ 0.15 GB + 36–45 KB per level-0 shard task**, linear from 64 to
65 536 tasks and independent of the pixel count (262144² at shard 8192 costs what 131072² at
shard 4096 costs). agea4 at shard 4096 is 553 × 535 × 4 = 1.18 M tasks: 45–55 GB before a
pixel is read. The reproduction on the real store (`memory_trace.py`, 30 GB cap) showed
exactly that: the main process went from 0.5 GB to a 29 GB peak in 19 minutes of graph
construction with the 8 workers flat at 1.1 GB, dask reported a 1.53 GiB serialized graph
(72× the 21 MiB of the 16 384-task run), and the run was stopped. The workers' memory scales
with the block instead: `workers × (base + k × chunk² × bands × itemsize)`.

† At shard 16384 the band-last read is a 1 GB (16384, 16384, 4) block per task before the
transpose and the four band splits; the worker tree peaked at 28 GB against an automatic
limit of 5.4 GB per worker, the batch write failed and so did the per-band retry (the run
was logged at the quiet level, so the nanny's own messages are not in the file). 8192 is
the largest chunk that fits 8 workers of this input on 43 GB.

### Time: a fixed cost per shard, and the link

`task_breakdown.py` on the 65536² all-nodata store: the pipeline's own task costs 0.1 ms per
shard, the time is in dask's `rechunk` tasks at 249 ms each (1 300 for 256 input blocks): the
(4096, 4096, 4) block is transposed and split into four band blocks, each of which then goes
through zarr's sharding codec on write. zarr itself reads a missing block in 25–31 ms; the
write of an all-fill shard costs 162 ms at chunk 256 (256 inner chunks), 69 ms at chunk 512 and
1024, 80 ms unsharded, against 1.6 ms for the `any()` that proves it empty. Skipping the zarr
write for all-nodata blocks (they are never stored anyway) brings the all-nodata run from
50.2 s to 21.1 s, i.e. **0.35 s → 0.13 s per empty shard**.

On real data (`s3_window.py`, a data-dense 32768² window of agea4, 8 workers): 98.7 s wall,
2.8 s of worker time per shard-band, of which the block read from MinIO is 6.1 s per
(4096, 4096, 4) block. MinIO delivers 107–112 MB/s to this box at 1, 8 or 16 streams, so the
1.18 TB of stored data alone is a 3-hour floor.

Projection for agea4 at shard 4096, 8 workers, if the memory held: ~137 k data shard-bands
× 2.8 s + ~1.05 M empty × 0.35 s ≈ 750 k worker-seconds ≈ 26 h; the empty shards are half
of it, and the skip cuts them to 0.13 s (≈ 18 h).

### What follows

- The memory law is a consequence of one graph per level. Writing level 0 in spatial windows
  (one compute per window of, say, 64 × 64 shards) bounds the graph at the window's size
  whatever the raster and keeps the per-window resume granularity; levels 2+ already work
  from the store. Not implemented.
- Skip the zarr write for all-nodata level-0 blocks (three lines in `_write_and_reduce`),
  worth ~8 h on this raster. Not implemented.
- Until then: `--shard-size 8192` quarters the graph (≈ 12 GB main process on agea4) and
  fits the workers; 16384 does not on 43 GB with a band-last input.
- The read side is bound by the MinIO link; more workers would not read faster.

## Windowed level 0 (changes 19–21, branch `windowed-level0`)

Verification of the windowed writer on 2026-09-11 (`bench/verify_windowed.sh`, log
`bench/out/verify_windowed.log`), with another session's notebook kernel holding 1–2 cores
throughout (load average 6–15), so wall times are noisier than the references.

| run | main | branch | note |
|---|---:|---:|---|
| 32768² window, 8 processes | 18.9 s (quiet) / 16.9 s (same-conditions) | 20.5 s / 16.1 s | output = baseline at 8 levels |
| 32768² window, 8 threads | 34.8 s (quiet) | 38.7 s | |
| full Italy, 8 processes | 367 s (quiet) / 534 s (same afternoon) | 379 s | output = baseline at all 10 levels, 5710 tasks over 4 windows |
| 262144² × 4 bands, shard 4096, metadata-only | 846 s, main process 0.80 GB | 352 s, **main process 0.27 GB** | 16 384 shard tasks in 16 windows |
| 65536² × 4 bands, all nodata | 50.2 s, 3032 tasks | 20.9 s, 599 tasks | one task per four-band block, empty shards skip the codec |
| agea4 32768² data window, MinIO | 98.7 s, 1519 tasks, 722 s worker time | 98.1 s, 622 tasks, 634 s worker time | bound by the MinIO read (5 s per 64 MB block) |

- **Memory.** The main process no longer grows with the raster: 0.27 GB at 16 384 shard tasks
  against 0.80 GB before, and the same for any larger raster since only one window
  (32² blocks, or 32² × bands with a band-last source) is in the graph at a time. On agea4 that
  replaces 45–55 GB with well under 1 GB.
- **Empty shards.** 0.35 s → 0.13 s each; on the all-nodata 65536² store the run halves.
- **Band-last.** One task per (bands, 4096, 4096) block, no dask rechunk: 5× fewer tasks on
  the all-nodata store, 2.4× fewer on the agea4 window, where the wall time is set by MinIO.
- **No regression on the reference cases.** Window and full Italy are within noise of main under
  the same load (16.1 s against 16.9 s side by side on the window; 379 s against 534 s on full
  Italy the same afternoon, 367 s on main the day before on a quiet machine). Outputs are
  identical to the baseline at every level in both.
- **Resume.** A run killed mid-way leaves `terrazarr:windows_done` on the level-0 group;
  the next run finishes the missing windows and does not touch the done ones
  (`tests/test_pipeline.py::test_window_failure_is_retried_then_resumed`). A level 0 without the
  completion flag is never taken as complete.
