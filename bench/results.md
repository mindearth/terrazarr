# Benchmark results

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
  memory by `spatial_chunk²` instead of growing with the time dimension.
- Output comparison (`bench/compare_outputs.py`): s1 (min) and s4 (median) are identical at every level.
  s2 and s3 (mean on uint8) differ by at most 1 per level, compounding to 4 at level 6, because the
  baseline truncated integer means (change 10). All shapes here are powers of two, so pixel sizes agree.

## Against MinIO: WSF3Dv3 Italy window

Input: a 32768×32768 float64 window (row 74000, col 78000) of `WSF3Dv3_Italy.tif`, extracted to a
2048-chunked zarr on MinIO with `bench/tif_to_zarr.py`. Both implementations write to the same
MinIO bucket with `--tile-width 256 --method mean --nodata 0 --sharding --threads 8`.
The baseline runs in `.venv-baseline` (botocore < 1.36, see README).

### chunk 4096

| metric | baseline | optimized | ratio |
|---|---:|---:|---:|
| wall time [s] | 61.34 | 57.79 | 0.94× |
| peak RSS [MB] | 2789.6 | 2653.0 | 0.95× |
| dask tasks | 821 | 822 | 1.00× |
| chunk GETs | 543 | 455 | 0.84× |
| chunk PUTs | 107 | 107 | 1.00× |
| metadata GETs | 281 | 289 | 1.03× |
| objects | 148 | 148 | 1.00× |

### chunk 8192

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
- At chunk 8192 the baseline issues 22 240 range requests for a 64-chunk input, i.e. it re-reads
  the input per output tile. The optimized module reads each input chunk about five times in total
  across the whole pyramid.
- Peak RSS at chunk 8192 follows the memory model in the README: 8 threads × 8192² × 8 bytes is
  4.3 GB of input blocks in flight, and the observed 8.4 GB is that times a factor of two. The
  baseline uses less memory at 8192 only because it reads small pieces, which is what costs it the
  22 240 GETs. Choose `--chunk-size`/`--threads` so that `threads × chunk² × itemsize × 2` fits.

## Against MinIO: full WSF3Dv3 Italy

`bench/run_full_italy.sh`: the whole `WSF3Dv3_Italy.tif` (178335×200599 float64, 1.3 % non-zero)
extracted to a 2048-chunked zarr on MinIO, then the optimized pyramid with
`--chunk-size 4096 --tile-width 256 --method mean --nodata 0 --sharding --threads 8`.
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

At chunk 4096 the two implementations do almost the same work on this raster: the input chunks
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
same settings as the MinIO runs (chunk 4096, tile 256, mean, nodata 0, sharding, 8 threads).
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
what each finding turned into is `bench/profile.md`.

`py-spy record` over `run_one.py --impl optimized`, chunk 4096, 8 threads, input and output on
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

Before changes 12–16 (tables above): chunk 4096 57.79 s / 822 tasks / 455 GETs, chunk 8192
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
- Peak RSS of the threaded window run at chunk 8192 (7.3 GB) is unchanged in kind from the
  earlier 8.4 GB: 8 threads × 8192² × 8 bytes of input in flight. Workers cap this per process.

## Change 17: one read of level 0 (local 32768² Italy window)

`run_one.py --impl optimized`, chunk 4096, tile 256, mean, nodata 0, sharding, 1 process × 8
threads, input and output on local NVMe. The three variants differ only in how level 1 is built.

| level 1 built by | wall [s] | dask tasks | chunk GETs | peak RSS [MB] |
|---|---:|---:|---:|---:|
| re-reading level 0 from the store (`GEOZARR_PYRAMID_FUSE_LEVEL_1=0`) | 43.9 | 125 | 453 | |
| the level-0 dask blocks, second consumer (change 13) | 36.7 | 207 | 645 | |
| the write task itself (change 17) | 28.9 | 140 | 389 | 2312 |

The first two rows are from 2026-09-06, the third from 2026-09-07 with another job holding two
cores, so its wall time is if anything pessimistic. 389 GETs is the 256 input chunks plus the
parent-shard reads of levels 2–7; level 0 is decoded once. The change-17 output equals
`win32k_baseline.zarr` at all 8 levels (`bench/compare_outputs.py`, max |diff| 0, 39 s, 2.1 GB
peak with the blockwise comparison).

## Suite v3: change 17 on the MinIO window and full Italy

Same runs as suite v2 (MinIO window at chunk 4096, full Italy on MinIO and on local NVMe, `t8`
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

`run_one.py` opens a GeoTIFF with rioxarray in `--chunk-size` blocks and `lock=False` (one GDAL
handle per dask thread), locally or through `/vsis3/` on MinIO (`bench/run_inputs_win32k.sh`,
`bench/run_inputs_full.sh`). Store counters cover the output only for GeoTIFF inputs. Same
pyramid settings as everywhere else: chunk 4096, tile 256, mean, nodata 0, sharding, 8 threads
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
  6.6 GB cache. The first attempt of this run failed after 184 s in the harness's
  output-listing step, a 3-minute MinIO timeout unrelated to the input.
