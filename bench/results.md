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
