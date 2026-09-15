# MinIO as input and output store

What the MinIO at `AWS_ENDPOINT_URL` (bucket `test`, over the local network) delivers to this
24-core / 43 GB box, measured with obstore from the benchmark venv, and what it means for a
pyramid run. Dates are 2026-09-07 to 2026-09-11.

## Throughput

| operation | streams | rate |
|---|---:|---:|
| GET, whole 36 MB shard | 1 | 112 MB/s |
| GET, 2 MB ranges of one object | 8 / 16 | 107 / 108 MB/s aggregate |
| zarr read of a (4096, 4096, 4) block through the shard index, 8 threads | 8 | 419 MB/s raw-equivalent (sparse block) |
| PUT, 64 MB object (multipart) | 1 | 26 MB/s |
| PUT, 64 MB objects (multipart) | 4 / 8 | 22 / 23 MB/s aggregate |
| PUT, 64 MB objects (multipart) | 16 | timeouts: 180 s per part, 5 retries, then failure |
| PUT, 8 MB objects (single part, a compressed band shard) | 1 | 11 MB/s |
| PUT, 8 MB objects | 4 / 8 | 20 / 22 MB/s aggregate |
| LIST, 5 508 keys under one prefix | 1 | 55 s while a benchmark was running, 79 s for 47 294 keys idle |

- Reads saturate at about 110 MB/s, a 1 Gbit/s link, at any concurrency from one stream up.
- Writes saturate at about 22 MB/s, a quarter of the read rate, from four streams up; the
  extra streams only queue. Sixteen concurrent multipart uploads made MinIO time out (180 s
  per request, obstore's 5 retries exhausted) and the process was killed by the harness.
- Listings are slow: 10–15 ms per key. `run_one.py` lists the output prefix before a run to
  delete it, which on 2026-09-07 timed out for 3 minutes and failed a run before it started.
- Latency per request is 13–480 ms depending on load; the pipeline's block reads are
  latency-bound, not bandwidth-bound: the agea4 window read 22 MB/s with 8 workers.

## Pyramid runs, MinIO against local NVMe

From `bench/results.md` (optimized module, shard 4096, chunk 256, mean, sharding):

| run | local | MinIO | penalty |
|---|---:|---:|---:|
| 32k window, 8 threads (suite v3) | 33.2 s | 49.7 s | +50 % |
| 32k window, 8 processes (suite v3) | 16.4 s | 38.8 s | +137 % |
| full Italy (2.7 GB out), 8 threads (suite v2, quiet) | 1051 s | 1203 s | +14 % |
| full Italy, 8 processes (suite v2, quiet) | 346 s | 460 s | +33 % |
| full Italy, 8 processes, COG input (input formats) | 387 s | 507 s | +31 % |

The penalty is the write side: full Italy writes 2.7 GB, which is 2 minutes at 22 MB/s,
close to the 114 s difference in the process layout. Reads are not the limit on these runs.

## agea4: output on the test bucket

`s3://test/agea4.zarr` (2 263 040 × 2 191 360 × 4 uint8, 1.18 TB stored) through the windowed
writer of branch `windowed-level0`, 8 workers, output on MinIO. The local disk (125 GB free)
cannot hold the output anyway.

| what | estimate |
|---|---:|
| bytes written: level 0 re-encoded ≈ input, plus a third for levels 1–9 | 1.4–1.7 TB |
| bytes read: input 1.2 TB, plus parents of levels 2–9 0.4 TB | 1.6 TB |
| level 0 + 1, 34 000 data blocks, write-bound at 34 MB per block / 22 MB/s | 14–15 h |
| level 0 + 1, 262 000 empty blocks (nothing written) | about 6 h |
| levels 2–9, 0.4 TB written | 5–6 h |
| **total** | **24–27 h** (19–20 h with a local output) |
| memory: main process < 0.5 GB at level 0, ≈ 3 GB at level 2; workers ≈ 0.6 GB each | 8–12 GB |

Because the cap is the write rate, more than 8 workers does not shorten the run and pushes
MinIO towards the timeouts above. Check that the bucket has 1.7 TB free first (not visible
through the S3 API), launch detached (`systemd-run --user --unit`), and rely on the
per-window resume markers if it is interrupted.

## Compression level against bytes written

See the table below (`bench/s3_window.py 32768 out.zarr 10240 1085440 s3://test/agea4.zarr <compressor> <level>`,
the data-dense agea4 window, 8 workers, output on local NVMe so that only the codec varies).

The window is 32768² × 4 bands, 4.3 GB raw at level 0, 5.7 GB raw for the 8 levels; 86 % of
its level-0 inner chunks hold data. Blosc byte shuffle is a no-op on uint8 (typesize 1), so
the level is the only codec lever. 2026-09-11, quiet machine for the first run, then the
runs' own tails raised the load for the next ones (wall times ±10 %).

| codec | bytes written | vs zstd-3 | wall | CPU |
|---|---:|---:|---:|---:|
| zstd 3 (default) | 2622 MB | | 50.2 s | 682 % |
| zstd 6 | 2614 MB | −0.3 % | 47.1 s | 926 % |
| zstd 9 | 2557 MB | −2.5 % | 72.8 s | 1441 % |
| lz4 5 | 3340 MB | +27 % | 60.4 s | 406 % |

The guess that a higher level would save 10–20 % of the bytes was wrong for this data: zstd 9
saves 2.5 % for 45 % more wall and twice the CPU. Orthophoto pixels are close to incompressible
by a general-purpose codec once the alpha and nodata blocks are out of the way; the 0.46 ratio
of the default comes from those. Stay on zstd 3; do not use lz4 for a MinIO output. What would
cut the bytes is a change of representation, e.g. dropping the constant alpha band or a
lossy image codec, which is outside this pipeline.

The output-to-input ratio on this window is about 0.9 (level 0 re-encoded per band at chunk 256
against the source's four-band 2048² chunks), so the agea4 estimate above stands: about
1.05 TB for level 0 and 1.4 TB for the pyramid.
