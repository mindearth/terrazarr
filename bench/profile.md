# Profile of the optimized pipeline

One profiling pass, taken on 2026-09-06 at commit `d221383`, before changes 12–17. It is what
motivated them; the shares below no longer describe the current code (see "Not measured since").

## Setup

`py-spy record -r 50 --format raw` over the whole of `bench/run_one.py --impl optimized` on the
local 32768² Italy window (`bench/data/italy/input/WSF3Dv3_Italy_win32k.zarr`), chunk 4096,
tile 256, mean, nodata 0, sharding, one process with 8 dask threads, input and output on local
NVMe. Two recordings: all threads (10 465 samples) and GIL-holding samples only (`--gil`,
1 314 samples). Wall 47 s at 279 % CPU of the 800 % available, peak RSS 2.7 GB.

Raw sample files (gitignored, local only): `bench/out/pyspy_win32k_t8_all.txt` and
`bench/out/pyspy_win32k_t8_gil.txt`. Each line is a stack with its sample count; they can be
fed to `flamegraph.pl` or aggregated with a few lines of Python.

## Time per level

From the run log of the same configuration.

| stage | time |
|---|---:|
| level-0 copy | ≈ 18 s |
| level 1 | 19.6 s |
| level 2 | 5.0 s |
| levels 3–7 | 2.5 s together |

## Where the samples landed (all threads)

| share | leaf frame | what it was doing |
|---:|---|---|
| 20.3 % | numpy `_wrapreduction` (`sum`) | the two block sums of the mean kernel: `valid.sum` and the masked sum over `axis=(1, 3)` |
| 13.6 % | zarr `__setitem__` in `read_batch` | copying decoded chunks into the output array, on zarr's single event-loop thread |
| 11.8 % | `ThreadPoolExecutor._worker` | dask threads idle, waiting on zarr |
| 11.0 % | dask task `__call__` | graph execution overhead |
| 10.4 % | `as_numpy_array_wrapper` under `_decode_sync` | blosc decode plus a copy to numpy |
| 8.3 % | `_encode_sync` | blosc encode |
| 4.6 % | dask `_concatenate2` | merging 2×2 reduced blocks into one shard-sized block |
| 1.8 % | `array_equal` in `write_batch` | zarr checking whether a chunk equals the fill value before writing |
| 1.5 % | `np.full` in `_merge_chunk_array` | zarr allocating fill-value chunks during the merge |

Inclusive shares of the same run: `sum` 20.4 %, `__setitem__` 13.6 %, `_decode_sync` 10.6 %,
`_encode_sync` 8.4 %, `_concatenate2` 4.6 %.

## GIL-only recording

12.6 % of all samples held the GIL, spread over asyncio plumbing rather than any one hot spot:
event-loop `_run_once` 7.2 %, `create_task` 7.1 %, `_run` 5.3 %, batching helpers and
`concurrent_map` 2–3 % each. Within that subset decode was 10.4 % inclusive and encode 5.4 %.
Nothing dominated. This is the finding that ruled out the GIL and pointed at zarr's one asyncio
event loop per process as the serialisation point; the free-threaded build (results.md,
"Free-threaded Python") later confirmed it by gaining only 6 %.

## Kernel micro-benchmark

One 4096² float64 block with 1.3 % non-zero pixels, like WSF3D; best of three.

| variant | time |
|---|---:|
| current kernel: `where(valid, blocks, 0)` plus sums over both axes | 437 ms |
| sum one axis, then the other | 506 ms |
| nodata-0 special case, no masked copy | 377 ms |
| same, with an all-nodata fast path | 396 ms |
| all-nodata block: `any()` alone | 6.9 ms |

The kernel gains little from reformulation (14 % for the nodata-0 case). The large win is
skipping all-nodata blocks entirely, which applies to about 40 % of this raster's input chunks.

## Process layouts

Same window and settings, measured right after the profile (`run_one.py --workers`).

| layout | wall | CPU | peak RSS |
|---|---:|---:|---:|
| 1 process × 8 threads | 48.2 s | 277 % | 2.6 GB |
| 4 processes × 2 threads | 29.0 s | 808 % | 0.9 GB (main process) |
| 8 processes × 1 thread | 23.4 s | 1051 % | 4.7 GB (sum over the process tree) |
| 4 processes × 4 threads | 30.1 s | 872 % | 1.3 GB (main process) |

Outputs were identical across layouts at every level.

## What came of it

| finding | change |
|---|---|
| 20 % in the reduction sums | 12: per-block kernel with the nodata fast path |
| 18 s level-0 copy plus a 19.6 s level 1 that re-read it | 13, then 17: level 1 from the level-0 write task, level 0 read once |
| concatenate and fill-value overheads in the overview merge | 14: one task per output shard, parents read from the store |
| idle threads, event-loop serialisation, GIL not the limit | 15: worker processes by default |

## Not measured since

There is no profile of the current code. The 8-process layout and changes 12–17 have moved the
balance: level 1 is now computed inside the level-0 write tasks, levels 2+ are about 15 % of the
run, and level 0 (read, decode, encode, write) is the clear majority of the time. A fresh
`py-spy` pass on one worker process (`py-spy record --pid <worker>` or `--subprocesses`) would
show what is left.
