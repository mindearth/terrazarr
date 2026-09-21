# Changes from the baseline

Baseline: the `eopf_geozarr.conversion.geozarr` module of [eopf-geozarr](https://github.com/EOPF-Explorer/data-model)
(Development Seed for ESA, Apache-2.0), in the internal fork that added `method` and `nodata_value`
to it; the public benchmark harness runs the published 0.7.1 as baseline. Finding ids refer to the audit (L = laziness, H = hardware, D = input data).

| # | Change | Finding | Where |
|---|---|---|---|
| 1 | Reload each written level with one dask block per shard (`chunks={"y": sc, "x": sc}`) instead of `chunks="auto"`. Whole-shard GETs, no rechunk before the write, no odd-chunk reshape penalty. | L1 | `_load_existing_dataset`, `create_geozarr_compliant_multiscales` |
| 2 | Level-0 dask block from the shard, or `shard_size` when unsharded, never from the chunk size. `shard_size % chunk_size == 0` is enforced. | L2 | `_dask_chunks_for`, `write_dataset_band_by_band_with_validation`, `create_geozarr_dataset` |
| 3 | Level-0 bands written in one `dask.compute`; per-band retry with cleanup only for bands that did not land. | L3 | `write_dataset_band_by_band_with_validation` |
| 4 | Min and max reduce in the source dtype (mask with the dtype extreme); median promotes to float32 for ≤16-bit integers. | H1 | `utils.downsample_2d_array` |
| 5 | `--threads-per-worker` and `--memory-limit` on the CLI and client factory. | H1 | `cli.py` |
| 6 | Shards are one slice on non-spatial dims: `(1, sc, sc)`. Shards are clipped to a chunk multiple never larger than the array. Overview blocks are one slice on non-spatial dims. | H2 | `_create_geozarr_encoding`, `create_geozarr_compliant_multiscales` |
| 7 | One zarr store object for reads, writes, attribute updates, existence checks and deletes (`store.delete_dir`). No s3fs path, no JSON editing of `zarr.json`. | H3 | throughout `geozarr.py`, `store.py` |
| 8 | Existing-band validation checks presence, shape, dtype, `_ARRAY_DIMENSIONS`, CRS and readability of one chunk. `standard_name` is no longer required and a NaN sample is not a failure. `utils.all_chunks_written` offers a strict check. | D1 | `utils.validate_existing_band_data` |
| 9 | Level L pixel size is exactly `2**L` native pixels, anchored top-left; level bbox follows the trimmed extent. Applied to level coordinates, `spatial_ref` GeoTransform, multiscales layout and tile matrix set. | D2 | `_level_pixel_size`, `create_overview_dataset_all_vars`, `create_native_crs_tile_matrix_set` |
| 10 | Integer overviews round to nearest before the cast. | D3 | `create_overview_dataset_all_vars` |
| 11 | `grid_mapping` is pinned into `attrs` before every write. An explicit `encoding=` passed to `to_zarr` replaces the variable encoding, where rioxarray keeps `grid_mapping`, so the baseline never stored it and a written level could not be decoded with a CRS (`ds.rio.crs` was `None`). Found on the MinIO test. | S3 test | `_pin_grid_mapping` |
| 12 | Per-block numpy reduction (`utils.reduce_block`) mapped over dask blocks instead of an array-wide reshape and reduce. A block without a valid pixel is answered without a reduction (7 ms against 440 ms for a 4096² float64 block); with nodata 0 the masked copy of the block is skipped; the final cast (rounded for integers) is folded into the task. | profile | `utils.reduce_block`, `utils.downsample_2d_array` |
| 13 | Level 1 is reduced from the level-0 blocks in the same compute as the level-0 write, so the largest level is read and decoded once instead of twice. A failed batch removes the partial level 1, which the pyramid then rebuilds from the store. `TERRAZARR_FUSE_LEVEL_1=0` disables it. | profile | `write_geozarr_group`, `write_dataset_band_by_band_with_validation` |
| 14 | Overview levels with at least 16 output shards are built by one task per output shard that reads its four parent shards from the store one at a time: no 2×2 block merge, task memory is one parent shard plus the output whatever the level, and a parent shard that holds no valid pixel costs one GET and a scan. Smaller levels keep one task per parent shard for parallelism (`TERRAZARR_MIN_BLOCKS_FROM_STORE`). | profile | `_overview_arrays_from_store`, `_read_reduce_block` |
| 15 | Dask worker processes by default (`--workers min(8, cpus)`, `--threads-per-worker 1`). Zarr assembles chunks on one asyncio loop per process, so threads queue on it: 8 single-threaded processes halve the wall time of one 8-thread process at identical output. | profile | `cli.py` |
| 16 | `--compressor {zstd,lz4,lz4hc,blosclz,zlib,none}` and `--clevel` for every level (`create_geozarr_dataset(compressor=...)`, `make_compressor`). | profile | `cli.py`, `make_compressor` |
| 17 | Level 0 is read once. Change 13 built level 1 from the same dask blocks as the level-0 write, but xarray's `to_zarr` graph and the reduction graph share no keys, so dask executed the block read twice (34 239 GETs on full Italy against 27 771 without fusing). Now one `map_blocks` task per level-0 shard writes the shard into the pre-created zarr array and returns the reduced block (`_write_and_reduce`); `to_zarr` only stores level-0 metadata and coordinates, and level 1 is built from the returned blocks (`fused_level = (path, spec, builder)`). Local 32768² window, 8 threads: 43.9 s / 453 GETs unfused, 36.7 s / 645 GETs with change 13, 28.9 s / 389 GETs now; output identical to the baseline at every level. | suite v2 | `_write_and_reduce`, `_fused_write_reduce_array`, `write_geozarr_group`, `write_dataset_band_by_band_with_validation` |
| 18 | A level-0 batch that fails part-way is written again in full. The old path re-validated each variable from the store and kept the ones that passed, but a missing shard reads as the fill value and shards that are all fill value are never stored, so a partial write is indistinguishable from a complete one; a test that fails one write-and-reduce task showed level 0 accepted with a hole and level 1 built from it. `utils.all_chunks_written` documents that only its True answer is conclusive. | test | `write_dataset_band_by_band_with_validation`, `tests/test_pipeline.py::test_fused_batch_failure_rebuilds_level0_and_level1` |
| 19 | Level 0 and the fused level 1 are written in windows of `window_shards`² dask blocks (default 32; `--window-shards`, `TERRAZARR_WINDOW_SHARDS`), one dask compute each and all bands together, into arrays created once up front; level-1 regions are shard-aligned. The graph held by the client and scheduler is bounded by the window (about 40 KB per block task) instead of growing with the raster: a 2.26 M × 2.19 M four-band image at shard 4096 is 1.18 M block tasks and needed 45–55 GB in the main process. Each window is retried `max_retries` times; the level-0 group records the windows done and a completion flag (`terrazarr:windows_done`, `terrazarr:level0_complete`), a resume skips a complete level and finishes an incomplete one from the missing windows, keeping the level-1 regions of the windows done. A partial level 0 without the flag is never taken as complete, which closes the gap left after change 18. The failure path of change 18 (one rewrite of every band) is gone. | scaling | `_write_level0_windows`, `_level0_windows`, `_read_window_marker`, `write_dataset_band_by_band_with_validation`, `write_geozarr_group` |
| 20 | A level-0 block equal to the array's fill value (NaN-aware) skips the zarr write: zarr stored nothing for it but walked every inner chunk of the shard to find out, 162 ms per 4096² shard at chunk 256 against 1.6 ms for the check. Zeros under a NaN fill are data and are stored as before. | scaling | `_all_fill`, `_write_and_reduce`, `_write_block` |
| 21 | A non-spatial dim of at most `MAX_LEAD_PER_TASK` (16) slices that the source holds in one block stays whole in the dask block: a band-last (y, x, band) image is read and transposed once and one task writes every band shard and reduces every band, instead of dask splitting the block band by band with a rechunk (249 ms and one 64 MB transfer per block). The fused task makes the strided transpose contiguous once. | scaling | `_dask_chunks_for`, `_write_and_reduce` |
| – | Multiscale failures are re-raised; `_create_encoding` call fixed (was a `TypeError` for non-GeoZarr groups); `--nodata` accepts floats; numpy fallback uses `chunks="auto"`. | D4, L4 | |

Recommended for imagery outputs: `--chunk-size 512`. The per-shard cost of the sharding codec is
per inner chunk (an all-fill 4096² shard costs 162 ms at chunk 256, 69 ms at 512 and 1024), and
RGBA tiles of 512² are what web clients ask for anyway; it changes no value.

## Output differences to expect

- Overview values of integer data differ by at most 1 where the baseline truncated a mean.
- For shapes that are not multiples of `2**L`, level L has the same shape but a slightly
  smaller pixel size and extent than the baseline (the baseline stretched the full native
  bounds over the trimmed pixels). Level 0 and shapes that are powers of two are identical.
- Group attributes gain a per-level `spatial:bbox` in `multiscales.layout`.
- `nearest` picks the pixel at the centre of each `2**L` block, consistent with the level
  transform. The baseline stretched the native extent over the trimmed shape, so for sizes that
  are not multiples of `2**L` its sampled rows and columns drift by up to one pixel.

## Behaviour kept

Function names and the flow (level 0 band by band, then overviews from the previous level
re-opened from the store, consolidation per level, group and root) are unchanged, as is the
GeoZarr metadata layout. Public signatures gain `store` and `s3_profile`; internal functions
take a `store` instead of `output_path`.
