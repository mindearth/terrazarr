# Changes from the baseline

Baseline: `me-geotools` `src/me_geotools/zarr_pyramid_v3` at commit `bd5ef00` (kept verbatim in
`bench/baseline/`). Finding ids refer to the audit (L = laziness, H = hardware, D = input data).

| # | Change | Finding | Where |
|---|---|---|---|
| 1 | Reload each written level with one dask block per shard (`chunks={"y": sc, "x": sc}`) instead of `chunks="auto"`. Whole-shard GETs, no rechunk before the write, no odd-chunk reshape penalty. | L1 | `_load_existing_dataset`, `create_geozarr_compliant_multiscales` |
| 2 | Level-0 dask block from the shard, or `spatial_chunk` when unsharded, never from the tile width. `spatial_chunk % tile_width == 0` is enforced. | L2 | `_dask_chunks_for`, `write_dataset_band_by_band_with_validation`, `create_geozarr_dataset` |
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
| 13 | Level 1 is reduced from the level-0 blocks in the same compute as the level-0 write, so the largest level is read and decoded once instead of twice. A failed batch removes the partial level 1, which the pyramid then rebuilds from the store. `GEOZARR_PYRAMID_FUSE_LEVEL_1=0` disables it. | profile | `write_geozarr_group`, `write_dataset_band_by_band_with_validation` |
| 14 | Overview levels with at least 16 output shards are built by one task per output shard that reads its four parent shards from the store one at a time: no 2×2 block merge, task memory is one parent shard plus the output whatever the level, and a parent shard that holds no valid pixel costs one GET and a scan. Smaller levels keep one task per parent shard for parallelism (`GEOZARR_PYRAMID_MIN_BLOCKS_FROM_STORE`). | profile | `_overview_arrays_from_store`, `_read_reduce_block` |
| 15 | Dask worker processes by default (`--workers min(8, cpus)`, `--threads-per-worker 1`). Zarr assembles chunks on one asyncio loop per process, so threads queue on it: 8 single-threaded processes halve the wall time of one 8-thread process at identical output. | profile | `cli.py` |
| 16 | `--compressor {zstd,lz4,lz4hc,blosclz,zlib,none}` and `--clevel` for every level (`create_geozarr_dataset(compressor=...)`, `make_compressor`). | profile | `cli.py`, `make_compressor` |
| 17 | Level 0 is read once. Change 13 built level 1 from the same dask blocks as the level-0 write, but xarray's `to_zarr` graph and the reduction graph share no keys, so dask executed the block read twice (34 239 GETs on full Italy against 27 771 without fusing). Now one `map_blocks` task per level-0 shard writes the shard into the pre-created zarr array and returns the reduced block (`_write_and_reduce`); `to_zarr` only stores level-0 metadata and coordinates, and level 1 is built from the returned blocks (`fused_level = (path, spec, builder)`). Local 32768² window, 8 threads: 43.9 s / 453 GETs unfused, 36.7 s / 645 GETs with change 13, 28.9 s / 389 GETs now; output identical to the baseline at every level. | suite v2 | `_write_and_reduce`, `_fused_write_reduce_array`, `write_geozarr_group`, `write_dataset_band_by_band_with_validation` |
| 18 | A level-0 batch that fails part-way is written again in full. The old path re-validated each variable from the store and kept the ones that passed, but a missing shard reads as the fill value and shards that are all fill value are never stored, so a partial write is indistinguishable from a complete one; a test that fails one write-and-reduce task showed level 0 accepted with a hole and level 1 built from it. `utils.all_chunks_written` documents that only its True answer is conclusive. | test | `write_dataset_band_by_band_with_validation`, `tests/test_pipeline.py::test_fused_batch_failure_rebuilds_level0_and_level1` |
| – | Multiscale failures are re-raised; `_create_encoding` call fixed (was a `TypeError` for non-GeoZarr groups); `--nodata` accepts floats; numpy fallback uses `chunks="auto"`. | D4, L4 | |

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
