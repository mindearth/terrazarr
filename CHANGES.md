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
| – | Multiscale failures are re-raised; `_create_encoding` call fixed (was a `TypeError` for non-GeoZarr groups); `--nodata` accepts floats; numpy fallback uses `chunks="auto"`. | D4, L4 | |

## Output differences to expect

- Overview values of integer data differ by at most 1 where the baseline truncated a mean.
- For shapes that are not multiples of `2**L`, level L has the same shape but a slightly
  smaller pixel size and extent than the baseline (the baseline stretched the full native
  bounds over the trimmed pixels). Level 0 and shapes that are powers of two are identical.
- Group attributes gain a per-level `spatial:bbox` in `multiscales.layout`.

## Behaviour kept

Function names and the flow (level 0 band by band, then overviews from the previous level
re-opened from the store, consolidation per level, group and root) are unchanged, as is the
GeoZarr metadata layout. Public signatures gain `store` and `s3_profile`; internal functions
take a `store` instead of `output_path`.
