# Tests

```bash
uv pip install --python .venv/bin/python -e . --group dev
.venv/bin/python -m pytest -q
```

`tests/conftest.py` starts one in-process dask client (two threads) for the session and puts
`bench/` on the path for the synthetic input generator `bench/synth.py`, which writes a small
raster with a nodata border and a nodata block so the valid-fraction rule is exercised.

## What is covered

| test | checks |
|---|---|
| `test_downsample.py` | the block reducer: mean with nodata, min/max in the source dtype, median in float32 for small integers, NaN rules, the valid-fraction rule, the all-nodata fast path, nearest at the block centre, leading dims, odd chunk alignment |
| `test_precondition_shard_size_multiple_of_chunk_size` | `shard_size % chunk_size == 0` is enforced |
| `test_encoding_shards_bounded_and_clipped` | shards are one slice on non-spatial dims and clipped to the array |
| `test_pipeline_2d_levels_and_georeferencing` | level count, exact `2**L` pixel size, top-left anchoring, CRS on every level, rasterio-consistent bounds, level-1 values against the rounded nodata-aware mean |
| `test_pipeline_3d_sharded` | a (t, y, x) input: shards `(1, s, s)`, level-1 values per slice |
| `test_resume_skips_written_bands_and_levels` | a second run over a complete store rewrites nothing (mtimes) |
| `test_validate_existing_band_accepts_nan_and_rejects_dtype_mismatch` | the existing-band validation |
| `test_fused_level1_store_levels_and_resume` | fused and unfused level 1, levels from the store and from memory, resume |
| `test_pipeline_float_nan_nodata_and_compressor_none` | NaN nodata, `--compressor none` |
| `test_window_failure_is_retried_then_resumed` | a failing window is retried; a permanent failure fails the level; the next run resumes from the marker without rewriting the windows done |
| `test_windowed_write_equals_reference` | window sizes 2, 4 and 64 give the same pyramid, partial edge windows included |
| `test_all_fill_blocks_skip_the_write` | an all-fill block leaves no shard; zeros under a NaN fill are stored |
| `test_band_last_source_is_one_task_per_block` | a (y, x, band) source keeps its bands in one dask block and gives the per-band reference |

The pipeline tests run the whole writer on 512²–1000² rasters and take about 40 s together.
The synthetic benchmark (`bench/compare.py --only s1`) is the smoke test of the harness.
