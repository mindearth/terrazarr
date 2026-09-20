# Benchmark harness

Scripts behind [docs/benchmarks.md](../docs/benchmarks.md); the "Reproducibility" section there
lists the machine, the data placeholders and one command per table.

| script | what |
|---|---|
| `fetch_data.py` | the public inputs, references and synthetic data (anonymous S3, see docs/benchmarks.md "Data") |
| `synth.py`, `compare.py` | synthetic inputs and the baseline-vs-optimized scenarios s1–s4 (no external data) |
| `run_one.py` | one run, one implementation, JSON metrics with provenance (git commit, versions, machine) |
| `run_suite.sh`, `run_full_italy.sh` | the window and full-Italy suites on an S3 store and locally |
| `tif_to_zarr.py`, `tiff_tiles.py` | GeoTIFF window extraction to zarr; tile directory statistics |
| `run_inputs_win32k.sh`, `run_inputs_full.sh` | the same raster as striped GeoTIFF, COG and zarr |
| `tools/` | runners for eopf-geozarr, topozarr and GDAL, and the shape-matched comparison |
| `scaling.py`, `memory_trace.py`, `task_breakdown.py`, `s3_window.py` | the scaling study |
| `compare_outputs.py` | level-by-level comparison of two pyramids, blockwise |
| `verify_windowed.sh` | verification of the windowed writer against the references |

`s3env.sh` exports the S3 credentials from a private env file; replace it with your own.
