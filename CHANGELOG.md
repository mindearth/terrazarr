# Changelog

All notable changes to this project are documented in this file.

## [0.1.2] - 2026-09-23

Initial release.

### Added

- `to_geozarr`, a lazy, windowed and resumable writer of GeoZarr multiscale pyramids to zarr v3
  stores with sharding, for any xarray `DataArray`, `Dataset` or `DataTree` with a CRS.
- `ds.terrazarr.to_geozarr(...)` xarray accessor and a `terrazarr` command-line interface, both
  wrapping the same writer.
- Flat, sub-gigabyte peak memory regardless of raster size: level 0 is one task per output shard,
  each level above is reduced from its parent shards read one at a time, and nothing is ever
  materialised as a whole array in memory.
- Resumable level-0 writes: each finished window is recorded, so an interrupted run continues
  from the missing windows instead of starting over.
- Store-aware I/O: one read per input block, shard-aligned reads and writes, all-nodata blocks
  skipped, band-last images transposed once per block.
- Exact overview semantics: `2**L` native pixels per pixel at level `L`, anchored top-left,
  configurable nodata rule (mean of valid pixels, block blanked below 30% valid), integer
  rounding, and a decodable CRS at every level.
- Local paths and `s3://` URLs supported for both input and output.
- Configurable chunk size, shard size, aggregation method, nodata value, Dask worker topology,
  and compressor/compression level.
- Documentation site with API reference, benchmarks against eopf-geozarr, topozarr and GDAL, and
  a live demo serving a GeoZarr pyramid through TiTiler.

[0.1.2]: https://github.com/mindearth/terrazarr/releases/tag/v0.1.2
