# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added

- `chunk_size` and `shard_size` (`--chunk-size`, `--shard-size`) also take a size per dim, e.g.
  `chunk_size={"y": 512, "x": 512, "band": 4}` (`--chunk-size y=512,x=512,band=4`) writes
  `(4, 512, 512)` chunks; the non-spatial dims were fixed at 1. An int still sets y and x, and y
  must equal x. A shard is a multiple of its chunk and is the dask block on that dim, so task
  memory is the product of the shard on every dim.
- A dim left out of `chunk_size` / `shard_size` has the chunk and shard of the source (the zarr
  encoding of its first data variable), and 256 and 4096 on y/x and 1 on another dim when the
  source has none. A source with a long chunk on a non-spatial dim now gives a long shard there,
  which raises task memory.

### Changed

- A level already on the store whose chunks or shards differ from the layout being written is
  rewritten instead of reused or resumed (resuming it could have two tasks rewrite one shard).

### Fixed

- The tile matrix set declared 256-pixel tiles whatever `chunk_size` was, which disagreed with the
  chunks and with `tile_matrix_limits`; its tile is now the y/x chunk.

## [0.1.4] - 2026-10-14

### Added

- `method="sum"` / `--method sum`: each output pixel is the sum of the valid source pixels (NaN and the numeric
  `nodata` are excluded) and is nodata only where none is valid, so totals such as population
  are conserved from level to level. The result keeps the source dtype; integers saturate.

## [0.1.3] - 2026-10-08

### Changed

- Level-0 resume is now opt-in: `resume=True` / `--resume` checks the completion marker, skips a
  complete level 0 and writes only the windows an interrupted run left out. By default a valid level
  0 already on the store is used as it is, and an absent or invalid one is written from scratch.

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

[Unreleased]: https://github.com/mindearth/terrazarr/compare/v0.1.4...HEAD
[0.1.4]: https://github.com/mindearth/terrazarr/releases/tag/v0.1.4
[0.1.3]: https://github.com/mindearth/terrazarr/releases/tag/v0.1.3
[0.1.2]: https://github.com/mindearth/terrazarr/releases/tag/v0.1.2
