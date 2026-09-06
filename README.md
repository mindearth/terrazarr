# geozarr-pyramid

GeoZarr multiscale pyramid writer (zarr v3, optional sharding), extracted from
`me-geotools/src/me_geotools/zarr_pyramid_v3` and optimized for laziness and scale.

- `src/geozarr_pyramid/` the optimized module (`geozarr.py`, `utils.py`, `store.py`, `cli.py`)
- `bench/baseline/geozarr_baseline/` the original module, verbatim except import paths, kept for comparison
- `bench/` synthetic inputs, a per-run harness and a baseline-vs-optimized driver
- `tests/` unit and end-to-end tests of the optimized module
- `CHANGES.md` what changed and why, with the audit findings each change addresses

## Install

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e . --group dev
```

Credentials for `s3://` paths come from the environment (`AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, `AWS_ENDPOINT_URL`); there is no dependency on private packages.

## Use

```bash
geozarr-pyramid --input in.zarr --output out.zarr \
  --chunk-size 4096 --tile-width 256 --sharding --method mean --nodata 0 \
  --workers 4 --threads-per-worker 4 --memory-limit 12GB
```

`--tile-width` is the zarr chunk on y/x (the tile served to clients). `--chunk-size` is the
shard on y/x with `--sharding`, and the dask block on y/x in every case; it must be a multiple of
the tile width. Peak memory per task is about one `chunk-size²` block times a small factor
(see `CHANGES.md`, H1), times `--threads-per-worker` per worker.

## Test and benchmark

Local, synthetic inputs:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python bench/compare.py              # writes bench/out/results.md
.venv/bin/python bench/compare.py --only s1    # one scenario
```

Each benchmark run is a separate process with an in-process dask cluster, so store traffic,
executed tasks and peak RSS of the whole pipeline are measured for both implementations on
identical inputs.

### Against MinIO

```bash
source bench/s3env.sh                       # exports MinIO credentials from ~/repos/.myenvs
.venv/bin/python bench/tif_to_zarr.py --src s3://bucket/x.tif --dst s3://bucket/x.zarr \
    --row0 74000 --col0 78000 --rows 32768 --cols 32768      # window of a striped GeoTIFF -> zarr input
.venv/bin/python bench/run_one.py --impl optimized --input s3://bucket/x.zarr --output s3://bucket/out.zarr \
    --chunk-size 4096 --tile-width 256 --sharding --method mean --nodata 0 --threads 8
```

The baseline needs `botocore < 1.36` against this MinIO (newer botocore omits the `Content-MD5`
header that MinIO requires on bulk deletes, which s3fs uses; me-geotools pins it for the same
reason). Build it once and run the baseline with that interpreter:

```bash
uv venv --python 3.11 .venv-baseline
uv pip install --python .venv-baseline/bin/python -e . --group dev "s3fs==2024.12.0" "aiobotocore==2.15.2" "botocore<1.36"
.venv-baseline/bin/python bench/run_one.py --impl baseline ...
```

The optimized module does not use s3fs and runs with current botocore. See `bench/results.md`.

#### Extract memory: strip size vs peak RSS

`tif_to_zarr.py` reads full-width row strips in parallel and every strip is one dask task, so
`--threads` strips are decoded at once and each is held until its `--chunk` row is written:

```
peak RSS  ≈  1.5 GB  +  threads × strip × width × itemsize
```

The constant is the GDAL block cache (1 GB), the curl cache (0.5 GB) and the interpreter.
Measured on `WSF3Dv3_Italy.tif` (width 200 599, float64, 8 threads):

| `--strip` | strip size | predicted | measured peak RSS |
|---:|---:|---:|---:|
| 2048 | 3.29 GB | 27.8 GB | 28.4 GB |
| 1024 | 1.64 GB | 14.6 GB | 13.8 GB |

So halving the strip halves the peak, and the peak scales linearly with the raster width and the
thread count. Pick `strip` so that `threads × strip × width × itemsize` is comfortably below the
free memory; the strip count only changes the number of range requests, not the total bytes read.
`bench/run_full_italy.sh` defaults to `STRIP=1024` for this reason. The pyramid step itself is
bounded by `threads × chunk-size² × itemsize × 2` instead (see above), 3.5 GB for the same raster
at chunk 4096.
