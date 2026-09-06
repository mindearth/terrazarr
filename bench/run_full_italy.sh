#!/usr/bin/env bash
# Full WSF3Dv3_Italy.tif (178335 x 200599 float64) -> zarr input -> GeoZarr pyramid on MinIO, optimized module.
# Memory: the extract holds up to THREADS full-width strips in flight (STRIP x 200599 x 8 B = 1.6 GB at STRIP=1024).
set -euo pipefail
cd "$(dirname "$0")/.."
source bench/s3env.sh
SRC=s3://wsf-artifacts/3d/local/italy/20240101/original/WSF3Dv3_Italy.tif
IN=s3://test/geozzar-pyramid/input/WSF3Dv3_Italy_full.zarr
OUT=s3://test/geozzar-pyramid/italy_full_optimized.zarr
THREADS=${THREADS:-8}
echo "== extract $(date +%T)"
.venv/bin/python bench/tif_to_zarr.py --src "$SRC" --dst "$IN" --strip "${STRIP:-1024}" --chunk 2048 --threads "$THREADS"
echo "== pyramid $(date +%T)"
.venv/bin/python bench/run_one.py --impl optimized --input "$IN" --output "$OUT" \
  --chunk-size 4096 --tile-width 256 --method mean --nodata 0 --sharding --threads "$THREADS" --quiet \
  2> bench/out/italy_full_optimized.stderr | tail -1 | tee bench/out/italy_full_optimized.json
echo "== done $(date +%T)"
