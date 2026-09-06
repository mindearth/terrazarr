#!/usr/bin/env bash
# Full WSF3Dv3_Italy.tif (178335 x 200599 float64) -> zarr input -> GeoZarr pyramid on MinIO.
#   IMPL=optimized|baseline   which module (baseline runs in .venv-baseline, see README)
#   EXTRACT=1|0               run the GeoTIFF -> zarr extract (0: reuse the input already on MinIO)
#   THREADS, STRIP            dask threads; rows per extract strip
# Extract memory: peak RSS ~ 1.5 GB + THREADS x STRIP x width x itemsize (see README).
set -euo pipefail
cd "$(dirname "$0")/.."
source bench/s3env.sh
IMPL=${IMPL:-optimized}
EXTRACT=${EXTRACT:-1}
THREADS=${THREADS:-8}
STRIP=${STRIP:-1024}
SRC=s3://wsf-artifacts/3d/local/italy/20240101/original/WSF3Dv3_Italy.tif
IN=s3://test/geozzar-pyramid/input/WSF3Dv3_Italy_full.zarr
OUT=s3://test/geozzar-pyramid/italy_full_${IMPL}.zarr
PY=.venv/bin/python
[ "$IMPL" = baseline ] && PY=.venv-baseline/bin/python
if [ "$EXTRACT" = 1 ]; then
  echo "== extract $(date +%T)"
  .venv/bin/python bench/tif_to_zarr.py --src "$SRC" --dst "$IN" --strip "$STRIP" --chunk 2048 --threads "$THREADS"
fi
echo "== pyramid $IMPL $(date +%T)"
"$PY" bench/run_one.py --impl "$IMPL" --input "$IN" --output "$OUT" \
  --chunk-size 4096 --tile-width 256 --method mean --nodata 0 --sharding --threads "$THREADS" --quiet \
  2> "bench/out/italy_full_${IMPL}.stderr" | tail -1 | tee "bench/out/italy_full_${IMPL}.json"
echo "== done $(date +%T)"
