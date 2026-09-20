#!/usr/bin/env bash
# Full WSF3Dv3_Italy.tif (178335 x 200599 float64) -> zarr input -> GeoZarr pyramid on MinIO.
#   IMPL=optimized|baseline   which module (baseline runs in .venv-baseline, see README)
#   EXTRACT=1|0               run the GeoTIFF -> zarr extract (0: reuse the input already on MinIO)
#   THREADS, STRIP            dask threads; rows per extract strip
#   ROOT                      store root for input and output (default MinIO; a local dir for a
#                             latency-free run, e.g. ROOT=bench/data/italy EXTRACT=0 TAG=_local)
#   TAG                       suffix for the output store and the bench/out result files
#   WORKERS                   dask worker processes (THREADS are split across them)
# Extract memory: peak RSS ~ 1.5 GB + THREADS x STRIP x width x itemsize (see README).
set -euo pipefail
cd "$(dirname "$0")/.."
source bench/s3env.sh
IMPL=${IMPL:-optimized}
EXTRACT=${EXTRACT:-1}
THREADS=${THREADS:-8}
WORKERS=${WORKERS:-1}
STRIP=${STRIP:-1024}
SRC=s3://wsf-artifacts/3d/local/italy/20240101/original/WSF3Dv3_Italy.tif
ROOT=${ROOT:-s3://test/geozzar-pyramid}
TAG=${TAG:-}
IN=$ROOT/input/WSF3Dv3_Italy_full.zarr
OUT=$ROOT/italy_full_${IMPL}${TAG}.zarr
PY=.venv/bin/python
[ "$IMPL" = baseline ] && PY=.venv-baseline/bin/python
if [ "$EXTRACT" = 1 ]; then
  echo "== extract $(date +%T)"
  .venv/bin/python bench/tif_to_zarr.py --src "$SRC" --dst "$IN" --strip "$STRIP" --chunk 2048 --threads "$THREADS"
fi
echo "== pyramid $IMPL $(date +%T)"
"$PY" bench/run_one.py --impl "$IMPL" --input "$IN" --output "$OUT" \
  --shard-size 4096 --chunk-size 256 --method mean --nodata 0 --sharding --threads "$THREADS" --workers "$WORKERS" --quiet \
  2> "bench/out/italy_full_${IMPL}${TAG}.stderr" | tail -1 | tee "bench/out/italy_full_${IMPL}${TAG}.json"
echo "== done $(date +%T)"
