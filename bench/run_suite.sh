#!/usr/bin/env bash
# Every benchmark of bench/results.md for the optimized module, in one go (about an hour):
# synthetic scenarios (baseline vs optimized), the MinIO Italy window at chunk 4096 and 8192,
# the full Italy raster on MinIO and on a local copy. Threaded runs carry the store counters;
# 8-process runs give the wall time (counters then cover the main process only).
#   TAG=_v2 bash bench/run_suite.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source bench/s3env.sh
TAG=${TAG:-_v2}
OUT=bench/out
WIN=s3://test/geozzar-pyramid/input/WSF3Dv3_Italy_win32k.zarr
echo "== synthetic $(date +%T)"
.venv/bin/python bench/compare.py --threads 4 > "$OUT/suite_synthetic$TAG.log" 2>&1
cp "$OUT/results.md" "$OUT/synthetic$TAG.md"
for C in 4096 8192; do
  for mode in "t8 1" "w8 8"; do set -- $mode; M=$1; W=$2
    echo "== minio window chunk $C $M $(date +%T)"
    .venv/bin/python bench/run_one.py --impl optimized --input "$WIN" --output "s3://test/geozzar-pyramid/win32k_c${C}_optimized${TAG}.zarr" \
      --chunk-size "$C" --tile-width 256 --method mean --nodata 0 --sharding --threads 8 --workers "$W" --quiet \
      2> "$OUT/s3_win32k_c${C}${TAG}_$M.stderr" | tail -1 | tee "$OUT/s3_win32k_c${C}${TAG}_$M.json"
  done
done
for root in "s3://test/geozzar-pyramid minio" "bench/data/italy local"; do set -- $root; R=$1; N=$2
  for mode in "t8 1" "w8 8"; do set -- $mode; M=$1; W=$2
    echo "== full italy $N $M $(date +%T)"
    ROOT=$R EXTRACT=0 IMPL=optimized TAG="${TAG}_$M" WORKERS=$W bash bench/run_full_italy.sh
  done
done
echo "== suite done $(date +%T)"
