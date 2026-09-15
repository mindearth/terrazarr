#!/usr/bin/env bash
# Verification of the windowed level-0 writer: window and full Italy against the baseline and the
# previous timings, the synthetic scaling point where the main process used to grow, the all-nodata
# run for the skip-empty saving, and a data window of agea4 for the band-last path.
set -uo pipefail
cd "$(dirname "$0")/.."; source bench/s3env.sh
IN=bench/data/italy/input/WSF3Dv3_Italy_win32k.zarr; D=bench/data/italy; OUT=bench/out
for mode in "w8 8" "t8 1"; do set -- $mode
  echo "== window $1 $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
  /usr/bin/time -f 'time: wall=%e cpu=%P maxrss=%MKB' .venv/bin/python bench/run_one.py --impl optimized --input $IN --output $D/win_windowed_$1.zarr --shard-size 4096 --chunk-size 256 --method mean --nodata 0 --sharding --threads 8 --workers $2 --quiet 2> $OUT/windowed_win32k_$1.stderr | tail -1 | tee $OUT/windowed_win32k_$1.json
  grep '^time:' $OUT/windowed_win32k_$1.stderr
done
echo "== compare window w8 vs baseline $(date +%T)"; .venv/bin/python bench/compare_outputs.py $D/win32k_baseline.zarr $D/win_windowed_w8.zarr --threads 8 2>&1 | grep -v -i warn
echo "== full italy w8 $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
/usr/bin/time -f 'time: wall=%e cpu=%P maxrss=%MKB' .venv/bin/python bench/run_one.py --impl optimized --input $D/input/WSF3Dv3_Italy_full.zarr --output $D/full_windowed_w8.zarr --shard-size 4096 --chunk-size 256 --method mean --nodata 0 --sharding --threads 8 --workers 8 --quiet 2> $OUT/windowed_full_w8.stderr | tail -1 | tee $OUT/windowed_full_w8.json
grep '^time:' $OUT/windowed_full_w8.stderr
echo "== scaling 262144 chunk 4096 $(date +%T)"; .venv/bin/python bench/scaling.py --sizes 262144 --workers 8 --out $OUT/scaling_windowed.jsonl 2>&1 | grep -v -i warn | grep "^{" | cut -c1-330
echo "== task breakdown 65536 all-nodata $(date +%T)"; .venv/bin/python bench/task_breakdown.py 65536 bench/data/scaling 2>&1 | grep -vE "Warning|warn|instead|Perhaps|^\s*$" | head -6
echo "== agea4 data window 32768 $(date +%T)"; .venv/bin/python bench/s3_window.py 32768 bench/data/scaling/agea4_win32k_windowed.zarr 2>&1 | grep -vE "Warning|warn|instead|Perhaps|^\s*$" | head -8
echo "== compare full w8 vs baseline $(date +%T)"; .venv/bin/python bench/compare_outputs.py $D/italy_full_baseline.zarr $D/full_windowed_w8.zarr --threads 8 2>&1 | grep -v -i warn
echo "== verify done $(date +%T)"
