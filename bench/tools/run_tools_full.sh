#!/usr/bin/env bash
# Full WSF3Dv3 Italy (local zarr input, 178335×200599 float64) through this module (8 processes),
# topozarr (8 workers) and GDAL 3.13.3, then each output compared with the baseline pyramid,
# levels matched by shape. The EOPF converter is not run: it computes every overview from the
# whole previous level as one numpy array (level 0 is 286 GB).
#   bash bench/tools/run_tools_full.sh          # bench/out/tools_full_*.json, log on stdout
set -uo pipefail
cd "$(dirname "$0")/../.."
IN=bench/data/italy/input/WSF3Dv3_Italy_full.zarr
D=bench/data/italy/tools; mkdir -p $D; OUT=bench/out; REF=bench/data/italy/italy_full_baseline.zarr
run() { # name cap timeout cmd...
  local name=$1 cap=$2 T=$3; shift 3
  echo "== $name $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
  systemd-run --user --scope -p MemoryMax=$cap -q timeout $T /usr/bin/time -f 'time: wall=%e cpu=%P maxrss=%MKB' "$@" 2> "$OUT/tools_full_$name.stderr" | tail -1 | tee "$OUT/tools_full_$name.json"
  echo "exit=$? $(grep '^time:' "$OUT/tools_full_$name.stderr" || echo 'time: (killed by timeout)')"; du -sh $D/full_$name.zarr 2>/dev/null | cut -f1 | sed 's/^/size: /'
}
run ours_w8 24G 3h .venv/bin/python bench/run_one.py --impl optimized --input $IN --output $D/full_ours_w8.zarr --shard-size 4096 --chunk-size 256 --method mean --nodata 0 --sharding --threads 8 --workers 8 --quiet
run topozarr 24G 3h .venv-topozarr/bin/python bench/tools/run_topozarr.py --input $IN --output $D/full_topozarr.zarr --levels 10 --workers 8
run gdal 24G 3h bash bench/tools/run_gdal.sh $IN $D/full_gdal.zarr 10 256
for name in ours_w8 topozarr gdal; do
  echo "== compare $name vs baseline $(date +%T)"
  systemd-run --user --scope -p MemoryMax=16G -q timeout 2h .venv/bin/python bench/tools/compare_any.py $REF $D/full_$name.zarr --threads 8 2>&1 | grep -v -i warn
done
echo "== tools full done $(date +%T)"
