#!/usr/bin/env bash
# The 32768² Italy window (local zarr input) through four GeoZarr pyramid writers with matched
# settings (mean, 8 levels down to 256 px, 8 threads or workers): this module (threaded and with 8
# processes), the upstream EOPF converter, topozarr and GDAL 3.13.2 (gdal_translate + gdaladdo).
# Each output is then compared with the baseline window, levels matched by shape.
#   bash bench/tools/run_tools_win32k.sh          # bench/out/tools_win32k_*.json, log on stdout
set -uo pipefail
cd "$(dirname "$0")/../.."
IN=bench/data/italy/input/WSF3Dv3_Italy_win32k.zarr
D=bench/data/italy/tools; mkdir -p $D; OUT=bench/out; REF=bench/data/italy/win32k_baseline.zarr
run() { # name cap cmd...
  local name=$1 cap=$2; shift 2
  echo "== $name $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
  systemd-run --user --scope -p MemoryMax=$cap -q /usr/bin/time -f 'time: wall=%e cpu=%P maxrss=%MKB' "$@" 2> "$OUT/tools_win32k_$name.stderr" | tail -1 | tee "$OUT/tools_win32k_$name.json"
  echo "exit=$? $(grep '^time:' "$OUT/tools_win32k_$name.stderr" || echo 'time: n/a')"; du -sh $D/$name.zarr 2>/dev/null | cut -f1 | sed 's/^/size: /'
}
run ours_t8 24G .venv/bin/python bench/run_one.py --impl optimized --input $IN --output $D/ours_t8.zarr --chunk-size 4096 --tile-width 256 --method mean --nodata 0 --sharding --threads 8 --workers 1 --quiet
run ours_w8 24G .venv/bin/python bench/run_one.py --impl optimized --input $IN --output $D/ours_w8.zarr --chunk-size 4096 --tile-width 256 --method mean --nodata 0 --sharding --threads 8 --workers 8 --quiet
run topozarr 24G .venv-topozarr/bin/python bench/tools/run_topozarr.py --input $IN --output $D/topozarr.zarr --levels 8 --workers 8
run gdal 24G bash bench/tools/run_gdal.sh $IN $D/gdal.zarr 8 256
run eopf 34G .venv-eopf/bin/python bench/tools/run_eopf.py --input $IN --output $D/eopf.zarr --chunk-size 4096 --tile-width 256 --min-dimension 256 --threads 8 --quiet
for name in ours_t8 ours_w8 topozarr gdal eopf; do
  echo "== compare $name vs baseline $(date +%T)"
  systemd-run --user --scope -p MemoryMax=16G -q .venv/bin/python bench/tools/compare_any.py $REF $D/$name.zarr --threads 8 2>&1 | grep -v -i warn
done
echo "== tools win32k done $(date +%T)"
