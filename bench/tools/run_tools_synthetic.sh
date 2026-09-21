#!/usr/bin/env bash
# The synthetic scenarios s1-s4 (bench/synth.py inputs, present under bench/data after
# bench/compare.py or bench/fetch_data.py) through the four GeoZarr writers with the mean method,
# 4 threads or workers, sharded 256-in-4096 where the writer takes it; then each output compared
# with terrazarr's, levels matched by shape. terrazarr additionally runs each scenario's own
# method (s1 min, s4 median) for the numbers of the terrazarr-vs-eopf table.
#   bash bench/tools/run_tools_synthetic.sh          # bench/out/tools_synth_*.json, log on stdout
set -uo pipefail
cd "$(dirname "$0")/../.."
D=bench/data/tools_synth; mkdir -p $D; OUT=bench/out
run() { # name cap cmd...
  local name=$1 cap=$2; shift 2
  echo "== $name $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
  systemd-run --user --scope -p MemoryMax=$cap -q /usr/bin/time -f 'time: wall=%e cpu=%P maxrss=%MKB' "$@" 2> "$OUT/tools_synth_$name.stderr" | tail -1 | tee "$OUT/tools_synth_$name.json"
  echo "exit=$? $(grep '^time:' "$OUT/tools_synth_$name.stderr" || echo 'time: n/a')"; du -sh $D/$name.zarr 2>/dev/null | cut -f1 | sed 's/^/size: /'
}
for s in s1 s2 s3 s4; do
  IN=bench/data/$s.zarr; SHARD="--sharding"; [ $s = s2 ] && SHARD=""
  run ${s}_terrazarr_w4 16G .venv/bin/python bench/run_one.py --impl optimized --input $IN --output $D/${s}_terrazarr_w4.zarr --shard-size 4096 --chunk-size 256 --method mean --nodata 0 $SHARD --threads 4 --workers 4 --quiet
  run ${s}_terrazarr_t4 16G .venv/bin/python bench/run_one.py --impl optimized --input $IN --output $D/${s}_terrazarr_t4.zarr --shard-size 4096 --chunk-size 256 --method mean --nodata 0 $SHARD --threads 4 --workers 1 --quiet
  run ${s}_eopf 32G .venv-eopf/bin/python bench/tools/run_eopf.py --input $IN --output $D/${s}_eopf.zarr --shard-size 4096 --min-dimension 256 --threads 4 --quiet
  run ${s}_topozarr 16G .venv-topozarr/bin/python bench/tools/run_topozarr.py --input $IN --output $D/${s}_topozarr.zarr --levels 7 --workers 4
  run ${s}_gdal 16G bash bench/tools/run_gdal.sh $IN $D/${s}_gdal.zarr 7 256
done
for s in s1 s4; do
  m=min; [ $s = s4 ] && m=median
  run ${s}_terrazarr_${m}_w4 16G .venv/bin/python bench/run_one.py --impl optimized --input bench/data/$s.zarr --output $D/${s}_terrazarr_${m}_w4.zarr --shard-size 4096 --chunk-size 256 --method $m --nodata 0 --sharding --threads 4 --workers 4 --quiet
done
for s in s1 s2 s3 s4; do for t in eopf topozarr gdal; do
  echo "== compare $s $t vs terrazarr $(date +%T)"
  systemd-run --user --scope -p MemoryMax=16G -q .venv/bin/python bench/tools/compare_any.py $D/${s}_terrazarr_w4.zarr $D/${s}_$t.zarr --threads 4 --chunk 2048 2>&1 | grep -v -i warn
done; done
echo "== tools synthetic done $(date +%T)"
