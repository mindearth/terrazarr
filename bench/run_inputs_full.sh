#!/usr/bin/env bash
# Full WSF3Dv3 Italy from three input formats: COG (512² tiles) and zarr (2048² chunks) on MinIO
# in both layouts and locally with 8 processes, then the striped GeoTIFF (one-row strips)
# threaded on MinIO with a 12 GB GDAL block cache and a time cap. Results in bench/out/inputs_full_*.json.
#   bash bench/run_inputs_full.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source bench/s3env.sh
OUT=bench/out
run() { # store input layout workers cache timeout
  local store=$1 input=$2 M=$3 W=$4 CACHE=$5 T=$6 IN DST
  if [ $store = local ]; then ROOT=bench/data/italy/input; DST=bench/data/italy/inputs_full_exp.zarr
  else ROOT=s3://test/geozzar-pyramid/input; DST=s3://test/geozzar-pyramid/italy_full_inputs_exp.zarr; fi
  case $input in
    striped) IN=$ROOT/WSF3Dv3_Italy_striped.tif ;;
    cog)     IN=$ROOT/WSF3Dv3_Italy_cog.tif ;;
    zarr)    IN=$ROOT/WSF3Dv3_Italy_full.zarr ;;
  esac
  echo "== $store $input $M $(date +%T)"
  GDAL_CACHEMAX=$CACHE timeout "$T" /usr/bin/time -f 'time: wall=%e cpu=%P maxrss=%MKB' \
    .venv/bin/python bench/run_one.py --impl optimized --input "$IN" --output "$DST" \
    --chunk-size 4096 --tile-width 256 --method mean --nodata 0 --sharding --threads 8 --workers "$W" --quiet \
    2> "$OUT/inputs_full_${store}_${input}_$M.stderr" | tail -1 | tee "$OUT/inputs_full_${store}_${input}_$M.json"
  echo "exit=$? $(grep '^time:' "$OUT/inputs_full_${store}_${input}_$M.stderr" || echo 'time: (killed by timeout)')"
}
run minio cog  t8 1 2048 3h
run minio cog  w8 8 1024 3h
run minio zarr t8 1 2048 3h
run minio zarr w8 8 1024 3h
run local cog  w8 8 1024 3h
run local zarr w8 8 1024 3h
run minio striped t8 1 12288 2h
echo "== inputs full done $(date +%T)"
