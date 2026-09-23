#!/usr/bin/env bash
# The 32768² Italy window from three input formats (striped GeoTIFF with one-row strips, COG
# with 512² tiles, zarr with 2048² chunks), on local NVMe and on MinIO, threaded (t8) and with
# 8 worker processes (w8). Same pyramid settings as the other window benchmarks.
#   bash bench/run_inputs_win32k.sh          # results in bench/out/inputs_win32k_*.json
# Striped GeoTIFFs need GDAL's block cache to hold a block-row of strips (4096 rows), otherwise
# every block of the row decodes them again: GDAL_CACHEMAX is set per layout.
set -euo pipefail
cd "$(dirname "$0")/.."
source bench/s3env.sh
OUT=bench/out
for store in local minio; do
  if [ $store = local ]; then ROOT=bench/data/italy/input; DST=bench/data/italy/inputs_exp.zarr
  else ROOT=s3://test/geozzar-pyramid/input; DST=s3://test/geozzar-pyramid/win32k_inputs_exp.zarr; fi
  for input in striped cog zarr; do
    case $input in
      striped) IN=$ROOT/WSF3Dv3_Italy_win32k_striped.tif ;;
      cog)     IN=$ROOT/WSF3Dv3_Italy_win32k_cog.tif ;;
      zarr)    IN=$ROOT/WSF3Dv3_Italy_win32k.zarr ;;
    esac
    for mode in "t8 1 8192" "w8 8 1024"; do set -- $mode; M=$1; W=$2; CACHE=$3
      echo "== $store $input $M $(date +%T)"
      GDAL_CACHEMAX=$CACHE /usr/bin/time -f 'time: wall=%e cpu=%P maxrss=%MKB' \
        .venv/bin/python bench/run_one.py --impl optimized --input "$IN" --output "$DST" \
        --shard-size 4096 --chunk-size 256 --method mean --nodata 0 --sharding --threads 8 --workers "$W" --quiet \
        2> "$OUT/inputs_win32k_${store}_${input}_$M.stderr" | tail -1 | tee "$OUT/inputs_win32k_${store}_${input}_$M.json"
      grep "^time:" "$OUT/inputs_win32k_${store}_${input}_$M.stderr" || true
    done
  done
done
echo "== inputs win32k done $(date +%T)"
