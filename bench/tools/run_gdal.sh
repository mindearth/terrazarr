#!/usr/bin/env bash
# GDAL 3.13.3 Zarr V3 pyramid: gdal_translate to a Zarr V3 array (256² chunks, zstd), then gdaladdo
# -r average writes the overviews as ovr_2x.. groups with a multiscales attribute (zarr-conventions v1).
# Runs in the official docker image (the system GDAL is 3.8). Reports wall, CPU, peak memory of the
# container (cgroup memory.peak) and object count as JSON.
#   bash bench/tools/run_gdal.sh IN.zarr OUT.zarr [levels=8] [blocksize=256]
set -uo pipefail
IN=$(realpath "$1"); OUT=$2; LEVELS=${3:-8}; BS=${4:-256}
IMG=ghcr.io/osgeo/gdal:ubuntu-full-3.13.3
rm -rf "$OUT"; mkdir -p "$(dirname "$OUT")"; OUTDIR=$(realpath "$(dirname "$OUT")"); NAME=$(basename "$OUT")
factors=""; f=2; for ((i=1; i<LEVELS; i++)); do factors="$factors $f"; f=$((f*2)); done
CID=gdalzarr_$$
t0=$(date +%s.%N)
docker run -d --name $CID -u "$(id -u):$(id -g)" -e GDAL_NUM_THREADS=8 -e GDAL_CACHEMAX=4096 \
  -v "$IN:/in:ro" -v "$OUTDIR:/out" $IMG bash -c "
    gdal_translate -q -of ZARR -co FORMAT=ZARR_V3 -co BLOCKSIZE=$BS,$BS -co COMPRESS=ZSTD 'ZARR:\"/in\":/data' /out/$NAME &&
    gdaladdo -q -r average /out/$NAME $factors" >/dev/null
CG=/sys/fs/cgroup/system.slice/docker-$(docker inspect -f '{{.Id}}' $CID).scope
peak=0
# the container's cgroup vanishes on exit, so its memory peak is polled while it runs
while [ "$(docker inspect -f '{{.State.Running}}' $CID 2>/dev/null)" = true ]; do
  v=$(cat $CG/memory.peak 2>/dev/null || echo 0); [ "$v" -gt "$peak" ] && peak=$v; sleep 1
done
rc=$(docker wait $CID)
wall=$(echo "$(date +%s.%N) - $t0" | bc)
docker logs $CID > "$OUTDIR/$NAME.gdal.stderr" 2>&1
docker rm $CID >/dev/null 2>&1
objects=$(find "$OUT" -type f 2>/dev/null | wc -l)
err=null; [ $rc -ne 0 ] && err="\"exit $rc: $(tail -1 "$OUTDIR/$NAME.gdal.stderr" | tr -d '"')\""
printf '{"impl": "gdal-3.13.2", "ok": %s, "error": %s, "wall_s": %.2f, "objects": %d, "peak_rss_mb": %.1f}\n' \
  "$([ $rc -eq 0 ] && echo true || echo false)" "$err" "$wall" "$objects" "$(echo "$peak / 1048576" | bc -l)"
