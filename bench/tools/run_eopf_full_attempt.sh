#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# eopf-geozarr 0.11.0 on full WSF-3D Italy under a memory cap, with dask spilling to the data disk
# and a sampler of resident memory and spill size, until it finishes, dies or is stopped:
#   systemctl --user stop terrazarr-eopf-full
# The end state goes to bench/out/eopf_full_attempt.log as an "== attempt ..." line, which
# bench/tools/summarize.py prints in the full-Italy table.
set -euo pipefail
cd "$(dirname "$0")/../.."
IN=bench/data/italy/input/WSF3Dv3_Italy_full.zarr
OUT=bench/data/italy/tools/full_eopf.zarr
LOG=bench/out/eopf_full_attempt.log
CAP=${CAP:-30G}; T=${T:-90m}; SPILL=$PWD/bench/data/dask-tmp
rm -rf "$OUT" "$SPILL"; mkdir -p bench/out
systemd-run --user --unit terrazarr-eopf-full -p MemoryMax=$CAP -q --working-directory="$PWD" bash -c "
  export DASK_TEMPORARY_DIRECTORY=$SPILL
  echo \"== eopf-geozarr 0.11.0 on full WSF-3D Italy, $CAP cap, dask spill on the data disk \$(date +%T)\" >> $LOG
  ( while true; do
      echo \"\$(date +%T) rss=\$(ps -o rss= -C python | awk '{s+=\$1} END{printf \"%.1f\", s/1048576}')GB spill=\$(du -sh $SPILL 2>/dev/null | cut -f1) free=\$(df -h --output=avail $SPILL | tail -1 | tr -d ' ')\" >> $LOG
      sleep 60; done ) &
  S=\$!
  timeout $T /usr/bin/time -f 'time: wall=%e cpu=%P maxrss=%MKB' .venv-eopf/bin/python bench/tools/run_eopf.py \
    --input $IN --output $OUT --shard-size 4096 --chunk-size 256 --min-dimension 256 --threads 8 --quiet >> $LOG 2>&1 \
    && echo \"== attempt done \$(date +%T): finished\" >> $LOG \
    || echo \"== attempt ended \$(date +%T) with exit \$?: not finished, see the lines above\" >> $LOG
  kill \$S"
echo "running as unit terrazarr-eopf-full; follow with: tail -f $LOG"
