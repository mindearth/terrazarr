#!/usr/bin/env bash
# Publish the benchmark inputs, the reference pyramids, the synthetic inputs and the results to the
# public bucket, then write MANIFEST.json (keys and sizes, what bench/fetch_data.py reads: the bucket
# allows anonymous GET but not LIST) and an attribution note. Needs write access to the bucket.
#   AWS_PROFILE=<writer profile> bash bench/publish_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."
B=${BUCKET:-s3://me-public-assets/terrazarr}; D=bench/data/italy
BUCKET_NAME=$(echo "$B" | sed -E 's#^s3://([^/]+).*#\1#'); KEY_PREFIX=$(echo "$B" | sed -E 's#^s3://[^/]+/##')
log() { echo "== $* $(date +%T)"; }
log results;   aws s3 sync bench/out "$B/results" --only-show-errors --exclude "*.zarr/*" --exclude "public/*"
log synthetic; for s in s1 s2 s3 s4 smoke_u8; do aws s3 sync "bench/data/$s.zarr" "$B/synthetic/$s.zarr" --only-show-errors; done
log inputs;    aws s3 sync "$D/input" "$B/inputs" --only-show-errors
log reference; for r in win32k_baseline italy_full_baseline; do aws s3 sync "$D/$r.zarr" "$B/reference/$r.zarr" --only-show-errors; done
log manifest
aws s3api list-objects-v2 --bucket "$BUCKET_NAME" --prefix "$KEY_PREFIX/" --query "Contents[].{path: Key, size: Size}" --output json \
  | python3 -c "
import json, sys, datetime
prefix = '$KEY_PREFIX/'
objs = [{'path': o['path'][len(prefix):], 'size': o['size']} for o in json.load(sys.stdin) if o['path'].startswith(prefix) and not o['path'].endswith(('MANIFEST.json', 'README.txt'))]
json.dump({'generated': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'), 'count': len(objs), 'bytes': sum(o['size'] for o in objs), 'objects': objs}, open('/tmp/terrazarr_manifest.json', 'w'))
print(len(objs), 'objects', round(sum(o['size'] for o in objs) / 1e9, 2), 'GB')"
aws s3 cp /tmp/terrazarr_manifest.json "$B/MANIFEST.json" --only-show-errors --content-type application/json
cat > /tmp/terrazarr_readme.txt <<'EOF2'
terrazarr benchmark data (https://github.com/mindearth/terrazarr)

inputs/     WSF3Dv3_Italy: the World Settlement Footprint 3D layer over Italy as the source striped
            GeoTIFF, as a COG, as a 2048-chunked zarr v3 store, and a 32768² window of each.
            License CC BY 4.0; see docs/benchmarks.md in the repository for the provenance.
reference/  the pyramids of the original module (eopf-geozarr fork) used by every comparison, CC BY 4.0.
synthetic/  the synthetic scenarios' inputs (bench/synth.py), Apache-2.0.
results/    the JSON and log files behind the benchmark tables, Apache-2.0.
MANIFEST.json  keys and sizes; bench/fetch_data.py downloads from it (anonymous GET, no LIST).
EOF2
aws s3 cp /tmp/terrazarr_readme.txt "$B/README.txt" --only-show-errors --content-type text/plain
log "published"
