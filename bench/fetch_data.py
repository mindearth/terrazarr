# SPDX-License-Identifier: Apache-2.0
"""Download the benchmark inputs, the reference pyramids and the synthetic inputs from the public bucket.

    .venv/bin/python bench/fetch_data.py                # window inputs, window reference, synthetic (about 2 GB)
    .venv/bin/python bench/fetch_data.py --full         # plus the full-Italy inputs and reference (about 9 GB more)
    .venv/bin/python bench/fetch_data.py --only synthetic results

Anonymous reads from s3://me-public-assets/terrazarr (eu-central-1), no credentials needed; every
file is also reachable at https://me-public-assets.s3.eu-central-1.amazonaws.com/terrazarr/<key>.
The bucket allows anonymous GET but not LIST, so the keys come from MANIFEST.json, written by
bench/publish_data.sh. The WSF3D Italy data is CC BY 4.0. A file already present locally with the
same size is skipped.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import obstore
from obstore.store import S3Store

BUCKET, REGION, PREFIX = "me-public-assets", "eu-central-1", "terrazarr"
BENCH = Path(__file__).resolve().parent
LOCAL = {  # remote top-level folder -> local folder
    "inputs": BENCH / "data" / "italy" / "input",
    "reference": BENCH / "data" / "italy",
    "synthetic": BENCH / "data",
    "results": BENCH / "out" / "public",
}
SETS = {  # name -> (remote folder, key filter)
    "synthetic": ("synthetic", lambda k: True),
    "window": ("inputs", lambda k: "win32k" in k),
    "window-reference": ("reference", lambda k: "win32k" in k),
    "full": ("inputs", lambda k: "win32k" not in k),
    "full-reference": ("reference", lambda k: "win32k" not in k),
    "results": ("results", lambda k: True),
}
DEFAULT = ["synthetic", "window", "window-reference"]


def fetch(store: S3Store, key: str, dest: Path, size: int) -> int:
    if dest.exists() and dest.stat().st_size == size:
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in obstore.get(store, key).stream():
            f.write(chunk)
    return size


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--full", action="store_true", help="also the full-Italy inputs and reference")
    p.add_argument("--only", nargs="*", choices=list(SETS), help="a subset of the data sets")
    p.add_argument("--threads", type=int, default=16)
    a = p.parse_args()
    sets = a.only or DEFAULT + (["full", "full-reference"] if a.full else [])
    store = S3Store(BUCKET, region=REGION, skip_signature=True, prefix=PREFIX)
    manifest = json.loads(bytes(obstore.get(store, "MANIFEST.json").bytes()))
    for name in sets:
        folder, keep = SETS[name]
        objs = [o for o in manifest["objects"] if o["path"].startswith(folder + "/") and keep(o["path"])]
        jobs = [(o["path"], LOCAL[folder] / o["path"][len(folder) + 1:], o["size"]) for o in objs]
        with ThreadPoolExecutor(a.threads) as ex:
            got = sum(ex.map(lambda j: fetch(store, *j), jobs))
        total = sum(o["size"] for o in objs)
        print(f"{name:16} {len(objs):6d} objects {total / 1e9:6.2f} GB, downloaded {got / 1e9:.2f} GB -> {LOCAL[folder]}")


if __name__ == "__main__":
    main()
