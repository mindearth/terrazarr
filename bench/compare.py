"""
Benchmark baseline vs optimized on synthetic inputs. Each run is a separate process.

    .venv/bin/python bench/compare.py            # all scenarios
    .venv/bin/python bench/compare.py --only s1  # one scenario
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable

SCENARIOS = {
    "s1": dict(desc="2D uint8 16384², sharded 4096, min, nodata 0", shape=(16384, 16384), dtype="uint8",
               args=["--chunk-size", "4096", "--tile-width", "256", "--method", "min", "--nodata", "0", "--sharding"]),
    "s2": dict(desc="2D uint8 16384², unsharded, mean, nodata 0", shape=(16384, 16384), dtype="uint8",
               args=["--chunk-size", "4096", "--tile-width", "256", "--method", "mean", "--nodata", "0"]),
    "s3": dict(desc="3D uint8 8×8192², sharded 4096, mean, nodata 0", shape=(8, 8192, 8192), dtype="uint8",
               args=["--chunk-size", "4096", "--tile-width", "256", "--method", "mean", "--nodata", "0", "--sharding"]),
    "s4": dict(desc="2D float32 12288², sharded 4096, median", shape=(12288, 12288), dtype="float32",
               args=["--chunk-size", "4096", "--tile-width", "256", "--method", "median", "--sharding"]),
}

METRICS = [
    ("wall_s", "wall time [s]"),
    ("peak_rss_mb", "peak RSS [MB]"),
    ("tasks_executed", "dask tasks"),
    ("store_get_chunk", "chunk GETs"),
    ("store_set_chunk", "chunk PUTs"),
    ("store_get_meta", "metadata GETs"),
    ("files_on_disk", "files on disk"),
]


def run(impl: str, inp: Path, out: Path, args: list[str], threads: int) -> dict:
    cmd = [PY, str(HERE / "run_one.py"), "--impl", impl, "--input", str(inp), "--output", str(out),
           "--threads", str(threads), "--quiet", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    last = [l for l in proc.stdout.splitlines() if l.startswith("{")]
    if not last:
        return {"impl": impl, "ok": False, "error": proc.stderr[-2000:]}
    return json.loads(last[-1])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", default=list(SCENARIOS))
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--data", default=str(HERE / "data"))
    p.add_argument("--out", default=str(HERE / "out"))
    a = p.parse_args()

    sys.path.insert(0, str(HERE))
    from synth import make_input

    data, out = Path(a.data), Path(a.out)
    data.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for key in a.only:
        sc = SCENARIOS[key]
        inp = data / f"{key}.zarr"
        if not (inp / "zarr.json").exists():
            print(f"[{key}] generating input {sc['shape']} {sc['dtype']} ...", flush=True)
            make_input(inp, shape=sc["shape"], dtype=sc["dtype"], input_chunk=2048,
                       nodata=0 if sc["dtype"] == "uint8" else None)
        results[key] = {}
        for impl in ("baseline", "optimized"):
            print(f"[{key}] {impl} ...", flush=True)
            r = run(impl, inp, out / f"{key}_{impl}.zarr", sc["args"], a.threads)
            results[key][impl] = r
            print(f"[{key}] {impl}: {json.dumps(r)}", flush=True)

    (out / "results.json").write_text(json.dumps(results, indent=2))

    lines = ["# Benchmark results", "", f"threads per run: {a.threads}", ""]
    for key, res in results.items():
        b, o = res["baseline"], res["optimized"]
        lines += [f"## {key}: {SCENARIOS[key]['desc']}", "", "| metric | baseline | optimized | ratio |", "|---|---:|---:|---:|"]
        for m, label in METRICS:
            bv, ov = b.get(m), o.get(m)
            ratio = f"{ov / bv:.2f}×" if isinstance(bv, (int, float)) and isinstance(ov, (int, float)) and bv else "n/a"
            lines.append(f"| {label} | {bv} | {ov} | {ratio} |")
        if not b.get("ok"):
            lines.append(f"\nbaseline error: `{b.get('error')}`")
        if not o.get("ok"):
            lines.append(f"\noptimized error: `{o.get('error')}`")
        lines.append("")
    (out / "results.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
