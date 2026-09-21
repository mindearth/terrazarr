# SPDX-License-Identifier: Apache-2.0
"""Markdown tables from the writer-comparison results (bench/out/tools_*.json, bench/out/tools_all.log).

    .venv/bin/python bench/tools/summarize.py [bench/out/tools_all.log] > /tmp/tables.md
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "out"
TOOLS = [("terrazarr_w4", "terrazarr, 4 processes"), ("terrazarr_t4", "terrazarr, 4 threads"), ("eopf", "eopf-geozarr 0.11.0"),
         ("topozarr", "topozarr 0.1.9"), ("gdal", "GDAL 3.13.3")]
SCEN = {"s1": "s1: uint8 16384², sharded", "s2": "s2: uint8 16384², unsharded", "s3": "s3: uint8 8×8192², sharded", "s4": "s4: float32 12288², sharded"}


def times(log: str) -> dict[str, tuple[float, str, float]]:
    """name -> (wall from /usr/bin/time, cpu, maxrss GB) parsed from the suite logs."""
    out = {}
    for m in re.finditer(r"^== (\S+) \d\d:\d\d:\d\d.*?\n(?:.*?\n)*?exit=\d+ time: wall=([\d.]+) cpu=(\d+)% maxrss=(\d+)KB", log, re.M):
        out[m.group(1)] = (float(m.group(2)), m.group(3) + " %", int(m.group(4)) / 2**20)
    return out


def compare(log: str, key: str) -> str:
    """Level-1 and last-level differences from a compare block titled '== compare <key> ...'."""
    m = re.search(rf"^== compare {re.escape(key)} .*?\n(.*?)(?=^== |\Z)", log, re.M | re.S)
    if not m:
        return "n/a"
    rows = [r.split() for r in m.group(1).splitlines() if re.match(r"\s*\S+\s+\(", r)]
    rows = [r for r in rows if r[-4] not in ("no",)]
    if len(rows) < 2:
        return "no matching level"
    def cell(r): return f"{float(r[-3]):.3g} / {100*float(r[-2]):.1f} %"
    return f"L1 {cell(rows[1])}; last {cell(rows[-1])}"


def load(name: str) -> dict | None:
    p = OUT / f"{name}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text().strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    log = Path(sys.argv[1]).read_text() if len(sys.argv) > 1 else (OUT / "tools_all.log").read_text()
    t = times(log)
    print("### Synthetic scenarios, mean, 4 workers or threads\n")
    print("| scenario | writer | wall [s] | CPU | peak RSS [GB] | objects | overview values vs terrazarr (max diff / pixels differing, level 1; last level) |")
    print("|---|---|---:|---:|---:|---:|---|")
    for s, desc in SCEN.items():
        for key, label in TOOLS:
            name = f"{s}_{key}"; j = load(f"tools_synth_{name}"); w = t.get(name)
            if j is None or w is None:
                print(f"| {desc} | {label} | – | – | – | – | not run |"); continue
            ok = j.get("ok", True)
            val = "reference" if key.startswith("terrazarr") else compare(log, f"{s} {key} vs terrazarr")
            print(f"| {desc} | {label} | {w[0]:.1f} | {w[1]} | {w[2]:.2f} | {j.get('objects', '–')} | {val if ok else 'failed: ' + str(j.get('error'))[:60]} |")
    for suite, title, tools in (("win32k", "WSF-3D Italy, 32768² window", [("ours_w8", "terrazarr, 8 processes"), ("ours_t8", "terrazarr, 8 threads"), ("eopf", "eopf-geozarr 0.11.0"), ("topozarr", "topozarr 0.1.9"), ("gdal", "GDAL 3.13.3")]),
                                ("full", "WSF-3D Italy, full", [("ours_w8", "terrazarr, 8 processes"), ("eopf", "eopf-geozarr 0.11.0"), ("topozarr", "topozarr 0.1.9"), ("gdal", "GDAL 3.13.3")])):
        print(f"\n### {title}\n")
        print("| writer | wall [s] | CPU | peak RSS [GB] | objects | overview values vs the reference (max diff / pixels differing, level 1; last level) |")
        print("|---|---:|---:|---:|---:|---|")
        for key, label in tools:
            j = load(f"tools_{suite}_{key}"); w = t.get(key)
            if j is None or w is None:
                print(f"| {label} | – | – | – | – | not run |"); continue
            ok = j.get("ok", True)
            val = compare(log, f"{key} vs baseline")
            print(f"| {label} | {w[0]:.1f} | {w[1]} | {w[2]:.2f} | {j.get('objects', '–')} | {val if ok else 'failed: ' + str(j.get('error'))[:80]} |")


if __name__ == "__main__":
    main()
