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


def times(prefix: str) -> dict[str, tuple[float, str, float]]:
    """name -> (wall from /usr/bin/time, cpu, maxrss GB) from the per-run stderr files of a suite."""
    out = {}
    for f in OUT.glob(f"{prefix}_*.stderr"):
        m = re.search(r"^time: wall=([\d.]+) cpu=(\d+)% maxrss=(\d+)KB", f.read_text(), re.M)
        if m:
            out[f.name[len(prefix) + 1:-len(".stderr")]] = (float(m.group(1)), m.group(2) + " %", int(m.group(3)) / 2**20)
    return out


def compare(log: str, key: str) -> str:
    """Level-1 and last-level differences from a compare block titled '== compare <key> ...'."""
    blocks = list(re.finditer(rf"^== compare {re.escape(key)} .*?\n(.*?)(?=^== |\Z)", log, re.M | re.S))
    if not blocks:
        return "n/a"
    m = blocks[-1]  # a comparison rerun after a fix appends a new block; the last one counts
    rows = []
    for line in m.group(1).splitlines():
        mm = re.match(r"\s*(\S+)\s+\(([\d, ]+)\)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)(?:\s+(\S+))?", line)
        if mm and mm.group(3) != "no":
            width = int(mm.group(2).split(",")[-1])
            # columns: max|diff|, frac diff, [frac>0.5], dtype; the older format has no frac>0.5
            f5 = mm.group(6) if mm.group(7) else None
            rows.append((width, mm.group(4), mm.group(5), f5))
    rows.sort(key=lambda r: -r[0])   # largest level first, whatever the tool's group naming
    if len(rows) < 2:
        return "no matching level"
    def cell(r):
        beyond = f" ({100*float(r[3]):.1f} % beyond rounding)" if r[3] is not None else ""
        return f"{float(r[1]):.3g} / {100*float(r[2]):.1f} %{beyond}"
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
    t = times("tools_synth")
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
            cpu, rss = (("container", j.get("peak_rss_mb", 0) / 1024) if key == "gdal" else (w[1], w[2]))
            print(f"| {desc} | {label} | {w[0]:.1f} | {cpu} | {rss:.2f} | {j.get('objects', '–')} | {val if ok else 'failed: ' + str(j.get('error'))[:60]} |")
    for suite, title, tools in (("win32k", "WSF-3D Italy, 32768² window", [("ours_w8", "terrazarr, 8 processes"), ("ours_t8", "terrazarr, 8 threads"), ("eopf", "eopf-geozarr 0.11.0"), ("topozarr", "topozarr 0.1.9"), ("gdal", "GDAL 3.13.3")]),
                                ("full", "WSF-3D Italy, full", [("ours_w8", "terrazarr, 8 processes"), ("eopf", "eopf-geozarr 0.11.0"), ("topozarr", "topozarr 0.1.9"), ("gdal", "GDAL 3.13.3")])):
        t = times(f"tools_{suite}")
        print(f"\n### {title}\n")
        print("| writer | wall [s] | CPU | peak RSS [GB] | objects | overview values vs the reference (max diff / pixels differing, level 1; last level) |")
        print("|---|---:|---:|---:|---:|---|")
        for key, label in tools:
            j = load(f"tools_{suite}_{key}"); w = t.get(key)
            if j is None or w is None:
                print(f"| {label} | – | – | – | – | not run |"); continue
            ok = j.get("ok", True)
            val = compare(log, f"{key} vs baseline")
            cpu, rss = (("container", j.get("peak_rss_mb", 0) / 1024) if key == "gdal" else (w[1], w[2]))
            print(f"| {label} | {w[0]:.1f} | {cpu} | {rss:.2f} | {j.get('objects', '–')} | {val if ok else 'failed: ' + str(j.get('error'))[:80]} |")


if __name__ == "__main__":
    main()
