"""Count the tiles of a (Big)TIFF and how many are unwritten (byte count 0, i.e. sparse).

    python bench/tiff_tiles.py file.tif        # first IFD (full resolution) and every overview
"""
from __future__ import annotations

import struct
import sys

TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 16: 8, 17: 8, 18: 8}
TYPE_FMT = {3: "H", 4: "I", 16: "Q"}


def ifds(f):
    f.seek(0)
    bo = "<" if f.read(2) == b"II" else ">"
    magic = struct.unpack(bo + "H", f.read(2))[0]
    big = magic == 43
    if big:
        f.seek(8); off = struct.unpack(bo + "Q", f.read(8))[0]
    else:
        off = struct.unpack(bo + "I", f.read(4))[0]
    while off:
        f.seek(off)
        n = struct.unpack(bo + ("Q" if big else "H"), f.read(8 if big else 2))[0]
        tags = {}
        for _ in range(n):
            raw = f.read(20 if big else 12)
            tag, typ = struct.unpack(bo + "HH", raw[:4])
            cnt = struct.unpack(bo + ("Q" if big else "I"), raw[4:12] if big else raw[4:8])[0]
            val = raw[12:] if big else raw[8:]
            size = TYPE_SIZE.get(typ, 1) * cnt
            if size > len(val):
                ptr = struct.unpack(bo + ("Q" if big else "I"), val)[0]
                here = f.tell(); f.seek(ptr); data = f.read(size); f.seek(here)
            else:
                data = val[:size]
            tags[tag] = (typ, cnt, data)
        off = struct.unpack(bo + ("Q" if big else "I"), f.read(8 if big else 4))[0]
        yield bo, tags


def main(path: str) -> None:
    with open(path, "rb") as f:
        for i, (bo, tags) in enumerate(ifds(f)):
            if 325 not in tags:
                continue
            typ, cnt, data = tags[325]
            counts = struct.unpack(bo + TYPE_FMT[typ] * cnt, data)
            w = struct.unpack(bo + TYPE_FMT[tags[256][0]], tags[256][2][: TYPE_SIZE[tags[256][0]]])[0]
            h = struct.unpack(bo + TYPE_FMT[tags[257][0]], tags[257][2][: TYPE_SIZE[tags[257][0]]])[0]
            empty = sum(1 for c in counts if c == 0)
            print(f"ifd {i}: {w}x{h}, tiles {cnt}, empty {empty} ({100 * empty / cnt:.1f} %), stored bytes {sum(counts) / 1e6:.1f} MB")


if __name__ == "__main__":
    main(sys.argv[1])
