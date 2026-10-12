#!/usr/bin/env python3
"""bitwise comparison of two prefill residual dumps: .rall (STRATA_PREFILL_DUMP_R_ALL, BF16 [n][D], every position)
and .r64 (STRATA_PREFILL_DUMP_R, fp32 every 64th position of the last prompt); plus the generated tokens in events.log"""
import sys, os, re
import numpy as np
a, b = sys.argv[1], sys.argv[2]   # prefixes, e.g. d_ref d_gs
D = int(sys.argv[3]) if len(sys.argv) > 3 else 0
for ext in ("rall", "r64"):
    pa, pb = f"{a}.{ext}", f"{b}.{ext}"
    da, db = open(pa, "rb").read(), open(pb, "rb").read()
    same = da == db
    print(f"{ext}: {len(da)} vs {len(db)} bytes, {'bitwise identical' if same else 'DIFFER'}")
    if not same and ext == "rall" and D and len(da) == len(db):
        xa = np.frombuffer(da, np.uint16).reshape(-1, D); xb = np.frombuffer(db, np.uint16).reshape(-1, D)
        rows = np.nonzero((xa != xb).any(1))[0]
        print(f"  {len(rows)} of {len(xa)} rows differ, first {rows[:10]}")
def toks(d):
    out = []
    for l in open(os.path.join(d, "events.log")):
        f = l.rstrip("\n").split(" ", 2)
        if f[1] == "O" and len(f) > 2 and f[2].startswith(("T ", "BT ")): out.append(f[2])
    return out
ta, tb = toks(a), toks(b)
print(f"generated token lines (T, BT): {len(ta)} vs {len(tb)}, {'identical' if ta == tb else 'DIFFER'}")
if ta != tb:
    for x, y in zip(ta, tb):
        if x != y: print("  first diff:", x[:150], "|", y[:150]); break
