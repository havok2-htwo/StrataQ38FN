#!/usr/bin/env python3
"""Compare two batch_test16/32 dumps ({"solo": [...], "batch": [...]}): per slot, are the token lists equal; first diff position."""
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
for key in ("solo", "batch"):
    xa, xb = a.get(key, []), b.get(key, [])
    same = 0; diffs = []
    for i, (u, v) in enumerate(zip(xa, xb)):
        if u == v: same += 1
        else:
            k = next((j for j, (p, q) in enumerate(zip(u, v)) if p != q), min(len(u), len(v)))
            diffs.append(f"slot {i}: first diff at token {k} (len {len(u)} vs {len(v)})")
    print(f"{key}: {same}/{max(len(xa), len(xb))} identical" + ("" if len(xa) == len(xb) else f" (count {len(xa)} vs {len(xb)})"))
    for d in diffs: print("   ", d)
