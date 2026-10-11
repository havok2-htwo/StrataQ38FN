#!/usr/bin/env python3
"""cmp_dump.py A.json B.json: batch_test16 dumps, solo and batch tokens per slot equal?"""
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
for k in ("solo", "batch"):
    x, y = a.get(k) or [], b.get(k) or []
    same = sum(1 for i in range(min(len(x), len(y))) if x[i] == y[i])
    print(f"{k}: {same}/{max(len(x), len(y))} identical")
