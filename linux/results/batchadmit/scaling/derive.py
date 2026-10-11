#!/usr/bin/env python3
"""Derived bound for batch admission from the plain arm (medians, phase dec15 and idle).
Stall of one solo admission (dec15) = drain + prompt(read+ckpt+rest) + first window + slot copy (send->BADM) + refill 24 ms
(refill after BADM from results/fixedcost/final, not visible here).  Combined admission of k prompts (upper bound):
one drain, one read of the concatenated tokens (measured 'concat' condition), k checkpoints, one merged first window
over k rows (+3 ms per extra row, #48: 2.4-4), one refill."""
import sys, statistics as st
sys.path.insert(0, sys.argv[1])
from analyze import parse
REFILL, ROW = 24.0, 3.0
rows = parse(sys.argv[1] + "/plain")
m = lambda key, f: st.median([x[f] for x in rows[key]])
for ph in ("idle", "dec15"):
    print(f"== {ph}")
    print(f"{'k x N':8} {'solo read':>9} {'k*solo':>7} {'1 read':>7} {'(1xkN)':>7} {'read save':>9} {'%':>4} | "
          f"{'solo stall':>10} {'k*stall':>8} {'comb stall':>10} {'save':>6} {'%':>4} {'per adm':>7}")
    for n in (75, 150, 600, 1200):
        s = (ph, "single", 1, n)
        sr = m(s, "read")
        stall = m(s, "badm") + (REFILL if ph == "dec15" else 0)
        for k in (2, 3, 4):
            c = (ph, "concat", k, n)
            if c not in rows:
                continue
            cr = m(c, "read")
            sk = (ph, "single", 1, k * n)
            alt = m(sk, "read") if sk in rows else float("nan")
            comb = (m(c, "drain") + cr + k * m(c, "ckpt") + m(c, "first") + (k - 1) * ROW + m(c, "copy") * k
                    + (REFILL if ph == "dec15" else 0))
            print(f"{k}x{n:<6} {sr:9.1f} {k*sr:7.1f} {cr:7.1f} {alt:7.1f} {k*sr-cr:9.1f} {100*(k*sr-cr)/(k*sr):4.0f} | "
                  f"{stall:10.1f} {k*stall:8.1f} {comb:10.1f} {k*stall-comb:6.1f} {100*(k*stall-comb)/(k*stall):4.0f} "
                  f"{(k*stall-comb)/k:7.1f}")
