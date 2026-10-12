#!/usr/bin/env python3
"""rows/s of the decode stretch from events.log (BT lines between the STRETCH marks), group-step intervals."""
import sys, statistics as st
for path in sys.argv[1:]:
    t0 = t1 = None; bts = []
    for l in open(path):
        f = l.split(" ", 3)
        t = float(f[0])
        if f[1] == "M" and f[2] == "STRETCH":
            if "END" in l: t1 = t
            else: t0 = t
        elif f[1] == "M" and "NSYS" in l:
            pass
        elif f[1] == "O" and f[2] == "BT" and t0 is not None and t1 is None:
            bts.append((t, int(f[3].split()[0])))
    a, b = t0 + 2, t1 - 0.5          # skip the ramp
    sel = [x for x in bts if a <= x[0] <= b]
    nslots = max(s for _, s in bts) + 1
    gs = 4 if nslots <= 16 else 8
    # group-step completions: BT bursts of one group
    steps = {}
    prev = None
    for t, s in sel:
        g = s // gs
        if prev is None or prev[1] != g or t - prev[0] > 0.0005:
            steps.setdefault(g, []).append(t)
        prev = (t, g)
    per = [d for g in steps for d in (y - x for x, y in zip(steps[g], steps[g][1:]))]
    nsteps = sum(len(v) for v in steps.values())
    print(f"{path}: slots {nslots}, {len(sel)} rows in {b-a:.2f} s = {len(sel)/(b-a):.1f} rows/s; "
          f"group-steps {nsteps/(b-a):.1f}/s; per-group step period median {1000*st.median(per):.2f} ms "
          f"(p10 {1000*sorted(per)[len(per)//10]:.2f}, p90 {1000*sorted(per)[9*len(per)//10]:.2f}); "
          f"=> one stage-window ~{1000*st.median(per)/4:.2f} ms")
