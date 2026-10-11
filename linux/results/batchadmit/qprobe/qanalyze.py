#!/usr/bin/env python3
# per arm: waiting=N histogram at each BGEN, BADM -> next BGEN gap, whether the next BGEN was already queued at BADM
import re, sys, collections, statistics
arms = collections.OrderedDict(); cur = None
for l in open(sys.argv[1]):
    f = l.split(" ", 2)
    if len(f) < 3: continue
    t, d, rest = float(f[0]), f[1], f[2].strip()
    if d == "ARM": cur = rest; arms[cur] = []; continue
    if cur is None: continue
    arms[cur].append((t, d, rest))
for arm, ev in arms.items():
    w = collections.Counter(); gaps = []; badm_t = None; n = 0; back = 0; bgen_t = []
    for t, d, r in ev:
        if d == ">" and r.startswith("BGEN"):
            m = re.search(r"waiting=(\d+)", r); k = int(m.group(1)) if m else -1
            w[k] += 1; n += 1; bgen_t.append(t)
            if badm_t is not None:
                g = (t - badm_t) * 1000; gaps.append(g)
                if g < 0: back += 1
            badm_t = None
        elif d == "<" and r.startswith("BADM"):
            badm_t = t
    gs = sorted(gaps)
    def pc(p): return gs[min(len(gs) - 1, int(p * len(gs)))] if gs else float("nan")
    print(f"{arm}: {n} BGEN; waiting= hist " + " ".join(f"{k}:{w[k]}" for k in sorted(w)))
    print(f"   BADM->next BGEN gap ms (only when a request was queued behind): n={len(gs)} p10 {pc(.1):.1f} p50 {pc(.5):.1f} p90 {pc(.9):.1f} max {gs[-1] if gs else 0:.1f}")
