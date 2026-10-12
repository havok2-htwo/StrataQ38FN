#!/usr/bin/env python3
"""STRATA_VERIFY_PROFILE batch lines -> table (ms per stage-window), GDN and QSA layers summed and grouped."""
import sys, re
GROUP = {"q8+qkv/q-idx gemv": "GDN in-proj (qkv,z gemv)", "conv": "GDN conv", "ab": "GDN gates a/b", "z": "GDN z gemv",
         "rec": "GDN recurrence", "out-proj": "attn/GDN out-proj",
         "q8+kv-idx": "QSA proj+attention", "k/v+norm-rope": "QSA proj+attention", "kv+idx append": "QSA proj+attention",
         "q+q-idx": "QSA proj+attention", "scores+topk": "QSA proj+attention", "kv-resolve": "QSA proj+attention",
         "attention": "QSA proj+attention", "gate": "QSA proj+attention",
         "hc-read1+router": "HC read (mid) + router", "shared+quant": "shared expert + quant", "waitA": "wait (plan)",
         "VRAM hits": "routed experts (VRAM)", "copy+combine": "expert combine", "(gap)": "gap between layers",
         "head": "head (LM head + sampling prep)", "hc0 norm": "HC read0 norm (+window start wait)", "hc0 down": "HC read0 down/up",
         "hc0 up": "HC read0 down/up"}
ORDER = ["HC read0 norm (+window start wait)", "HC read0 down/up", "GDN in-proj (qkv,z gemv)", "GDN z gemv", "GDN conv",
         "GDN gates a/b", "GDN recurrence", "QSA proj+attention", "attn/GDN out-proj", "HC read (mid) + router",
         "shared expert + quant", "routed experts (VRAM)", "expert combine", "wait (plan)", "gap between layers",
         "head (LM head + sampling prep)"]
cols = []
for path in sys.argv[1:]:
    fc = []
    for l in open(path):
        m = re.search(r"strata batch GPU stages, stage (\d) \(ms/window\):(.*)", l)
        if not m: continue
        d = {}
        body = m.group(2).split("| total")[0]
        for part in body.split("|"):
            part = re.sub(r"^\s*(GDN|QSA) layers:", "", part)
            for k, v in re.findall(r"\s*([a-zA-Z0-9+/()\- ]+?) (\d+\.\d+)", part):
                k = k.strip(); g = GROUP.get(k, k)
                d[g] = d.get(g, 0) + float(v)
        tot = float(re.search(r"total ([\d.]+) ms/window over (\d+)", l).group(1))
        fc.append((f"{path.split('/')[0][:5]} S{m.group(1)}", d, tot))
    cols += fc[-4:]   # the decode stretch (the earlier reports are the warm-up's single-row windows)
print(f"{'phase (ms per stage-window)':38s}" + "".join(f"{c[0]:>13s}" for c in cols))
for k in ORDER:
    print(f"{k:38s}" + "".join(f"{c[1].get(k, 0):13.2f}" for c in cols))
print(f"{'total stamped':38s}" + "".join(f"{c[2]:13.2f}" for c in cols))
