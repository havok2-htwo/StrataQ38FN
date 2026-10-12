#!/usr/bin/env python3
"""per admission: read time, drain, send->BADM; with --lines the prefill timing lines in order (relative ms)"""
import sys, re, statistics as st, collections
path = sys.argv[1]; show = "--lines" in sys.argv
cur = None; res = collections.defaultdict(list); lines = []
for l in open(path):
    f = l.rstrip("\n").split(" ", 2)
    t = float(f[0]); msg = f[2] if len(f) > 2 else ""
    if f[1] == "M" and msg.startswith("ADMIT slot"):
        tag = re.search(r"tag=(\S+)", msg).group(1)
        cur = {"tag": tag, "t0": t, "drain": 0.0, "read": None, "lines": []}
    elif cur is None: continue
    elif "drain for BGEN" in msg: cur["drain"] = float(re.search(r"([\d.]+) ms", msg).group(1))
    elif "tokens (batched) in" in msg: cur["read"] = float(re.search(r"in ([\d.]+) ms", msg).group(1)); cur["rt"] = t
    elif "prompt chunk" in msg or "prefill timing" in msg: cur["lines"].append((t, msg))
    elif f[1] == "M" and msg.startswith("ADMITTED"):
        cur["adm"] = (t - cur["t0"]) * 1000
        if not cur["tag"].startswith(("warmup", "fill")):
            k = ":".join(cur["tag"].split(":")[:2]); res[k].append(cur)
        cur = None
for k, v in res.items():
    r = [x["read"] for x in v if x["read"]]
    print(f"{k:12s} n={len(v)} read med {st.median(r):.1f} ms (min {min(r):.1f} max {max(r):.1f}); drain med "
          f"{st.median(x['drain'] for x in v):.1f}; send->BADM med {st.median(x['adm'] for x in v):.1f} ms")
    if show:
        x = v[len(v) // 2]
        print("  example", x["tag"])
        t0 = x["lines"][0][0] if x["lines"] else 0
        for t, m in x["lines"]:
            m = re.sub(r" \(\d+\.\d%\)", "", m)
            print(f"   {1000*(t-t0):7.1f} {m[:330]}")
