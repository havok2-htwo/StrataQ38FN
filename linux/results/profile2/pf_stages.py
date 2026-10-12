#!/usr/bin/env python3
"""Per-stage, per-chunk prefill phases (STRATA_PREFILL_TIMING) and the chunk pipeline, averaged over reps.
Stage attribution by the print order of the stage threads (verified against the nsys trace):
  1 chunk  : S0 S1 S2 S3
  3 chunks : S1c0 S2c0 S1c1 S0(all) S3c0 S2c1 S1c2 S3c1 S2c2 S3c2
start = print time - wall; gpu_end = start + GPU timeline (S1-S3: one chunk per run); times relative to the read start."""
import sys, re, collections, statistics as st
ORDER3 = ["S1c0", "S2c0", "S1c1", "S0", "S3c0", "S2c1", "S1c2", "S3c1", "S2c2", "S3c2"]
ORDER1 = ["S0", "S1", "S2", "S3"]
path = sys.argv[1]
reads = []; cur = None
for l in open(path):
    f = l.rstrip("\n").split(" ", 2); t = float(f[0]); msg = f[2] if len(f) > 2 else ""
    if f[1] == "M" and msg.startswith("ADMIT slot"):
        tag = re.search(r"tag=(\S+)", msg).group(1)
        cur = None if tag.startswith(("warmup", "fill")) else {"tag": tag, "tl": [], "host": [], "c0": []}
        if cur: reads.append(cur)
    elif cur is None: continue
    elif "prompt chunk" in msg and " of 1193" in msg or "prompt chunk 0 of 143" in msg and not cur["c0"]:
        cur["c0"].append(t)
    elif msg.startswith("strata prefill timing: host:"):
        cur["host"].append(dict((k, float(v)) for k, v in re.findall(r"(chunk setup|waiting for each chunk|after each chunk|PLE)[^\d]*?([\d.]+) ms", msg)))
    elif msg.startswith("strata prefill timing:"):
        m = re.match(r"strata prefill timing: (\d+) tokens, GPU timeline (\d+) ms, wall (\d+) ms, host staging \d+ ms:(.*)", msg)
        ph = {k.strip(): float(pc) * float(m.group(2)) / 100 for k, v, pc in re.findall(r" ([a-z+/ ]+?) (\d+) \(([\d.]+)%\)", m.group(4))}
        cur["tl"].append({"t": t, "n": int(m.group(1)), "gpu": float(m.group(2)), "wall": float(m.group(3)), "ph": ph})
    elif "tokens (batched) in" in msg:
        cur["read"] = float(re.search(r"in ([\d.]+) ms", msg).group(1))
        cur["rend"] = t
groups = collections.defaultdict(list)
for r in reads:
    k = ":".join(r["tag"].split(":")[:2]); groups[k].append(r)
PH = ["embed+steps", "hc read", "gdn", "gdn conv+gates", "gdn recurrence", "gdn out proj", "qsa proj", "qsa attn",
      "router+shared", "host grouping", "gather", "wait copy", "dequant", "gemm gate/up", "gemm down", "combine", "ple"]
for k, rs in groups.items():
    order = ORDER3 if len(rs[0]["tl"]) == 10 else ORDER1
    print(f"== {k}  reps {len(rs)}  read med {st.median(r['read'] for r in rs):.1f} ms")
    print(f"{'unit':5s} {'tok':>4s} {'start':>6s} {'gpuEnd':>6s} {'hand':>6s} {'wall':>5s} {'GPU':>5s} | " + " ".join(f"{p[:9]:>9s}" for p in PH))
    for i, name in enumerate(order):
        rows = []
        for r in rs:
            if len(r["tl"]) != len(order): continue
            x = r["tl"][i]; t0 = r["c0"][0] if r["c0"] else r["tl"][0]["t"] - r["tl"][0]["wall"] / 1000
            rows.append((x, t0))
        x0 = rows[0][0]
        med = lambda fn: st.median(fn(x, t0) for x, t0 in rows)
        start = med(lambda x, t0: (x["t"] - t0) * 1000 - x["wall"])
        gpu = med(lambda x, t0: x["gpu"]); wall = med(lambda x, t0: x["wall"])
        print(f"{name:5s} {x0['n']:4d} {start:6.1f} {start+gpu:6.1f} {start+wall:6.1f} {wall:5.0f} {gpu:5.0f} | " +
              " ".join(f"{med(lambda x, t0: x['ph'].get(p, 0.0)):9.1f}" for p in PH))
