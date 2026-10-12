#!/usr/bin/env python3
"""nsys sqlite analysis for the T16-142fca profile.
  nsys_an.py decode FILE.sqlite      per device: busy (union), category kernel time per window, window spans, gaps
  nsys_an.py prefill FILE.sqlite     per device: the prompt read's chunks (busy segments), category time, gaps
"""
import sqlite3, sys, collections, statistics as st

CATS = [  # (category, name prefixes)  - first match wins
    ("MoE experts", ["native_gu_multi", "native_down_multi", "native_down_kernel", "swiglu_q8_1_entries",
                     "native_swiglu_quantize", "combine_k10", "quantize_q8_1_kernel", "mul_mat_q_tab",
                     "mul_mat_q_stream_k_fixup", "quantize_mmq_q8_1", "swiglu_kernel", "prep_group", "moe_combine",
                     "swiglu_pair"]),
    ("router+plan", ["route_multi", "route_kernel", "resident_plan", "sigmoid_scale_rows"]),
    ("dense GEMV/GEMM", ["native_mmvq_multi", "native_quantize_q8_1_kernel", "bf16_f32_mmvf_multi", "ampere_",
                         "Kernel2", "splitKreduce", "kernel"]),
    ("dequant q8_0", ["dequant_q8_0"]),
    ("GDN", ["gdn_"]),
    ("QSA attention", ["attn_chunk", "attn_merge", "block_scores", "block_topk", "append", "kv_append", "norm_rope",
                       "gate", "prompt_attn", "split_q", "rope_kernel", "gate_attn"]),
    ("HC read/write", ["gr_"]),
    ("head+sampling", ["native_mmvq_il", "native_q8_1_interleave", "sampler_", "argmax", "post"]),
    ("PLE/embed (st.0)", ["native_small_mmvq", "bf16_f32_mmvf_kernel", "weighted_rms_norm", "conv_residual",
                          "history_advance", "copy_indexed", "broadcast", "embed_rows", "add_streams"]),
    ("hand-off/flags", ["copy_from_mapped", "copy_i32_from_mapped", "wait_flag", "copy_i32_kernel"]),
]


def cat(name, dev):
    if name == "gate_kernel":
        return "PLE/embed (st.0)"
    for c, ps in CATS:
        for p in ps:
            if name == p or name.startswith(p):
                return c
    return "other"


def union(iv):
    iv = sorted(iv)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def load(path, maxdur=50e6):
    db = sqlite3.connect(path)
    names = dict(db.execute("select id, value from StringIds"))
    ks = [(d, s, e, names[n], g, c, sid) for d, s, e, n, g, c, sid in db.execute(
        "select deviceId, start, end, shortName, graphNodeId, correlationId, streamId from CUPTI_ACTIVITY_KIND_KERNEL")
          if e - s < maxdur]
    rt = [(s, e, names[n], c, tid) for s, e, n, c, tid in db.execute(
        "select start, end, nameId, correlationId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME")]
    mc = list(db.execute("select deviceId, start, end, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY"))
    return ks, rt, mc


def decode(path):
    ks, rt, mc = load(path)
    t0 = min(k[1] for k in ks)
    t1 = max(k[2] for k in ks)
    devs = sorted(set(k[0] for k in ks))
    launch = {c: (s, e) for s, e, n, c, tid in rt if n.startswith("cudaGraphLaunch")}
    print(f"== {path}: capture {(t1 - t0) / 1e6:.1f} ms")
    allcats = [c for c, _ in CATS] + ["other"]
    rows = {}
    for d in devs:
        kd = [k for k in ks if k[0] == d]
        # window = the graph launch that runs the head of the layer stack (the window graph): group by correlation id
        byc = collections.defaultdict(list)
        for k in kd:
            byc[k[5]].append(k)
        mx = max(len(v) for v in byc.values())
        wins = sorted([(min(x[1] for x in v), max(x[2] for x in v), len(v), c) for c, v in byc.items() if len(v) >= 0.6 * mx])
        coms = [(min(x[1] for x in v), max(x[2] for x in v)) for c, v in byc.items() if 50 <= len(v) < 0.6 * mx]
        # complete windows only
        nw = len(wins)
        span = sum(e - s for s, e, _, _ in wins) / max(nw, 1) / 1e6
        u = union([(k[1], k[2]) for k in kd])
        busy = sum(e - s for s, e in u)
        lo, hi = wins[0][0], wins[-1][1]
        u_in = union([(max(k[1], lo), min(k[2], hi)) for k in kd if k[2] > lo and k[1] < hi])
        busy_in = sum(e - s for s, e in u_in)
        # gaps inside windows vs between windows
        gin, gbt = 0, []
        for (s, e, _, _), (s2, _, _, _) in zip(wins, wins[1:]):
            gbt.append(s2 - e)
        for (s, e, _, _), (s2, _, _, _) in zip(wins, wins[1:]):
            uu = union([(k[1], k[2]) for k in kd if k[1] >= s and k[2] <= s2])
            gin += (s2 - s) - sum(b - a for a, b in uu)
        lat = [wins[i][0] - launch[wins[i][3]][0] for i in range(nw) if wins[i][3] in launch]
        lcall = [launch[w[3]][1] - launch[w[3]][0] for w in wins if w[3] in launch]
        per = (hi - lo) / max(nw - 1, 1) / 1e6
        cs = collections.Counter()
        for k in kd:
            if lo <= k[1] <= hi:
                cs[cat(k[3], d)] += k[2] - k[1]
        rows[d] = cs
        print(f"GPU{d}: {nw} windows, period {per:.2f} ms, window span {span:.2f} ms, busy {busy / (t1 - t0) * 100:.1f} % of capture "
              f"({busy_in / (hi - lo) * 100:.1f} % between first and last window); idle per period (window start to next start) {gin / (nw - 1) / 1e6:.2f} ms; "
              f"window end to next window start {st.mean(gbt) / 1e6:.2f} ms (median {st.median(gbt) / 1e6:.2f}); launch->first kernel "
              f"median {st.median(lat) / 1e6:.2f} ms; cudaGraphLaunch call median {st.median(lcall) / 1e3:.0f} us; "
              f"second (commit) graph per window: {len(coms)}, span {st.mean([e - s for s, e in coms]) / 1e6 if coms else 0:.2f} ms")
        rows[d]["_nw"] = nw
    print(f"{'category (ms/window)':22s}" + "".join(f"{'GPU' + str(d):>8s}" for d in devs))
    for c in allcats:
        if any(rows[d][c] for d in devs):
            print(f"{c:22s}" + "".join(f"{rows[d][c] / rows[d]['_nw'] / 1e6:8.3f}" for d in devs))
    print(f"{'sum':22s}" + "".join(f"{sum(v for k, v in rows[d].items() if k != '_nw') / rows[d]['_nw'] / 1e6:8.3f}" for d in devs))
    # host: graph launches and syncs per thread
    by = collections.defaultdict(lambda: collections.Counter())
    for s, e, n, c, tid in rt:
        by[n.split("_v")[0]]["n"] += 1
        by[n.split("_v")[0]]["t"] += e - s
    tw = sum(rows[d]["_nw"] for d in devs)
    print("host CUDA API (all threads), per stage-window: " + ", ".join(
        f"{n} {v['n'] / tw:.2f}x {v['t'] / tw / 1e3:.0f} us" for n, v in sorted(by.items(), key=lambda x: -x[1]["t"])[:6]))
    if mc:
        print("memcpy: " + ", ".join(f"GPU{d} kind{k}: {n} x, {b / 1e6:.1f} MB, {t / 1e6:.2f} ms" for d, k, n, b, t in
                                     sqlite3.connect(path).execute(
                                         "select deviceId, copyKind, count(*), sum(bytes), sum(end-start) from CUPTI_ACTIVITY_KIND_MEMCPY group by 1,2")))


def prefill(path, gap_ms=1.0):
    ks, rt, mc = load(path)
    devs = sorted(set(k[0] for k in ks))
    # the read: from the first prompt MMQ on GPU0 to the last one on the last GPU (+ its chunk tail)
    mm = [k for k in ks if k[3] == "mul_mat_q_tab"]
    a = min(k[1] for k in mm if k[0] == devs[0]) - 3e6
    b = max(k[2] for k in mm if k[0] == devs[-1]) + 3e6
    print(f"== {path}: read window {(b - a) / 1e6:.1f} ms (first..last prompt MMQ +-3 ms)")
    allcats = [c for c, _ in CATS] + ["other"]
    for d in devs:
        # prefill = non-graph kernels (decode windows and the last stage's draft layer run as graphs)
        kd = [k for k in ks if k[0] == d and k[4] is None and a <= k[1] <= b and not k[3].startswith("sampler_")]
        kg = [k for k in ks if k[0] == d and k[4] is not None and a <= k[1] <= b]
        u = union([(k[1], k[2]) for k in kd])
        segs = []
        for s, e in u:
            if segs and s - segs[-1][1] < gap_ms * 1e6:
                segs[-1][1] = e
                segs[-1][2] += e - s
            else:
                segs.append([s, e, e - s])
        segs = [x for x in segs if x[1] - x[0] > 2e6]
        ug = union([(k[1], k[2]) for k in kg])
        gbusy = sum(e - s for s, e in ug)
        print(f"GPU{d}: prompt segments (gaps < {gap_ms} ms merged): " + "; ".join(
            f"{(s - a) / 1e6:6.1f}-{(e - a) / 1e6:6.1f} ({(e - s) / 1e6:5.1f} ms, busy {bz / (e - s) * 100:4.1f} %)" for s, e, bz in segs)
              + f" | graph kernels in the read (decode windows / draft layer): {len(kg)}, {gbusy / 1e6:.1f} ms busy")
        for i, (s, e, bz) in enumerate(segs):
            cs = collections.Counter()
            for k in kd:
                if s <= k[1] <= e:
                    cs[cat(k[3], d)] += k[2] - k[1]
            print(f"   seg{i}: " + ", ".join(f"{c} {cs[c] / 1e6:.1f}" for c in allcats if cs[c] > 0.05e6)
                  + f" | gaps {((e - s) - bz) / 1e6:.1f} ms")
    # host<->device hand-off copies in the read
    db = sqlite3.connect(path)
    q = db.execute("select deviceId, copyKind, count(*), sum(bytes), sum(end-start) from CUPTI_ACTIVITY_KIND_MEMCPY "
                   "where start between ? and ? group by 1,2", (a, b)).fetchall()
    print("memcpy in the read: " + ", ".join(f"GPU{d} kind{k} {n}x {by / 1e6:.1f} MB {t / 1e6:.2f} ms" for d, k, n, by, t in q))
    sy = db.execute("select count(*), sum(end-start) from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION where start between ? and ?", (a, b)).fetchone()
    print(f"sync calls in the read: {sy[0]}, {(sy[1] or 0) / 1e6:.1f} ms")


if __name__ == "__main__":
    {"decode": decode, "prefill": prefill}[sys.argv[1]](sys.argv[2])
