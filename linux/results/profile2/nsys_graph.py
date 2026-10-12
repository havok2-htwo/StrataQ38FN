#!/usr/bin/env python3
"""graph-level nsys trace (--cuda-graph-trace=graph) of the pipelined decode: per stage the window graph and the commit
graph, GPU busy, idle between windows, the host's launch timing, and the stage-to-stage hand-off latency."""
import sqlite3, sys, collections, statistics as st
path = sys.argv[1]
db = sqlite3.connect(path)
names = dict(db.execute("select id, value from StringIds"))
gr = list(db.execute("select deviceId, start, end, graphExecId, correlationId from CUPTI_ACTIVITY_KIND_GRAPH_TRACE order by start"))
rt = {c: (s, e, tid) for s, e, n, c, tid in db.execute("select start, end, nameId, correlationId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME")}
krn = list(db.execute("select deviceId, start, end from CUPTI_ACTIVITY_KIND_KERNEL"))
sync = list(db.execute("select r.start, r.end from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id = r.nameId where s.value like 'cudaStreamSynchronize%'"))
t0 = min(g[1] for g in gr); t1 = max(g[2] for g in gr)
devs = sorted(set(g[0] for g in gr))
# window graph = the longer of the two graph execs per device (by mean span)
win = {}
for d in devs:
    by = collections.defaultdict(list)
    for g in gr:
        if g[0] == d: by[g[3]].append(g)
    mspan = {k: st.mean(x[2] - x[1] for x in v) for k, v in by.items()}
    top = max(mspan.values())
    win[d] = (sorted(x for k, v in by.items() if mspan[k] > 0.4 * top for x in v),
              [x for k, v in by.items() if mspan[k] <= 0.4 * top for x in v])
print(f"== {path}: {(t1 - t0) / 1e6:.0f} ms")
print(f"{'GPU':4s} {'win':>4s} {'period':>7s} {'winSpan':>7s} {'commit':>7s} {'busy%':>6s} {'idle/per':>8s} {'launch':>7s} {'lat':>6s} {'handoff':>8s}")
for d in devs:
    W, C = win[d]
    W = [w for w in W if t0 + 20e6 < w[1] < t1 - 20e6]
    per = (W[-1][1] - W[0][1]) / (len(W) - 1) / 1e6
    lo, hi = W[0][1], W[-1][1]
    ivs = sorted([(g[1], g[2]) for g in gr if g[0] == d and lo <= g[1] < hi] + [(k[1], k[2]) for k in krn if k[0] == d and lo <= k[1] < hi])
    u = []
    for s, e in ivs:
        if u and s <= u[-1][1]: u[-1][1] = max(u[-1][1], e)
        else: u.append([s, e])
    busy = sum(min(e, hi) - s for s, e in u) / (hi - lo)
    lcall = [rt[w[4]][1] - rt[w[4]][0] for w in W if w[4] in rt]
    lat = [w[1] - rt[w[4]][1] for w in W if w[4] in rt]   # launch call returned -> GPU start
    # hand-off: previous stage's window end -> this window's launch call (host reaction), d > first
    ho = []
    if d != devs[0]:
        Wp = [w for w in win[d - 1][0]]
        ends = sorted(w[2] for w in Wp)
        import bisect
        for w in W:
            if w[4] not in rt: continue
            ls = rt[w[4]][0]
            i = bisect.bisect_right(ends, ls) - 1
            if i >= 0: ho.append(ls - ends[i])
    print(f"{d:4d} {len(W):4d} {per:7.2f} {st.mean(w[2] - w[1] for w in W) / 1e6:7.2f} {st.mean(c[2] - c[1] for c in C) / 1e6:7.2f} "
          f"{busy * 100:6.1f} {per * (1 - busy):8.2f} {st.median(lcall) / 1e3:6.0f}u {st.median(lat) / 1e3:5.0f}u "
          f"{(st.median(ho) / 1e3 if ho else float('nan')):7.0f}u")
tids = collections.Counter(); tt = collections.Counter()
for s, e, n, c, tid in db.execute("select start, end, nameId, correlationId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME"):
    tids[tid] += 1; tt[tid] += e - s
for tid in tids: print(f"host thread {tid % 1000000}: {tids[tid]} CUDA calls, {tt[tid] / 1e6:.1f} ms in calls ({tt[tid] / (t1 - t0) * 100:.1f} % of the capture)")
print(f"cudaStreamSynchronize (head sampling on the last stage): {len(sync)} calls, median {st.median(e - s for s, e in sync) / 1e3:.0f} us")
print("columns: period = window start to next start (ms); winSpan = window graph GPU span; commit = second graph span; busy% = GPU busy "
      "(graphs + kernels) between first and last window; idle/per = idle ms per period; launch = cudaGraphLaunch call (host us); "
      "lat = call return -> GPU start; handoff = previous stage window end -> this stage's launch call (host reaction)")
