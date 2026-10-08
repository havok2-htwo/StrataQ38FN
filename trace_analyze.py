"""Expert-routing trace analysis (from --dump-routing in --serve): is expert access predictable?

Record format (src/program/generate.cpp drive_pool_multi): int32 layer, int32 k, k int32 ids, k float weights.
In the serve path a verify window writes, per layer, one record per token; consecutive records with the same layer
form one block, and the blocks of a window run layer 0..47.

usage: python trace_analyze.py TRACE.bin
"""
import struct
import sys
from collections import Counter, defaultdict

import numpy as np

N_LAYER, N_EXP = 48, 512


def load(path):
    data = open(path, "rb").read()
    off, recs = 0, []
    while off + 8 <= len(data):
        layer, k = struct.unpack_from("<ii", data, off)
        off += 8
        ids = struct.unpack_from(f"<{k}i", data, off)
        off += 8 * k  # ids + weights
        recs.append((layer, ids))
    # blocks of equal layer -> windows -> tokens
    blocks, cur = [], None
    for layer, ids in recs:
        if cur is None or cur[0] != layer:
            cur = (layer, [])
            blocks.append(cur)
        cur[1].append(ids)
    tokens, win = [], {}
    for layer, rows in blocks:
        if layer == 0 and win:
            tokens += flush(win)
            win = {}
        win[layer] = rows
    tokens += flush(win)
    return recs, tokens


def flush(win):
    if len(win) != N_LAYER:
        return []
    n = min(len(r) for r in win.values())
    return [[set(win[l][t]) for l in range(N_LAYER)] for t in range(n)]


def main():
    recs, toks = load(sys.argv[1])
    print(f"records {len(recs)}, complete tokens {len(toks)}")
    if len(toks) < 200:
        print("too few tokens for statistics")
        return
    half = len(toks) // 2
    train, test = toks[:half], toks[half:]

    # 1. concentration per layer: experts needed for 50/90% of routings
    freq = [Counter() for _ in range(N_LAYER)]
    for t in toks:
        for l in range(N_LAYER):
            freq[l].update(t[l])
    need90, need50 = [], []
    for l in range(N_LAYER):
        c = np.array(sorted(freq[l].values(), reverse=True), dtype=float)
        cs = np.cumsum(c) / c.sum()
        need50.append(int(np.searchsorted(cs, 0.5)) + 1)
        need90.append(int(np.searchsorted(cs, 0.9)) + 1)
    print(f"\n[concentration] experts/layer for 50% of routings: median {int(np.median(need50))}, "
          f"for 90%: median {int(np.median(need90))} (of {N_EXP}); distinct used/layer median "
          f"{int(np.median([len(f) for f in freq]))}")

    # "cold" = outside the per-layer top 38% by train frequency (the VRAM tier holds 9332/24576 = 38% of pairs)
    trainfreq = [Counter() for _ in range(N_LAYER)]
    for t in train:
        for l in range(N_LAYER):
            trainfreq[l].update(t[l])
    hot = [set(e for e, _ in trainfreq[l].most_common(int(0.38 * N_EXP))) for l in range(N_LAYER)]
    cold_share = np.mean([len(t[l] - hot[l]) / 10 for t in test for l in range(N_LAYER)])
    print(f"[static top-38% per layer from train half] covers {100 * (1 - cold_share):.1f}% of test routings "
          f"(misses {100 * cold_share:.1f}%)")

    # 2. cross-layer co-occurrence: L -> L+1
    co = [defaultdict(Counter) for _ in range(N_LAYER - 1)]
    for t in train:
        for l in range(N_LAYER - 1):
            nxt = t[l + 1]
            for e in t[l]:
                co[l][e].update(nxt)

    # 3. temporal: same layer, previous token
    print("\n[prediction of layer L+1's experts]  recall = share of the true 10 experts in the predicted set")
    print(f"{'predictor':42s} {'@10':>6s} {'@20':>6s} {'@40':>6s} | cold-only @20 @40")
    for name in ("global frequency (L+1)", "co-occurrence L->L+1", "previous token same layer",
                 "co-occ + prev token"):
        rec = {10: [], 20: [], 40: []}
        rec_cold = {20: [], 40: []}
        prev = None
        for t in test:
            for l in range(N_LAYER - 1):
                truth = t[l + 1]
                score = Counter()
                if name.startswith("global"):
                    score = trainfreq[l + 1]
                if "co-occ" in name:
                    for e in t[l]:
                        score.update(co[l][e])
                if "prev" in name and prev is not None:
                    for e in prev[l + 1]:
                        score[e] += 10 ** 6
                ranked = [e for e, _ in score.most_common(40)]
                for n in rec:
                    rec[n].append(len(truth & set(ranked[:n])) / 10)
                cold = truth - hot[l + 1]
                if cold:
                    for n in rec_cold:
                        rec_cold[n].append(len(cold & set(ranked[:n])) / len(cold))
            prev = t
        print(f"{name:42s} {100*np.mean(rec[10]):5.1f}% {100*np.mean(rec[20]):5.1f}% {100*np.mean(rec[40]):5.1f}% | "
              f"{100*np.mean(rec_cold[20]):5.1f}% {100*np.mean(rec_cold[40]):5.1f}%")

    # temporal reuse across tokens
    reuse = [len(test[i][l] & test[i - 1][l]) / 10 for i in range(1, len(test)) for l in range(N_LAYER)]
    reuse4 = [len(test[i][l] & set().union(*[test[i - j][l] for j in range(1, 5)])) / 10
              for i in range(4, len(test)) for l in range(N_LAYER)]
    print(f"\n[temporal] experts reused from previous token (same layer): {100*np.mean(reuse):.1f}%; "
          f"from the previous 4 tokens: {100*np.mean(reuse4):.1f}%")

    # strongest pairs: "if 35 then 85"
    print("\n[strongest cross-layer pairs, P(e' in L+1 | e in L), support >= 50]")
    best = []
    for l in range(N_LAYER - 1):
        for e, cnt in co[l].items():
            n = trainfreq[l][e]
            if n >= 50:
                e2, c2 = cnt.most_common(1)[0]
                best.append((c2 / n, l, e, e2, n))
    best.sort(reverse=True)
    for p, l, e, e2, n in best[:10]:
        print(f"  layer {l} expert {e} -> layer {l+1} expert {e2}: {100*p:.0f}% (n={n})")
    ps = np.array([b[0] for b in best])
    print(f"  over all {len(ps)} (layer, expert) with support>=50: median top-1 conditional "
          f"{100*np.median(ps):.0f}%, share with >=50%: {100*np.mean(ps >= 0.5):.1f}%")


if __name__ == "__main__":
    main()
