#!/usr/bin/env python3
"""fixedcost step 0: per-phase medians from the engine's STRATA_TRACE admission lines in a fc_driver run, the
BYIELD cost (drain + refill, or the BT gap after a dropped one), and a token comparison between runs.

  fc_analyze.py RUN_DIR [OTHER_RUN_DIR ...]   (the first run's tokens are compared with each other run's)
"""
import json, re, statistics, sys
from pathlib import Path

ADM = re.compile(r"admission of slot (\d+): drain ([\d.]+), prompt ([\d.]+) \(read ([\d.]+), windows ([\d.]+), (\d+) "
                 r"checkpoint\(s\) ([\d.]+) = sync ([\d.]+) \+ save ([\d.]+) \+ evict ([\d.]+), rest (-?[\d.]+)\), "
                 r"first window ([\d.]+), slot copy ([\d.]+), slot freed ([\d.]+), checkpoint copy ([\d.]+)")
KEYS = ["drain", "prompt", "read", "windows", "ckpts", "ckpt", "sync", "save", "evict", "rest", "first_window",
        "slot_copy", "slot_freed", "ckpt_copy"]


def load(run):
    ev = []
    for l in (Path(run) / "events.log").read_text(errors="replace").splitlines():
        p = l.split(" ", 2)
        if len(p) == 3:
            ev.append((float(p[0]), p[1], p[2]))
    return ev


def analyze(run):
    ev = load(run)
    bt = [t for t, s, r in ev if s == "O" and r.startswith("BT ")]
    import bisect
    adm, cur = [], None
    for i, (t, s, r) in enumerate(ev):
        if s == "M" and r.startswith("ADMIT "):
            m = dict(kv.split("=", 1) for kv in r.split()[1:])
            cur = {"cls": m["cls"], "by": m.get("byield", "none"), "n": int(m["n"])}
            adm.append(cur)
        elif cur is not None and s == "E":
            mm = ADM.search(r)
            if mm:
                for k, v in zip(KEYS, mm.groups()[1:]):
                    cur[k] = float(v)
            elif "first rows" in r and "after the admission" in r and "refill" not in cur:
                cur["refill"] = float(re.search(r"first rows ([\d.]+) ms", r).group(1))
            elif r.startswith("strata trace: drain for BYIELD"):
                cur["byield_drain"] = float(re.search(r"([\d.]+) ms$", r).group(1))
                cur["byield_inflight"] = int(re.search(r": (\d+) groups", r).group(1))
            elif "after the drain for BYIELD" in r:
                cur["byield_stall"] = float(re.search(r"first rows ([\d.]+) ms", r).group(1))
            elif r.startswith("strata trace: BYIELD dropped"):
                cur["byield_dropped_inflight"] = int(re.search(r"dropped, (\d+) groups", r).group(1))
        elif cur is not None and s == "M" and r.startswith("LATE BYIELD"):
            cur["t_late"] = t
    # BT gap after a late BYIELD (both arms): the longest gap between rows in the 300 ms after it was sent
    for a in adm:
        if "t_late" in a:
            k0 = bisect.bisect_left(bt, a["t_late"])
            k1 = bisect.bisect_left(bt, a["t_late"] + 0.3)
            ts = bt[max(k0 - 1, 0):k1 + 1]
            a["late_gap"] = round(1000 * max(b - c for b, c in zip(ts[1:], ts[:-1])), 1) if len(ts) > 1 else None
            # reference: the longest gap in the 300 ms before it
            j0 = bisect.bisect_left(bt, a["t_late"] - 0.3)
            ts0 = bt[j0:k0]
            n_after = bisect.bisect_left(bt, a["t_late"] + 0.5) - k0
            n_ref = bisect.bisect_left(bt, a["t_late"] + 2.0) - bisect.bisect_left(bt, a["t_late"] + 1.5)
            a["rows_lost"] = n_ref - n_after
            a["pre_gap"] = round(1000 * max(b - c for b, c in zip(ts0[1:], ts0[:-1])), 1) if len(ts0) > 1 else None
    return adm


def med(v):
    v = [x for x in v if x is not None]
    return f"{statistics.median(v):7.1f}" if v else "    n/a"


runs = sys.argv[1:]
res = {r: analyze(r) for r in runs}
for r, adm in res.items():
    meas = adm[16:]   # the 16 fills first (m0_driver), then the measured admissions
    print(f"== {r}: {len(meas)} measured admissions (+ {len(adm) - len(meas)} fills)")
    for cls in ("short", "long"):
        rs = [a for a in meas if a["cls"] == cls and "prompt" in a]
        if not rs:
            continue
        print(f"  {cls} (n={len(rs)}, ~{int(statistics.median([a['n'] for a in rs]))} tokens): " +
              ", ".join(f"{k} {med([a.get(k) for a in rs]).strip()}" for k in KEYS + ["refill"] if k != "ckpts"))
    for by in ("read", "late"):
        rs = [a for a in meas if a["by"] == by]
        if not rs:
            continue
        print(f"  BYIELD {by} (n={len(rs)}): drained {sum('byield_drain' in a for a in rs)}x, groups in flight at the drain "
              f"{[a.get('byield_inflight') for a in rs if 'byield_inflight' in a]}, drain {med([a.get('byield_drain') for a in rs])} ms, "
              f"drain+refill until rows {med([a.get('byield_stall') for a in rs])} ms; dropped without drain "
              f"{sum('byield_dropped_inflight' in a and 'byield_drain' not in a for a in rs)}x; "
              f"longest BT gap in the 300 ms after it {med([a.get('late_gap') for a in rs])} ms, rows in the 500 ms after it "
              f"minus rows in [+1.5, +2.0] s {med([a.get('rows_lost') for a in rs])}")
    fills = [a for a in adm[:16] if "ckpt_copy" in a]
    if fills:
        print(f"  fills (n={len(fills)}): ckpt_copy {med([a['ckpt_copy'] for a in fills])}, save {med([a.get('save') for a in fills])}")
def prompt_keys(run):
    # request number -> its prompt kind: the prompt (and the seed) is a function of (number, kind); spontaneous ends
    # (EOS) come at other times in another run, so the numbering diverges and only equal pairs are comparable
    k, uid = {}, 0
    for t, s, r in load(run):
        if s == "M" and r.startswith("ADMIT "):
            uid += 1
            k[str(uid)] = r.split("cls=")[1].split()[0].replace("short-refill", "short")
    return k


if len(runs) > 1:
    t0 = json.loads((Path(runs[0]) / "tokens.json").read_text())
    k0 = prompt_keys(runs[0])
    for r in runs[1:]:
        t1 = json.loads((Path(r) / "tokens.json").read_text())
        k1 = prompt_keys(r)
        same = diff = 0
        n_cmp = 0
        for u in t0:
            if u not in t1 or k0.get(u) != k1.get(u) or k0.get(u) == "fill" and False:
                continue
            a, b = t0[u], t1[u]
            m = min(len(a), len(b))
            n_cmp += m
            if a[:m] == b[:m]:
                same += 1
            else:
                k = next(j for j in range(m) if a[j] != b[j])
                diff += 1
                print(f"  request {u}: first diff at token {k} of {m}")
        print(f"tokens {runs[0]} vs {r}: {same} requests identical over their common prefix, {diff} differ "
              f"({n_cmp} tokens compared)")
