#!/usr/bin/env python3
"""Summarize scale_driver runs: per condition and phase the median read / admission split, and (timing arm) the
per-stage GPU phase split.  analyze.py DIR [DIR...]"""
import re, statistics as st, sys
from collections import defaultdict
from pathlib import Path

EXPERT = {"router+shared", "host grouping", "gather", "wait copy", "dequant", "gemm gate/up", "gemm down", "combine"}
ATTN = {"qsa proj", "qsa indexer", "qsa select", "qsa attn", "kv stage"}
GDN = {"gdn", "gdn conv+gates", "gdn recurrence", "gdn out proj"}
RX_ADM = re.compile(r"admission of slot \d+: drain ([\d.]+), prompt ([\d.]+) \(read ([\d.]+), windows ([\d.]+), "
                    r"(\d+) checkpoint\(s\) ([\d.]+).*?rest ([\d.]+)\), first window ([\d.]+), slot copy ([\d.]+)")
RX_READ = re.compile(r"strata trace: read (\d+) tokens \(batched\) in ([\d.]+) ms")
RX_PT = re.compile(r"strata prefill timing: (\d+) tokens, GPU timeline (\d+) ms, wall (\d+) ms, host staging \d+ ms:(.*)")
RX_PH = re.compile(r" ([a-z+/ ]+?) (\d+) \(([\d.]+)%\)")


def med(v):
    return st.median(v) if v else float("nan")


def parse(d):
    rows = defaultdict(list)   # (phase, kind, k, n) -> list of dict
    cur = None
    for line in open(Path(d) / "events.log", errors="replace"):
        t, src, rest = line.rstrip("\n").split(" ", 2)
        if src == "M" and rest.startswith("ADMIT slot="):
            m = re.search(r"tag=(\S+) n=(\d+) active=(\d+)", rest)
            tag = m.group(1)
            if ":" not in tag:
                cur = None
                continue
            ph, kind, kn, _ = tag.split(":")
            k, n = map(int, kn.split("x"))
            cur = {"key": (ph, kind, k, n), "N": int(m.group(2)), "active": int(m.group(3)), "stages": [],
                   "t_send": float(t)}
        elif cur is None:
            continue
        elif src == "E" and (m := RX_READ.search(rest)):
            cur["n_read"] = int(m.group(1))
            cur["read"] = float(m.group(2))
        elif src == "E" and (m := RX_PT.search(rest)):
            phs = {a.strip(): float(b) for a, b, _ in RX_PH.findall(" " + m.group(4))}
            cur["stages"].append({"gpu": float(m.group(2)), "wall": float(m.group(3)), "ph": phs})
        elif src == "E" and (m := RX_ADM.search(rest)):
            cur.update(drain=float(m.group(1)), prompt=float(m.group(2)), ckpt=float(m.group(6)),
                       first=float(m.group(8)), copy=float(m.group(9)))
        elif src == "M" and rest.startswith("ADMITTED"):
            cur["badm"] = (float(t) - cur["t_send"]) * 1000
            rows[cur["key"]].append(cur)
            cur = None
    return rows


def main():
    for d in sys.argv[1:]:
        rows = parse(d)
        print(f"== {d}")
        print(f"{'phase':6} {'cond':10} {'N':>5} {'nread':>5} {'reps':>4} {'read':>7} {'min':>7} {'max':>7} "
              f"{'drain':>6} {'first':>6} {'ckpt':>5} {'send>BADM':>9} {'ms/100tok':>9} | per-stage GPU ms: "
              f"total expert attn gdn other (sum of 4 stages)")
        for key in sorted(rows, key=lambda k: (k[0] != "idle", k[3] * k[2], k[1])):
            ph, kind, k, n = key
            r = rows[key]
            rd = [x["read"] for x in r if "read" in x]
            line = (f"{ph:6} {(f'{k}x{n}' if kind == 'concat' else str(n)):10} {r[0]['N']:5} "
                    f"{r[0].get('n_read', 0):5} {len(r):4} {med(rd):7.1f} {min(rd):7.1f} {max(rd):7.1f} "
                    f"{med([x.get('drain', 0) for x in r]):6.1f} {med([x.get('first', 0) for x in r]):6.1f} "
                    f"{med([x.get('ckpt', 0) for x in r]):5.1f} {med([x['badm'] for x in r]):9.1f} "
                    f"{100 * med(rd) / r[0].get('n_read', 1):9.1f}")
            if r[0]["stages"]:
                agg = defaultdict(list)
                for x in r:
                    tot = defaultdict(float)
                    for s in x["stages"]:
                        tot["total"] += s["gpu"]
                        for name, v in s["ph"].items():
                            cls = "expert" if name in EXPERT else "attn" if name in ATTN else "gdn" if name in GDN else "other"
                            tot[cls] += v
                            tot["ph:" + name] += v
                    for kk, v in tot.items():
                        agg[kk].append(v)
                line += " | " + " ".join(f"{med(agg[c]):5.0f}" for c in ("total", "expert", "attn", "gdn", "other"))
                line += "  [" + ", ".join(f"{kk[3:]} {med(v):.0f}" for kk, v in sorted(agg.items())
                                         if kk.startswith("ph:") and med(v) >= 1) + "]"
            print(line)


if __name__ == "__main__":
    main()
