#!/usr/bin/env python3
"""Split every admission of an m0_driver.py run into its phases (from the timestamped events.log).

  m0_analyze.py RUN_DIR [--json out.json]
"""
import json, re, statistics, sys
from pathlib import Path

run = Path(sys.argv[1])
ev = []
for l in (run / "events.log").read_text(errors="replace").splitlines():
    t, src, rest = l.split(" ", 2) if l.count(" ") >= 2 else (l.split(" ") + [""])[:3]
    ev.append((float(t), src, rest))
bt_t = [t for t, s, r in ev if s == "O" and r.startswith("BT ")]
import bisect


def rows_between(a, b):
    return bisect.bisect_left(bt_t, b) - bisect.bisect_left(bt_t, a)


# D_ref: decode-only stretches (skip 0.3 s after the stretch start: the pipeline ramp after BADM / BDONE)
stretch = {}
cur = None
for t, s, r in ev:
    if s == "M" and r.startswith("STRETCH decode-only"):
        cur = (t, int(r.split("active=")[1]))
    elif s == "M" and r.startswith("STRETCH end") and cur:
        a, k = cur[0] + 0.3, cur[1]
        n = rows_between(a, t)
        stretch.setdefault(k, []).append(n / (t - a))
        cur = None
D = {k: statistics.mean(v) for k, v in stretch.items()}
Dref = statistics.mean([D.get(15, 0), D.get(16, 0)]) if 15 in D and 16 in D else max(D.values())

adm = []
i = 0
while i < len(ev):
    t, s, r = ev[i]
    if s == "M" and r.startswith("ADMIT "):
        m = dict(kv.split("=") for kv in r.split()[1:])
        a = {"cls": m["cls"], "slot": int(m["slot"]), "n": int(m["n"]), "active_before": int(m["active"])}
        segs, j = [], i + 1
        while j < len(ev):
            tj, sj, rj = ev[j]
            if sj == "S" and rj.startswith("BGEN"):
                a["t_send"] = tj
            elif sj == "E" and rj.startswith("strata trace: request"):
                a["t_req"] = tj
            elif sj == "E" and rj.startswith("strata trace: prompt start"):
                a["t_pstart"] = tj
            elif sj == "E" and rj.startswith("strata trace: refill start"):
                a["t_refill0"] = tj
            elif sj == "E" and rj.startswith("strata trace: refilled"):
                a["refill_ms"] = a.get("refill_ms", 0) + float(re.search(r"in ([\d.]+) ms", rj).group(1))
            elif sj == "E" and rj.startswith("strata trace: lent"):
                a["lent"] = rj.split("lent ")[1].split(" slots")[0]
            elif sj == "E" and rj.startswith("strata trace: read "):
                mm = re.search(r"read (\d+) tokens \((\w+)\) in ([\d.]+) ms", rj)
                segs.append({"kind": mm.group(2), "tokens": int(mm.group(1)), "ms": float(mm.group(3)),
                             "t_end": tj, "t_start": tj - float(mm.group(3)) / 1000})
            elif sj == "E" and rj.startswith("strata trace: prompt done"):
                a["t_pdone"] = tj
            elif sj == "O" and rj.startswith("REUSED"):
                a["t_reused"] = tj
            elif sj == "E" and rj.startswith("strata trace: window") and "t_win" not in a:
                a["t_win"] = tj
            elif sj == "O" and rj.startswith("DONE"):
                a["t_done"] = tj
                a["first_window_ms_engine"] = float(rj.split()[4])
            elif sj == "E" and "takes" in rj and rj.startswith("strata batch: slot"):
                a["copy_ms"] = float(re.search(r"copied in ([\d.]+) ms", rj).group(1))
                a["t_copied"] = tj
            elif sj == "E" and rj.startswith("strata serve: prompt "):
                a["serve_line"] = rj
            elif sj == "O" and rj.startswith("BADM"):
                a["t_badm"] = tj
                break
            j += 1
        a["segs"] = segs
        # first BT after BADM
        k = bisect.bisect_right(bt_t, a.get("t_badm", 0))
        a["t_bt_after"] = bt_t[k] if k < len(bt_t) else None
        kb = bisect.bisect_left(bt_t, a.get("t_req", a.get("t_send", 0))) - 1
        a["t_bt_before"] = bt_t[kb] if kb >= 0 else None
        adm.append(a)
        i = j
    i += 1


def ms(x, y):
    return None if x is None or y is None else round((y - x) * 1000, 1)


rows = []
for a in adm:
    if "t_send" not in a or "t_badm" not in a:
        continue
    segs = a["segs"]
    batched = [s for s in segs if s["kind"] == "batched"]
    windows = [s for s in segs if s["kind"] == "windows"]
    # checkpoint: the gaps between consecutive read segments (checkpoint_at runs after a cut, before the next read)
    gaps = [ms(segs[q]["t_end"], segs[q + 1]["t_start"]) for q in range(len(segs) - 1)]
    tail_gap = ms(segs[-1]["t_end"], a.get("t_pdone")) if segs else None
    W = 2.0     # the window after BGEN, for the effective stall
    T = (a["t_badm"] - a["t_send"]) + W
    lost = T - rows_between(a["t_send"], a["t_send"] + T) / Dref
    r = {
        "cls": a["cls"], "slot": a["slot"], "n": a["n"], "active_before": a["active_before"], "lent": a.get("lent"),
        "drain_ms": ms(a["t_send"], a["t_bt_before"]) if a["t_bt_before"] and a["t_bt_before"] > a["t_send"] else 0.0,
        "send_to_request_ms": ms(a["t_send"], a.get("t_req")),
        "pre_read_ms": ms(max(a["t_send"], a["t_bt_before"] or 0), a.get("t_req")),
        "request_to_prompt_start_ms": ms(a.get("t_req"), a.get("t_pstart")),
        "sp_run_ms": round(sum(s["ms"] for s in batched), 1), "sp_run_tokens": sum(s["tokens"] for s in batched),
        "n_batched_segs": len(batched),
        "header_windows_ms": round(sum(s["ms"] for s in windows), 1), "header_tokens": sum(s["tokens"] for s in windows),
        "checkpoint_gaps_ms": gaps, "tail_gap_ms": tail_gap,
        "pdone_to_window_ms": ms(a.get("t_pdone"), a.get("t_win")),
        "first_window_ms": ms(a.get("t_win"), a.get("t_done")), "first_window_engine_ms": a.get("first_window_ms_engine"),
        "copy_ms": a.get("copy_ms"), "done_to_badm_ms": ms(a.get("t_done"), a["t_badm"]),
        "refill_ms": a.get("refill_ms", 0.0),
        "badm_to_first_bt_ms": ms(a["t_badm"], a["t_bt_after"]),
        "send_to_badm_ms": ms(a["t_send"], a["t_badm"]),
        "stall_send_to_first_bt_ms": ms(a["t_send"], a["t_bt_after"]),
        "R_eff_ms": round(lost * 1000, 1),
    }
    rows.append(r)

print(f"run {run}: D (rows/s, decode-only, by active slots): " +
      ", ".join(f"{k}: {statistics.mean(v):.1f} (n={len(v)}, min {min(v):.1f}, max {max(v):.1f})"
                for k, v in sorted(stretch.items())) + f"; D_ref used {Dref:.1f}")
cols = ["cls", "n", "drain_ms", "pre_read_ms", "request_to_prompt_start_ms", "sp_run_ms", "header_windows_ms",
        "checkpoint_gaps_ms", "tail_gap_ms", "pdone_to_window_ms", "first_window_ms", "copy_ms", "done_to_badm_ms",
        "refill_ms", "badm_to_first_bt_ms", "send_to_badm_ms", "stall_send_to_first_bt_ms", "R_eff_ms"]
print("\t".join(cols))
for r in rows:
    print("\t".join(str(r[c]) for c in cols))
print()
for cls in ("short", "long", "fill"):
    rs = [r for r in rows if r["cls"] == cls and (cls != "fill" or r["active_before"] >= 8)]
    if not rs:
        continue
    print(f"== {cls}: {len(rs)} admissions, median per phase")
    for c in cols[1:]:
        if c == "checkpoint_gaps_ms":
            v = [sum(r[c]) for r in rs if r[c] is not None]
        else:
            v = [r[c] for r in rs if r[c] is not None]
        if v:
            print(f"   {c:28s} median {statistics.median(v):8.1f}  min {min(v):8.1f}  max {max(v):8.1f}")
if len(sys.argv) > 3 and sys.argv[2] == "--json":
    Path(sys.argv[3]).write_text(json.dumps({"D": {k: v for k, v in stretch.items()}, "Dref": Dref, "rows": rows},
                                            indent=1))
