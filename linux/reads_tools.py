#!/usr/bin/env python3
"""Helpers of the GPU window window-reads.sh (branch ll5-reads: shorter prompt reads, phase 1 - STRATA_SR_CKPT_WARM,
STRATA_SR_CKPT_LATE, STRATA_SR_NODRAIN, umbrella STRATA_SHORT_READS).  No GPU in here: every mode reads files the window
wrote (engine logs, dumps, sweep rows) or measures prompts with the model's tokenizer.  Verdict lines start with '==> '
and end in PASS, FAIL or INFO (window-reads.sh copies them into its window.log).

Modes:
  tokcheck  the HTTP workloads' prompts measured with the tokenizer (ovkv_ident.py's conversations, bench_multiturn.py's
            agent mix): every request incl. its output < --max-context, the background documents > --kv-resident
  refok     may the lanes4 window's serial solo reference (results/lanes4/ref.json) stand in for this window's?  Same
            binary as production's (sha256 in that window's prep line), same config apart from the lanes, same prompts
            (lens), complete, no errors; exit 0 yes, 1 no (then the window runs the reference itself)
  prefsum   phase 0: lanes_stdin.py --mode prefill rows + the engine log (STRATA_TRACE / STRATA_PREFILL_TIMING): per phase
            and size send->BADM, the read, the overhead outside it and the admission's phases; deltas vs the first arm
  engstat   per engine log: the admission phases (all slots), STRATA_SR_* on/off lines, the 'short reads:' counters, the
            no-drain lines, LATE_VERIFY / TEST_FAULT / SLOT_PARK_VERIFY counts, ERR / watchdog / engine starts; checks
  bootcmp   the boot lines of two engine logs: expert cache slots, experts resident, VRAM free per card, the staging
  btcmp     batch_test16 dumps vs the base arm's: token identity per slot
  abttft    ab_quick / run_strata_variant rows of sweep.jsonl: agg, steady, TTFT p50 / p95 / max per arm; deltas
  srab      phase 1 (--short-read A/B): abttft + reply identity (INFO: another path, not an exactness claim) + the
            multi-turn rows (results of bench_multiturn.py --out)
"""
import argparse, glob, json, os, re, statistics as st, subprocess, sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
MAXCTX = 262144


def say(m):
    print(m, flush=True)


def verdict(ok):
    return "PASS" if ok else "FAIL"


def med(v):
    v = [x for x in v if x is not None]
    return st.median(v) if v else None


def f1(x):
    return "n/a" if x is None else f"{x:.1f}"


# ------------------------------------------------------------------------------------------------------- log parsing
ADM_RE = re.compile(r"strata trace: admission of slot (\d+): drain ([\d.]+), prompt ([\d.]+) \(read ([\d.]+), windows "
                    r"([\d.]+), (\d+) checkpoint\(s\) ([\d.]+) = sync ([\d.]+) \+ save ([\d.]+) \+ evict ([\d.]+), rest "
                    r"(-?[\d.]+)\), first window ([\d.]+), slot copy ([\d.]+), slot freed ([\d.]+), checkpoint copy ([\d.]+)")
ADM_F = ["drain", "prompt", "read", "windows", "ckpts", "ckpt", "sync", "save", "evict", "rest", "first_window",
         "slot_copy", "slot_freed", "ckpt_copy"]
PT_RE = re.compile(r"strata prefill timing: (\d+) tokens, GPU timeline (\d+) ms, wall (\d+) ms, host staging (\d+) ms:(.*)")
PTH_RE = re.compile(r"strata prefill timing: host: chunk setup .*? (\d+) ms, waiting for each chunk (\d+) ms, after each "
                    r"chunk .*? (\d+) ms, PLE (\d+) ms")
OV_RE = re.compile(r"overlapped read of (\d+) tokens in (\d+) ms \(drain ([\d.]+) ms\)")
ND_SKIP = "drain for BGEN skipped (STRATA_SR_NODRAIN)"
ND_DEF_RE = re.compile(r"deferred drain \(STRATA_SR_NODRAIN\) for (.*?): (\d+) groups in flight, ([\d.]+) ms")
SRC_RE = re.compile(r"short reads: warm pool (\d+) takes, (\d+) found it empty, (\d+) made; late checkpoints (\d+), (\d+) "
                    r"waited for \(([\d.]+) ms\); BGEN drains skipped (\d+), deferred (\d+), reads beside groups (\d+)")
SRC_F = ["warm_takes", "warm_empty", "warm_made", "late", "late_waited", "late_wait_ms", "nd_skipped", "nd_deferred",
         "nd_beside"]


def read_lines(path):
    try:
        return Path(path).read_text(errors="replace").splitlines()
    except OSError:
        return None


def admissions(lines):
    """[(slot, {field: value}, [the lines since the previous admission line])] in log order."""
    out, seg = [], []
    for l in lines:
        m = ADM_RE.search(l)
        if m:
            v = [float(x) for x in m.groups()[1:]]
            out.append((int(m.group(1)), dict(zip(ADM_F, v)), seg))
            seg = []
        else:
            seg.append(l)
    return out


def seg_facts(seg):
    pt = [PT_RE.search(l) for l in seg]
    pt = [m for m in pt if m]
    ph = [PTH_RE.search(l) for l in seg]
    ph = [m for m in ph if m]
    ov = [OV_RE.search(l) for l in seg]
    ov = [m for m in ov if m]
    return {"pt_gpu": sum(float(m.group(2)) for m in pt) if pt else None,
            "pt_wall": sum(float(m.group(3)) for m in pt) if pt else None,
            "pt_host_stage": sum(float(m.group(4)) for m in pt) if pt else None,
            "pt_phases": pt[-1].group(5).strip() if pt else "",
            "pt_setup": sum(float(m.group(1)) for m in ph) if ph else None,
            "pt_wait": sum(float(m.group(2)) for m in ph) if ph else None,
            "pt_after": sum(float(m.group(3)) for m in ph) if ph else None,
            "ov_ms": sum(float(m.group(2)) for m in ov) if ov else None,
            "ov_drain": sum(float(m.group(3)) for m in ov) if ov else None,
            "nd_skip": sum(ND_SKIP in l for l in seg),
            "nd_def": sum(1 for l in seg if ND_DEF_RE.search(l))}


def counters(lines):
    last = None
    for l in lines:
        m = SRC_RE.search(l)
        if m:
            last = dict(zip(SRC_F, [float(x) for x in m.groups()]))
    return last


# ------------------------------------------------------------------------------------------------------------ modes
def mode_tokcheck(a):
    import lanes_stdin as LS          # (the model's tokenizer, as the drivers use it)
    import ovkv_ident as OK
    import bench_multiturn as MT
    cfg = json.loads(Path(a.config).read_text())
    tok = LS.tokenizer(cfg)
    over = 24                          # per message: the chat template's role / start / end tokens (generous)
    n = lambda s: len(tok.encode(s))
    bad = 0
    # ovkv_ident.py: per round r and join JT: B background requests (document + question), the join, its second turn
    joins = [int(x) for x in a.ov_joins.split(",")]
    worst = 0
    bgmin = None
    for r, jt in enumerate(joins):
        bg_max = a.ov_bg_tok + jt // 100
        for u in range(a.ov_bg):
            t = n(f"Document {u}:\n{OK.doc(500 + u, a.ov_bg_ctx)}\n\nQuestion {r}: write a long, detailed story that uses "
                  f"the words of this document, part {r + 1}.") + over
            bgmin = t if bgmin is None else min(bgmin, t)
            worst = max(worst, t + bg_max)
        j = n(f"Report {r}-{a.ov_seed}:\n{OK.doc(9000 + 100 * a.ov_seed + r, jt)}\n\nTask: name the three most frequent "
              f"words and summarise the report.") + over
        j2 = j + 96 + over + n("Now name the least frequent word, and why it matters.") + over
        say(f"    ovkv round {r}: join {jt} (estimated by words) = {j} tokens (+96 out), its second turn <= {j2} (+64 out)")
        worst = max(worst, j + 96, j2 + 64)
    ok_ctx = worst + 64 <= MAXCTX
    ok_kv = bgmin is not None and bgmin > int(a.kvres)
    bad += not (ok_ctx and ok_kv)
    say(f"==> tokcheck ovkv_ident: longest request incl. its output {worst} tokens (+64 margin <= {MAXCTX}); background "
        f"documents >= {bgmin} tokens (want > --kv-resident {a.kvres}) {verdict(ok_ctx and ok_kv)}")
    # bench_multiturn.py: the whole history of the longest user's last turn
    worst = 0
    for uid in range(a.mt_users):
        hist = n(f"You are agent #{uid}. Answer concisely and precisely.") + over
        for t in range(a.mt_turns):
            nn = a.mt_doc if t == 0 else a.mt_add
            msg = n(f"[agent {uid}, step {t + 1}] New material:\n{MT.doc(uid, t, nn)}\n\nTask: summarise what is new in "
                    f"this material and how it relates to the earlier steps.")
            hist += msg + over
            worst = max(worst, hist + a.mt_max_tok)
            if t == 1 and uid == 0:
                say(f"    multiturn: a follow-up's new user message is {msg} tokens (+ the template's few; uid 0, turn 2: "
                    f"--short-read 16/32/64 decides its path once the history is reused)")
            hist += a.mt_max_tok + over
    ok = worst + 64 <= MAXCTX
    bad += not ok
    say(f"==> tokcheck bench_multiturn ({a.mt_users} users x {a.mt_turns} turns, doc {a.mt_doc}, add {a.mt_add}, "
        f"{a.mt_max_tok} out): longest request incl. its output <= {worst} tokens {verdict(ok)}")
    return 1 if bad else 0


def mode_refok(a):
    import lanes_stdin as LS
    why = []
    try:
        ref = json.loads(Path(a.ref).read_text())
    except Exception as e:  # noqa: BLE001
        say(f"refok: no reference ({e})")
        return 1
    if ref.get("label") != "REF" or ref.get("errors"):
        why.append(f"label {ref.get('label')} errors {ref.get('errors')}")
    seq = LS.SEQS["ref4"]
    miss = [x for x in seq if x not in (ref.get("cases") or {})]
    if miss:
        why.append(f"cases missing {miss[:5]}")
    if str((ref.get("info") or {}).get("kv_resident")) != a.kvres:
        why.append(f"kv_resident {(ref.get('info') or {}).get('kv_resident')} != {a.kvres}")
    # the prompts: today's drivers build the same ones (lengths of every case; the content follows from the code)
    if os.path.getmtime(os.path.join(HERE, "lanes_stdin.py")) > os.path.getmtime(a.ref):
        why.append("lanes_stdin.py is newer than the reference")
    cfg = json.loads(Path(a.prod).read_text())
    tok = LS.tokenizer(cfg)
    _, lens = LS.build_cases(tok, None)
    diff = [k for k in seq if lens.get(k) != (ref.get("lens") or {}).get(k)]
    if diff:
        why.append(f"prompt lengths differ for {diff[:5]}")
    # the reference's engine: the same binary production runs, the same config apart from the lanes
    try:
        first = Path(a.ref_txt).read_text(errors="replace").splitlines()[0]
        exe = re.search(r"starting (\S+);", first).group(1)
    except Exception:  # noqa: BLE001
        exe = None
    if exe is None or os.path.realpath(exe) != os.path.realpath(cfg["exe"]):
        why.append(f"reference exe {exe} is not production's {cfg['exe']}")
    wl = Path(a.window_log).read_text(errors="replace") if Path(a.window_log).is_file() else ""
    m = re.search(r"window prep: (\S+) sha256 ([0-9a-f]{16})", wl)
    if not m or m.group(1) != os.path.basename(cfg["exe"]) or m.group(2) != a.prod_sha:
        why.append(f"lanes4 prep line {m.group(0) if m else None} does not name production's binary sha {a.prod_sha}")
    rc = json.loads(Path(a.ref_cfg).read_text())
    strip = {"STRATA_CONV_TIER_DIR", "STRATA_SPEC_LANES", "STRATA_SOLO_PIPELINE"}
    pe = {k: v for k, v in cfg["env"].items() if k not in strip}
    re_ = {k: v for k, v in rc["env"].items() if k not in strip}
    if pe != re_ or cfg["args"] != rc["args"] or cfg.get("layer_split") != rc.get("layer_split") \
            or cfg.get("parallel") != rc.get("parallel") or cfg.get("gpu") != rc.get("gpu"):
        why.append("the reference's config differs from production's beyond the lanes, the tier dir and the log")
    if why:
        say(f"refok: NO - {'; '.join(why)}")
        return 1
    say(f"refok: yes - {a.ref} ({len(seq)} requests, production's binary, its config without the lanes)")
    return 0


def mode_prefsum(a):
    arms = []
    for spec in a.arms:
        label, rows_p, log_p = spec.split("=", 2)
        try:
            rows = json.loads(Path(rows_p).read_text())
        except Exception as e:  # noqa: BLE001
            say(f"==> prefsum {label}: no rows ({e}) FAIL")
            continue
        lines = read_lines(log_p)
        if lines is None:
            say(f"==> prefsum {label}: no engine log {log_p} FAIL")
            continue
        meas = [x for x in admissions(lines) if x[0] == a.slot]
        n_warm = len(meas) - len(rows)
        if n_warm != a.warmups:
            say(f"    prefsum {label}: {len(meas)} admissions of slot {a.slot} for {len(rows)} rows + {a.warmups} warm-ups: "
                f"matched from the end (INFO)")
        meas = meas[-len(rows):] if rows and len(meas) >= len(rows) else []
        for r, (s, f, seg) in zip(rows, meas):
            r["adm"] = f
            r["seg"] = seg_facts(seg)
        arms.append((label, rows, counters(lines)))
    if not arms:
        return 1
    keys = []
    for _, rows, _ in arms:
        for r in rows:
            k = (r["phase"], r["n"])
            if k not in keys:
                keys.append(k)
    base = {}
    for label, rows, cnt in arms:
        for k in keys:
            v = [r for r in rows if (r["phase"], r["n"]) == k]
            if not v:
                continue
            w = med([r["wall_ms"] for r in v])
            rd = med([r.get("read_ms") for r in v])
            ad = lambda f: med([r["adm"][f] for r in v if "adm" in r])
            sg = lambda f: med([r["seg"][f] for r in v if "seg" in r])
            outside = None if w is None or rd is None else w - rd
            line = (f"==> prefsum {label} {k[0]:5} {k[1]:5} tok: send->BADM {f1(w)} ms, read {f1(rd)} ms, outside the read "
                    f"{f1(outside)} ms | admission: drain {f1(ad('drain'))}, ckpt {f1(ad('ckpt'))} (sync {f1(ad('sync'))} "
                    f"save {f1(ad('save'))} evict {f1(ad('evict'))}), rest {f1(ad('rest'))}, first window "
                    f"{f1(ad('first_window'))}, slot copy {f1(ad('slot_copy'))}, slot freed {f1(ad('slot_freed'))}, ckpt "
                    f"copy {f1(ad('ckpt_copy'))} | prefill timing GPU {f1(sg('pt_gpu'))} wall {f1(sg('pt_wall'))} host "
                    f"staging {f1(sg('pt_host_stage'))} setup {f1(sg('pt_setup'))} wait {f1(sg('pt_wait'))} after "
                    f"{f1(sg('pt_after'))}; overlapped {f1(sg('ov_ms'))} (drain {f1(sg('ov_drain'))}); BGEN drains skipped "
                    f"{sum(r['seg']['nd_skip'] for r in v if 'seg' in r)}/{len(v)}")
            if label != arms[0][0] and k in base and w is not None and base[k][0] is not None:
                bw, bo = base[k]
                line += f" | vs {arms[0][0]}: send->BADM {w - bw:+.1f} ms"
                if outside is not None and bo is not None:
                    line += f", outside the read {outside - bo:+.1f} ms"
            say(line + " INFO")
            if label == arms[0][0]:
                base[k] = (w, outside)
                ph = [r["seg"]["pt_phases"] for r in v if r.get("seg", {}).get("pt_phases")]
                if ph:
                    say(f"    {label} {k[0]} {k[1]}: prefill timing phases (one read): {ph[0][:400]}")
        if cnt:
            say(f"==> prefsum {label}: short reads counters at the end {json.dumps(cnt)} INFO")
    return 0


OFF_RE = re.compile(r"(STRATA_SR_\w+|STRATA_SHORT_READS)[^:]* is off: (.*)")
ON_PAT = {"WARM": re.compile(r"STRATA_SR_CKPT_WARM=\d+: the checkpoint pool keeps"),
          "LATE": re.compile(r"STRATA_SR_CKPT_LATE: turn checkpoints go to the host after BADM"),
          "NODRAIN": None}


def eng_facts(lines):
    adm = admissions(lines)
    c = {"adm_n": len(adm)}
    for f in ("drain", "ckpt", "save", "evict", "slot_copy", "slot_freed", "ckpt_copy", "rest"):
        c[f] = med([x[1][f] for x in adm])
        c[f + "_max"] = max([x[1][f] for x in adm], default=None)
    c["off"] = sorted({m.group(0)[:200] for l in lines for m in [OFF_RE.search(l)] if m})
    c["on_warm"] = any(ON_PAT["WARM"].search(l) for l in lines)
    c["on_late"] = any(ON_PAT["LATE"].search(l) for l in lines)
    c["nd_off"] = any("STRATA_SR_NODRAIN is off" in l for l in lines)
    c["nd_skip"] = sum(ND_SKIP in l for l in lines)
    c["nd_def"] = sum(1 for l in lines if ND_DEF_RE.search(l))
    c["nd_drained_why"] = sorted({l.split("drained for a BGEN: ", 1)[1][:120] for l in lines if "drained for a BGEN: " in l})
    c["nd_read_drains"] = sorted({l.split("the read drains first: ", 1)[1][:120] for l in lines
                                  if "STRATA_SR_NODRAIN: the read drains first: " in l})
    c["cnt"] = counters(lines)
    c["verified"] = sum("verified: the same bytes as a direct save" in l for l in lines)
    c["differs"] = sum("DIFFERS from a direct save" in l for l in lines)
    c["late_fail"] = sum(bool(re.search(r"STRATA_SR_CKPT_LATE: the copy of the checkpoint at \d+ to the host failed", l))
                         for l in lines)
    c["park_ok"] = sum("STRATA_SLOT_PARK_VERIFY: slot" in l and "bytes identical" in l for l in lines)
    c["park_bad"] = sum("differs from a plain capture" in l for l in lines)
    c["staging"] = [l.split("strata serve: ", 1)[-1][:160] for l in lines if "MiB of staging" in l]
    c["err"] = [l[:200] for l in lines if l.startswith("ERR") or " ERR " in l or "GPU stopped responding" in l
                or "not a live token prefix" in l]
    c["watchdog"] = sum("no progress for" in l for l in lines)
    c["starts"] = sum("engine started" in l for l in lines)
    return c


def mode_engstat(a):
    facts = {}
    for spec in a.arms:
        label, path = spec.split("=", 1)
        files = sorted(glob.glob(path))
        if not files:
            say(f"    engstat {label}: no log {path} (step skipped?) INFO")
            continue
        lines = [l for f in files for l in (read_lines(f) or [])]
        c = eng_facts(lines)
        facts[label] = c
        say(f"==> engstat {label}: {c['adm_n']} admissions; median drain {f1(c['drain'])} (max {f1(c['drain_max'])}), ckpt "
            f"{f1(c['ckpt'])} = save {f1(c['save'])} + evict {f1(c['evict'])} (evict max {f1(c['evict_max'])}), slot copy "
            f"{f1(c['slot_copy'])}, slot freed {f1(c['slot_freed'])} (max {f1(c['slot_freed_max'])}), ckpt copy "
            f"{f1(c['ckpt_copy'])} ms; on: warm {c['on_warm']} late {c['on_late']} nodrain-off-line {c['nd_off']}; "
            f"BGEN drains skipped {c['nd_skip']}, deferred {c['nd_def']}; counters {json.dumps(c['cnt'])}; verify "
            f"{c['verified']} same / {c['differs']} DIFFERS; late copy failed {c['late_fail']}; park verify {c['park_ok']} "
            f"identical / {c['park_bad']} differ; ERR {len(c['err'])}, watchdog {c['watchdog']}, engine starts "
            f"{c['starts']} INFO")
        for l in c["off"]:
            say(f"    {label}: {l}")
        for l in c["nd_drained_why"][:6]:
            say(f"    {label}: BGEN drained: {l}")
        for l in c["nd_read_drains"][:6]:
            say(f"    {label}: read drained: {l}")
        for l in c["staging"][:4]:
            say(f"    {label}: {l}")
        for l in c["err"][:4]:
            say(f"    {label}: {l}")
    # checks (an arm without a log: SKIP, no verdict)
    for spec in a.want_on:
        label, sw = spec.split(":", 1)
        c = facts.get(label)
        if c is None:
            continue
        miss = []
        for s in sw.split(","):
            if s == "WARM" and (not c["on_warm"] or any("STRATA_SR_CKPT_WARM" in l for l in c["off"])):
                miss.append(s)
            if s == "LATE" and (not c["on_late"] or any("STRATA_SR_CKPT_LATE" in l for l in c["off"])):
                miss.append(s)
            if s == "NODRAIN" and c["nd_off"]:
                miss.append(s)
        say(f"==> engstat {label}: switches on {sw} (missing/off: {miss or 'none'}) {verdict(not miss)}")
    if a.want_off_all:
        for label in a.want_off_all.split(","):
            c = facts.get(label)
            if c is None:
                continue
            on = c["on_warm"] or c["on_late"] or c["nd_skip"] or c["cnt"] is not None or c["staging"]
            say(f"==> engstat {label}: no short-reads switch active (env unset: default path) {verdict(not on)}")
    if a.want_beside_any:
        arms = [x for x in a.want_beside_any.split(",") if x in facts]
        per = {x: int((facts[x]["cnt"] or {}).get("nd_beside", 0)) for x in arms}
        tot = sum(per.values())
        if arms:
            say(f"==> engstat STRATA_SR_NODRAIN took its path (reads beside groups in flight > 0 in some T0c arm): {per} "
                f"{verdict(tot > 0)}")
    for label in (a.want_verify.split(",") if a.want_verify else []):
        c = facts.get(label)
        if c is None:
            continue
        say(f"==> engstat {label}: STRATA_SR_CKPT_LATE_VERIFY {c['verified']} late checkpoints the same bytes as a direct "
            f"save, {c['differs']} DIFFERS (want > 0 and 0) {verdict(c['verified'] > 0 and c['differs'] == 0)}")
    for label in (a.want_fault.split(",") if a.want_fault else []):
        c = facts.get(label)
        if c is None:
            continue
        say(f"==> engstat {label}: STRATA_SR_CKPT_LATE_TEST_FAULT=copy: {c['late_fail']} failed late copies (want > 0), "
            f"park verify {c['park_ok']} identical / {c['park_bad']} differ, ERR {len(c['err'])}, watchdog "
            f"{c['watchdog']}, engine starts {c['starts']} (want 1) "
            f"{verdict(c['late_fail'] > 0 and not c['err'] and c['watchdog'] == 0 and c['starts'] <= 1 and c['park_bad'] == 0)}")
    for label in (a.want_park.split(",") if a.want_park else []):
        c = facts.get(label)
        if c is None:
            continue
        say(f"==> engstat {label}: STRATA_SLOT_PARK_VERIFY {c['park_ok']} parks identical, {c['park_bad']} differ "
            f"(want > 0 and 0) {verdict(c['park_ok'] > 0 and c['park_bad'] == 0)}")
    for label in (a.want_clean.split(",") if a.want_clean else []):
        c = facts.get(label)
        if c is None:
            continue
        say(f"==> engstat {label}: ERR {len(c['err'])}, watchdog {c['watchdog']}, engine starts {c['starts']} (want 0, 0, "
            f"<= 1) {verdict(not c['err'] and c['watchdog'] == 0 and c['starts'] <= 1)}")
    if a.compare:      # 'BASE,GOT': the admission phases T0a / T0b are for (INFO)
        b, g = a.compare.split(",")
        if b in facts and g in facts:
            fb, fg = facts[b], facts[g]
            parts = []
            for f in ("drain", "ckpt", "save", "evict", "slot_freed", "slot_copy", "ckpt_copy"):
                if fb[f] is not None and fg[f] is not None:
                    parts.append(f"{f} {fb[f]:.1f} -> {fg[f]:.1f}")
            say(f"==> engstat {g} vs {b}: admission medians (ms) {', '.join(parts)} INFO")
    return 0


CACHE0_RE = re.compile(r"strata generate: expert cache (\d+) slots, ([\d.]+) GiB")
CACHEK_RE = re.compile(r"layer split: (CUDA\d) runs layers [\d-]+, expert cache (\d+) slots \(([\d.]+) GiB\)")
FREE_RE = re.compile(r"strata serve: (?:(CUDA\d): )?(\d+) MiB of VRAM free with everything loaded(.*)")
RES_RE = re.compile(r"(\d+)% of the experts resident")


def boot(lines):
    cache, free, res, low = {}, {}, [], []
    for l in lines:
        m = CACHE0_RE.search(l)
        if m and "CUDA0" not in cache:
            cache["CUDA0"] = int(m.group(1))
        m = CACHEK_RE.search(l)
        if m:
            cache[m.group(1)] = int(m.group(2))
        m = FREE_RE.search(l)
        if m:
            free[m.group(1) or "CUDA0"] = int(m.group(2))
            if "LOW" in m.group(3) or "WARNING" in l:
                low.append(l[:160])
        m = RES_RE.search(l)
        if m:
            res.append(int(m.group(1)))
    return cache, free, res, low


def mode_bootcmp(a):
    lb, lg = read_lines(a.base), read_lines(a.got)
    if lb is None or lg is None:
        say(f"    bootcmp {a.what}: a log is missing ({a.base}, {a.got}): SKIP")
        return 0
    cb, fb, rb, _ = boot(lb)
    cg, fg, rg, low = boot(lg)
    cards = sorted(set(cb) | set(cg) | set(fb) | set(fg))
    say(f"==> bootcmp {a.what}: " + "; ".join(
        f"{c} expert cache {cb.get(c)} -> {cg.get(c)} slots, VRAM free {fb.get(c)} -> {fg.get(c)} MiB" for c in cards)
        + " INFO")
    for l in [l for l in lg if "MiB of staging" in l or "kept out of its expert cache" in l][:6]:
        say(f"    {a.what}: {l.split('strata ', 1)[-1][:200]}")
    ok = bool(rg) and all(x == 100 for x in rg) and not low and all(v >= 256 for v in fg.values()) and bool(fg)
    say(f"==> bootcmp {a.what}: experts resident {rg} (base {rb}), LOW/WARNING lines {len(low)}, VRAM free per card "
        f"{fg} (want 100 %, none, >= 256 MiB) {verdict(ok)}")
    return 0


def rc_ran(d, step):
    try:
        return Path(f"{d}/rc_{step}").read_text().strip() != "skip"
    except OSError:
        return False


def mode_btcmp(a):
    def dump(step):
        if not rc_ran(a.dir, step):
            say(f"    btcmp {step}: SKIP (step skipped)")
            return None
        try:
            return json.loads(Path(f"{a.dir}/{step}.json").read_text())["batch"]
        except Exception as e:  # noqa: BLE001
            say(f"==> btcmp {step}: no dump ({e}) FAIL")
            return None
    b = dump(a.base)
    if b is None:
        say(f"==> btcmp {a.what}: no base dump ({a.base}) FAIL")
        return 0
    for step in a.arms.split(","):
        g = dump(step)
        if g is None:
            continue
        bad = [i for i in range(max(len(b), len(g))) if i >= len(b) or i >= len(g) or b[i] != g[i]]
        first = {i: next((k for k in range(min(len(b[i]), len(g[i]))) if b[i][k] != g[i][k]), min(len(b[i]), len(g[i])))
                 for i in bad if i < len(b) and i < len(g)}
        say(f"==> btcmp {a.what} {step} vs {a.base}: {len(b) - len([i for i in bad if i < len(b)])}/{len(b)} slots "
            f"token-identical" + (f" (differ: {first}) FAIL" if bad or not b else " PASS"))
    return 0


def sweep_rows(path, since):
    rows = {}
    for i, line in enumerate(open(path)):
        if i < since or not line.strip():
            continue
        r = json.loads(line)
        rows[(r["label"], r["users"])] = r
    return rows


def ab_table(a, arms, reps):
    rows = sweep_rows(a.sweep, a.since_line)

    def lab(arm, k, kind):
        return f"ab-{a.tag}-{arm}" + (f"-r{k}" if reps > 1 else "") + f"-{kind}"
    levels = sorted({u for (l, u) in rows if l.startswith(f"ab-{a.tag}-")})
    res = {}
    for kind in ("short", "long"):
        for u in levels:
            for arm in arms:
                v = [rows.get((lab(arm, k, kind), u)) for k in range(1, reps + 1)]
                if not all(v):
                    continue
                res[(kind, u, arm)] = {
                    "agg": st.mean(x["agg_out_tps"] for x in v),
                    "steady": st.mean((x.get("agg_out_tps_steady") or 0) for x in v),
                    "dec": st.mean(x["dec_tps_p50"] for x in v),
                    "p50": st.mean(x["ttft_p50"] for x in v), "p95": st.mean(x["ttft_p95"] for x in v),
                    "max": max(x["ttft_max"] for x in v), "err": sum(x["errors"] for x in v)}
            base = res.get((kind, u, arms[0]))
            for arm in arms:
                x = res.get((kind, u, arm))
                if not x:
                    continue
                d = ""
                if base and arm != arms[0]:
                    d = (f" | vs {arms[0]}: agg {(x['agg'] / base['agg'] - 1) * 100:+.2f} %, steady "
                         f"{(x['steady'] / base['steady'] - 1) * 100 if base['steady'] else 0:+.2f} %, TTFT p50 "
                         f"{x['p50'] - base['p50']:+.2f} s, p95 {x['p95'] - base['p95']:+.2f} s, max "
                         f"{x['max'] - base['max']:+.2f} s")
                say(f"==> {a.tag} {kind:>5} {u:>2} users {arm:<10} agg {x['agg']:.1f} steady {x['steady']:.1f} dec p50 "
                    f"{x['dec']:.1f} TTFT p50 {x['p50']:.2f} p95 {x['p95']:.2f} max {x['max']:.2f} s (mean of {reps}; max: "
                    f"over them), errors {x['err']}{d} INFO")
    return res, levels, lab


def mode_abttft(a):
    arms = a.arms_list.split(",")
    res, _, _ = ab_table(a, arms, a.reps)
    errs = sum(x["err"] for x in res.values())
    say(f"==> {a.tag}: rows {len(res)}, request errors {errs} {verdict(bool(res) and errs == 0)}")
    return 0


def mode_srab(a):
    arms = a.arms_list.split(",")
    res, levels, lab = ab_table(a, arms, 1)
    for kind in ("short", "long"):
        for u in levels:
            for arm in arms[1:]:
                p = subprocess.run([sys.executable, "-I", os.path.join(HERE, "compare_replies.py"), a.requests,
                                    lab(arms[0], 1, kind), lab(arm, 1, kind), "--users", str(u), "--show", "0"],
                                   capture_output=True, text=True)
                say(f"==> {a.tag} {kind} {u} users {arm} vs {arms[0]} replies: {p.stdout.strip().splitlines()[0] if p.stdout.strip() else p.stderr.strip()[:200]} INFO")
    mt = {}
    try:
        for i, line in enumerate(open(a.mt)):
            if line.strip():
                r = json.loads(line)
                mt[r["label"]] = r
    except OSError:
        pass
    base = mt.get(f"{a.tag}-{arms[0]}-mt")
    for arm in arms:
        r = mt.get(f"{a.tag}-{arm}-mt")
        if not r:
            say(f"    {a.tag} {arm}: no multi-turn row")
            continue
        turns = [r[k] for k in sorted(k for k in r if re.fullmatch(r"turn\d+", k))]
        tt = "; ".join(f"t{i + 1} prompt {t['prompt']} cached {t['cached']} TTFT p50 {t['ttft_p50']:.2f} p95 "
                       f"{t['ttft_p95']:.2f} dec {t['dec_p50']:.1f}" for i, t in enumerate(turns))
        d = ""
        if base and arm != arms[0]:
            same = sum(1 for x, y in zip(base["replies"], r["replies"]) if x == y)
            d = (f" | vs {arms[0]}: agg {(r['agg_out_tps'] / base['agg_out_tps'] - 1) * 100:+.2f} %, follow-up TTFT p50 "
                 f"{r['follow_ttft_p50'] - base['follow_ttft_p50']:+.2f} s, replies identical {same}/{len(base['replies'])}")
        say(f"==> {a.tag} multi-turn {arm}: wall {r['wall_s']} s, agg {r['agg_out_tps']} tok/s, follow-up TTFT p50 "
            f"{r['follow_ttft_p50']} s; {tt}{d} INFO")
    errs = sum(x["err"] for x in res.values())
    say(f"==> {a.tag}: sweep rows {len(res)}, multi-turn rows {sum(1 for arm in arms if f'{a.tag}-{arm}-mt' in mt)}/"
        f"{len(arms)}, request errors {errs} {verdict(bool(res) and errs == 0)}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["tokcheck", "refok", "prefsum", "engstat", "bootcmp", "btcmp",
                                                      "abttft", "srab"])
    ap.add_argument("--config", default="")
    ap.add_argument("--kvres", default="28672")
    ap.add_argument("--ov-bg", type=int, default=4)
    ap.add_argument("--ov-bg-ctx", type=int, default=45000)
    ap.add_argument("--ov-bg-tok", type=int, default=400)
    ap.add_argument("--ov-joins", default="1000,8000,32000")
    ap.add_argument("--ov-seed", type=int, default=7)
    ap.add_argument("--mt-users", type=int, default=16)
    ap.add_argument("--mt-turns", type=int, default=4)
    ap.add_argument("--mt-doc", type=int, default=2000)
    ap.add_argument("--mt-add", type=int, default=24)
    ap.add_argument("--mt-max-tok", type=int, default=192)
    ap.add_argument("--ref", default="")
    ap.add_argument("--ref-txt", default="")
    ap.add_argument("--ref-cfg", default="")
    ap.add_argument("--window-log", default="")
    ap.add_argument("--prod", default="")
    ap.add_argument("--prod-sha", default="")
    ap.add_argument("--arms", nargs="*", default=[], help="prefsum: LABEL=ROWS=LOG; engstat: LABEL=LOG(glob)")
    ap.add_argument("--slot", type=int, default=15)
    ap.add_argument("--warmups", type=int, default=3)
    ap.add_argument("--want-on", action="append", default=[], help="engstat: LABEL:WARM,LATE,NODRAIN")
    ap.add_argument("--want-off-all", default="")
    ap.add_argument("--want-beside-any", default="")
    ap.add_argument("--want-verify", default="")
    ap.add_argument("--want-fault", default="")
    ap.add_argument("--want-park", default="")
    ap.add_argument("--want-clean", default="")
    ap.add_argument("--compare", default="")
    ap.add_argument("--base", default="")
    ap.add_argument("--got", default="")
    ap.add_argument("--what", default="")
    ap.add_argument("--dir", default="")
    ap.add_argument("--arms-list", default="")
    ap.add_argument("--sweep", default=os.path.join(HERE, "results/sweep.jsonl"))
    ap.add_argument("--requests", default=os.path.join(HERE, "results/sweep.requests.jsonl"))
    ap.add_argument("--since-line", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--mt", default="")
    a = ap.parse_args()
    if a.mode == "btcmp":
        a.arms = ",".join(a.arms)
    return {"tokcheck": mode_tokcheck, "refok": mode_refok, "prefsum": mode_prefsum, "engstat": mode_engstat,
            "bootcmp": mode_bootcmp, "btcmp": mode_btcmp, "abttft": mode_abttft, "srab": mode_srab}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
