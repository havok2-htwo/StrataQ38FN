#!/usr/bin/env python3
"""Drivers of the GPU window window-lanes.sh (M1a STRATA_SOLO_PIPELINE=1, M1b STRATA_SPEC_LANES=2, branch ll3-all).

The engine is spoken to over its --serve stdin/stdout protocol (serve/server.py's lines: GEN / GENI alone,
BGEN / BGENI <slot> <max_new> <keys> [<embeddings file>] <ids> into a batch slot, BSTOP, QUIT; answers RESUME, PP,
REUSED, T, DONE, BADM <slot> <0|1>, and the slots' BT <slot> <tok> / BDONE <slot> <produced> <finish> <ms>), through
vision_batch_check.py's Engine class (Strata-1441/tools).  Every engine started here gets the config's env, its GPUs
in PCI order, its library dirs and a fresh conversation-tier dir under /media/raid0/strata-kvspill-bench/lanes
(never production's), and writes its stderr to the log given with --log (never production's).

Modes (each prints its verdicts as lines starting with '==> ', ending in PASS, FAIL or INFO):
  encode   pictures A and B through the config's image encoder (CPU: "gpu": false), before the window; --out DIR
  boot     start, READY, one short GEN, the engine's own boot lines (loans, VRAM free with everything loaded, the
           M1a / M1b cost lines), nvidia-smi, QUIT.  --expect off|m1a|m1b; --base <boot JSON of the env-off boot>
  solo     GEN requests in a fixed order (--seq ref|m1a|fm), tokens and each request's log lines -> --out JSON
  lanes    the lane exactness drive: per group BGEN into each of its slots in turn (0, 4[, 8[, 12]]), each waiting
           for its BADM (STRATA_SPEC_LANES_HOLD=k, k = the group's size, keeps the earlier slots untouched), every
           slot's tokens until BDONE -> --out JSON (--pairs all|one|ov: M1b's pairs; g2s|g3|g4: groups of 2/3/4)
  switch   window-lanes4.sh's mode switches on one engine (STRATA_SPEC_LANES_HOLD=2): GEN (1 user, pipelined), then
           the lane slots one after the other while the earlier ones decode (2 -> 3 -> 4 lanes), BSTOP down to two,
           then one, then GEN again (1 -> 2 -> 3 -> 4 -> 2 -> 1) -> --out JSON
  compare  --ref <solo JSON> --got <solo, lanes or switch JSON>: token identity per request / per lane slot (a lane
           slot's lane phase: its contiguous lane windows from its admission on, from the 'lanes start' / 'lanes ran'
           lines)
  prefill  prefill speed by prompt length (slot 15, phases idle and dec15 = 15 slots decoding beside the read)
  sweep    per-arm speed from results/sweep.jsonl (ab_quick labels): agg tok/s, per-user decode p50, deltas
  abcheck  ab_quick's reply-identity brackets -> PASS / FAIL (or INFO)
  lanelog  'lanes start' / 'lanes ran' / watchdog / ERR counts in engine logs
  pllog    'request pipelined=0|1 (reason)' counts in engine logs

Prompt lengths are measured with the model's tokenizer (Prompts.text_like / exact slices), never estimated from
words; every request is checked to fit --max-context 262144 with its max_new.
"""
import argparse, collections, glob, json, os, queue, re, statistics as st, subprocess, sys, tempfile, time
from pathlib import Path

TOOLS = "/media/raid0/StrataQ38FN/Strata-1441/tools"
sys.path.insert(0, TOOLS)
import vision_batch_check as V  # noqa: E402

# (window-lanes4.sh sets LANES_TIER_ROOT to its own dir under /media/raid0/strata-kvspill-bench)
TIER_ROOT = os.environ.get("LANES_TIER_ROOT") or "/media/raid0/strata-kvspill-bench/lanes"
if not TIER_ROOT.startswith("/media/raid0/strata-kvspill-bench/"):
    raise SystemExit(f"LANES_TIER_ROOT {TIER_ROOT}: not under /media/raid0/strata-kvspill-bench (never production's)")
MAXCTX = 262144
# exactness settings of every token comparison (both sides): no adaptive swaps, no PCIe share, no suffix drafter
# (its drafts would make the serial reference's windows differ from the lanes' for a reason that is not the lanes)
EXACT = "--pcie-frac 0 --adapt-every 0 --suffix-draft 0"
T07 = "temperature=0.7 seed=1234"
Q_LONG = V.Q_TEXT_LONG

SOLO_SEQ = ["s_short1", "s_short2", "s_1k2", "s_4k", "s_16k", "s_48k", "t_short", "t_4k", "r_turn1", "r_other",
            "r_follow"]
FM_SEQ = ["s_short1", "s_1k2", "s_4k", "t_short", "s_16k"]
PAIRS = {"all": ["L_short", "L_long", "L_t07", "L_32k", "L_32kr", "L_pcie", "L_vis"],
         "one": ["L_short", "L_long", "L_t07", "L_32k"],
         "ov": ["L_short", "L_32kr"],
         # window-lanes4.sh (up to four lanes): 2 lanes in the shared profile, 3 and 4 lanes (HOLD=2 / 3 / 4)
         "g2s": ["L_short", "L_t07", "L_32k", "L_32kr", "L_vis"],
         "g3": ["G3_short", "G3_t07", "G3_32k", "G3_32kr"],
         "g4": ["G4_short", "G4_mix", "G4_32k"]}
LANE_REF_SEQ = [f"{p}.{s}" for p in PAIRS["all"] for s in (0, 4)]
G_REF_SEQ = [f"{g}.{s}" for g in PAIRS["g3"] for s in (0, 4, 8)] + [f"{g}.{s}" for g in PAIRS["g4"] for s in (0, 4, 8, 12)]
SW_SLOTS = (0, 4, 8, 12)
SW_MAX_NEW = 1500
SW_STEP = 48                      # tokens per user between two mode switches
SW_REF_SEQ = ["sw_s1"] + [f"sw.{s}" for s in SW_SLOTS] + ["sw_s2"]
SEQS = {"ref": SOLO_SEQ + LANE_REF_SEQ, "m1a": SOLO_SEQ, "fm": FM_SEQ,
        "ref4": SOLO_SEQ + LANE_REF_SEQ + G_REF_SEQ + SW_REF_SEQ}

LOG_KEEP = re.compile(r"pipelined=|KV streaming:|strata serve: prompt \d+ tokens|conversation cache: (restored|parked)|strata pipeline: \d+ windows|"
                      r"STRATA_SPEC_LANES|STRATA_SOLO_PIPELINE|overlapped read|no progress for|ERR|WARNING")
FACT_RE = re.compile(r"no loan|borrows|VRAM free with everything loaded|kept out|keeping it in VRAM|STRATA_SOLO_PIPELINE|"
                     r"STRATA_SPEC_LANES|experts resident|too little room|is off|stays serial|INFO spec_lanes|"
                     r"STRATA_PIPE_OVERLAP|KV streaming: \d+ of")


def say(msg):
    print(msg, flush=True)


def verdict(ok):
    return "PASS" if ok else "FAIL"


# ------------------------------------------------------------------------------------------------------------ engine
def engine_env(cfg, extra_env, drop):
    """serve/server.py's child_env (the GPUs in PCI order, the config's env, its library dirs), a fresh tier dir."""
    env = dict(os.environ)
    if cfg.get("gpu"):
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in cfg["gpu"])
    env.update({k: str(v) for k, v in (cfg.get("env") or {}).items()})
    for k in drop:
        env.pop(k, None)
    env["LD_LIBRARY_PATH"] = ":".join(cfg.get("lib_dirs", []) + [env.get("LD_LIBRARY_PATH", "")])
    os.makedirs(TIER_ROOT, exist_ok=True)
    env["STRATA_CONV_TIER_DIR"] = tempfile.mkdtemp(prefix="tier-", dir=TIER_ROOT)
    env.update(extra_env)
    return env


def kv_list(items):
    out = {}
    for kv in items or []:
        k, _, v = kv.partition("=")
        out[k] = v
    return out


def start(a, cfg, name):
    exe = a.exe or cfg["exe"]
    env = engine_env(cfg, kv_list(a.env), a.drop_env or [])
    shown = {k: env[k] for k in sorted(env) if k.startswith("STRATA_") and k not in (cfg.get("env") or {})}
    gone = [k for k in (a.drop_env or []) if k in (cfg.get("env") or {})]
    say(f"[{name}] starting {exe}; extra args '{a.extra}'; env beyond the config's: {shown}; dropped: {gone}; "
        f"log {a.log}")
    batch = int(cfg.get("parallel") or 16)
    return V.Engine(exe, cfg, batch, True, a.extra.split(), env, a.log, name, line_timeout=a.line_timeout,
                    start_timeout=900.0)


def smi():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        return [[int(x) for x in l.split(",")] for l in r.stdout.strip().splitlines()]
    except Exception:  # noqa: BLE001
        return []


def boot_facts(text):
    lines = [l.strip() for l in text.splitlines() if FACT_RE.search(l)]
    free = {}
    for l in lines:
        m = re.search(r"(?:(CUDA\d): )?(\d+) MiB of VRAM free with everything loaded", l)
        if m:
            free[m.group(1) or "CUDA0"] = int(m.group(2))
    # every '<N>% of the experts resident' line (one per layer-split stage line the engine writes), in order
    res = [int(m.group(1)) for l in lines for m in [re.search(r"(\d+)% of the experts resident", l)] if m]
    return {"lines": lines, "free": free, "res_pct": res or None,
            "no_loan": sum("no loan" in l for l in lines),
            "borrows": [l for l in lines if "borrows" in l],
            "low": [l for l in lines if "VRAM free with everything loaded" in l and ("LOW" in l or "WARNING" in l)]}


# ----------------------------------------------------------------------------------------------------------- prompts
def load_pics(path):
    if not path or not Path(path).is_file():
        return None
    p = json.loads(Path(path).read_text())
    return p if all(Path(x["file"]).is_file() for x in p.values()) else None


def build_cases(tok, pics):
    P = V.Prompts(tok, pics or {}, 0)
    c = {}

    def add(name, ids, keys="", max_new=160, emb=None):
        if len(ids) + max_new + 64 > MAXCTX:
            raise SystemExit(f"{name}: {len(ids)} + {max_new} tokens do not fit --max-context {MAXCTX}")
        c[name] = {"ids": ids, "keys": keys, "max_new": max_new, "emb": emb}

    # M1a (solo path): < 32K, >= 48K, sampled, and a conversation resumed from the RAM cache above 32K
    add("s_short1", P.text(0))
    add("s_short2", P.text(1))
    add("s_1k2", P.text_like(1200, 1, Q_LONG))
    add("s_4k", P.text_like(4000, 2, Q_LONG))
    add("s_16k", P.text_like(16000, 3, Q_LONG))
    add("s_48k", P.text_like(49000, 4, Q_LONG))
    add("t_short", P.text(2), T07)
    add("t_4k", P.text_like(4000, 5, Q_LONG), T07)
    turn1 = P.text_like(40000, 6, V.Q_TEXT_TURN1)
    add("r_turn1", turn1, max_new=64)
    add("r_other", P.text(3), max_new=32)               # another conversation: r_turn1 goes to the RAM cache
    # restored from there, > 32K of context; a longer decode, so its own windows stream K/V from RAM (the
    # KV-streaming line's delta of this request must be > 0)
    add("r_follow", P.follow_up(turn1, V.CANNED_TEXT, V.Q_FOLLOW_TEXT), max_new=400)
    # M1b lane pairs (slot 0, slot 4)
    add("L_short.0", P.text(4))
    add("L_short.4", P.text(5))
    add("L_long.0", P.text_like(1200, 11, Q_LONG))
    add("L_long.4", P.text_like(4000, 12, Q_LONG))
    add("L_t07.0", P.text(6), T07)
    add("L_t07.4", P.text(7), T07)
    add("L_32k.0", P.text_like(40000, 13, Q_LONG))
    add("L_32k.4", P.text(8))
    add("L_32kr.0", P.text(14))                         # the long read is the second admission: beside the held slot 0
    add("L_32kr.4", P.text_like(40000, 14, Q_LONG))
    add("L_pcie.0", P.text(9), "pcie_frac=0.5")         # its own PCIe share beside the other's (per-slot pcie_num)
    add("L_pcie.4", P.text(10))
    add("L_vis.0", P.text(12))                          # text in the slots that held pictures (set_rope_base)
    add("L_vis.4", P.text(13))
    add("V_text1", P.text(11), max_new=400)             # beside the pictures (not a lane slot: HOLD stays out)
    if pics:
        add("V_imgA", P.image("A", V.Q_IMAGE), max_new=48, emb=pics["A"]["file"])
        add("V_imgB", P.image("B", V.Q_IMAGE2), max_new=48, emb=pics["B"]["file"])
    # window-lanes4.sh: groups of 3 (slots 0, 4, 8) and 4 (0, 4, 8, 12): greedy, T=0.7 seed 1234, a > 32K context in
    # the first (held) and in the last admission
    add("G3_short.0", P.text(16))
    add("G3_short.4", P.text(17))
    add("G3_short.8", P.text(18))
    add("G3_t07.0", P.text(19), T07)
    add("G3_t07.4", P.text_like(1200, 30, Q_LONG))
    add("G3_t07.8", P.text(20), T07)
    add("G3_32k.0", P.text_like(40000, 31, Q_LONG))
    add("G3_32k.4", P.text(21))
    add("G3_32k.8", P.text(22), T07)
    add("G3_32kr.0", P.text(23))
    add("G3_32kr.4", P.text(24), T07)
    add("G3_32kr.8", P.text_like(40000, 32, Q_LONG))
    add("G4_short.0", P.text(25))
    add("G4_short.4", P.text(26))
    add("G4_short.8", P.text(27))
    add("G4_short.12", P.text(28))
    add("G4_mix.0", P.text(29), T07)
    add("G4_mix.4", P.text_like(1200, 33, Q_LONG))
    add("G4_mix.8", P.text(30), T07)
    add("G4_mix.12", P.text_like(4000, 34, Q_LONG))
    add("G4_32k.0", P.text(31))
    add("G4_32k.4", P.text_like(40000, 35, Q_LONG))
    add("G4_32k.8", P.text(32), T07)
    add("G4_32k.12", P.text(33))
    # the mode switches (mode_switch): long replies (an agenda item by item), so every user still decodes at the
    # switches; slot 8 has > 32K of context; the two GEN around them are the 1-user (pipelined) mode
    add("sw_s1", P.text_like(1200, 40, Q_LONG), max_new=300)
    add("sw.0", P.text_like(2000, 36, Q_LONG), max_new=SW_MAX_NEW)
    add("sw.4", P.text_like(1500, 37, Q_LONG), T07, max_new=SW_MAX_NEW)
    add("sw.8", P.text_like(40000, 38, Q_LONG), max_new=SW_MAX_NEW)
    add("sw.12", P.text_like(2500, 39, Q_LONG), T07, max_new=SW_MAX_NEW)
    add("sw_s2", P.text_like(1600, 41, Q_LONG), max_new=300)
    lens = {k: len(v["ids"]) for k, v in c.items()}
    for k, lo in (("s_48k", 48000), ("r_turn1", 34000), ("L_32k.0", 34000), ("L_32kr.4", 34000), ("G3_32k.0", 34000),
                  ("G3_32kr.8", 34000), ("G4_32k.4", 34000), ("sw.8", 34000)):
        if lens[k] < lo:
            raise SystemExit(f"{k}: {lens[k]} tokens, wanted >= {lo}")
    if len(c["r_follow"]["ids"]) <= 32768:
        raise SystemExit("r_follow is not above 32K")
    return c, lens


def tokenizer(cfg):
    return V.tokenizer(cfg["tokenizer"])


# ------------------------------------------------------------------------------------------------------- log slices
def slices(e, marks):
    """Each request's engine log lines: from its own start to the next request's start (late lines included)."""
    try:
        text = Path(e.log_path).read_bytes()      # (byte offsets, as the marks: log_size)
    except OSError:
        text = b""
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1] if i + 1 < len(marks) else len(text)
        seg = text[m:end].decode(errors="replace")
        out.append([l for l in seg.splitlines() if LOG_KEEP.search(l)])
    return out


def settle(e, mark, kvres, timeout=15.0):
    """Wait until this request's end-of-request block is in the log: the 'strata serve: prompt N tokens' line (written
    after DONE) and, with --kv-resident, the cumulative 'KV streaming: ... MiB read from RAM' line after it - so the next
    request's mark is taken after them and they are not charged to the next request."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        t = e.log_text(mark)
        i = t.find("strata serve: prompt ")
        if i >= 0 and (not kvres or "KV streaming: " in t[i:] or time.monotonic() > end - timeout + 5.0):
            time.sleep(0.2)
            return
        time.sleep(0.1)


PROMPT_LINE = re.compile(r"strata serve: prompt (\d+) tokens = ")
KV_LINE = re.compile(r"strata serve: KV streaming: .*?([\d.]+) MiB read from RAM")


def kv_by_request(log_path, reqs):
    """Each request's cumulative 'KV streaming: ... MiB read from RAM' value, found by walking the whole log after the
    run: the request's own 'strata serve: prompt <n> tokens = ...' line (n = its prompt length, searched in request
    order after the previous request's line), then the first KV-streaming line after it and before the next request's
    prompt line (the engine writes it right after the prompt line).  reqs: [(name, n)].  -> {name: MiB or None}"""
    try:
        lines = Path(log_path).read_text(errors="replace").splitlines()
    except OSError:
        lines = []
    out, i = {}, 0
    for name, n in reqs:
        out[name] = None
        j = next((k for k in range(i, len(lines)) for m in [PROMPT_LINE.search(lines[k])] if m and int(m.group(1)) == n),
                 None)
        if j is None:
            continue
        i = j + 1
        for k in range(j + 1, len(lines)):
            if PROMPT_LINE.search(lines[k]):
                break
            m = KV_LINE.search(lines[k])
            if m:
                out[name] = float(m.group(1))
                break
    return out


def save(path, obj):
    tmp = path + ".tmp"
    Path(tmp).write_text(json.dumps(obj))
    os.replace(tmp, path)


# -------------------------------------------------------------------------------------------------------------- modes
def mode_encode(a, cfg):
    work = Path(a.out)
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join(cfg.get("lib_dirs", []) + [env.get("LD_LIBRARY_PATH", "")])
    if cfg["vision"].get("gpu"):
        raise SystemExit("the config's encoder runs on a GPU: not before the window")
    enc = V.Encoder(cfg["vision"], work, env, work / "encoder.log", 600.0, 300.0)
    try:
        pics = {k: enc.encode(V.picture_png(k), f"pic{k}") for k in V.PICTURES}
    finally:
        enc.close()
    save(str(work / "pics.json"), pics)
    shown = ", ".join("%s %d tokens" % (k, p["n"]) for k, p in pics.items())
    say(f"==> encode: pictures {shown} -> {work}/pics.json PASS")


def mode_boot(a, cfg):
    e = start(a, cfg, a.label)
    res = {"label": a.label, "info": e.info}
    try:
        tok = tokenizer(cfg)
        P = V.Prompts(tok, {}, 0)
        mark = e.log_size()
        r = e.solo(16, P.text(15), name="boot-gen")
        time.sleep(0.5)
        res["gen"] = {"tokens": len(r["tokens"]), "log": [l for l in e.log_text(mark).splitlines() if LOG_KEEP.search(l)]}
        res["smi"] = smi()
        res["rss_mib"] = rss_mib(e.p.pid)
    finally:
        e.quit()
    text = Path(a.log).read_text(errors="replace")
    f = boot_facts(text)
    res.update(f)
    save(a.out, res)
    for l in f["lines"]:
        say("    " + l[:260])
    for l in res["gen"]["log"]:
        say("    (first GEN) " + l[:200])
    say(f"    INFO: {' '.join(f'{k}={v}' for k, v in sorted(e.info.items()))}")
    say("    nvidia-smi after the first GEN (index, used MiB, free MiB): " + "; ".join(",".join(map(str, x)) for x in res["smi"]))
    cards = [f"CUDA{i}" for i in range(4)]
    free = [f["free"].get(c) for c in cards]
    ok = f["no_loan"] == 4 and not f["borrows"] and not f["low"] and all(x is not None and x >= 512 for x in free)
    why = []
    if a.expect == "m1a":
        took = any("second verifiers, hand-offs and GDN snapshots took" in l for l in f["lines"])
        serial = [l for l in f["lines"] if "stays serial" in l or "too little room" in l]
        pl1 = any("request pipelined=1" in l for l in res["gen"]["log"])
        ok = ok and took and not serial and pl1
        why.append(f"cost line {'yes' if took else 'NO'}, serial/too-little-room lines {len(serial)}, first GEN "
                   f"pipelined=1 {'yes' if pl1 else 'NO'}")
    elif a.expect == "m1b":
        on = e.info.get("spec_lanes") == "2" and e.info.get("lane_slots") == "0,4"
        off = [l for l in f["lines"] if "STRATA_SPEC_LANES=2 is off" in l or "STRATA_SPEC_LANES is off" in l
               or "too little room" in l]
        keep = [l for l in f["lines"] if "keeping it in VRAM" in l]
        took = any("lane verifiers, hand-offs, the drafters' graphs" in l for l in f["lines"])
        ok = ok and on and not off and not keep and took
        why.append(f"INFO spec_lanes=2 lane_slots=0,4 {'yes' if on else 'NO'}, cost line {'yes' if took else 'NO'}, "
                   f"off/too-little-room lines {len(off)}, 'keeping it in VRAM' {len(keep)}")
    elif a.expect == "lc":
        # window-lanes4.sh: up to four lanes (STRATA_SPEC_LANES=N): the lanes on, how many the boot fitted, the solo
        # pipeline folded in (the first GEN pipelined), no lane dropped for want of a drafter
        n = int(e.info.get("spec_lanes") or 0)
        off = [l for l in f["lines"] if "STRATA_SPEC_LANES is off" in l or "is not supported" in l]
        took = any("lane verifiers, hand-offs, the drafters' graphs" in l for l in f["lines"])
        pl1 = any("request pipelined=1" in l for l in res["gen"]["log"])
        drops = [l for l in f["lines"] if re.search(r"lane \d+ goes|did not load", l)]
        ok = ok and n >= max(2, a.want_lanes) and (a.max_lanes == 0 or n <= a.max_lanes) and not off and took \
            and (pl1 or not a.want_pipelined)
        why.append(f"INFO spec_lanes={n} lane_slots={e.info.get('lane_slots')} (want >= {max(2, a.want_lanes)}"
                   f"{', <= ' + str(a.max_lanes) if a.max_lanes else ''}), cost line {'yes' if took else 'NO'}, off lines "
                   f"{len(off)}, first GEN pipelined=1 {'yes' if pl1 else 'NO'}{'' if a.want_pipelined else ' (not asked)'}, "
                   f"lanes dropped at boot {len(drops)}")
        res["lanes"] = n
        save(a.out, res)
    if a.want_pipelined and a.expect in ("m1b",):
        pl1 = any("request pipelined=1" in l for l in res["gen"]["log"])
        ok = ok and pl1
        why.append(f"first GEN pipelined=1 {'yes' if pl1 else 'NO'}")
    delta = ""
    bj = json.loads(Path(a.base).read_text()) if a.base and Path(a.base).is_file() else None
    bname = a.base_name or "env-off"
    extra = []        # verdicts of their own (they do not gate the later steps: the main line does)
    if bj:
        b = bj["free"]
        d = {c: f["free"].get(c, 0) - b.get(c, 0) for c in cards}
        delta = f" (vs {bname}: " + ", ".join(f"{c} {d[c]:+d}" for c in cards) + " MiB)"
        if a.vram_tol:
            within = all(f["free"].get(c) is not None and b.get(c) is not None and abs(d[c]) <= a.vram_tol for c in cards)
            extra.append(f"VRAM free within +-{a.vram_tol} MiB of {bname} on every card: "
                         f"{', '.join(f'{c} {d[c]:+d}' for c in cards)} MiB {verdict(within)}")
    elif a.vram_tol:
        extra.append(f"VRAM vs {bname}: no {bname} boot JSON FAIL")
    if a.all_resident:
        full = f["res_pct"] is not None and all(x == 100 for x in f["res_pct"])
        ok = ok and full
        why.append(f"experts 100 % resident {'yes' if full else 'NO (' + str(f['res_pct']) + ')'}")
    if res.get("rss_mib") is not None:
        brss = (bj or {}).get("rss_mib")
        drss = res["rss_mib"] - brss if brss is not None else None
        why.append(f"engine RSS {res['rss_mib']} MiB" + (f" ({drss:+d} vs {bname})" if drss is not None else ""))
        if a.rss_max:
            extra.append(f"host RAM: engine RSS after the first GEN {res['rss_mib']} MiB, "
                         + (f"{drss:+d} MiB vs {bname} (want <= +{a.rss_max}) {verdict(drss <= a.rss_max)}"
                            if drss is not None else f"no {bname} RSS INFO"))
    if a.expect in ("m1a", "m1b", "lc") and a.base:     # (no --base: this boot is the base itself)
        # the room an arm keeps must not push experts out of VRAM: its exactness drives compare with an env-off
        # reference whose experts were resident (CPU vs GPU expert math would be a difference that is not the arm's).
        # Both boots must have said '<N>% of the experts resident' and the values must be equal; no env-off boot
        # JSON (its step failed or was skipped) is a FAIL, not a pass
        rb, rg = (bj or {}).get("res_pct"), f["res_pct"]
        same = bj is not None and rb is not None and rg is not None and rb == rg
        ok = ok and same
        why.append(f"experts resident {rg}% vs {bname} {rb if bj else '(no ' + bname + ' boot JSON)'}% "
                   f"{'same' if same else 'DIFFERENT or missing'}")
    say(f"==> boot {a.label}: no loan on {f['no_loan']}/4 cards, {len(f['borrows'])} borrow lines, VRAM free with "
        f"everything loaded {'/'.join(str(x) for x in free)} MiB{delta}, LOW lines {len(f['low'])}"
        f"{'; ' + '; '.join(why) if why else ''} {verdict(ok)}")
    for x in extra:
        say(f"==> boot {a.label}: {x}")


def rss_mib(pid):
    try:
        for l in Path(f"/proc/{pid}/status").read_text().splitlines():
            if l.startswith("VmRSS:"):
                return int(l.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def mode_solo(a, cfg):
    tok = tokenizer(cfg)
    cases, lens = build_cases(tok, None)
    seq = SEQS[a.seq]
    e = start(a, cfg, a.label)
    out = {"label": a.label, "seq": seq, "lens": lens, "cases": {}, "info": e.info, "errors": []}
    marks = []
    kvres = "--kv-resident" in (cfg.get("args") or [])
    try:
        for name in seq:
            c = cases[name]
            marks.append(e.log_size())
            t0 = time.monotonic()
            try:
                r = e.solo(c["max_new"], c["ids"], keys=c["keys"], name=name)
            except V.EngineError as x:
                out["errors"].append(f"{name}: {x}")
                say(f"{name}: ERROR {x}")
                break
            f = (r.get("done") or "").split()
            dec_ms = float(f[4]) if len(f) > 4 and re.fullmatch(r"[\d.]+", f[4]) else None
            out["cases"][name] = {"tokens": r["tokens"], "prompt_tokens": len(c["ids"]), "keys": c["keys"],
                                  "resume": r.get("resume"), "reused": r.get("reused"), "done": r.get("done"),
                                  "wall_s": round(time.monotonic() - t0, 2),
                                  "dec_tps": round((len(r["tokens"]) - 1) / dec_ms * 1000, 1) if dec_ms else None}
            say(f"{name}: prompt {len(c['ids'])}, {len(r['tokens'])} tokens, reused {r.get('reused')}, "
                f"{out['cases'][name]['dec_tps']} tok/s decode, {out['cases'][name]['wall_s']} s")
            settle(e, marks[-1], kvres)
            save(a.out, out)
    finally:
        e.quit()
    kvs = kv_by_request(a.log, [(n, out["cases"][n]["prompt_tokens"]) for n in seq if n in out["cases"]])
    prev = None
    for name, seg in zip(seq, slices(e, marks)):
        if name not in out["cases"]:
            continue
        x = out["cases"][name]
        x["log"] = seg[:40]
        m = [re.search(r"request pipelined=(\d) \((.*)\)", l) for l in seg]
        m = [y for y in m if y]
        x["pipelined"] = int(m[0].group(1)) if m else None
        x["pl_reason"] = m[0].group(2) if m else None
        x["restored"] = any("conversation cache: restored" in l for l in seg)
        x["rolled_back"] = sum(int(y.group(1)) for l in seg for y in [re.search(r"(\d+) rolled back", l)] if y)
        x["spec_windows"] = sum(int(y.group(1)) for l in seg for y in [re.search(r"(\d+) speculative, ", l)] if y)
        kv = kvs.get(name)     # (the line is cumulative over the process: the delta to the previous request's value)
        x["kv_ram_mib_delta"] = None if kv is None else round(kv - (prev or 0.0), 1)
        if kv is not None:
            prev = kv
    save(a.out, out)
    n = len(out["cases"])
    pl = collections.Counter(str(x.get("pipelined")) for x in out["cases"].values())
    say(f"==> solo {a.label}: {n}/{len(seq)} requests done, pipelined lines {dict(pl)}, errors {len(out['errors'])} "
        f"{verdict(n == len(seq) and not out['errors'])}")


START_RE = re.compile(r"slot (\d+) at (\d+) \((with drafts|no drafts), drafter (\w+)\)")
RAN_RE = re.compile(r"slot (\d+): (\d+) windows, (\d+) tokens")


def parse_phases(seg):
    """The lane phases in log order: each 'lanes start:' line (slot -> [position, drafts, drafter]) with the 'lanes
    ran' line that ends it (slot -> [windows, tokens]; a slot without windows in that phase may be left out of it).
    Slot keys are strings (as they come back from the JSON)."""
    phases = []
    for l in seg:
        if "STRATA_SPEC_LANES: lanes start:" in l:
            phases.append({"start": {m.group(1): [int(m.group(2)), m.group(3), m.group(4)] for m in START_RE.finditer(l)},
                           "ran": None})
        elif "STRATA_SPEC_LANES: lanes ran" in l:
            ran = {m.group(1): [int(m.group(2)), int(m.group(3))] for m in RAN_RE.finditer(l)}
            if phases and phases[-1]["ran"] is None:
                phases[-1]["ran"] = ran
            else:                       # (a 'lanes ran' without its start line: STRATA_SPEC_LANES_LOG off?)
                phases.append({"start": {}, "ran": ran, "orphan": True})
    return phases


def lane_cover(phases, s, prompt, ntok):
    """How much of slot s's reply its lanes produced, from its admission on: k = the admission's token + every lane
    phase's tokens while each phase starts where the one before ended (no batch window in between).  The tokens[:k]
    are the lane phase (they must equal serial solo spec's); lane_only: the whole reply (k = all its tokens)."""
    ph = [p for p in phases if s in p["start"]]
    res = {"k": min(1, ntok), "first": None, "phases": len(ph), "gap": None, "sizes": [len(p["start"]) for p in ph]}
    if not ph:
        res["gap"] = "no lane start"
        res["lane_only"] = ntok <= 1
        return res
    first = ph[0]["start"][s]
    res["first"] = first
    if first[0] - prompt not in (0, 1):
        res["gap"] = f"the first lane start at {first[0]}, the prompt is {prompt}: batch windows before it"
        res["lane_only"] = ntok <= 1
        return res
    k, prev = 1, None
    for i, p in enumerate(ph):
        pos = p["start"][s][0]
        if prev is not None and pos != prev:
            res["gap"] = f"lane phase {i + 1} starts at {pos}, phase {i} ended at {prev}: batch windows in between"
            break
        if p.get("ran") is None:
            res["gap"] = f"lane phase {i + 1} has no 'lanes ran' line"
            break
        t = (p["ran"].get(s) or [0, 0])[1]
        k += t
        prev = pos + t
    res["k"] = min(k, ntok)
    res["lane_only"] = k >= ntok
    return res


def group_slots(cases, g):
    return sorted(int(k.rsplit(".", 1)[1]) for k in cases if "." in k and k.rsplit(".", 1)[0] == g)


def lane_log_settle(e, mark, timeout=3.0):
    """Until every 'lanes start' since mark has its 'lanes ran' line (the last drain writes it after the BDONE)."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        t = e.log_text(mark)
        if t.count("STRATA_SPEC_LANES: lanes start:") <= t.count("STRATA_SPEC_LANES: lanes ran"):
            break
        time.sleep(0.1)
    time.sleep(0.2)
    return [l for l in e.log_text(mark).splitlines() if LOG_KEEP.search(l)]


def lane_group(e, cases, g, pics, rec):
    """One group: (pictures first for L_vis), BGEN into each of its slots in turn, each to its BADM (HOLD=k, k = the
    group's size: the earlier slots stay untouched), then every slot to its BDONE."""
    slots = group_slots(cases, g)
    if g == "L_vis":
        if not pics:
            rec["skip"] = "no pictures (encode step failed)"
            return
        e.admit(1, cases["V_text1"]["max_new"], cases["V_text1"]["ids"], name="V_text1")
        ra = e.admit(0, 48, cases["V_imgA"]["ids"], emb=cases["V_imgA"]["emb"], name="V_imgA")
        rb = e.admit(4, 48, cases["V_imgB"]["ids"], emb=cases["V_imgB"]["emb"], name="V_imgB")
        e.wait_done(0)
        e.wait_done(4)
        if 1 in e.active:
            e.bstop(1)
        e.wait_done(1)
        rec["pictures"] = {"A": len(ra["tokens"]), "B": len(rb["tokens"])}
    mark = e.log_size()
    recs, own = {}, {}
    for s in slots:
        c = cases[f"{g}.{s}"]
        recs[s] = e.admit(s, c["max_new"], c["ids"], keys=c["keys"], name=f"{g}.{s}")
        own[s] = len(recs[s]["tokens"])
    at_last = {s: len(recs[s]["tokens"]) for s in slots}
    rec["held"] = all(at_last[s] == own[s] for s in slots)       # no BT of an earlier slot before the last BADM
    rec["tokens_at_last_badm"] = {str(s): at_last[s] for s in slots}
    for s in slots:
        e.wait_done(s)
    seg = lane_log_settle(e, mark)
    rec["phases"] = parse_phases(seg)
    rec["slots"] = {str(s): {"tokens": list(recs[s]["tokens"]), "prompt_tokens": len(cases[f"{g}.{s}"]["ids"]),
                             "keys": cases[f"{g}.{s}"]["keys"], "bdone": recs[s].get("bdone"),
                             "finish": recs[s].get("finish"), "reused": recs[s].get("reused")} for s in slots}
    for s, x in rec["slots"].items():
        x["cover"] = lane_cover(rec["phases"], s, x["prompt_tokens"], len(x["tokens"]))
    rec["lanes_ran"] = [l[l.find("lanes ran"):][:300] for l in seg if "lanes ran" in l]
    rec["ov_off"] = [l[l.find("overlapped read off"):][:200] for l in seg if "overlapped read off" in l]
    rec["ov_on"] = sum("overlapped read of" in l for l in seg)
    rec["log"] = seg[:80]


def mode_lanes(a, cfg):
    tok = tokenizer(cfg)
    pics = load_pics(a.pics)
    cases, lens = build_cases(tok, pics)
    pairs = PAIRS[a.pairs]
    e = start(a, cfg, a.label)
    out = {"label": a.label, "planned": pairs, "pairs": {}, "info": e.info, "errors": [], "lens": lens}
    try:
        say(f"[{a.label}] INFO spec_lanes={e.info.get('spec_lanes')} lane_slots={e.info.get('lane_slots')}")
        for p in pairs:
            rec = out["pairs"].setdefault(p, {})
            try:
                lane_group(e, cases, p, pics, rec)
            except (V.EngineError, RuntimeError) as x:
                out["errors"].append(f"{p}: {x}")
                rec["error"] = str(x)
                say(f"{p}: ERROR {x}")
                if not e.alive():
                    break
                try:
                    e.drain_all()       # (the next group starts with every slot idle)
                except V.EngineError:
                    break
                continue
            if "skip" in rec:
                say(f"{p}: SKIP ({rec['skip']})")
                continue
            s = rec["slots"]
            say(f"{p}: " + "; ".join(f"slot {k} {len(x['tokens'])} tokens (lane phase {x['cover']['k']}"
                                     f"{'' if x['cover']['lane_only'] else ', ' + str(x['cover']['gap'])})"
                                     for k, x in s.items())
                + f"; held {rec['held']}; lane phases {[len(ph['start']) for ph in rec['phases']]}; overlapped reads "
                  f"{rec['ov_on']}, refused {len(rec['ov_off'])}")
            for l in rec["lanes_ran"][:3]:
                say("    " + l)
            save(a.out, out)
    finally:
        save(a.out, out)
        e.quit()
    done = sum(1 for p in pairs if out["pairs"].get(p, {}).get("slots"))
    say(f"==> lanes {a.label}: {done}/{len(pairs)} groups ran, errors {len(out['errors'])} "
        f"{verdict(not out['errors'] and done == len([p for p in pairs if p != 'L_vis' or pics]))}")


def solo_logged(e, c, name, kvres):
    """A GEN with its own log slice: 'request pipelined=' and the pipeline's rollbacks."""
    mark = e.log_size()
    r = e.solo(c["max_new"], c["ids"], keys=c["keys"], name=name)
    settle(e, mark, kvres)
    seg = [l for l in e.log_text(mark).splitlines() if LOG_KEEP.search(l)]
    m = [y for y in (re.search(r"request pipelined=(\d) \((.*)\)", l) for l in seg) if y]
    return {"tokens": list(r["tokens"]), "prompt_tokens": len(c["ids"]), "keys": c["keys"], "done": r.get("done"),
            "pipelined": int(m[0].group(1)) if m else None, "pl_reason": m[0].group(2) if m else None,
            "rolled_back": sum(int(y.group(1)) for l in seg for y in [re.search(r"(\d+) rolled back", l)] if y),
            "spec_windows": sum(int(y.group(1)) for l in seg for y in [re.search(r"(\d+) speculative, ", l)] if y),
            "log": seg[:30]}


def mode_switch(a, cfg):
    """1 -> 2 -> 3 -> 4 -> 2 -> 1 users on one engine (STRATA_SPEC_LANES_HOLD=2: the first lane slot waits for the
    second, so both run in lanes from their admission on): GEN; BGEN into the lane slots one after the other, SW_STEP
    tokens per user between two admissions (each BGEN drains the running lanes, reads the prompt, and the lanes start
    again with one more); BSTOP the last ones down to two, then one (the lanes keep running: STRATA_SPEC_LANES_KEEP,
    the shared profile's default); BSTOP the last; GEN again."""
    tok = tokenizer(cfg)
    cases, lens = build_cases(tok, None)
    e = start(a, cfg, a.label)
    kvres = "--kv-resident" in (cfg.get("args") or [])
    ls = [int(x) for x in (e.info.get("lane_slots") or "").split(",") if x.strip().isdigit()]
    out = {"label": a.label, "switch": True, "info": e.info, "lane_slots": ls, "lens": lens, "solo": {}, "slots": {},
           "events": [], "errors": []}
    recs = {}

    def ev(what):
        act = sorted(s for s in e.active if s in ls)
        out["events"].append({"what": what, "active_lane_slots": act,
                              "tokens": {str(s): len(e.slot_tok.get(s, [])) for s in recs}})
        say(f"  {what}: lane slots decoding {act}, tokens so far "
            f"{ {s: len(e.slot_tok.get(s, [])) for s in recs} }")

    def more(slots, n):
        base = {s: len(e.slot_tok.get(s, [])) for s in slots}
        for s in slots:
            if s in e.active:
                e.wait_tokens(s, base[s] + n)

    def stop(s):
        if s in e.active:
            e.bstop(s)
        e.wait_done(s)

    mark = e.log_size()
    try:
        say(f"[{a.label}] INFO spec_lanes={e.info.get('spec_lanes')} lane_slots={e.info.get('lane_slots')}")
        if len(ls) < 2:
            raise RuntimeError(f"lane_slots '{e.info.get('lane_slots')}': the lanes are off")
        out["solo"]["sw_s1"] = solo_logged(e, cases["sw_s1"], "sw_s1", kvres)
        ev("GEN sw_s1 done (1 user)")
        for i, s in enumerate(ls):
            c = cases[f"sw.{s}"]
            recs[s] = e.admit(s, c["max_new"], c["ids"], keys=c["keys"], name=f"sw.{s}")
            if i == 0:
                continue                              # (held until the second one: HOLD=2)
            ev(f"BADM {s} ({i + 1} users)")
            more([x for x in recs if x in e.active], SW_STEP)
            ev(f"{SW_STEP} more tokens each")
        for s in reversed(ls[2:]):
            stop(s)
        ev("BSTOP down to two users")
        more([x for x in ls[:2] if x in e.active], SW_STEP)
        stop(ls[1])
        ev("BSTOP down to one user")
        more([ls[0]], SW_STEP)
        stop(ls[0])
        ev("BSTOP of the last user")
        e.drain_all()
        out["solo"]["sw_s2"] = solo_logged(e, cases["sw_s2"], "sw_s2", kvres)
        ev("GEN sw_s2 done (1 user)")
    except (V.EngineError, RuntimeError) as x:
        out["errors"].append(str(x))
        say(f"ERROR {x}")
    finally:
        # (a dead engine's log is read too: its last lane lines show where it stopped)
        seg = lane_log_settle(e, mark, 3.0 if e.alive() else 0.0)
        out["phases"] = parse_phases(seg)
        for s, r in recs.items():
            c = cases[f"sw.{s}"]
            x = {"tokens": list(r["tokens"]), "prompt_tokens": len(c["ids"]), "keys": c["keys"], "bdone": r.get("bdone"),
                 "finish": r.get("finish")}
            x["cover"] = lane_cover(out["phases"], str(s), x["prompt_tokens"], len(x["tokens"]))
            out["slots"][str(s)] = x
        out["log"] = seg[:200]
        save(a.out, out)
        e.quit()
    sizes = [len(p["start"]) for p in out["phases"]]
    for s, x in out["slots"].items():
        say(f"sw.{s}: {len(x['tokens'])} tokens ({x['finish']}), lane phase {x['cover']['k']}, phases {x['cover']['sizes']}"
            f"{'' if x['cover']['lane_only'] else ', ' + str(x['cover']['gap'])}")
    say(f"==> switch {a.label}: lane phases by lanes {sizes}; GEN before/after pipelined "
        f"{[out['solo'].get(n, {}).get('pipelined') for n in ('sw_s1', 'sw_s2')]}; errors {len(out['errors'])} "
        f"{verdict(not out['errors'] and len(out['solo']) == 2)}")


def first_diff(x, y):
    return next((k for k in range(min(len(x), len(y))) if x[k] != y[k]), None if len(x) == len(y) else min(len(x), len(y)))


def mode_compare(a, _cfg):
    ref = json.loads(Path(a.ref).read_text())["cases"]
    got = json.loads(Path(a.got).read_text())
    if got.get("switch"):                                       # mode_switch's run
        return compare_switch(a, ref, got)
    if "pairs" not in got:                                      # solo vs solo
        # the denominator is the planned sequence (the run's own 'seq', else SEQS by --seq), never only what ran:
        # a request the run did not reach counts as not identical / not pipelined
        planned = a.cases.split(",") if a.cases else list(got.get("seq") or SEQS[a.seq])
        missing = [n for n in planned if n not in got["cases"]]
        names = [n for n in planned if n in got["cases"]]
        for n in missing:
            say(f"{n}: NOT RUN, counted as not identical (the run ended before it: {got.get('errors')})")
        same = 0
        for n in names:
            r, g = ref.get(n, {}).get("tokens"), got["cases"][n]["tokens"]
            ok = r is not None and r == g
            same += ok
            say(f"{n}: {'IDENTICAL' if ok else 'DIFFERS at ' + str(first_diff(r or [], g))} ({len(g)} tokens, ref "
                f"{len(r or [])}; prompt {got['cases'][n]['prompt_tokens']}, keys '{got['cases'][n]['keys']}'; decode "
                f"{got['cases'][n].get('dec_tps')} vs ref {ref.get(n, {}).get('dec_tps')} tok/s; pipelined "
                f"{got['cases'][n].get('pipelined')} ({got['cases'][n].get('pl_reason')}))")
        want = len(planned)
        say(f"==> {a.what}: {same}/{want} requests token-identical to the env-off reference {verdict(same == want and want > 0)}")
        if a.need_pipelined:
            pl = [n for n in names if got["cases"][n].get("pipelined") == 1]
            bad = {n: got["cases"][n].get("pl_reason") for n in names if got["cases"][n].get("pipelined") != 1}
            bad.update({n: "not run" for n in missing})
            say(f"==> {a.what}: 'request pipelined=1' for {len(pl)}/{len(planned)} requests {bad if bad else ''} "
                f"{verdict(len(pl) == len(planned) and bool(planned))}")
        if a.need_resume:
            x = got["cases"].get("r_follow", {})
            t1 = got["cases"].get("r_turn1", {})
            # the restored follow-up itself must have streamed (turn 1 is a fresh 40K read: it streams anyway, INFO)
            ok = bool(x.get("restored")) and (x.get("reused") or 0) > 32768 and (x.get("kv_ram_mib_delta") or 0) > 0
            say(f"==> {a.what}: turn 1 of the resumed conversation (prompt {t1.get('prompt_tokens')}) read "
                f"{t1.get('kv_ram_mib_delta')} MiB K/V from RAM by its KV-streaming line INFO")
            say(f"==> {a.what}: resume above 32K from the conversation cache (prompt {x.get('prompt_tokens')}, REUSED "
                f"{x.get('reused')}, 'conversation cache: restored' {'yes' if x.get('restored') else 'NO'}, the "
                f"follow-up's own K/V read from RAM by its KV-streaming line {x.get('kv_ram_mib_delta')} MiB, want > 0) "
                f"{verdict(ok)}")
            x48 = got["cases"].get("s_48k", {})
            say(f"==> {a.what}: >= 48K prompt ({x48.get('prompt_tokens')} tokens) streamed K/V from RAM: "
                f"{x48.get('kv_ram_mib_delta')} MiB {verdict(bool(x48.get('kv_ram_mib_delta')))}")
        if a.need_rollback:
            # the 'rolled back' count includes natural misses: the forced ones must show as clearly more rollbacks
            # than the same requests had without FORCE_MISS (--base-got, the m1a_exact run)
            rb = sum(got["cases"][n].get("rolled_back") or 0 for n in names)
            spec = sum(got["cases"][n].get("spec_windows") or 0 for n in names)
            nat = None
            if a.base_got and Path(a.base_got).is_file():
                bc = json.loads(Path(a.base_got).read_text())["cases"]
                if all(n in bc for n in names):
                    nat = sum(bc[n].get("rolled_back") or 0 for n in names)
            ok = rb > 0 and nat is not None and rb > nat and rb >= spec // 3 - len(names)
            say(f"==> {a.what}: rollbacks with FORCE_MISS {rb} of {spec} speculative windows (forced every 3rd: "
                f">= ~{spec // 3}) vs {nat if nat is not None else '?'} without it (m1a_exact, same requests) "
                f"{verdict(ok)}")
        return
    # lanes vs the serial solo reference
    # the planned groups, not only those that ran: a group the run never reached counts as its slots failed
    pairs = a.cases.split(",") if a.cases else list(got.get("planned") or PAIRS[a.pairs])
    n_lane = n_full = n_all = n_start = n_held = n_only = 0
    for p in pairs:
        rec = got["pairs"].get(p, {})
        if "skip" in rec:
            say(f"{p}: SKIP {rec['skip']}")
            continue
        # (the group's slots by its name, not by what the reference or the run has: planned denominators)
        planned = ["0", "4", "8", "12"] if p.startswith("G4_") else ["0", "4", "8"] if p.startswith("G3_") else ["0", "4"]
        if not rec.get("slots"):
            say(f"{p}: no result ({rec.get('error') or 'not reached: ' + str(got.get('errors'))})")
            n_all += len(planned)
            continue
        for s in planned:
            n_all += 1
            g = rec["slots"].get(s)
            if g is None:
                say(f"{p} slot {s}: no result")
                continue
            r = ref.get(f"{p}.{s}", {}).get("tokens")
            cov = g.get("cover") or lane_cover(rec.get("phases") or [], s, g["prompt_tokens"], len(g["tokens"]))
            k = cov["k"]
            lane_ok = r is not None and k >= 2 and g["tokens"][:k] == r[:k]
            full_ok = r is not None and g["tokens"] == r
            n_lane += lane_ok
            n_full += full_ok
            n_only += bool(cov["lane_only"])
            fs = cov["first"]
            st_ok = fs is not None and fs[1] == "with drafts" and fs[2] == "level" and fs[0] - g["prompt_tokens"] in (0, 1)
            n_start += st_ok
            say(f"{p} slot {s}: lane phase {k} of {len(g['tokens'])} tokens {'IDENTICAL' if lane_ok else 'DIFFERS'}"
                f"{'' if cov['lane_only'] else ' (' + str(cov['gap']) + ')'}; full "
                f"{'IDENTICAL' if full_ok else 'differs at ' + str(first_diff(r or [], g['tokens']))} (ref {len(r or [])}); "
                f"lane phases {cov['sizes']}; first lane start {fs} for prompt {g['prompt_tokens']} "
                f"{'ok' if st_ok else 'NOT ok'}")
        n_held += bool(rec.get("held"))
        if a.ov:
            say(f"{p}: overlapped reads {rec.get('ov_on')}, refused {rec.get('ov_off')}")
    if a.full:   # (KEEP: the whole reply runs in lanes, admission to end, and equals serial solo spec)
        say(f"==> {a.what}: {n_full}/{n_all} lane slots token-identical to serial solo spec over the full length, "
            f"{n_only}/{n_all} replies wholly in lanes {verdict(n_full == n_all and n_only == n_all and n_all > 0)}")
    else:
        say(f"==> {a.what}: {n_lane}/{n_all} lane slots token-identical to serial solo spec over the lane phase "
            f"(full length {n_full}/{n_all}, wholly in lanes {n_only}/{n_all}) {verdict(n_lane == n_all and n_all > 0)}")
    say(f"==> {a.what}: lanes started at each slot's admission position with drafts and a level drafter: "
        f"{n_start}/{n_all} {verdict(n_start == n_all and n_all > 0)}")
    ran = [p for p in pairs if got["pairs"].get(p, {}).get("slots")]
    if a.hold:
        want = [p for p in pairs if "skip" not in got["pairs"].get(p, {})]
        say(f"==> {a.what}: HOLD kept the earlier slots untouched until the last slot's BADM in {n_held}/{len(want)} "
            f"groups {verdict(n_held == len(want) and bool(want))}")
        # every group's first lane phase ran all its slots at once (HOLD=k with k = the group's size)
        whole = sum(1 for p in ran if (got["pairs"][p].get("phases") or [{}])[0].get("start") is not None
                    and len(got["pairs"][p]["phases"][0]["start"]) == len(got["pairs"][p]["slots"]))
        say(f"==> {a.what}: the first lane phase ran every slot of its group in {whole}/{len(ran)} groups "
            f"{verdict(whole == len(want) and bool(want))}")
    if a.ov:   # (the engine says each refusal reason once per process)
        refused = sum(any("lane slot is held" in l for l in got["pairs"][p].get("ov_off") or []) for p in ran)
        on = sum(got["pairs"][p].get("ov_on") or 0 for p in ran)
        say(f"==> {a.what}: the overlap gate refused with 'a lane slot is held' ({refused} pair logs; said once per "
            f"process) and no read beside a held lane slot overlapped ({on} overlapped reads) "
            f"{verdict(refused >= 1 and on == 0 and bool(ran))}")


def compare_switch(a, ref, got):
    """mode_switch's run vs the serial solo reference: the two GEN identical and pipelined; every lane slot's reply
    wholly in lanes (KEEP, no batch window between two lane phases) and identical to serial solo spec - in full when it
    ended by itself, as a prefix of the reference when a BSTOP ended it; the drive reached every mode."""
    ok_solo = 0
    for n in ("sw_s1", "sw_s2"):
        g, r = got["solo"].get(n), ref.get(n, {}).get("tokens")
        same = g is not None and r is not None and g["tokens"] == r
        pl = g is not None and g.get("pipelined") == 1
        ok_solo += same and pl
        say(f"{n}: {'IDENTICAL' if same else 'DIFFERS at ' + str(first_diff(r or [], (g or {}).get('tokens') or []))} "
            f"({len((g or {}).get('tokens') or [])} tokens, ref {len(r or [])}); pipelined {(g or {}).get('pipelined')} "
            f"({(g or {}).get('pl_reason')}); rollbacks {(g or {}).get('rolled_back')}")
    say(f"==> {a.what}: the 1-user GEN before and after the lanes token-identical and pipelined: {ok_solo}/2 "
        f"{verdict(ok_solo == 2)}")
    if a.need_rollback:
        # FORCE_MISS: the two GEN's rollbacks (natural misses included) must exceed the same GEN's in the run without it
        # (--base-got: the plain switch run) - else no miss was forced and the run proves nothing about the restores
        names = ("sw_s1", "sw_s2")
        rb = sum((got["solo"].get(n) or {}).get("rolled_back") or 0 for n in names)
        spec = sum((got["solo"].get(n) or {}).get("spec_windows") or 0 for n in names)
        nat = None
        if a.base_got and Path(a.base_got).is_file():
            bs = json.loads(Path(a.base_got).read_text()).get("solo") or {}
            if all(n in bs for n in names):
                nat = sum(bs[n].get("rolled_back") or 0 for n in names)
        ok = rb > 0 and nat is not None and rb > nat
        say(f"==> {a.what}: rollbacks of the two GEN with FORCE_MISS {rb} of {spec} speculative windows vs "
            f"{nat if nat is not None else '?'} without it (--base-got) {verdict(ok)}")
    # (planned: the lane slots the engine said it runs, each with a sw.<slot> reference; none = a FAIL below)
    ls = [str(s) for s in got.get("lane_slots") or []]
    n_ok = n_only = n_start = 0
    for s in ls:
        g = got["slots"].get(s)
        r = ref.get(f"sw.{s}", {}).get("tokens")
        if g is None or r is None:
            say(f"sw.{s}: no result (ref {'yes' if r is not None else 'no'})")
            continue
        cov = g["cover"]
        t, k = g["tokens"], cov["k"]
        natural = g.get("finish") in ("stop", "length")
        same = t == r if natural else t == r[:len(t)]
        lane_ok = k >= 2 and t[:k] == r[:k]
        fs = cov["first"]
        st_ok = fs is not None and fs[1] == "with drafts" and fs[2] == "level" and fs[0] - g["prompt_tokens"] in (0, 1)
        n_ok += same and lane_ok
        n_only += bool(cov["lane_only"])
        n_start += st_ok
        say(f"sw.{s}: {len(t)} tokens ({g.get('finish')}), {'IDENTICAL' if same else 'DIFFERS at ' + str(first_diff(r, t))} "
            f"to serial solo spec ({'in full' if natural else 'as a prefix'}, ref {len(r)}); lane phase {k} "
            f"{'IDENTICAL' if lane_ok else 'DIFFERS'}{'' if cov['lane_only'] else ' (' + str(cov['gap']) + ')'}; lane phases "
            f"{cov['sizes']}; first start {fs} {'ok' if st_ok else 'NOT ok'}")
    say(f"==> {a.what}: {n_ok}/{len(ls)} lane slots token-identical to serial solo spec, {n_only}/{len(ls)} wholly in "
        f"lanes, {n_start}/{len(ls)} started at admission with drafts {verdict(n_ok == n_only == n_start == len(ls) and len(ls) >= 2)}")
    sizes = [len(p["start"]) for p in got.get("phases") or []]
    want = list(range(2, len(ls) + 1))
    downs = [x["active_lane_slots"] for x in got.get("events") or [] if x["what"].startswith("BSTOP down")]
    reached = sizes[:len(want)] == want
    say(f"==> {a.what}: lane phases by lanes {sizes} (want {want} up, then KEEP: no new start on the way down); lane "
        f"slots decoding after the BSTOPs down {downs} {verdict(reached and len(sizes) == len(want))}")
    # the way down (4 -> 2 -> 1): the users still decoding after each BSTOP (a reply that ended by itself before its
    # switch leaves that mode untested: FAIL, the replies were meant to be long enough)
    want_down = [ls[:2], ls[:1]]
    got_down = [[str(x) for x in d] for d in downs]
    say(f"==> {a.what}: the way down ran {got_down} lane slots (want {want_down}) {verdict(got_down == want_down)}")
    say(f"==> {a.what}: errors {got.get('errors')} {verdict(not got.get('errors'))}")


def mode_prefill(a, cfg):
    tok = tokenizer(cfg)
    P = V.Prompts(tok, {}, 0)
    corpus = "\n".join(p.read_text(errors="replace") for p in sorted(Path("/media/raid0/StrataQ38FN/Strata-142").rglob("*.md"))
                       if ".git" not in p.parts)
    corp = tok.encode(corpus[:1500000])
    w0 = P.encode("<|im_start|>user\n")
    w1 = P.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    sizes = [int(x) for x in a.sizes.split(",")]
    if max(sizes) + 64 > MAXCTX or max(sizes) > len(corp) // 2:
        raise SystemExit(f"sizes {sizes}: too long (context {MAXCTX}, corpus {len(corp)} tokens)")
    uid = [0]

    def prompt(n):
        uid[0] += 1
        head = tok.encode(f"[request {uid[0]} / prefill / {time.time_ns()}] Summarize the following notes:\n")
        need = n - len(w0) - len(w1) - len(head)
        off = (uid[0] * 104729) % (len(corp) - need)
        ids = w0 + head + corp[off:off + need] + w1
        assert len(ids) == n, (len(ids), n)
        return ids

    a.env = (a.env or []) + ["STRATA_TRACE=1"]
    e = start(a, cfg, a.label)
    MEAS = int(cfg.get("parallel") or 16) - 1
    rows = []
    fill_k = [0]
    fill_ids = [P.text_like(900, 20 + k, Q_LONG) for k in range(8)]

    def pump_for(sec):
        end = time.monotonic() + sec
        while True:
            left = end - time.monotonic()
            if left <= 0:
                return
            try:
                line = e.q.get(timeout=left)
            except queue.Empty:
                return
            if line is None:
                e.q.put(None)
                raise V.EngineError("the engine ended")
            if line.startswith("ERR"):
                raise V.EngineError(line)
            e._line(line)

    def refill():
        for s in range(MEAS):
            if s not in e.active:
                fill_k[0] += 1
                # an agenda to go through item by item: long replies, so the slots keep decoding through a 64K read
                # (a short TEXTS answer ends at EOS within seconds); the counts at send / BADM are recorded per row
                e.admit(s, 12000, fill_ids[fill_k[0] % len(fill_ids)], keys=f"temperature=0.7 seed={fill_k[0]}",
                        name=f"fill{s}")

    def measure(n, tag):
        ids = prompt(n)
        mark = e.log_size()
        dec_send = len([s for s in e.active if s != MEAS])
        t0 = time.monotonic()
        e.admit(MEAS, 4, ids, name=tag)
        wall = (time.monotonic() - t0) * 1000
        dec_badm = len([s for s in e.active if s != MEAS])
        if MEAS in e.active:
            e.bstop(MEAS)
        e.wait_done(MEAS)
        time.sleep(0.2)
        seg = e.log_text(mark)
        rd = [(int(m.group(1)), float(m.group(3)), m.group(2))
              for m in re.finditer(r"strata trace: read (\d+) tokens \(([^)]*)\) in ([\d.]+) ms", seg)]
        ov = [(int(m.group(1)), float(m.group(2)), "overlapped")
              for m in re.finditer(r"overlapped read of (\d+) tokens in ([\d.]+) ms", seg)]
        r = ov or rd
        return {"tag": tag, "n": n, "wall_ms": round(wall, 1), "read_ms": round(sum(x[1] for x in r), 1) if r else None,
                "read_tok": sum(x[0] for x in r) if r else None, "kind": ",".join(sorted({x[2] for x in r})),
                "dec_send": dec_send, "dec_badm": dec_badm}

    try:
        for n in (300, 1200, 150):
            measure(n, "warmup")
        for phase in a.phases.split(","):
            if phase == "dec15":
                refill()
                pump_for(5)
            for rep in range(a.reps):
                for n in sizes:
                    if phase == "dec15":
                        refill()
                        pump_for(1.0)
                    x = measure(n, f"{phase}:{n}:r{rep}")
                    x["phase"] = phase
                    rows.append(x)
                    say(f"{x['tag']}: send->BADM {x['wall_ms']} ms, read {x['read_ms']} ms of {x['read_tok']} tokens "
                        f"({x['kind']}); slots decoding at send {x['dec_send']}, at BADM {x['dec_badm']}")
                    save(a.out, rows)
            if phase == "dec15":
                for s in list(e.active):
                    e.bstop(s)
                e.drain_all()
    finally:
        save(a.out, rows)
        e.quit()
    for phase in a.phases.split(","):
        for n in sizes:
            v = [x for x in rows if x["phase"] == phase and x["n"] == n]
            if not v:
                continue
            w = st.median(x["wall_ms"] for x in v)
            rr = [x["read_ms"] for x in v if x["read_ms"]]
            r = st.median(rr) if rr else None
            ds = [x.get("dec_send") for x in v]
            db = [x.get("dec_badm") for x in v]
            say(f"==> prefill {phase:5} {n:6} tokens: read {r if r is not None else '?'} ms = "
                f"{(n / r * 1000) if r else float('nan'):7.0f} tok/s; send->BADM {w:.0f} ms = {n / w * 1000:7.0f} tok/s "
                f"({len(v)} reps, {v[0]['kind'] or 'no read line'}; slots decoding at send {ds}, at BADM {db}; "
                f"STRATA_TRACE=1 on: its syncs may lower these slightly) INFO")


def mode_sweep(a, _cfg):
    rows = {}
    for i, line in enumerate(open(a.sweep)):
        if i < a.since_line or not line.strip():
            continue                   # rows of earlier windows with the same labels do not count
        r = json.loads(line)
        rows[(r["label"], r["users"])] = r
    arms = a.arms.split(",")

    def lab(arm, k, kind):
        return f"ab-{a.tag}-{arm}" + (f"-r{k}" if a.reps > 1 else "") + f"-{kind}"
    levels = sorted({u for (l, u) in rows if l.startswith(f"ab-{a.tag}-")})
    errs = 0
    res = {}
    for kind in ("short", "long"):
        for u in levels:
            for arm in arms:
                v = [rows.get((lab(arm, k, kind), u)) for k in range(1, a.reps + 1)]
                if not all(v):
                    continue
                errs += sum(x["errors"] for x in v)
                res[(kind, u, arm)] = {"agg": st.mean(x["agg_out_tps"] for x in v),
                                       "dec": st.mean(x["dec_tps_p50"] for x in v),
                                       "ttft": st.mean(x["ttft_p50"] for x in v)}
            base = res.get((kind, u, arms[0]))
            for arm in arms[1:]:
                x = res.get((kind, u, arm))
                if not base or not x:
                    continue
                da, dd = (x["agg"] / base["agg"] - 1) * 100, (x["dec"] / base["dec"] - 1) * 100
                say(f"{kind:>5} {u:>2} users: {arms[0]} agg {base['agg']:.1f} dec p50/user {base['dec']:.1f} TTFT "
                    f"{base['ttft']:.2f} | {arm} agg {x['agg']:.1f} ({da:+.2f} %) dec p50/user {x['dec']:.1f} ({dd:+.2f} %) "
                    f"TTFT {x['ttft']:.2f}")
                if u == a.target_users and a.target_dec:
                    say(f"==> {a.tag} {kind} {u} users: decode p50 per user {base['dec']:.1f} -> {x['dec']:.1f} tok/s "
                        f"(target >= {a.target_dec:.0f}) {verdict(x['dec'] >= a.target_dec)}")
                if a.gain_users and u == a.gain_users:
                    say(f"==> {a.tag} {kind} {u} user(s): agg {da:+.2f} % (want >= +{a.gain_min:.1f} %) "
                        f"{verdict(da >= a.gain_min)}")
                lim = dict((int(k), float(v)) for k, v in (x.split(":") for x in a.dec_min.split(",") if x)) \
                    if a.dec_min else {}
                if u in lim:   # per-user decode p50 (window-lanes4.sh): no loss at 1-2 and 5+ users, a gain at 3-4
                    say(f"==> {a.tag} {kind} {u} users: decode p50 per user {base['dec']:.1f} -> {x['dec']:.1f} tok/s "
                        f"({dd:+.2f} %, want >= {lim[u]:+.1f} %); agg {da:+.2f} % {verdict(dd >= lim[u])}")
                if a.flat_users and str(u) in a.flat_users.split(","):
                    say(f"==> {a.tag} {kind} {u} users: agg {da:+.2f} %, dec p50 {dd:+.2f} % (no loss beyond "
                        f"-{a.flat_tol:.1f} %) {verdict(da >= -a.flat_tol)}")
    if a.dec_min:      # (planned levels: a level of --dec-min without rows for both arms is a FAIL, not a pass)
        for kind in ("short", "long"):
            for u in sorted(int(x.split(":")[0]) for x in a.dec_min.split(",") if x):
                miss = [arm for arm in arms if (kind, u, arm) not in res]
                if miss:
                    say(f"==> {a.tag} {kind} {u} users: no rows for {miss} (all {a.reps} repetitions needed) FAIL")
    say(f"==> {a.tag}: request errors over all arms and levels: {errs} {verdict(errs == 0 and bool(res))}")


def mode_abcheck(a, _cfg):
    lines = [l.rstrip() for l in open(a.file) if " vs " in l and "users" in l]
    if not lines:
        say(f"==> {a.what}: no comparison lines in {a.file} FAIL")
        return
    for l in lines:
        br = re.findall(r"\[r\d+: ([^\]]*)\]", l)
        # every bracket must say identical N > 0, different 0, and neither errored uids nor uids in one arm only
        # (compare_replies.py leaves errored uids out of both counts and does not compare the one-arm ones); a
        # bracket that is not closed on this line (compare_replies said more than one line) counts as failed
        opened = len(re.findall(r"\[r\d+:", l))
        ok = bool(br) and len(br) == opened
        for b in br:
            ident = re.search(r"identical (\d+)", b)
            diff = re.search(r"different (\d+)", b)
            errd = re.search(r"errored (\d+)", b)
            only = re.findall(r"only in [AB] (\d+)", b)
            ok = ok and bool(ident) and int(ident.group(1)) > 0 and bool(diff) and int(diff.group(1)) == 0 \
                and not (errd and int(errd.group(1)) > 0) and not any(int(x) > 0 for x in only)
        head = " ".join(l.split()[:4])
        strict = a.expect == "identical" and (not a.strict_users or l.split()[1] in a.strict_users.split(","))
        say(f"==> {a.what} {head}: replies {'; '.join(br)}{'' if len(br) == opened else f' ({opened - len(br)} unreadable brackets)'} "
            + (verdict(ok) if strict else "INFO"))


def mode_lanelog(a, _cfg):
    files = [f for p in a.files for f in glob.glob(p)]
    starts = ran = wd = err = 0
    tps = []
    sizes = collections.Counter()
    for f in files:
        for l in open(f, errors="replace"):
            starts += "STRATA_SPEC_LANES: lanes start" in l
            if "STRATA_SPEC_LANES: lanes start:" in l:
                sizes[len(START_RE.findall(l))] += 1
            if "STRATA_SPEC_LANES: lanes ran" in l:
                ran += 1
                tps += [float(x) for x in re.findall(r"\(([\d.]+) tok/s\)", l)]
            wd += "no progress for" in l
            err += l.startswith("ERR") or " ERR " in l
    med = f"{st.median(tps):.1f}" if tps else "n/a"
    if a.expect == "none":
        ok = starts == 0 and wd == 0 and err == 0
    elif a.expect == "any":    # (lane starts counted, not wanted either way: no watchdog, no ERR)
        ok = wd == 0 and err == 0
    else:
        ok = starts > 0 and ran > 0 and wd == 0 and err == 0
    say(f"==> {a.what}: {len(files)} logs, {starts} 'lanes start', {ran} 'lanes ran' (per-slot lane tok/s median {med}), "
        f"{wd} watchdog, {err} ERR (want {'no lane start' if a.expect == 'none' else 'no watchdog, no ERR' if a.expect == 'any' else 'lane starts'}) "
        f"{verdict(ok)}")
    if a.want_sizes:   # (STRATA_SPEC_LANES_LOG=1: the slots per 'lanes start' line)
        want = [int(x) for x in a.want_sizes.split(",") if x]
        say(f"==> {a.what}: lane starts by lanes {dict(sorted(sizes.items()))} (want each of {want}) "
            f"{verdict(all(sizes.get(w, 0) > 0 for w in want))}")


def mode_pllog(a, _cfg):
    files = [f for p in a.files for f in glob.glob(p)]
    c = collections.Counter()
    for f in files:
        for l in open(f, errors="replace"):
            m = re.search(r"request pipelined=(\d) \((.*)\)", l)
            if m:
                c[(m.group(1), m.group(2) if m.group(1) == "0" else "")] += 1
    p1 = sum(v for (k, _), v in c.items() if k == "1")
    p0 = {r: v for (k, r), v in c.items() if k == "0"}
    say(f"==> {a.what}: {len(files)} logs, pipelined=1 {p1}, pipelined=0 {sum(p0.values())} {p0} {verdict(p1 > 0)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["encode", "boot", "solo", "lanes", "switch", "compare", "prefill",
                                                      "sweep", "abcheck", "lanelog", "pllog"])
    ap.add_argument("--config")
    ap.add_argument("--exe", default="")
    ap.add_argument("--label", default="run")
    ap.add_argument("--log", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--env", action="append", default=[], help="K=V for the engine (beyond the config's env)")
    ap.add_argument("--drop-env", action="append", default=[], help="a config env key left out")
    ap.add_argument("--extra", default="", help="more engine arguments in one string (e.g. the exactness settings)")
    ap.add_argument("--line-timeout", type=float, default=300.0)
    ap.add_argument("--expect", default="off")
    ap.add_argument("--base", default="")
    ap.add_argument("--seq", default="ref", choices=list(SEQS))
    ap.add_argument("--pairs", default="all", choices=list(PAIRS))
    ap.add_argument("--pics", default="")
    ap.add_argument("--ref", default="")
    ap.add_argument("--got", default="")
    ap.add_argument("--base-got", default="", help="compare --need-rollback: the same requests without FORCE_MISS")
    ap.add_argument("--cases", default="")
    ap.add_argument("--what", default="compare")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--hold", action="store_true")
    ap.add_argument("--ov", action="store_true")
    ap.add_argument("--need-pipelined", action="store_true")
    ap.add_argument("--need-resume", action="store_true")
    ap.add_argument("--need-rollback", action="store_true")
    ap.add_argument("--sizes", default="170,512,1200,4000,16000,64000")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--phases", default="idle,dec15")
    ap.add_argument("--sweep", default="/media/raid0/StrataQ38FN/linux/results/sweep.jsonl")
    ap.add_argument("--since-line", type=int, default=0, help="sweep: ignore the first N lines of --sweep")
    ap.add_argument("--tag", default="")
    ap.add_argument("--arms", default="")
    ap.add_argument("--target-users", type=int, default=0)
    ap.add_argument("--target-dec", type=float, default=0.0)
    ap.add_argument("--gain-users", type=int, default=0)
    ap.add_argument("--gain-min", type=float, default=5.0)
    ap.add_argument("--flat-users", default="")
    ap.add_argument("--flat-tol", type=float, default=1.5)
    ap.add_argument("--dec-min", default="", help="sweep: 'users:min %%,...' per-user decode p50 delta wanted")
    ap.add_argument("--want-sizes", default="", help="lanelog: lane starts with each of these lane counts")
    ap.add_argument("--want-lanes", type=int, default=0, help="boot lc: at least this many lanes")
    ap.add_argument("--max-lanes", type=int, default=0, help="boot lc: at most this many lanes")
    ap.add_argument("--want-pipelined", action="store_true", help="boot: the first GEN pipelined=1")
    ap.add_argument("--all-resident", action="store_true", help="boot: every '%% of the experts resident' line says 100")
    ap.add_argument("--vram-tol", type=int, default=0, help="boot: VRAM free within this many MiB of --base per card")
    ap.add_argument("--base-name", default="", help="boot: what --base is (for the verdict lines)")
    ap.add_argument("--rss-max", type=int, default=0, help="boot: engine RSS growth vs --base at most this many MiB")
    ap.add_argument("--file", default="")
    ap.add_argument("--strict-users", default="", help="abcheck: judge only these levels (others INFO)")
    ap.add_argument("files", nargs="*")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text()) if a.config else {}
    if a.mode in ("boot", "solo", "lanes", "switch", "prefill") and not (a.config and a.log and a.out):
        raise SystemExit("--config, --log and --out are needed")
    try:
        {"encode": mode_encode, "boot": mode_boot, "solo": mode_solo, "lanes": mode_lanes, "switch": mode_switch,
         "compare": mode_compare,
         "prefill": mode_prefill, "sweep": mode_sweep, "abcheck": mode_abcheck, "lanelog": mode_lanelog,
         "pllog": mode_pllog}[a.mode](a, cfg)
    except V.EngineError as x:
        say(f"==> {a.mode} {a.label}: the engine failed: {x} FAIL")
        return 1
    finally:
        for e in list(V.LIVE):
            try:
                e.quit()
            except Exception:  # noqa: BLE001
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
