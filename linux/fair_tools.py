#!/usr/bin/env python3
"""Helpers of the GPU window window-fair.sh (branch ll6-reads: stage 1 FAIR - a prompt read gives way at a chunk boundary
to short requests, engine STRATA_READ_YIELD=1 + server STRATA_YIELD_SHORT=N).  Verdict lines start with '==> ' and end in
PASS, FAIL or INFO (the window copies them into its window.log).  Only the mode `scen` talks to a server (the test server
on 127.0.0.1:8091 the window started); every other mode reads files.

Modes:
  tokcheck  every request of the scenarios measured with the model's tokenizer (the same builder `scen` uses): each
            request incl. its output (and every second turn incl. its history) < 262,144 tokens; the win scenario's
            long reads 100K-150K; the exactness scenario's long read > 90K; the short requests <= --short new tokens
  scen      run one scenario against --url and write every reply, its send / first-token / done times and its
            prompt length to --out (JSON).  exact: four rounds (a lone 48K read + a short; a 100K read beside 4
            decoding streams + 3 shorts, one at T=0.7 seeded; a 40K read at T=0.7 seeded beside 2 streams + a seeded
            short; a 64K read + a burst of 8 shorts), each long read's and some shorts' second turn (a resume from
            the slot / conversation cache: a slot state that differs gives another reply).  win (--rep k): w1 a cold
            ~125K read beside 2 decoding streams, 4 shorts arriving at +3/6/9/12 s; w2 a cold ~105K read, a burst of
            16 shorts at +4 s.  Prompts depend only on the scenario, the round and --rep: every arm gets the same ones
  cmp       two scen JSONs: every reply identical (text and error-free) -> PASS / FAIL
  win       the win scenario's TTFTs per arm (--arms LABEL=GLOB of scen JSONs): shorts' TTFT p50 / max, the burst's
            time to all first tokens, p50 / p95, the long read's TTFT and total; deltas vs the first arm (judged: w1
            every short's TTFT = its max, w2 the time to all 16 first tokens, each <= --gain x the first arm's)
  yieldstat engine logs (--arms LABEL=GLOB): 'gives way' lines (q, copy ms), 'yslot=... holds no read' (an unpinned
            resume), SLOT_PARK_VERIFY identical / differ, ERR, watchdog, engine starts; --want-yield / --want-none /
            --want-clean / --test-at LABEL:k
  absum     ab_quick.sh's summary for the window's own ABBA runner (each arm with its own server checkout): lines in
            ab_quick's format, so lanes_stdin.py --mode abcheck reads them
  readstat  (window-reads2.sh, stage 2 STRATA_READS) engine logs (--arms LABEL=GLOB): reads handed to the background
            (at, n, part tokens), ready / dropped (cancel, err), runs beside the groups vs drained, paused for a line,
            'bg=... not taken' reasons, unpinned finishes, park verify, ERR, watchdog, the pad-row guard;
            --want-bg / --want-none / --want-clean
Scenario `two` (window-reads2.sh, --rep k): t1 a cold ~90K read beside 2 decoding streams and a second cold ~40K read at
+2 s, then 3 shorts; t2 a cold ~60K read with ~50K and ~30K ones at +1 / +2 s beside 3 streams, then 2 shorts - reads in
the background at once (STRATA_READS=2: the third reads on the control lines); its "shorts" include those long ones
(tokcheck allows them up to 60,000 tokens)
Scenario `solo` (--rep k): o1 a lone cold ~100K conversation (384 tokens) and two follow-up turns (256 each: a solo
conversation's follow-ups, STRATA_READS_SOLO); o2 a cold ~100K read beside 2 decoding streams (a background read with
STRATA_SERVER_READS) and its follow-up.  dec: the long replies' decode tok/s and draft acceptance per arm (the server's
timings), the streams' longest stall (gap_max: a drained background read stalls them).
Scenario `vis` (rep 0, tokcheck --two): pictures beside background reads (v1: 2 pictures and a picture + 20K text arriving
during a cold 60K text read beside 2 streams, one seeded T=0.7) and a 30K text + picture long read with a 20K text
request beside it (v2), follow-up turns; --scenario takes a comma list (exact,vis: one run, one JSON).
mix: bench_multiturn / bench_agentic rows of the window's ABBA agent mix (--dir, --arms-list, --reps, --gain): reply
identity per repetition, follow-up and per-role TTFT vs the first arm.
"""
import argparse, glob, json, os, re, statistics as st, subprocess, sys, threading, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
MAXCTX = 262144
OVER = 24                 # per message: the chat template's role / start / end tokens (generous)
G = {"temperature": 0}


def say(m):
    print(m, flush=True)


def verdict(ok):
    return "PASS" if ok else "FAIL"


def pct(v, p):
    v = sorted(x for x in v if x is not None)
    if not v:
        return None
    k = (len(v) - 1) * p / 100.0
    i = int(k)
    return v[i] if i + 1 >= len(v) else v[i] + (v[i + 1] - v[i]) * (k - i)


def f2(x):
    return "n/a" if x is None else f"{x:.2f}"


# ------------------------------------------------------------------------------------------------------ the prompts
def t07(seed):
    return {"temperature": 0.7, "seed": seed}


def scenario(name, rep):
    """Rounds: bg (count, doc tokens, reply tokens), long (seed, tokens, max_tok, sampling), shorts [(offset s, seed,
    tokens, max_tok, sampling)], burst (offset s or None: the shorts all at once), turn2 [(key, max_tok, sampling)].
    The background replies are long enough (~100 tok/s per stream with 2-4 users) to keep decoding through most of the
    long read, so the read and its stops happen beside decode windows (the overlapped read path), not only at its start."""
    if name == "exact":
        return [
            {"r": "x0", "bg": (0, 0, 0), "long": (101, 48000, 48, G), "shorts": [(2.0, 111, 400, 48, G)],
             "turn2": [("long", 48, G), ("s0", 32, G)]},
            {"r": "x1", "bg": (4, 3000, 1600), "long": (201, 100000, 64, G),
             "shorts": [(3.0, 211, 300, 48, G), (5.0, 212, 1500, 48, G), (7.0, 213, 800, 48, t07(11))],
             "turn2": [("long", 48, G), ("s1", 32, G)]},
            {"r": "x2", "bg": (2, 3000, 1000), "long": (301, 40000, 64, t07(1234)), "shorts": [(1.5, 311, 500, 48, t07(5))],
             "turn2": [("long", 48, t07(99))]},
            {"r": "x3", "bg": (0, 0, 0), "long": (401, 64000, 48, G),
             "shorts": [(2.0, 411 + i, 200 + 50 * i, 24, G) for i in range(8)], "burst": 2.0,
             "turn2": [("long", 32, G)]},
        ]
    if name == "two":
        b = 5000 + 1000 * rep
        return [
            {"r": "t1", "bg": (2, 3000, 2400), "long": (b + 1, 90000, 32, G),
             "shorts": [(2.0, b + 21, 40000, 32, G), (3.5, b + 22, 300, 32, G), (5.0, b + 23, 1200, 32, G),
                        (6.5, b + 24, 600, 32, t07(7))],
             "turn2": [("long", 32, G), ("s0", 32, G)]},
            {"r": "t2", "bg": (3, 3000, 2400), "long": (b + 2, 60000, 32, G),
             "shorts": [(1.0, b + 31, 50000, 32, G), (2.0, b + 32, 30000, 32, t07(3)), (4.0, b + 33, 400, 32, G),
                        (5.0, b + 34, 800, 32, G)],
             "turn2": [("s0", 32, G)]},
        ]
    if name == "solo":
        b = 8000 + 100 * rep
        return [
            {"r": "o1", "bg": (0, 0, 0), "long": (b + 1, 100000, 384, G), "shorts": [],
             "turn2": [("long", 256, G), ("long", 256, G)]},
            {"r": "o2", "bg": (2, 3000, 2400), "long": (b + 2, 100000, 384, G), "shorts": [],
             "turn2": [("long", 256, G)]},
        ]
    if name == "vis":
        # pictures beside background reads (window-reads2.sh): a picture request is never read in the background (the
        # server keeps it out, the engine refuses bg= with images); it must still give the same reply when it arrives
        # during a background read (v1: two pictures and a picture + 20K text at +2/3.5/5 s, one seeded T=0.7, beside a
        # cold 60K text read and 2 streams) and when it is the long read itself (v2: 30K text + a picture, read
        # serially, with a 20K text request at +2 s that may go to the background meanwhile); follow-up turns of both
        return [
            {"r": "v1", "bg": (2, 3000, 2400), "long": (9101, 60000, 48, G), "shorts": [],
             "pics": [(2.0, 9111, 0, "red-square", 48, G), (3.5, 9112, 0, "blue-circle", 48, t07(21)),
                      (5.0, 9113, 20000, "text:STRATA 4711", 48, G)],
             "turn2": [("long", 48, G), ("p0", 32, G), ("p2", 32, G)]},
            {"r": "v2", "bg": (2, 3000, 2400), "long": (9201, 30000, 48, G, "green-triangle"),
             "shorts": [(2.0, 9211, 20000, 48, G), (3.0, 9212, 400, 48, t07(8))],
             "turn2": [("long", 48, G), ("s0", 32, G)]},
        ]
    if name == "win":
        b = 1000 * rep
        return [
            {"r": "w1", "bg": (2, 3000, 2400), "long": (b + 1, 125000, 32, G),
             "shorts": [(3.0 * (i + 1), b + 11 + i, t, 32, G) for i, t in enumerate((200, 600, 1200, 1800))],
             "turn2": []},
            {"r": "w2", "bg": (0, 0, 0), "long": (b + 2, 105000, 32, G),
             "shorts": [(4.0, b + 31 + i, 150 + 20 * i, 16, G) for i in range(16)], "burst": 4.0, "turn2": []},
        ]
    raise SystemExit(f"unknown scenario {name}")


class Docs:
    """Deterministic documents of a measured length (ovkv_ident.doc's words, scaled to the target in two passes)."""
    def __init__(self, tok):
        import ovkv_ident as OK
        self.OK, self.tok = OK, tok

    def n(self, s):
        return len(self.tok.encode(s))

    def doc(self, seed, target):
        n = max(8, int(target / 1.56))
        for _ in range(4):
            d = self.OK.doc(seed, n)
            k = self.n(d)
            if abs(k - target) <= max(16, target // 100):
                break
            n = max(8, int(n * target / k))
        return d, k


def msgs_long(D, seed, tokens, r):
    d, _ = D.doc(seed, tokens)
    return [{"role": "user", "content": f"Report {r}-{seed}:\n{d}\n\nTask: name the three most frequent words and "
                                         f"summarise the report."}]


def msgs_short(D, seed, tokens, r):
    d, _ = D.doc(seed, tokens)
    return [{"role": "user", "content": f"Note {r}-{seed}:\n{d}\n\nQuestion: which word comes first, and what is the "
                                         f"note about?"}]


IMG_TOK = 1600          # tokcheck: what one 800x600 test picture may cost at most (vision tokens + markers, generous)
_PICS = {}


def pic(kind):
    if kind not in _PICS:
        import vision_check as VC      # (PIL; the same pictures as vision_check.py)
        _PICS[kind] = VC.picture(kind)
    return _PICS[kind]


def with_pic(text, kind):
    return [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": pic(kind)}}]


def msgs_long_any(D, long, r):
    """The round's long request; a 5th element: a picture with it (read serially: never in the background)."""
    m = msgs_long(D, long[0], long[1], r)
    if len(long) > 4:
        m[0]["content"] = with_pic(m[0]["content"] + " Also say what the picture shows.", long[4])
    return m


def msgs_pic(D, seed, tokens, kind, r):
    import vision_check as VC
    q = VC.Q_TEXT if kind.startswith("text:") else VC.Q_SHAPE
    head = f"Note {r}-{seed}:\n{D.doc(seed, tokens)[0]}\n\n" if tokens else f"Picture {r}-{seed}. "
    return [{"role": "user", "content": with_pic(head + q, kind)}]


def msgs_bg(D, u, tokens, r):
    d, _ = D.doc(700 + u, tokens)
    return [{"role": "user", "content": f"Document {u}:\n{d}\n\nQuestion {r}: write a long, detailed story that uses the "
                                         f"words of this document."}]


Q2 = "Now name the least frequent word, and why it matters."
Q3 = "Which sentence of the report would you cut first, and why?"


def plen(D, msgs):
    def n(c):
        if isinstance(c, str):
            return D.n(c)
        return sum(D.n(p["text"]) if p.get("type") == "text" else IMG_TOK for p in c)
    return sum(n(m["content"]) + OVER for m in msgs)


# -------------------------------------------------------------------------------------------------------- HTTP
def chat(a, messages, max_tok, sampling, rec):
    body = {"model": a.model, "messages": messages, "max_tokens": max_tok, "stream": True, "reasoning_effort": "none"}
    body.update(sampling)
    req = urllib.request.Request(a.url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    text, err = [], None
    rec["t_send"] = time.perf_counter()
    rec["t_first"] = None
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                c = json.loads(line[5:])
                if c.get("timings"):
                    rec["timings"] = c["timings"]
                for ch in c.get("choices") or []:
                    d = (ch.get("delta") or {}).get("content")
                    if d:
                        now = time.perf_counter()
                        if rec["t_first"] is None:
                            rec["t_first"] = now
                        else:      # the longest wait between two pieces of this reply (a stall of its stream)
                            rec["gap_max"] = max(rec.get("gap_max", 0.0), now - rec["t_last"])
                        rec["t_last"] = now
                        text.append(d)
                        rec["deltas"] = rec.get("deltas", 0) + 1
    except Exception as e:  # noqa: BLE001 - recorded: an arm with an error never matches
        err = repr(e)[:300]
    rec["t_done"] = time.perf_counter()
    rec["text"] = "".join(text)
    rec["error"] = err


def run_round(a, D, rnd, res):
    r = rnd["r"]
    ths = []

    def go(key, msgs, max_tok, sampling):
        rec = {"prompt_tokens": plen(D, msgs), "max_tok": max_tok, "sampling": sampling, "msgs_n": len(msgs)}
        res[key] = rec
        th = threading.Thread(target=chat, args=(a, msgs, max_tok, sampling, rec), daemon=True)
        th.start()
        return th
    nb, bt, bo = rnd["bg"]
    bg = [go(f"{r}-bg{u}", msgs_bg(D, u, bt, r), bo, G) for u in range(nb)]
    t_s = time.perf_counter()
    while any(res[f"{r}-bg{u}"].get("deltas", 0) < 16 and bg[u].is_alive() for u in range(nb)) \
            and time.perf_counter() - t_s < 600:
        time.sleep(0.05)
    lm, lsamp = rnd["long"][2], rnd["long"][3]
    lmsgs = msgs_long_any(D, rnd["long"], r)
    t_l = time.perf_counter()
    tl = go(f"{r}-long", lmsgs, lm, lsamp)
    smsgs = {}
    sh = []
    ev = [(off, f"s{i}", (lambda seed=seed, tok=tok: msgs_short(D, seed, tok, r)), mt, samp)
          for i, (off, seed, tok, mt, samp) in enumerate(rnd["shorts"])]
    ev += [(off, f"p{i}", (lambda seed=seed, tok=tok, kind=kind: msgs_pic(D, seed, tok, kind, r)), mt, samp)
           for i, (off, seed, tok, kind, mt, samp) in enumerate(rnd.get("pics", []))]
    for off, key, build, mt, samp in sorted(ev, key=lambda e: e[0]):
        m = build()
        while time.perf_counter() < t_l + off:
            time.sleep(0.01)
        smsgs[key] = m
        sh.append(go(f"{r}-{key}", m, mt, samp))
    for th in [tl] + sh + bg:
        th.join()
    res[f"{r}-meta"] = {"t_long_send": t_l, "burst": rnd.get("burst"),
                        "t_burst": (t_l + rnd["burst"]) if rnd.get("burst") is not None else None}
    conv = {}                              # key -> (its messages so far, its last reply's result key, turns)
    for key, mt, samp in rnd["turn2"]:     # one after the other: each resumes its conversation (a key again: a 3rd turn)
        m1, last, n = conv.get(key) or ((lmsgs if key == "long" else smsgs[key]), f"{r}-{key}", 1)
        prev = res[last]
        m2 = m1 + [{"role": "assistant", "content": prev["text"]}, {"role": "user", "content": Q2 if n == 1 else Q3}]
        rk = f"{r}-{key}-turn{n + 1}"
        go(rk, m2, mt, samp).join()
        conv[key] = (m2, rk, n + 1)
    bad = [k for k, v in res.items() if k.startswith(f"{r}-") and isinstance(v, dict) and v.get("error")]
    say(f"round {r}: long {res[f'{r}-long']['prompt_tokens']} tok TTFT "
        f"{res[f'{r}-long']['t_first'] - res[f'{r}-long']['t_send'] if res[f'{r}-long']['t_first'] else float('nan'):.2f} s; "
        f"shorts TTFT {[round(res[k]['t_first'] - res[k]['t_send'], 2) if res[k]['t_first'] else None for k in [f'{r}-s{i}' for i in range(len(rnd['shorts']))] + [f'{r}-p{i}' for i in range(len(rnd.get('pics', [])))]]}"
        + (f"; errors {bad}" if bad else ""))


def tokenizer(config):
    import lanes_stdin as LS
    return LS.tokenizer(json.load(open(config)))


def mode_scen(a):
    D = Docs(tokenizer(a.config))
    res = {"_scenario": a.scenario, "_rep": a.rep, "_url": a.url}
    t0 = time.perf_counter()
    for rnd in [x for name in a.scenario.split(",") for x in scenario(name, a.rep)]:
        if a.rounds and rnd["r"] not in a.rounds.split(","):
            continue
        run_round(a, D, rnd, res)
    res["_wall_s"] = round(time.perf_counter() - t0, 1)
    json.dump(res, open(a.out, "w"), indent=1)
    errs = [k for k, v in res.items() if isinstance(v, dict) and v.get("error")]
    say(f"==> scen {a.scenario} rep {a.rep} ({a.label}): {sum(1 for v in res.values() if isinstance(v, dict) and 'text' in v)} "
        f"replies in {res['_wall_s']} s, errors {errs[:4]} {verdict(not errs)}")
    return 0


def mode_tokcheck(a):
    D = Docs(tokenizer(a.config))
    bad = 0
    lists = [("exact", [0]), ("win", [int(x) for x in a.reps_list.split(",")])]
    if a.two:
        lists.append(("two", [int(x) for x in a.reps_list.split(",")]))
        lists.append(("solo", [int(x) for x in a.reps_list.split(",")]))
        lists.append(("vis", [0]))
    for name, reps in lists:
        for rep in reps:
            for rnd in scenario(name, rep):
                r = rnd["r"]
                lm = rnd["long"][2]
                lmsgs = msgs_long_any(D, rnd["long"], r)
                L = plen(D, lmsgs)
                worst = L + lm
                hist = L + lm
                for key, mt, _ in rnd["turn2"]:      # history + the earlier replies (<= their max_tok) + the questions
                    if key == "long":
                        hist += OVER + max(D.n(Q2), D.n(Q3)) + OVER + mt
                        worst = max(worst, hist)
                sl = []
                for i, (_, seed, tok, mt, _) in enumerate(rnd["shorts"]):
                    s = plen(D, msgs_short(D, seed, tok, r))
                    sl.append(s)
                    worst = max(worst, s + mt + 2 * OVER + D.n(Q2) + 64)
                for i, (_, seed, tok, kind, mt, _) in enumerate(rnd.get("pics", [])):
                    s = plen(D, msgs_pic(D, seed, tok, kind, r))
                    sl.append(s)
                    worst = max(worst, s + mt + 2 * OVER + D.n(Q2) + 64)
                nb, bt, bo = rnd["bg"]
                bgl = [plen(D, msgs_bg(D, u, bt, r)) + bo for u in range(nb)]
                worst = max([worst] + bgl)
                ok = worst + 64 <= MAXCTX and (max(sl) if sl else 0) <= (60000 if name in ("two", "vis") else a.short)
                want = ""
                if name == "win":
                    ok = ok and 100000 <= L <= 150000
                    want = ", long read within 100K-150K"
                if r == "x1":
                    ok = ok and L >= 90000
                    want = ", long read >= 90K"
                bad += not ok
                say(f"==> tokcheck {name} rep {rep} {r}: long {L} tok, shorts {min(sl) if sl else 0}-{max(sl) if sl else 0} "
                    f"tok (<= {60000 if name in ('two', 'vis') else a.short}), bg {bgl[:1] or 0}; longest request incl. output {worst} (+64 <= {MAXCTX}{want}) "
                    f"{verdict(ok)}")
    return 1 if bad else 0


def mode_cmp(a):
    A, B = json.load(open(a.a)), json.load(open(a.b))
    keys = sorted(k for k in set(A) | set(B) if not k.startswith("_") and not k.endswith("-meta"))
    if a.rounds:          # only these rounds' replies (an arm that ran some rounds of the scenario)
        keys = [k for k in keys if k.split("-", 1)[0] in a.rounds.split(",")]
    bad = []
    for k in keys:
        x, y = A.get(k), B.get(k)
        if not x or not y or x.get("error") or y.get("error") or x.get("text") != y.get("text"):
            bad.append(k)
            tx, ty = (x or {}).get("text", ""), (y or {}).get("text", "")
            n = next((i for i, (p, q) in enumerate(zip(tx, ty)) if p != q), min(len(tx), len(ty)))
            say(f"    {k}: differs at char {n} of {len(tx)}/{len(ty)} (errors: {(x or {}).get('error')} / "
                f"{(y or {}).get('error')})")
    say(f"==> {a.what}: {len(keys) - len(bad)}/{len(keys)} replies identical {verdict(bool(keys) and not bad)}")
    return 0


def win_facts(d):
    out = {}
    for r in ("w1", "w2"):
        L = d.get(f"{r}-long")
        if not L:
            continue
        sh = [v for k, v in d.items() if k.startswith(f"{r}-s") and "-turn2" not in k]
        tt = [v["t_first"] - v["t_send"] if v.get("t_first") else None for v in sh]
        m = d.get(f"{r}-meta") or {}
        f = {"long_ttft": (L["t_first"] - L["t_send"]) if L.get("t_first") else None,
             "long_total": L["t_done"] - L["t_send"], "s_p50": pct(tt, 50), "s_p95": pct(tt, 95),
             "s_max": max([x for x in tt if x is not None], default=None), "s_miss": sum(x is None for x in tt),
             "long_tok": L.get("prompt_tokens")}
        if m.get("t_burst") is not None:
            firsts = [v.get("t_first") for v in sh]
            f["all_first"] = (max(firsts) - m["t_burst"]) if firsts and all(firsts) else None
        out[r] = f
    return out


def mode_win(a):
    arms = []
    for spec in a.arms:
        label, pat = spec.split("=", 1)
        files = sorted(glob.glob(pat))
        if not files:
            say(f"    win {label}: no scenario files {pat} (skipped) INFO")
            continue
        facts = []
        for fp in files:
            try:
                facts.append(win_facts(json.load(open(fp))))
            except Exception as e:  # noqa: BLE001
                say(f"==> win {label}: {fp} unreadable ({e}) FAIL")
        arms.append((label, facts))
    if not arms:
        say("==> win: no arm ran FAIL")
        return 0
    agg = {}
    for label, facts in arms:
        for r in ("w1", "w2"):
            v = [f[r] for f in facts if r in f]
            if not v:
                continue
            g = {k: st.median([x[k] for x in v if x.get(k) is not None]) if any(x.get(k) is not None for x in v) else None
                 for k in ("long_ttft", "long_total", "s_p50", "s_p95", "s_max", "all_first", "long_tok")}
            g["n"], g["miss"] = len(v), sum(x["s_miss"] for x in v)
            agg[(label, r)] = g
            what = ("w1: a cold ~125K read beside 2 decoding streams, 4 shorts at +3/6/9/12 s" if r == "w1" else
                    "w2: a cold ~105K read, 16 shorts at once at +4 s")
            say(f"==> win {label} {what}: shorts TTFT p50 {f2(g['s_p50'])} p95 {f2(g['s_p95'])} max {f2(g['s_max'])} s"
                + (f", all 16 first tokens after {f2(g['all_first'])} s" if r == "w2" else "")
                + f"; long read ({g['long_tok']} tok) TTFT {f2(g['long_ttft'])} s, total {f2(g['long_total'])} s "
                f"(median of {g['n']} runs; {g['miss']} shorts without a token) INFO")
    base = arms[0][0]
    for label, _ in arms[1:]:
        for r in ("w1", "w2"):
            b, x = agg.get((base, r)), agg.get((label, r))
            if not b or not x:
                continue
            # w1: every short (its max): the p50 of 4 shorts passes when 2 of them still wait out the read (a server
            # STRATA_YIELDS_MAX below the shorts' count), which is not "shorts no longer wait behind long reads"
            key = "s_max" if r == "w1" else "all_first"
            ok = x[key] is not None and b[key] is not None and x[key] <= a.gain * b[key] and x["miss"] == 0
            if r == "w1":
                say(f"    win {label} vs {base} w1: shorts TTFT p50 {f2(b['s_p50'])} -> {f2(x['s_p50'])} s INFO")
            say(f"==> win {label} vs {base} {r}: {'every short TTFT (max)' if r == 'w1' else 'time to all 16 first tokens'} "
                f"{f2(b[key])} -> {f2(x[key])} s (want <= {a.gain:.2f} x), long read TTFT {f2(b['long_ttft'])} -> "
                f"{f2(x['long_ttft'])} s, total {f2(b['long_total'])} -> {f2(x['long_total'])} s {verdict(ok)}")
            if a.long_tol is not None:      # (the kill rule's second half: the long read not slower than the base's)
                okl = bool(b["long_total"] and x["long_total"]) and x["long_total"] <= (1.0 + a.long_tol) * b["long_total"]
                say(f"==> win {label} vs {base} {r}: the long read's total {f2(b['long_total'])} -> {f2(x['long_total'])} s "
                    f"(want <= {1.0 + a.long_tol:.2f} x) {verdict(okl)}")
            elif b["long_total"] and x["long_total"]:
                say(f"==> win {label} vs {base} {r}: the long read's cost {x['long_total'] - b['long_total']:+.2f} s "
                    f"({(x['long_total'] / b['long_total'] - 1) * 100:+.1f} %) INFO")
    return 0


GIVE_RE = re.compile(r"the prompt read gives way at (\d+) of (\d+) tokens; slot (\d+) holds it \(copied in ([\d.]+) ms")


def mode_yieldstat(a):
    facts = {}
    for spec in a.arms:
        label, pat = spec.split("=", 1)
        files = sorted(glob.glob(pat))
        if not files:
            say(f"    yieldstat {label}: no log {pat} (step skipped?) INFO")
            continue
        lines, starts = [], 0
        for fp in files:
            try:
                fl = open(fp, errors="replace").read().splitlines()
            except OSError:
                fl = []
            lines += fl
            starts = max(starts, sum("engine started" in l for l in fl))   # (per engine log: one server start each)
        # STRATA_READ_YIELD's own stops end in "(copied in X ms, STRATA_READ_YIELD)"; the old #656 path (production's
        # binary and the new one with the env unset, BYIELD from a server's half-length rule) prints the same line
        # without the suffix - an old-path stop is production behaviour, not a yield of the new mechanism
        g_all = [(GIVE_RE.search(l), "STRATA_READ_YIELD)" in l) for l in lines]
        g = [m for m, new in g_all if m and new]
        c = {"yields": len(g), "old656": sum(1 for m, new in g_all if m and not new),
             "q": [(int(m.group(1)), int(m.group(2))) for m in g],
             "copy": [float(m.group(4)) for m in g],
             "unpinned": sum("holds no read of this prompt that gave way" in l for l in lines),
             "park_ok": sum("STRATA_SLOT_PARK_VERIFY: slot" in l and "bytes identical" in l for l in lines),
             "park_bad": sum("differs from a plain capture" in l or "was written before its capture" in l for l in lines),
             "err": [l[:200] for l in lines if l.startswith("ERR") or " ERR " in l or "GPU stopped responding" in l
                     or "not a live token prefix" in l],
             "watchdog": sum("no progress for" in l for l in lines),
             "starts": starts}
        facts[label] = c
        cp = sorted(c["copy"])
        say(f"==> yieldstat {label}: {c['yields']} reads gave way (STRATA_READ_YIELD; q of n: {c['q'][:8]}"
            f"{' ...' if len(c['q']) > 8 else ''}; old #656 path {c['old656']}), "
            f"copy ms median {f2(pct(cp, 50))} max {f2(cp[-1] if cp else None)}; unpinned resumes {c['unpinned']}; park "
            f"verify {c['park_ok']} identical / {c['park_bad']} differ; ERR {len(c['err'])}, watchdog {c['watchdog']}, "
            f"engine starts {c['starts']} (most in one log) INFO")
        for l in c["err"][:4]:
            say(f"    {label}: {l}")
    for label in (a.want_yield.split(",") if a.want_yield else []):
        c = facts.get(label)
        if c is not None:
            say(f"==> yieldstat {label}: reads gave way {c['yields']} times (want > 0: else its exactness proves nothing "
                f"about the yield), every resume pinned to its paused part (unpinned {c['unpinned']}, want 0) "
                f"{verdict(c['yields'] > 0 and c['unpinned'] == 0)}")
    for label in (a.want_none.split(",") if a.want_none else []):
        c = facts.get(label)
        if c is not None:
            say(f"==> yieldstat {label}: no STRATA_READ_YIELD stop (env unset / production: {c['yields']}; old #656 path "
                f"{c['old656']}, as production) {verdict(c['yields'] == 0)}")
    for label in (a.want_clean.split(",") if a.want_clean else []):
        c = facts.get(label)
        if c is not None:
            ok = not c["err"] and c["watchdog"] == 0 and c["starts"] <= a.max_starts and c["park_bad"] == 0
            say(f"==> yieldstat {label}: ERR {len(c['err'])}, watchdog {c['watchdog']}, engine starts {c['starts']} "
                f"(<= {a.max_starts}), park verify differ {c['park_bad']} {verdict(ok)}")
    for spec in a.test_at:
        label, k = spec.split(":")
        c = facts.get(label)
        if c is not None:
            lo = int(k) * 512
            say(f"==> yieldstat {label}: STRATA_READ_YIELD_TEST_AT={k} (from ~{lo} tokens into a segment): stops at "
                f"{[q for q, _ in c['q']]} INFO")
    return 0


BG_RE = re.compile(r"the prompt read goes to the background at (\d+) of (\d+) tokens: slot (\d+) reads (\d+) tokens")
RDY_RE = re.compile(r"background read of slot (-?\d+)'s request is ready in slot (\d+): (\d+) tokens read in (\d+) ms")
WAITS_RE = re.compile(r"background read in slot \d+ waits: ")
DRAINED_RE = re.compile(r"background read in slot \d+: \d+ tokens with the groups drained in (\d+) ms")
BESIDE_RE = re.compile(r"background read in slot (\d+): (\d+) tokens beside the groups in (\d+) ms \(drain ([\d.]+) \+ ([\d.]+) ms\)")


def mode_readstat(a):
    facts = {}
    for spec in a.arms:
        label, pat = spec.split("=", 1)
        files = sorted(glob.glob(pat))
        if not files:
            say(f"    readstat {label}: no log {pat} (step skipped?) INFO")
            continue
        lines, starts = [], 0
        for fp in files:
            try:
                fl = open(fp, errors="replace").read().splitlines()
            except OSError:
                fl = []
            lines += fl
            starts = max(starts, sum("engine started" in l for l in fl))
        bg = [BG_RE.search(l) for l in lines]
        bg = [m for m in bg if m]
        rdy = [m for m in (RDY_RE.search(l) for l in lines) if m]
        beside = [m for m in (BESIDE_RE.search(l) for l in lines) if m]
        drops = [l[:200] for l in lines if "background read of slot" in l and " ends at " in l]
        c = {"bg": len(bg), "ready": len(rdy), "drops": drops,
             "cancel": sum(": cancel" in l for l in drops), "err_drops": sum(": err" in l for l in drops),
             "at": [(int(m.group(1)), int(m.group(2)), int(m.group(4))) for m in bg],
             "rate": [int(m.group(3)) * 1000.0 / max(1, int(m.group(4))) for m in rdy],
             "beside": len(beside), "paused": sum("paused for a line" in l for l in lines),
             "drained": sum("with the groups drained" in l for l in lines),
             "waits": sum(bool(WAITS_RE.search(l)) for l in lines),
             "drained_ms": sum(float(m.group(1)) for m in (DRAINED_RE.search(l) for l in lines) if m),
             "drain_ms": [float(m.group(4)) + float(m.group(5)) for m in beside],
             "not_taken": sorted({l.split("not taken, the prompt is read here: ", 1)[1][:80] for l in lines
                                  if "not taken, the prompt is read here: " in l}),
             "unpinned": sum("holds no read of this prompt that gave way" in l for l in lines),
             "park_ok": sum("STRATA_SLOT_PARK_VERIFY: slot" in l and "bytes identical" in l for l in lines),
             "park_bad": sum("differs from a plain capture" in l or "was written before its capture" in l for l in lines),
             "padguard": sum("which a background read reads into" in l for l in lines),
             "err": [l[:200] for l in lines if l.startswith("ERR") or " ERR " in l or "GPU stopped responding" in l
                     or "not a live token prefix" in l],
             "watchdog": sum("no progress for" in l or "background read finished for" in l for l in lines),
             "starts": starts}
        facts[label] = c
        dm = sorted(c["drain_ms"])
        say(f"==> readstat {label}: {c['bg']} reads to the background (at, n, tokens: {c['at'][:6]}"
            f"{' ...' if len(c['at']) > 6 else ''}), {c['ready']} ready (tok/s median {f2(pct(c['rate'], 50))}), dropped "
            f"{c['cancel']} cancel / {c['err_drops']} err; {c['beside']} turns beside the groups ({c['paused']} paused for a "
            f"line, drain ms median {f2(pct(dm, 50))} max {f2(dm[-1] if dm else None)}), {c['drained']} drained "
            f"({c['drained_ms']:.0f} ms in all), gate refused {c['waits']} time(s); not taken: "
            f"{c['not_taken'][:4]}; unpinned finishes {c['unpinned']}; park verify {c['park_ok']} identical / "
            f"{c['park_bad']} differ; ERR {len(c['err'])}, watchdog {c['watchdog']}, pad guard {c['padguard']}, engine "
            f"starts {c['starts']} INFO")
        for l in (c["err"] + c["drops"])[:6]:
            say(f"    {label}: {l}")
    for label in (a.want_bg.split(",") if a.want_bg else []):
        c = facts.get(label)
        if c is not None:
            ok = c["bg"] > 0 and c["ready"] == c["bg"] and c["err_drops"] == 0 and c["unpinned"] == 0
            say(f"==> readstat {label}: {c['bg']} background reads (want > 0: else its exactness proves nothing about "
                f"them), {c['ready']} ready (want all), {c['err_drops']} dropped with err (want 0), unpinned finishes "
                f"{c['unpinned']} (want 0) {verdict(ok)}")
    for label in (a.want_none.split(",") if a.want_none else []):
        c = facts.get(label)
        if c is not None:
            say(f"==> readstat {label}: no background read (env unset / production: {c['bg']}) {verdict(c['bg'] == 0)}")
    for label in (a.want_clean.split(",") if a.want_clean else []):
        c = facts.get(label)
        if c is not None:
            ok = not c["err"] and c["watchdog"] == 0 and c["starts"] <= a.max_starts and c["park_bad"] == 0 and \
                c["padguard"] == 0
            say(f"==> readstat {label}: ERR {len(c['err'])}, watchdog {c['watchdog']}, engine starts {c['starts']} "
                f"(<= {a.max_starts}), park verify differ {c['park_bad']}, pad guard {c['padguard']} {verdict(ok)}")
    return 0


def dec_facts(v):
    """A reply's decode rate (the engine's timings when the server sent them, else pieces / streaming time), its
    drafts' acceptance (accepted / offered: None without drafts), TTFT and longest stall."""
    t = v.get("timings") or {}
    rate = t.get("predicted_per_second")
    if rate is None and v.get("t_first") and v.get("deltas", 0) > 1 and v["t_done"] > v["t_first"]:
        rate = (v["deltas"] - 1) / (v["t_done"] - v["t_first"])
    acc = (t["draft_n_accepted"] / t["draft_n"]) if t.get("draft_n") else None
    return {"rate": rate, "acc": acc, "ttft": (v["t_first"] - v["t_send"]) if v.get("t_first") else None,
            "gap": v.get("gap_max")}


def mode_dec(a):
    """The long requests' decode after their read (keys with '-long', every turn): decode tok/s, draft acceptance,
    TTFT per arm (--arms LABEL=GLOB, the first the base); judged: each later arm's median decode rate >= (1 - --tol) x
    the base's and its median acceptance >= the base's - --tol (the background read builds no draft K/V: E-9).  The
    other replies' (the streams') longest stall is printed (INFO)."""
    arms = []
    for spec in a.arms:
        label, pat = spec.split("=", 1)
        files = sorted(glob.glob(pat))
        per = {}
        for fp in files:
            d = json.load(open(fp))
            for k, v in d.items():
                if a.rounds and k.split("-", 1)[0] not in a.rounds.split(","):
                    continue
                if isinstance(v, dict) and "text" in v and not v.get("error"):
                    per.setdefault(k, []).append(dec_facts(v))
        if not files:
            say(f"    dec {label}: no scenario files {pat} (skipped) INFO")
            continue
        arms.append((label, per))
    if len(arms) < 2:
        say("==> dec: fewer than two arms ran INFO")
        return 0

    def med(per, pick, what):
        xs = [f[what] for k, fs in per.items() if pick(k) for f in fs if f[what] is not None]
        return pct(xs, 50), (max(xs) if xs else None), len(xs)
    is_long = lambda k: "-long" in k
    is_other = lambda k: "-long" not in k
    base_l, base = arms[0]
    b_rate, _, _ = med(base, is_long, "rate")
    b_acc, _, _ = med(base, is_long, "acc")
    for label, per in arms:
        rate, _, n = med(per, is_long, "rate")
        acc, _, na = med(per, is_long, "acc")
        ttft, tmax, _ = med(per, is_long, "ttft")
        _, gmax, _ = med(per, is_other, "gap")
        say(f"==> dec {label}: {n} long replies, decode tok/s median {f2(rate)}, draft acceptance median {f2(acc)} "
            f"({na} with drafts), TTFT median {f2(ttft)} max {f2(tmax)} s; the other replies' longest stall "
            f"{f2(gmax)} s INFO")
        for k in sorted(per):
            if is_long(k):
                f = per[k]
                say(f"    {label} {k}: tok/s {[f2(x['rate']) for x in f]} acc {[f2(x['acc']) for x in f]} "
                    f"TTFT {[f2(x['ttft']) for x in f]}")
    for label, per in arms[1:]:
        rate, _, _ = med(per, is_long, "rate")
        acc, _, _ = med(per, is_long, "acc")
        ok_r = rate is not None and b_rate is not None and rate >= (1.0 - a.tol) * b_rate
        ok_a = acc is None or b_acc is None or acc >= b_acc - a.tol
        say(f"==> dec {label} vs {base_l}: long replies' decode {f2(rate)} vs {f2(b_rate)} tok/s (>= {1 - a.tol:.2f}x), "
            f"draft acceptance {f2(acc)} vs {f2(b_acc)} (>= base - {a.tol}) {verdict(ok_r and ok_a)}")
    return 0


def mode_mix(a):
    """The agent mix (window-reads2.sh step 4): bench_multiturn (16 users x 3 turns, greedy) and bench_agentic rows of
    --dir's multiturn.jsonl / agentic.jsonl, labels mix-<ARM>-r<k>-mt / -agentic (--arms-list, --reps; the first arm
    the base).  Judged: every multiturn reply (uid, turn) identical to the base arm's run of the same repetition, and
    the follow-up TTFT p50 (turns 2-3: the requests that wait behind other users' document reads) and the agentic
    roles' TTFT p50 at most --gain x the base's (mean over the repetitions); the rest INFO."""
    def rows(name, cut):
        out = {}
        fp = os.path.join(a.dir, name)
        for line in (open(fp) if os.path.exists(fp) else []):
            if line.strip():
                r = json.loads(line)
                out[r["label"][:-cut]] = r
        return out
    mt, ag = rows("multiturn.jsonl", 3), rows("agentic.jsonl", 8)
    arms = a.arms_list.split(",")
    lab = lambda arm, k: f"mix-{arm}-r{k}"
    for arm in arms:
        for k in range(1, a.reps + 1):
            r = mt.get(lab(arm, k))
            if r:
                say(f"    mt {lab(arm, k)}: TTFT p50/p95 turn1 {r['turn1']['ttft_p50']:.2f}/{r['turn1']['ttft_p95']:.2f} "
                    f"turn2 {r['turn2']['ttft_p50']:.2f}/{r['turn2']['ttft_p95']:.2f} turn3 {r['turn3']['ttft_p50']:.2f}/"
                    f"{r['turn3']['ttft_p95']:.2f}, follow-up p50 {r['follow_ttft_p50']:.2f} s, wall {r['wall_s']:.0f} s, "
                    f"{r['agg_out_tps']:.1f} tok/s")
            g = ag.get(lab(arm, k))
            if g:
                say(f"    agentic {lab(arm, k)}: " + "  ".join(f"{x} {g[x]['ttft_p50']}/{g[x]['ttft_p95']}/{g[x]['ttft_max']}"
                                                         for x in ("main", "sub", "chat") if x in g)
                    + f"  wall {g['wall_s']:.0f} s, {g['agg_out_tps']} tok/s")
    for arm in arms:
        for k in range(1, a.reps + 1):
            g = ag.get(lab(arm, k))
            if g is None:
                say(f"==> mix {arm} r{k}: agentic run missing FAIL")
                continue
            errs = {x: g[x].get("err", 0) for x in ("main", "sub", "chat") if x in g}
            say(f"==> mix {arm} r{k}: agentic request errors {errs} {verdict(not any(errs.values()))}")
    mean = lambda xs: st.mean(xs) if xs else None
    base = arms[0]
    agg = {}
    for arm in arms:
        m = [mt[lab(arm, k)] for k in range(1, a.reps + 1) if lab(arm, k) in mt]
        g = [ag[lab(arm, k)] for k in range(1, a.reps + 1) if lab(arm, k) in ag]
        agg[arm] = {"follow": mean([r["follow_ttft_p50"] for r in m]), "t1": mean([r["turn1"]["ttft_p50"] for r in m]),
                    "tps": mean([r["agg_out_tps"] for r in m]), "n": len(m), "ng": len(g),
                    **{x: mean([r[x]["ttft_p50"] for r in g if x in r and r[x].get("ttft_p50") is not None])
                       for x in ("main", "sub", "chat")},
                    "awall": mean([r["wall_s"] for r in g])}
        v = agg[arm]
        say(f"==> mix {arm} (mean of {v['n']} multiturn / {v['ng']} agentic runs): multiturn turn1 TTFT p50 {f2(v['t1'])} s, "
            f"follow-up p50 {f2(v['follow'])} s, {f2(v['tps'])} tok/s; agentic TTFT p50 main {f2(v['main'])} sub "
            f"{f2(v['sub'])} chat {f2(v['chat'])} s, wall {f2(v['awall'])} s INFO")
    for arm in arms[1:]:
        for k in range(1, a.reps + 1):
            x, y = mt.get(lab(base, k)), mt.get(lab(arm, k))
            if not x or not y:
                say(f"==> mix {arm} r{k} vs {base}: multiturn run missing FAIL")
                continue
            ra = {(u, t): (gn, h) for u, t, gn, h in x["replies"]}
            rb = {(u, t): (gn, h) for u, t, gn, h in y["replies"]}
            diff = sorted(kk for kk in set(ra) | set(rb) if ra.get(kk) != rb.get(kk))
            say(f"==> mix {arm} r{k} vs {base}: multiturn replies {len(ra) - sum(kk in ra for kk in diff)}/{len(ra)} "
                f"identical" + (f" (first differ uid/turn {diff[:4]})" if diff else "") + f" {verdict(bool(ra) and not diff)}")
        b, v = agg[base], agg[arm]
        # judged: follow-ups and the agents; INFO: turn 1 (16 cold documents at once: background reads trade it) and the
        # chat user (a handful of requests at random times: timing dependent, TESTLOG #87)
        for key, what, judged in (("follow", "multiturn follow-up TTFT p50", True), ("main", "agentic main TTFT p50", True),
                                  ("sub", "agentic sub-agent TTFT p50", True), ("t1", "multiturn turn-1 TTFT p50", False),
                                  ("chat", "agentic chat TTFT p50", False)):
            if b[key] is None or v[key] is None:
                say(f"==> mix {arm} vs {base}: {what} not measured INFO")
                continue
            # (+0.25 s: a sub-second TTFT moves by a fraction of a second with one request's timing)
            ok = v[key] <= a.gain * b[key] + 0.25
            say(f"==> mix {arm} vs {base}: {what} {f2(b[key])} -> {f2(v[key])} s"
                + (f" (want <= {a.gain:.2f} x + 0.25 s) {verdict(ok)}" if judged else " INFO"))
    return 0


def mode_absum(a):
    rows = {}
    for i, line in enumerate(open(a.sweep)):
        if i < a.since_line or not line.strip():
            continue
        r = json.loads(line)
        rows[(r["label"], r["users"])] = r
    arms = a.arms_list.split(",")

    def lab(arm, rep, kind):
        return f"ab-{a.tag}-{arm}" + (f"-r{rep}" if a.reps > 1 else "") + f"-{kind}"
    levels = sorted({u for (l, u) in rows if l.startswith(f"ab-{a.tag}-")})
    n = 0
    for kind in ("short", "long"):
        for u in levels:
            base = {}
            for arm in arms:
                v = [rows.get((lab(arm, k, kind), u)) for k in range(1, a.reps + 1)]
                if not all(v):
                    continue
                tps = [x["agg_out_tps"] for x in v]
                std = [x.get("agg_out_tps_steady") or float("nan") for x in v]
                line = (f"{kind:>5} {u:>3} users {arm:<24} agg {st.mean(tps):7.1f}"
                        + (f" (sd {st.stdev(tps) / st.mean(tps) * 100:.2f} %)" if a.reps > 1 else "")
                        + f"  steady {st.mean(std):7.1f}  TTFT p50 {st.mean(x['ttft_p50'] for x in v):.2f}")
                if not base:
                    base = {"tps": tps, "arm": arm}
                else:
                    d = [(b / x - 1) * 100 for x, b in zip(base["tps"], tps)]
                    line += f"  vs {base['arm']}: {st.mean(d):+.2f} %" + (
                        f" (pairs {', '.join(f'{x:+.2f}' for x in d)})" if a.reps > 1 else "")
                    for k in range(1, a.reps + 1):
                        p = subprocess.run([sys.executable, "-I", os.path.join(HERE, "compare_replies.py"), a.requests,
                                            lab(base["arm"], k, kind), lab(arm, k, kind), "--users", str(u), "--show", "0"],
                                           capture_output=True, text=True)
                        line += f"  [r{k}: {p.stdout.split(' - ')[-1].strip() if ' - ' in p.stdout else p.stdout.strip()}]"
                print(line, flush=True)
                n += " vs " in line
    say(f"==> absum {a.tag}: {n} comparison lines (arms {a.arms_list}, {a.reps} repetitions) {verdict(n > 0)}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True,
                    choices=["tokcheck", "scen", "cmp", "win", "yieldstat", "absum", "readstat", "dec", "mix"])
    ap.add_argument("--config", default="")
    ap.add_argument("--url", default="http://127.0.0.1:8091")
    ap.add_argument("--model", default="strata")
    ap.add_argument("--scenario", default="exact", help="exact, win, two, solo, vis or a comma list (one run, in order)")
    ap.add_argument("--tol", type=float, default=0.10, help="dec: the decode rate / acceptance tolerance")
    ap.add_argument("--two", action="store_true", help="tokcheck: the `two` scenario too (window-reads2.sh)")
    ap.add_argument("--want-bg", default="", help="readstat: arms that must have read in the background")
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--reps-list", default="1,2")
    ap.add_argument("--rounds", default="", help="scen: only these rounds (comma list)")
    ap.add_argument("--short", type=int, default=2048, help="tokcheck: a short request's prompt at most this long")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--a", default="")
    ap.add_argument("--b", default="")
    ap.add_argument("--what", default="replies")
    ap.add_argument("--arms", nargs="*", default=[], help="win / yieldstat: LABEL=GLOB")
    ap.add_argument("--gain", type=float, default=0.5, help="win: the new arm's TTFT at most this fraction of the base's")
    ap.add_argument("--long-tol", type=float, default=None,
                    help="win: also judge the long read's total <= (1 + this) x the base's (default: INFO only)")
    ap.add_argument("--want-yield", default="")
    ap.add_argument("--want-none", default="")
    ap.add_argument("--want-clean", default="")
    ap.add_argument("--max-starts", type=int, default=1)
    ap.add_argument("--test-at", action="append", default=[], help="yieldstat: LABEL:k")
    ap.add_argument("--sweep", default=os.path.join(HERE, "results/sweep.jsonl"))
    ap.add_argument("--requests", default=os.path.join(HERE, "results/sweep.requests.jsonl"))
    ap.add_argument("--since-line", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--arms-list", default="")
    ap.add_argument("--dir", default="", help="mix: the window's results dir (multiturn.jsonl, agentic.jsonl)")
    a = ap.parse_args()
    return {"tokcheck": mode_tokcheck, "scen": mode_scen, "cmp": mode_cmp, "win": mode_win, "yieldstat": mode_yieldstat,
            "absum": mode_absum, "readstat": mode_readstat, "dec": mode_dec, "mix": mode_mix}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
