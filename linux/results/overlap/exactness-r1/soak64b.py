#!/usr/bin/env python3
"""Exactness-r1 soak (written for this run, not in the repo): N requests through --batch slots, greedy.
Every slot admission after the first ones is read while other slots decode (STRATA_PIPE_OVERLAP=1: overlapped).
Prompts cycle ~150/600/1200/3000 filler tokens; max_new cycles 16..96 so slots end at different times; every
8th request (from #8 on) is the next turn of an earlier finished one, preferably into its parent's slot (reuse);
requests with index % 16 == 7 get a BSTOP after 10 tokens (prefix check).  Every request must equal its solo.
v2: max_new 64..300 (the 16 slots stay busy), a finished parent's slot is kept free for its next turn, BSTOP also
from BT lines read inside an admission.
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, "/media/raid0/StrataQ38FN/Strata/tools")
from batch_test import Engine, tokenizer          # noqa: E402
from batch_overlap_test import QUESTIONS, chat, Driver, EngineErr, check_log  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--exe", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--extra", default="")
ap.add_argument("--env", action="append", default=[])
ap.add_argument("--dump", default="")
ap.add_argument("--solo-json", default="")
ap.add_argument("--compare", default="")
a = ap.parse_args()
cfg = json.loads(Path(a.config).read_text())
tok = tokenizer(cfg["tokenizer"])
FILLS = [150, 600, 1200, 3000]
NEWS = [96, 160, 240, 128, 200, 64, 300, 180]
P, parent = {}, {}
for i in range(a.n):
    name = f"r{i:02d}"
    if i >= 8 and i % 8 == 0:
        parent[name] = f"r{i - 7:02d}"   # filled in after the parent's solo
        P[name] = None
        child_of = None
    else:
        P[name] = (chat(tok, QUESTIONS[i % len(QUESTIONS)], fill=FILLS[i % 4], seed=100 + i), NEWS[i % 8])
env = {"STRATA_IQ_MT_MIN": "1", "STRATA_TRACE": "1", **(cfg.get("env") or {})}
for kv in a.env:
    k, _, v = kv.partition("=")
    env[k] = v
env_on = env.get("STRATA_PIPE_OVERLAP", "0") not in ("", "0")
eng = Engine(a.exe, cfg, a.batch, env, a.extra.split())
d = Driver(eng)
print(f"engine {a.exe} split {cfg.get('layer_split')} env {sorted(env.items())}", flush=True)
solo = {}
if a.solo_json:
    old = json.loads(Path(a.solo_json).read_text())
    solo = old["solo"]
    print("solo reused from", a.solo_json)
t0 = time.time()
for k in P:
    if k in parent:
        pk = parent[k]
        ids = P[pk][0] + solo[pk] + tok.encode("<|im_end|>\n<|im_start|>user\nGo on, in more detail.<|im_end|>\n"
                                               "<|im_start|>assistant\n<think>\n\n</think>\n\n", parse_special=True)
        P[k] = (ids, 96)
    if k not in solo:
        solo[k] = d.solo(*P[k])
if a.solo_json:
    old = json.loads(Path(a.solo_json).read_text())
    assert old["prompts"] == {k: v[0] for k, v in P.items()}, "solo json is for other prompts"
print(f"solo: {len(P)} in {time.time() - t0:.0f} s; prompt tokens {sorted(len(v[0]) for v in P.values())}", flush=True)

stop_at = {k for i, k in enumerate(P) if i % 16 == 7}
stopped, last_slot, pending = set(), {}, list(P)
children = {v: k for k, v in parent.items()}
_handle = d.handle
def handle(line):   # BSTOP after 10 tokens of a stop_at request, wherever its BT line is read (also inside admit)
    r = _handle(line)
    if line.startswith("BT "):
        s = int(line.split()[1])
        k = d.slot_req.get(s)
        if k in stop_at and k not in stopped and len(d.toks[k]) >= 10:
            eng.send(f"BSTOP {s}")
            stopped.add(k)
    return r
d.handle = handle
def reserved():     # the slot of a finished parent whose next turn is not admitted yet
    return {last_slot[p] for p, c in children.items() if p in last_slot and c in pending}
t0 = time.time()
admit_t = []
try:
    while pending or d.slot_req:
        free = [s for s in range(a.batch) if s not in d.slot_req]
        res = reserved()
        ready = [k for k in pending if k not in parent or (parent[k] in d.fin and parent[k] not in d.slot_req.values())]
        k = None
        for cand in ready:
            ps = last_slot.get(parent.get(cand), -1)
            if ps in free:
                k, slot = cand, ps
                break
            other = [f for f in free if f not in res]
            if other:
                k, slot = cand, other[0]
                break
        if k is not None:
            pending.remove(k)
            last_slot[k] = slot
            ta = time.time()
            d.admit(slot, k, *P[k])
            admit_t.append((k, len(P[k][0]), slot, len(d.slot_req) - 1, time.time() - ta))
            continue
        d.handle(d.get())
except EngineErr as x:
    print("ENGINE ERR:", x)
    print("RESULT: FAIL")
    sys.exit(2)
print(f"batch phase: {len(P)} requests in {time.time() - t0:.1f} s", flush=True)
for k, n, s, busy, dt in admit_t:
    print(f"   admit {k} ({n} tok) slot {s} beside {busy} decoding slots: {dt:.2f} s")
ok = True
nfail = 0
for k in P:
    b, s = d.toks.get(k, []), solo[k]
    fin = d.fin.get(k, "?")
    if fin == "cancel":
        same = b == s[: len(b)]
        v = f"stopped at {len(b)} of {len(s)}: {'PREFIX OK' if same else 'DIFFERS'}"
    else:
        same = b == s
        first = next((i for i in range(min(len(b), len(s))) if b[i] != s[i]), None)
        v = f"{len(b)} tokens ({fin}), solo {len(s)}: " + ("IDENTICAL" if same else f"DIFFERS at {first}")
    if k in parent:
        v += f"; next turn of {parent[k]}, RESUME {d.resume.get(k)} in slot {last_slot[k]} (parent slot {last_slot[parent[k]]})"
    ok &= same
    nfail += not same
    print(f"{k}: {v}")
eng.send("QUIT")
try:
    eng.p.wait(timeout=180)
except Exception:
    eng.p.kill()
eng.log.close()
ok &= check_log(eng.log_path, env_on)
bt = {k: d.toks.get(k, []) for k in P}
if a.dump:
    Path(a.dump).write_text(json.dumps({"prompts": {k: v[0] for k, v in P.items()}, "solo": solo, "batch": bt,
                                        "resume": d.resume, "fin": d.fin}))
if a.compare:
    o = json.loads(Path(a.compare).read_text())
    diff = [k for k in P if k not in stop_at and o["batch"].get(k) != bt[k]]
    diff += [k for k in stop_at if o["batch"][k][:min(len(o["batch"][k]), len(bt[k]))] != bt[k][:min(len(o["batch"][k]), len(bt[k]))]]
    rdiff = [k for k in P if o["resume"].get(k) != d.resume.get(k)]
    print(f"compare with {a.compare}: " + ("token-identical" if not diff else f"DIFFERS {diff}"))
    print(f"compare with {a.compare}: RESUME " + ("identical" if not rdiff else f"DIFFERS {[(k, o['resume'].get(k), d.resume.get(k)) for k in rdiff]}"))
    ok &= not diff
print(f"{nfail} requests differ from solo")
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 2)
