#!/usr/bin/env python3
"""exactness-r1: the next turn of r09/r25/r33/r41 (soak2) continued from the parent's slot, deterministically: 8
background slots decode (max_new 600), each parent is admitted into the last slot, ends (length 160), then its next
turn is admitted into the same slot (RESUME from the slot).  Prompts and solo references from soak2-on.json; the
children's tokens are compared with solo and with the soak2 runs' batch tokens."""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, "/media/raid0/StrataQ38FN/Strata/tools")
from batch_test import Engine            # noqa: E402
from batch_overlap_test import Driver, EngineErr  # noqa: E402
ap = argparse.ArgumentParser()
ap.add_argument("--exe", required=True); ap.add_argument("--config", required=True)
ap.add_argument("--extra", default=""); ap.add_argument("--env", action="append", default=[])
ap.add_argument("--dump", default=""); ap.add_argument("--bg", type=int, default=8)
a = ap.parse_args()
R = Path("/media/raid0/StrataQ38FN/linux/results/overlap/exactness-r1")
src = json.loads((R / "soak2-on.json").read_text())
others = {n: json.loads((R / f"{n}.json").read_text()) for n in ("soak2-on", "soak2-on2", "soak2-off", "soak2-ref")}
cfg = json.loads(Path(a.config).read_text())
env = {"STRATA_IQ_MT_MIN": "1", "STRATA_TRACE": "1", **(cfg.get("env") or {})}
for kv in a.env:
    k, _, v = kv.partition("="); env[k] = v
eng = Engine(a.exe, cfg, 16, env, a.extra.split())
d = Driver(eng)
print("engine", a.exe, sorted(env.items()), flush=True)
P, S = src["prompts"], src["solo"]
bgs = ["r02", "r03", "r05", "r06", "r10", "r11", "r13", "r14"][: a.bg]
for i, k in enumerate(bgs):
    d.admit(i, "bg_" + k, P[k], 600)
ok = True
pairs = [("r09", "r16"), ("r25", "r32"), ("r33", "r40"), ("r41", "r48"), ("r17", "r24"), ("r49", "r56")]
for p, c in pairs:
    d.admit(15, p, P[p], 160); d.wait_done(p)
    d.admit(15, c, P[c], 96); d.wait_done(c)
    same_p = d.toks[p] == S[p]
    same = d.toks[c] == S[c]
    first = next((i for i in range(min(len(d.toks[c]), len(S[c]))) if d.toks[c][i] != S[c][i]), None)
    eq = {n: (o["batch"].get(c) == d.toks[c], o["resume"].get(c)) for n, o in others.items()}
    print(f"{p}: {'IDENTICAL' if same_p else 'DIFFERS'} to solo; {c}: RESUME {d.resume.get(c)}, "
          f"{'IDENTICAL' if same else f'DIFFERS at {first}'} to solo; equal to soak runs' {c} (equal, their RESUME): {eq}", flush=True)
    ok &= same and same_p
d.stop_all()
eng.send("QUIT")
try:
    eng.p.wait(timeout=120)
except Exception:
    eng.p.kill()
if a.dump:
    Path(a.dump).write_text(json.dumps({"batch": d.toks, "resume": d.resume}))
print("RESULT:", "PASS" if ok else "FAIL")
