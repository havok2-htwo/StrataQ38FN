#!/usr/bin/env python3
"""Corrected reference for the "next turn from the slot differs from a fresh read" row (TESTLOG row after #40):
soak64b computed the child's solo after six other GENs, so the prompt cache (6 checkpoints) had evicted the parent
and the solo child re-read the parent's reply by prefill, while the slot continued from decode state - two numeric
paths.  Here each child's solo runs RIGHT AFTER its parent's solo (the child continues from the parent's decode
state on the solo chain, as the slot does), then the same pairs run in slot 15 beside 8 decoding slots (reuse4)."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, "/media/raid0/StrataQ38FN/Strata-142/tools")
from batch_test import Engine            # noqa: E402
from batch_overlap_test import Driver    # noqa: E402
ap = argparse.ArgumentParser()
ap.add_argument("--exe", required=True); ap.add_argument("--config", required=True)
ap.add_argument("--extra", default=""); ap.add_argument("--dump", default="")
a = ap.parse_args()
R = Path("/media/raid0/StrataQ38FN/linux/results/overlap/exactness-r1")
src = json.loads((R / "soak2-on.json").read_text())
cfg = json.loads(Path(a.config).read_text())
env = {"STRATA_IQ_MT_MIN": "1", "STRATA_TRACE": "1", **(cfg.get("env") or {})}
eng = Engine(a.exe, cfg, 16, env, a.extra.split())
d = Driver(eng)
P, Sold = src["prompts"], src["solo"]
pairs = [("r09", "r16"), ("r25", "r32"), ("r33", "r40"), ("r41", "r48"), ("r17", "r24"), ("r49", "r56")]
solo = {}
for p, c in pairs:                       # parent solo, child solo right after it (continues from the live chain)
    solo[p] = d.solo(P[p], 160); solo[c] = d.solo(P[c], 96)
for i, k in enumerate(["r02", "r03", "r05", "r06", "r10", "r11", "r13", "r14"]):
    d.admit(i, "bg_" + k, P[k], 600)
ok = True
for p, c in pairs:
    d.admit(15, p, P[p], 160); d.wait_done(p)
    d.admit(15, c, P[c], 96); d.wait_done(c)
    fd = lambda x, y: next((i for i in range(min(len(x), len(y))) if x[i] != y[i]), None)
    same_p, same_c = d.toks[p] == solo[p], d.toks[c] == solo[c]
    old = d.toks[c] == Sold[c]
    print(f"{p}: {'IDENTICAL' if same_p else 'DIFFERS at %s' % fd(d.toks[p], solo[p])} to solo; {c} from slot "
          f"(RESUME {d.resume.get(c)}): {'IDENTICAL' if same_c else 'DIFFERS at %s' % fd(d.toks[c], solo[c])} to the "
          f"corrected solo, {'equal' if old else 'not equal'} to the old soak64b solo", flush=True)
    ok &= same_p and same_c
d.stop_all(); eng.send("QUIT")
try:
    eng.p.wait(timeout=120)
except Exception:
    eng.p.kill()
if a.dump:
    Path(a.dump).write_text(json.dumps({"solo": solo, "batch": d.toks, "resume": d.resume}))
print("RESULT:", "PASS" if ok else "FAIL")
