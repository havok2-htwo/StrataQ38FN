#!/usr/bin/env python3
"""fixedcost step S1 edge cases (STRATA_SLOT_CKPT_MOVE): the turn checkpoint moves into the slot; the request path
finds it in the slots.  Seeded sampling; slots 0-3 / 4-7 / 8-11 / 12-15 are the 4 pipelined groups.
  W0 (4)          warm-up chat
  S  (3)          2.3K system prompt: root + turn checkpoint (turn moved into slot 3, root stays in the chain)
  S2 (3)          S's next turn right after it: the main session still holds S -> restored from slot 3's checkpoint
                  on the main session (no K/V copy); slot 3 is the admit slot -> moved back, then on into slot 3
  S3 (8)          S2's next turn into another slot: restored from slot 3's (kept there)
  G  (9)          S3's prompt again (regenerate, no new turn checkpoint): from slot 8's, COPIED into slot 9
  G2 (8)          the same into slot 8: moved back from slot 8 and moved on (moved_in)
  R  (10)         a new chat with S's system prompt: the root from the chain
  D  (12, 400)    new chat, keeps decoding
  D1 (13, 400)    D's prompt while D decodes (duplicate): from ACTIVE slot 12's checkpoint on the main session
  E  (5)          another chat (the main session moves on)
  D2 (6)          D's prompt again: env off reads it all (slot 12/13 active); env on: active slot source -> RESUME
  T  (15), X (0), T2 (15): T's next turn after another chat: idle slot 15's checkpoint, slot source == admit slot (moved)
review fix (finding 1): after D2, F (6) refills slot 6 with another chat, then D3 = D's prompt as a solo GEN while slots
  12/13 still decode: with --mtp (no --batch-mtp) a solo GEN drafts from the main drafter, so the active slots are no
  source for it (RESUME 0 as env off; the S1 build gave 12)
Every request's RESUME and tokens to --dump; --compare checks them against another run (--allow-resume NAME: tokens only).
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, "/media/raid0/StrataQ38FN/Strata-142/tools")
from batch_test16 import Engine, tokenizer  # noqa: E402

KEYS = "temperature=0.7 seed=11"
RULE = ("Rule %d: answer precisely, cite the relevant section of the handbook, keep the tone neutral and never "
        "invent facts that the handbook does not state. ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--extra", default="")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--dump", required=True)
    ap.add_argument("--compare", default="")
    ap.add_argument("--allow-resume", action="append", default=[])
    ap.add_argument("--max-new", type=int, default=32)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    tok = tokenizer(cfg["tokenizer"])
    enc = lambda s: tok.encode(s, parse_special=True)
    HDR = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    user = lambda q: f"<|im_start|>user\n{q}<|im_end|>\n"
    env = dict(kv.split("=", 1) for kv in a.env)
    eng = Engine(a.exe, cfg, 16, env, a.extra.split())
    out = eng.lines()
    res, of_slot, done = {}, {}, set()
    reading = [None]
    solo = [False]

    def step():
        line = next(out)
        f = line.split()
        if not f:
            return
        if f[0] == "RESUME":
            res[reading[0]]["resume"] = int(f[1])
        elif f[0] == "T":
            res[reading[0]]["tokens"].append(int(f[1]))
        elif f[0] == "BT":
            res[of_slot[int(f[1])]]["tokens"].append(int(f[2]))
        elif f[0] == "ERR":
            print(reading[0], line)
            raise SystemExit(1)
        elif f[0] == "DONE" and solo[0]:
            done.add(reading[0])
            reading[0] = None
            solo[0] = False
        elif f[0] == "BADM":
            name = reading[0]
            reading[0] = None
            if f[2] == "0":
                done.add(name)
        elif f[0] == "BDONE":
            done.add(of_slot[int(f[1])])

    def admit(name, s, text, max_new=None, wait=True):
        ids = enc(text)
        res[name] = {"n": len(ids), "resume": None, "tokens": []}
        of_slot[s] = name
        reading[0] = name
        eng.send(f"BGEN {s} {max_new or a.max_new} {KEYS} {','.join(map(str, ids))}")
        while reading[0] is not None:
            step()
        if wait:
            wait_done(name)
        print(f"{name}: slot {s}, n={len(ids)}, RESUME {res[name]['resume']}", flush=True)

    def gen(name, text, max_new=None):
        ids = enc(text)
        res[name] = {"n": len(ids), "resume": None, "tokens": []}
        reading[0] = name
        solo[0] = True
        eng.send(f"GEN {max_new or a.max_new} {KEYS} {','.join(map(str, ids))}")
        while reading[0] is not None:
            step()
        print(f"{name}: solo GEN, n={len(ids)}, RESUME {res[name]['resume']}; D/D1 still decoding: "
              f"{'D' not in done and 'D1' not in done}", flush=True)

    def wait_done(*names):
        while not all(n in done for n in names):
            step()

    other = "<|im_start|>assistant\nA different reply than the one generated.<|im_end|>\n"
    system = "<|im_start|>system\n" + "".join(RULE % i for i in range(70)) + "<|im_end|>\n"
    admit("W0", 4, user("Explain how a B-tree works.") + HDR)
    s1 = system + user("Which rule covers citations?")
    admit("S", 3, s1 + HDR)
    s2 = s1 + other + user("And tone?")
    admit("S2", 3, s2 + HDR)
    s3 = s2 + other + user("And neutrality?")
    admit("S3", 8, s3 + HDR)
    admit("G", 9, s3 + HDR)
    admit("G2", 8, s3 + HDR)
    admit("R", 10, system + user("Summarize the rules in one line.") + HDR)
    d = user("Describe the water cycle in detail.") + HDR
    admit("D", 12, d, 400, wait=False)
    admit("D1", 13, d, 400, wait=False)
    admit("E", 5, user("Why is the sky blue?") + HDR)
    admit("D2", 6, d)
    admit("F", 6, user("Name three prime numbers.") + HDR)
    gen("D3", d)
    wait_done("D", "D1")
    t = user("What is a closure in programming?")
    admit("T", 15, t + HDR)
    admit("X", 0, user("Explain gradient descent.") + HDR)
    admit("T2", 15, t + other + user("Give an example.") + HDR)
    eng.send("QUIT")
    Path(a.dump).write_text(json.dumps(res))
    for k, v in res.items():
        print(f"  {k}: n={v['n']} RESUME {v['resume']} {len(v['tokens'])} tokens {ascii(tok.decode(v['tokens'])[:50])}")
    if a.compare:
        ref = json.loads(Path(a.compare).read_text())
        bad = [k for k in res if res[k]["tokens"] != ref.get(k, {}).get("tokens")
               or (k not in a.allow_resume and res[k]["resume"] != ref.get(k, {}).get("resume"))]
        rdiff = {k: (ref.get(k, {}).get("resume"), res[k]["resume"]) for k in res
                 if res[k]["resume"] != ref.get(k, {}).get("resume")}
        print(f"compare with {a.compare}: {len(res) - len(bad)}/{len(res)} requests identical"
              f" (tokens; RESUME except {a.allow_resume})" + (f"; DIFFER: {bad}" if bad else "")
              + (f"; RESUME ref->this: {rdiff}" if rdiff else ""))
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
