#!/usr/bin/env python3
"""fixedcost step K1+K2-lite edge cases (STRATA_CKPT_POOL, STRATA_CKPT_THREADS): checkpoints saved into the storage of
dropped ones and copied into a slot through it, then restored.  One admission at a time, seeded sampling:
  - warm-up chats W0..W3 (slots 4..7): every later admission prunes the previous one's turn checkpoint (pool fed);
  - A (slot 0), B (slot 1), C (slot 2): turn checkpoints saved into pooled vectors, copied into the slots;
  - A2: A's next turn with a DIFFERENT reply in the history (it matches A's slot only up to the turn checkpoint) into
    slot 0: the slot's own checkpoint is restored (RESUME = A's turn checkpoint);
  - B2: B's next turn the same way into slot 9: from slot 1's checkpoint (a slot source);
  - S: a ~2.3K-token system prompt (past --prompt-cache-root): a root and a turn checkpoint in one admission; then S2
    (S's next turn, divergent reply) restores S's turn checkpoint from its slot;
  - R: a new chat with S's system prompt: mounts on the root checkpoint (main cache).
Every request's RESUME and tokens to --dump; --compare checks them against another run.

  ckpt_edge.py --exe ... --config T16-142.json --extra "..." --dump out.json [--compare other.json]
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
    res = {}

    def admit(name, s, text):
        ids = enc(text)
        eng.send(f"BGEN {s} {a.max_new} {KEYS} {','.join(map(str, ids))}")
        got, resume = [], None
        for line in out:
            if line.startswith("RESUME "):
                resume = int(line.split()[1])
            elif line.startswith("T "):
                got.append(int(line.split()[1]))
            elif line.startswith("BT ") and int(line.split()[1]) == s:
                got.append(int(line.split()[2]))
            elif line.startswith("ERR"):
                print(name, line)
                break
            elif line.startswith("BADM ") and line.split()[2] == "0":
                break
            elif line.startswith("BDONE ") and int(line.split()[1]) == s:
                break
        res[name] = {"n": len(ids), "resume": resume, "tokens": got}
        print(f"{name}: slot {s}, n={len(ids)}, RESUME {resume} -> {len(got)} tokens {ascii(tok.decode(got)[:60])}",
              flush=True)

    qs = ["Explain how a B-tree works.", "Why is the sky blue?", "What is a closure in programming?",
          "Describe the water cycle briefly."]
    for i, q in enumerate(qs):
        admit(f"W{i}", 4 + i, user(q) + HDR)
    turn = {}
    for name, s, q in (("A", 0, "Compare TCP and QUIC in a few sentences."),
                       ("B", 1, "How does public-key cryptography work?"),
                       ("C", 2, "Explain gradient descent.")):
        turn[name] = user(q)
        admit(name, s, turn[name] + HDR)
    other = "<|im_start|>assistant\nA different reply than the one generated.<|im_end|>\n"
    admit("A2", 0, turn["A"] + other + user("Shorter, please.") + HDR)
    admit("B2", 9, turn["B"] + other + user("Give an example.") + HDR)
    system = "<|im_start|>system\n" + "".join(RULE % i for i in range(70)) + "<|im_end|>\n"
    admit("S", 3, system + user("Which rule covers citations?") + HDR)
    admit("S2", 3, system + user("Which rule covers citations?") + other + user("And tone?") + HDR)
    admit("R", 10, system + user("Summarize the rules in one line.") + HDR)
    eng.send("QUIT")
    Path(a.dump).write_text(json.dumps(res))
    if a.compare:
        ref = json.loads(Path(a.compare).read_text())
        bad = [k for k in res if res[k] != ref.get(k)]
        print(f"compare with {a.compare}: {len(res) - len(bad)}/{len(res)} requests identical (tokens and RESUME)"
              + (f"; DIFFER: {bad}" if bad else ""))
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
