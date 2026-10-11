#!/usr/bin/env python3
"""fixedcost step AB edge cases (STRATA_ADMIT_MERGE_FIRST): admissions whose last windowed part is shorter than a
window (a prompt of a few tokens: the merged window starts at 0), longer than one (raw text of 9..64 tokens, a chat
header with an assistant prefill: the part's windows, then the last 8 tokens with the first token), the plain chat
headers (thinking off / on) and a next turn (a windowed new message, its checkpoint, then the header).  Each prompt
greedy, seeded-sampled and seeded-sampled with penalties.  One admission at a time (BGEN, then the slot runs alone to
BDONE); every request's tokens (first token + slot tokens) to --dump, compared between two runs with --compare.

  ab_edge.py --exe ... --config T16-142.json --extra "..." --dump out.json [--compare other.json] [--env K=V]
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, "/media/raid0/StrataQ38FN/Strata-142/tools")
from batch_test16 import Engine, tokenizer  # noqa: E402

KEYS = ["", "temperature=0.7 seed=11",
        "temperature=0.8 seed=5 penalty_last_n=64 penalty_repeat=1.15 penalty_present=0.3"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--extra", default="")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--dump", required=True)
    ap.add_argument("--compare", default="")
    ap.add_argument("--max-new", type=int, default=24)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    tok = tokenizer(cfg["tokenizer"])
    enc = lambda s: tok.encode(s, parse_special=True)
    raw = ("The quick brown fox jumps over the lazy dog while the committee reviews the long agenda, notes the "
           "budget, the schedule and every open risk, and asks the staff to report back next quarter with numbers.")
    rw = enc(raw)
    chat = lambda q, hdr: enc(f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{hdr}")
    prompts = {
        "raw3": rw[:3], "raw9": rw[:9], "raw40": rw[:40], "raw64": (rw * 3)[:65],
        "chat_off": chat("Explain how a B-tree works.", "<think>\n\n</think>\n\n"),
        "chat_on": chat("What is the capital of France? Answer briefly.", "<think>\n"),
        "chat_prefill": chat("Write a haiku about the sea.", "<think>\n\n</think>\n\nHere is a haiku about the sea, "
                             "in the classic five-seven-five form:\n\n"),
    }
    env = {}
    for kv in a.env:
        k, v = kv.split("=", 1)
        env[k] = v
    eng = Engine(a.exe, cfg, 16, env, a.extra.split())
    out = eng.lines()
    res, slot = {}, 0

    def admit(name, ids):
        nonlocal slot
        s = slot
        slot = (slot + 1) % 16
        eng.send(f"BGEN {s} {a.max_new} {keys} {','.join(map(str, ids))}".replace("  ", " "))
        got = []
        for line in out:
            if line.startswith("T "):
                got.append(int(line.split()[1]))
            elif line.startswith("BT "):
                f = line.split()
                if int(f[1]) == s:
                    got.append(int(f[2]))
            elif line.startswith("ERR"):
                print(name, line)
                break
            elif line.startswith("BADM ") and line.split()[2] == "0":
                break
            elif line.startswith("BDONE ") and int(line.split()[1]) == s:
                break
        res[name] = {"n": len(ids), "tokens": got}
        print(f"{name}: n={len(ids)} -> {len(got)} tokens {ascii(tok.decode(got)[:70])}", flush=True)
        return got

    for ki, keys in enumerate(KEYS):
        for name, ids in prompts.items():
            got = admit(f"{name}/k{ki}", ids)
            if name == "chat_off":   # the next turn of this chat
                nxt = ids + got + enc("<|im_end|>\n<|im_start|>user\nAnd a B+ tree?<|im_end|>\n"
                                      "<|im_start|>assistant\n<think>\n\n</think>\n\n")
                admit(f"chat_off_turn2/k{ki}", nxt)
    eng.send("QUIT")
    Path(a.dump).write_text(json.dumps(res))
    if a.compare:
        ref = json.loads(Path(a.compare).read_text())
        bad = [k for k in res if res[k] != ref.get(k)]
        print(f"compare with {a.compare}: {len(res) - len(bad)}/{len(res)} requests identical"
              + (f"; DIFFER: {bad}" if bad else ""))
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
