#!/usr/bin/env python3
"""review fix (finding 2): STRATA_CKPT_THREADS workers under --aux-cpus auto, then a request with aux_cpus=0.
Prints every thread's CPU list at READY, after the first admission (the checkpoint threads exist then) and after a
request that turns aux off: the threads new since READY must not end on the host thread's one CPU."""
import argparse, json, os, sys
from pathlib import Path

sys.path.insert(0, "/media/raid0/StrataQ38FN/Strata-142/tools")
from batch_test16 import Engine, tokenizer  # noqa: E402


def cpus(pid):
    m = {}
    for t in os.listdir(f"/proc/{pid}/task"):
        try:
            for l in open(f"/proc/{pid}/task/{t}/status"):
                if l.startswith("Cpus_allowed_list:"):
                    m[int(t)] = l.split()[1]
        except OSError:
            pass
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--extra", default="")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    tok = tokenizer(cfg["tokenizer"])
    enc = lambda s: tok.encode(s, parse_special=True)
    eng = Engine(a.exe, cfg, 16, {}, a.extra.split())
    out = eng.lines()
    pid = eng.p.pid
    ready = cpus(pid)
    print("at READY: single-CPU threads", sorted({v for v in ready.values() if "-" not in v and "," not in v}))

    def bgen(s, q, keys=""):
        ids = enc(f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        eng.send(f"BGEN {s} 8 {keys} {','.join(map(str, ids))}".replace("  ", " "))
        adm = False
        for line in out:
            f = line.split()
            if f and f[0] == "ERR":
                raise SystemExit(line)
            if f and f[0] == "BADM":
                if f[2] == "0":
                    return
                adm = True
            if adm and f and f[0] == "BDONE":
                return

    bgen(0, "Explain how a B-tree works.")
    after1 = cpus(pid)
    new = sorted(set(after1) - set(ready))
    print("threads new after the first admission:", {t: after1[t] for t in new})
    bgen(1, "Why is the sky blue?", "aux_cpus=0")
    after2 = cpus(pid)
    print("after the aux_cpus=0 request:        ", {t: after2.get(t) for t in new})
    host = [v for t, v in ready.items() if t != pid and "-" not in v and "," not in v]
    print("single-CPU masks at READY (host thread):", host)
    bad = [t for t in new if after2.get(t) in host]
    print("RESULT:", "FAIL, on the host's CPU:" if bad else "PASS", bad)
    eng.send("QUIT")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
