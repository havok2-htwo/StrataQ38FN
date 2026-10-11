#!/usr/bin/env python3
"""fixedcost step 0 / O1 driver (m0_driver.py + BYIELD arms, deterministic prompts, tokens per request). M0 baseline driver: runs the engine with the production args of a server config
(+ --batch N, --layer-split from the config), keeps N slots decoding, and admits short / long prompts into a freed
slot while the others decode.  Every stdout and stderr line is written with a monotonic timestamp to
<out>/events.log ("O"/"E" prefix) so the admission stall can be split into its phases offline (m0_analyze.py).

  m0_driver.py --config configs/T16-mmqtab.json --out DIR [--extra "..."] [--env K=V ...] [--admissions 16]
"""
import argparse, json, os, queue, subprocess, sys, threading, time
from pathlib import Path

sys.path.insert(0, "/media/raid0/StrataQ38FN/Strata/tools")
from batch_test import tokenizer  # noqa: E402

TOPICS = [
    "how NVMe SSD wear leveling works", "the CAP theorem in distributed systems",
    "TCP congestion control (CUBIC and BBR)", "why B-trees are used in database indexes",
    "virtual memory and page tables", "event-driven vs synchronous server architectures",
    "how GPU tensor cores execute matrix multiplication", "cache coherency protocols in multi-core CPUs",
    "the Raft consensus algorithm", "garbage collection in the JVM", "how a compiler performs register allocation",
    "the design of the Linux CFS/EEVDF scheduler", "public-key infrastructure and TLS handshakes",
    "how LSM-trees work in RocksDB", "mixture-of-experts language models", "speculative decoding for LLMs",
]
FILLER = (
    "The system under discussion consists of several cooperating components. Each component exposes a narrow "
    "interface, keeps its own state and communicates through explicit messages. Failures are expected and handled "
    "locally where possible; otherwise they are escalated with enough context for an operator to act. Performance "
    "is measured end to end, and every optimisation is justified by a measurement rather than by intuition. "
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--exe", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--extra", default="")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--admissions", type=int, default=16, help="measured admissions (short/long alternating)")
    ap.add_argument("--max-new", type=int, default=4000)
    ap.add_argument("--startup-only", action="store_true")
    ap.add_argument("--settle", type=float, default=4.0, help="s of decode after each admission")
    ap.add_argument("--pre", type=float, default=3.0, help="s of decode with N-1 slots before each admission")
    ap.add_argument("--byield", default="none,read,late", help="per measured admission, round robin: none | read "
                    "(BYIELD right after BGEN: queued during the read) | late (BYIELD --late-ms after BADM: groups in flight)")
    ap.add_argument("--late-ms", type=float, default=60.0)
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(a.config).read_text())
    tok = tokenizer(cfg["tokenizer"])
    args = list(cfg["args"])
    if len(cfg.get("gpu") or []) > 1 and "--layer-split" not in a.extra:
        args += ["--layer-split", str(cfg.get("layer_split") or "auto")]
    args += ["--batch", str(a.batch)] + a.extra.split()
    env = dict(os.environ)
    env.update(cfg.get("env") or {})
    for kv in a.env:
        k, v = kv.split("=", 1)
        env[k] = v
    env["LD_LIBRARY_PATH"] = ":".join(cfg.get("lib_dirs", []) + [env.get("LD_LIBRARY_PATH", "")])
    exe = a.exe or cfg["exe"]
    (out_dir / "cmd.json").write_text(json.dumps({"exe": exe, "args": args, "env": {k: env[k] for k in env
                                                  if k.startswith("STRATA") or k.startswith("CUDA")}}, indent=1))
    p = subprocess.Popen([exe, "--serve", *args], cwd=cfg.get("cwd"), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, bufsize=1, env=env)
    T0 = time.monotonic()
    ev = open(out_dir / "events.log", "w", buffering=1)
    ev_lock = threading.Lock()
    q = queue.Queue()

    def put(src, line):
        t = time.monotonic() - T0
        with ev_lock:
            ev.write(f"{t:.4f} {src} {line}\n")
        return t

    def rd_err():
        for line in p.stderr:
            put("E", line.rstrip("\n"))

    def rd_out():
        for line in p.stdout:
            line = line.rstrip("\n")
            q.put((put("O", line), line))
        q.put((time.monotonic() - T0, None))

    threading.Thread(target=rd_err, daemon=True).start()
    threading.Thread(target=rd_out, daemon=True).start()

    def send(line):
        put("S", line[:60] + (" ..." if len(line) > 60 else ""))
        p.stdin.write(line + "\n")
        p.stdin.flush()

    def mark(text):
        put("M", text)
        print(f"[{time.monotonic() - T0:8.2f}] {text}", flush=True)

    def nxt(timeout=None):
        t, l = q.get(timeout=timeout)
        if l is None:
            raise SystemExit("engine ended (see events.log)")
        return t, l

    while True:
        _, l = nxt()
        if l.startswith("READY"):
            break
    mark("READY")
    if a.startup_only:
        send("QUIT")
        p.wait(timeout=300)
        return 0

    uid = [0]

    def prompt(cls):
        uid[0] += 1
        n_fill = 16 if cls == "long" else 1
        doc = FILLER * n_fill
        text = (f"[request {uid[0]} / fc]\nBackground notes:\n{doc}\n\n"
                f"Task: write a long, thorough and well-structured technical explanation of "
                f"{TOPICS[uid[0] % len(TOPICS)]}. Use several sections and concrete examples. Do not stop early.")
        return tok.encode(f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                          parse_special=True)

    active, free = set(), set(range(a.batch))
    slot_uid, toks = {}, {}

    def handle(l):
        if l.startswith("BT "):
            f = l.split()
            u = slot_uid.get(int(f[1]))
            if u is not None:
                toks[u].append(int(f[2]))
        if l.startswith("BDONE "):
            s = int(l.split()[1])
            active.discard(s)
            free.add(s)
        elif l.startswith("ERR"):
            mark("ENGINE " + l)

    def admit(slot, cls, by="none"):
        ids = prompt(cls)
        u = uid[0]
        toks[u] = []
        mark(f"ADMIT slot={slot} cls={cls} n={len(ids)} active={len(active)} byield={by}")
        send(f"BGEN {slot} {a.max_new} temperature=0.7 seed={uid[0]} {','.join(map(str, ids))}")
        if by == "read":
            send(f"BYIELD {slot}")
        free.discard(slot)
        slot_uid.pop(slot, None)
        while True:
            _, l = nxt()
            if l.startswith("T "):
                toks[u].append(int(l[2:]))
            if l.startswith("BADM "):
                f = l.split()
                if int(f[1]) == slot:
                    if f[2] == "1":
                        active.add(slot)
                        slot_uid[slot] = u
                    else:
                        free.add(slot)
                    mark(f"ADMITTED slot={slot} cont={f[2]}")
                    if by == "late":
                        run_for(a.late_ms / 1000.0)
                        mark(f"LATE BYIELD slot={slot}")
                        send(f"BYIELD {slot}")
                    return
            if l.startswith("ERR"):
                mark(f"ADMIT FAILED slot={slot}: {l}")
                free.add(slot)
                return
            handle(l)

    def run_for(sec):
        end = time.monotonic() + sec
        while time.monotonic() < end:
            try:
                _, l = nxt(timeout=max(0.01, end - time.monotonic()))
            except queue.Empty:
                break
            handle(l)

    def refill_spontaneous(keep=None):
        # slots that ended on their own (EOS / max_new): back in with a short prompt, flagged; `keep` stays free
        while len(active) < a.batch - 1 and (free - {keep}):
            admit(min(free - {keep}), "short-refill")

    # fill N-1 slots
    for s in range(a.batch - 1):
        admit(s, "fill")
    mark(f"STRETCH decode-only active={len(active)}")
    run_for(8)
    mark("STRETCH end")
    admit(a.batch - 1, "fill")
    mark(f"STRETCH decode-only active={len(active)}")
    run_for(8)
    mark("STRETCH end")
    victim = 0
    modes = a.byield.split(",")
    for i in range(a.admissions):
        cls = "short" if i % 2 == 0 else "long"
        by = modes[i % len(modes)]
        refill_spontaneous()
        while victim not in active:
            victim = (victim + 1) % a.batch
        v = victim
        victim = (victim + 1) % a.batch
        send(f"BSTOP {v}")
        while v in active:
            _, l = nxt()
            handle(l)
        refill_spontaneous(keep=v)
        mark(f"STRETCH decode-only active={len(active)}")
        run_for(a.pre)
        mark("STRETCH end")
        admit(v, cls, by)
        mark(f"STRETCH decode-only active={len(active)}")
        run_for(a.settle)
        mark("STRETCH end")
    for s in list(active):
        send(f"BSTOP {s}")
    while active:
        _, l = nxt()
        handle(l)
    send("QUIT")
    try:
        p.wait(timeout=300)
    except subprocess.TimeoutExpired:
        p.kill()
    mark("END")
    (out_dir / "tokens.json").write_text(json.dumps(toks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
