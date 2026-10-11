#!/usr/bin/env python3
"""[pipecheck variant: --prefill override, chunk-multiple prompts] Prefill read-time scaling vs token count (batch-admission bound).  No engine change.

Runs the engine (T16-142fc args, --batch 16) over stdin, admits fresh prompts of fixed token counts into slot 15
and logs every stdout/stderr line with a monotonic timestamp (events.log).  Two phases:
  idle : no other slot decoding
  dec15: slots 0..14 decoding (long generations, refilled if they end), slot 15 measured
Conditions: single prompts of N tokens, and "concat" prompts = k distinct chat prompts of N tokens back-to-back in
ONE token sequence (one read; stands in for reading k queued prompts together: same routing diversity).
Text comes from slices of the repo's Markdown corpus (diverse routing, unlike a repeated filler); every prompt starts
with a unique header, so no prefix-cache reuse.
  scale_driver.py --config CFG --out DIR [--env K=V ...] [--reps 4] [--phases idle,dec15]
"""
import argparse, json, os, queue, random, subprocess, sys, threading, time
from pathlib import Path

sys.path.insert(0, "/media/raid0/StrataQ38FN/Strata/tools")
from batch_test import tokenizer  # noqa: E402

SINGLE = [int(x) for x in __import__("os").environ["PC_SINGLE"].split(",")]
CONCAT = []
FILL_TOPICS = ["how NVMe SSD wear leveling works", "the CAP theorem", "TCP congestion control", "B-tree indexes",
               "virtual memory", "event-driven servers", "GPU tensor cores", "cache coherency", "Raft consensus",
               "JVM garbage collection", "register allocation", "the Linux scheduler", "TLS handshakes",
               "LSM-trees in RocksDB", "mixture-of-experts models", "speculative decoding"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--phases", default="idle,dec15")
    ap.add_argument("--pause", type=float, default=1.0, help="s of decode between measured admissions (dec15)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(a.config).read_text())
    tok = tokenizer(cfg["tokenizer"])
    corpus = "\n".join(p.read_text(errors="replace") for p in
                       sorted(Path("/media/raid0/StrataQ38FN/Strata-142").rglob("*.md")) if ".git" not in p.parts)
    rng = random.Random(1234)
    args = list(cfg["args"]); args[args.index("--prefill") + 1] = __import__("os").environ["PC_PREFILL"]; args += __import__("os").environ.get("PC_EXTRA", "").split(); args += ["--layer-split", str(cfg["layer_split"]), "--batch", "16"]
    env = dict(os.environ)
    env.update(cfg.get("env") or {})
    env["STRATA_TRACE"] = "1"
    for kv in a.env:
        k, v = kv.split("=", 1)
        env[k] = v
    env["LD_LIBRARY_PATH"] = ":".join(cfg.get("lib_dirs", []) + [env.get("LD_LIBRARY_PATH", "")])
    (out / "cmd.json").write_text(json.dumps({"exe": cfg["exe"], "args": args, "env": {
        k: env[k] for k in env if k.startswith("STRATA") or k.startswith("CUDA")}}, indent=1))
    p = subprocess.Popen([cfg["exe"], "--serve", *args], cwd=cfg.get("cwd"), stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, env=env)
    T0 = time.monotonic()
    ev = open(out / "events.log", "w", buffering=1)
    lk = threading.Lock()
    q = queue.Queue()

    def put(src, line):
        with lk:
            ev.write(f"{time.monotonic() - T0:.4f} {src} {line}\n")

    def rd_err():
        for line in p.stderr:
            put("E", line.rstrip("\n"))

    def rd_out():
        for line in p.stdout:
            line = line.rstrip("\n")
            put("O", line)
            q.put(line)
        q.put(None)

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
        l = q.get(timeout=timeout)
        if l is None:
            raise SystemExit("engine ended (see events.log)")
        return l

    while not nxt().startswith("READY"):
        pass
    mark("READY")
    uid = [0]
    WRAP0 = tok.encode("<|im_start|>user\n", parse_special=True)
    WRAP1 = tok.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", parse_special=True)

    def chat_prompt(n):
        uid[0] += 1
        head = tok.encode(f"[request {uid[0]} / scale / {time.time_ns()}] Summarize the following notes:\n")
        need = n - len(WRAP0) - len(WRAP1) - len(head)
        off = rng.randrange(0, len(corpus) - 40000)
        body = tok.encode(corpus[off:off + max(400, need * 6)])[:need]
        ids = WRAP0 + head + body + WRAP1
        assert len(ids) == n, (len(ids), n)
        return ids

    active, free = set(), set(range(16))
    MEAS = 15

    def handle(l):
        if l.startswith("BDONE "):
            s = int(l.split()[1])
            active.discard(s)
            free.add(s)
        elif l.startswith("ERR"):
            mark("ENGINE " + l)

    def admit(slot, ids, max_new, temp, tag):
        mark(f"ADMIT slot={slot} tag={tag} n={len(ids)} active={len(active)}")
        send(f"BGEN {slot} {max_new} temperature={temp} seed={uid[0]} {','.join(map(str, ids))}")
        free.discard(slot)
        while True:
            l = nxt()
            if l.startswith("BADM "):
                f = l.split()
                if int(f[1]) == slot:
                    (active.add if f[2] == "1" else free.add)(slot)
                    mark(f"ADMITTED slot={slot} cont={f[2]}")
                    return
            if l.startswith("ERR"):
                mark(f"ADMIT FAILED slot={slot}: {l}")
                free.add(slot)
                return
            handle(l)

    def stop(slot):
        if slot in active:
            send(f"BSTOP {slot}")
        while slot in active:
            handle(nxt())

    def run_for(sec):
        end = time.monotonic() + sec
        while time.monotonic() < end:
            try:
                l = nxt(timeout=max(0.01, end - time.monotonic()))
            except queue.Empty:
                break
            handle(l)

    def fill_prompt():
        uid[0] += 1
        t = (f"[request {uid[0]} / fill / {time.time_ns()}]\nWrite a long, thorough technical explanation of "
             f"{FILL_TOPICS[uid[0] % len(FILL_TOPICS)]}. Use many sections and examples. Do not stop early.")
        return tok.encode(f"<|im_start|>user\n{t}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                          parse_special=True)

    def refill():
        while len(active - {MEAS}) < 15 and (free - {MEAS}):
            admit(min(free - {MEAS}), fill_prompt(), 12000, 0.7, "fill")

    conds = [("single", 1, n) for n in SINGLE] + [("concat", k, n) for k, n in CONCAT]
    # warm-up (first reads pay one-off costs)
    for n in (300, 1200, 150):
        admit(MEAS, chat_prompt(n), 4, 0, "warmup")
        stop(MEAS)
    for phase in a.phases.split(","):
        if phase == "dec15":
            refill()
            mark(f"STRETCH decode-only active={len(active)}")
            run_for(6)
        for rep in range(a.reps):
            order = list(conds)
            random.Random(100 + rep).shuffle(order)
            for kind, k, n in order:
                if phase == "dec15":
                    refill()
                    run_for(a.pause)
                ids = []
                for _ in range(k):
                    ids += chat_prompt(n)
                tag = f"{phase}:{kind}:{k}x{n}:r{rep}"
                admit(MEAS, ids, 4, 0, tag)
                stop(MEAS)
                if phase == "idle":
                    run_for(0.2)
        if phase == "dec15":
            for s in list(active):
                send(f"BSTOP {s}")
            while active:
                handle(nxt())
    send("QUIT")
    try:
        p.wait(timeout=300)
    except subprocess.TimeoutExpired:
        p.kill()
    mark("END")


if __name__ == "__main__":
    main()
