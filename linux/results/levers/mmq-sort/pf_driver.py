#!/usr/bin/env python3
"""Profile driver for the T16-142fca default (no engine change).  Derived from batchadmit/scaling/scale_driver.py.

Runs the engine over stdin (--serve), logs every stdout/stderr line with a monotonic timestamp (events.log).
  --mode prefill : warm-up, then per phase (idle, dec15) admit fresh prompts of --sizes tokens into the last slot,
                   --reps times each (dec15 = all other slots decoding, refilled).
  --mode decode  : admit --batch long generations, let them decode --seconds, then BSTOP all (the engine prints its
                   idle report: rows/s and, with STRATA_VERIFY_PROFILE, the per-stage window profile).
  --nsys NAME    : the engine runs under `nsys launch --session-new=NAME`; captures (nsys start/stop) are taken around
                   the admissions named by --nsys-cap (prefill: "phase:size:rep,...") or for --nsys-sec seconds in the
                   middle of the decode stretch.
"""
import argparse, json, os, queue, random, subprocess, sys, threading, time
from pathlib import Path

sys.path.insert(0, "/media/raid0/StrataQ38FN/Strata/tools")
from batch_test import tokenizer  # noqa: E402

FILL_TOPICS = ["how NVMe SSD wear leveling works", "the CAP theorem", "TCP congestion control", "B-tree indexes",
               "virtual memory", "event-driven servers", "GPU tensor cores", "cache coherency", "Raft consensus",
               "JVM garbage collection", "register allocation", "the Linux scheduler", "TLS handshakes",
               "LSM-trees in RocksDB", "mixture-of-experts models", "speculative decoding"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--mode", choices=["prefill", "decode"], required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--sizes", default="150,1200")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--phases", default="idle,dec15")
    ap.add_argument("--pause", type=float, default=1.0)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--nsys", default="")
    ap.add_argument("--nsys-cap", default="")
    ap.add_argument("--nsys-sec", type=float, default=1.5)
    ap.add_argument("--nsys-graph", default="node")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(a.config).read_text())
    tok = tokenizer(cfg["tokenizer"])
    corpus = "\n".join(p.read_text(errors="replace") for p in
                       sorted(Path("/media/raid0/StrataQ38FN/Strata-142").rglob("*.md")) if ".git" not in p.parts)
    rng = random.Random(1234)
    args = list(cfg["args"]) + ["--layer-split", str(cfg["layer_split"]), "--batch", str(a.batch)]
    env = dict(os.environ)
    env.update(cfg.get("env") or {})
    env["STRATA_TRACE"] = "1"
    for kv in a.env:
        k, v = kv.split("=", 1)
        env[k] = v
    env["LD_LIBRARY_PATH"] = ":".join(cfg.get("lib_dirs", []) + [env.get("LD_LIBRARY_PATH", "")])
    cmd = [cfg["exe"], "--serve", *args]
    if a.nsys:
        cmd = ["nsys", "launch", f"--session-new={a.nsys}", "--trace=cuda", f"--cuda-graph-trace={a.nsys_graph}",
               "--wait=primary", *cmd]
    (out / "cmd.json").write_text(json.dumps({"cmd": cmd, "env": {
        k: env[k] for k in env if k.startswith("STRATA") or k.startswith("CUDA")}}, indent=1))
    p = subprocess.Popen(cmd, cwd=cfg.get("cwd"), stdin=subprocess.PIPE,
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

    def nsys_start(name):
        r = subprocess.run(["nsys", "start", f"--session={a.nsys}", "-o", str(out / name), "--force-overwrite=true"],
                           capture_output=True, text=True)
        mark(f"NSYS START {name} rc={r.returncode}")

    def nsys_stop(name):
        mark(f"NSYS STOP {name}")
        r = subprocess.run(["nsys", "stop", f"--session={a.nsys}"], capture_output=True, text=True)
        mark(f"NSYS STOPPED {name} rc={r.returncode} {r.stderr.strip()[-200:]}")

    while not nxt().startswith("READY"):
        pass
    mark("READY")
    uid = [0]
    WRAP0 = tok.encode("<|im_start|>user\n", parse_special=True)
    WRAP1 = tok.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", parse_special=True)

    def chat_prompt(n):
        uid[0] += 1
        head = tok.encode(f"[request {uid[0]} / prof / {os.environ.get('PF_STAMP') or time.time_ns()}] Summarize the following notes:\n")
        need = n - len(WRAP0) - len(WRAP1) - len(head)
        off = rng.randrange(0, len(corpus) - 40000)
        body = tok.encode(corpus[off:off + max(400, need * 6)])[:need]
        ids = WRAP0 + head + body + WRAP1
        assert len(ids) == n, (len(ids), n)
        return ids

    NS = a.batch
    active, free = set(), set(range(NS))
    MEAS = NS - 1

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

    def stop_all():
        for s in list(active):
            send(f"BSTOP {s}")
        while active:
            handle(nxt())

    if a.mode == "prefill":
        def refill():
            while len(active - {MEAS}) < NS - 1 and (free - {MEAS}):
                admit(min(free - {MEAS}), fill_prompt(), 12000, 0.7, "fill")

        caps = set(x for x in a.nsys_cap.split(",") if x)
        sizes = [int(x) for x in a.sizes.split(",")]
        for n in (300, 1200, 150):
            admit(MEAS, chat_prompt(n), 4, 0, "warmup")
            stop(MEAS)
        for phase in a.phases.split(","):
            if phase == "dec15":
                refill()
                mark(f"STRETCH decode-only active={len(active)}")
                run_for(6)
            for rep in range(a.reps):
                for n in sizes:
                    if phase == "dec15":
                        refill()
                        run_for(a.pause)
                    ids = chat_prompt(n)
                    tag = f"{phase}:{n}:r{rep}"
                    key = f"{phase}:{n}:{rep}"
                    if a.nsys and key in caps:
                        nsys_start(f"nsys_pf_{phase}_{n}_r{rep}")
                        run_for(0.15)
                    admit(MEAS, ids, 4, 0, tag)
                    stop(MEAS)
                    if a.nsys and key in caps:
                        run_for(0.15)
                        nsys_stop(f"nsys_pf_{phase}_{n}_r{rep}")
                    if phase == "idle":
                        run_for(0.2)
            if phase == "dec15":
                stop_all()
                run_for(1.0)
    else:
        # warm-up (one-off costs of the first reads and windows)
        for n in (300, 150):
            admit(0, chat_prompt(n), 8, 0, "warmup")
            stop(0)
        run_for(1.0)
        for s in range(NS):
            admit(s, fill_prompt(), 12000, 0.7, "fill")
        mark(f"STRETCH decode-only active={len(active)}")
        half = a.seconds / 2
        run_for(half)
        if a.nsys:
            nsys_start(f"nsys_dec_b{NS}")
            run_for(a.nsys_sec)
            nsys_stop(f"nsys_dec_b{NS}")
            run_for(half)
        else:
            run_for(half)
        mark(f"STRETCH END active={len(active)}")
        stop_all()
        run_for(2.0)
    send("QUIT")
    try:
        p.wait(timeout=300)
    except subprocess.TimeoutExpired:
        p.kill()
    mark("END")


if __name__ == "__main__":
    main()
