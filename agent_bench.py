"""Agentic A/B runner for Strata: start a config variant on its own port, replay an agent-like workload, stop it.

The workload mimics the live traffic: a main agent with a long context (~80K tokens) that grows turn by turn,
a sub-agent with a different ~18K context in between (ping-pong), and long decodes on top of the long context.

usage: python agent_bench.py NAME [--set flag=VALUE] [--add flag[=VALUE]] [--drop flag] [--env K=V] [--port 8090]
                             [--cfg key=JSON] [--server-arg ARG] [--tune JSON] [--workload agent|session|parallel]
                             [--decode-runs 2] [--keep]
Appends one JSON line per arm to agent_results.jsonl; engine log in logs/engine-NAME.log.
"""
import argparse
import ctypes
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
STRATA = HERE / "Strata"
PY = STRATA / ".venv" / "Scripts" / "python.exe"
sys.path.insert(0, str(HERE))
from run_variant import edit_args, kill_port, wait_ready  # noqa: E402
from bench_strata import PROMPTS  # noqa: E402


# ---------------------------------------------------------------- system memory sampling (commit is the bottleneck)
class PERFINFO(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("CommitTotal", ctypes.c_size_t), ("CommitLimit", ctypes.c_size_t),
                ("CommitPeak", ctypes.c_size_t), ("PhysicalTotal", ctypes.c_size_t),
                ("PhysicalAvailable", ctypes.c_size_t), ("SystemCache", ctypes.c_size_t),
                ("KernelTotal", ctypes.c_size_t), ("KernelPaged", ctypes.c_size_t),
                ("KernelNonpaged", ctypes.c_size_t), ("PageSize", ctypes.c_size_t),
                ("HandleCount", ctypes.c_ulong), ("ProcessCount", ctypes.c_ulong), ("ThreadCount", ctypes.c_ulong)]


def mem_now():
    pi = PERFINFO()
    pi.cb = ctypes.sizeof(pi)
    ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(pi), pi.cb)
    g = pi.PageSize / 2**30
    return {"commit_free_gib": round((pi.CommitLimit - pi.CommitTotal) * g, 2),
            "phys_avail_gib": round(pi.PhysicalAvailable * g, 2)}


def vram_free_mib():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return int(r.stdout.split()[0])
    except Exception:
        return None


class MemSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.samples, self.stop = [], False

    def run(self):
        while not self.stop:
            self.samples.append(mem_now())
            time.sleep(2)

    def summary(self):
        if not self.samples:
            return {}
        return {"commit_free_min_gib": min(s["commit_free_gib"] for s in self.samples),
                "phys_avail_min_gib": min(s["phys_avail_gib"] for s in self.samples)}


# ---------------------------------------------------------------- workload
def corpus_main():
    # ~80K tokens of English technical prose: the repo docs
    parts = []
    for p in sorted((STRATA / "docs").glob("*.md")):
        parts.append(f"=== FILE docs/{p.name} ===\n" + p.read_text(encoding="utf-8", errors="replace"))
    return "\n\n".join(parts)[:300_000]


def corpus_sub():
    # ~18K tokens of C++: a different context, like a sub-agent reading code
    return (STRATA / "src" / "core" / "expert_source.cpp").read_text(encoding="utf-8", errors="replace")[:60_000]


def tool_result(i, n_chars):
    # a deterministic "tool output" suffix (grep-like lines)
    lines, k = [], 0
    while sum(len(x) for x in lines) < n_chars:
        lines.append(f"src/file_{(i * 31 + k) % 97}.cpp:{(k * 17 + i) % 4000}: cache_slot[{k}] = lookup(expert {k * 7 % 512}, layer {k % 48});")
        k += 1
    return "Tool result (grep):\n" + "\n".join(lines)


SYS_MAIN = "You are a senior engineer agent working on the Strata inference engine. Answer precisely and briefly."
SYS_SUB = "You are a code-reading sub-agent. You read C++ and report facts with line references. Be terse."


TUNE = None   # --tune JSON: strata_tune keys sent with every request (e.g. {"pcie_frac": 0.7})


def chat(url, messages, max_tokens):
    req_body = {"model": "strata", "messages": messages, "temperature": 0, "reasoning_effort": "none",
                "max_tokens": max_tokens, "stream": True, "stream_options": {"include_usage": True}}
    if TUNE:
        req_body["strata_tune"] = TUNE
    body = json.dumps(req_body).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = last = None
    usage, text, n = None, [], 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                d = ch.get("delta", {})
                piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
                if piece:
                    now = time.perf_counter()
                    first = first or now
                    last = now
                    n += 1
                    text.append(piece)
    t_end = time.perf_counter()
    out_tok = (usage or {}).get("completion_tokens") or n
    dec = (last - first) if first and last and last > first else None
    return {"ttft_s": round((first or t_end) - t0, 3), "decode_tps": round((out_tok - 1) / dec, 2) if dec else None,
            "out_tokens": out_tok, "prompt_tokens": (usage or {}).get("prompt_tokens"), "text": "".join(text)}


def workload(url, decode_runs):
    rows = []

    def rec(step, r):
        r = dict(r)
        r.pop("text", None)
        r["step"] = step
        rows.append(r)
        print(f"   {step:14s} prompt {r['prompt_tokens']!s:>6} TTFT {r['ttft_s']:7.2f}s decode {r['decode_tps']!s:>7} tok/s "
              f"out {r['out_tokens']}", flush=True)

    # warm-up (not recorded): graph capture, first-request startup prefill
    chat(url, [{"role": "user", "content": "Write a short poem about the sea."}], 128)
    chat(url, [{"role": "user", "content": "Explain recursion with an example in Python."}], 128)

    # D: short-context decode
    for name in ("chat", "code", "de"):
        for i in range(decode_runs):
            rec(f"dec-{name}{i}", chat(url, [{"role": "user", "content": f"[run {i}] " + PROMPTS[name]}], 768))

    main = [{"role": "system", "content": SYS_MAIN},
            {"role": "user", "content": corpus_main() + "\n\nQuestion: list the expert cache tiers Strata has, one line each."}]
    sub = [{"role": "system", "content": SYS_SUB},
           {"role": "user", "content": corpus_sub() + "\n\nWhat does FileExpertSource do with a RAM budget? 3 bullets."}]

    def turn(conv, step, i, max_tokens, suffix_chars=6000):
        r = chat(url, conv, max_tokens)
        rec(step, r)
        conv.append({"role": "assistant", "content": r["text"]})
        conv.append({"role": "user", "content": tool_result(i, suffix_chars) + "\n\nContinue: what next? Be brief."})

    turn(main, "M0-cold", 0, 200)          # cold ~80K prefill
    turn(main, "M1-append", 1, 200)        # same conversation + suffix: prefix reuse
    turn(sub, "S0-cold", 2, 200)           # sub-agent: evicts main's KV unless parked
    turn(main, "M2-return", 3, 200)        # return to main: re-prefill or restore
    turn(sub, "S1-return", 4, 200)
    turn(main, "M3-return", 5, 200)
    turn(main, "M4-longdec", 6, 1500)      # long decode at ~85K context
    return rows


def slot(url, action, filename):
    body = json.dumps({"filename": filename}).encode()
    req = urllib.request.Request(f"{url}/slots/0?action={action}", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            out = json.load(r)
    except urllib.error.HTTPError as e:
        out = {"http_error": e.code, "body": e.read().decode("utf-8", "replace")[:400]}
    out["wall_s"] = round(time.perf_counter() - t0, 3)
    print(f"   slot {action:7s} {filename:14s} -> {json.dumps(out)[:300]}", flush=True)
    return out


def session_workload(url, restart):
    """KV to SSD: save the main agent's conversation to X: and C:, overwrite it with a sub-agent, restore it from
    each drive and continue; then restart the engine and restore it from X: (vs the cold first read)."""
    rows, slots = [], {}

    def rec(step, r):
        r = dict(r)
        text = r.pop("text", "")
        r["step"], r["head"] = step, text[:120].replace("\n", " ")
        rows.append(r)
        print(f"   {step:14s} prompt {r['prompt_tokens']!s:>6} TTFT {r['ttft_s']:7.2f}s decode {r['decode_tps']!s:>7} "
              f"out {r['out_tokens']} | {r['head'][:70]}", flush=True)
        return text

    chat(url, [{"role": "user", "content": "Write a short poem about the sea."}], 128)
    main = [{"role": "system", "content": SYS_MAIN},
            {"role": "user", "content": corpus_main() + "\n\nQuestion: list the expert cache tiers Strata has, one line each."}]
    sub = [{"role": "system", "content": SYS_SUB},
           {"role": "user", "content": corpus_sub() + "\n\nWhat does FileExpertSource do with a RAM budget? 3 bullets."}]
    t = rec("M0-cold", chat(url, main, 200))
    main += [{"role": "assistant", "content": t}, {"role": "user", "content": tool_result(1, 6000) + "\n\nContinue: what next? Be brief."}]
    # the drive is decided by --server-arg --slot-save-path DIR (one arm per drive)
    slots["save"] = slot(url, "save", "main.ses")
    slots["save-again"] = slot(url, "save", "main2.ses")   # second save: warm file cache vs new file
    rec("S0-cold", chat(url, sub, 200))
    slots["restore"] = slot(url, "restore", "main.ses")
    rec("M1-restored", chat(url, main, 200))               # must reuse ~91K
    url = restart()
    chat(url, [{"role": "user", "content": "Write a short poem about the sea."}], 128)
    slots["restore-after-restart"] = slot(url, "restore", "main.ses")
    rec("M1-after-restart", chat(url, main, 200))
    return rows, slots


# ---------------------------------------------------------------- engine log parsing
RE_PROMPT = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms .*?, (\d+) generated in (\d+) ms "
                       r"\(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)")
RE_HIT = re.compile(r"decode expert cache hit rate: ([\d.]+)% \((\d+) hits / (\d+) lookups\); (\d+) more read by the GPU "
                    r"over PCIe.*?of all (\d+) routed")
RE_FILES = re.compile(r"files (\d+) blobs ([\d.]+) MB read")
RE_EXCH = re.compile(r"(\d+) exchanged with the VRAM tier")


def parse_log(path):
    reqs, cur = [], None
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        m = RE_PROMPT.search(line)
        if m:
            cur = dict(zip(["prompt", "reused", "read", "read_ms", "gen", "gen_ms", "eng_tps", "acc", "drafts"],
                           [float(x) for x in m.groups()]))
            reqs.append(cur)
            continue
        if cur is None:
            continue
        m = RE_HIT.search(line)
        if m:
            hits, _, pcie, routed = (float(m.group(i)) for i in (2, 3, 4, 5))
            cur["vram_share"] = hits / routed if routed else None
            cur["pcie_share"] = pcie / routed if routed else None
        m = RE_FILES.search(line)
        if m:
            cur["files_blobs"], cur["files_mb"] = float(m.group(1)), float(m.group(2))
        m = RE_EXCH.search(line)
        if m:
            cur["exchanged"] = float(m.group(1))
    return reqs


def diverge_workload(url):
    """Partial prefix reuse: the context diverges late (last message replaced), at ~75% and at ~50%.
    Each step shares its prefix with the step before, so the reusable part is set by the checkpoints."""
    rows = []

    def rec(step, r):
        r = dict(r)
        r.pop("text", None)
        r["step"] = step
        rows.append(r)
        print(f"   {step:14s} prompt {r['prompt_tokens']!s:>6} TTFT {r['ttft_s']:7.2f}s", flush=True)

    chat(url, [{"role": "user", "content": "Write a short poem about the sea."}], 128)
    base = corpus_main()
    alt = (STRATA / "src" / "core" / "verify.cpp").read_text(encoding="utf-8", errors="replace")
    q = "\n\nQuestion: list the expert cache tiers Strata has, one line each."

    def conv(doc, last):
        return [{"role": "system", "content": SYS_MAIN}, {"role": "user", "content": doc + q},
                {"role": "assistant", "content": "Noted. Ready for the tool results."},
                {"role": "user", "content": last + "\n\nContinue: what next? Be brief."}]

    n = len(base)
    rec("D0-cold", chat(url, conv(base, tool_result(1, 6000)), 120))
    rec("D1-same", chat(url, conv(base, tool_result(1, 6000)), 120))            # identical prompt: full reuse
    rec("D2-edit-last", chat(url, conv(base, tool_result(2, 6000)), 120))       # last message replaced (~94%)
    cut75 = int(n * 0.75)
    doc75 = base[:cut75] + "\n=== FILE src/core/verify.cpp ===\n" + alt[: n - cut75]
    rec("D3-div75", chat(url, conv(doc75, tool_result(2, 6000)), 120))          # diverges at ~75% of the doc
    cut50 = int(n * 0.50)
    doc50 = base[:cut50] + "\n=== FILE src/core/verify.cpp ===\n" + alt[: n - cut50]
    rec("D4-div50", chat(url, conv(doc50, tool_result(2, 6000)), 120))          # diverges at ~50%
    rec("D5-edit-last", chat(url, conv(doc50, tool_result(3, 6000)), 120))      # again: last message replaced
    return rows


NEEDLE = "IMPORTANT FACT: the maintenance code of the Kestrel cluster is 7319-KESTREL."


def corpus_long():
    """~380K tokens: all docs, then the C++ sources; deterministic order."""
    parts = []
    for p in sorted((STRATA / "docs").glob("*.md")):
        parts.append(f"=== FILE docs/{p.name} ===\n" + p.read_text(encoding="utf-8", errors="replace"))
    for sub in ("core", "kernels", "prefill", "spec", "program"):
        for p in sorted((STRATA / "src" / sub).glob("*.c*")):
            parts.append(f"=== FILE src/{sub}/{p.name} ===\n" + p.read_text(encoding="utf-8", errors="replace"))
    return "\n\n".join(parts)


def longctx_workload(url, max_ctx, decode_runs):
    """Context-size sweep: short decodes, then growing prompts (~115K / ~185K / ~250K tokens) with a needle at
    40%, each prompt a prefix-extension of the one before (reuse via checkpoints), plus a forced long answer."""
    rows = []

    def rec(step, r, extra=None):
        r = dict(r)
        text = r.pop("text", "")
        r["step"] = step
        r.update(extra or {})
        rows.append(r)
        print(f"   {step:14s} prompt {r['prompt_tokens']!s:>6} TTFT {r['ttft_s']:7.2f}s decode {r['decode_tps']!s:>7} "
              f"out {r['out_tokens']} {extra or ''}", flush=True)
        return text

    chat(url, [{"role": "user", "content": "Write a short poem about the sea."}], 128)
    chat(url, [{"role": "user", "content": "Explain recursion with an example in Python."}], 128)
    for name in ("chat", "code", "de"):
        for i in range(decode_runs):
            rec(f"dec-{name}{i}", chat(url, [{"role": "user", "content": f"[run {i}] " + PROMPTS[name]}], 768))

    full = corpus_long()
    needle_at = int(380_000 * 0.4)   # inside the first (shared) part of every prompt
    for label, chars in (("L115k", 380_000), ("L180k", 580_000), ("L240k", 790_000)):
        est_tokens = chars / 3.3
        if est_tokens > max_ctx - 3000:
            print(f"   {label}: skipped (~{est_tokens:.0f} tokens > --max-context {max_ctx})", flush=True)
            continue
        doc = full[:needle_at] + "\n" + NEEDLE + "\n" + full[needle_at:chars]
        msgs = [{"role": "system", "content": SYS_MAIN},
                {"role": "user", "content": doc + "\n\nQuestion: what is the maintenance code of the Kestrel cluster? "
                                                  "Answer with the code only."}]
        try:
            r = chat(url, msgs, 40)
            rec(f"{label}-needle", r, {"found": "7319" in r["text"]})
            msgs2 = msgs[:1] + [{"role": "user", "content": doc + "\n\nWrite a detailed, 500-word overview of the "
                                                                  "engine described above, section by section."}]
            rec(f"{label}-longdec", chat(url, msgs2, 700))
        except Exception as e:
            print(f"   {label}: ERROR {e!r}", flush=True)
            rows.append({"step": f"{label}-error", "error": repr(e), "ttft_s": None, "decode_tps": None,
                         "prompt_tokens": None, "out_tokens": None})
    return rows


def parallel_workload(url, rounds=(1, 2, 3)):
    """Batching: N concurrent decodes (different topics), aggregate generated tok/s per round."""
    import concurrent.futures
    chat(url, [{"role": "user", "content": "Write a short poem about the sea."}], 128)
    topics = ["how NVMe SSD wear leveling works", "the CAP theorem", "how TCP cubic congestion control works",
              "why B-trees are used in database indexes"]
    rows = []
    for n in rounds:
        for rep in range(2):
            def one(i):
                return chat(url, [{"role": "user", "content": f"[r{n}.{rep}.{i}] Explain in detail {topics[i % 4]}."}], 600)
            t0 = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
                res = list(ex.map(one, range(n)))
            wall = time.perf_counter() - t0
            gen = sum(r["out_tokens"] for r in res)
            row = {"step": f"par{n}-{rep}", "n": n, "wall_s": round(wall, 2), "agg_tps": round(gen / wall, 1),
                   "per_stream_tps": [r["decode_tps"] for r in res], "ttft_s": max(r["ttft_s"] for r in res),
                   "prompt_tokens": None, "decode_tps": round(gen / wall, 1), "out_tokens": gen}
            rows.append(row)
            print(f"   par N={n} rep{rep}: aggregate {row['agg_tps']} tok/s, per stream {row['per_stream_tps']}", flush=True)
    return rows


def main():
    global TUNE
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--base", default="strata-unsloth-ud-iq4_xs.json")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--add", action="append", default=[])
    ap.add_argument("--drop", action="append", default=[])
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--cfg", action="append", default=[], help="config key=JSON value, e.g. parallel=2")
    ap.add_argument("--server-arg", action="append", default=[], help="extra server.py argument (repeat)")
    ap.add_argument("--tune", default="", help="strata_tune JSON sent with every request")
    ap.add_argument("--workload", default="agent", choices=["agent", "session", "parallel", "diverge", "longctx"])
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--decode-runs", type=int, default=2)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    TUNE = json.loads(a.tune) if a.tune else None

    cfg = json.loads((STRATA / a.base).read_text(encoding="utf-8"))
    cfg["args"] = edit_args(cfg["args"], a.set, a.add, a.drop)
    for kv in a.cfg:
        k, v = kv.split("=", 1)
        cfg[k] = json.loads(v)
    log = HERE / "logs" / f"engine-{a.name}.log"
    log.parent.mkdir(exist_ok=True)
    if log.exists():
        log.unlink()
    cfg["log"] = str(log)
    cfg["port"] = a.port
    cfg_path = HERE / "configs" / f"strata-{a.name}.json"
    cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")

    env = dict(os.environ, PYTHONUNBUFFERED="1", STRATA_DECODE_TIMING="1")
    for kv in a.env:
        k, v = kv.split("=", 1)
        env[k] = v

    url = f"http://127.0.0.1:{a.port}"
    out = open(HERE / "logs" / f"server-{a.name}.out", "w", encoding="utf-8")
    state = {}

    def start():
        kill_port(a.port)
        t0 = time.time()
        proc = subprocess.Popen([str(PY), str(STRATA / "serve" / "server.py"), "--engine", "strata", "--config",
                                 str(cfg_path), "--port", str(a.port), "--host", "127.0.0.1"] + a.server_arg,
                                cwd=str(STRATA), env=env, stdout=out, stderr=subprocess.STDOUT)
        ok = wait_ready(url, proc)
        state.setdefault("load_s", round(time.time() - t0))
        print(f"[{a.name}] ready={ok} after {time.time() - t0:.0f}s", flush=True)
        if not ok:
            out.flush()
            print(open(HERE / "logs" / f"server-{a.name}.out", encoding="utf-8", errors="replace").read()[-3000:])
            kill_port(a.port)
            sys.exit(1)
        time.sleep(15)
        return url

    mem_before = mem_now()
    print(f"[{a.name}] changes: set={a.set} add={a.add} drop={a.drop} env={a.env} cfg={a.cfg} "
          f"server={a.server_arg} tune={a.tune} workload={a.workload}", flush=True)
    start()
    mem_loaded, vram_loaded = mem_now(), vram_free_mib()
    keys = re.compile(r"expert cache \d+ slots|RAM budget|resident RAM mode|file tier reads|borrows|batch|"
                      r"lendable|conversation cache|VRAM free with|prefetch|lookahead|row cache|kv |WARNING|error|failed", re.I)
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if keys.search(line) and "verify:" not in line:
            print("   log:", line[:260], flush=True)
    out.flush()
    for line in open(HERE / "logs" / f"server-{a.name}.out", encoding="utf-8", errors="replace"):
        if "slot save" in line or "batch" in line.lower():
            print("   srv:", line.strip()[:260], flush=True)

    sampler = MemSampler()
    sampler.start()
    slots, err, rows = None, None, []
    try:
        if a.workload == "agent":
            rows = workload(url, a.decode_runs)
        elif a.workload == "session":
            rows, slots = session_workload(url, start)
        elif a.workload == "diverge":
            rows = diverge_workload(url)
        elif a.workload == "longctx":
            mc = cfg["args"][cfg["args"].index("--max-context") + 1] if "--max-context" in cfg["args"] else "131072"
            rows = longctx_workload(url, int(mc), a.decode_runs)
        else:
            rows = parallel_workload(url)
    except Exception as e:  # keep what we have
        err = repr(e)
        print("   WORKLOAD ERROR:", err, flush=True)
    sampler.stop = True
    time.sleep(3)
    eng = parse_log(log)
    if a.workload == "agent":
        # the engine log has one entry per request incl. 2 warm-ups first
        for r, e in zip(rows, eng[2:]):
            r["eng"] = e
    eng_w = eng[2:] if a.workload == "agent" else eng
    if a.workload != "agent":
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if (" reused " in line or "session" in line.lower() or "batch" in line.lower()) and "verify:" not in line:
                print("   eng:", line.replace("strata serve: ", "")[:240], flush=True)
    dec = [r["decode_tps"] for r in rows if r["step"].startswith("dec-") and r["decode_tps"]]
    by = {r["step"]: r for r in rows}
    summary = {
        "name": a.name, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "set": a.set, "add": a.add, "drop": a.drop,
        "env": a.env, "cfg": a.cfg, "server_arg": a.server_arg, "tune": a.tune, "workload": a.workload,
        "error": err, "load_s": state.get("load_s"), "vram_free_loaded_mib": vram_loaded,
        "mem_before": mem_before, "mem_loaded": mem_loaded, "mem_during": sampler.summary(),
        "mem_end": mem_now(), "vram_free_end_mib": vram_free_mib(),
        "dec_short_median": statistics.median(dec) if dec else None,
        "dec_longctx": by.get("M4-longdec", {}).get("decode_tps"),
        "ttft": {k: v["ttft_s"] for k, v in by.items() if not k.startswith("dec-")},
        "reused": {k: v.get("eng", {}).get("reused") for k, v in by.items() if not k.startswith("dec-")},
        "vram_share_med": statistics.median([e["vram_share"] for e in eng_w if e.get("vram_share")]) if eng_w else None,
        "files_mb_total": eng[-1].get("files_mb") if eng else None,
        "slots": slots,
        "rows": rows,
    }
    with open(HERE / "agent_results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(f"[{a.name}] SUMMARY dec_short_med={summary['dec_short_median']} dec_longctx={summary['dec_longctx']} "
          f"vram_share={summary['vram_share_med']} ttft={summary['ttft']} reused={summary['reused']} "
          f"commit_free_min={summary['mem_during'].get('commit_free_min_gib')} vram_free={vram_loaded}", flush=True)
    if not a.keep:
        kill_port(a.port)


if __name__ == "__main__":
    main()
