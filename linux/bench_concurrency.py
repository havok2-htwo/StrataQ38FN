#!/usr/bin/env python3
"""Concurrency sweep against an OpenAI-compatible endpoint (Strata, llama.cpp, vLLM).

N closed-loop users each send ROUNDS streamed chat requests one after another; every request carries a unique
prefix (so no prompt/conversation cache hit), a ~PROMPT_TOK-token document and asks for a long answer capped at
MAX_TOK.  Per level it reports the aggregate output rate (completion tokens / wall), TTFT, per-request decode rate
(tokens after the first / time after the first) and end-to-end latency, and appends a JSON line to --out.

  python3 bench_concurrency.py --url http://127.0.0.1:8080 --levels 1,2,4,8,16 --label batch8
"""
import argparse
import concurrent.futures as cf
import json
import statistics
import threading
import time
import urllib.request

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


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def one_request(a, uid):
    topic = TOPICS[uid % len(TOPICS)]
    doc = FILLER * max(1, a.prompt_tok // 60)
    prompt = (f"[request {uid} / {a.label} / {time.time_ns()}]\nBackground notes:\n{doc}\n\n"
              f"Task: write a long, thorough and well-structured technical explanation of {topic}. "
              f"Use several sections and concrete examples. Do not stop early.")
    max_tok = a.max_tok
    if a.max_tok_min and a.max_tok_min < a.max_tok:   # mixed reply lengths, the same per request id in every run
        max_tok = a.max_tok_min + (uid * 2654435761) % (a.max_tok - a.max_tok_min + 1)
    body = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tok,
            "stream": True, "temperature": a.temperature, "seed": uid, "reasoning_effort": "none",
            "stream_options": {"include_usage": True}}
    if a.vllm:
        body["chat_template_kwargs"] = {"enable_thinking": False}
        body.pop("reasoning_effort")
    hdr = {"Content-Type": "application/json"}
    if a.key:
        hdr["Authorization"] = "Bearer " + a.key
    urls = a.url.split(",")          # several instances: round-robin by request id (data-parallel setups)
    req = urllib.request.Request(urls[uid % len(urls)].rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=hdr)
    t0 = time.perf_counter()
    t_first = None
    chunks = 0
    usage = None
    err = None
    try:
        with urllib.request.urlopen(req, timeout=a.timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    c = json.loads(data)
                except ValueError:
                    continue
                if c.get("error"):
                    err = str(c["error"])[:200]
                if c.get("usage"):
                    usage = c["usage"]
                for ch in c.get("choices") or []:
                    d = ch.get("delta") or {}
                    if d.get("content") or d.get("reasoning_content"):
                        if t_first is None:
                            t_first = time.perf_counter()
                        chunks += 1
    except Exception as e:  # noqa: BLE001 - a failed request is a result too
        err = repr(e)[:200]
    t1 = time.perf_counter()
    gen = (usage or {}).get("completion_tokens", chunks)
    ptok = (usage or {}).get("prompt_tokens", 0)
    cached = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    ttft = (t_first - t0) if t_first else float("nan")
    dec = (gen - 1) / (t1 - t_first) if t_first and t1 > t_first and gen > 1 else float("nan")
    return {"uid": uid, "t0": t0, "t1": t1, "ttft": ttft, "e2e": t1 - t0, "gen": gen, "prompt": ptok,
            "cached": cached, "dec_tps": dec, "err": err}


def run_level(a, n, base_uid):
    results, lock = [], threading.Lock()

    def user(i):
        for r in range(a.rounds):
            res = one_request(a, base_uid + i * a.rounds + r)
            with lock:
                results.append(res)

    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        list(ex.map(user, range(n)))
    wall = time.perf_counter() - t0
    ok = [r for r in results if not r["err"] and r["gen"] > 0]
    errs = [r["err"] for r in results if r["err"]]
    tot_gen = sum(r["gen"] for r in ok)
    tot_prompt = sum(r["prompt"] for r in ok)
    ttft = [r["ttft"] for r in ok if r["ttft"] == r["ttft"]]
    dec = [r["dec_tps"] for r in ok if r["dec_tps"] == r["dec_tps"]]
    e2e = [r["e2e"] for r in ok]
    row = {
        "label": a.label, "users": n, "rounds": a.rounds, "requests": len(results), "ok": len(ok), "errors": len(errs),
        "err_sample": errs[:3], "wall_s": round(wall, 2), "max_tok": a.max_tok, "prompt_tok_avg": round(tot_prompt / max(1, len(ok))),
        "out_tok_total": tot_gen, "agg_out_tps": round(tot_gen / wall, 1),
        "agg_total_tps": round((tot_gen + tot_prompt) / wall, 1),
        "ttft_p50": round(pct(ttft, 50), 2), "ttft_p95": round(pct(ttft, 95), 2), "ttft_max": round(max(ttft) if ttft else float("nan"), 2),
        "dec_tps_p50": round(pct(dec, 50), 1), "dec_tps_p5": round(pct(dec, 5), 1), "dec_tps_p95": round(pct(dec, 95), 1),
        "e2e_p50": round(pct(e2e, 50), 1), "e2e_p95": round(pct(e2e, 95), 1),
        "cached_tok": sum(r["cached"] for r in ok),
    }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--key", default="")
    ap.add_argument("--label", default="run")
    ap.add_argument("--levels", default="1,2,4,8,16")
    ap.add_argument("--rounds", type=int, default=2, help="requests per user, sent back to back")
    ap.add_argument("--max-tok", type=int, default=512)
    ap.add_argument("--max-tok-min", type=int, default=0,
                    help="> 0: each request's max_tokens is spread over [max-tok-min, max-tok] (fixed per request id), "
                         "so the slots do not finish in lockstep")
    ap.add_argument("--prompt-tok", type=int, default=1000)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--out", default="")
    ap.add_argument("--model", default="strata")
    ap.add_argument("--vllm", action="store_true", help="thinking off via chat_template_kwargs (vLLM)")
    a = ap.parse_args()
    for w in range(a.warmup):
        one_request(a, 10_000_000 + w)
    print(f"# {a.label}: prompt~{a.prompt_tok} tok, max_tokens {a.max_tok}, {a.rounds} request(s) per user, "
          f"temperature {a.temperature}", flush=True)
    print(f"{'users':>5} {'ok/req':>7} {'wall':>7} {'agg out':>8} {'dec p50':>8} {'dec p5':>7} {'TTFT p50':>9} "
          f"{'TTFT p95':>9} {'TTFT max':>9} {'e2e p50':>8} {'e2e p95':>8}", flush=True)
    for i, n in enumerate(int(x) for x in a.levels.split(",")):
        row = run_level(a, n, 1_000_000 * (i + 1))
        print(f"{n:>5} {row['ok']:>3}/{row['requests']:<3} {row['wall_s']:>7.1f} {row['agg_out_tps']:>8.1f} "
              f"{row['dec_tps_p50']:>8.1f} {row['dec_tps_p5']:>7.1f} {row['ttft_p50']:>9.2f} {row['ttft_p95']:>9.2f} "
              f"{row['ttft_max']:>9.2f} {row['e2e_p50']:>8.1f} {row['e2e_p95']:>8.1f}"
              + (f"  ERR {row['err_sample']}" if row["errors"] else ""), flush=True)
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")
        time.sleep(2)


if __name__ == "__main__":
    main()
