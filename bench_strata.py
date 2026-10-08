"""Serial Strata benchmark: decode tok/s + TTFT over a fixed prompt set.

usage: python bench_strata.py LABEL [--url http://127.0.0.1:8080] [--runs 2] [--max-tokens 384]
Appends one JSON line per run to bench_results.jsonl next to this script.
"""
import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

PROMPTS = {
    "chat": "Explain in detail how a CPU branch predictor works, including two-level adaptive predictors and TAGE.",
    "code": "Write a complete, well-commented Python implementation of an LRU cache with O(1) get and put, "
            "plus unit tests using unittest.",
    "de": "Erkläre ausführlich, wie eine Wärmepumpe funktioniert und wann sie sich in einem Altbau lohnt.",
    # ~4K-token prompt: repeated technical text + a question at the end
    "long4k": ("The following is a log of design notes.\n" + "\n".join(
        f"Note {i}: the cache layer {i % 7} evicts entries using a decayed frequency counter; "
        f"hits on shard {i % 13} were {i * 37 % 1000} per second and latency p99 was {i * 11 % 90} ms."
        for i in range(1, 140)) + "\n\nSummarize the main patterns in these notes in a structured way."),
    # ~32K-token prompt (prefill test)
    "long32k": ("The following is a long log of design notes.\n" + "\n".join(
        f"Entry {i}: service {i % 23} migrated table t{i % 97} to partition p{i % 11}; replication lag peaked at "
        f"{i * 13 % 700} ms, {i * 7 % 50} retries, owner team-{i % 17}, ticket OPS-{1000 + i}."
        for i in range(1, 580)) +"\n\nWhich services had the most retries? Answer briefly."),
}
DEFAULT = ["chat", "code", "de", "long4k"]


def run_one(url, prompt, max_tokens):
    body = json.dumps({
        "model": "strata", "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "reasoning_effort": "none", "max_tokens": max_tokens,
        "stream": True, "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = last = None
    n_chunks = 0
    usage = None
    text = []
    with urllib.request.urlopen(req, timeout=1800) as r:
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
                    if first is None:
                        first = now
                    last = now
                    n_chunks += 1
                    text.append(piece)
    t_end = time.perf_counter()
    out_tok = (usage or {}).get("completion_tokens") or n_chunks
    prompt_tok = (usage or {}).get("prompt_tokens")
    decode_s = (last - first) if (first and last and last > first) else float("nan")
    return {
        "ttft_s": round((first or t_end) - t0, 3),
        "decode_tps": round((out_tok - 1) / decode_s, 2) if decode_s == decode_s else None,
        "e2e_tps": round(out_tok / (t_end - t0), 2),
        "out_tokens": out_tok, "prompt_tokens": prompt_tok,
        "head": "".join(text)[:160].replace("\n", " "),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    names = a.only.split(",") if a.only else DEFAULT
    # warmup (not recorded): first request after load pays graph capture etc.
    time.sleep(20)
    run_one(a.url, "Write a short poem about the sea.", 256)
    run_one(a.url, "Explain recursion with an example in Python.", 256)
    summary = {}
    with open(HERE / "bench_results.jsonl", "a", encoding="utf-8") as f:
        for name in names:
            rows = []
            for i in range(a.runs):
                # unique prefix so the prompt cache cannot reuse the previous run
                r = run_one(a.url, f"[run {i}] " + PROMPTS[name], a.max_tokens)
                r.update(label=a.label, prompt=name, run=i, ts=time.strftime("%Y-%m-%d %H:%M:%S"))
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                rows.append(r)
                print(f"{a.label:24s} {name:7s} run{i}: decode {r['decode_tps']} tok/s | "
                      f"TTFT {r['ttft_s']}s | out {r['out_tokens']} | prompt {r['prompt_tokens']}", flush=True)
            summary[name] = statistics.median(r["decode_tps"] for r in rows if r["decode_tps"])
    print("MEDIAN decode tok/s:", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
