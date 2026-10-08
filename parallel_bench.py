import json
import statistics
import time
import urllib.request
import concurrent.futures
import sys
from pathlib import Path

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080") + "/v1/chat/completions"
KEY = (Path(__file__).resolve().parent / "api_key.txt").read_text().strip()
LABEL = sys.argv[2] if len(sys.argv) > 2 else ""
MAXTOK = 1000
ROUNDS = [1, 2, 4]

TOPICS = [
    "how NVMe SSD wear leveling works",
    "the CAP theorem in distributed systems",
    "how TCP congestion control (cubic) works",
    "why B-trees are used in database indexes",
    "how virtual memory and page tables work",
    "the tradeoffs of event-driven vs synchronous architectures",
    "how GPU tensor cores execute matrix operations",
    "why cache coherency protocols exist in multi-core CPUs",
]


def one_request(round_id, idx):
    prompt = (
        f"[bench round={round_id} worker={idx}] You are a technical documentation agent. "
        f"Write a thorough technical explanation of: {TOPICS[idx % len(TOPICS)]}. "
        f"Be detailed and precise."
    )
    body = json.dumps(
        {
            "model": "qwen",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAXTOK,
            "stream": False, "temperature": 0, "reasoning_effort": "none",
        }
    ).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        data = json.loads(r.read())
    wall = time.time() - t0
    u = data.get("usage", {})
    gen = u.get("completion_tokens", 0)
    return {
        "idx": idx, "wall": wall, "gen_tps": gen / wall, "prompt_tps": u.get("prompt_tokens", 0) / wall,
        "prompt_n": u.get("prompt_tokens", 0), "gen_n": gen,
        "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0,
    }


def run_round(n):
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        futs = [ex.submit(one_request, n, i) for i in range(n)]
        t0 = time.time()
        results = [f.result() for f in futs]
    wall = time.time() - t0
    gen_tps = sorted(r["gen_tps"] for r in results)
    prompt_tps = sorted(r["prompt_tps"] for r in results)
    total_out = sum(r["gen_n"] for r in results)
    agg = total_out / wall
    cached = sum(r["cached"] for r in results)
    print(
        f"N={n:<3} wall={wall:7.1f}s | agg_gen={agg:7.1f} tok/s | "
        f"gen_tps min/med/max = {gen_tps[0]:5.1f}/{gen_tps[len(gen_tps)//2]:5.1f}/{gen_tps[-1]:5.1f} | "
        f"prompt_tps min/med/max = {prompt_tps[0]:6.1f}/{prompt_tps[len(prompt_tps)//2]:6.1f}/{prompt_tps[-1]:6.1f} | "
        f"cached_tokens={cached}",
        flush=True,
    )
    print(
        f"      per-request wall: "
        + " ".join(f"{r['idx']}:{r['wall']:.1f}s" for r in sorted(results, key=lambda x: x['idx'])),
        flush=True,
    )
    return wall, agg


print(LABEL, "parallel scaling test: 128 output tokens per request, unique prompt per round", flush=True)
walls = []
for n in ROUNDS:
    wall, agg = run_round(n)
    walls.append((n, wall, agg))
    time.sleep(3)

print()
print("scaling: wall time vs N (ideal linear = 1.0x per doubling, perfect parallel = 0.5x)")
prev = None
for n, wall, agg in walls:
    if prev:
        print(f"  N={n:<3} wall={wall:7.1f}s | ratio vs prev = {wall/prev[1]:.2f} | agg={agg:6.1f} tok/s")
    else:
        print(f"  N={n:<3} wall={wall:7.1f}s |                          | agg={agg:6.1f} tok/s")
    prev = (n, wall, agg)
