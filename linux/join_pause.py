#!/usr/bin/env python3
"""Cold long join beside decoding streams: how long do the running streams stop while the new prompt is read?

B background users stream long replies (100-token prompt, --bg-tok output, greedy).  Once every background stream has
produced --warm tokens, one cold request with a unique ~--join-tok document joins (its prompt shares nothing with any
earlier one).  Per background stream: the longest gap between two content chunks from the join's send to its first
token + 1 s (the read and its first window), against the stream's median gap before the join.  Repeated --reps times
(each rep a fresh set of streams and a fresh document).  One JSON line per rep to --out, a summary to stdout.

  python3 join_pause.py --url http://127.0.0.1:8090 --label gap-join-64k --bg 4 --join-tok 32000 --reps 3
"""
import argparse, json, statistics as st, threading, time, urllib.request

WORDS = ("orbit lattice ferry granite meadow cipher lantern harbor quartz willow signal canyon ember tundra mosaic "
         "beacon delta prism glacier saddle vortex nimbus cobalt thistle anchor falcon summit bramble marble").split()


def doc(seed, n_tok):
    out, k = [], seed * 2654435761 + 12345
    for i in range(int(n_tok / 1.3)):
        k = (k * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        out.append(WORDS[(k >> 33) % len(WORDS)])
        if i % 19 == 18:
            out.append(f"(item {seed}.{i})\n")
    return " ".join(out)


def stream(a, messages, max_tok, times, done):
    body = {"model": a.model, "messages": messages, "max_tokens": max_tok, "temperature": 0, "stream": True,
            "reasoning_effort": "none", "stream_options": {"include_usage": True}}
    req = urllib.request.Request(a.url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                c = json.loads(line[5:])
                for ch in c.get("choices") or []:
                    if (ch.get("delta") or {}).get("content"):
                        times.append(time.perf_counter())
    finally:
        done.set()


def rep(a, k):
    bg = []
    for u in range(a.bg):
        times, done = [], threading.Event()
        msgs = [{"role": "user", "content": f"[{k}.{u}] Write a long, detailed essay (at least 3000 words) about the "
                                            f"history of lighthouses on stormy coasts, section {u + 1}."}]
        th = threading.Thread(target=stream, args=(a, msgs, a.bg_tok, times, done), daemon=True)
        th.start()
        bg.append((times, done, th))
    t_start = time.perf_counter()
    while any(len(t) < a.warm for t, d, _ in bg if not d.is_set()):
        if time.perf_counter() - t_start > 600:
            break
        time.sleep(0.05)
    jt, jd = [], threading.Event()
    msgs = [{"role": "user", "content": f"Document {k}-{a.seed}:\n{doc(1000 * a.seed + k, a.join_tok)}\n\n"
                                        f"Task: name the three most frequent words."}]
    t_join = time.perf_counter()
    jth = threading.Thread(target=stream, args=(a, msgs, 32, jt, jd), daemon=True)
    jth.start()
    jth.join()
    t_first = jt[0] if jt else time.perf_counter()
    win_end = t_first + 1.0
    time.sleep(1.5)
    rows = []
    for u, (times, done, th) in enumerate(bg):
        ts = list(times)
        pre = [y - x for x, y in zip(ts, ts[1:]) if y <= t_join]
        inwin = [x for x in ts if t_join - 5 <= x <= win_end]
        gaps = [y - x for x, y in zip(inwin, inwin[1:])]
        alive = any(x > win_end for x in ts)
        rows.append({"stream": u, "alive_after": alive, "tokens_before": sum(1 for x in ts if x <= t_join),
                     "median_gap_ms": round(st.median(pre) * 1000, 1) if pre else None,
                     "max_gap_ms": round(max(gaps) * 1000, 1) if gaps else None})
    for times, done, th in bg:
        th.join()
    return {"rep": k, "join_tok": a.join_tok, "bg": a.bg, "join_ttft_s": round(t_first - t_join, 3), "streams": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="strata")
    ap.add_argument("--label", required=True)
    ap.add_argument("--bg", type=int, default=4)
    ap.add_argument("--bg-tok", type=int, default=3000)
    ap.add_argument("--warm", type=int, default=64)
    ap.add_argument("--join-tok", type=int, default=32000)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1, help="changes every join document (cold prompts across arms too)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    res = []
    for k in range(a.reps):
        r = rep(a, k)
        r["label"] = a.label
        res.append(r)
        live = [s for s in r["streams"] if s["alive_after"] and s["max_gap_ms"] is not None]
        print(f"[{a.label}] rep {k}: join {a.join_tok} tok TTFT {r['join_ttft_s']:.2f} s; background streams alive "
              f"{len(live)}/{a.bg}, longest pause " + ", ".join(f"{s['max_gap_ms']:.0f}" for s in live) +
              " ms (median gap before " + ", ".join(f"{s['median_gap_ms']}" for s in live) + " ms)", flush=True)
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(r) + "\n")
    pauses = [s["max_gap_ms"] for r in res for s in r["streams"] if s["alive_after"] and s["max_gap_ms"] is not None]
    if pauses:
        print(f"[{a.label}] summary: join TTFT mean {st.mean(r['join_ttft_s'] for r in res):.2f} s, longest stream "
              f"pause mean {st.mean(pauses):.0f} ms, max {max(pauses):.0f} ms over {len(pauses)} stream-joins")


if __name__ == "__main__":
    main()
