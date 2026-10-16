#!/usr/bin/env python3
"""Token identity of the overlapped read with a streamed K/V cache (--kv-resident), over HTTP, greedy.

Per round: B background conversations, each a ~--bg-ctx token document (longer than --kv-resident, so the slots'
selections miss and CLOCK evicts while they decode) plus a round-specific question (round 1 reads them cold, later
rounds resume them from the conversation cache: a read with p0 > 0 that stages a restored host copy).  Once every
background stream has produced --warm tokens, one cold join of JT tokens is sent (JT from --joins).  When everything
has finished, a second turn of the join conversation (history + reply + a new question) resumes it from its parked
slot or the RAM cache.  Every reply's full text goes to --out as JSON: run it once per arm, then --compare A.json C.json.

  python3 -I ovkv_ident.py --url http://127.0.0.1:8090 --out results/ovkv/hident-A.json
  python3 -I ovkv_ident.py --compare results/ovkv/hident-A.json results/ovkv/hident-C.json
"""
import argparse, json, sys, threading, time, urllib.request

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


def chat(a, messages, max_tok, res, key, times=None):
    body = {"model": a.model, "messages": messages, "max_tokens": max_tok, "temperature": 0, "stream": True,
            "reasoning_effort": "none"}
    req = urllib.request.Request(a.url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    text, err = [], None
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                c = json.loads(line[5:])
                for ch in c.get("choices") or []:
                    d = (ch.get("delta") or {}).get("content")
                    if d:
                        text.append(d)
                        if times is not None:
                            times.append(time.perf_counter())
    except Exception as e:  # recorded: an arm with an error never matches
        err = repr(e)
    res[key] = {"text": "".join(text), "error": err}


def run(a):
    res, joins = {}, [int(x) for x in a.joins.split(",")]
    bg_docs = [doc(500 + u, a.bg_ctx) for u in range(a.bg)]
    for r, jt in enumerate(joins):
        t0 = time.perf_counter()
        bg_max = a.bg_tok + jt // 100   # long enough to keep decoding through the join's read
        ths, times = [], []
        for u in range(a.bg):
            msgs = [{"role": "user", "content": f"Document {u}:\n{bg_docs[u]}\n\nQuestion {r}: write a long, detailed "
                                                f"story that uses the words of this document, part {r + 1}."}]
            t = []
            th = threading.Thread(target=chat, args=(a, msgs, bg_max, res, f"r{r}-bg{u}", t), daemon=True)
            th.start()
            ths.append(th)
            times.append(t)
        t_s = time.perf_counter()
        while any(len(t) < a.warm for t, th in zip(times, ths) if th.is_alive()) and time.perf_counter() - t_s < 900:
            time.sleep(0.05)
        jmsgs = [{"role": "user", "content": f"Report {r}-{a.seed}:\n{doc(9000 + 100 * a.seed + r, jt)}\n\n"
                                             f"Task: name the three most frequent words and summarise the report."}]
        chat(a, jmsgs, 96, res, f"r{r}-join{jt}")
        for th in ths:
            th.join()
        # the join conversation's second turn: resumes it (its parked slot or the RAM conversation cache)
        jmsgs2 = jmsgs + [{"role": "assistant", "content": res[f"r{r}-join{jt}"]["text"]},
                          {"role": "user", "content": "Now name the least frequent word, and why it matters."}]
        chat(a, jmsgs2, 64, res, f"r{r}-join{jt}-turn2")
        bad = [k for k, v in res.items() if k.startswith(f"r{r}-") and v["error"]]
        print(f"round {r}: join {jt} beside {a.bg} x {a.bg_ctx} tok streams, {time.perf_counter() - t0:.0f} s"
              + (f", errors: {bad}" if bad else ""), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


def compare(pa, pb):
    A, B = json.load(open(pa)), json.load(open(pb))
    keys = sorted(set(A) | set(B))
    bad = [k for k in keys if k not in A or k not in B or A[k]["error"] or B[k]["error"] or A[k]["text"] != B[k]["text"]]
    for k in bad:
        x, y = A.get(k, {}), B.get(k, {})
        tx, ty = x.get("text", ""), y.get("text", "")
        n = next((i for i, (p, q) in enumerate(zip(tx, ty)) if p != q), min(len(tx), len(ty)))
        print(f"  {k}: differs at char {n} (errors: {x.get('error')} / {y.get('error')})")
    print(f"==> http identity {pa} vs {pb}: {len(keys) - len(bad)}/{len(keys)} replies identical "
          + ("PASS" if not bad and keys else "FAIL"))
    return 0 if not bad and keys else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--model", default="strata")
    ap.add_argument("--bg", type=int, default=4)
    ap.add_argument("--bg-ctx", type=int, default=45000, help="background documents (tokens): > --kv-resident")
    ap.add_argument("--bg-tok", type=int, default=400, help="background reply tokens (+ join/100)")
    ap.add_argument("--warm", type=int, default=32)
    ap.add_argument("--joins", default="1000,32000,100000,200000")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="")
    ap.add_argument("--compare", nargs=2, default=None)
    a = ap.parse_args()
    if a.compare:
        sys.exit(compare(*a.compare))
    run(a)


if __name__ == "__main__":
    main()
