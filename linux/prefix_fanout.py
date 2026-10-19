#!/usr/bin/env python3
"""Prefix fan-out: N requests that share one long prefix x (a document) and differ only in the last question - do they
all read just their own end and decode at the same time?

  prefix_fanout.py --url http://127.0.0.1:8091 --n 16 --doc-tok 20000 --mode warm|cold --label L --out F.json

warm: one request x + question 0 first (x is then in a slot / the conversation cache), then N requests at once.
cold: N requests at once with a new x nobody sent before.
Per request: TTFT, prompt / cached tokens (usage), decode tok/s, its decode interval; plus /v1/load every 0.2 s
(slots decoding / reading / waiting).  Printed: TTFT p50/max, cached share, the most requests decoding at the same time
(from the stream intervals and from /v1/load).  Greedy, thinking off (reasoning_effort none)."""
import argparse, json, threading, time, urllib.request
import concurrent.futures as cf

QUESTIONS = ["Wie viele Abschnitte nennt das Dokument?", "Welcher Wert steht in Abschnitt 7?",
             "Fasse Abschnitt 3 in einem Satz zusammen.", "Welcher Abschnitt hat den größten Wert?",
             "Nenne die Werte der Abschnitte 1 bis 4.", "Gibt es Abschnitte mit gleichem Wert?",
             "Was ist der Wert von Abschnitt 12?", "Wie lautet die Überschrift des Dokuments?",
             "Welcher Abschnitt kommt nach Abschnitt 20?", "Ist der Wert in Abschnitt 5 gerade?",
             "Zähle drei Abschnitte mit kleinem Wert auf.", "Welche Farbe wird in Abschnitt 9 genannt?",
             "Wie viele Wörter hat Abschnitt 2 ungefähr?", "Welcher Ort steht in Abschnitt 15?",
             "Was ist die Summe der Werte von Abschnitt 1 und 2?", "Nenne den letzten Abschnitt.",
             "Was sagt Abschnitt 11?", "Welcher Wert steht in Abschnitt 30?"]
COLORS = ["rot", "blau", "grün", "gelb", "violett", "orange", "grau", "weiß"]
PLACES = ["Leipzig", "Dresden", "Halle", "Jena", "Erfurt", "Chemnitz", "Zwickau", "Gera"]


def doc(tag: str, tok: int) -> str:
    out, i = [f"Dokument {tag}: Messprotokoll."], 0
    while sum(len(s) for s in out) < tok * 3.6:          # ~3.6 characters per token for this text
        i += 1
        out.append(f"Abschnitt {i} ({tag}): Der gemessene Wert beträgt {(i * 37 + len(tag)) % 997}, die Farbe ist "
                   f"{COLORS[i % 8]}, der Ort ist {PLACES[(i * 3) % 8]}, Prüfnummer {i * 7919 % 100003}.")
    return "\n".join(out)


def one(url, d, q, max_tok, uid):
    body = {"model": "strata", "messages": [{"role": "system", "content": "Antworte kurz auf Deutsch."},
                                            {"role": "user", "content": d + "\n\nFrage: " + q}],
            "max_tokens": max_tok, "stream": True, "temperature": 0, "seed": uid, "reasoning_effort": "none",
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); tf = None; usage = None; err = None; n = 0; txt = []
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                    continue
                c = json.loads(line[5:])
                usage = c.get("usage") or usage
                for ch in c.get("choices") or []:
                    dd = ch.get("delta") or {}
                    if dd.get("content") or dd.get("reasoning_content"):
                        tf = tf or time.time(); n += 1; txt.append((dd.get("reasoning_content") or "") + (dd.get("content") or ""))
    except Exception as e:  # noqa: BLE001
        err = repr(e)[:200]
    t1 = time.time()
    u = usage or {}
    gen = u.get("completion_tokens", n)
    return {"uid": uid, "t0": t0, "tf": tf, "t1": t1, "ttft": (tf - t0) if tf else None, "gen": gen,
            "prompt": u.get("prompt_tokens"), "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "dec": (gen - 1) / (t1 - tf) if tf and t1 > tf and gen > 1 else None, "err": err, "text": "".join(txt)}


def poll(url, stop, rows):
    while not stop.is_set():
        try:
            s = json.load(urllib.request.urlopen(url + "/v1/load", timeout=2))["slots"]
            rows.append((time.time(), s.get("decoding", 0), s.get("reading_prompt", 0), s.get("waiting", 0)))
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8091")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--doc-tok", type=int, default=20000)
    ap.add_argument("--max-tok", type=int, default=128)
    ap.add_argument("--mode", choices=["warm", "cold"], default="warm")
    ap.add_argument("--tag", default="")
    ap.add_argument("--label", default="fanout")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    d = doc(a.tag or f"{a.mode}-{int(time.time())}", a.doc_tok)
    pre = None
    if a.mode == "warm":
        pre = one(a.url, d, "Antworte nur mit OK.", 8, 999)
        time.sleep(1.0)
    stop, load = threading.Event(), []
    th = threading.Thread(target=poll, args=(a.url, stop, load), daemon=True); th.start()
    with cf.ThreadPoolExecutor(a.n) as ex:
        rs = list(ex.map(lambda i: one(a.url, d, QUESTIONS[i % len(QUESTIONS)] + f" (#{i})", a.max_tok, i), range(a.n)))
    time.sleep(0.5); stop.set(); th.join()
    ok = [r for r in rs if r["tf"] and not r["err"]]
    t0 = min(r["t0"] for r in rs)
    ev = sorted([(r["tf"], 1) for r in ok] + [(r["t1"], -1) for r in ok])
    cur = peak = 0
    for _, s in ev:
        cur += s; peak = max(peak, cur)
    tt = sorted(r["ttft"] for r in ok)
    p = lambda q: tt[min(len(tt) - 1, int(q * len(tt)))] if tt else float("nan")
    first_all = max(r["tf"] for r in ok) - t0 if ok else float("nan")
    last_start = max(r["tf"] for r in ok) if ok else 0
    first_end = min(r["t1"] for r in ok) if ok else 0
    print(f"# {a.label}: mode {a.mode}, {a.n} requests sharing a ~{a.doc_tok}-token document, {a.max_tok} tokens out")
    if pre:
        print(f"warm-up: prompt {pre['prompt']}, cached {pre['cached']}, TTFT {pre['ttft']:.2f} s")
    print(f"{'#':>3} {'prompt':>7} {'cached':>7} {'read':>6} {'TTFT':>6} {'first@':>7} {'done@':>7} {'tok':>4} {'dec/s':>6}")
    for r in sorted(rs, key=lambda r: r["tf"] or 1e18):
        if r["err"] or not r["tf"]:
            print(f"{r['uid']:>3} ERROR {r['err']}"); continue
        rd = (r["prompt"] or 0) - (r["cached"] or 0)
        print(f"{r['uid']:>3} {r['prompt']:>7} {r['cached']:>7} {rd:>6} {r['ttft']:>6.2f} {r['tf'] - t0:>7.2f} "
              f"{r['t1'] - t0:>7.2f} {r['gen']:>4} {r['dec'] or 0:>6.1f}")
    cached_share = sum(r["cached"] or 0 for r in ok) / max(1, sum(r["prompt"] or 0 for r in ok))
    print(f"==> TTFT p50 {p(0.5):.2f} s, max {tt[-1] if tt else float('nan'):.2f} s; all {len(ok)} first tokens after "
          f"{first_all:.2f} s; prompt tokens from cache {100 * cached_share:.1f} %; most decoding at once (streams) "
          f"{peak}; (load poll) max decoding {max((x[1] for x in load), default=0)}, max reading {max((x[2] for x in load), default=0)}, "
          f"max waiting {max((x[3] for x in load), default=0)}; all {len(ok)} decoding together: "
          f"{'yes' if last_start < first_end else 'no'} (last start {last_start - t0:.2f} s, first end {first_end - t0:.2f} s); "
          f"errors {len(rs) - len(ok)}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"label": a.label, "mode": a.mode, "pre": pre, "rows": rs, "load": load}, f)


if __name__ == "__main__":
    main()
