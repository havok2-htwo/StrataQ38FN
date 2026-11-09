#!/usr/bin/env python3
"""The request shapes Genesis sends through G3_vLLM, against one server (a test server on 127.0.0.1:8091 in a window):
  a  thinking off, short answer               b  response_format json_object
  c  streamed tool call: tools + reasoning_budget 2048 + chat_template_kwargs reasoning_effort low (G3's form)
  d  the tool result as G3 sends it back (a user turn "[Runtime System Instruction] Tool Result:"), streamed
  e  reasoning_effort xhigh with reasoning_budget 512: the reasoning stops at the budget, an answer follows
  f  an image (a red square on white, PNG made here) with thinking off
One line per shape and a verdict line '==> smoke <label>: ... PASS|FAIL' (window-v31.sh greps '^==>').
  python3 v31_smoke.py --url http://127.0.0.1:8091 --model qwen3.8-flash-next-unsloth-ud-iq4_xs --label V
"""
import argparse, base64, json, struct, sys, time, urllib.request, zlib


def png_square(n=96, sq=48):
    rows = b""
    for y in range(n):
        row = bytearray(b"\x00")
        for x in range(n):
            inside = (n - sq) // 2 <= x < (n + sq) // 2 and (n - sq) // 2 <= y < (n + sq) // 2
            row += b"\xdc\x14\x14" if inside else b"\xff\xff\xff"
        rows += bytes(row)
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", n, n, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))


def call(url, body, stream):
    body = dict(body, stream=stream)
    if stream:
        body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; content = reasoning = ""; calls = {}; finish = usage = None; events = 0
    with urllib.request.urlopen(req, timeout=300) as r:
        if not stream:
            d = json.load(r); m = d["choices"][0]["message"]
            for i, tc in enumerate(m.get("tool_calls") or []):
                calls[i] = {"id": tc.get("id"), "name": tc["function"]["name"], "args": tc["function"].get("arguments") or ""}
            return dict(ttft=round(time.time() - t0, 2), events=1, content=m.get("content") or "",
                        reasoning=m.get("reasoning_content") or "", calls=list(calls.values()),
                        finish=d["choices"][0].get("finish_reason"), usage=d.get("usage"))
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            d = json.loads(data); events += 1
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                dl = ch.get("delta", {})
                if (dl.get("content") or dl.get("reasoning_content") or dl.get("tool_calls")) and ttft is None:
                    ttft = time.time() - t0
                content += dl.get("content") or ""; reasoning += dl.get("reasoning_content") or ""
                for tc in dl.get("tool_calls") or []:
                    c = calls.setdefault(tc.get("index", 0), {"id": None, "name": "", "args": ""})
                    c["id"] = tc.get("id") or c["id"]; f = tc.get("function") or {}
                    c["name"] += f.get("name") or ""; c["args"] += f.get("arguments") or ""
                finish = ch.get("finish_reason") or finish
    return dict(ttft=round(ttft or -1, 2), events=events, content=content, reasoning=reasoning,
                calls=list(calls.values()), finish=finish, usage=usage)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="strata")
    ap.add_argument("--label", default="smoke")
    ap.add_argument("--out", default="", help="append the shapes' results as one JSON line")
    a = ap.parse_args()
    M = a.model
    tools = [{"type": "function", "function": {"name": "get_time", "description": "Liefert die aktuelle Uhrzeit",
                                               "parameters": {"type": "object", "properties": {}}}}]
    base = [{"role": "system", "content": "Du bist ein Testagent."},
            {"role": "user", "content": "Wie spät ist es? Nutze das Werkzeug."}]
    res, ok = {}, {}

    def run(key, fn):
        try:
            res[key] = fn()
        except Exception as e:  # noqa: BLE001
            res[key] = {"error": repr(e)[:300]}

    run("a", lambda: call(a.url, dict(model=M, messages=[{"role": "user", "content": "Antworte nur mit OK."}],
                                      temperature=0, max_tokens=24, chat_template_kwargs={"enable_thinking": False}), False))
    ok["a"] = "OK" in (res["a"].get("content") or "")
    run("b", lambda: call(a.url, dict(model=M, messages=[{"role": "user", "content": "Gib ein JSON-Objekt mit dem Schlüssel ok=true zurück."}],
                                      temperature=0, max_tokens=64, chat_template_kwargs={"enable_thinking": False},
                                      response_format={"type": "json_object"}), False))
    try:
        ok["b"] = json.loads(res["b"].get("content") or "").get("ok") is True
    except Exception:  # noqa: BLE001
        ok["b"] = False
    run("c", lambda: call(a.url, dict(model=M, messages=base, temperature=0.1, max_tokens=2048, tools=tools, reasoning_budget=2048,
                                      chat_template_kwargs={"reasoning_effort": "low"}), True))
    cc = (res["c"].get("calls") or [{}])[0]
    try:
        json.loads(cc.get("args") or "{}"); args_ok = True
    except Exception:  # noqa: BLE001
        args_ok = False
    ok["c"] = cc.get("name") == "get_time" and args_ok and res["c"].get("finish") == "tool_calls"
    tc = {"id": cc.get("id") or "call_1", "type": "function", "function": {"name": "get_time", "arguments": cc.get("args") or "{}"}}
    msgs = base + [{"role": "assistant", "content": None, "tool_calls": [tc]},
                   {"role": "user", "content": "[Runtime System Instruction]\nTool Result:\n08:01"}]
    run("d", lambda: call(a.url, dict(model=M, messages=msgs, temperature=0.1, max_tokens=2048, tools=tools, reasoning_budget=2048,
                                      chat_template_kwargs={"reasoning_effort": "low"}), True))
    ok["d"] = "08:01" in (res["d"].get("content") or "") or "8:01" in (res["d"].get("content") or "")
    run("e", lambda: call(a.url, dict(model=M, messages=[{"role": "user", "content": "Wie viele Primzahlen liegen zwischen 1 und 60? Nenne am Ende nur die Zahl."}],
                                      temperature=0, max_tokens=1024, reasoning_effort="xhigh", reasoning_budget=512), True))
    ru = ((res["e"].get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens")
    ok["e"] = bool((res["e"].get("content") or "").strip()) and res["e"].get("finish") in ("stop", "length") \
        and (ru is None or ru <= 600)
    img = "data:image/png;base64," + base64.b64encode(png_square()).decode()
    run("f", lambda: call(a.url, dict(model=M, messages=[{"role": "user", "content": [
        {"type": "text", "text": "Welche Farbe hat das Quadrat in der Mitte des Bildes? Antworte mit einem Wort."},
        {"type": "image_url", "image_url": {"url": img}}]}], temperature=0, max_tokens=24,
        chat_template_kwargs={"enable_thinking": False}), False))
    ok["f"] = any(w in (res["f"].get("content") or "").lower() for w in ("rot", "red"))
    for k in "abcdef":
        r = dict(res[k]); r["reasoning_chars"] = len(r.pop("reasoning", "") or "")
        r["content"] = (r.get("content") or "")[:120]
        print(f"{k}) {'ok ' if ok[k] else 'BAD'} {json.dumps(r, ensure_ascii=False)}")
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps({"label": a.label, "ok": ok, "res": {k: {**v, "reasoning": (v.get("reasoning") or "")[:200]}
                                                                      for k, v in res.items()}}, ensure_ascii=False) + "\n")
    bad = [k for k in "abcdef" if not ok[k]]
    print(f"==> smoke {a.label}: " + " ".join(f"{k}={'ok' if ok[k] else 'BAD'}" for k in "abcdef")
          + (" PASS" if not bad else f" FAIL ({','.join(bad)})"))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
