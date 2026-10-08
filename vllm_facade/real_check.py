"""Smoke test of the vLLM facade against the REAL Strata (facade on 127.0.0.1:8000)."""
import json
import time
import urllib.request

from openai import OpenAI

B = "http://127.0.0.1:8000"
c = OpenAI(base_url=B + "/v1", api_key="EMPTY")


def get(path):
    with urllib.request.urlopen(B + path, timeout=30) as r:
        return r.status, r.read().decode()


def post(path, body):
    req = urllib.request.Request(B + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def check(name, fn):
    t0 = time.time()
    try:
        out = fn()
        print(f"[OK]   {name} ({time.time() - t0:.1f}s): {str(out)[:220]}", flush=True)
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {str(e)[:300]}", flush=True)


M = c.models.list().data[0].id
check("health", lambda: get("/health")[0])
check("version", lambda: get("/version")[1])
check("models", lambda: [(m.id, m.owned_by, m.model_extra.get("max_model_len")) for m in c.models.list().data])
check("chat no-think", lambda: c.chat.completions.create(
    model=M, messages=[{"role": "user", "content": "Nenne drei Primfarben, nur die Wörter."}], max_tokens=60,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}}).choices[0].message.content)


def think():
    r = c.chat.completions.create(model=M, messages=[{"role": "user", "content": "What is 17*23? Answer with the number."}],
                                  max_tokens=600, reasoning_effort="low")
    m = r.choices[0].message
    return {"content": m.content, "reasoning_len": len((m.model_extra or {}).get("reasoning") or ""),
            "usage": r.usage.model_dump() if r.usage else None}


check("chat thinking", think)


def stream():
    s = c.chat.completions.create(model=M, messages=[{"role": "user", "content": "Count from 1 to 10."}], stream=True,
                                  stream_options={"include_usage": True}, max_tokens=80,
                                  extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    chunks = list(s)
    text = "".join(ch.choices[0].delta.content or "" for ch in chunks if ch.choices)
    return {"n_chunks": len(chunks), "last_usage": chunks[-1].usage.total_tokens if chunks[-1].usage else None,
            "text": text}


check("stream+usage", stream)
tools = [{"type": "function", "function": {"name": "get_weather", "description": "current weather",
                                           "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                          "required": ["city"]}}}]


def tool():
    r = c.chat.completions.create(model=M, messages=[{"role": "user", "content": "Wie ist das Wetter in Berlin?"}],
                                  tools=tools, max_tokens=300, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    ch = r.choices[0]
    return ch.finish_reason, [(t.id[:20], t.function.name, t.function.arguments) for t in (ch.message.tool_calls or [])]


check("tools", tool)


def structured():
    r = c.chat.completions.create(model=M, messages=[{"role": "user", "content": "Gib eine Stadt in Bayern und ihre Einwohnerzahl."}],
                                  max_tokens=200, extra_body={
                                      "guided_json": {"type": "object", "properties": {"city": {"type": "string"},
                                                                                      "population": {"type": "integer"}},
                                                      "required": ["city", "population"]},
                                      "chat_template_kwargs": {"enable_thinking": False}})
    return json.loads(r.choices[0].message.content)


check("guided_json", structured)
check("guided_choice", lambda: c.chat.completions.create(
    model=M, messages=[{"role": "user", "content": "Ist Wasser nass? Antworte ja oder nein."}], max_tokens=50,
    extra_body={"guided_choice": ["ja", "nein"], "chat_template_kwargs": {"enable_thinking": False}}).choices[0].message.content)
check("completions", lambda: c.completions.create(model=M, prompt="Die Hauptstadt von Frankreich ist", max_tokens=16).choices[0].text)
check("tokenize", lambda: post("/tokenize", {"model": M, "prompt": "Hallo Welt"}))
check("detokenize", lambda: post("/detokenize", {"model": M, "tokens": post("/tokenize", {"prompt": "Hallo Welt"})["tokens"]}))
check("metrics", lambda: [l for l in get("/metrics")[1].splitlines()
                          if l.startswith(("vllm:generation_tokens_total", "vllm:num_requests_running"))])


def wrong_model():
    try:
        c.chat.completions.create(model="nope", messages=[{"role": "user", "content": "x"}], max_tokens=5)
        return "NO ERROR?!"
    except Exception as e:
        return f"{type(e).__name__}: {str(e)[:120]}"


check("unknown model -> 404", wrong_model)


def too_long():
    try:
        c.chat.completions.create(model=M, messages=[{"role": "user", "content": "x"}], max_tokens=10 ** 6)
        return "NO ERROR?!"
    except Exception as e:
        return f"{type(e).__name__}: {str(e)[:160]}"


check("context overflow -> 400", too_long)
