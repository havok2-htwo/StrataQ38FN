#!/usr/bin/env python3
"""test_facade.py - automated tests of vllm_facade.py against Strata's mock engine (no GPU).

Starts three Strata mock servers (8081 greeting, 8082 tool call, 8083 JSON) and eight facades (8001-8008) with
different flags, runs the tests with urllib, and stops every process it started (by PID).  Ports 8000 and 8080 are
never touched.

    X:/dev/StrataQ38FN/Strata/.venv/Scripts/python.exe test_facade.py [-k substring] [--keep-logs]

When .testvenv (with openai installed) exists next to this file, an openai-python smoke test runs as well.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import select
import threading
import shutil
import socket
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
STRATA = Path(os.environ.get("STRATA_ROOT", "X:/dev/StrataQ38FN/Strata"))
PY = str(STRATA / ".venv" / "Scripts" / "python.exe")
CONFIG = STRATA / "strata-unsloth-ud-iq4_xs.json"
TOKDIR = Path("X:/dev/StrataQ38FN/Strata-data/packs/unsloth-ud-iq4_xs/tokenizer")
LOGS = HERE / ".testlogs"

REASONING = "Thinking about it."
ANSWER = "Hello from the mock engine."
TOOL_SCRIPT = ("Let me check.</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nBerlin\n"
               "</parameter>\n</function>\n</tool_call>")
JSON_SCRIPT = 'ok</think>\n\n{"value": "yes", "a": 1}'
MOCKS = {8081: None, 8082: TOOL_SCRIPT, 8083: JSON_SCRIPT}
NAMES = ["--served-model-name", "test-model"]
FACADES = {
    8001: (8081, ["--served-model-name", "test-model", "alias-b"]),
    8002: (8082, NAMES),
    8003: (8083, NAMES),
    8004: (8081, NAMES + ["--api-key", "secret-key", "--reasoning-output", "reasoning",
                          "--enable-prompt-tokens-details", "--enable-request-id-headers",
                          "--enable-force-include-usage"]),
    8005: (8081, NAMES + ["--reasoning-output", "inline", "--model-check", "lenient"]),
    8006: (8081, ["--reasoning-output", "reasoning_content", "--completions", "off"]),
    8007: (8089, NAMES),                         # nothing listens on 8089: the backend is down
    8008: (8084, NAMES + ["--client-timeout", "3"]),   # SlowUpstream (in this process): scripted answers
}
SLOW_PORT = 8084
FORBIDDEN = {8000, 8080}
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "weather of a city",
                                           "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                          "required": ["city"]}}}]
HI = [{"role": "user", "content": "hi"}]


# ------------------------------------------------------------------------------------------------ helpers
class Resp:
    def __init__(self, status, headers, body: bytes):
        self.status, self.headers, self.body = status, headers, body

    def json(self):
        return json.loads(self.body)

    def header(self, name):
        return self.headers.get(name)

    def events(self) -> list:
        """SSE data payloads (parsed JSON, or the string "[DONE]"), comments skipped."""
        out = []
        for block in self.body.decode("utf-8").split("\n\n"):
            lines = [x for x in block.split("\n") if x.startswith("data:")]
            if not lines:
                continue
            raw = "\n".join(x[5:].lstrip() for x in lines)
            out.append(raw if raw == "[DONE]" else json.loads(raw))
        return out


def call(port, method, path, body=None, headers=None, timeout=60) -> Resp:
    assert port not in FORBIDDEN
    data = None
    h = dict(headers or {})
    if body is not None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        h.setdefault("Content-Type", "application/json")
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return Resp(resp.status, resp.headers, resp.read())
    except urllib.error.HTTPError as e:
        return Resp(e.code, e.headers, e.read())


def chat(port, body, **kw) -> Resp:
    body = dict({"model": "test-model", "messages": HI}, **body)
    return call(port, "POST", "/v1/chat/completions", body, **kw)


def err(r: Resp, status: int, etype: str, contains: str = "") -> dict:
    assert r.status == status, (r.status, r.body[:400])
    e = r.json()["error"]
    assert set(e) >= {"message", "type", "param", "code"}, e
    assert e["type"] == etype and e["code"] == status, e
    assert contains in e["message"], e
    return e


def hello_ids() -> list[int]:
    return call(8001, "POST", "/tokenize", {"prompt": "Hello world"}).json()["tokens"]


def port_free(port) -> bool:
    s = socket.socket()
    s.settimeout(0.3)
    try:
        return s.connect_ex(("127.0.0.1", port)) != 0
    finally:
        s.close()


# ------------------------------------------------------------------------------------------------ slow upstream
GONE = {"stream": threading.Event(), "nonstream": threading.Event()}
RECORDED: list = []                                # "record" requests as the facade sent them
HANG_RELEASE = threading.Event()                   # ends the "hang" requests
UP_MODE = {"get": "ok"}                            # "incomplete": GETs announce 500 bytes and send 8


class SlowUpstream(BaseHTTPRequestHandler):
    """A fake Strata (HTTP/1.0, like Strata).  Chat requests are scripted by the last user message:
    "record" answers at once and keeps the request; "hang" accepts and never answers nor reads (a frozen backend);
    "cut" streams one chunk and closes; anything else takes 10 s (keep-alives when streamed) and notes when the
    facade closes the connection, which is what the facade does when its own client goes away."""
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):
        pass

    def _json(self, obj, status=200):
        b = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _sse_head(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

    def _sse(self, obj, event=None):
        self.wfile.write((f"event: {event}\n" if event else "").encode() + b"data: " + json.dumps(obj).encode() +
                         b"\n\n")
        self.wfile.flush()

    def do_GET(self):
        if UP_MODE["get"] == "incomplete":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "500")
            self.end_headers()
            self.wfile.write(b'{"x": 1}')
            return
        if self.path == "/v1/models":
            self._json({"object": "list", "data": [{"id": "slow", "object": "model", "meta": {"n_ctx": 4096}}]})
        elif self.path == "/metrics":
            self._json({"engine": {"max_context": 4096}, "live": {"state": "idle", "queued": 0}, "totals": {}})
        else:
            self._json({"status": "ok", "max_context": 4096, "model": "slow", "loaded": True})

    def _gone(self) -> bool:
        try:
            r, _, _ = select.select([self.connection], [], [], 0)
            return bool(r) and self.connection.recv(1, socket.MSG_PEEK) == b""
        except OSError:
            return True

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        msgs = req.get("messages") or [{}]
        content = msgs[-1].get("content") if isinstance(msgs[-1], dict) else None
        stream = bool(req.get("stream"))
        if content == "hang":
            HANG_RELEASE.wait(30)
            return
        if self.path == "/v1/messages":
            msg = {"id": "msg_1", "type": "message", "role": "assistant", "model": "slow", "content": [],
                   "stop_reason": None, "usage": {"input_tokens": 3, "output_tokens": 0}}
            self._sse_head()
            self._sse({"type": "message_start", "message": msg}, "message_start")
            if content == "cut":
                return                              # closed after one event: a stream that broke off
            self._sse({"type": "message_stop"}, "message_stop")
            return
        chunk = {"id": "chatcmpl-up1", "object": "chat.completion.chunk", "created": 1, "model": "slow"}
        if content == "record":
            RECORDED.append(req)
            usage = {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
            if not stream:
                return self._json({"id": "chatcmpl-up1", "object": "chat.completion", "created": 1, "model": "slow",
                                   "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                                                "finish_reason": "stop"}], "usage": usage})
            self._sse_head()
            self._sse(dict(chunk, choices=[{"index": 0, "delta": {"role": "assistant", "content": ""},
                                            "finish_reason": None}]))
            self._sse(dict(chunk, choices=[{"index": 0, "delta": {"content": "ok"}, "finish_reason": None}]))
            self._sse(dict(chunk, choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}], usage=usage))
            self.wfile.write(b"data: [DONE]\n\n")
            return
        if content == "cut":
            self._sse_head()
            self._sse(dict(chunk, choices=[{"index": 0, "delta": {"role": "assistant", "content": ""},
                                            "finish_reason": None}]))
            self._sse(dict(chunk, choices=[{"index": 0, "delta": {"content": "par"}, "finish_reason": None}]))
            return                                  # no finish, no [DONE]
        mode = "stream" if stream else "nonstream"
        if mode == "stream":
            self._sse_head()
        for _ in range(100):
            if self._gone():
                GONE[mode].set()
                return
            try:
                if mode == "stream":
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
            except OSError:
                GONE[mode].set()
                return
            time.sleep(0.1)


# ------------------------------------------------------------------------------------------------ processes
class Procs:
    def __init__(self):
        self.procs: list[subprocess.Popen] = []
        self.files = []
        self.servers = []

    def start(self, name, args):
        LOGS.mkdir(exist_ok=True)
        f = open(LOGS / f"{name}.log", "w", encoding="utf-8")
        self.files.append(f)
        p = subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT, cwd=str(HERE),
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.procs.append(p)
        print(f"  started {name} (pid {p.pid})", flush=True)
        return p

    def stop(self):
        for p in self.procs:
            if p.poll() is None:
                p.terminate()                     # TerminateProcess on this PID only
        for p in self.procs:
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()
        for f in self.files:
            f.close()
        HANG_RELEASE.set()
        for srv in self.servers:
            srv.shutdown()
            srv.server_close()
        print(f"  stopped {len(self.procs)} processes: {', '.join(str(p.pid) for p in self.procs)}", flush=True)


def wait_ready(port, path, deadline):
    while time.time() < deadline:
        try:
            r = call(port, "GET", path, timeout=3)
            if r.status == 200:
                return
        except OSError:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"port {port} did not become ready ({path})")


def start_all(procs: Procs):
    ports = list(MOCKS) + list(FACADES) + [8089, SLOW_PORT]
    busy = [p for p in ports if not port_free(p)]
    if busy:
        raise SystemExit(f"ports already in use: {busy} (stop whatever listens there first)")
    srv = ThreadingHTTPServer(("127.0.0.1", SLOW_PORT), SlowUpstream)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    procs.servers.append(srv)
    for port, script in MOCKS.items():
        args = [PY, str(STRATA / "serve" / "server.py"), "--engine", "mock", "--port", str(port),
                "--tokenizer", str(TOKDIR)]
        if script:
            args += ["--script", script]
        procs.start(f"mock{port}", args)
    for port, (up, extra) in FACADES.items():
        procs.start(f"facade{port}", [PY, str(HERE / "vllm_facade.py"), "--host", "127.0.0.1", "--port", str(port),
                                      "--upstream", f"http://127.0.0.1:{up}", "--strata-root", str(STRATA),
                                      "--strata-config", str(CONFIG), *extra])
    deadline = time.time() + 120
    for port in MOCKS:
        wait_ready(port, "/health", deadline)
    for port in FACADES:
        wait_ready(port, "/version", deadline)


# ------------------------------------------------------------------------------------------------ tests
TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


@test
def health_ping_version_load():
    r = call(8001, "GET", "/health")
    assert r.status == 200 and r.body == b"" and r.header("Content-Length") == "0", (r.status, r.body)
    assert call(8001, "GET", "/ping").status == 200
    assert call(8001, "POST", "/ping", {}).status == 200
    r = call(8001, "GET", "/version")
    assert r.json() == {"version": "0.31.0+strata"}, r.body
    assert call(8001, "GET", "/load").json() == {"server_load": 0}


@test
def models():
    r = call(8001, "GET", "/v1/models")
    assert r.status == 200, r.body
    j = r.json()
    assert j["object"] == "list" and [d["id"] for d in j["data"]] == ["test-model", "alias-b"], j
    d = j["data"][0]
    assert d["owned_by"] == "vllm" and d["object"] == "model" and d["parent"] is None, d
    assert d["max_model_len"] == 32760, d
    assert d["root"] == "qwen3.8-flash-next-unsloth-ud-iq4_xs", d
    assert len(d["permission"]) == 1 and d["permission"][0]["object"] == "model_permission", d
    assert d["permission"][0]["id"].startswith("modelperm-")
    # the default served name is Strata's model id (from the config)
    j = call(8006, "GET", "/v1/models").json()
    assert [d["id"] for d in j["data"]] == ["qwen3.8-flash-next-unsloth-ud-iq4_xs"], j


@test
def tokenize_detokenize():
    r = call(8001, "POST", "/tokenize", {"prompt": "Hello world", "return_token_strs": True})
    assert r.status == 200, r.body
    j = r.json()
    assert j["count"] == 2 and len(j["tokens"]) == 2 and j["max_model_len"] == 32760, j
    assert j["token_strs"] == ["Hello", "Ġworld"], j
    assert call(8001, "POST", "/tokenize", {"prompt": "x"}).json()["token_strs"] is None
    r = call(8001, "POST", "/detokenize", {"tokens": j["tokens"]})
    assert r.json() == {"prompt": "Hello world"}, r.body
    err(call(8001, "POST", "/detokenize", {"tokens": [10 ** 9]}), 400, "BadRequestError", "out of vocabulary")
    # chat form == the prompt Strata builds for the same messages
    n = call(8001, "POST", "/tokenize", {"messages": HI}).json()["count"]
    u = chat(8001, {"max_tokens": 50}).json()["usage"]
    assert n == u["prompt_tokens"], (n, u)
    n_off = call(8001, "POST", "/tokenize", {"messages": HI,
                                             "chat_template_kwargs": {"enable_thinking": False}}).json()["count"]
    u_off = chat(8001, {"chat_template_kwargs": {"enable_thinking": False}}).json()["usage"]
    assert n_off == u_off["prompt_tokens"] and n_off != n, (n_off, u_off, n)


@test
def chat_nonstream_shape():
    r = chat(8001, {"model": "alias-b", "max_tokens": 300})
    assert r.status == 200, r.body
    j = r.json()
    assert j["object"] == "chat.completion" and j["id"].startswith("chatcmpl-"), j
    assert j["model"] == "test-model", j                      # the first served name, never the request's
    assert j["system_fingerprint"].startswith("vllm-0.31.0-"), j
    for k in ("service_tier", "prompt_logprobs", "prompt_token_ids", "kv_transfer_params"):
        assert k in j and j[k] is None, k
    assert "timings" not in j
    c = j["choices"][0]
    assert c["index"] == 0 and c["finish_reason"] == "stop" and c["stop_reason"] is None and c["logprobs"] is None, c
    m = c["message"]
    assert m["role"] == "assistant" and m["content"] == ANSWER, m
    assert m["reasoning"] == REASONING and m["reasoning_content"] == REASONING, m     # --reasoning-output both
    for k in ("refusal", "annotations", "audio", "function_call"):
        assert k in m and m[k] is None, k
    assert "tool_calls" not in m and "strata_mcp" not in m
    u = j["usage"]
    assert u["prompt_tokens"] > 0 and u["completion_tokens"] > 0 and \
        u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"], u
    assert u.get("prompt_tokens_details") is None, u
    # no model -> accepted
    r = call(8001, "POST", "/v1/chat/completions", {"messages": HI})
    assert r.status == 200 and r.json()["model"] == "test-model", r.body


@test
def chat_stream_include_usage():
    r = chat(8001, {"stream": True, "stream_options": {"include_usage": True}})
    assert r.status == 200, r.body
    assert r.header("Content-Type").startswith("text/event-stream"), r.headers
    assert (r.header("Transfer-Encoding") or "").lower() == "chunked", r.headers
    ev = r.events()
    assert ev[-1] == "[DONE]", ev[-3:]
    chunks = ev[:-1]
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant", chunks[0]
    assert all(c["object"] == "chat.completion.chunk" and c["model"] == "test-model" for c in chunks), chunks
    content = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c["choices"])
    reasoning = "".join(c["choices"][0]["delta"].get("reasoning") or "" for c in chunks if c["choices"])
    reasoning_c = "".join(c["choices"][0]["delta"].get("reasoning_content") or "" for c in chunks if c["choices"])
    assert content == ANSWER and reasoning == REASONING and reasoning_c == REASONING, (content, reasoning)
    fin = [c for c in chunks if c["choices"] and c["choices"][0].get("finish_reason")]
    assert len(fin) == 1 and fin[0]["choices"][0]["finish_reason"] == "stop", fin
    assert "usage" not in fin[0] and fin[0]["choices"][0]["stop_reason"] is None, fin[0]
    assert fin[0]["choices"][0]["logprobs"] is None
    last = chunks[-1]
    assert last["choices"] == [] and last["usage"]["total_tokens"] > 0, last
    assert set(last) >= {"id", "object", "created", "model", "choices", "usage"}, last
    assert last["system_fingerprint"].startswith("vllm-"), last
    assert all("timings" not in c for c in chunks)


@test
def chat_stream_without_usage():
    ev = chat(8001, {"stream": True}).events()
    assert ev[-1] == "[DONE]"
    chunks = ev[:-1]
    assert all(c["choices"] for c in chunks), chunks[-1]          # no usage chunk
    assert all("usage" not in c for c in chunks)
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop" and "system_fingerprint" in chunks[-1]
    # --enable-force-include-usage (8004): a usage chunk without stream_options, with prompt_tokens_details
    ev = chat(8004, {"stream": True}, headers={"Authorization": "Bearer secret-key"}).events()
    last = ev[-2]
    assert last["choices"] == [] and "cached_tokens" in last["usage"]["prompt_tokens_details"], last


@test
def reasoning_modes():
    auth = {"Authorization": "Bearer secret-key"}
    m = chat(8004, {}, headers=auth).json()["choices"][0]["message"]           # reasoning
    assert m["reasoning"] == REASONING and "reasoning_content" not in m, m
    m = chat(8006, {"model": None}).json()["choices"][0]["message"]           # reasoning_content
    assert m["reasoning_content"] == REASONING and "reasoning" not in m, m
    m = chat(8005, {}).json()["choices"][0]["message"]                       # inline
    assert m["content"] == f"<think>\n{REASONING}\n</think>\n\n{ANSWER}", m
    assert "reasoning" not in m and "reasoning_content" not in m, m
    ev = chat(8005, {"stream": True}).events()
    text = "".join(c["choices"][0]["delta"].get("content") or "" for c in ev[:-1] if c["choices"])
    assert text == f"<think>\n{REASONING}\n</think>\n\n{ANSWER}", repr(text)
    m = chat(8001, {"include_reasoning": False}).json()["choices"][0]["message"]
    assert m.get("reasoning") is None and m.get("reasoning_content") is None and m["content"] == ANSWER, m
    ev = chat(8001, {"stream": True, "include_reasoning": False}).events()
    assert not any(c["choices"][0]["delta"].get("reasoning") for c in ev[:-1] if c["choices"])
    ev = chat(8004, {"stream": True}, headers=auth).events()
    assert "".join(c["choices"][0]["delta"].get("reasoning") or "" for c in ev[:-1] if c["choices"]) == REASONING
    assert not any("reasoning_content" in c["choices"][0]["delta"] for c in ev[:-1] if c["choices"])


@test
def enable_thinking_mapping():
    on = chat(8001, {}).json()
    off = chat(8001, {"chat_template_kwargs": {"enable_thinking": False}}).json()
    alt = chat(8001, {"chat_template_kwargs": {"thinking": False}}).json()
    eff = chat(8001, {"reasoning_effort": "none"}).json()
    assert on["choices"][0]["message"]["reasoning"] == REASONING
    for j in (off, alt, eff):
        m = j["choices"][0]["message"]
        assert m["reasoning"] is None and m["reasoning_content"] is None, m      # thinking off: no reasoning parse
        assert j["usage"]["prompt_tokens"] == off["usage"]["prompt_tokens"], j["usage"]
    assert off["usage"]["prompt_tokens"] != on["usage"]["prompt_tokens"]
    r = chat(8001, {"chat_template_kwargs": {"enable_thinking": True, "foo": 1}, "user": "u1", "ignore_eos": True})
    assert r.status == 200, r.body
    ignored = r.header("X-Facade-Ignored").split(",")
    assert {"chat_template_kwargs.foo", "user", "ignore_eos"} <= set(ignored), ignored
    err(chat(8001, {"chat_template_kwargs": "x"}), 400, "BadRequestError", "chat_template_kwargs")


@test
def tools_passthrough_and_ids():
    r = chat(8002, {"tools": TOOLS, "messages": [{"role": "user", "content": "Weather in Berlin?"}]})
    assert r.status == 200, r.body
    c = r.json()["choices"][0]
    assert c["finish_reason"] == "tool_calls", c
    tc = c["message"]["tool_calls"]
    assert len(tc) == 1 and tc[0]["type"] == "function" and tc[0]["function"]["name"] == "get_weather", tc
    assert tc[0]["id"].startswith("chatcmpl-tool-") and len(tc[0]["id"]) == len("chatcmpl-tool-") + 32, tc
    assert json.loads(tc[0]["function"]["arguments"]) == {"city": "Berlin"}, tc
    assert c["message"]["reasoning"] == "Let me check."
    # streamed: one id, rewritten consistently; the joined arguments parse
    ev = chat(8002, {"tools": TOOLS, "stream": True}).events()
    pieces = [x for c in ev[:-1] if c["choices"] for x in c["choices"][0]["delta"].get("tool_calls") or []]
    ids = {p["id"] for p in pieces if p.get("id")}
    assert len(ids) == 1 and next(iter(ids)).startswith("chatcmpl-tool-"), pieces
    args = "".join(p["function"].get("arguments") or "" for p in pieces)
    assert json.loads(args) == {"city": "Berlin"}, args
    assert [c for c in ev[:-1] if c["choices"] and c["choices"][0]["finish_reason"]][0]["choices"][0][
        "finish_reason"] == "tool_calls"
    # a multi-turn tool conversation (assistant tool_calls + tool message, arguments as string and dict)
    hist = [{"role": "user", "content": "Weather in Berlin?"},
            {"role": "assistant", "content": None, "reasoning": "need the tool",
             "tool_calls": [{"id": tc[0]["id"], "type": "function",
                             "function": {"name": "get_weather", "arguments": '{"city": "Berlin"}'}}]},
            {"role": "tool", "tool_call_id": tc[0]["id"], "content": "sunny, 21 C"},
            {"role": "assistant", "content": "Sunny.", "tool_calls": [
                {"id": "x", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Rome"}}}]},
            {"role": "tool", "tool_call_id": "x", "content": "rain"}]
    r = chat(8002, {"tools": TOOLS, "messages": hist})
    assert r.status == 200, r.body
    assert chat(8001, {"tools": []}).status == 200                          # tools: [] is dropped, not a 400


@test
def structured_outputs():
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
    j = chat(8003, {"guided_json": schema}).json()
    assert json.loads(j["choices"][0]["message"]["content"])["a"] == 1, j
    j = chat(8003, {"structured_outputs": {"json": json.dumps(schema)}}).json()
    assert json.loads(j["choices"][0]["message"]["content"])["a"] == 1, j
    j = chat(8003, {"guided_choice": ["yes", "no"]}).json()
    assert j["choices"][0]["message"]["content"] == "yes", j                   # wrapped and unwrapped
    j = chat(8003, {"structured_outputs": {"choice": ["yes", "no"]}}).json()
    assert j["choices"][0]["message"]["content"] == "yes", j
    ev = chat(8003, {"guided_choice": ["yes", "no"], "stream": True}).events()
    assert "".join(c["choices"][0]["delta"].get("content") or "" for c in ev[:-1] if c["choices"]) == "yes", ev
    j = chat(8003, {"response_format": {"type": "json_schema",
                                        "json_schema": {"name": "s", "schema": {"type": "string"}}}}).json()
    assert j["choices"][0]["message"]["content"] == '"yes"', j               # non-object root: wrapped
    j = chat(8003, {"response_format": {"type": "json_object"}}).json()
    assert json.loads(j["choices"][0]["message"]["content"]) == {"value": "yes", "a": 1}, j
    err(chat(8003, {"guided_regex": "[a-z]+"}), 400, "BadRequestError", "regex structured output is not supported")
    err(chat(8003, {"structured_outputs": {"grammar": "root ::= x"}}), 400, "BadRequestError", "grammar")
    err(chat(8003, {"response_format": {"type": "structural_tag"}}), 400, "BadRequestError", "structural_tag")
    err(chat(8003, {"guided_json": schema, "tools": TOOLS}), 400, "BadRequestError", "tools")
    # the greeting is no JSON: Strata's 502 structured_output_failed -> 500
    err(chat(8001, {"guided_json": schema}), 500, "InternalServerError")


@test
def model_check():
    e = err(chat(8001, {"model": "nope"}), 404, "NotFoundError", "Valid aliases: test-model, alias-b.")
    assert e["param"] == "model" and e["message"].startswith("The model `nope` does not exist."), e
    assert chat(8005, {"model": "nope"}).status == 200                     # --model-check lenient
    err(call(8001, "POST", "/v1/completions", {"model": "nope", "prompt": "x"}), 404, "NotFoundError")


@test
def rejected_fields():
    err(chat(8001, {"logprobs": True}), 400, "BadRequestError", "logprobs are not supported")
    err(chat(8001, {"top_logprobs": 2}), 400, "BadRequestError", "logprobs")
    err(chat(8001, {"prompt_logprobs": 1}), 400, "BadRequestError", "logprobs")
    assert chat(8001, {"logprobs": False, "logit_bias": None, "top_logprobs": 0}).status == 200
    err(chat(8001, {"n": 2, "stream": True}), 400, "BadRequestError", "n > 1 with streaming")
    err(chat(8001, {"stop": ["a", "b", "c", "d", "e"]}), 400, "BadRequestError", "at most 4 stop")
    err(chat(8001, {"continue_final_message": True}), 400, "BadRequestError", "continue_final_message")
    err(chat(8001, {"chat_template": "{{x}}"}), 400, "BadRequestError", "--trust-request-chat-template")
    err(chat(8001, {"logit_bias": {"1": 5}}), 400, "BadRequestError", "logit_bias")
    err(chat(8001, {"messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {}}]}]}),
        400, "BadRequestError", "input_audio content is not supported")


@test
def n_sequential():
    j = chat(8001, {"n": 2}).json()
    assert [c["index"] for c in j["choices"]] == [0, 1], j
    one = chat(8001, {}).json()["usage"]
    u = j["usage"]
    assert u["prompt_tokens"] == one["prompt_tokens"] and u["completion_tokens"] == 2 * one["completion_tokens"], u


@test
def context_length_error():
    e = err(chat(8001, {"max_tokens": 40000}), 400, "BadRequestError",
            "This model's maximum context length is 32760 tokens. However, you requested 40000 output tokens")
    assert e["param"] == "input_tokens" and "Please reduce the length" in e["message"], e
    err(call(8001, "POST", "/v1/completions", {"prompt": "x", "max_tokens": 40000}), 400, "BadRequestError",
        "maximum context length is 32760")


@test
def completions_emulation():
    r = call(8001, "POST", "/v1/completions", {"model": "test-model", "prompt": "Say hi"})
    assert r.status == 200, r.body
    assert r.header("X-Facade-Completions") == "chat-emulated"
    j = r.json()
    assert j["object"] == "text_completion" and j["id"].startswith("cmpl-") and j["model"] == "test-model", j
    c = j["choices"][0]
    assert c["index"] == 0 and c["text"] and c["logprobs"] is None and c["stop_reason"] is None, c
    assert REASONING not in c["text"] or "</think>" in c["text"]          # thinking is off: no reasoning channel
    assert j["usage"]["total_tokens"] > 0
    j = call(8001, "POST", "/v1/completions", {"prompt": "Say hi", "max_tokens": 3}).json()
    assert j["choices"][0]["finish_reason"] == "length" and j["usage"]["completion_tokens"] == 3, j
    j = call(8001, "POST", "/v1/completions", {"prompt": "Say hi", "echo": True}).json()
    assert j["choices"][0]["text"].startswith("Say hi") and len(j["choices"][0]["text"]) > 6, j
    j = call(8001, "POST", "/v1/completions", {"prompt": ["a", "b"], "max_tokens": 5}).json()
    assert [c["index"] for c in j["choices"]] == [0, 1], j
    j = call(8001, "POST", "/v1/completions", {"prompt": hello_ids(), "echo": True, "max_tokens": 5}).json()
    assert j["choices"][0]["text"].startswith("Hello world"), j
    j = call(8001, "POST", "/v1/completions", {"prompt": [hello_ids()[:1], hello_ids()[1:]], "max_tokens": 2}).json()
    assert len(j["choices"]) == 2, j
    r = call(8001, "POST", "/v1/completions", {"prompt": "Say hi", "stream": True, "echo": True,
                                               "stream_options": {"include_usage": True}})
    ev = r.events()
    assert ev[-1] == "[DONE]" and all(c["object"] == "text_completion" for c in ev[:-1]), ev
    text = "".join(c["choices"][0]["text"] for c in ev[:-1] if c["choices"])
    assert text.startswith("Say hi") and len(text) > 6, text
    assert ev[-2]["choices"] == [] and ev[-2]["usage"]["total_tokens"] > 0, ev[-2]
    assert any(c["choices"] and c["choices"][0]["finish_reason"] for c in ev[:-1])
    err(call(8001, "POST", "/v1/completions", {"prompt": hello_ids(), "stream": True, "echo": True}), 400,
        "BadRequestError", "echo")
    err(call(8001, "POST", "/v1/completions", {"prompt": "x", "suffix": "y"}), 400, "BadRequestError", "suffix")
    err(call(8001, "POST", "/v1/completions", {"prompt": "x", "logprobs": 1}), 400, "BadRequestError", "logprobs")
    err(call(8006, "POST", "/v1/completions", {"prompt": "x"}), 404, "Not Found")        # --completions off
    j = call(8001, "POST", "/generate", {"prompt": "Hi", "max_tokens": 4}).json()
    assert isinstance(j["text"], list) and j["text"][0].startswith("Hi"), j


@test
def auth():
    r = call(8004, "GET", "/v1/models")
    assert r.status == 401 and r.json() == {"error": "Unauthorized"}, (r.status, r.body)
    assert call(8004, "GET", "/v1/models", headers={"Authorization": "Bearer wrong"}).status == 401
    assert chat(8004, {}).status == 401
    assert call(8004, "GET", "/v1/models", headers={"Authorization": "Bearer secret-key"}).status == 200
    for path in ("/health", "/version", "/metrics", "/load", "/ping"):
        assert call(8004, "GET", path).status == 200, path
    assert call(8004, "POST", "/tokenize", {"prompt": "hi"}).status == 200
    assert call(8004, "OPTIONS", "/v1/chat/completions").status == 200
    r = chat(8004, {}, headers={"Authorization": "Bearer secret-key", "X-Request-Id": "abc-123"})
    assert r.status == 200 and r.header("X-Request-Id") == "abc-123", r.headers
    assert "cached_tokens" in r.json()["usage"]["prompt_tokens_details"]    # --enable-prompt-tokens-details
    assert call(8001, "GET", "/v1/models").header("X-Request-Id") is None


@test
def error_format_and_routes():
    for path in ("/v1/embeddings", "/settings", "/v1/load", "/props", "/", "/rerank", "/v1/responses/abc"):
        r = call(8001, "POST", path, {})
        assert r.status == 404 and r.json() == {"error": {"message": "Not Found", "type": "Not Found",
                                                          "param": None, "code": 404}}, (path, r.body)
    assert call(8001, "GET", "/status").status == 404
    err(call(8001, "POST", "/v1/chat/completions", b"{not json"), 400, "BadRequestError")
    err(call(8001, "GET", "/v1/chat/completions"), 405, "Method Not Allowed")
    # HEAD and PATCH: vLLM's 405 (404 for an unknown path), not http.server's 501 HTML page; HEAD has no body
    r = call(8001, "HEAD", "/health")
    assert r.status == 405 and r.body == b"" and r.header("Content-Type") == "application/json", (r.status, r.body)
    assert call(8001, "HEAD", "/nope").status == 404
    err(call(8001, "PATCH", "/v1/models", {}), 405, "Method Not Allowed")
    assert call(8001, "GET", "/v1/models", headers={"Origin": "http://evil.example"}).header(
        "Access-Control-Allow-Origin") is None


@test
def passthrough():
    r = call(8001, "POST", "/v1/messages/count_tokens", {"model": "test-model", "messages": HI})
    assert r.status == 200 and r.json()["input_tokens"] > 0, r.body
    r = call(8001, "POST", "/v1/messages", {"model": "test-model", "max_tokens": 50, "messages": HI})
    assert r.status == 200 and r.json()["type"] == "message", r.body
    r = call(8001, "POST", "/v1/messages", {"model": "test-model", "max_tokens": 50, "messages": HI, "stream": True})
    assert r.status == 200 and b"event: message_stop" in r.body, r.body[-300:]
    # vLLM answers with its served name, not Strata's model id (non-streamed and in the stream's events)
    r = call(8001, "POST", "/v1/messages", {"model": "alias-b", "max_tokens": 50, "messages": HI})
    assert r.json()["model"] == "test-model", r.body
    r = call(8001, "POST", "/v1/messages", {"model": "test-model", "max_tokens": 50, "messages": HI, "stream": True})
    starts = [json.loads(x[5:]) for x in r.body.decode().splitlines() if x.startswith("data:") and "message_start" in x]
    assert starts and starts[0]["message"]["model"] == "test-model", r.body[:400]
    assert b"qwen3.8-flash-next" not in r.body and b"event: error" not in r.body, r.body[:400]
    # facade errors on /v1/messages use vLLM's Anthropic error shape
    r = call(8001, "POST", "/v1/messages", {"model": "nope", "messages": HI})
    assert r.status == 404, r.body
    j = r.json()
    assert j["type"] == "error" and j["error"]["type"] == "NotFoundError" and "nope" in j["error"]["message"], j
    r = call(8001, "POST", "/v1/messages/count_tokens", {"model": "nope", "messages": HI})
    assert r.status == 404 and r.json()["type"] == "error", r.body
    r = call(8001, "POST", "/v1/responses", {"model": "test-model", "input": "hi"})
    assert r.status == 200 and r.json()["object"] == "response", r.body
    assert r.json()["model"] == "test-model", r.body
    r = call(8001, "POST", "/v1/responses", {"model": "test-model", "input": "hi", "stream": True})
    assert r.status == 200 and b"response.completed" in r.body and b"qwen3.8-flash-next" not in r.body, r.body[-300:]
    err(call(8001, "POST", "/v1/responses", {"model": "nope", "input": "hi"}), 404, "NotFoundError")


@test
def keep_alive_http11():
    c = http.client.HTTPConnection("127.0.0.1", 8001, timeout=30)
    try:
        for _ in range(2):
            c.request("POST", "/v1/chat/completions", json.dumps({"messages": HI, "stream": True}),
                      {"Content-Type": "application/json"})
            r = c.getresponse()
            body = r.read()
            assert r.status == 200 and body.rstrip().endswith(b"data: [DONE]"), body[-100:]
        c.request("GET", "/version")
        assert json.loads(c.getresponse().read())["version"] == "0.31.0+strata"
    finally:
        c.close()


@test
def backend_down():
    r = call(8007, "GET", "/health")
    assert r.status == 503 and r.body == b"", r.status
    err(call(8007, "GET", "/v1/models"), 503, "ServiceUnavailableError")
    err(chat(8007, {}), 503, "ServiceUnavailableError", "backend unavailable")


@test
def client_disconnect():
    # non-streamed: the client times out after 1 s; the facade's watcher closes the upstream socket
    try:
        chat(8008, {}, timeout=1)
        raise AssertionError("the slow upstream answered")
    except (TimeoutError, socket.timeout, urllib.error.URLError):
        pass
    assert GONE["nonstream"].wait(5), "the facade kept the upstream request open after its client left"
    # streamed: read the first keep-alive, then hang up
    c = http.client.HTTPConnection("127.0.0.1", 8008, timeout=10)
    c.request("POST", "/v1/chat/completions", json.dumps({"messages": HI, "stream": True}),
              {"Content-Type": "application/json"})
    r = c.getresponse()
    assert r.status == 200 and r.read1(100).find(b"keep-alive") >= 0
    c.sock.shutdown(socket.SHUT_RDWR)
    c.close()
    assert GONE["stream"].wait(5), "the facade kept the upstream stream open after its client left"
    deadline = time.time() + 5
    while time.time() < deadline:
        t = call(8008, "GET", "/metrics").body.decode()
        line = [x for x in t.splitlines() if 'finished_reason="abort"' in x][0]
        if float(line.split()[-1]) >= 2:
            break
        time.sleep(0.2)
    assert float(line.split()[-1]) >= 2, line
    assert call(8008, "GET", "/load").json() == {"server_load": 0}


@test
def metrics():
    r = call(8001, "GET", "/metrics")
    assert r.status == 200 and r.header("Content-Type").startswith("text/plain; version=0.0.4"), r.headers
    t = r.body.decode()
    lab = 'model_name="test-model",engine="0"'
    for name in ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc",
                 "vllm:prompt_tokens_total", "vllm:generation_tokens_total", "vllm:prefix_cache_queries_total",
                 "vllm:prefix_cache_hits_total", "vllm:num_preemptions_total"):
        assert f"{name}{{{lab}}} " in t, name
    vals = {line.split(" ")[0]: float(line.split(" ")[1]) for line in t.splitlines()
            if line and not line.startswith("#")}
    assert vals[f"vllm:prompt_tokens_total{{{lab}}}"] > 0
    assert vals[f'vllm:request_success_total{{{lab},finished_reason="stop"}}'] >= 1
    assert vals[f'vllm:request_success_total{{{lab},finished_reason="length"}}'] >= 1
    assert vals[f"vllm:e2e_request_latency_seconds_count{{{lab}}}"] >= 1
    assert f'vllm:time_to_first_token_seconds_bucket{{{lab},le="+Inf"}}' in vals
    assert f'vllm:request_prompt_tokens_bucket{{{lab},le="+Inf"}}' in vals
    assert vals[f'vllm:engine_sleep_state{{{lab},sleep_state="awake"}}'] == 1.0
    assert call(8001, "GET", "/load").json() == {"server_load": 0}


# ------------------------------------------------------------------------------------------------ review regressions
def raw(port, data: bytes, timeout=10.0) -> tuple[bytes, float]:
    """Send raw bytes, read until the facade closes (or timeout) -> (answer, seconds until closed)."""
    assert port not in FORBIDDEN
    t0 = time.time()
    s = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    out = b""
    try:
        s.sendall(data)
        while True:
            try:
                b = s.recv(65536)
            except ConnectionError:
                break
            if not b:
                break
            out += b
    except socket.timeout:
        pass
    finally:
        s.close()
    return out, time.time() - t0


def metric(port, reason) -> float:
    t = call(port, "GET", "/metrics").body.decode()
    return float([x for x in t.splitlines() if f'finished_reason="{reason}"' in x][0].split()[-1])


@test
def header_injection():
    """Unknown request field names reach X-Facade-Ignored percent-encoded: no CRLF injection, no Latin-1 crash."""
    evil = "x\r\nSet-Cookie: injected=1\r\nX-Evil"
    for stream in (False, True):
        c = http.client.HTTPConnection("127.0.0.1", 8001, timeout=30)
        try:
            c.request("POST", "/v1/chat/completions", json.dumps({"messages": HI, evil: 1, "中": 2, "stream": stream}),
                      {"Content-Type": "application/json"})
            r = c.getresponse()
            body = r.read()
            assert r.status == 200, (r.status, body[:300])
            assert r.getheader("Set-Cookie") is None and r.getheader("X-Evil") is None, r.getheaders()
            ign = r.getheader("X-Facade-Ignored").split(",")
            assert "x%0D%0ASet-Cookie%3A%20injected%3D1%0D%0AX-Evil" in ign and "%E4%B8%AD" in ign, ign
        finally:
            c.close()


@test
def body_limits_and_framing():
    # a huge Content-Length is refused at once (no allocation, no waiting for the body)
    out, dt = raw(8001, b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 300000000\r\n\r\n{")
    assert out.startswith(b"HTTP/1.1 413") and dt < 5, (out[:200], dt)
    # malformed lengths are the client's fault (400), not an internal error
    for hdr in (b"Content-Length: abc\r\n", b"Content-Length: -5\r\n", b"Content-Length: 1_0\r\n"):
        out, _ = raw(8001, b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n" + hdr + b"\r\n{}")
        assert out.startswith(b"HTTP/1.1 400") and b"invalid Content-Length" in out, (hdr, out[:300])
    out, _ = raw(8001, b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\nzz\r\n")
    assert out.startswith(b"HTTP/1.1 400") and b"malformed chunked" in out, out[:300]
    # a well-formed chunked body still works
    body = json.dumps({"messages": HI, "max_tokens": 5}).encode()
    out, _ = raw(8001, b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nConnection: close\r\n"
                       b"Content-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n" +
                 b"%x\r\n%s\r\n0\r\n\r\n" % (len(body), body))
    assert out.startswith(b"HTTP/1.1 200"), out[:300]
    # a client that stops sending its body is dropped after --client-timeout (3 s on 8008), and so is an idle one
    out, dt = raw(8008, b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 100\r\n\r\n{", timeout=15)
    assert out == b"" and dt < 10, (out[:200], dt)
    out, dt = raw(8008, b"", timeout=15)
    assert out == b"" and dt < 10, dt
    assert call(8008, "GET", "/load").json() == {"server_load": 0}


@test
def auth_generate():
    assert call(8004, "POST", "/generate", {"prompt": "Hi", "max_tokens": 4}).status == 401
    r = call(8004, "POST", "/generate", {"prompt": "Hi", "max_tokens": 4},
             headers={"Authorization": "Bearer secret-key"})
    assert r.status == 200 and r.json()["text"][0].startswith("Hi"), r.body


@test
def completions_stream_errors_keep_status():
    # the first upstream answer decides the status line: a real 400 / 503, not 200 with an error event
    err(call(8001, "POST", "/v1/completions", {"prompt": "Hi", "max_tokens": 100000, "stream": True}), 400,
        "BadRequestError", "maximum context length is 32760")
    err(call(8007, "POST", "/v1/completions", {"prompt": "Hi", "stream": True}), 503, "ServiceUnavailableError")
    ev = call(8001, "POST", "/v1/completions", {"prompt": ["a", "b"], "max_tokens": 3, "stream": True,
                                                "echo": True}).events()
    assert ev[-1] == "[DONE]" and {c["choices"][0]["index"] for c in ev[:-1]} == {0, 1}, ev
    assert ev[0]["choices"][0]["text"] == "a", ev[0]


@test
def named_tool_choice_finish_reason():
    named = {"type": "function", "function": {"name": "get_weather"}}
    j = chat(8002, {"tools": TOOLS, "tool_choice": named}).json()
    c = j["choices"][0]
    assert c["finish_reason"] == "stop" and c["message"]["tool_calls"][0]["function"]["name"] == "get_weather", c
    ev = chat(8002, {"tools": TOOLS, "tool_choice": named, "stream": True}).events()
    fins = [c["choices"][0]["finish_reason"] for c in ev[:-1] if c["choices"] and c["choices"][0]["finish_reason"]]
    assert fins == ["stop"], fins
    assert chat(8002, {"tools": TOOLS, "tool_choice": "auto"}).json()["choices"][0]["finish_reason"] == "tool_calls"


@test
def response_id_from_request_id():
    j = chat(8001, {}, headers={"X-Request-Id": "abc"}).json()
    assert j["id"] == "chatcmpl-abc", j["id"]
    r = chat(8001, {"request_id": "xyz"})
    assert r.json()["id"] == "chatcmpl-xyz" and r.header("X-Facade-Ignored") is None, (r.json()["id"], r.headers)
    assert chat(8001, {"request_id": "xyz"}, headers={"X-Request-Id": "hdr"}).json()["id"] == "chatcmpl-hdr"
    ev = chat(8001, {"stream": True, "stream_options": {"include_usage": True}},
              headers={"X-Request-Id": "abc"}).events()
    assert {c["id"] for c in ev[:-1]} == {"chatcmpl-abc"}, {c["id"] for c in ev[:-1]}
    assert call(8001, "POST", "/v1/completions", {"prompt": "x", "max_tokens": 2},
                headers={"X-Request-Id": "abc"}).json()["id"] == "cmpl-abc"
    j = chat(8001, {}).json()
    assert j["id"].startswith("chatcmpl-") and j["id"] != "chatcmpl-abc", j["id"]


@test
def n_gets_distinct_seeds():
    rec = [{"role": "user", "content": "record"}]
    RECORDED.clear()
    j = chat(8008, {"messages": rec, "n": 3, "seed": 7}).json()
    assert [c["index"] for c in j["choices"]] == [0, 1, 2], j
    assert [r.get("seed") for r in RECORDED] == [7, 8, 9], [r.get("seed") for r in RECORDED]
    RECORDED.clear()
    call(8008, "POST", "/v1/completions", {"prompt": "record", "n": 2, "seed": 5, "max_tokens": 2})
    assert [r.get("seed") for r in RECORDED] == [5, 6], [r.get("seed") for r in RECORDED]
    RECORDED.clear()
    chat(8008, {"messages": rec, "n": 2})
    assert [r.get("seed") for r in RECORDED] == [None, None], RECORDED


@test
def continuous_usage():
    def check(chunks):
        final = chunks[-1]
        assert final["choices"] == [] and final["usage"]["total_tokens"] > 0, final
        with_choices = [c for c in chunks if c["choices"]]
        assert with_choices and all("usage" in c for c in with_choices), with_choices
        cts = [c["usage"]["completion_tokens"] for c in with_choices]
        assert cts == sorted(cts) and cts[0] == 0, cts
        assert all(c["usage"]["prompt_tokens"] == final["usage"]["prompt_tokens"] for c in with_choices), \
            [c["usage"] for c in with_choices]
        fin = [c for c in with_choices if c["choices"][0]["finish_reason"]][0]
        assert fin["usage"]["completion_tokens"] == final["usage"]["completion_tokens"], (fin, final)
    # --enable-force-include-usage = usage at the end and on every chunk (vLLM's should_include_usage)
    check(chat(8004, {"stream": True}, headers={"Authorization": "Bearer secret-key"}).events()[:-1])
    check(chat(8001, {"stream": True, "stream_options": {"include_usage": True,
                                                        "continuous_usage_stats": True}}).events()[:-1])
    ev = call(8004, "POST", "/v1/completions", {"prompt": "Say hi", "stream": True, "max_tokens": 5},
              headers={"Authorization": "Bearer secret-key"}).events()
    assert all("usage" in c for c in ev[:-1]) and ev[-2]["choices"] == [], ev


@test
def frozen_upstream_disconnect():
    """The backend accepts and never answers nor reads: the client's disconnect still frees the handler."""
    before = metric(8008, "abort")
    try:
        chat(8008, {"messages": [{"role": "user", "content": "hang"}]}, timeout=1)
        raise AssertionError("the hanging upstream answered")
    except (TimeoutError, socket.timeout, urllib.error.URLError):
        pass
    deadline = time.time() + 6
    while time.time() < deadline and call(8008, "GET", "/load").json()["server_load"] != 0:
        time.sleep(0.2)
    assert call(8008, "GET", "/load").json() == {"server_load": 0}, "the handler is still blocked on the backend"
    assert metric(8008, "abort") >= before + 1


@test
def truncated_streams_are_not_complete():
    before = metric(8008, "error")
    r = call(8008, "POST", "/v1/messages", {"model": "test-model", "max_tokens": 9, "stream": True,
                                            "messages": [{"role": "user", "content": "cut"}]})
    assert r.status == 200 and b"event: error" in r.body and b"ended the stream early" in r.body, r.body
    assert b'"model": "test-model"' in r.body or b'"model":"test-model"' in r.body, r.body
    r = call(8008, "POST", "/v1/messages", {"model": "test-model", "max_tokens": 9, "stream": True,
                                            "messages": [{"role": "user", "content": "whole"}]})
    assert b"message_stop" in r.body and b"event: error" not in r.body, r.body
    c = http.client.HTTPConnection("127.0.0.1", 8008, timeout=30)
    try:
        c.request("POST", "/generate", json.dumps({"prompt": "cut", "stream": True}),
                  {"Content-Type": "application/json"})
        resp = c.getresponse()
        assert resp.status == 200
        try:
            body = resp.read()
            raise AssertionError(f"a cut /generate stream ended like a complete one: {body!r}")
        except http.client.IncompleteRead:
            pass
    finally:
        c.close()
    ev = chat(8008, {"messages": [{"role": "user", "content": "cut"}], "stream": True}).events()
    assert any("error" in e for e in ev[:-1] if isinstance(e, dict)), ev
    deadline = time.time() + 3
    while time.time() < deadline and metric(8008, "error") < before + 3:
        time.sleep(0.2)
    assert metric(8008, "error") >= before + 3, (before, metric(8008, "error"))


@test
def broken_upstream_reply():
    """A reply shorter than its Content-Length is "backend unavailable", not an internal facade error."""
    UP_MODE["get"] = "incomplete"
    try:
        assert call(8008, "GET", "/health").status == 503
        assert call(8008, "GET", "/metrics").status == 200
        r = chat(8008, {"messages": [{"role": "user", "content": "record"}]})
        assert r.status == 200, r.body
    finally:
        UP_MODE["get"] = "ok"
    assert call(8008, "GET", "/health").status == 200


@test
def backend_down_is_fast():
    for _ in range(2):
        t0 = time.time()
        err(chat(8007, {}), 503, "ServiceUnavailableError")
        assert time.time() - t0 < 4.5, time.time() - t0          # one refused connect (~2 s on Windows), not three


@test
def in_process_units():
    sys.path.insert(0, str(HERE))
    sys.dont_write_bytecode = True                 # no __pycache__ next to the facade
    import vllm_facade as vf
    assert vf.Upstream("http://localhost:8089", "").connect_host == "127.0.0.1"
    assert vf.Upstream("http://localhost:8089", "").netloc == "localhost:8089"     # Host header unchanged
    assert vf.Upstream("http://127.0.0.1:8089", "").connect_host == "127.0.0.1"
    assert vf.header_safe("a\r\nSet-Cookie: x") == "aSet-Cookie: x" and vf.header_safe("中") == "?"
    ctx = vf.Ctx()
    ctx.ignored = ["a,b", "c d", "x" * 5000]
    h = ctx.ignored_header()
    assert h.startswith("a%2Cb,c%20d,") and len(h) < 2100, h[:50]
    old = sys.stderr
    sys.stderr = open(os.devnull, "w")
    try:
        for i in range(vf.LOG_ONCE_MAX * 2):
            vf.log_once(f"ignored:{i}" + "y" * 1000, "x")
    finally:
        sys.stderr.close()
        sys.stderr = old
    assert len(vf._logged) <= vf.LOG_ONCE_MAX + 1 and all(len(k) <= 200 for k in vf._logged)
    ctx.named_tool = True
    assert vf.finish_reason("tool_calls", ctx) == "stop" and vf.finish_reason("length", ctx) == "length"
    assert vf.nth_seed({"seed": 2 ** 32 - 1}, 1)["seed"] == 2 ** 32 - 2 and vf.nth_seed({}, 1) == {}


OPENAI_SCRIPT = r'''
import json, sys, openai
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="EMPTY", max_retries=0)
M = c.models.list().data[0].id
assert M == "test-model", M
r = c.chat.completions.create(model=M, messages=[{"role": "user", "content": "hi"}], max_tokens=100)
assert r.choices[0].message.content == "Hello from the mock engine."
assert getattr(r.choices[0].message, "reasoning", None) == "Thinking about it."
s = c.chat.completions.create(model=M, messages=[{"role": "user", "content": "hi"}], stream=True,
                              stream_options={"include_usage": True})
chunks = list(s)
assert chunks[-1].choices == [] and chunks[-1].usage.total_tokens > 0
tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object",
          "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
t = OpenAI(base_url="http://127.0.0.1:8002/v1", api_key="EMPTY", max_retries=0)
r = t.chat.completions.create(model=M, messages=[{"role": "user", "content": "Weather in Berlin?"}], tools=tools)
tc = r.choices[0].message.tool_calls[0]
assert r.choices[0].finish_reason == "tool_calls" and tc.id.startswith("chatcmpl-tool-")
args = ""
for ch in t.chat.completions.create(model=M, messages=[{"role": "user", "content": "x"}], tools=tools, stream=True):
    for d in (ch.choices[0].delta.tool_calls or []) if ch.choices else []:
        args += d.function.arguments or ""
assert json.loads(args) == {"city": "Berlin"}
j = OpenAI(base_url="http://127.0.0.1:8003/v1", api_key="EMPTY", max_retries=0)
r = j.chat.completions.create(model=M, messages=[{"role": "user", "content": "JSON with a=1"}],
    extra_body={"structured_outputs": {"json": {"type": "object", "properties": {"a": {"type": "integer"}},
                "required": ["a"]}}, "top_k": 20, "min_p": 0.05, "repetition_penalty": 1.05})
assert json.loads(r.choices[0].message.content)["a"] == 1
print(c.completions.create(model=M, prompt="Once upon a time", max_tokens=5).choices[0].text[:40])
try:
    c.chat.completions.create(model="nope", messages=[{"role": "user", "content": "x"}])
    raise SystemExit("no 404")
except openai.NotFoundError as e:
    assert "Valid aliases" in str(e)
try:
    c.chat.completions.create(model=M, messages=[{"role": "user", "content": "x"}], max_tokens=10**6)
    raise SystemExit("no 400")
except openai.BadRequestError as e:
    assert "maximum context length" in str(e)
a = OpenAI(base_url="http://127.0.0.1:8004/v1", api_key="wrong", max_retries=0)
try:
    a.models.list()
    raise SystemExit("no 401")
except openai.AuthenticationError:
    pass
print("openai", openai.__version__, "ok")
'''


@test
def openai_client():
    py = HERE / ".testvenv" / "Scripts" / "python.exe"
    if not py.exists():
        print("    (skipped: no .testvenv with openai)")
        return
    p = subprocess.run([str(py), "-c", OPENAI_SCRIPT], capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    print("    " + p.stdout.strip().replace("\n", "\n    "))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", default="", help="run only tests whose name contains this")
    ap.add_argument("--keep-logs", action="store_true")
    a = ap.parse_args()
    procs = Procs()
    failed = []
    try:
        print("starting mocks and facades ...", flush=True)
        start_all(procs)
        for fn in TESTS:
            if a.k not in fn.__name__:
                continue
            t0 = time.time()
            try:
                fn()
                print(f"PASS {fn.__name__} ({time.time() - t0:.1f}s)", flush=True)
            except Exception:
                failed.append(fn.__name__)
                print(f"FAIL {fn.__name__}\n" + traceback.format_exc(), flush=True)
    finally:
        procs.stop()
        if not a.keep_logs and not failed:
            shutil.rmtree(LOGS, ignore_errors=True)
    n = len([f for f in TESTS if a.k in f.__name__])
    print(f"\n{n - len(failed)}/{n} passed" + (f"; failed: {', '.join(failed)} (logs in {LOGS})" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
