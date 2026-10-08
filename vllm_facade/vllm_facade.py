#!/usr/bin/env python3
"""vllm_facade.py - a vLLM-compatible OpenAI server in front of Strata.

Clients written for vLLM (openai-python, LiteLLM hosted_vllm/, LangChain, LlamaIndex, ops tools) talk to this
facade; it translates to Strata's own API on --upstream (default http://127.0.0.1:8080) and normalizes the answers
to vLLM's shapes (v0.31).  Stdlib only, plus the Strata tokenizer/template for /tokenize and /detokenize.

    python vllm_facade.py --upstream http://127.0.0.1:8080 --port 8000 \
        --strata-root X:/dev/StrataQ38FN/Strata --strata-config X:/dev/StrataQ38FN/Strata/strata-unsloth-ud-iq4_xs.json \
        --served-model-name <the name the client sends>
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import http.client
import json
import os
import re
import select
import socket
import _socket
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, urlsplit

DEFAULT_MODEL = "qwen3.8-flash-next"
CTX_SLACK = 8                                    # Strata rejects prompt + max + 8 > ctx
ARGS = None                                      # the parsed command line (main)
UP = None                                        # Upstream
INFO = None                                      # Info
METRICS = None                                   # Metrics
CFG: dict = {}                                   # the Strata run config (only tokenizer, model_name are read)
GEN_DEFAULTS: dict = {}                          # --generation-config
_logged: set = set()
_log_lock = threading.Lock()
LOG_ONCE_MAX = 512                               # keys come from client input (field names, model names): bounded


def log(msg: str) -> None:
    print(f"[vllm-facade] {msg}", file=sys.stderr, flush=True)


def log_once(key: str, msg: str) -> None:
    key = key[:200]
    with _log_lock:
        if key in _logged:
            return
        if len(_logged) >= LOG_ONCE_MAX:
            if "\0full" in _logged:
                return
            key, msg = "\0full", f"{LOG_ONCE_MAX} distinct notices logged: further new ones are not logged"
        _logged.add(key)
    log(msg)


def header_safe(value) -> str:
    """A header value that cannot break the response: no control characters (CR/LF header injection), Latin-1
    only (http.server encodes headers as Latin-1), bounded."""
    v = re.sub(r"[\x00-\x1f\x7f]", "", str(value))
    return v.encode("latin-1", "replace").decode("latin-1")[:4096]


ERROR_TYPES = {400: "BadRequestError", 401: "Unauthorized", 404: "NotFoundError", 405: "MethodNotAllowed",
               409: "ConflictError", 413: "RequestTooLargeError", 422: "UnprocessableEntityError",
               500: "InternalServerError", 501: "NotImplementedError", 503: "ServiceUnavailableError"}


class FacadeError(Exception):
    def __init__(self, status: int, message: str, etype: str | None = None, param: str | None = None):
        super().__init__(message)
        self.status, self.message, self.param = status, message, param
        self.etype = etype or ERROR_TYPES.get(status, "InternalServerError")

    def body(self) -> dict:
        return {"error": {"message": self.message, "type": self.etype, "param": self.param, "code": self.status}}


class ClientGone(Exception):
    """The client closed its connection: the request is aborted."""


def bad(message: str, param: str | None = None) -> FacadeError:
    return FacadeError(400, message, "BadRequestError", param)


def route_error(status: int) -> FacadeError:
    """No such route / method: vLLM's HTTPException handler uses the status phrase as the type."""
    phrase = {404: "Not Found", 405: "Method Not Allowed"}[status]
    return FacadeError(status, phrase, phrase)


def anthropic_error(e: FacadeError) -> dict:
    """/v1/messages errors in vLLM's Anthropic shape."""
    return {"type": "error", "error": {"type": e.etype, "message": e.message}}


def backend_check() -> None:
    """Refresh what is known about Strata; 503 at once when it just refused the connection."""
    if INFO.refresh():
        raise FacadeError(503, "backend unavailable", "ServiceUnavailableError")


# ------------------------------------------------------------------------------------------------ upstream
class Upstream:
    def __init__(self, url: str, key: str):
        u = urlsplit(url if "://" in url else "http://" + url)
        self.https = u.scheme == "https"
        self.host = u.hostname or "127.0.0.1"
        self.port = u.port or (443 if self.https else 80)
        self.netloc = f"{self.host}:{self.port}"
        self.prefix = u.path.rstrip("/")
        self.key = key
        # "localhost" resolves to ::1 first on Windows; a refused ::1 costs ~2 s per connection before 127.0.0.1 is
        # tried.  Strata listens on IPv4, so plain http to localhost goes to 127.0.0.1 (the Host header is unchanged).
        self.connect_host = "127.0.0.1" if self.host.lower() == "localhost" and not self.https else self.host

    def send(self, method: str, path: str, body=None, timeout: float | None = None):
        """-> conn with the request sent.  Only Content-Type, Content-Length, Host and the upstream key are sent:
        never the client's Origin, Host, Cookie or Authorization."""
        cls = http.client.HTTPSConnection if self.https else http.client.HTTPConnection
        conn = cls(self.connect_host, self.port, timeout=timeout)
        headers = {"Host": self.netloc}
        data = None
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        try:
            conn.request(method, self.prefix + path, body=data, headers=headers)
            return conn
        except (ConnectionRefusedError, socket.timeout, socket.gaierror):
            conn.close()
            raise FacadeError(503, "backend unavailable", "ServiceUnavailableError")
        except (http.client.HTTPException, OSError) as e:
            conn.close()
            raise FacadeError(500, f"backend closed the connection ({type(e).__name__})", "InternalServerError")

    @staticmethod
    def recv(conn):
        try:
            return conn.getresponse()
        except socket.timeout:
            conn.close()
            raise FacadeError(503, "backend did not answer in time", "ServiceUnavailableError")
        except (http.client.HTTPException, OSError) as e:
            conn.close()
            raise FacadeError(500, f"backend closed the connection ({type(e).__name__})", "InternalServerError")

    def open(self, method: str, path: str, body=None, timeout: float | None = None):
        conn = self.send(method, path, body, timeout)
        return conn, self.recv(conn)

    def get_json(self, path: str, timeout: float = 2.0):
        """-> (status, obj); (None, None) when unreachable, (-1, None) when the reply broke off."""
        try:
            conn = self.send("GET", path, timeout=timeout)
        except FacadeError:
            return None, None
        try:
            resp = self.recv(conn)
            raw = resp.read()
            try:
                return resp.status, json.loads(raw or b"null")
            except ValueError:
                return resp.status, None
        except (FacadeError, OSError, http.client.HTTPException):
            return -1, None
        finally:
            conn.close()


CTX_RE1 = re.compile(r"prompt \((\d+) tokens\) \+ max tokens \((\d+)\) exceeds the context \((\d+)\)")
CTX_RE2 = re.compile(r"prompt \((\d+) tokens\) leaves no room .*?\((\d+)\)")


def upstream_error(status: int, raw: bytes) -> FacadeError:
    """A Strata error status + body -> the facade's (vLLM-shaped) error."""
    try:
        err = json.loads(raw or b"{}").get("error")
    except (ValueError, AttributeError):
        err = None
    if isinstance(err, dict):
        msg, etype = str(err.get("message") or ""), err.get("type")
    else:
        msg, etype = (str(err) if err else raw[:500].decode("utf-8", "replace")), None
    if status == 400:
        m = CTX_RE1.search(msg)
        if m:
            n, mx, c = (int(x) for x in m.groups())
            return FacadeError(400, f"This model's maximum context length is {c - CTX_SLACK} tokens. However, you "
                                    f"requested {mx} output tokens and your prompt contains {n} input tokens, for a "
                                    f"total of {n + mx} tokens. Please reduce the length of the input prompt or the "
                                    "number of requested output tokens.", "BadRequestError", "input_tokens")
        m = CTX_RE2.search(msg)
        if m:
            n, c = (int(x) for x in m.groups())
            return FacadeError(400, f"Input length ({n}) exceeds model's maximum context length ({c - CTX_SLACK}).",
                               "BadRequestError", "input_tokens")
        return FacadeError(400, msg or "bad request", "BadRequestError")
    if status == 404:
        return FacadeError(404, msg or "Not Found", "NotFoundError")
    if status == 502 and etype == "structured_output_failed":
        return FacadeError(500, msg, "InternalServerError")
    if status == 503:
        return FacadeError(503, msg or "backend unavailable", "ServiceUnavailableError")
    if status in (401, 403, 415):
        log_once(f"misconf{status}", f"Strata rejected the facade's request ({status}: {msg}): check "
                                     "--upstream-api-key / STRATA_API_KEY")
        return FacadeError(500, f"facade misconfigured: upstream rejected ({status})", "InternalServerError")
    if status == 409:
        return FacadeError(409, msg or "conflict", "ConflictError")
    return FacadeError(500, msg or f"backend error ({status})", "InternalServerError")


class Watch:
    """Closes the upstream socket when the client goes away (Strata's own watcher then cancels the engine)."""

    def __init__(self, client_sock, conn):
        self.sock, self.conn, self.gone = client_sock, conn, False
        self.done = threading.Event()
        threading.Thread(target=self._run, daemon=True, name="facade-watch").start()

    def _run(self):
        while not self.done.wait(0.5):
            try:
                readable, _, _ = select.select([self.sock], [], [], 0)
                gone = bool(readable) and self.sock.recv(1, socket.MSG_PEEK) == b""
            except (ConnectionError, TimeoutError):
                gone = True
            except (OSError, ValueError):
                return
            if gone:
                self.gone = True
                self.kill()
                return

    def kill(self):
        s = self.conn.sock
        if s is None:
            return
        try:
            s.shutdown(socket.SHUT_RDWR)            # the FIN: Strata's watcher cancels the engine
        except OSError:
            pass
        if os.name == "nt":
            # On Windows shutdown() does not wake a recv() another thread is blocked in (the handler, waiting for a
            # frozen upstream): only closing the handle does.  socket.close() alone would wait for the response's
            # makefile reference, so the handle itself is closed; the later conn.close() is then a no-op.
            try:
                _socket.socket.close(s)
            except OSError:
                pass

    def stop(self):
        self.done.set()


# ------------------------------------------------------------------------------------------------ upstream info
class Info:
    """Strata's model id, aliases, context and engine version, cached for 10 s (the last known values are kept
    while Strata is unloaded or unreachable)."""

    def __init__(self, model_id: str | None):
        self.lock = threading.Lock()
        self.model_id = model_id
        self.aliases: list[str] = []
        self.n_ctx: int | None = None
        self.reached = False
        self.t = 0.0
        self.engine_version: str | None = None
        self.metrics: dict = {}
        self.metrics_t = 0.0
        self._fp: str | None = None

    def refresh(self, force: bool = False) -> bool:
        """-> True when this call tried Strata and could not connect (the caller answers 503 at once instead of
        paying a second refused connect).  Throttled: every 10 s once reached, every 3 s while not."""
        with self.lock:
            if not force and time.time() - self.t < (10 if self.reached else 3):
                return False
            self.t = time.time()
        model_id, aliases, n_ctx, ok = None, None, None, False
        st, obj = UP.get_json("/v1/models", 3.0)
        if st is None:
            return True
        if st == 200 and isinstance(obj, dict) and obj.get("data"):
            data = [d for d in obj["data"] if isinstance(d, dict)]
            main = next((d for d in data if not d.get("alias_of")), data[0] if data else {})
            model_id, aliases = main.get("id"), list(main.get("aliases") or [])
            n_ctx = (main.get("meta") or {}).get("n_ctx") or None
            ok = True
        if not ok or not n_ctx:
            st, h = UP.get_json("/health", 2.0)
            if st == 200 and isinstance(h, dict):
                model_id = model_id or h.get("model")
                n_ctx = n_ctx or (h.get("max_context") or None)
                ok = True
        with self.lock:
            if ok:
                self.reached = True
                if model_id:
                    self.model_id = model_id
                if aliases is not None:
                    self.aliases = aliases
                if n_ctx:
                    self.n_ctx = int(n_ctx)
        if ok and self.engine_version is None:
            self.upstream_metrics(max_age=0)
        return False

    def upstream_metrics(self, max_age: float = 1.0) -> dict:
        with self.lock:
            if time.time() - self.metrics_t < max_age:
                return self.metrics
        st, obj = UP.get_json("/metrics", 2.0)
        with self.lock:
            self.metrics_t = time.time()
            if st == 200 and isinstance(obj, dict):
                self.metrics = obj
                v = (obj.get("engine") or {}).get("version")
                if v and v != self.engine_version:
                    self.engine_version, self._fp = str(v), None
            return self.metrics

    def strata_model(self) -> str:
        return self.model_id or CFG.get("model_name") or DEFAULT_MODEL

    def max_model_len(self) -> int | None:
        return self.n_ctx - CTX_SLACK if self.n_ctx and self.n_ctx > CTX_SLACK else None

    def fingerprint(self) -> str:
        if self._fp is None:
            base = ARGS.report_version.split("+")[0]
            h = hashlib.sha1((self.strata_model() + (self.engine_version or "")).encode()).hexdigest()[:8]
            self._fp = f"vllm-{base}-{h}"
        return self._fp


def served() -> str:
    return ARGS.served_model_name[0]


def valid_names() -> list[str]:
    out = []
    for n in [*ARGS.served_model_name, *INFO.aliases]:
        if n not in out:
            out.append(n)
    return out


def model_check(req: dict) -> None:
    m = req.get("model")
    if m is None or m == "":
        return
    if m in valid_names():
        return
    if ARGS.model_check == "lenient":
        log_once("model:" + str(m), f"unknown model {m!r} served anyway (--model-check lenient)")
        return
    raise FacadeError(404, f"The model `{m}` does not exist. Valid aliases: {', '.join(ARGS.served_model_name)}.",
                      "NotFoundError", "model")


# ------------------------------------------------------------------------------------------------ metrics
REQUEST_LATENCY_BUCKETS = (0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0,
                           240.0, 480.0, 960.0, 1920.0, 7680.0)
TTFT_BUCKETS = (0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0,
                40.0, 80.0, 160.0, 640.0, 2560.0)


def buckets_1_2_5(max_value: int) -> list[float]:
    out, e = [], 0
    while True:
        for m in (1, 2, 5):
            v = m * 10 ** e
            if v > max_value:
                return out
            out.append(v)
        e += 1


class Hist:
    def __init__(self, buckets):
        self.b = list(buckets)
        self.counts = [0] * len(self.b)
        self.sum, self.n = 0.0, 0

    def observe(self, v: float) -> None:
        self.sum += v
        self.n += 1
        for i, le in enumerate(self.b):
            if v <= le:
                self.counts[i] += 1
                break

    def render(self, name: str, labels: str) -> list[str]:
        out, acc = [], 0
        for le, c in zip(self.b, self.counts):
            acc += c
            out.append(f'{name}_bucket{{{labels},le="{float(le)}"}} {float(acc)}')
        out.append(f'{name}_bucket{{{labels},le="+Inf"}} {float(self.n)}')
        out.append(f"{name}_count{{{labels}}} {float(self.n)}")
        out.append(f"{name}_sum{{{labels}}} {self.sum}")
        return out


class Req:
    """One facade request's timings and outcome, for the histograms."""

    def __init__(self):
        self.t0 = time.monotonic()
        self.t_queue = self.t_first = None
        self.finish = None
        self.prompt = self.gen = 0

    def mark_queue(self):
        if self.t_queue is None:
            self.t_queue = time.monotonic()

    def mark_first(self):
        self.mark_queue()
        if self.t_first is None:
            self.t_first = time.monotonic()

    def add_usage(self, u: dict | None, prompt_once: bool = False):
        if u:
            if not (prompt_once and self.prompt):
                self.prompt += int(u.get("prompt_tokens") or 0)
            self.gen += int(u.get("completion_tokens") or 0)


class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.in_flight = 0
        self.success = {"stop": 0, "length": 0, "abort": 0, "error": 0, "repetition": 0}
        self.e2e = Hist(REQUEST_LATENCY_BUCKETS)
        self.queue = Hist(REQUEST_LATENCY_BUCKETS)
        self.ttft = Hist(TTFT_BUCKETS)
        self.tok_buckets = None
        self.prompt_h = self.gen_h = None

    def begin(self) -> Req:
        with self.lock:
            self.in_flight += 1
        return Req()

    def end(self, r: Req) -> None:
        now = time.monotonic()
        with self.lock:
            self.in_flight -= 1
            if r.finish is None:
                return                                  # refused before it ran (4xx): not a finished request
            reason = {"tool_calls": "stop"}.get(r.finish, r.finish)
            self.success[reason if reason in self.success else "stop"] += 1
            self.e2e.observe(now - r.t0)
            if reason == "abort":
                return
            self.queue.observe((r.t_queue or now) - r.t0)
            self.ttft.observe((r.t_first or now) - r.t0)
            if self.prompt_h is None:
                b = buckets_1_2_5(INFO.max_model_len() or 32768)
                self.prompt_h, self.gen_h = Hist(b), Hist(b)
            self.prompt_h.observe(r.prompt)
            self.gen_h.observe(r.gen)

    def render(self) -> str:
        m = INFO.upstream_metrics()
        live, totals, engine = m.get("live") or {}, m.get("totals") or {}, m.get("engine") or {}
        state = live.get("state")
        busy = state in ("reading", "generating")
        running = int(live.get("running") or 0) if live.get("parallel") else int(busy)
        waiting = int(live.get("queued") or 0) + int(live.get("waiting") or 0)
        ctx = engine.get("max_context") or INFO.n_ctx or 0
        kv = ((live.get("prompt_tokens") or 0) + (live.get("generated") or 0)) / ctx if busy and ctx else 0.0
        name = served().replace("\\", "\\\\").replace('"', '\\"')
        L = f'model_name="{name}",engine="0"'
        out = ["# vllm-facade for Strata: *_total counters come from Strata (they include its web UI's traffic); the "
               "histograms and request_success_total cover facade traffic only."]

        def series(n, kind, help_, value, labels=L):
            out.extend([f"# HELP {n} {help_}", f"# TYPE {n} {kind}", f"{n}{{{labels}}} {float(value)}"])

        series("vllm:num_requests_running", "gauge", "Number of requests in model execution batches.", running)
        series("vllm:num_requests_waiting", "gauge", "Number of requests waiting to be processed.", waiting)
        series("vllm:kv_cache_usage_perc", "gauge", "KV-cache usage. 1 means 100 percent usage (approximate).",
               min(1.0, kv))
        awake = state != "unloaded"
        out += ["# HELP vllm:engine_sleep_state Engine sleep state; awake = 0 means engine is sleeping.",
                "# TYPE vllm:engine_sleep_state gauge",
                f'vllm:engine_sleep_state{{{L},sleep_state="awake"}} {float(awake)}',
                f'vllm:engine_sleep_state{{{L},sleep_state="weights_offloaded"}} {float(not awake)}']
        for n, h, v in (("vllm:prompt_tokens_total", "Number of prefill tokens processed.", totals.get("prompt_tokens")),
                        ("vllm:generation_tokens_total", "Number of generation tokens processed.",
                         totals.get("output_tokens")),
                        ("vllm:prefix_cache_queries_total", "Prefix cache queries, in terms of number of queried "
                         "tokens.", totals.get("prompt_tokens")),
                        ("vllm:prefix_cache_hits_total", "Prefix cache hits, in terms of number of cached tokens.",
                         totals.get("reused")),
                        ("vllm:prompt_tokens_cached_total", "Number of cached prompt tokens (local prefix cache).",
                         totals.get("reused")),
                        ("vllm:num_preemptions_total", "Cumulative number of preemption from the engine.", 0)):
            series(n, "counter", h, v or 0)
        with self.lock:
            out += ["# HELP vllm:request_success_total Count of successfully processed requests (facade traffic).",
                    "# TYPE vllm:request_success_total counter"]
            out += [f'vllm:request_success_total{{{L},finished_reason="{k}"}} {float(v)}'
                    for k, v in self.success.items()]
            b = buckets_1_2_5(INFO.max_model_len() or 32768)
            hists = (("vllm:e2e_request_latency_seconds", "Histogram of e2e request latency in seconds.", self.e2e),
                     ("vllm:time_to_first_token_seconds", "Histogram of time to first token in seconds.", self.ttft),
                     ("vllm:request_queue_time_seconds", "Histogram of time spent in WAITING phase for request.",
                      self.queue),
                     ("vllm:request_prompt_tokens", "Number of prefill tokens processed.", self.prompt_h or Hist(b)),
                     ("vllm:request_generation_tokens", "Number of generation tokens processed.",
                      self.gen_h or Hist(b)))
            for n, h, hist in hists:
                out += [f"# HELP {n} {h}", f"# TYPE {n} histogram"] + hist.render(n, L)
        return "\n".join(out) + "\n"


# ------------------------------------------------------------------------------------------------ tokenizer
class Tokenizer:
    lock = threading.Lock()
    loaded = None                                   # (tok, tpl, frontend, literals)

    @classmethod
    def get(cls):
        with cls.lock:
            if cls.loaded is not None:
                return cls.loaded
            tokdir = Path(CFG.get("tokenizer") or "")
            if not CFG.get("tokenizer") or not (tokdir / "vocab.json").exists():
                raise FacadeError(501, "the tokenizer is not available (set --strata-config to a Strata run config "
                                       "whose \"tokenizer\" directory exists)", "NotImplementedError")
            root = Path(ARGS.strata_root)
            for sub in ("tools", "serve"):
                p = str(root / sub)
                if p not in sys.path:
                    sys.path.append(p)
            import strata_tokenizer
            import frontend
            vocab = json.loads((tokdir / "vocab.json").read_text(encoding="utf-8"))
            tokens = [None] * len(vocab)
            for t, i in vocab.items():
                tokens[i] = t
            tok = strata_tokenizer.Tokenizer(tokens, (tokdir / "merges.txt").read_text(encoding="utf-8").split("\n"),
                                             json.loads((tokdir / "token_type.json").read_text()))
            tpl_path = tokdir / "chat_template.jinja"
            tpl = frontend.ChatTemplate(tpl_path if tpl_path.exists() else root / "serve" / "chat_template.jinja")
            cls.loaded = (tok, tpl, frontend, frontend.literal_tags(getattr(tok, "control_tokens", ())))
            log(f"tokenizer loaded from {tokdir}")
            return cls.loaded


def chat_prompt_ids(req: dict) -> list[int]:
    """The token ids of the prompt Strata builds for these messages/tools (its template and tokenizer)."""
    tok, tpl, fe, literals = Tokenizer.get()
    try:
        messages, tools, kw = fe.openai_to_messages({k: req.get(k) for k in (
            "messages", "tools", "reasoning_effort", "chat_template_kwargs") if req.get(k) is not None})
        marked, marked_tools, changed = fe.mark_think_literals(messages, tools, literals)
        prompt = tpl.render(marked, tools=marked_tools,
                            add_generation_prompt=req.get("add_generation_prompt", True) is not False, **kw)
        plain = ()
        if changed:
            prompt, plain = fe.unmark_think_literals(prompt, literals)
        return tok.encode(prompt, parse_special=True, plain=plain) if plain else \
            tok.encode(prompt, parse_special=True)
    except ValueError as e:
        raise bad(str(e)) from None


def prompt_estimate(up: dict) -> int:
    """Continuous usage stats: the prompt's token count before Strata reports it (0 when it cannot be counted)."""
    try:
        return len(chat_prompt_ids(up))
    except Exception:
        return 0


def decode_ids(ids) -> str:
    tok = Tokenizer.get()[0]
    if not isinstance(ids, list) or not all(isinstance(i, int) and not isinstance(i, bool) for i in ids):
        raise bad("tokens must be a list of integers", "tokens")
    n = len(tok.tokens)
    for i in ids:
        if i < 0 or i >= n:
            raise bad(f"token id {i} is out of vocabulary (0..{n - 1})", "tokens")
    return tok.decode(ids)


# ------------------------------------------------------------------------------------------------ chat translation
FORWARD = {"messages", "stream", "tools", "tool_choice", "response_format", "max_tokens", "max_completion_tokens",
           "temperature", "top_p", "min_p", "repetition_penalty", "frequency_penalty", "presence_penalty", "stop",
           "reasoning_effort"}
HANDLED = {"model", "n", "top_k", "seed", "request_id", "chat_template_kwargs","thinking_token_budget", "include_reasoning",
           "structured_outputs", "guided_json", "guided_choice", "guided_regex", "guided_grammar", "stream_options",
           "logprobs", "top_logprobs", "prompt_logprobs", "return_tokens_as_token_ids", "continue_final_message",
           "add_generation_prompt", "chat_template", "logit_bias", "allowed_token_ids", "bad_words"}
STRIP = {"strata_mcp", "strata_tune", "strata_checkpoint", "return_progress", "experimental_speed_projection"}
UNSUPPORTED_PARTS = {"video_url", "audio_url", "input_audio", "file"}
SO_KINDS = ("json", "regex", "choice", "grammar", "json_object", "structural_tag")
SCHEMA_ROOT_KEYS = ("$defs", "definitions", "$schema", "$id")


class Ctx:
    """What the facade itself does with one request's answer."""

    def __init__(self):
        self.n, self.stream, self.include_usage, self.include_reasoning = 1, False, False, True
        self.continuous_usage = False               # usage on every chunk (vLLM's continuous_usage_stats)
        self.wrap = None                            # None | "json" | "choice": unwrap {"value": ...}
        self.named_tool = False                     # tool_choice names one function: vLLM finishes with "stop"
        self.request_id = None                      # X-Request-Id header or the body's request_id (the response id)
        self.ignored: list[str] = []

    def ignore(self, name: str):
        name = str(name)
        if name not in self.ignored:
            self.ignored.append(name)
            log_once("ignored:" + name, f"request field {name[:100]!r} is not supported by Strata: dropped "
                                        "(listed in the X-Facade-Ignored header)")

    def ignored_header(self) -> str:
        """The X-Facade-Ignored value: client-chosen names, percent-encoded (no CR/LF/comma/non-Latin-1 can reach
        the header), bounded."""
        out = ",".join(quote(n, safe="") for n in self.ignored)
        if len(out) > 2048:
            out = out[:2048].rsplit(",", 1)[0] + ",..."
        return out

    def resp_id(self, prefix: str, fallback: str | None) -> str:
        """vLLM's response id: <prefix>-<X-Request-Id or request_id>, else the backend's (or a random) id."""
        if self.request_id:
            return f"{prefix}-{self.request_id}"
        return fallback or f"{prefix}-{uuid.uuid4().hex}"


def only_objects(node, root, refs=()) -> bool:
    """True when every value the schema accepts is a JSON object (Strata's own rule, serve/structured.py)."""
    if not isinstance(node, dict):
        return False
    if node.get("type") == "object":
        return True
    ref = node.get("$ref")
    if isinstance(ref, str) and ref.startswith("#") and ref not in refs:
        target = root
        for part in ref[1:].split("/")[1:]:
            part = part.replace("~1", "/").replace("~0", "~")
            if not isinstance(target, dict) or part not in target:
                return False
            target = target[part]
        return only_objects(target, root, refs + (ref,))
    for key in ("anyOf", "oneOf"):
        br = node.get(key)
        if isinstance(br, list) and br and all(only_objects(b, root, refs) for b in br):
            return True
    br = node.get("allOf")
    return isinstance(br, list) and any(only_objects(b, root, refs) for b in br)


def wrap_schema(schema: dict) -> dict:
    hoisted = {k: schema[k] for k in SCHEMA_ROOT_KEYS if k in schema}
    inner = {k: v for k, v in schema.items() if k not in hoisted}
    return {"type": "object", "properties": {"value": inner}, "required": ["value"], "additionalProperties": False,
            **hoisted}


def schema_format(schema: dict, name: str = "structured_output", extra: dict | None = None):
    if only_objects(schema, schema):
        return {"type": "json_schema", "json_schema": {**(extra or {}), "name": name, "schema": schema}}, None
    return {"type": "json_schema", "json_schema": {**(extra or {}), "name": name, "schema": wrap_schema(schema)}}, "json"


def unwrap(content: str, kind: str) -> str:
    try:
        v = json.loads(content)["value"]
    except (ValueError, TypeError, KeyError):
        return content
    if kind == "choice" and isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"))


def structured(req: dict, ctx: Ctx):
    """-> the response_format to send (or None)."""
    so, kind, spec = req.get("structured_outputs"), None, None
    if so is not None:
        if not isinstance(so, dict):
            raise bad("structured_outputs must be an object", "structured_outputs")
        kinds = [k for k in SO_KINDS if so.get(k) not in (None, False)]
        if len(kinds) > 1:
            raise bad("structured_outputs: give exactly one of " + ", ".join(SO_KINDS), "structured_outputs")
        if kinds:
            kind, spec = kinds[0], so[kinds[0]]
        for k, v in so.items():
            if k not in SO_KINDS and v not in (None, False):
                ctx.ignore("structured_outputs." + k)
    if kind is None:
        for legacy, k in (("guided_json", "json"), ("guided_choice", "choice"), ("guided_regex", "regex"),
                          ("guided_grammar", "grammar")):
            if req.get(legacy) is not None:
                kind, spec = k, req[legacy]
                break
    if kind in ("regex", "grammar", "structural_tag"):
        raise bad(f"{kind} structured output is not supported by this backend (prompt+validate only)")
    if kind == "json":
        if isinstance(spec, str):
            try:
                spec = json.loads(spec)
            except ValueError:
                raise bad("the structured output JSON schema is not valid JSON") from None
        if not isinstance(spec, dict):
            raise bad("the structured output JSON schema must be an object")
        rf, ctx.wrap = schema_format(spec)
        return rf
    if kind == "json_object":
        return {"type": "json_object"}
    if kind == "choice":
        if not isinstance(spec, list) or not spec:
            raise bad("choice must be a non-empty list")
        ctx.wrap = "choice"
        return {"type": "json_schema", "json_schema": {"name": "structured_output",
                                                       "schema": wrap_schema({"enum": spec})}}
    rf = req.get("response_format")
    if isinstance(rf, dict):
        t = rf.get("type")
        if t == "structural_tag":
            raise bad("structural_tag structured output is not supported by this backend (prompt+validate only)")
        js = rf.get("json_schema")
        if t == "json_schema" and isinstance(js, dict) and isinstance(js.get("schema"), dict):
            extra = {k: v for k, v in js.items() if k not in ("name", "schema")}
            out, ctx.wrap = schema_format(js["schema"], js.get("name") or "structured_output", extra)
            return out
    return rf


def translate_chat(req: dict, ctx: Ctx) -> dict:
    """A vLLM chat request -> Strata's.  Raises FacadeError for what must be refused."""
    up: dict = {}
    n = req.get("n")
    n = 1 if n is None else n
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise bad("n must be a positive integer", "n")
    ctx.n, ctx.stream = n, bool(req.get("stream"))
    if n > 1 and ctx.stream:
        raise bad("n > 1 with streaming is not supported", "n")
    tl = req.get("top_logprobs")
    if req.get("logprobs") or (isinstance(tl, (int, float)) and tl > 0) or req.get("prompt_logprobs") is not None \
            or req.get("return_tokens_as_token_ids"):
        raise bad("logprobs are not supported by this backend", "logprobs")
    if req.get("continue_final_message") or req.get("add_generation_prompt") is False:
        raise bad("continue_final_message / add_generation_prompt=false not supported")
    if req.get("chat_template") is not None:
        raise bad("Chat template is passed with request, but --trust-request-chat-template is not set. Refused "
                  "request with untrusted chat template.", "chat_template")
    for k in ("logit_bias", "allowed_token_ids", "bad_words"):
        if req.get(k):
            raise bad(f"{k} is not supported by this backend", k)
    stop = req.get("stop")
    if isinstance(stop, list) and len(stop) > 4:
        raise bad("at most 4 stop sequences are supported", "stop")
    for k, v in req.items():
        if k in FORWARD:
            up[k] = v
        elif k in STRIP or k in HANDLED:
            continue
        elif k == "guided_decoding_backend" or k == "guided_whitespace_pattern":
            if v is not None:
                ctx.ignore(k)
        elif v is not None:
            ctx.ignore(k)
    # messages (2.2)
    msgs = req.get("messages")
    if isinstance(msgs, list):
        out = []
        for m in msgs:
            if isinstance(m, dict):
                m = dict(m)
                if m.get("role") == "assistant" and m.get("reasoning") and not m.get("reasoning_content"):
                    m["reasoning_content"] = m["reasoning"]
                content = m.get("content")
                if isinstance(content, list):
                    for p in content:
                        if isinstance(p, dict) and p.get("type") in UNSUPPORTED_PARTS:
                            raise bad(f"{p['type']} content is not supported", "messages")
            out.append(m)
        up["messages"] = out
    # tools
    if "tools" in up and not up["tools"]:
        up.pop("tools")
        up.pop("tool_choice", None)
    tc = up.get("tool_choice")
    ctx.named_tool = isinstance(tc, dict) and isinstance(tc.get("function"), dict) and bool(tc["function"].get("name"))
    rid = req.get("request_id")
    if isinstance(rid, str) and rid:
        ctx.request_id = rid
    # top_k, seed
    top_k = req.get("top_k")
    if top_k is not None and top_k not in (-1, 0):
        up["top_k"] = top_k
    seed = req.get("seed")
    if seed is not None:
        if isinstance(seed, int) and not isinstance(seed, bool) and seed > 0:
            up["seed"] = seed
        else:
            ctx.ignore("seed")
    # thinking
    ctk = req.get("chat_template_kwargs")
    if ctk is not None:
        if not isinstance(ctk, dict):
            raise bad("chat_template_kwargs must be an object", "chat_template_kwargs")
        kw = {}
        for k, v in ctk.items():
            if k == "enable_thinking" and isinstance(v, bool):
                kw["enable_thinking"] = v
            elif k == "thinking" and isinstance(v, bool):
                if v is False and "enable_thinking" not in ctk:
                    kw["enable_thinking"] = False
            elif k == "reasoning_effort":
                kw["reasoning_effort"] = v
            else:
                ctx.ignore("chat_template_kwargs." + k)
        if kw:
            up["chat_template_kwargs"] = kw
    if req.get("thinking_token_budget") is not None:
        up["reasoning_budget_tokens"] = req["thinking_token_budget"]
    ctx.include_reasoning = req.get("include_reasoning") is not False
    so = req.get("stream_options")
    # vLLM's should_include_usage: the force flag means usage at the end *and* on every chunk
    if ARGS.enable_force_include_usage:
        ctx.include_usage = ctx.continuous_usage = True
    elif isinstance(so, dict):
        ctx.include_usage = bool(so.get("include_usage"))
        ctx.continuous_usage = ctx.include_usage and bool(so.get("continuous_usage_stats"))
    # structured outputs (2.4)
    rf = structured(req, ctx)
    up.pop("response_format", None)
    if rf is not None:
        up["response_format"] = rf
        if isinstance(rf, dict) and rf.get("type") not in (None, "text") and up.get("tools"):
            raise bad("structured output together with tools is not supported by this backend")
    # sampling defaults (Strata is greedy by default)
    for k, v in GEN_DEFAULTS.items():
        if req.get(k) is None and not (k == "top_k" and top_k in (-1, 0)):
            up[k] = v
    up["model"] = INFO.strata_model()
    return up


# ------------------------------------------------------------------------------------------------ normalization
def norm_usage(u: dict | None, drop_none: bool = False) -> dict | None:
    if not isinstance(u, dict):
        return None
    pt, ct = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
    out = {"prompt_tokens": pt, "total_tokens": int(u.get("total_tokens") or pt + ct), "completion_tokens": ct,
           "prompt_tokens_details": None, "completion_tokens_details": None}
    if ARGS.enable_prompt_tokens_details:
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        out["prompt_tokens_details"] = {"cached_tokens": cached, "created_cache_tokens": 0, "multimodal_tokens": None}
    if drop_none:
        out = {k: v for k, v in out.items() if v is not None}
        if "prompt_tokens_details" in out:
            out["prompt_tokens_details"] = {k: v for k, v in out["prompt_tokens_details"].items() if v is not None}
    return out


class IdMap(dict):
    def get_id(self, strata_id):
        if strata_id is None:
            return None
        if strata_id not in self:
            self[strata_id] = "chatcmpl-tool-" + uuid.uuid4().hex
        return self[strata_id]


def rewrite_call(c: dict, ids: IdMap) -> dict:
    c = dict(c)
    if "id" in c:
        c["id"] = ids.get_id(c["id"])
    return c


def set_reasoning(m: dict, r: str | None, ctx: Ctx) -> None:
    if not ctx.include_reasoning:
        r = None
    mode = ARGS.reasoning_output
    if mode == "inline":
        if r:
            m["content"] = "<think>\n" + r + "\n</think>\n\n" + (m.get("content") or "")
        return
    if mode in ("reasoning", "both"):
        m["reasoning"] = r
    if mode in ("reasoning_content", "both"):
        m["reasoning_content"] = r


def normalize_chat(r: dict, ctx: Ctx) -> dict:
    ids = IdMap()
    out = {"id": ctx.resp_id("chatcmpl", r.get("id")), "object": "chat.completion",
           "created": r.get("created") or int(time.time()), "model": served(), "choices": [],
           "service_tier": None, "system_fingerprint": INFO.fingerprint(), "usage": norm_usage(r.get("usage")),
           "prompt_logprobs": None, "prompt_token_ids": None, "kv_transfer_params": None}
    for ch in r.get("choices") or []:
        msg = ch.get("message") or {}
        content = msg.get("content")
        if content and ctx.wrap:
            content = unwrap(content, ctx.wrap)
        m = {"role": msg.get("role") or "assistant", "content": content, "refusal": None, "annotations": None,
             "audio": None, "function_call": None}
        calls = [rewrite_call(c, ids) for c in msg.get("tool_calls") or []]
        if calls:
            m["tool_calls"] = calls
        set_reasoning(m, msg.get("reasoning_content"), ctx)
        out["choices"].append({"index": ch.get("index", 0), "message": m, "logprobs": None,
                               "finish_reason": finish_reason(ch.get("finish_reason"), ctx), "stop_reason": None})
    return out


def finish_reason(fin, ctx: Ctx):
    """vLLM reports "stop" (not "tool_calls") when tool_choice named the function."""
    return "stop" if fin == "tool_calls" and ctx.named_tool else fin


def nth_seed(up: dict, i: int) -> dict:
    """The i-th of n sequential calls: with a seed, seed+i, so the n samples differ (vLLM gives n distinct ones)."""
    seed = up.get("seed")
    if i == 0 or not isinstance(seed, int):
        return up
    s = seed + i
    return dict(up, seed=s if s < 2 ** 32 else max(1, seed - i))


def merge_choices(results: list[dict]) -> dict:
    """n sequential answers -> one response (choices[i].index = i, prompt tokens counted once)."""
    out = dict(results[0], choices=[])
    if out.get("usage"):
        out["usage"] = dict(out["usage"])
    comp = 0
    for i, r in enumerate(results):
        for ch in r["choices"]:
            out["choices"].append(dict(ch, index=i))
        comp += (r.get("usage") or {}).get("completion_tokens") or 0
    if out.get("usage"):
        out["usage"]["completion_tokens"] = comp
        out["usage"]["total_tokens"] = out["usage"]["prompt_tokens"] + comp
    return out


class ChatStream:
    """Strata chat.completion.chunk -> vLLM's."""

    def __init__(self, ctx: Ctx, rec: Req):
        self.ctx, self.rec, self.ids = ctx, rec, IdMap()
        self.usage = None
        self.cid, self.created = None, None
        self.think_open = self.think_closed = False
        self.finish = None
        self.errored = False
        self.prompt_est = 0                          # continuous usage: the prompt's tokens (local tokenizer)
        self.pieces = 0                              # ... and the generated pieces so far (~ tokens)

    def transform(self, c: dict) -> dict | None:
        if "error" in c:
            self.errored = True
            e = c["error"] if isinstance(c["error"], dict) else {"message": str(c["error"])}
            return {"error": {"message": str(e.get("message") or "backend error"), "type": "InternalServerError",
                              "param": None, "code": 500}}
        final = None
        if "usage" in c:
            self.usage = final = c.pop("usage")
            self.rec.add_usage(self.usage)
        c.pop("timings", None)
        c.pop("prompt_progress", None)
        c.pop("strata_mcp", None)
        c["id"] = self.ctx.resp_id("chatcmpl", c.get("id"))
        self.cid, self.created = c.get("id"), c.get("created")
        c["model"] = served()
        keep = False
        mode = ARGS.reasoning_output
        for ch in c.get("choices") or []:
            d = ch.get("delta") or {}
            new = {}
            if "role" in d:
                new["role"] = d["role"]
                keep = True
            r = d.get("reasoning_content") if self.ctx.include_reasoning else None
            content = d.get("content")
            if content and self.ctx.wrap:
                content = unwrap(content, self.ctx.wrap)
            calls = [rewrite_call(x, self.ids) for x in d.get("tool_calls") or []]
            finish = finish_reason(ch.get("finish_reason"), self.ctx)
            ch["finish_reason"] = finish
            if d.get("reasoning_content") or d.get("content") or d.get("tool_calls"):
                self.pieces += 1
            if r or content or calls:
                self.rec.mark_first()
            if mode == "inline":
                text = ""
                if r:
                    if not self.think_open:
                        text, self.think_open = "<think>\n", True
                    text += r
                if self.think_open and not self.think_closed and (content or calls or finish is not None):
                    text += "\n</think>\n\n"
                    self.think_closed = True
                if text or "content" in d:
                    new["content"] = text + (content or "")
            else:
                if r and mode in ("reasoning", "both"):
                    new["reasoning"] = r
                if r and mode in ("reasoning_content", "both"):
                    new["reasoning_content"] = r
                if "content" in d and (content or "role" in d):
                    new["content"] = content
            if calls:
                new["tool_calls"] = calls
            ch["delta"] = new
            ch["logprobs"] = None
            if finish is not None:
                ch["stop_reason"] = None
                self.finish = finish
                if not self.ctx.include_usage:
                    c["system_fingerprint"] = INFO.fingerprint()
            if new or finish is not None:
                keep = True
        if keep and self.ctx.continuous_usage:
            if final:
                pt, ct = int(final.get("prompt_tokens") or 0), int(final.get("completion_tokens") or 0)
            else:
                pt, ct = self.prompt_est, self.pieces
            c["usage"] = {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}
        return c if keep else None

    def usage_chunk(self) -> dict | None:
        if not self.ctx.include_usage or self.usage is None:
            return None
        return {"id": self.cid, "object": "chat.completion.chunk", "created": self.created, "model": served(),
                "choices": [], "usage": norm_usage(self.usage, drop_none=True),
                "system_fingerprint": INFO.fingerprint()}


PASSTHROUGH_ENDS = {"message_stop", "error", "response.completed", "response.incomplete", "response.failed"}


def _rename(obj) -> bool:
    """Strata's model id -> the served name (vLLM answers with its served name), at the top and in the
    message/response objects of Anthropic and Responses events."""
    changed = False
    for o in (obj, obj.get("message"), obj.get("response")):
        if isinstance(o, dict) and isinstance(o.get("model"), str):
            o["model"] = served()
            changed = True
    return changed


def rename_model(raw: bytes) -> bytes:
    try:
        obj = json.loads(raw)
    except ValueError:
        return raw
    if isinstance(obj, dict) and _rename(obj):
        return json.dumps(obj, ensure_ascii=False).encode("utf-8")
    return raw


def passthrough_event(lines: list[bytes]) -> tuple[bytes, str | None]:
    """One SSE event of a passthrough stream -> (its bytes with the model renamed, its type)."""
    out, kind, data_at, data = [], None, None, []
    for line in lines:
        s = line.rstrip(b"\r\n")
        if s.startswith(b"event:"):
            kind = s[6:].strip().decode("utf-8", "replace")
        elif s.startswith(b"data:"):
            if data_at is None:
                data_at = len(out)
                out.append(None)
            data.append(s[5:].lstrip())
            continue
        out.append(s)
    if data_at is not None:
        out[data_at] = b"\n".join(b"data: " + d for d in data)
        try:
            obj = json.loads(b"\n".join(data))
        except ValueError:
            obj = None
        if isinstance(obj, dict):
            if isinstance(obj.get("type"), str):
                kind = kind or obj["type"]
            if _rename(obj):
                out[data_at] = b"data: " + json.dumps(obj, ensure_ascii=False).encode("utf-8")
    return b"\n".join(out) + b"\n", kind


def sse_events(resp):
    """Strata's SSE -> ("comment", line) | ("data", obj) | ("done", None); ends at EOF."""
    data = []
    while True:
        line = resp.readline()
        if not line:
            return
        line = line.rstrip(b"\r\n")
        if not line:
            if data:
                raw = b"\n".join(data)
                data = []
                if raw.strip() == b"[DONE]":
                    yield "done", None
                    continue
                try:
                    yield "data", json.loads(raw)
                except ValueError:
                    continue
            continue
        if line.startswith(b":"):
            yield "comment", line
        elif line.startswith(b"data:"):
            data.append(line[5:].lstrip())


# ------------------------------------------------------------------------------------------------ HTTP server
# vLLM guards /v1 (and its other API prefixes); /generate is the facade's own generation route, so it is guarded too
AUTH_PREFIXES = ("/v1", "/v2", "/inference", "/cohere", "/generate")
ANTHROPIC_PATHS = {"/v1/messages", "/v1/messages/count_tokens"}
GEN_PATHS = {"/v1/chat/completions", "/v1/completions", "/v1/messages", "/v1/responses", "/generate"}
PASSTHROUGH = {"/v1/messages", "/v1/messages/count_tokens", "/v1/responses"}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "vllm-facade"
    sys_version = ""

    def log_message(self, fmt, *args):
        if not ARGS.quiet:
            sys.stderr.write("[vllm-facade] %s %s\n" % (self.client_address[0], fmt % args))

    # ---- plumbing
    def setup(self):
        # an idle keep-alive connection, or a client that stops sending its request or reading its answer, is
        # dropped after --client-timeout seconds instead of holding a thread forever
        self.timeout = ARGS.client_timeout if ARGS.client_timeout > 0 else None
        super().setup()
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def _reset(self, method: str = ""):
        self.rec = None
        self.extra = []
        self.sent = False
        self.chunked = False
        self.raw_body = b""
        self.head_only = method == "HEAD"
        self.request_id = None

    def framing_error(self, status: int, message: str) -> FacadeError:
        """The body's length is unknown or refused: answer, then close (the rest is not read)."""
        self.close_connection = True
        return FacadeError(status, message, None, None)

    def read_exact(self, n: int) -> bytes:
        parts, left = [], n
        try:
            while left > 0:
                b = self.rfile.read(min(left, 1 << 20))      # read as it arrives: no up-front allocation of n
                if not b:
                    raise ClientGone()
                parts.append(b)
                left -= len(b)
        except OSError:                                     # incl. the client timeout
            raise ClientGone() from None
        return b"".join(parts)

    def read_line(self) -> bytes:
        try:
            line = self.rfile.readline(1024)
        except OSError:
            raise ClientGone() from None
        if not line:
            raise ClientGone()
        return line

    def read_body(self) -> bytes:
        limit = ARGS.max_body_mb * 1024 * 1024
        te = (self.headers.get("Transfer-Encoding") or "").lower()
        if "chunked" in te:
            parts, total = [], 0
            while True:
                hexsize = self.read_line().split(b";")[0].strip()
                if not re.fullmatch(rb"[0-9A-Fa-f]{1,16}", hexsize):
                    raise self.framing_error(400, "malformed chunked request body")
                size = int(hexsize, 16)
                if size == 0:
                    for _ in range(64):                     # trailers, up to the blank line
                        if self.read_line() in (b"\r\n", b"\n"):
                            break
                    break
                total += size
                if total > limit:
                    raise self.framing_error(413, f"request body larger than {ARGS.max_body_mb} MiB")
                parts.append(self.read_exact(size))
                self.read_line()
            return b"".join(parts)
        cl = self.headers.get("Content-Length")
        if cl is None:
            return b""
        if not re.fullmatch(r"[0-9]{1,19}", cl.strip()):
            raise self.framing_error(400, "invalid Content-Length")
        n = int(cl.strip())
        if n > limit:
            raise self.framing_error(413, f"request body larger than {ARGS.max_body_mb} MiB")
        return self.read_exact(n) if n else b""

    def json_body(self) -> dict:
        try:
            obj = json.loads(self.raw_body or b"{}")
        except ValueError as e:
            raise bad(f"JSON decode error: {e}") from None
        if not isinstance(obj, dict):
            raise bad("the request body must be a JSON object")
        return obj

    def add_cors(self):
        origins = ARGS.allowed_origins
        origin = self.headers.get("Origin")
        if not origins or not origin:
            return
        if "*" in origins:
            self.extra.append(("Access-Control-Allow-Origin", "*"))
        elif origin.rstrip("/") in origins:
            self.extra += [("Access-Control-Allow-Origin", origin), ("Vary", "Origin")]
        else:
            return
        self.extra += [("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
                       ("Access-Control-Allow-Headers", self.headers.get("Access-Control-Request-Headers") or
                        "Authorization, Content-Type")]

    def head(self, status: int, ctype: str | None, length: int | None):
        self.sent = True
        try:
            self.send_response(status)
            if ctype:
                self.send_header("Content-Type", ctype)
            for k, v in self.extra:
                self.send_header(k, header_safe(v))
            if length is None:
                if self.request_version == "HTTP/1.1":
                    self.chunked = True
                    self.send_header("Transfer-Encoding", "chunked")
                else:
                    self.close_connection = True
                    self.send_header("Connection", "close")
            else:
                self.send_header("Content-Length", str(length))
            self.end_headers()
        except OSError:
            raise ClientGone() from None

    def send_bytes(self, status: int, data: bytes, ctype: str | None = "application/json"):
        self.head(status, ctype if data or ctype != "application/json" else None, len(data))
        if data and not self.head_only:
            try:
                self.wfile.write(data)
            except OSError:
                raise ClientGone() from None

    def send_json(self, status: int, obj) -> None:
        self.send_bytes(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def write(self, data: bytes) -> None:
        """One piece of a streamed body; ClientGone when the client went away."""
        try:
            if self.chunked:
                self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
            else:
                self.wfile.write(data)
            self.wfile.flush()
        except OSError:
            raise ClientGone() from None

    def end_stream(self) -> None:
        if self.chunked:
            try:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except OSError:
                raise ClientGone() from None

    def sse(self, obj) -> None:
        self.write(b"data: " + (obj if isinstance(obj, bytes) else
                                json.dumps(obj, ensure_ascii=False).encode("utf-8")) + b"\n\n")

    # ---- dispatch
    def do_GET(self):
        self.dispatch("GET")

    def do_POST(self):
        self.dispatch("POST")

    def do_OPTIONS(self):
        self.dispatch("OPTIONS")

    def do_PUT(self):
        self.dispatch("PUT")

    def do_DELETE(self):
        self.dispatch("DELETE")

    def do_PATCH(self):
        self.dispatch("PATCH")

    def do_HEAD(self):
        self.dispatch("HEAD")                   # no route takes HEAD (as in vLLM): 405/404, headers only

    def dispatch(self, method: str):
        self._reset(method)
        path = self.path.split("?")[0]
        if len(path) > 1:
            path = path.rstrip("/")
        rec = None
        try:
            rid = self.headers.get("X-Request-Id")
            self.request_id = rid or None
            if ARGS.enable_request_id_headers:
                self.extra.append(("X-Request-Id", rid or uuid.uuid4().hex))
            self.add_cors()
            if ARGS.api_key and method != "OPTIONS" and path.startswith(AUTH_PREFIXES) and not self.authorized():
                self.close_connection = True    # refused before its body is read (no memory spent on it)
                return self.send_json(401, {"error": "Unauthorized"})
            self.raw_body = self.read_body()
            if method == "OPTIONS":
                return self.send_bytes(200, b"", None)
            route = ROUTES.get(path)
            if route is None:
                raise route_error(404)
            fn = route.get(method)
            if fn is None:
                raise route_error(405)
            if path in GEN_PATHS:
                rec = METRICS.begin()
                self.rec = rec
            fn(self)
        except FacadeError as e:
            if not self.sent:
                try:
                    self.send_json(e.status, anthropic_error(e) if path in ANTHROPIC_PATHS else e.body())
                except ClientGone:
                    self.close_connection = True
            else:
                self.close_connection = True
        except ClientGone:
            if rec is not None:
                rec.finish = "abort"
            self.close_connection = True
        except Exception:
            log("internal error:\n" + traceback.format_exc())
            if rec is not None:
                rec.finish = "error"
            if not self.sent:
                try:
                    self.send_json(500, FacadeError(500, "internal facade error").body())
                except ClientGone:
                    pass
            self.close_connection = True
        finally:
            if rec is not None:
                METRICS.end(rec)

    def authorized(self) -> bool:
        auth = self.headers.get("Authorization") or ""
        given = auth[7:].strip() if auth[:7].lower() == "bearer " else ""
        ok = False
        for k in ARGS.api_key:
            ok |= hmac.compare_digest(given.encode(), k.encode())
        return ok

    # ---- upstream calls
    def upstream_call(self, path: str, body: dict):
        """-> (conn, resp, watch): the client is watched while Strata queues and works."""
        conn = UP.send("POST", path, body)
        watch = Watch(self.connection, conn)
        try:
            resp = UP.recv(conn)
        except FacadeError:
            watch.stop()
            conn.close()
            if watch.gone:
                raise ClientGone() from None
            raise
        return conn, resp, watch

    def call_json(self, path: str, body: dict) -> dict:
        conn, resp, watch = self.upstream_call(path, body)
        try:
            try:
                raw = resp.read()
            except (OSError, http.client.HTTPException):
                if watch.gone:
                    raise ClientGone() from None
                raise FacadeError(500, "backend closed the connection", "InternalServerError")
            if watch.gone:
                raise ClientGone()
        finally:
            watch.stop()
            conn.close()
        if resp.status != 200:
            raise upstream_error(resp.status, raw)
        try:
            return json.loads(raw)
        except ValueError:
            raise FacadeError(500, "backend sent a body that is not JSON", "InternalServerError") from None

    def open_stream(self, path: str, body: dict):
        conn, resp, watch = self.upstream_call(path, body)
        if resp.status != 200:
            try:
                raw = resp.read()
            except (OSError, http.client.HTTPException):
                raw = b""
            finally:
                watch.stop()
                conn.close()
            raise upstream_error(resp.status, raw)
        return conn, resp, watch

    # ---- simple routes
    def r_health(self):
        st, _ = UP.get_json("/health", 2.0)
        self.send_bytes(200 if st == 200 else 503, b"", None)

    def r_version(self):
        self.send_json(200, {"version": ARGS.report_version})

    def r_load(self):
        with METRICS.lock:
            n = METRICS.in_flight
        self.send_json(200, {"server_load": n})

    def r_metrics(self):
        self.send_bytes(200, METRICS.render().encode("utf-8"), "text/plain; version=0.0.4; charset=utf-8")

    def r_models(self):
        INFO.refresh()
        if not INFO.reached:
            raise FacadeError(503, "the backend (Strata) is not reachable", "ServiceUnavailableError")
        now = int(time.time())
        cards = []
        for name in ARGS.served_model_name:
            cards.append({"id": name, "object": "model", "created": now, "owned_by": "vllm", "root": ARGS.root,
                          "parent": None, "max_model_len": INFO.max_model_len(),
                          "permission": [{"id": "modelperm-" + uuid.uuid4().hex, "object": "model_permission",
                                          "created": now, "allow_create_engine": False, "allow_sampling": True,
                                          "allow_logprobs": True, "allow_search_indices": False, "allow_view": True,
                                          "allow_fine_tuning": False, "organization": "*", "group": None,
                                          "is_blocking": False}]})
        self.send_json(200, {"object": "list", "data": cards})

    def r_tokenize(self):
        req = self.json_body()
        model_check(req)
        tok, tpl, fe, literals = Tokenizer.get()
        if req.get("messages") is not None:
            if req.get("chat_template") is not None:
                raise bad("Chat template is passed with request, but --trust-request-chat-template is not set.",
                          "chat_template")
            if req.get("continue_final_message"):
                raise bad("continue_final_message is not supported")
            ctk = req.get("chat_template_kwargs")
            if ctk is not None and not isinstance(ctk, dict):
                raise bad("chat_template_kwargs must be an object", "chat_template_kwargs")
            ids = chat_prompt_ids(req)
        elif isinstance(req.get("prompt"), str):
            ids = tok.encode(req["prompt"], parse_special=True)
        else:
            raise bad("send \"prompt\" (a string) or \"messages\"")
        self.send_json(200, {"count": len(ids), "max_model_len": INFO.max_model_len(), "tokens": ids,
                             "token_strs": [tok.tokens[i] for i in ids] if req.get("return_token_strs") else None})

    def r_detokenize(self):
        req = self.json_body()
        model_check(req)
        self.send_json(200, {"prompt": decode_ids(req.get("tokens"))})

    # ---- chat
    def r_chat(self):
        req = self.json_body()
        model_check(req)
        backend_check()
        ctx = Ctx()
        up = translate_chat(req, ctx)
        ctx.request_id = self.request_id or ctx.request_id
        if ctx.ignored:
            self.extra.append(("X-Facade-Ignored", ctx.ignored_header()))
        rec = self.rec
        if not ctx.stream:
            up["stream"] = False
            results = []
            for i in range(ctx.n):
                r = normalize_chat(self.call_json("/v1/chat/completions", nth_seed(up, i)), ctx)
                rec.add_usage(r.get("usage"), prompt_once=True)
                results.append(r)
            out = merge_choices(results) if ctx.n > 1 else results[0]
            rec.mark_first()
            rec.finish = (out["choices"][0]["finish_reason"] if out["choices"] else "stop") or "stop"
            return self.send_json(200, out)
        st = ChatStream(ctx, rec)
        if ctx.continuous_usage:
            st.prompt_est = prompt_estimate(up)
        conn, resp, watch = self.open_stream("/v1/chat/completions", up)
        try:
            self.head(200, "text/event-stream", None)
            done = False
            try:
                for kind, obj in sse_events(resp):
                    if kind == "comment":
                        rec.mark_queue()
                        self.write(b": keep-alive\n\n")
                    elif kind == "done":
                        done = True
                        break
                    else:
                        rec.mark_queue()
                        out = st.transform(obj)
                        if out is not None:
                            self.sse(out)
            except (OSError, http.client.HTTPException):
                if watch.gone:
                    raise ClientGone() from None
            if watch.gone:
                raise ClientGone()
            if not done and not st.errored:
                st.errored = True
                self.sse({"error": {"message": "the backend ended the stream early", "type": "InternalServerError",
                                    "param": None, "code": 500}})
            if not st.errored:
                u = st.usage_chunk()
                if u is not None:
                    self.sse(u)
            self.sse(b"[DONE]")
            self.end_stream()
            rec.finish = "error" if st.errored else (st.finish or "stop")
        except ClientGone:
            watch.kill()
            raise
        finally:
            watch.stop()
            conn.close()

    # ---- completions (emulated over chat)
    def completion_prompts(self, req: dict) -> list[tuple[str, bool]]:
        p = req.get("prompt")
        if isinstance(p, str):
            return [(p, False)]
        if isinstance(p, list) and p:
            if all(isinstance(x, str) for x in p):
                return [(x, False) for x in p]
            if all(isinstance(x, int) and not isinstance(x, bool) for x in p):
                return [(decode_ids(p), True)]
            if all(isinstance(x, list) for x in p):
                return [(decode_ids(x), True) for x in p]
        raise bad("prompt must be a string, a list of strings, a list of token ids or a list of token id lists",
                  "prompt")

    def r_completions(self, legacy: bool = False):
        if ARGS.completions == "off":
            raise route_error(404)
        req = self.json_body()
        model_check(req)
        backend_check()
        if req.get("suffix") is not None:
            raise bad("suffix is not currently supported", "suffix")
        if req.get("logprobs") is not None:
            raise bad("logprobs are not supported by this backend", "logprobs")
        if req.get("prompt_embeds") is not None:
            raise bad("prompt_embeds are not supported by this backend", "prompt_embeds")
        prompts = self.completion_prompts(req)
        echo = bool(req.get("echo")) or legacy
        stream = bool(req.get("stream"))
        if echo and stream and any(ids for _, ids in prompts) and not legacy:
            raise bad("echo with token-id prompts is not supported when streaming", "echo")
        log_once("completions", "/v1/completions is emulated over chat: the prompt is sent as one user message, so "
                                "the model answers it instead of continuing it")
        self.extra.append(("X-Facade-Completions", "chat-emulated"))
        chat = {k: v for k, v in req.items() if k not in ("prompt", "echo", "suffix", "logprobs", "prompt_embeds",
                                                          "add_special_tokens", "best_of")}
        if chat.get("max_tokens") is None and chat.get("max_completion_tokens") is None:
            chat["max_tokens"] = 16
        ctk = chat.get("chat_template_kwargs")
        if ctk is None or (isinstance(ctk, dict) and "enable_thinking" not in ctk and "thinking" not in ctk):
            chat["chat_template_kwargs"] = {**(ctk or {}), "enable_thinking": False}
        chat["messages"] = [{"role": "user", "content": "x"}]
        ctx = Ctx()
        up = translate_chat(chat, ctx)
        ctx.request_id = self.request_id or ctx.request_id
        if ctx.ignored:
            self.extra.append(("X-Facade-Ignored", ctx.ignored_header()))
        cid, created, rec = ctx.resp_id("cmpl", None), int(time.time()), self.rec
        if legacy:
            return self.generate_legacy(up, ctx, prompts, stream)
        if not stream:
            choices, usage = [], {"prompt_tokens": 0, "completion_tokens": 0}
            for pi, (text, _) in enumerate(prompts):
                body = dict(up, messages=[{"role": "user", "content": text}], stream=False)
                for j in range(ctx.n):
                    r = self.call_json("/v1/chat/completions", nth_seed(body, j))
                    ch = (r.get("choices") or [{}])[0]
                    content = (ch.get("message") or {}).get("content") or ""
                    if content and ctx.wrap:
                        content = unwrap(content, ctx.wrap)
                    u = r.get("usage") or {}
                    if j == 0:
                        usage["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
                    usage["completion_tokens"] += int(u.get("completion_tokens") or 0)
                    fin = {"tool_calls": "stop"}.get(ch.get("finish_reason"), ch.get("finish_reason"))
                    choices.append({"index": pi * ctx.n + j, "text": (text if echo else "") + content,
                                    "logprobs": None, "finish_reason": fin, "stop_reason": None,
                                    "prompt_logprobs": None})
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
            rec.add_usage(usage)
            rec.mark_first()
            rec.finish = choices[0]["finish_reason"] or "stop"
            return self.send_json(200, {"id": cid, "object": "text_completion", "created": created,
                                        "model": served(), "choices": choices, "service_tier": None,
                                        "system_fingerprint": INFO.fingerprint(), "usage": norm_usage(usage),
                                        "kv_transfer_params": None})
        # streaming: one prompt after the other, each with its own index.  The status line waits for the first
        # upstream answer, so a refused first request (context length, backend down) is a real 400/503, as in vLLM.
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        errored, finish = False, None
        live = {"pt": 0, "ct": 0}                   # continuous usage of the current prompt

        def chunk(i, text, fin=None, final_usage=None):
            c = {"id": cid, "object": "text_completion", "created": created, "model": served(),
                 "choices": [{"index": i, "text": text, "logprobs": None, "finish_reason": fin, "stop_reason": None}]}
            if fin is not None and not ctx.include_usage:
                c["system_fingerprint"] = INFO.fingerprint()
            if ctx.continuous_usage:
                pt, ct = live["pt"], live["ct"]
                if final_usage:
                    pt, ct = int(final_usage.get("prompt_tokens") or 0), int(final_usage.get("completion_tokens") or 0)
                c["usage"] = {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}
            return c

        for pi, (text, _) in enumerate(prompts):
            if errored:
                break
            body = dict(up, messages=[{"role": "user", "content": text}], stream=True)
            if ctx.continuous_usage:
                live["pt"], live["ct"] = prompt_estimate(body), 0
            try:
                conn, resp, watch = self.open_stream("/v1/chat/completions", body)
            except FacadeError as e:
                if not self.sent:
                    raise
                self.sse({"error": {"message": e.message, "type": e.etype, "param": e.param, "code": e.status}})
                errored = True
                break
            try:
                if not self.sent:
                    self.head(200, "text/event-stream", None)
                if echo:
                    self.sse(chunk(pi, text))
            except ClientGone:
                watch.kill()
                watch.stop()
                conn.close()
                raise
            done = False
            try:
                try:
                    for kind, obj in sse_events(resp):
                        if kind == "comment":
                            rec.mark_queue()
                            self.write(b": keep-alive\n\n")
                            continue
                        if kind == "done":
                            done = True
                            break
                        rec.mark_queue()
                        if "error" in obj:
                            e = obj["error"] if isinstance(obj["error"], dict) else {"message": str(obj["error"])}
                            self.sse({"error": {"message": str(e.get("message")), "type": "InternalServerError",
                                                "param": None, "code": 500}})
                            errored = True
                            continue
                        ch = (obj.get("choices") or [{}])[0]
                        piece = (ch.get("delta") or {}).get("content")
                        if piece and ctx.wrap:
                            piece = unwrap(piece, ctx.wrap)
                        fin = ch.get("finish_reason")
                        d = ch.get("delta") or {}
                        if d.get("content") or d.get("reasoning_content") or d.get("tool_calls"):
                            live["ct"] += 1
                        if piece:
                            rec.mark_first()
                            self.sse(chunk(pi, piece))
                        if obj.get("usage"):
                            u = obj["usage"]
                            usage["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
                            usage["completion_tokens"] += int(u.get("completion_tokens") or 0)
                        if fin is not None:
                            finish = {"tool_calls": "stop"}.get(fin, fin)
                            self.sse(chunk(pi, "", finish, obj.get("usage")))
                except (OSError, http.client.HTTPException):
                    if watch.gone:
                        raise ClientGone() from None
                if watch.gone:
                    raise ClientGone()
            except ClientGone:
                watch.kill()
                raise
            finally:
                watch.stop()
                conn.close()
            if not done and not errored:
                self.sse({"error": {"message": "the backend ended the stream early", "type": "InternalServerError",
                                    "param": None, "code": 500}})
                errored = True
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        rec.add_usage(usage)
        if ctx.include_usage and not errored:
            self.sse({"id": cid, "object": "text_completion", "created": created, "model": served(), "choices": [],
                      "usage": norm_usage(usage, drop_none=True), "system_fingerprint": INFO.fingerprint()})
        self.sse(b"[DONE]")
        self.end_stream()
        rec.finish = "error" if errored else (finish or "stop")

    def r_completions_v1(self):
        self.r_completions(False)

    def r_generate(self):
        self.r_completions(True)

    def generate_legacy(self, up: dict, ctx: Ctx, prompts, stream: bool):
        """vLLM's legacy POST /generate: {"text": [prompt + completion]}; streamed as cumulative NUL-separated
        JSON chunks."""
        rec = self.rec
        text, _ = prompts[0]
        body = dict(up, messages=[{"role": "user", "content": text}], stream=stream)
        if not stream:
            outs = []
            for i in range(ctx.n):
                r = self.call_json("/v1/chat/completions", nth_seed(body, i))
                ch = (r.get("choices") or [{}])[0]
                outs.append(text + ((ch.get("message") or {}).get("content") or ""))
                rec.add_usage(r.get("usage"), prompt_once=True)
                rec.finish = ch.get("finish_reason") or "stop"
            rec.mark_first()
            return self.send_json(200, {"text": outs})
        if ctx.n > 1:
            raise bad("n > 1 with streaming is not supported", "n")
        conn, resp, watch = self.open_stream("/v1/chat/completions", body)
        try:
            self.head(200, "application/json", None)
            acc = text
            done = False
            try:
                for kind, obj in sse_events(resp):
                    if kind != "data":
                        if kind == "done":
                            done = True
                            break
                        continue
                    if "error" in obj:
                        break
                    ch = (obj.get("choices") or [{}])[0]
                    piece = (ch.get("delta") or {}).get("content")
                    rec.add_usage(obj.get("usage"))
                    if ch.get("finish_reason"):
                        rec.finish = ch["finish_reason"]
                    if piece:
                        rec.mark_first()
                        acc += piece
                        self.write(json.dumps({"text": [acc]}, ensure_ascii=False).encode("utf-8") + b"\0")
            except (OSError, http.client.HTTPException):
                if watch.gone:
                    raise ClientGone() from None
            if watch.gone:
                raise ClientGone()
            if not done:
                # a backend error or a stream that broke off: this format has no error channel, so the answer is
                # cut (no terminating chunk, connection closed) instead of ending like a complete one
                rec.finish = "error"
                self.close_connection = True
                return
            self.end_stream()
            rec.finish = rec.finish or "stop"
        except ClientGone:
            watch.kill()
            raise
        finally:
            watch.stop()
            conn.close()

    # ---- passthrough (/v1/messages, /v1/messages/count_tokens, /v1/responses)
    def r_passthrough(self):
        req = self.json_body()
        model_check(req)
        backend_check()
        path = self.path.split("?")[0].rstrip("/")
        anthropic = path in ANTHROPIC_PATHS
        for k in STRIP:
            req.pop(k, None)
        req["model"] = INFO.strata_model()
        conn, resp, watch = self.upstream_call(path, req)
        rec = self.rec
        try:
            if resp.status in (401, 403, 415):
                raise upstream_error(resp.status, resp.read())
            ctype = resp.getheader("Content-Type") or "application/json"
            if "text/event-stream" not in ctype:
                try:
                    raw = resp.read()
                except (OSError, http.client.HTTPException):
                    if watch.gone:
                        raise ClientGone() from None
                    raise FacadeError(500, "backend closed the connection", "InternalServerError")
                if resp.status == 200:
                    raw = rename_model(raw)
                if rec is not None:
                    rec.mark_first()
                    rec.finish = "stop" if resp.status == 200 else None
                return self.send_bytes(resp.status, raw, ctype)
            self.head(resp.status, ctype, None)
            # SSE, event by event: the model is renamed, and an end is recognized.  A stream without its last
            # event (message_stop / response.completed|incomplete|failed / error) broke off: it gets an error event
            # instead of looking like a complete answer.
            ended, failed, block = False, False, []
            try:
                while True:
                    line = resp.readline()
                    if not line:
                        break
                    if line.strip():
                        block.append(line)
                        continue
                    if not block:
                        continue
                    data, kind = passthrough_event(block)
                    block = []
                    if kind in PASSTHROUGH_ENDS:
                        ended = True
                        failed = failed or kind in ("error", "response.failed")
                    if rec is not None:
                        rec.mark_first()
                    self.write(data + b"\n")
            except (OSError, http.client.HTTPException):
                if watch.gone:
                    raise ClientGone() from None
            if watch.gone:
                raise ClientGone()
            if block:                                        # a last event without its blank line
                data, kind = passthrough_event(block)
                ended = ended or kind in PASSTHROUGH_ENDS
                self.write(data + b"\n")
            if not ended:
                msg = "the backend ended the stream early"
                ev = {"type": "error", "error": {"type": "api_error", "message": msg}} if anthropic else \
                    {"type": "error", "code": "server_error", "message": msg, "param": None}
                self.write(b"event: error\ndata: " + json.dumps(ev).encode() + b"\n\n")
            self.end_stream()
            if rec is not None:
                rec.finish = "stop" if ended and not failed else "error"
        except ClientGone:
            watch.kill()
            raise
        finally:
            watch.stop()
            conn.close()

    def r_ping(self):
        self.r_health()


ROUTES = {
    "/health": {"GET": Handler.r_health},
    "/ping": {"GET": Handler.r_ping, "POST": Handler.r_ping},
    "/version": {"GET": Handler.r_version},
    "/load": {"GET": Handler.r_load},
    "/metrics": {"GET": Handler.r_metrics},
    "/v1/models": {"GET": Handler.r_models},
    "/tokenize": {"POST": Handler.r_tokenize},
    "/detokenize": {"POST": Handler.r_detokenize},
    "/v1/chat/completions": {"POST": Handler.r_chat},
    "/v1/completions": {"POST": Handler.r_completions_v1},
    "/generate": {"POST": Handler.r_generate},
    "/v1/messages": {"POST": Handler.r_passthrough},
    "/v1/messages/count_tokens": {"POST": Handler.r_passthrough},
    "/v1/responses": {"POST": Handler.r_passthrough},
}


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, *a, **kw):
        self.active, self.active_lock = 0, threading.Lock()
        super().__init__(*a, **kw)

    def process_request(self, request, client_address):
        """At most --max-connections connections (= handler threads) at a time; one more gets a 503 and is closed."""
        with self.active_lock:
            full = self.active >= ARGS.max_connections
            if not full:
                self.active += 1
        if full:
            log_once("maxconn", f"--max-connections {ARGS.max_connections} reached: refusing connections")
            try:
                request.settimeout(1.0)
                body = json.dumps(FacadeError(503, "too many connections").body()).encode()
                request.sendall(b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\n"
                                b"Connection: close\r\nContent-Length: %d\r\n\r\n%s" % (len(body), body))
            except OSError:
                pass
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            with self.active_lock:
                self.active -= 1
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            with self.active_lock:
                self.active -= 1

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionError, TimeoutError)):
            return
        super().handle_error(request, client_address)


# ------------------------------------------------------------------------------------------------ main
def generation_config(value: str) -> dict:
    if value == "auto":
        return {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
    if value == "vllm":
        return {"temperature": 1.0}
    p = Path(value)
    if p.is_dir():
        p = p / "generation_config.json"
    try:
        obj = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else json.loads(value)
    except (ValueError, OSError) as e:
        raise SystemExit(f"--generation-config: not auto, vllm, a JSON object or a generation_config.json ({e})")
    if not isinstance(obj, dict):
        raise SystemExit("--generation-config: expected a JSON object")
    keys = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty", "frequency_penalty")
    return {k: obj[k] for k in keys if obj.get(k) is not None}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="vLLM-compatible OpenAI server in front of Strata")
    ap.add_argument("--upstream", default=os.environ.get("STRATA_UPSTREAM", "http://127.0.0.1:8080"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--strata-root", default="X:/dev/StrataQ38FN/Strata")
    ap.add_argument("--strata-config", default=None, help="Strata run config (only tokenizer and model_name are read)")
    ap.add_argument("--served-model-name", nargs="+", default=None)
    ap.add_argument("--api-key", nargs="+", default=None, help="client keys (also $VLLM_API_KEY)")
    ap.add_argument("--upstream-api-key", default=os.environ.get("STRATA_API_KEY", ""))
    ap.add_argument("--model-check", choices=["strict", "lenient"], default="strict")
    ap.add_argument("--reasoning-output", choices=["reasoning", "reasoning_content", "both", "inline"],
                    default="both")
    ap.add_argument("--generation-config", default="auto")
    ap.add_argument("--enable-force-include-usage", action="store_true")
    ap.add_argument("--enable-prompt-tokens-details", action="store_true")
    ap.add_argument("--enable-request-id-headers", action="store_true")
    ap.add_argument("--allowed-origins", nargs="*", default=[])
    ap.add_argument("--completions", choices=["chat", "off"], default="chat")
    ap.add_argument("--report-version", default="0.31.0+strata")
    ap.add_argument("--root", default=None, help="ModelCard root (default: Strata's model id)")
    ap.add_argument("--quiet", action="store_true", help="no access log")
    ap.add_argument("--client-timeout", type=float, default=60.0,
                    help="seconds a client may stay silent (idle keep-alive, unsent body, unread answer); 0 = none")
    ap.add_argument("--max-body-mb", type=int, default=64, help="largest request body (413 above)")
    ap.add_argument("--max-connections", type=int, default=256, help="concurrent client connections")
    a = ap.parse_args(argv)
    if a.api_key is None:
        env = os.environ.get("VLLM_API_KEY", "").strip()
        a.api_key = [env] if env else []
    a.api_key = [k for k in a.api_key if k]
    return a


def main(argv=None) -> int:
    global ARGS, UP, INFO, METRICS, CFG, GEN_DEFAULTS
    ARGS = parse_args(argv)
    if ARGS.strata_config:
        try:
            CFG = json.loads(Path(ARGS.strata_config).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as e:
            log(f"cannot read --strata-config {ARGS.strata_config}: {e} (tokenizer endpoints are off)")
            CFG = {}
    GEN_DEFAULTS = generation_config(ARGS.generation_config)
    UP = Upstream(ARGS.upstream, ARGS.upstream_api_key)
    INFO = Info(CFG.get("model_name"))
    METRICS = Metrics()
    INFO.refresh(force=True)
    if not ARGS.served_model_name:
        ARGS.served_model_name = [CFG.get("model_name") or INFO.model_id or DEFAULT_MODEL]
    if ARGS.root is None:
        ARGS.root = CFG.get("model_name") or INFO.model_id or DEFAULT_MODEL
    try:
        srv = Server((ARGS.host, ARGS.port), Handler)
    except OSError as e:
        log(f"cannot listen on {ARGS.host}:{ARGS.port}: {e}")
        return 1
    if ARGS.host not in ("127.0.0.1", "localhost", "::1") and not ARGS.api_key:
        log(f"WARNING: listening on {ARGS.host} without --api-key: anyone on the network can use the model")
    log(f"vLLM facade {ARGS.report_version} on http://{ARGS.host}:{ARGS.port} -> Strata {ARGS.upstream} "
        f"({'reachable' if INFO.reached else 'NOT reachable yet'}, model {INFO.strata_model()}, "
        f"n_ctx {INFO.n_ctx})")
    log(f"served model names: {', '.join(ARGS.served_model_name)}; reasoning output: {ARGS.reasoning_output}; "
        f"sampling defaults: {GEN_DEFAULTS}")
    try:
        srv.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
