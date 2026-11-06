#!/usr/bin/env python3
"""Reproduction: a long conversation loses its cache when one short other request runs in between.

The case (2026-11-06): a ~136k-token agent chat (A) ends, one unrelated ~900-token request (B) runs, and A's next
turn (A2 = A + the reply + a new user message) reads everything again (~25 s) - while a ~30k-token chat in the same
situation keeps its cache.  The engine log has the reason for the big one:

  strata serve: conversation cache: skip parking (physical RAM admission; need 3438 MiB plus 16384 MiB floor,
  18102 MiB available, source cgroup; evicted 14, 0 still parked)

i.e. B overwrote A's batch slot, A's slot image went to the conversation cache, the RAM gate wanted the snapshot
plus the 16 GiB floor, evicted every parked conversation and still refused (the evictions did not raise
"available").  A 30k snapshot (~0.8 GiB) fits into the ~1.7 GiB above the floor, a 136k one (~3.4 GiB) does not.

Per scenario and size this script sends, sequentially, on an otherwise idle TEST server:
  [fillers]  --fillers short conversations (each >= --conversation-cache-min-tokens) that get parked, so the
             RAM gate has something to evict (did eviction give RAM back?  then the park is admitted)
  A          system prompt + tools + a multi-turn agent history of ~SIZE tokens, a final user question
  [pressure] --pressure cgroup: the test scope's MemoryMax is set to (its usage - reclaimable) + --avail-mib, i.e.
             the engine sees the same "available" as production (18102 MiB = the 16 GiB floor + ~1.7 GiB) - in
             every arm, also one with another floor (the RAM situation is production's, the policy the arm's)
  B          ~900 tokens, unrelated (no tools, other system prompt) - lands in A's slot (see SLOT FORCING)
  A2         A + the assistant reply + a new user message; variants:
               control    no B at all (A's slot is never overwritten: the baseline)
               interleave B in between, the reply exactly as returned (content, reasoning_content, tool_calls)
               rerender   B in between, the reply re-rendered (no reasoning_content, whitespace normalised,
                          tool-call arguments in another key order) - hypothesis 2: prefix mismatch
               effort     B in between, the reply as returned, but A2 with reasoning_effort --a2-effort (low): the
                          template puts the effort into the system turn (effort_position "start"), so the prefix
                          differs from token ~0 on - hypothesis 2, the strong form
Recorded per request: prompt tokens, reused tokens (/metrics requests[].reused and usage.prompt_tokens_details.
cached_tokens), prompt_read, wall time, time to first token (any delta: reasoning, content or a tool call), which slot
held it afterwards, conversation-cache deltas (parks / evictions / restores) and the test engine's own
"conversation cache:" log lines of that request.  One JSON line per request and one per scenario go to
results/repro_park_ram.jsonl; --report prints the before/after table over all labels in that file.

SLOT FORCING (why B lands in A's slot without any trick on a production-like config).  Strata-143-prod/serve/
server.py pick_slot(), line ~2351: with pipelined batch groups (a layer split -> the engine reports batch_groups > 1,
slot_gs > 0) the server does NOT prefer the slot that holds a request's prefix ("the engine keeps no pipelined slot as
a conversation cache"); it takes the free slot that fills a hole below a busy slot of its group, else the first free
one in slot_order = [0, gs, 2gs, .., 1, gs+1, ..].  On an idle server no slot is busy, so every request sequentially
sent - fillers, A, B, A2 - goes to slot 0.  With STRATA_SLOT_PARK=1 (production) A's slot image is captured when A
ends and handed to the conversation cache "when the slot is overwritten" (generate.cpp slot_park_env, ~line 423):
B's admission into slot 0 is exactly that moment, and the RAM gate (generate.cpp ~line 9915-9955) decides.  A2 then
also lands in slot 0 (holding B) and can only come back from the conversation cache.  The script checks this from
/metrics live.slots[].held_tokens after A and after B and prints "B in A's slot: yes/no".
Without pipelined groups (no layer split) pick_slot prefers the held prefix, then an EMPTY slot, then the least
recently used one: B would go to an empty slot and A's slot would stay untouched.  For that case --occupy N sends N
tiny requests after A (each takes another empty slot; use N = --batch - 1) so that A's slot is the least recently
used one when B comes.  --occupy is not needed (and not wanted) on the production config.

  # static checks (no server):
  python3 -I repro_park_ram.py --dry-run
  /media/raid0/StrataQ38FN/Strata/.venv/bin/python -I repro_park_ram.py --dry-run --exact   # template + tokenizer
  # GPU window, production stopped (systemctl --user stop strata.service), test port only:
  python3 -I repro_park_ram.py --write-config configs/REPRO-prod.json --arm prod
  python3 -I repro_park_ram.py --launch configs/REPRO-prod.json --label before --pressure cgroup
  python3 -I repro_park_ram.py --write-config configs/REPRO-malloc.json --arm malloc
  python3 -I repro_park_ram.py --launch configs/REPRO-malloc.json --label after-malloc --pressure cgroup
  python3 -I repro_park_ram.py --report
  # a server already running on the test port (started by hand in the strata-bench scope):
  python3 -I repro_park_ram.py --label before --pressure cgroup --unit strata-bench.scope \\
      --engine-log results/engine-repro_park_ram.log
"""
import argparse, datetime, json, random, shlex, subprocess, sys, time, urllib.parse, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROD_CONFIG = HERE / "configs" / "production.json"
SERVER_DIR = ROOT / "Strata-143-prod"            # production's HTTP layer (see strata-prod-ram-kv-cache)
VENV_PY = ROOT / "Strata" / ".venv" / "bin" / "python"
RESULTS = HERE / "results"
OUT_JSONL = RESULTS / "repro_park_ram.jsonl"
BENCH_TIER = "/media/raid0/strata-kvspill-bench"  # never production's tier dir
PROD_PORTS = (8090, 8000)                         # production Strata and the G3 proxy
CHARS_PER_TOKEN = 3.55                            # mixed content (only the dry run's per-message column uses it)
# chars per token of each generator below, measured with production's tokenizer (tools/strata_tokenizer.py, 2026-11-06)
CPT = {"prose": 4.96, "code": 3.59, "log": 2.61, "web": 4.27, "sql": 2.96, "grep": 2.35, "json": 3.0}
CACHE_LINE_KEYS = ("conversation cache", "STRATA_SLOT_PARK", "memory guard", "skip parking", "restor", "evict")

# ------------------------------------------------------------------------------------------- deterministic content
NOUNS = ("scheduler latency budget request queue worker shard replica index snapshot checkpoint migration rollback "
         "deployment pipeline artifact invoice customer contract supplier warehouse forecast report dashboard metric "
         "alert incident ticket release branch review fixture dataset model prompt token cache memory thread "
         "session account payment order shipment region cluster node volume backup policy quota").split()
VERBS = ("reduces increases blocks delays replaces validates duplicates splits merges schedules retries drops "
         "rewrites caches stores reads writes reports flags exports imports reconciles").split()
ADJS = ("stale partial nightly regional critical optional legacy upstream downstream cold warm idle pending "
        "failed archived encrypted sharded batched incremental").split()
NAMES = ("Berger Okafor Lindqvist Moreau Tanaka Novak Haddad Ferreira Kowalski Brennan Iyer Schulz").split()
IDENTS = ("buffer offset length handler config parser retry timeout payload header record field schema batch "
          "slot window cursor result error state limit count total item entry key value node edge").split()
FILES = ("src/scheduler/queue.py src/billing/invoice.py src/api/routes.py src/core/cache.py docs/runbook.md "
         "src/etl/loader.py config/deploy.yaml src/models/order.py tests/test_cache.py src/workers/export.py").split()

TOOLS = [
    {"type": "function", "function": {"name": n, "description": d, "parameters": {
        "type": "object", "properties": {k: {"type": t, "description": kd} for k, t, kd in props},
        "required": [props[0][0]]}}}
    for n, d, props in [
        ("read_file", "Read a text file of the workspace. Returns at most 2000 lines; use offset for more.",
         [("path", "string", "path relative to the workspace root"), ("offset", "integer", "first line, 1-based")]),
        ("write_file", "Create or overwrite a file with the given content.",
         [("path", "string", "target path"), ("content", "string", "the full new content")]),
        ("edit_file", "Replace one exact occurrence of a string in a file.",
         [("path", "string", "file"), ("old", "string", "exact text to replace"), ("new", "string", "replacement")]),
        ("search", "Search the workspace for a regular expression; returns file:line:text matches.",
         [("pattern", "string", "regular expression"), ("glob", "string", "optional file filter")]),
        ("run_shell", "Run a shell command in the workspace and return stdout and stderr (timeout 120 s).",
         [("command", "string", "the command line")]),
        ("web_search", "Search the web and return the top results with title, url and snippet.",
         [("query", "string", "search terms"), ("count", "integer", "number of results")]),
        ("fetch_url", "Download a web page and return its readable text.", [("url", "string", "absolute URL")]),
        ("query_db", "Run a read-only SQL query against the analytics database.",
         [("sql", "string", "a SELECT statement"), ("limit", "integer", "row limit")]),
        ("send_email", "Draft an email for the user to review (never sends by itself).",
         [("to", "string", "recipient"), ("subject", "string", "subject"), ("body", "string", "text")]),
        ("todo_write", "Replace the task list shown to the user.",
         [("items", "string", "one task per line, prefixed with [ ] or [x]")]),
    ]]


def est(s, cpt=CHARS_PER_TOKEN):
    return int(len(s) / cpt) + 1


def sentence(r):
    return (f"The {r.choice(ADJS)} {r.choice(NOUNS)} {r.choice(VERBS)} the {r.choice(NOUNS)} of the "
            f"{r.choice(ADJS)} {r.choice(NOUNS)}" + r.choice((".", ", which is why we keep it.", " every night.",
                                                              f" since build {r.randint(100, 9999)}.",
                                                              f" (see {r.choice(FILES)}).")))


def prose(r, n_tok):
    out, n = [], 0
    while n < n_tok:
        para = " ".join(sentence(r) for _ in range(r.randint(3, 7)))
        out.append(para)
        n += est(para, CPT["prose"]) + 1
    return "\n\n".join(out)


def code(r, n_tok):
    out, n = [], 0
    while n < n_tok:
        name = "_".join(r.sample(IDENTS, 2))
        args = ", ".join(r.sample(IDENTS, r.randint(1, 3)))
        body = [f"def {name}({args}):", f'    """{sentence(r)}"""']
        for _ in range(r.randint(3, 9)):
            a, b = r.sample(IDENTS, 2)
            body.append(r.choice((f"    {a} = {b}.get({r.randint(0, 64)}, {a!r})",
                                  f"    if {a} > {r.randint(1, 4096)}:\n        raise ValueError(f\"{a} {{{b}}}\")",
                                  f"    for {a} in {b}[{r.randint(0, 8)}:]:\n        total += len({a})",
                                  f"    {a}.append({b})  # {r.choice(NOUNS)}")))
        body.append(f"    return {r.choice(IDENTS)}\n")
        blk = "\n".join(body)
        out.append(blk)
        n += est(blk, CPT["code"])
    return "\n".join(out)


def logtext(r, n_tok):
    out, n, t = [], 0, 1793900000 + r.randint(0, 86400)
    while n < n_tok:
        t += r.randint(0, 9)
        ln = (f"{datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} "
              f"{r.choice(('INFO', 'INFO', 'INFO', 'WARN', 'ERROR', 'DEBUG'))} {r.choice(IDENTS)}.{r.choice(IDENTS)} "
              f"req={r.getrandbits(32):08x} {r.choice(NOUNS)}={r.randint(0, 99999)} {sentence(r)}")
        out.append(ln)
        n += est(ln, CPT["log"]) + 1
    return "\n".join(out)


def search_results(r, n_tok):
    out, n = [], 0
    while n < n_tok:
        x = {"title": f"{r.choice(ADJS).title()} {r.choice(NOUNS)} {r.choice(NOUNS)} guide",
             "url": f"https://docs.example.org/{r.choice(NOUNS)}/{r.getrandbits(24):06x}", "snippet": prose(r, 60)}
        out.append(x)
        n += est(json.dumps(x), CPT["web"]) + 2
    return json.dumps(out, indent=1)


def tool_result(r, n_tok):
    """(tool name, arguments, result text) of about n_tok tokens."""
    kind = r.choice(("read", "read", "search", "shell", "web", "sql"))
    if kind == "read":
        f = r.choice(FILES)
        return "read_file", {"path": f}, f"# {f}\n" + code(r, n_tok)
    if kind == "search":
        p = r.choice(IDENTS)
        lines, n = [], 0
        while n < n_tok:
            lines.append(f"{r.choice(FILES)}:{r.randint(1, 900)}:    {p} = {r.choice(IDENTS)}({r.choice(IDENTS)})")
            n += est(lines[-1], CPT["grep"]) + 1
        return "search", {"pattern": p}, "\n".join(lines)
    if kind == "shell":
        return "run_shell", {"command": f"journalctl -u {r.choice(NOUNS)} --since today"}, logtext(r, n_tok)
    if kind == "web":
        return "web_search", {"query": f"{r.choice(ADJS)} {r.choice(NOUNS)} {r.choice(NOUNS)}"}, search_results(r, n_tok)
    rows, n = [], 0
    while n < n_tok:
        rows.append({"id": len(rows), r.choice(IDENTS): r.randint(0, 10 ** 6), "name": r.choice(NAMES),
                     "note": sentence(r)})
        n += est(json.dumps(rows[-1]), CPT["sql"]) + 1
    return "query_db", {"sql": f"SELECT * FROM {r.choice(NOUNS)}s WHERE {r.choice(IDENTS)} > {r.randint(1, 99)}"}, \
        json.dumps(rows)


def system_prompt(r):
    rules = "\n".join(f"{i + 1}. {sentence(r)}" for i in range(40))
    return ("You are an autonomous assistant for a small company: you research, write code, analyse data and draft "
            "correspondence for the user, using the tools below. Work step by step, read before you change anything, "
            "cite the files and sources you used, and ask before any irreversible action.\n\nHouse rules:\n" + rules +
            f"\n\nToday is 2026-11-06. Workspace: /srv/work/{r.choice(NOUNS)}-{r.randint(10, 99)}.")


def conversation(seed, n_tok):
    """(messages, tools) of an agent conversation of about n_tok template tokens, ending in a user question."""
    r = random.Random(seed)
    msgs = [{"role": "system", "content": system_prompt(random.Random(1))}]   # one system prompt for every A
    # what is added is counted as asked (each generator meets its size in real tokens, see CPT); the system prompt,
    # the tool list and the template's tool preamble as measured (~1.9k tokens with production's template)
    n = est(msgs[0]["content"], CPT["prose"]) + est(json.dumps(TOOLS), CPT["json"]) + 20
    k, call = 0, 0
    while n < n_tok - 400:
        k += 1
        q, extra = r.randint(40, 220), (r.randint(800, 5000) if r.random() < 0.3 else 0)
        user = (f"[{r.choice(NAMES)}] " + prose(r, q) +
                (("\n\nHere is the material:\n" + r.choice((code, logtext, prose))(r, extra)) if extra else ""))
        msgs.append({"role": "user", "content": user})
        n += q + extra + 10
        for _ in range(r.randint(0, 3)):                       # tool rounds of this turn
            say, size = r.randint(10, 50), min(max(300, n_tok - n - 600), r.randint(600, 9000))
            name, args, res = tool_result(r, size)
            call += 1
            cid = f"call_{seed % 100000:05d}_{call:04d}"
            msgs.append({"role": "assistant", "content": prose(r, say),
                         "tool_calls": [{"id": cid, "type": "function",
                                         "function": {"name": name, "arguments": json.dumps(args)}}]})
            msgs.append({"role": "tool", "tool_call_id": cid, "content": res})
            n += say + est(json.dumps(args), CPT["json"]) + size + 40
            if n >= n_tok - 400:
                break
        ans = r.randint(120, 700)
        msgs.append({"role": "assistant", "content": prose(r, ans)})
        n += ans + 10
    msgs.append({"role": "user", "content": f"[{r.choice(NAMES)}] Given everything above, what are the three most "
                                            f"important open problems, and what would you do first? " + prose(r, 60)})
    return msgs, TOOLS


def short_other(seed, n_tok):
    """B: an unrelated short chat (another client: no tools, its own system prompt), ~n_tok tokens."""
    r = random.Random(seed)
    return [{"role": "system", "content": "You are a helpful assistant. Answer briefly."},
            {"role": "user", "content": "Summarise this email in three bullet points:\n\n" + prose(r, n_tok - 90)}], None


def rerender(msg):
    """The reply as another client would send it back: no reasoning, whitespace normalised, arguments re-ordered."""
    out = {"role": "assistant", "content": " ".join((msg.get("content") or "").split())}
    if msg.get("tool_calls"):
        calls = []
        for c in msg["tool_calls"]:
            fn = dict(c.get("function") or {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
                fn["arguments"] = json.dumps(dict(sorted(args.items(), reverse=True)), separators=(",", ":"))
            except (ValueError, AttributeError):
                pass
            calls.append({**c, "function": fn})
        out["tool_calls"] = calls
    return out


# --------------------------------------------------------------------------------------------------- the client
def port_of(url):
    u = urllib.parse.urlparse(url)
    return u.port or (443 if u.scheme == "https" else 80)


def get_json(a, path, timeout=10):
    try:
        with urllib.request.urlopen(a.url.rstrip("/") + path, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:                               # noqa: BLE001 - a missing endpoint is "unknown", not a crash
        return None


def chat(a, messages, tools, max_tok, effort):
    """One streamed request -> the row (timings, usage) and the assistant message as returned."""
    body = {"model": a.model, "messages": messages, "max_tokens": max_tok, "temperature": a.temperature,
            "stream": True, "stream_options": {"include_usage": True}}
    if tools:
        body["tools"] = tools
    if effort:
        body["reasoning_effort"] = effort
    headers = {"Content-Type": "application/json"}
    if a.key:
        headers["Authorization"] = f"Bearer {a.key}"
    req = urllib.request.Request(a.url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers=headers)
    t_wall, t0 = time.time(), time.perf_counter()
    t_first, content, reasoning, calls, usage, err = None, [], [], {}, {}, None
    try:
        with urllib.request.urlopen(req, timeout=a.timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                c = json.loads(line[5:])
                if c.get("usage"):
                    usage = c["usage"]
                for ch in c.get("choices") or []:
                    d = ch.get("delta") or {}
                    got = False
                    if d.get("content"):
                        content.append(d["content"]); got = True
                    if d.get("reasoning_content"):
                        reasoning.append(d["reasoning_content"]); got = True
                    for tc in d.get("tool_calls") or []:
                        x = calls.setdefault(tc.get("index", 0), {"id": "", "type": "function",
                                                                  "function": {"name": "", "arguments": ""}})
                        x["id"] = tc.get("id") or x["id"]
                        fn = tc.get("function") or {}
                        x["function"]["name"] += fn.get("name") or ""
                        x["function"]["arguments"] += fn.get("arguments") or ""
                        got = True
                    if got and t_first is None:
                        t_first = time.perf_counter()
    except Exception as e:                           # noqa: BLE001 - a failed request is a result
        err = f"{type(e).__name__}: {e}"
    t_end = time.perf_counter()
    msg = {"role": "assistant", "content": "".join(content)}
    if reasoning:
        msg["reasoning_content"] = "".join(reasoning)
    if calls:
        msg["tool_calls"] = [calls[i] for i in sorted(calls)]
    row = {"t_send": round(t_wall, 3), "wall_s": round(t_end - t0, 3),
           "ttft_s": round((t_first or t_end) - t0, 3), "prompt_tokens": usage.get("prompt_tokens", 0),
           "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
           "completion_tokens": usage.get("completion_tokens", 0), "err": err}
    return row, msg


def metrics_row(m, t_send):
    """The /metrics request entry of the request sent at t_send (the newest one that ended after it)."""
    for x in (m or {}).get("requests") or []:                  # newest first
        if (x.get("time") or 0) >= t_send:
            return x
    return None


def held(m):
    return [s.get("held_tokens") for s in ((m or {}).get("live") or {}).get("slots") or []]


def cache_stats(m):
    c = (m or {}).get("conversation_cache") or {}
    return {k: c.get(k) for k in ("parked", "bytes", "parks", "evictions", "restores")}


def wait_idle(a, settle):
    """Until the server runs nothing and the conversation cache has not changed for `settle` s (a park's capture)."""
    t_end, last, since = time.time() + 600, None, time.time()
    while time.time() < t_end:
        m = get_json(a, "/metrics")
        live = (m or {}).get("live") or {}
        busy = (live.get("running") or 0) + (live.get("queued") or 0)
        key = json.dumps(cache_stats(m), sort_keys=True)
        if busy or key != last:
            last, since = key, time.time()
        elif time.time() - since >= settle:
            return m
        time.sleep(0.5)
    print("  ! the server did not become idle within 600 s", flush=True)
    return get_json(a, "/metrics")


class EngineLog:
    def __init__(self, path):
        self.path = (Path(path) if Path(path).is_absolute() else HERE / path) if path else None
        self.pos = self.size()

    def size(self):
        try:
            return self.path.stat().st_size if self.path else 0
        except OSError:
            return 0

    def new_lines(self):
        """The engine's cache lines written since the last call (the test engine's own log only)."""
        if not self.path:
            return []
        end = self.size()
        if end < self.pos:
            self.pos = 0
        try:
            with open(self.path, "rb") as f:
                f.seek(self.pos)
                data = f.read(min(end - self.pos, 8 << 20))
        except OSError:
            return []
        self.pos = end
        return [ln for ln in data.decode("utf-8", "replace").splitlines() if any(k in ln for k in CACHE_LINE_KEYS)][-30:]


# ------------------------------------------------------------------------------------------------ RAM pressure
def run(cmd, check=True):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if check and p.returncode:
        raise RuntimeError(f"{shlex.join(cmd)}: {p.stderr.strip() or p.returncode}")
    return p.stdout.strip()


class Pressure:
    """--pressure cgroup: the test scope's MemoryMax set to (usage - reclaimable cache) + --avail-mib, as the
    engine's guard counts it (conversation_memory.cpp combine(): limit - (memory.current - inactive_file)), and put
    back afterwards.  Never production's unit."""

    def __init__(self, a):
        self.a, self.unit, self.saved = a, a.unit, None
        if self.unit in ("strata.service", "strata"):
            sys.exit("refusing: --pressure never touches production's strata.service")
        cg = run(["systemctl", "--user", "show", "-p", "ControlGroup", "--value", self.unit])
        if not cg:
            sys.exit(f"--pressure cgroup: unit {self.unit} has no cgroup (is the test server running in it?)")
        if cg.rstrip("/").endswith("/strata.service"):
            sys.exit("refusing: that is production's cgroup")
        self.dir = Path("/sys/fs/cgroup" + cg)

    def used_net(self):
        cur = int((self.dir / "memory.current").read_text())
        stat = dict(ln.split() for ln in (self.dir / "memory.stat").read_text().splitlines())
        credit = max(0, min(int(stat.get("inactive_file", 0)), cur) - int(stat.get("file_dirty", 0))
                     - int(stat.get("file_writeback", 0)))
        return cur - credit

    def available_mib(self):
        try:
            lim = (self.dir / "memory.max").read_text().strip()
            return None if lim == "max" else (int(lim) - self.used_net()) >> 20
        except OSError:
            return None

    def apply(self):
        if self.saved is None:
            self.saved = (self.dir / "memory.max").read_text().strip()
        target = self.used_net() + (self.a.avail_mib << 20)
        run(["systemctl", "--user", "set-property", "--runtime", self.unit, f"MemoryMax={target}"])
        return self.available_mib()

    def release(self):
        if self.saved is not None:
            run(["systemctl", "--user", "set-property", "--runtime", self.unit,
                 f"MemoryMax={'infinity' if self.saved == 'max' else self.saved}"], check=False)
            self.saved = None


# ---------------------------------------------------------------------------------------------- the scenarios
SCENARIOS = ("control", "interleave", "rerender", "effort")


def scenario(a, name, size, idx, log, pressure, rows_out):
    seed = a.seed * 1000003 + size * 7 + idx * 101
    rows, ctx = [], {"label": a.label, "scenario": name, "size": size, "seed": seed}

    def step(role, messages, tools, max_tok, effort):
        m0 = wait_idle(a, a.settle)
        row, msg = chat(a, messages, tools, max_tok, effort)
        m1 = wait_idle(a, a.settle)
        x = metrics_row(m1, row["t_send"]) or {}
        c0, c1 = cache_stats(m0), cache_stats(m1)
        row.update(ctx, role=role, effort=effort, est_tokens=sum(est(json.dumps(mm)) for mm in messages),
                   reused=x.get("reused"), prompt_read=x.get("prompt_read"), prompt_ms=x.get("prompt_ms"),
                   server_prompt_tokens=x.get("prompt_tokens"), held_after=held(m1),
                   cache_delta={k: (c1[k] - c0[k]) if isinstance(c1.get(k), int) and isinstance(c0.get(k), int)
                                else None for k in c1},
                   parked_after=c1.get("parked"), avail_mib=pressure.available_mib() if pressure else None,
                   foreign=((m1 or {}).get("totals", {}).get("requests", 0) -
                            (m0 or {}).get("totals", {}).get("requests", 0)) - 1,
                   log=log.new_lines(), ts=datetime.datetime.now().astimezone().isoformat(timespec="seconds"))
        rows.append(row)
        if a.verbose or role in ("A", "B", "A2"):
            print(f"  {role:7s} prompt {row['prompt_tokens']:7d} reused {row['reused'] if row['reused'] is not None else '?':>7} "
                  f"ttft {row['ttft_s']:7.2f}s wall {row['wall_s']:7.2f}s parks {row['cache_delta'].get('parks')} "
                  f"evict {row['cache_delta'].get('evictions')} restores {row['cache_delta'].get('restores')}"
                  f"{'  ERR ' + row['err'] if row['err'] else ''}{'  ! foreign requests' if row['foreign'] > 0 else ''}",
                  flush=True)
            for ln in row["log"]:
                if "skip parking" in ln or "evicted" in ln or "restor" in ln:
                    print(f"          | {ln[:220]}", flush=True)
        return row, msg

    print(f"# {a.label}: {name} @ {size} tokens (seed {seed})", flush=True)
    for f in range(a.fillers):                                    # parked conversations for the gate to evict
        fm, ft = conversation(seed + 17 + f, a.filler_tokens)
        step(f"filler{f}", fm, ft, 16, a.effort)
    msgs, tools = conversation(seed, size)
    ra, reply = step("A", msgs, tools, a.reply_tokens, a.effort)
    a_slot = None
    if ra["held_after"]:
        cands = [(abs((h or 0) - ra["prompt_tokens"] - ra["completion_tokens"]), i)
                 for i, h in enumerate(ra["held_after"]) if (h or 0) >= ra["prompt_tokens"] > 0]
        a_slot = min(cands)[1] if cands else None
    avail_b = None
    try:
        if pressure:
            avail_b = pressure.apply()
            print(f"  pressure: test scope MemoryMax lowered, guard-style available {avail_b} MiB "
                  f"(asked {a.avail_mib}; the server's floor {a.floor_mib})", flush=True)
        b_took = None
        if name != "control":
            for k in range(a.occupy):                             # non-pipelined configs only (see SLOT FORCING)
                om, _ = short_other(seed + 900 + k, 120)
                step(f"occupy{k}", om, None, 8, a.effort)
            bm, bt = short_other(seed + 1, a.b_tokens)
            rb, _ = step("B", bm, bt, a.b_reply_tokens, a.effort)
            if a_slot is not None and rb["held_after"] and a_slot < len(rb["held_after"]):
                b_took = (rb["held_after"][a_slot] or 0) < ra["prompt_tokens"]
            print(f"  B in A's slot ({a_slot}): {'yes' if b_took else 'NO - nothing was parked' if b_took is False else '?'}",
                  flush=True)
        back = rerender(reply) if name == "rerender" else reply
        if not back.get("content") and not back.get("tool_calls"):
            back["content"] = "(no answer)"
        a2 = msgs + [back, {"role": "user", "content": "Thanks. Now do the first of those and show me the result. " +
                            prose(random.Random(seed + 2), a.a2_tokens)}]
        r2, _ = step("A2", a2, tools, a.a2_reply_tokens, a.a2_effort if name == "effort" else a.effort)
    finally:
        if pressure:
            pressure.release()
    ratio = (r2["reused"] or 0) / r2["prompt_tokens"] if r2["prompt_tokens"] else 0.0
    summ = {**ctx, "kind": "scenario", "a_slot": a_slot, "b_in_a_slot": b_took, "avail_mib_before_b": avail_b,
            "a_prompt": ra["prompt_tokens"], "a_ttft_s": ra["ttft_s"], "a2_prompt": r2["prompt_tokens"],
            "a2_reused": r2["reused"], "a2_cached": r2["cached_tokens"], "a2_ttft_s": r2["ttft_s"],
            "a2_wall_s": r2["wall_s"], "a2_reuse": round(ratio, 4), "hit": ratio >= 0.9,
            "refusals": sum(1 for row in rows for ln in row["log"] if "physical RAM admission" in ln),
            "args": {k: v for k, v in vars(a).items() if k not in ("key",)}}
    rows_out.extend(rows)
    return summ


def table(summaries, title):
    print(f"\n{title}")
    print(f"{'label':16s} {'scenario':10s} {'size':>7s} {'B@A':>4s} {'avail':>6s} {'A ttft':>7s} {'A2 prompt':>9s} "
          f"{'reused':>7s} {'reuse%':>6s} {'A2 ttft':>8s} {'refuse':>6s} hit")
    for s in summaries:
        b = {True: "yes", False: "no", None: "-"}[s.get("b_in_a_slot")]
        print(f"{s['label'][:16]:16s} {s['scenario']:10s} {s['size']:>7d} {b:>4s} {s.get('avail_mib_before_b') or '-':>6} "
              f"{s['a_ttft_s']:>7.2f} {s['a2_prompt']:>9d} {s['a2_reused'] if s['a2_reused'] is not None else '?':>7} "
              f"{100 * s['a2_reuse']:>5.1f}% {s['a2_ttft_s']:>8.2f} {s['refusals']:>6d} {'HIT' if s['hit'] else 'MISS'}")


def report(path):
    rows = []
    try:
        for ln in open(path, encoding="utf-8"):
            x = json.loads(ln)
            if x.get("kind") == "scenario":
                rows.append(x)
    except OSError:
        sys.exit(f"no results in {path}")
    rows.sort(key=lambda s: (s["size"], s["scenario"], s.get("ts", ""), s["label"]))
    table(rows, f"# before/after: every scenario row in {path} (label = arm)")


# ------------------------------------------------------------------------------------------- config and launch
def write_config(a):
    c = json.load(open(PROD_CONFIG))
    args, env = c["args"], c.setdefault("env", {})
    c["log"] = str(RESULTS / "engine-repro_park_ram.log")
    env["STRATA_CONV_TIER_DIR"] = BENCH_TIER
    if a.exe:
        c["exe"] = a.exe
    floor = a.fix_floor_mib if a.fix_floor_mib is not None else (4096 if a.arm == "floor" else None)
    if floor is not None:
        if "--conversation-cache-min-free-mib" in args:
            args[args.index("--conversation-cache-min-free-mib") + 1] = str(floor)
        else:
            args += ["--conversation-cache-min-free-mib", str(floor)]
    if a.arm == "malloc":
        # PROPOSAL, unverified: glibc trims a non-main arena heap only when its free top exceeds M_TRIM_THRESHOLD, and
        # a heap is at most 64 MiB (HEAP_MAX on 64-bit), so production's 64 MiB threshold means arena heaps never
        # shrink; 4 MiB lets them.  The mmap threshold stays at 16 MiB.
        env["MALLOC_TRIM_THRESHOLD_"] = "4194304"
    for kv in a.fix_env:
        k, _, v = kv.partition("=")
        if v == "" and kv.endswith("="):
            env.pop(k, None)
        else:
            env[k] = v
    c["repro_arm"] = {"arm": a.arm, "floor_mib": floor, "fix_env": a.fix_env, "from": str(PROD_CONFIG.resolve())}
    Path(a.write_config).parent.mkdir(parents=True, exist_ok=True)
    json.dump(c, open(a.write_config, "w"), indent=1)
    print(f"wrote {a.write_config}: arm {a.arm}, engine {c['exe']}, log {c['log']}, tier {BENCH_TIER}, floor "
          f"{floor if floor is not None else 'as production'}, env changes "
          f"{[k for k in env if k.startswith('MALLOC')]} + {a.fix_env}")


def launch(a):
    if run(["systemctl", "--user", "is-active", "strata.service"], check=False) == "active":
        sys.exit("refusing to launch: production (strata.service) holds the GPUs; stop it in a GPU window first")
    if get_json(a, "/health", timeout=3) is not None:
        sys.exit(f"refusing to launch: something already answers on {a.url}")
    cfg = str(Path(a.launch).resolve())
    RESULTS.mkdir(exist_ok=True)
    run(["systemctl", "--user", "stop", a.unit], check=False)
    run(["systemctl", "--user", "reset-failed", a.unit], check=False)
    u = urllib.parse.urlparse(a.url)
    cmd = ["systemd-run", "--user", "--scope", f"--unit={a.unit.removesuffix('.scope')}", "-p", f"MemoryMax={a.memmax}",
           "-p", "MemorySwapMax=0", str(VENV_PY), str(SERVER_DIR / "serve" / "server.py"), "--engine", "strata",
           "--config", cfg, "--port", str(port_of(a.url)), "--host", u.hostname or "127.0.0.1"]
    print("launch:", shlex.join(cmd), flush=True)
    out = open(RESULTS / "server-repro_park_ram.out", "ab")
    subprocess.Popen(cmd, cwd=SERVER_DIR, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                     start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < 1200:
        h = get_json(a, "/health", timeout=3)
        if h and h.get("loaded"):
            print(f"test server loaded after {time.time() - t0:.0f} s", flush=True)
            return cfg
        time.sleep(5)
    run(["systemctl", "--user", "stop", a.unit], check=False)
    sys.exit("test server did not load within 1200 s (see results/server-repro_park_ram.out)")


def config_floor(path):
    try:
        args = json.load(open(path))["args"]
        return int(args[args.index("--conversation-cache-min-free-mib") + 1])
    except (OSError, ValueError, KeyError, IndexError):
        return None


# --------------------------------------------------------------------------------------------------- dry run
def dry_run(a):
    tmpl = tok = None
    if a.exact:
        sys.path[:0] = [str(SERVER_DIR / "serve"), str(SERVER_DIR / "tools")]
        import frontend, strata_tokenizer as ST            # noqa: E401 - needs jinja2 and regex (Strata's venv)
        tdir = Path(json.load(open(PROD_CONFIG))["tokenizer"])
        vocab = json.loads((tdir / "vocab.json").read_text(encoding="utf-8"))
        tokens = [None] * len(vocab)
        for t, i in vocab.items():
            tokens[i] = t
        tok = ST.Tokenizer(tokens, (tdir / "merges.txt").read_text(encoding="utf-8").split("\n"),
                           json.loads((tdir / "token_type.json").read_text()))
        tmpl = frontend.ChatTemplate(tdir / "chat_template.jinja")

    def ids(messages, tools, effort):
        m, t, kw = frontend.openai_to_messages({"messages": messages, "tools": tools, "reasoning_effort": effort})
        return tok.encode(tmpl.render(m, t, **kw), parse_special=True)

    def count(messages, tools, effort):
        return len(ids(messages, tools, effort)) if tmpl else None

    def chars(messages, tools):
        return sum(est(json.dumps(m)) for m in messages) + (est(json.dumps(tools)) if tools else 0)

    print(f"# dry run: sizes {a.sizes}, scenarios {a.scenarios}, {a.fillers} fillers x {a.filler_tokens}, "
          f"B {a.b_tokens}, A2 + {a.a2_tokens}; 'chars' = JSON chars / {CHARS_PER_TOKEN}"
          + (", 'exact' = production's chat template + tokenizer (Strata-143-prod)" if tmpl else
             " (--exact with Strata's venv python counts the real tokens)"))
    print(f"{'what':26s} {'msgs':>5s} {'asked':>7s} {'chars':>7s} {'exact':>7s}")
    total = 0
    for size in a.sizes:
        for idx, name in enumerate(a.scenarios):
            seed = a.seed * 1000003 + size * 7 + idx * 101
            msgs, tools = conversation(seed, size)
            x = count(msgs, tools, a.effort)
            total += 2 * (x or size) + a.fillers * a.filler_tokens     # A, A2 (if it misses) and the fillers
            print(f"{f'A {name} @{size}':26s} {len(msgs):>5d} {size:>7d} {chars(msgs, tools):>7d} "
                  f"{x if x is not None else '-':>7}")
            if idx == 0 and tmpl:     # how much of A's prompt each A2 variant keeps (a reply of the usual shape)
                fake = {"role": "assistant", "content": prose(random.Random(5), a.reply_tokens),
                        "reasoning_content": prose(random.Random(6), 80),
                        "tool_calls": [{"id": "c", "type": "function",
                                        "function": {"name": "search", "arguments": '{"pattern": "x", "glob": "*.py"}'}}]}
                base = ids(msgs, tools, a.effort)
                for v, back, eff in (("exact", fake, a.effort), ("rerender", rerender(fake), a.effort),
                                     ("effort", fake, a.a2_effort)):
                    a2 = msgs + [back, {"role": "user", "content": "Thanks. " + prose(random.Random(seed + 2), a.a2_tokens)}]
                    full = ids(a2, tools, eff)
                    common = next((i for i, (p, q) in enumerate(zip(full, base)) if p != q), min(len(full), len(base)))
                    print(f"{'  A2 ' + v:26s} {len(a2):>5d} {'':>7s} {'':>7s} {len(full):>7d}  first {common} tokens "
                          f"= A's prompt ({len(base)})")
            if idx == 0:
                bm, _ = short_other(seed + 1, a.b_tokens)
                fm, ft = conversation(seed + 17, a.filler_tokens)
                print(f"{'B':26s} {len(bm):>5d} {a.b_tokens:>7d} {chars(bm, None):>7d} {count(bm, None, a.effort) or '-':>7}")
                print(f"{'filler (1 of ' + str(a.fillers) + ')':26s} {len(fm):>5d} {a.filler_tokens:>7d} "
                      f"{chars(fm, ft):>7d} {count(fm, ft, a.effort) or '-':>7}")
    print(f"prompt tokens to read at most (every A2 a miss): {total} - at ~5-6k tok/s prefill about "
          f"{total / 5500 / 60:.0f} min per arm")


# ------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8091", help="test server (default: the test port 8091)")
    ap.add_argument("--allow-prod", action="store_true", help="allow port 8090 / 8000 (production, the G3 proxy)")
    ap.add_argument("--model", default="strata")
    ap.add_argument("--key", default="")
    ap.add_argument("--label", default="repro", help="the arm's name in the JSONL and the table (before, after-...)")
    ap.add_argument("--sizes", type=lambda s: [int(x) for x in s.split(",")], default=[136000, 30000],
                    help="A's sizes in tokens, comma-separated")
    ap.add_argument("--scenarios", type=lambda s: [x for x in s.split(",") if x], default=list(SCENARIOS),
                    help=f"comma-separated subset of {','.join(SCENARIOS)}")
    ap.add_argument("--b-tokens", type=int, default=900)
    ap.add_argument("--a2-tokens", type=int, default=300, help="new user message tokens in A2")
    ap.add_argument("--reply-tokens", type=int, default=200, help="max_tokens of A (its reply goes into A2)")
    ap.add_argument("--b-reply-tokens", type=int, default=120)
    ap.add_argument("--a2-reply-tokens", type=int, default=32)
    ap.add_argument("--effort", default="none", help="reasoning_effort of every request ('' = the server's default)")
    ap.add_argument("--a2-effort", default="low", help="A2's reasoning_effort in the 'effort' scenario")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fillers", type=int, default=16, help="parked conversations before A (0 = none)")
    ap.add_argument("--filler-tokens", type=int, default=6000,
                    help="each filler's size: >= --conversation-cache-min-tokens (1024), < STRATA_CONV_TIER_MIN_TOKENS "
                         "(8192) so an evicted filler is dropped, not spilled to the SSD tier")
    ap.add_argument("--occupy", type=int, default=0, help="tiny requests after A (non-pipelined configs: --batch - 1)")
    ap.add_argument("--settle", type=float, default=3.0, help="idle seconds (and no cache change) before each request")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--pressure", choices=("none", "cgroup"), default="none",
                    help="cgroup: lower the test scope's MemoryMax to usage + --avail-mib between A and A2")
    ap.add_argument("--unit", default="strata-bench.scope", help="the test server's systemd unit (--pressure, --launch)")
    ap.add_argument("--avail-mib", type=int, default=18102,
                    help="--pressure: the RAM the engine's guard sees as available at B (production's log: 18102)")
    ap.add_argument("--floor-mib", type=int, default=None,
                    help="the server's --conversation-cache-min-free-mib, for the output only (default: the config's)")
    ap.add_argument("--engine-log", default="", help="the test engine's log (default: the launched config's 'log')")
    ap.add_argument("--write-config", default="", help="write a test config derived from configs/production.json")
    ap.add_argument("--arm", choices=("prod", "floor", "malloc", "custom"), default="prod",
                    help="--write-config: prod = as production; floor = --conversation-cache-min-free-mib 4096 (or "
                         "--fix-floor-mib); malloc = MALLOC_TRIM_THRESHOLD_=4194304; custom = only --fix-*")
    ap.add_argument("--fix-env", action="append", default=[], help="--write-config: KEY=VALUE (KEY= removes it)")
    ap.add_argument("--fix-floor-mib", type=int, default=None)
    ap.add_argument("--exe", default="", help="--write-config: another engine binary")
    ap.add_argument("--launch", default="", help="start a test server with this config in --unit (production stopped)")
    ap.add_argument("--memmax", default="224G", help="--launch: the scope's MemoryMax (production's: 224G)")
    ap.add_argument("--keep", action="store_true", help="--launch: leave the test server running afterwards")
    ap.add_argument("--out", default=str(OUT_JSONL))
    ap.add_argument("--dry-run", action="store_true", help="build the prompts and print token estimates; no server")
    ap.add_argument("--exact", action="store_true", help="--dry-run: count with production's template + tokenizer")
    ap.add_argument("--report", action="store_true", help="print the before/after table from --out and exit")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    bad = [s for s in a.scenarios if s not in SCENARIOS]
    if bad:
        ap.error(f"unknown scenarios {bad}")

    if a.report:
        return report(a.out)
    if a.write_config:
        return write_config(a)
    if a.dry_run:
        return dry_run(a)
    if port_of(a.url) in PROD_PORTS and not a.allow_prod:
        ap.error(f"{a.url} is production's port; use the test port 8091 (or --allow-prod)")
    if a.allow_prod and (a.pressure != "none" or a.launch):
        ap.error("--pressure and --launch are for the test server only")
    cfg = launch(a) if a.launch else None
    if a.floor_mib is None:
        a.floor_mib = (config_floor(cfg) if cfg else None) or 16384
    if not a.engine_log and cfg:
        a.engine_log = json.load(open(cfg)).get("log", "")
    try:
        if get_json(a, "/health") is None:
            sys.exit(f"no server on {a.url}")
        pressure = Pressure(a) if a.pressure == "cgroup" else None
        log = EngineLog(a.engine_log)
        m = get_json(a, "/metrics") or {}
        print(f"# {a.label}: {a.url}, engine {((m.get('engine') or {}).get('version'))}, slots "
              f"{(m.get('live') or {}).get('parallel')}, pressure {a.pressure} (available {a.avail_mib} MiB, floor "
              f"{a.floor_mib} MiB), engine log {a.engine_log or '-'}", flush=True)
        rows, summaries = [], []
        for size in a.sizes:
            for idx, name in enumerate(a.scenarios):
                s = scenario(a, name, size, idx, log, pressure, rows)
                s["ts"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
                summaries.append(s)
                Path(a.out).parent.mkdir(parents=True, exist_ok=True)
                with open(a.out, "a", encoding="utf-8") as f:     # per scenario, so an abort keeps what ran
                    for r in rows:
                        f.write(json.dumps({"kind": "request", **r}) + "\n")
                    f.write(json.dumps(s) + "\n")
                rows.clear()
        table(summaries, f"# {a.label}: A2 after A (and B) - HIT = >= 90 % of A2's prompt reused")
    finally:
        if a.launch and not a.keep:
            run(["systemctl", "--user", "stop", a.unit], check=False)
            print(f"stopped {a.unit}", flush=True)


if __name__ == "__main__":
    main()
