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
  checkbranch (not in the default set; --scenarios checkbranch) - the Genesis agent's turn pattern (2026-11-06,
             production log: 132219 read -> check call 132214 reused -> "dropped 1 superseded copy of this
             conversation" -> ~17 other requests -> the next turn 134737 tokens = only 8192 reused):
               A       as above (ends with a user message)
               CHECK   A + the reply + a short check question (--check-question, ~236 chars): one live-slot hit
               OTHER   one unrelated request of --other-tokens (default 30000; 0 = none, the control) - on an idle
                       pipelined server it lands in the same slot (SLOT FORCING), so CHECK's slot is parked
               A2      A + the reply + a new user message WITHOUT the check question: it branches off CHECK's
                       snapshot just before the check question.  Recorded: A2's reuse against the ideal
                       (A's prompt + the reply = CHECK's prompt minus the question; --dry-run --exact prints the
                       exact common prefix), every "superseded" drop and the engine's "prompt N tokens = X reused"
                       line of each step.  The question it answers: does the parked snapshot carry (and use) its
                       checkpoints on a branch, or is only the whole snapshot / the shared 8192 root reusable?
             OTHER uses its own system prompt and no tools by default, so it shares nothing with A and cannot take
             A's parked snapshot (the production failure needs a sibling that shares the head): --other-shares-head
             makes OTHER a sub-agent with A's tool list + MESSAGE/STOP (shares A's head up to its tool #N+1), and
             --checkbranch-head genesis gives A the 42-tool Genesis head of 'pingpong' (default small: 10 tools).
  pingpong  (not in the default set; --scenarios pingpong) - production's Genesis main agent / sub-agent ping-pong:
               A       a Genesis-like head: 42 tools rendered first (realistic JSON schemas, ~--head-tokens 13000
                       tokens), then a long system prompt (--pp-system-tokens), then a multi-turn history to ~SIZE
               CHECK   A + the reply + the short check question (~236 chars)
               SUB     a sub-agent of ~--sub-tokens (29500): the SAME first 42 tools, then MESSAGE and STOP, then its own
                       system prompt and history - it shares A's prefix up to the start of tool #43 (> 8192 tokens,
                       below A's next checkpoint); on an idle pipelined server it lands in CHECK's slot (SLOT FORCING)
               A2      A + the reply + a new user message (without the check question): branches off CHECK's snapshot
             then --cycles - 1 more cycles at once: CHECK2 on A2, SUB2 (a sibling of SUB: its history, another final
             task), A3 = A2 + A2's reply + a new message, ...  Recorded: A2/A3 reuse (PASS: >= --pp-target 95 % AND
             more than the --grid point below the branch: at 136k a restore from the 131072 grid checkpoint alone is
             95.2 % of A2 - the percentage alone would pass it; --dry-run --exact prints both),
             SUB/SUB2 reuse, the engine's borrowed / restored / superseded / skip-parking lines per step and the park
             capture times (capture_ms / wait_ms); one '==> pingpong ...' verdict line per scenario.
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
  # the Genesis ping-pong (2 cycles) at 136k, engine log of the test server:
  python3 -I repro_park_ram.py --label P --scenarios pingpong --sizes 136000 --engine-log results/engine-repro.log
  # a server already running on the test port (started by hand in the strata-bench scope):
  python3 -I repro_park_ram.py --label before --pressure cgroup --unit strata-bench.scope \\
      --engine-log results/engine-repro_park_ram.log
"""
import argparse, datetime, hashlib, json, random, re, shlex, subprocess, sys, time, urllib.parse, urllib.request
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
PP_FIT = 1.04          # pingpong: the history generator lands ~4 % short of the asked size with the Genesis head
CPT_TOOLS = 4.1        # json.dumps of a tool list as the template renders it (tojson, compact), measured 2026-11-06
CACHE_LINE_KEYS = ("conversation cache", "STRATA_SLOT_PARK", "memory guard", "skip parking", "restor", "evict",
                   "superseded", "borrow", "strata serve: prompt ")
# checkbranch: the check question (~236 chars, like the Genesis agent's)
CHECK_QUESTION = ("Bezogen auf deine letzte Antwort: ist sie vollständig und korrekt, oder fehlt ein wichtiger Punkt, "
                  "eine Annahme oder ein Risiko, das der Nutzer kennen sollte? Antworte kurz mit JA oder NEIN und, falls "
                  "NEIN, einem einzigen Satz zur Begründung.")

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


# --- pingpong: a Genesis-like head (42 tools first in the system turn, see the chat template), then MESSAGE / STOP
GENESIS_EXTRA = ("calendar_list_events calendar_create_event contacts_lookup crm_get_account crm_update_opportunity "
                 "ticket_create ticket_comment ticket_search git_status git_diff git_commit ci_get_pipeline ci_rerun_job "
                 "deploy_status metrics_query logs_search alert_ack doc_search doc_read doc_write spreadsheet_read "
                 "spreadsheet_update image_describe pdf_extract translate_text summarize_url memory_store memory_recall "
                 "schedule_task list_tasks http_request python_exec").split()          # 32 + TOOLS' 10 = 42
SUB_EXTRA_TOOLS = [
    {"type": "function", "function": {"name": "MESSAGE", "description": "Send an interim report or a question to the "
     "main agent. The main agent reads it after this sub-agent's turn; keep it short and factual.",
     "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "the report"},
                                                     "urgent": {"type": "boolean", "description": "interrupt now"}},
                    "required": ["text"]}}},
    {"type": "function", "function": {"name": "STOP", "description": "End this sub-agent and hand the final result "
     "to the main agent. Call it exactly once, after the task is done or cannot be done.",
     "parameters": {"type": "object", "properties": {"result": {"type": "string", "description": "the final result"},
                                                     "status": {"type": "string", "enum": ["done", "failed", "partial"],
                                                                "description": "how the task ended"}},
                    "required": ["result", "status"]}}}]
_HEAD_CACHE = {}


def genesis_tools(head_tokens):
    """TOOLS' 10 + 32 generated tools with realistic JSON schemas (nested objects, enums, arrays), deterministic; each
    generated tool is grown to an equal share of ~head_tokens (estimated; --dry-run --exact prints the real head)."""
    if head_tokens in _HEAD_CACHE:
        return _HEAD_CACHE[head_tokens]
    budget = max(120, (head_tokens - est(json.dumps([t["function"] for t in TOOLS]), CPT_TOOLS) - 40)
                 // len(GENESIS_EXTRA))
    out = list(TOOLS)
    for i, name in enumerate(GENESIS_EXTRA):
        r = random.Random(4242 + i)
        area = name.split("_")[0]
        props = {}
        for j in range(r.randint(3, 6)):
            k = f"{r.choice(IDENTS)}_{j}"
            kind = r.choice(("string", "string", "integer", "boolean", "enum", "array", "object"))
            d = sentence(r)
            if kind == "enum":
                props[k] = {"type": "string", "enum": r.sample(ADJS, 4), "description": d}
            elif kind == "array":
                props[k] = {"type": "array", "description": d, "items": {"type": "object", "properties": {
                    r.choice(IDENTS): {"type": "string", "description": sentence(r)},
                    r.choice(NOUNS): {"type": "integer", "minimum": 0, "description": sentence(r)}}}}
            elif kind == "object":
                props[k] = {"type": "object", "description": d, "properties": {
                    "id": {"type": "string", "description": f"the {area} {r.choice(NOUNS)} id"},
                    "filter": {"type": "string", "description": sentence(r)}}, "additionalProperties": False}
            else:
                props[k] = {"type": kind, "description": d}
        desc = [f"{name.replace('_', ' ').capitalize()} in the {area} system.", sentence(r)]
        tool = {"type": "function", "function": {"name": name, "description": "", "parameters": {
            "type": "object", "properties": props, "required": sorted(r.sample(sorted(props), 1))}}}
        while True:
            tool["function"]["description"] = " ".join(desc)
            if est(json.dumps(tool["function"]), CPT_TOOLS) >= budget:
                break
            desc.append(sentence(r))
        out.append(tool)
    _HEAD_CACHE[head_tokens] = out
    return out


def genesis_system(n_tok):
    """The main agent's long system prompt (~n_tok tokens), the same for every A."""
    r = random.Random(7)
    parts, k = ["You are Genesis, the main agent of a multi-agent assistant. You plan, delegate work to sub-agents, "
                "check their results and answer the user. Rules by area:"], 0
    while est("\n".join(parts), CPT["prose"]) < n_tok:
        k += 1
        parts.append(f"\n## {k}. {r.choice(ADJS).title()} {r.choice(NOUNS)}\n" + prose(r, 180))
    return "\n".join(parts)


def sub_system(seed, n_tok):
    r = random.Random(seed)
    return ("You are a sub-agent of Genesis. Do exactly the task below with the tools, report with MESSAGE when asked, "
            "and end with STOP.\n\n" + prose(r, n_tok))


def conversation(seed, n_tok, system=None, tools=None):
    """(messages, tools) of an agent conversation of about n_tok template tokens, ending in a user question (default:
    the small 10-tool head and one system prompt for every A; pingpong passes the Genesis head / a sub-agent's)."""
    r = random.Random(seed)
    tools = TOOLS if tools is None else tools
    msgs = [{"role": "system", "content": system if system is not None else system_prompt(random.Random(1))}]
    # what is added is counted as asked (each generator meets its size in real tokens, see CPT); the system prompt,
    # the tool list and the template's tool preamble as measured (~1.9k tokens with production's template)
    # (the small head keeps its old estimate, so the earlier scenarios' prompts stay as they were)
    n = est(msgs[0]["content"], CPT["prose"]) + (est(json.dumps(tools), CPT["json"]) if tools is TOOLS else
                                                 est(json.dumps([t["function"] for t in tools]), CPT_TOOLS)) + 20
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
    return msgs, tools


def short_other(seed, n_tok):
    """B: an unrelated short chat (another client: no tools, its own system prompt), ~n_tok tokens."""
    r = random.Random(seed)
    return [{"role": "system", "content": "You are a helpful assistant. Answer briefly."},
            {"role": "user", "content": "Summarise this email in three bullet points:\n\n" + prose(r, n_tok - 90)}], None


def other_request(a, seed, a_tools):
    """checkbranch's OTHER: unrelated (short_other) or, --other-shares-head, a sub-agent with A's tools + MESSAGE/STOP
    (it shares A's prefix up to the start of the first extra tool, like production's Genesis sub-agents)."""
    if not a.other_shares_head:
        return short_other(seed, a.other_tokens)
    tools = list(a_tools) + SUB_EXTRA_TOOLS
    return conversation(seed, a.other_tokens, sub_system(seed, a.sub_system_tokens), tools)


def checkbranch_head(a):
    """(system, tools) of checkbranch's A: the default small head, or (--checkbranch-head genesis) pingpong's."""
    if a.checkbranch_head == "genesis":
        return genesis_system(a.pp_system_tokens), genesis_tools(a.head_tokens)
    return None, None


def answer_of(reply):
    """The reply as the client sends it back (an empty one as '(no answer)', so the turn still renders)."""
    back = dict(reply)
    if not back.get("content") and not back.get("tool_calls"):
        back["content"] = "(no answer)"
    return back


def new_turn(seed, n_tok):
    return {"role": "user", "content": "Thanks. Now do the first of those and show me the result. " +
            prose(random.Random(seed), n_tok)}


def sub_task(seed):
    r = random.Random(seed)
    return {"role": "user", "content": f"[main agent] Task {r.randint(100, 999)}: check the {r.choice(ADJS)} "
                                       f"{r.choice(NOUNS)} of {r.choice(FILES)} and report the result with STOP. " +
            prose(r, 80)}


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
    if effort and a.genesis_shape:             # Genesis' agent shape: the effort as a template kwarg + a budget
        body["chat_template_kwargs"] = {"reasoning_effort": effort}
        if a.reasoning_budget:
            body["reasoning_budget"] = a.reasoning_budget
    elif effort:
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
           "completion_tokens": usage.get("completion_tokens", 0), "err": err,
           # greedy (--temperature 0): the same prompt must give the same reply in every exact arm
           "reply_sha": hashlib.sha256(json.dumps(msg, sort_keys=True).encode()).hexdigest()[:16]}
    return row, msg


def metrics_row(m, t_send, prompt_tokens=None):
    """The /metrics request entry of the request sent at t_send (the newest one that ended after it; with
    prompt_tokens the oldest such one of that prompt size - production has other clients' requests in between)."""
    after = [x for x in (m or {}).get("requests") or [] if (x.get("time") or 0) >= t_send]   # newest first
    if prompt_tokens:
        same = [x for x in after if x.get("prompt_tokens") == prompt_tokens]
        if same:
            return same[-1]                                    # the oldest of that size after t_send = ours
    return after[0] if after else None                        # the newest one (as before)


def held(m):
    return [s.get("held_tokens") for s in ((m or {}).get("live") or {}).get("slots") or []]


def cache_stats(m):
    c = (m or {}).get("conversation_cache") or {}
    return {k: c.get(k) for k in ("parked", "bytes", "parks", "evictions", "restores")}


def wait_idle(a, settle):
    """Until the server runs nothing and the conversation cache has not changed for `settle` s (a park's capture)."""
    t_end, last, since = time.time() + a.idle_timeout, None, time.time()
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
    print(f"  ! the server did not become idle within {a.idle_timeout:.0f} s", flush=True)
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
        return [ln for ln in data.decode("utf-8", "replace").splitlines() if any(k in ln for k in CACHE_LINE_KEYS)][-60:]


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
SCENARIOS = ("control", "interleave", "rerender", "effort", "checkbranch", "pingpong")
DEFAULT_SCENARIOS = ("control", "interleave", "rerender", "effort")
LOUD = ("skip parking", "evicted", "restor", "superseded", "borrow")     # engine lines printed under a step


def engine_prompt_line(lines):
    """(prompt, reused, read) of the last 'strata serve: prompt N tokens = X reused + Y read' line, else None."""
    for ln in reversed(lines):
        mm = re.search(r"prompt (\d+) tokens = (\d+) reused(?: \+ (\d+) read)?", ln)
        if mm:
            return [int(mm.group(1)), int(mm.group(2)), int(mm.group(3) or 0)]
    return None


def log_counts(lines):
    """The conversation-cache events of a step's engine lines: borrowed / restored / superseded drops / skipped
    parks / parks, and the parks' capture_ms / wait_ms (SLOT_PARK + PARK_ASYNC lines)."""
    cc = [ln for ln in lines if "conversation cache" in ln]
    out = {"borrowed": sum("borrow" in ln for ln in cc), "restored": sum("conversation cache: restored" in ln for ln in cc),
           "superseded": sum("superseded" in ln for ln in cc), "skip_parking": sum("skip parking" in ln for ln in cc),
           "parked": sum(bool(re.search(r"conversation cache: parked \d+ tokens", ln)) for ln in cc),
           "capture_ms": [], "wait_ms": []}
    for ln in cc:
        for k in ("capture_ms", "wait_ms"):
            mm = re.search(k + r"=([\d.]+)", ln)
            if mm:
                out[k].append(float(mm.group(1)))
    return out


def stepper(a, name, ctx, rows, log, pressure):
    """step(role, messages, tools, max_tok, effort) -> (row, reply): one request between two idle waits, with the
    /metrics entry, the conversation-cache deltas and the engine's cache lines of that request."""
    def step(role, messages, tools, max_tok, effort):
        m0 = wait_idle(a, a.settle)
        row, msg = chat(a, messages, tools, max_tok, effort)
        m1 = wait_idle(a, a.settle)
        x = metrics_row(m1, row["t_send"], row["prompt_tokens"]) or {}
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
        row["engine_prompt"] = engine_prompt_line(row["log"])
        row["superseded"] = [ln for ln in row["log"] if "superseded" in ln]
        row["events"] = log_counts(row["log"])
        if row["reused"] is None and row["cached_tokens"] is not None:   # no /metrics entry: the usage field
            row["reused"] = row["cached_tokens"]
        rows.append(row)
        if a.verbose or not role.startswith(("filler", "occupy")):
            print(f"  {role:7s} prompt {row['prompt_tokens']:7d} reused {row['reused'] if row['reused'] is not None else '?':>7} "
                  f"ttft {row['ttft_s']:7.2f}s wall {row['wall_s']:7.2f}s parks {row['cache_delta'].get('parks')} "
                  f"evict {row['cache_delta'].get('evictions')} restores {row['cache_delta'].get('restores')}"
                  f"{'  ERR ' + row['err'] if row['err'] else ''}{'  ! foreign requests' if row['foreign'] > 0 else ''}",
                  flush=True)
            for ln in row["log"]:
                if any(k in ln for k in LOUD) or (name in ("checkbranch", "pingpong") and "strata serve: prompt " in ln):
                    print(f"          | {ln[:220]}", flush=True)
        return row, msg
    return step


def slot_of(r):
    """The slot that holds the request of row r afterwards (held_tokens closest to its prompt + completion)."""
    cands = [(abs((h or 0) - r["prompt_tokens"] - r["completion_tokens"]), i)
             for i, h in enumerate(r.get("held_after") or []) if (h or 0) >= r["prompt_tokens"] > 0]
    return min(cands)[1] if cands else None


def took_slot(r_other, slot, own_prompt):
    """Did the request of row r_other overwrite `slot` (it holds less than own_prompt now)?  None = unknown."""
    if slot is None or not r_other.get("held_after") or slot >= len(r_other["held_after"]):
        return None
    return (r_other["held_after"][slot] or 0) < own_prompt


def ratio_of(r):
    return round((r["reused"] or 0) / r["prompt_tokens"], 4) if r and r.get("prompt_tokens") else 0.0


def scenario(a, name, size, idx, log, pressure, rows_out):
    seed = a.seed * 1000003 + size * 7 + idx * 101
    rows, ctx = [], {"label": a.label, "scenario": name, "size": size, "seed": seed}
    step = stepper(a, name, ctx, rows, log, pressure)
    if name == "pingpong":
        return pingpong(a, size, seed, ctx, rows, step, pressure, rows_out)

    print(f"# {a.label}: {name} @ {size} tokens (seed {seed})", flush=True)
    for f in range(a.fillers):                                    # parked conversations for the gate to evict
        fm, ft = conversation(seed + 17 + f, a.filler_tokens)
        step(f"filler{f}", fm, ft, 16, a.effort)
    system, head = checkbranch_head(a) if name == "checkbranch" else (None, None)
    msgs, tools = conversation(seed, size, system, head)
    ra, reply = step("A", msgs, tools, a.reply_tokens, a.effort)
    a_slot = slot_of(ra)
    avail_b = None
    try:
        if pressure:
            avail_b = pressure.apply()
            print(f"  pressure: test scope MemoryMax lowered, guard-style available {avail_b} MiB "
                  f"(asked {a.avail_mib}; the server's floor {a.floor_mib})", flush=True)
        b_took, rc, ro = None, None, None
        if name == "checkbranch":
            rc, _ = step("CHECK", msgs + [answer_of(reply), {"role": "user", "content": a.check_question}], tools,
                         a.check_reply_tokens, a.effort)
            if a.other_tokens > 0:
                om, ot = other_request(a, seed + 3, tools)
                ro, _ = step("OTHER", om, ot, a.b_reply_tokens, a.effort)
                b_took = took_slot(ro, a_slot, ra["prompt_tokens"])
                print(f"  OTHER in A's slot ({a_slot}): "
                      f"{'yes' if b_took else 'NO - CHECK stayed live' if b_took is False else '?'}"
                      f"{' (shares the head)' if a.other_shares_head else ''}", flush=True)
        elif name != "control":
            for k in range(a.occupy):                             # non-pipelined configs only (see SLOT FORCING)
                om, _ = short_other(seed + 900 + k, 120)
                step(f"occupy{k}", om, None, 8, a.effort)
            bm, bt = short_other(seed + 1, a.b_tokens)
            rb, _ = step("B", bm, bt, a.b_reply_tokens, a.effort)
            b_took = took_slot(rb, a_slot, ra["prompt_tokens"])
            print(f"  B in A's slot ({a_slot}): {'yes' if b_took else 'NO - nothing was parked' if b_took is False else '?'}",
                  flush=True)
        back = answer_of(rerender(reply) if name == "rerender" else reply)
        a2 = msgs + [back, new_turn(seed + 2, a.a2_tokens)]
        r2, _ = step("A2", a2, tools, a.a2_reply_tokens, a.a2_effort if name == "effort" else a.effort)
    finally:
        if pressure:
            pressure.release()
    ratio = ratio_of(r2)
    extra = {}
    if name == "checkbranch":
        # the ideal: everything before the check question (A's prompt + the reply as rendered back); CHECK's prompt
        # minus the question's own tokens is the upper bound, A's prompt the lower one (A's own park)
        q_tok = est(a.check_question, CPT["prose"]) + 8
        extra = {"check_prompt": rc["prompt_tokens"], "check_reused": rc["reused"], "check_ttft_s": rc["ttft_s"],
                 "other_prompt": ro["prompt_tokens"] if ro else None, "other_reused": ro["reused"] if ro else None,
                 "other_tokens": a.other_tokens, "other_shares_head": a.other_shares_head,
                 "checkbranch_head": a.checkbranch_head, "a2_ideal_reuse_est": rc["prompt_tokens"] - q_tok,
                 "a2_reused_vs_a": (r2["reused"] or 0) - ra["prompt_tokens"],
                 "a2_engine_prompt": r2.get("engine_prompt"),
                 "superseded": {row["role"]: len(row["superseded"]) for row in rows if row["superseded"]},
                 "superseded_total": sum(len(row["superseded"]) for row in rows),
                 "foreign": {row["role"]: row["foreign"] for row in rows if row["foreign"] > 0},
                 "check_question_chars": len(a.check_question)}
        print(f"  A2 reused {r2['reused']} of {r2['prompt_tokens']} (ideal ~{extra['a2_ideal_reuse_est']}, A's prompt "
              f"{ra['prompt_tokens']}); superseded drops {extra['superseded'] or 'none'}; engine {r2.get('engine_prompt')}"
              f"{'; ! foreign ' + str(extra['foreign']) if extra['foreign'] else ''}", flush=True)
    summ = {**ctx, "kind": "scenario", "a_slot": a_slot, "b_in_a_slot": b_took, "avail_mib_before_b": avail_b,
            "a_prompt": ra["prompt_tokens"], "a_ttft_s": ra["ttft_s"], "a_reply_sha": ra["reply_sha"],
            "a2_prompt": r2["prompt_tokens"], "a2_reused": r2["reused"], "a2_cached": r2["cached_tokens"],
            "a2_ttft_s": r2["ttft_s"], "a2_wall_s": r2["wall_s"], "a2_reuse": ratio, "hit": ratio >= 0.9,
            "refusals": sum(1 for row in rows for ln in row["log"] if "physical RAM admission" in ln),
            "events": sum_events(rows),
            **extra, "args": {k: v for k, v in vars(a).items() if k not in ("key",)}}
    ev = summ["events"]
    print(f"==> {name} {a.label} @{size}: A2 reused {r2['reused']} of {r2['prompt_tokens']} ({100 * ratio:.1f}%), "
          f"B in A's slot {b_took}, refusals {summ['refusals']}, restored {ev['restored']} borrowed {ev['borrowed']} "
          f"superseded {ev['superseded']} {'HIT' if summ['hit'] else 'MISS'}", flush=True)
    rows_out.extend(rows)
    return summ


def sum_events(rows):
    tot = {"borrowed": 0, "restored": 0, "superseded": 0, "skip_parking": 0, "parked": 0, "capture_ms": [], "wait_ms": []}
    for row in rows:
        for k, v in (row.get("events") or {}).items():
            tot[k] = tot[k] + v
    return tot


def pingpong(a, size, seed, ctx, rows, step, pressure, rows_out):
    """A (Genesis head) -> per cycle: CHECK on the current turn, SUB (shares the head up to tool #43), the next turn."""
    head = genesis_tools(a.head_tokens)
    sub_tools = head + SUB_EXTRA_TOOLS
    print(f"# {a.label}: pingpong @ {size} tokens, {a.cycles} cycles, head {len(head)} tools (~{a.head_tokens} tok), "
          f"SUB ~{a.sub_tokens} with {len(sub_tools)} tools (seed {seed})", flush=True)
    for f in range(a.fillers):
        fm, ft = conversation(seed + 17 + f, a.filler_tokens)
        step(f"filler{f}", fm, ft, 16, a.effort)
    msgs, _ = conversation(seed, int(size * PP_FIT), genesis_system(a.pp_system_tokens), head)
    ra, reply = step("A", msgs, head, a.reply_tokens, a.effort)
    a_slot = slot_of(ra)
    sub_base, _ = conversation(seed + 5, int(a.sub_tokens * PP_FIT), sub_system(seed + 5, a.sub_system_tokens),
                       sub_tools)
    cycles, avail_b, q_tok = [], None, est(a.check_question, CPT["prose"]) + 8
    cur, cur_reply, cur_row = msgs, reply, ra
    try:
        if pressure:
            avail_b = pressure.apply()
            print(f"  pressure: guard-style available {avail_b} MiB (asked {a.avail_mib}; floor {a.floor_mib})", flush=True)
        for c in range(a.cycles):
            sfx = "" if c == 0 else str(c + 1)
            back = answer_of(cur_reply)
            rc, _ = step("CHECK" + sfx, cur + [back, {"role": "user", "content": a.check_question}], head,
                         a.check_reply_tokens, a.effort)
            if a.pp_other:   # an unrelated request overwrites CHECK's slot: the main chat is then only in the RAM cache,
                om, ot = short_other(seed + 70 + c, a.pp_other)   # and SUB's 8192-token root match restores from there
                step("OTHER" + sfx, om, ot, a.b_reply_tokens, a.effort)   # (production's Genesis case, TESTLOG #94)
            sm = sub_base if c == 0 else sub_base[:-1] + [sub_task(seed + 50 + c)]   # SUB2..: a sibling of SUB
            rs, _ = step("SUB" + sfx, sm, sub_tools, a.sub_reply_tokens, a.effort)
            took = took_slot(rs, slot_of(rc), rc["prompt_tokens"])
            nxt = cur + [back, new_turn(seed + 2 + c, a.a2_tokens)]
            rn, rep = step(f"A{c + 2}", nxt, head, a.a2_reply_tokens, a.effort)
            cy = {"cycle": c + 1, "turn": f"A{c + 2}", "check_prompt": rc["prompt_tokens"], "check_reused": rc["reused"],
                  "check_reuse": ratio_of(rc), "sub_prompt": rs["prompt_tokens"], "sub_reused": rs["reused"],
                  "sub_reuse": ratio_of(rs), "sub_ttft_s": rs["ttft_s"], "sub_in_check_slot": took,
                  "prompt": rn["prompt_tokens"], "reused": rn["reused"], "cached": rn["cached_tokens"],
                  "reuse": ratio_of(rn), "ttft_s": rn["ttft_s"], "wall_s": rn["wall_s"],
                  "ideal_reuse_est": rc["prompt_tokens"] - q_tok, "engine_prompt": rn.get("engine_prompt"),
                  # the periodic checkpoint below the branch point: a turn that reuses only that much fell back to the
                  # grid (no branch restore), however close its percentage comes to the target
                  "grid": a.grid, "grid_below": (rc["prompt_tokens"] - q_tok) // a.grid * a.grid,
                  "reply_sha": rn["reply_sha"], "prev_prompt": cur_row["prompt_tokens"],
                  "events": sum_events([rc, rs, rn]),
                  "events_turn": rn.get("events"), "events_sub": rs.get("events")}
            cy["above_grid"] = (rn["reused"] or 0) > cy["grid_below"]
            cy["ok"] = cy["reuse"] >= a.pp_target and cy["above_grid"]
            cycles.append(cy)
            print(f"  cycle {c + 1}: {cy['turn']} reused {rn['reused']} of {rn['prompt_tokens']} ({100 * cy['reuse']:.1f} %, "
                  f"ideal ~{cy['ideal_reuse_est']}, grid point below {cy['grid_below']}{'' if cy['above_grid'] else ' = GRID FALLBACK'}); SUB{sfx} reused {rs['reused']} of {rs['prompt_tokens']}; SUB in "
                  f"CHECK's slot: {'yes' if took else 'no' if took is False else '?'}; engine {rn.get('engine_prompt')}",
                  flush=True)
            cur, cur_reply, cur_row = nxt, rep, rn
    finally:
        if pressure:
            pressure.release()
    ev = sum_events(rows)
    first = cycles[0] if cycles else {}
    ok = bool(cycles) and all(cy["ok"] for cy in cycles)
    summ = {**ctx, "kind": "scenario", "a_slot": a_slot, "b_in_a_slot": first.get("sub_in_check_slot"),
            "avail_mib_before_b": avail_b, "a_prompt": ra["prompt_tokens"], "a_ttft_s": ra["ttft_s"],
            "a_reply_sha": ra["reply_sha"], "a2_prompt": first.get("prompt", 0), "a2_reused": first.get("reused"),
            "a2_cached": first.get("cached"), "a2_ttft_s": first.get("ttft_s", 0.0), "a2_wall_s": first.get("wall_s"),
            "a2_reuse": first.get("reuse", 0.0), "hit": ok, "target": a.pp_target, "grid": a.grid, "cycles": cycles,
            "turn_reuse": {cy["turn"]: cy["reuse"] for cy in cycles},
            "sub_reuse": {f"SUB{'' if cy['cycle'] == 1 else cy['cycle']}": [cy["sub_reused"], cy["sub_prompt"]]
                          for cy in cycles},
            "refusals": sum(1 for row in rows for ln in row["log"] if "physical RAM admission" in ln),
            "events": ev, "head_tools": len(head), "sub_tools": len(sub_tools),
            "args": {k: v for k, v in vars(a).items() if k not in ("key",)}}
    cap = sorted(ev["capture_ms"])
    print(f"==> pingpong {a.label} @{size}: " + ", ".join(f"{cy['turn']} {100 * cy['reuse']:.1f}%"
                                                          f"{'' if cy['above_grid'] else ' (grid fallback)'}" for cy in cycles) +
          f" (target {100 * a.pp_target:.0f}% and above the {a.grid} grid point below the branch); " +
          ", ".join(f"SUB{'' if cy['cycle'] == 1 else cy['cycle']} {cy['sub_reused']}/{cy['sub_prompt']}" for cy in cycles) +
          f"; borrowed {ev['borrowed']} restored {ev['restored']} superseded {ev['superseded']} skip-parking "
          f"{ev['skip_parking']}; capture_ms p50 {cap[len(cap) // 2] if cap else '-'} max {cap[-1] if cap else '-'} "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    rows_out.extend(rows)
    return summ


def table(summaries, title):
    print(f"\n{title}")
    print(f"{'label':16s} {'scenario':11s} {'size':>7s} {'B@A':>4s} {'avail':>6s} {'A ttft':>7s} {'A2 prompt':>9s} "
          f"{'reused':>7s} {'reuse%':>6s} {'A2 ttft':>8s} {'refuse':>6s} hit")
    for s in summaries:
        b = {True: "yes", False: "no", None: "-"}[s.get("b_in_a_slot")]
        print(f"{s['label'][:16]:16s} {s['scenario']:11s} {s['size']:>7d} {b:>4s} {s.get('avail_mib_before_b') or '-':>6} "
              f"{s['a_ttft_s']:>7.2f} {s['a2_prompt']:>9d} {s['a2_reused'] if s['a2_reused'] is not None else '?':>7} "
              f"{100 * s['a2_reuse']:>5.1f}% {s['a2_ttft_s']:>8.2f} {s['refusals']:>6d} {'HIT' if s['hit'] else 'MISS'}")
    pp = [s for s in summaries if s.get("scenario") == "pingpong" and s.get("cycles")]
    if pp:
        pingpong_table(pp)


def pingpong_table(summaries):
    """pingpong: every cycle's turn (A2, A3, ..) and its SUB, plus the cache events of the scenario."""
    print(f"\n# pingpong: per cycle - CHECK (live-slot hit), SUB (shares the head up to tool #43), the next turn "
          f"(target >= its 'target' reused)")
    print(f"{'label':16s} {'size':>7s} {'cyc':>3s} {'turn':>4s} {'prompt':>7s} {'reused':>7s} {'reuse%':>6s} {'ttft':>6s} "
          f"{'CHECK%':>6s} {'SUB prompt':>10s} {'SUB reused':>10s} {'S@C':>3s} {'borrow':>6s} {'restor':>6s} {'supers':>6s} "
          f"{'skip':>4s} {'cap ms':>7s} verdict")
    for s in summaries:
        for cy in s["cycles"]:
            ev = cy.get("events") or {}
            cap = sorted(ev.get("capture_ms") or [])
            sc = {True: "yes", False: "no", None: "-"}[cy.get("sub_in_check_slot")]
            print(f"{s['label'][:16]:16s} {s['size']:>7d} {cy['cycle']:>3d} {cy['turn']:>4s} {cy['prompt']:>7d} "
                  f"{cy['reused'] if cy['reused'] is not None else '?':>7} {100 * cy['reuse']:>5.1f}% {cy['ttft_s']:>6.2f} "
                  f"{100 * cy['check_reuse']:>5.1f}% {cy['sub_prompt']:>10d} "
                  f"{cy['sub_reused'] if cy['sub_reused'] is not None else '?':>10} {sc:>3s} {ev.get('borrowed', 0):>6d} "
                  f"{ev.get('restored', 0):>6d} {ev.get('superseded', 0):>6d} {ev.get('skip_parking', 0):>4d} "
                  f"{(cap[-1] if cap else 0):>7.1f} "
                  f"{'GRID' if cy.get('above_grid') is False else 'ok' if cy['reuse'] >= s.get('target', 0.95) else 'LOW'}")


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


def config_every(path):
    """The config's --prompt-cache-every (the checkpoint grid), else None."""
    try:
        args = json.load(open(path))["args"]
        return int(args[args.index("--prompt-cache-every") + 1])
    except (OSError, ValueError, KeyError, IndexError):
        return None


# --------------------------------------------------------------------------------------------------- dry run
def dry_run(a):
    tmpl = tok = None
    if a.exact:
        sys.dont_write_bytecode = True                     # production's server checkout stays untouched
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
    grid = a.grid

    def common(x, y):
        return next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))

    for size in a.sizes:
        for idx, name in enumerate(a.scenarios):
            seed = a.seed * 1000003 + size * 7 + idx * 101
            if name == "pingpong":
                total += dry_pingpong(a, size, seed, ids, count, chars, common, grid, tmpl)
                if idx == 0:
                    fm, ft = conversation(seed + 17, a.filler_tokens)
                    print(f"{'filler (1 of ' + str(a.fillers) + ')':26s} {len(fm):>5d} {a.filler_tokens:>7d} "
                          f"{chars(fm, ft):>7d} {count(fm, ft, a.effort) or '-':>7}")
                continue
            system, head = checkbranch_head(a) if name == "checkbranch" else (None, None)
            msgs, tools = conversation(seed, size, system, head)
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
                    print(f"{'  A2 ' + v:26s} {len(a2):>5d} {'':>7s} {'':>7s} {len(full):>7d}  first {common(full, base)} "
                          f"tokens = A's prompt ({len(base)})")
            if name == "checkbranch":
                om, ot = other_request(a, seed + 3, tools) if a.other_tokens > 0 else (None, None)
                total += a.other_tokens
                print(f"{'  CHECK question':26s} {'':>5s} {'':>7s} {est(a.check_question):>7d} "
                      f"{'':>7s}  {len(a.check_question)} chars")
                if om:
                    print(f"{'  OTHER':26s} {len(om):>5d} {a.other_tokens:>7d} {chars(om, ot):>7d} "
                          f"{count(om, ot, a.effort) or '-':>7}"
                          + (f"  shares {common(ids(om, ot, a.effort), ids(msgs, tools, a.effort))} tokens with A"
                             if tmpl else "") + ("" if a.other_shares_head else "  (own system prompt, no tools)"))
                if tmpl:              # where A2 leaves CHECK's prompt: the branch point a snapshot has to serve
                    fake = {"role": "assistant", "content": prose(random.Random(5), a.reply_tokens)}
                    chk = ids(msgs + [fake, {"role": "user", "content": a.check_question}], tools, a.effort)
                    a2 = ids(msgs + [fake, new_turn(seed + 2, a.a2_tokens)], tools, a.effort)
                    c = common(a2, chk)
                    print(f"{'  A2 vs CHECK':26s} {'':>5s} {'':>7s} {'':>7s} {len(a2):>7d}  first {c} tokens = "
                          f"CHECK's prompt ({len(chk)}; question {len(chk) - c} tokens); {grid} grid point below: "
                          f"{c // grid * grid}")
            if idx == 0:
                bm, _ = short_other(seed + 1, a.b_tokens)
                fm, ft = conversation(seed + 17, a.filler_tokens)
                print(f"{'B':26s} {len(bm):>5d} {a.b_tokens:>7d} {chars(bm, None):>7d} {count(bm, None, a.effort) or '-':>7}")
                print(f"{'filler (1 of ' + str(a.fillers) + ')':26s} {len(fm):>5d} {a.filler_tokens:>7d} "
                      f"{chars(fm, ft):>7d} {count(fm, ft, a.effort) or '-':>7}")
    print(f"prompt tokens to read at most (every A2 a miss): {total} - at ~5-6k tok/s prefill about "
          f"{total / 5500 / 60:.0f} min per arm")


def dry_pingpong(a, size, seed, ids, count, chars, common, grid, tmpl):
    """The pingpong prompts: sizes, the shared head (A vs SUB: up to tool #43), the branch points; -> tokens to read."""
    head = genesis_tools(a.head_tokens)
    sub_tools = head + SUB_EXTRA_TOOLS
    msgs, _ = conversation(seed, int(size * PP_FIT), genesis_system(a.pp_system_tokens), head)
    sub, _ = conversation(seed + 5, int(a.sub_tokens * PP_FIT), sub_system(seed + 5, a.sub_system_tokens),
                       sub_tools)
    x, xs = count(msgs, head, a.effort), count(sub, sub_tools, a.effort)
    print(f"{f'A pingpong @{size}':26s} {len(msgs):>5d} {size:>7d} {chars(msgs, head):>7d} {x if x is not None else '-':>7}"
          f"  {len(head)} tools, system prompt ~{a.pp_system_tokens}")
    print(f"{'  SUB':26s} {len(sub):>5d} {a.sub_tokens:>7d} {chars(sub, sub_tools):>7d} {xs if xs is not None else '-':>7}"
          f"  {len(sub_tools)} tools")
    print(f"{'  head (42 tools, est)':26s} {'':>5s} {a.head_tokens:>7d} {est(json.dumps([t['function'] for t in head]), CPT_TOOLS):>7d}")
    read = a.fillers * a.filler_tokens + (x or size) + a.cycles * (2 * (x or size) + (xs or a.sub_tokens))
    if not tmpl:
        return read
    ia, isub = ids(msgs, head, a.effort), ids(sub, sub_tools, a.effort)
    h = common(ia, isub)
    print(f"{'  A vs SUB':26s} {'':>5s} {'':>7s} {'':>7s} {h:>7d}  shared head (to tool #43): grid {grid} -> "
          f"{h // grid * grid} reusable on the grid; {'> ' if h > grid else '<= '}{grid}, "
          f"{'<' if h < 2 * grid else '>='} the next point {2 * grid}")
    cur = msgs
    fake = {"role": "assistant", "content": prose(random.Random(5), a.reply_tokens)}
    for c in range(a.cycles):
        chk = ids(cur + [fake, {"role": "user", "content": a.check_question}], head, a.effort)
        nxt = cur + [fake, new_turn(seed + 2 + c, a.a2_tokens)]
        i_n = ids(nxt, head, a.effort)
        cm = common(i_n, chk)
        print(f"{f'  A{c + 2} vs CHECK{c + 1 if c else ""}':26s} {len(nxt):>5d} {'':>7s} {'':>7s} {len(i_n):>7d}  first {cm} "
              f"tokens = CHECK's prompt ({len(chk)}): ideal reuse {100 * cm / len(i_n):.1f} % "
              f"(grid point below {cm // grid * grid} = {100 * (cm // grid * grid) / len(i_n):.1f} %"
              f"{', >= the target: PASS also needs reused > the grid point' if cm // grid * grid >= a.pp_target * len(i_n) else ''})")
        if c:
            s2 = ids(sub[:-1] + [sub_task(seed + 50 + c)], sub_tools, a.effort)
            print(f"{f'  SUB{c + 1} vs SUB':26s} {'':>5s} {'':>7s} {'':>7s} {len(s2):>7d}  first {common(s2, isub)} tokens "
                  f"= SUB's prompt ({len(isub)})")
        cur = nxt
    return read


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
    ap.add_argument("--scenarios", type=lambda s: [x for x in s.split(",") if x], default=list(DEFAULT_SCENARIOS),
                    help=f"comma-separated subset of {','.join(SCENARIOS)} (default {','.join(DEFAULT_SCENARIOS)})")
    ap.add_argument("--b-tokens", type=int, default=900)
    ap.add_argument("--a2-tokens", type=int, default=300, help="new user message tokens in A2")
    ap.add_argument("--reply-tokens", type=int, default=200, help="max_tokens of A (its reply goes into A2)")
    ap.add_argument("--b-reply-tokens", type=int, default=120, help="max_tokens of B and of checkbranch's OTHER")
    ap.add_argument("--other-tokens", type=int, default=30000,
                    help="checkbranch: the unrelated request between CHECK and A2 (0 = none: the live-slot control)")
    ap.add_argument("--check-question", default=CHECK_QUESTION, help="checkbranch: the check question (~236 chars)")
    ap.add_argument("--check-reply-tokens", type=int, default=32, help="checkbranch/pingpong: max_tokens of CHECK")
    ap.add_argument("--other-shares-head", action="store_true",
                    help="checkbranch: OTHER is a sub-agent with A's tools + MESSAGE/STOP (shares A's head) instead of "
                         "an unrelated chat with its own system prompt and no tools")
    ap.add_argument("--checkbranch-head", choices=("small", "genesis"), default="small",
                    help="checkbranch: A's tool head - small (10 tools, ~1.9k tokens) or pingpong's Genesis head")
    ap.add_argument("--head-tokens", type=int, default=13000,
                    help="pingpong: the 42 tools' size in tokens (estimated; --dry-run --exact prints the real head)")
    ap.add_argument("--pp-system-tokens", type=int, default=6000, help="pingpong: the main agent's system prompt")
    ap.add_argument("--sub-tokens", type=int, default=29500, help="pingpong: the sub-agent's prompt size")
    ap.add_argument("--sub-system-tokens", type=int, default=1500, help="pingpong / OTHER: the sub-agent's system prompt")
    ap.add_argument("--sub-reply-tokens", type=int, default=64, help="pingpong: max_tokens of SUB")
    ap.add_argument("--cycles", type=int, default=2, help="pingpong: CHECK/SUB/next-turn cycles (2: A2 and A3)")
    ap.add_argument("--pp-other", type=int, default=0,
                    help="pingpong: an unrelated request of this many tokens between CHECK and SUB (0 = none): it takes "
                         "CHECK's slot, so SUB's shared head is found in the RAM cache (the main chat's parked copy)")
    ap.add_argument("--pp-target", type=float, default=0.95, help="pingpong: a turn's reuse needed for PASS")
    ap.add_argument("--grid", type=int, default=None,
                    help="the test config's --prompt-cache-every (default: production.json's, else 8192); pingpong "
                         "PASS also needs a turn to reuse more than the grid point below its branch")
    ap.add_argument("--genesis-shape", action="store_true",
                    help="send the effort as chat_template_kwargs.reasoning_effort (+ --reasoning-budget), as Genesis")
    ap.add_argument("--reasoning-budget", type=int, default=0, help="--genesis-shape: reasoning_budget (0 = none)")
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
    ap.add_argument("--idle-timeout", type=float, default=600,
                    help="give up waiting for idle after this many s (production is rarely idle: e.g. 60)")
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
    if a.cycles < 1:
        ap.error("--cycles must be >= 1")
    if a.grid is None:
        a.grid = config_every(a.launch or PROD_CONFIG) or 8192

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
