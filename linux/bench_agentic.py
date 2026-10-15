#!/usr/bin/env python3
"""A synthetic coding-agent workload against an OpenAI-compatible endpoint (Strata, vLLM): closer to real use than
lockstep users.

Per session (a developer with an agentic coding client):
  - one main agent: a long shared system prompt (the client's tool list, identical in every agent of every session) and
    a first user turn with ~--context tokens of repository context and a task;
  - every main step: the full history + an EPHEMERAL reminder message at the end (as Qwen Code / Claude Code append one
    per request and drop it in the next) -> a reply -> a tool result (--tool-min..--tool-max tokens) appended;
  - at --spawn-at steps the main agent spawns --subagents sub-agents that work in PARALLEL (same system prompt, their
    own task and a slice of context, --sub-steps steps with tool results each); the main agent waits for all of them
    and gets their final answers as one tool result.
Beside the sessions: a chat user (an Open WebUI tab) asks short independent questions at random intervals (mean
--chat-every s), some with a follow-up - the requests that should not wait behind the agents.

What it reports per role (main / sub / chat): requests, prompt and cached tokens, TTFT p50/p95/max, decode p50; overall:
wall time, output tokens, aggregate output tok/s and how long each session took.  Deterministic texts; --temperature 0
makes two arms comparable.  With --scale the token sizes shrink for a quick smoke run.

  python3 bench_agentic.py --url http://127.0.0.1:8090 --label strata-agentic
  python3 bench_agentic.py --url http://127.0.0.1:8096 --model qwen3.8-flash-next --vllm --label vllm-agentic
"""
import argparse, concurrent.futures as cf, datetime, json, random, statistics, threading, time, urllib.request

WORDS = ("def class return import self value index buffer cache request handler config parser token error retry "
         "timeout async await thread lock queue worker batch slot kernel tensor stream offset length header payload "
         "client server route schema field record table query commit branch merge test assert mock fixture").split()


def text(seed, n_tok):
    """Deterministic code-ish text of about n_tok tokens (~1.3 words per token), unique per seed."""
    out, k = [], seed * 2654435761 + 97
    for i in range(max(1, int(n_tok / 1.3))):
        k = (k * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        out.append(WORDS[(k >> 33) % len(WORDS)])
        if i % 13 == 12:
            out.append(f"# L{seed % 100000}.{i}\n")
    return " ".join(out)


class Client:
    def __init__(self, a):
        self.a = a
        self.rows, self.lock = [], threading.Lock()
        self.t0 = time.perf_counter()

    def call(self, role, who, messages, max_tok):
        a = self.a
        body = {"model": a.model, "messages": messages, "max_tokens": max_tok, "temperature": a.temperature,
                "stream": True, "stream_options": {"include_usage": True}}
        if a.vllm:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        else:
            body["reasoning_effort"] = "none"
        headers = {"Content-Type": "application/json"}
        if a.key:
            headers["Authorization"] = f"Bearer {a.key}"
        req = urllib.request.Request(a.url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers=headers)
        t_send = time.perf_counter()
        t_first, parts, usage, err = None, [], {}, None
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
                        d = (ch.get("delta") or {}).get("content")
                        if d:
                            if t_first is None:
                                t_first = time.perf_counter()
                            parts.append(d)
        except Exception as e:                       # noqa: BLE001 - a failed request is a result, not a crash
            err = f"{type(e).__name__}: {e}"
        t_end = time.perf_counter()
        gen = usage.get("completion_tokens", len(parts))
        row = {"role": role, "who": who, "at": round(t_send - self.t0, 2), "ttft": (t_first or t_end) - t_send,
               "e2e": t_end - t_send, "gen": gen, "prompt": usage.get("prompt_tokens", 0),
               "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0,
               "dec": (gen - 1) / (t_end - t_first) if t_first and gen > 1 and t_end > t_first else 0.0,
               "err": err}
        with self.lock:
            self.rows.append(row)
        if a.verbose:
            print(f"[{row['at']:7.1f}s] {role:4s} {who:10s} prompt {row['prompt']:6d} cached {row['cached']:6d} "
                  f"ttft {row['ttft']:6.2f}s gen {gen:4d} {row['dec']:6.1f} tok/s{'  ERR ' + err if err else ''}",
                  flush=True)
        return "".join(parts) or "(no answer)"


def sizes(a, n):
    return max(16, int(n * a.scale))


def subagent(c, a, sess, k, spawn, system):
    who = f"s{sess}.sub{spawn}.{k}"
    rnd = random.Random(sess * 1000 + spawn * 10 + k)
    msgs = [system, {"role": "user", "content": f"[sub-agent {who}] Task: investigate part {k} of the change and "
                                                 f"report findings with file names.\n\n"
                                                 f"{text(sess * 100003 + spawn * 1009 + k, sizes(a, a.sub_context))}"}]
    reply = ""
    for step in range(a.sub_steps):
        reply = c.call("sub", who, msgs, a.sub_out)
        msgs.append({"role": "assistant", "content": reply})
        tool = text(sess * 7919 + spawn * 104729 + k * 31 + step, sizes(a, rnd.randint(a.tool_min, a.tool_max)))
        msgs.append({"role": "user", "content": f"Tool result (read_file / grep, step {step + 1}):\n{tool}"})
    return reply


def session(c, a, sess, system):
    time.sleep(sess * a.stagger)
    who = f"s{sess}.main"
    rnd = random.Random(sess)
    t_start = time.perf_counter()
    msgs = [system, {"role": "user", "content": f"[session {sess}] Here is the repository context. Implement the feature "
                                                 f"described at the end, step by step, using the tools.\n\n"
                                                 f"{text(sess * 31337, sizes(a, a.context))}\n\nFeature: add retry "
                                                 f"handling with backoff to the request client."}]
    spawn_at = {int(x) for x in a.spawn_at.split(",") if x.strip()}
    for step in range(a.main_steps):
        reminder = {"role": "user", "content": f"<system-reminder>todo list and open files, step {step + 1}:\n"
                                                f"{text(sess * 977 + step, sizes(a, a.reminder))}</system-reminder>"}
        reply = c.call("main", who, msgs + [reminder], a.main_out)        # the reminder is not kept
        msgs.append({"role": "assistant", "content": reply})
        if step in spawn_at and a.subagents > 0:
            with cf.ThreadPoolExecutor(a.subagents) as ex:
                answers = list(ex.map(lambda k: subagent(c, a, sess, k, step, system), range(a.subagents)))
            msgs.append({"role": "user", "content": "Tool result (agent): the sub-agents report:\n" +
                         "\n\n".join(f"[sub-agent {k}] {ans}" for k, ans in enumerate(answers))})
        else:
            tool = text(sess * 524287 + step, sizes(a, rnd.randint(a.tool_min, a.tool_max)))
            msgs.append({"role": "user", "content": f"Tool result (step {step + 1}):\n{tool}"})
        time.sleep(a.think)
    return time.perf_counter() - t_start


def chat_user(c, a, stop):
    rnd = random.Random(424242)
    n = 0
    while not stop.wait(rnd.expovariate(1.0 / a.chat_every)):
        n += 1
        msgs = [{"role": "user", "content": f"Quick question {n}: explain this snippet briefly.\n"
                                            f"{text(900000 + n, sizes(a, rnd.randint(300, 1500)))}"}]
        reply = c.call("chat", f"chat{n}", msgs, a.chat_out)
        if rnd.random() < 0.4 and not stop.is_set():
            msgs += [{"role": "assistant", "content": reply}, {"role": "user", "content": "And how would you test it?"}]
            c.call("chat", f"chat{n}b", msgs, a.chat_out)


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="strata")
    ap.add_argument("--key", default="")
    ap.add_argument("--label", required=True)
    ap.add_argument("--vllm", action="store_true", help="thinking off via chat_template_kwargs (vLLM)")
    ap.add_argument("--sessions", type=int, default=2, help="agentic coding sessions at once")
    ap.add_argument("--stagger", type=float, default=20.0, help="seconds between the sessions' starts")
    ap.add_argument("--system", type=int, default=6000, help="tokens of the shared system prompt (tool list)")
    ap.add_argument("--context", type=int, default=20000, help="tokens of repository context in the first turn")
    ap.add_argument("--reminder", type=int, default=2500, help="tokens of the ephemeral reminder per request")
    ap.add_argument("--main-steps", type=int, default=10)
    ap.add_argument("--main-out", type=int, default=400)
    ap.add_argument("--tool-min", type=int, default=800)
    ap.add_argument("--tool-max", type=int, default=4000)
    ap.add_argument("--spawn-at", default="2,6", help="main steps (0-based) that spawn sub-agents")
    ap.add_argument("--subagents", type=int, default=3)
    ap.add_argument("--sub-context", type=int, default=6000)
    ap.add_argument("--sub-steps", type=int, default=3)
    ap.add_argument("--sub-out", type=int, default=300)
    ap.add_argument("--chat-every", type=float, default=25.0, help="mean seconds between chat questions (0 = none)")
    ap.add_argument("--chat-out", type=int, default=300)
    ap.add_argument("--think", type=float, default=1.0, help="seconds an agent waits between its steps")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every token size (0.05 for a smoke run)")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--out", default="", help="append one JSON summary line (and the rows) to this file")
    ap.add_argument("--verbose", action="store_true", help="one line per request")
    a = ap.parse_args()

    system = {"role": "system", "content": "You are a coding agent. Tools: read_file, write_file, edit, grep, glob, "
                                           "run_shell_command, agent, todo_write. Tool reference:\n" +
                                           text(1, sizes(a, a.system))}
    c = Client(a)
    stop = threading.Event()
    chat = threading.Thread(target=chat_user, args=(c, a, stop), daemon=True) if a.chat_every > 0 else None
    if chat:
        chat.start()
    with cf.ThreadPoolExecutor(a.sessions) as ex:
        took = list(ex.map(lambda s: session(c, a, s, system), range(a.sessions)))
    stop.set()
    if chat:
        chat.join(timeout=a.timeout)
    wall = time.perf_counter() - c.t0
    rows = c.rows

    print(f"# {a.label}: {a.sessions} sessions x {a.main_steps} main steps, {a.subagents} sub-agents at steps "
          f"{a.spawn_at} ({a.sub_steps} steps each), context {sizes(a, a.context)}, chat every ~{a.chat_every:g} s")
    print(f"{'role':5s} {'req':>4s} {'err':>3s} {'prompt':>7s} {'cached':>7s} {'TTFT p50':>9s} {'p95':>6s} {'max':>6s} "
          f"{'dec p50':>8s}")
    summary = {"label": a.label, "ts": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
               "args": vars(a), "wall_s": round(wall, 1), "session_s": [round(t, 1) for t in took]}
    for role in ("main", "sub", "chat"):
        rs = [r for r in rows if r["role"] == role]
        if not rs:
            continue
        ok = [r for r in rs if not r["err"]]
        tt = [r["ttft"] for r in ok]
        row = {"req": len(rs), "err": len(rs) - len(ok),
               "prompt": int(statistics.mean(r["prompt"] for r in ok)) if ok else 0,
               "cached": int(statistics.mean(r["cached"] for r in ok)) if ok else 0,
               "ttft_p50": round(statistics.median(tt), 2) if tt else None, "ttft_p95": round(pct(tt, .95), 2),
               "ttft_max": round(max(tt), 2) if tt else None,
               "dec_p50": round(statistics.median(r["dec"] for r in ok), 1) if ok else None}
        summary[role] = row
        print(f"{role:5s} {row['req']:>4d} {row['err']:>3d} {row['prompt']:>7d} {row['cached']:>7d} "
              f"{row['ttft_p50'] or 0:>9.2f} {row['ttft_p95']:>6.2f} {row['ttft_max'] or 0:>6.2f} {row['dec_p50'] or 0:>8.1f}")
    out_tok = sum(r["gen"] for r in rows if not r["err"])
    summary.update(out_tok=out_tok, agg_out_tps=round(out_tok / wall, 1),
                   prompt_tok=sum(r["prompt"] for r in rows), cached_tok=sum(r["cached"] for r in rows))
    print(f"wall {wall:.1f} s, sessions took {', '.join(f'{t:.0f} s' for t in took)}, {out_tok} output tokens, "
          f"{summary['agg_out_tps']} tok/s aggregate, {summary['prompt_tok']} prompt tokens "
          f"({summary['cached_tok']} cached as reported)")
    if a.out:
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps({**summary, "rows": rows}) + "\n")


if __name__ == "__main__":
    main()
