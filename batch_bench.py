"""Batching-Messung: Decode und Prefill mit 1/2/4 gleichzeitigen Requests, plus Decode neben einem langen Prompt.

Startet Strata (Server + Engine) auf eigenem Port mit einer Config-Variante, misst, beendet es wieder.

usage: python batch_bench.py NAME [--strata Strata-1402] [--exe <strata.exe>] [--parallel N] [--add flag[=VALUE]]
                             [--set flag=VALUE] [--drop flag] [--env K=V] [--port 8090] [--keep]
Schreibt eine JSON-Zeile je Arm nach batch_results.jsonl; Engine-Log in logs/engine-NAME.log.
"""
import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from agent_bench import MemSampler, chat, corpus_main, mem_now, vram_free_mib  # noqa: E402
from run_variant import edit_args, kill_port, wait_ready  # noqa: E402

PY = HERE / "Strata" / ".venv" / "Scripts" / "python.exe"
TOPICS = ["how NVMe SSD wear leveling works", "the CAP theorem in distributed systems",
          "how TCP cubic congestion control works", "why B-trees are used in database indexes",
          "how virtual memory and page tables work", "how GPU tensor cores execute matrix operations"]
CORPUS = corpus_main()


def decode_prompt(i):
    # eindeutiger Anfang: kein Prompt-Cache-Treffer zwischen den Requests
    return [{"role": "user", "content": f"[{uuid.uuid4().hex[:8]}] Explain in great detail {TOPICS[i % len(TOPICS)]}. "
                                        f"Write at least 1500 words with sections and examples."}]


def prefill_prompt(k, n_chars):
    off = (k * 37_000) % max(1, len(CORPUS) - n_chars)
    doc = CORPUS[off:off + n_chars]
    return [{"role": "user", "content": f"[{uuid.uuid4().hex[:8]}] Read this documentation:\n\n{doc}\n\n"
                                        f"Summarize it in three sentences."}]


def concurrent_round(url, n, make, max_tokens):
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        res = list(ex.map(lambda i: chat(url, make(i), max_tokens), range(n)))
    wall = time.perf_counter() - t0
    for r in res:
        r.pop("text", None)
    return wall, res


def decode_test(url, n, max_tokens=600):
    wall, res = concurrent_round(url, n, decode_prompt, max_tokens)
    out = sum(r["out_tokens"] for r in res)
    row = {"test": "decode", "n": n, "wall_s": round(wall, 2), "agg_out_tps": round(out / wall, 1),
           "per_stream_tps": [r["decode_tps"] for r in res], "ttft_s": [r["ttft_s"] for r in res],
           "out_tokens": [r["out_tokens"] for r in res]}
    print(f"   decode N={n}: gesamt {row['agg_out_tps']} tok/s, je Stream {row['per_stream_tps']}, "
          f"TTFT {row['ttft_s']}, wall {row['wall_s']} s", flush=True)
    return row


PREFILL_K = [0]


def prefill_test(url, n, n_chars=32_000):
    base = PREFILL_K[0]
    PREFILL_K[0] += n
    wall, res = concurrent_round(url, n, lambda i: prefill_prompt(base + i, n_chars), 16)
    pt = sum(r["prompt_tokens"] or 0 for r in res)
    row = {"test": "prefill", "n": n, "wall_s": round(wall, 2), "prompt_tokens": [r["prompt_tokens"] for r in res],
           "agg_prompt_tps": round(pt / max(r["ttft_s"] for r in res), 1), "ttft_s": [r["ttft_s"] for r in res]}
    print(f"   prefill N={n}: {pt} Prompt-Tokens, gesamt {row['agg_prompt_tps']} tok/s, TTFT {row['ttft_s']}",
          flush=True)
    return row


def mixed_test(url, n_chars=64_000):
    """Ein Stream decodiert 1500 Tokens; nach 4 s kommt ein langer Prompt dazu."""
    res = {}

    def long_decode():
        res["dec"] = chat(url, decode_prompt(0), 1500)

    th = threading.Thread(target=long_decode)
    th.start()
    time.sleep(4)
    res["pf"] = chat(url, prefill_prompt(PREFILL_K[0] + 7, n_chars), 200)
    PREFILL_K[0] += 1
    th.join()
    d, p = res["dec"], res["pf"]
    row = {"test": "mixed", "dec_tps": d["decode_tps"], "dec_out": d["out_tokens"], "dec_ttft_s": d["ttft_s"],
           "pf_prompt_tokens": p["prompt_tokens"], "pf_ttft_s": p["ttft_s"], "pf_dec_tps": p["decode_tps"]}
    print(f"   mixed: Decode-Stream {row['dec_tps']} tok/s ({row['dec_out']} Tok.), langer Prompt "
          f"{row['pf_prompt_tokens']} Tok. TTFT {row['pf_ttft_s']} s, danach {row['pf_dec_tps']} tok/s", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--base", default="strata-unsloth-ud-iq4_xs.json", help="Config in Strata\\")
    ap.add_argument("--strata", default="Strata-1402", help="Strata-Verzeichnis (serve\\, engine\\)")
    ap.add_argument("--parallel", type=int, default=0)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--add", action="append", default=[])
    ap.add_argument("--drop", action="append", default=[])
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--rounds", default="1,2,4")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()

    strata = (HERE / a.strata).resolve()
    cfg = json.loads((HERE / "Strata" / a.base).read_text(encoding="utf-8"))
    cfg["exe"] = str(strata / "engine" / "strata.exe")
    cfg["args"] = edit_args(cfg["args"], a.set, a.add, a.drop)
    if a.parallel:
        cfg["parallel"] = a.parallel
    else:
        cfg.pop("parallel", None)
    log = HERE / "logs" / f"engine-{a.name}.log"
    log.parent.mkdir(exist_ok=True)
    log.unlink(missing_ok=True)
    cfg["log"] = str(log)
    cfg["port"] = a.port
    cfg.pop("cors_origins", None)
    cfg_path = HERE / "configs" / f"strata-{a.name}.json"
    cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")

    env = dict(os.environ, PYTHONUNBUFFERED="1")
    for kv in a.env:
        k, v = kv.split("=", 1)
        env[k] = v
    url = f"http://127.0.0.1:{a.port}"
    out_path = HERE / "logs" / f"server-{a.name}.out"
    out = open(out_path, "w", encoding="utf-8")

    print(f"[{a.name}] engine={cfg['exe']} parallel={a.parallel} add={a.add} set={a.set} drop={a.drop}", flush=True)
    kill_port(a.port)
    mem_before = mem_now()
    t0 = time.time()
    proc = subprocess.Popen([str(PY), str(strata / "serve" / "server.py"), "--engine", "strata", "--config",
                             str(cfg_path), "--port", str(a.port), "--host", "127.0.0.1"],
                            cwd=str(strata), env=env, stdout=out, stderr=subprocess.STDOUT)
    if not wait_ready(url, proc):
        out.flush()
        print(out_path.read_text(encoding="utf-8", errors="replace")[-3000:])
        kill_port(a.port)
        sys.exit(1)
    load_s = round(time.time() - t0)
    time.sleep(10)
    mem_loaded, vram_loaded = mem_now(), vram_free_mib()
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if any(k in line.lower() for k in ("batch", "expert cache", "warning", "error", "failed", "slot")) \
                and "verify:" not in line:
            print("   log:", line[:240], flush=True)
    print(f"[{a.name}] bereit nach {load_s} s, VRAM frei {vram_loaded} MiB", flush=True)

    sampler = MemSampler()
    sampler.start()
    rows, err = [], None
    rounds = [int(x) for x in a.rounds.split(",")]
    try:
        chat(url, [{"role": "user", "content": "Write a short poem about the sea."}], 128)
        chat(url, decode_prompt(5), 200)
        for n in rounds:
            rows.append(decode_test(url, n))
        for n in rounds:
            rows.append(prefill_test(url, n))
        rows.append(mixed_test(url))
    except Exception as e:  # was bis dahin gemessen ist, bleibt
        err = repr(e)
        print("   FEHLER:", err, flush=True)
    sampler.stop = True
    time.sleep(2)
    alive = proc.poll() is None
    summary = {"name": a.name, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "engine": cfg["exe"],
               "parallel": a.parallel, "add": a.add, "set": a.set, "drop": a.drop, "env": a.env, "error": err,
               "server_alive": alive, "load_s": load_s, "vram_free_loaded_mib": vram_loaded,
               "mem_before": mem_before, "mem_loaded": mem_loaded, "mem_during": sampler.summary(), "rows": rows}
    with open(HERE / "batch_results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    errs = [l for l in log.read_text(encoding="utf-8", errors="replace").splitlines()
            if ("failed" in l or "ERR" in l or "error" in l.lower()) and "verify:" not in l]
    for l in errs[-8:]:
        print("   log:", l[:240], flush=True)
    print(f"[{a.name}] fertig, Server lebt: {alive}, Commit frei min "
          f"{summary['mem_during'].get('commit_free_min_gib')} GiB", flush=True)
    if not a.keep:
        kill_port(a.port)


if __name__ == "__main__":
    main()
