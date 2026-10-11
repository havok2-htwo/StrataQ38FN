#!/usr/bin/env python3
"""Small, reproducible quality check against any OpenAI-compatible endpoint (vLLM, Strata, llama.cpp).

  MMLU-Pro: 240 questions, the same rows every run (stratified over the 14 categories), 10 options, CoT, "The answer is (X)"
  GSM8K:    150 questions (every 8th test row), CoT, last number = answer
Greedy (temperature 0), thinking off.  Data comes from the HF datasets-server JSON API and is cached in data/.

  python3 quick_eval.py --url http://127.0.0.1:8003 --model qwen3.8-flash-next --label vllm-full --vllm
"""
import argparse, concurrent.futures as cf, json, os, re, sys, time, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
API = "https://datasets-server.huggingface.co/rows"


def fetch_rows(dataset, config, split, offset, length):
    q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(f"{API}?{q}", timeout=60) as r:
                return [x["row"] for x in json.load(r)["rows"]]
        except Exception:  # noqa: BLE001
            time.sleep(5 * (attempt + 1))
    raise SystemExit(f"cannot fetch {dataset} {offset}")


def load_sets():
    os.makedirs(DATA, exist_ok=True)
    p = os.path.join(DATA, "sets.json")
    if os.path.exists(p):
        return json.load(open(p))
    mmlu_all = []
    for off in range(0, 12100, 100):
        rows = fetch_rows("TIGER-Lab/MMLU-Pro", "default", "test", off, 100)
        if not rows:
            break
        mmlu_all += rows
    by_cat = {}
    for r in mmlu_all:
        by_cat.setdefault(r["category"], []).append(r)
    mmlu = []
    per = 240 // len(by_cat) + 1
    for cat in sorted(by_cat):
        rows = by_cat[cat]
        step = max(1, len(rows) // per)
        mmlu += rows[::step][:per]
    mmlu = mmlu[:240]
    gsm = []
    for off in range(0, 1319, 100):
        gsm += fetch_rows("openai/gsm8k", "main", "test", off, 100)
    gsm = gsm[::8][:150]
    sets = {"mmlu_pro": [{"q": r["question"], "options": r["options"], "answer": r["answer"], "cat": r["category"]}
                         for r in mmlu],
            "gsm8k": [{"q": r["question"], "answer": r["answer"].split("####")[-1].strip().replace(",", "")}
                      for r in gsm]}
    json.dump(sets, open(p, "w"))
    return sets


def ask(a, prompt):
    body = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": a.max_tokens,
            "temperature": 0, "stream": False}
    if a.vllm:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        body["reasoning_effort"] = "none"
    req = urllib.request.Request(a.url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                return json.load(r)["choices"][0]["message"].get("content") or ""
        except Exception as e:  # noqa: BLE001
            err = repr(e)
            time.sleep(3)
    return f"ERROR {err}"


def grade_mmlu(item, text):
    m = re.findall(r"answer is \(?([A-J])\)?", text)
    if not m:
        m = re.findall(r"\b([A-J])\b", text[-40:])
    return bool(m) and m[-1] == item["answer"], (m[-1] if m else None)


def grade_gsm(item, text):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace("$", ""))
    if not nums:
        return False, None
    got = nums[-1].replace(",", "").rstrip(".")
    try:
        return abs(float(got) - float(item["answer"])) < 1e-6, got
    except ValueError:
        return False, got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="x")
    ap.add_argument("--label", required=True)
    ap.add_argument("--vllm", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=2048)
    a = ap.parse_args()
    sets = load_sets()
    letters = "ABCDEFGHIJ"
    jobs = []
    for i, it in enumerate(sets["mmlu_pro"]):
        opts = "\n".join(f"({letters[k]}) {o}" for k, o in enumerate(it["options"]))
        p = (f"The following is a multiple choice question about {it['cat']}. Think step by step, then finish with "
             f"\"The answer is (X)\" where X is the letter of the correct option.\n\nQuestion: {it['q']}\nOptions:\n{opts}")
        jobs.append(("mmlu_pro", i, it, p))
    for i, it in enumerate(sets["gsm8k"]):
        p = f"Solve the problem step by step. End with the final numeric answer on its own line.\n\n{it['q']}"
        jobs.append(("gsm8k", i, it, p))
    t0 = time.time()
    res = {"mmlu_pro": [], "gsm8k": []}
    out = open(os.path.join(HERE, f"answers-{a.label}.jsonl"), "w")
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(ask, a, p): (s, i, it) for s, i, it, p in jobs}
        for f in cf.as_completed(futs):
            s, i, it = futs[f]
            text = f.result()
            ok, got = (grade_mmlu if s == "mmlu_pro" else grade_gsm)(it, text)
            res[s].append(ok)
            out.write(json.dumps({"set": s, "i": i, "ok": ok, "got": got, "want": it["answer"], "text": text[-400:]}) + "\n")
    row = {"label": a.label, "mmlu_pro": round(100 * sum(res["mmlu_pro"]) / len(res["mmlu_pro"]), 1),
           "gsm8k": round(100 * sum(res["gsm8k"]) / len(res["gsm8k"]), 1),
           "n": {k: len(v) for k, v in res.items()}, "minutes": round((time.time() - t0) / 60, 1)}
    print(json.dumps(row))
    with open(os.path.join(HERE, "eval_results.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    sys.exit(main())
