# replay_user.py <url> <uid> <turns> <out.json>: one bench_multiturn user alone (greedy, the same messages), every
# turn's prompt/cached/gen/sha256 and the replies' text
import hashlib, json, sys, types
sys.path.insert(0, "/media/raid0/StrataQ38FN/linux")
import bench_multiturn as B
url, uid, turns, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
a = types.SimpleNamespace(url=url, model="x", vllm=False, max_tok=400, temperature=0.0, doc=8000, add=3000)
msgs = [{"role": "system", "content": f"You are agent #{uid}. Answer concisely and precisely."}]
rows = []
for t in range(turns):
    n = a.doc if t == 0 else a.add
    msgs.append({"role": "user", "content": f"[agent {uid}, step {t + 1}] New material:\n{B.doc(uid, t, n)}\n\n"
                                            f"Task: summarise what is new in this material and how it relates "
                                            f"to the earlier steps."})
    r = B.call(a, msgs)
    msgs.append({"role": "assistant", "content": r["text"]})
    r["sha256"] = hashlib.sha256(r["text"].encode("utf-8")).hexdigest()
    rows.append({k: r[k] for k in ("prompt", "cached", "gen", "sha256", "text")})
    print(t + 1, r["prompt"], r["cached"], r["gen"], r["sha256"][:16], flush=True)
json.dump(rows, open(out, "w"), indent=1)
