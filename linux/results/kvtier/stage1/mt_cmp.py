# mt_cmp.py <multiturn.jsonl> <labelA> <labelB>: reply sha256 per (uid, turn), A vs B
import json, sys
rows = {}
for l in open(sys.argv[1]):
    r = json.loads(l); rows[r["label"]] = r
def m(lab):
    r = rows[lab]; f = r["reply_fields"]
    return {(x[f.index("uid")], x[f.index("turn")]): x[f.index("sha256")] for x in r["replies"]}
a, b = m(sys.argv[2]), m(sys.argv[3])
keys = sorted(set(a) | set(b))
same = [k for k in keys if a.get(k) == b.get(k)]
diff = [k for k in keys if a.get(k) != b.get(k)]
print(f"{sys.argv[2]} vs {sys.argv[3]}: {len(same)}/{len(keys)} replies identical")
for t in sorted({k[1] for k in keys}):
    kk = [k for k in keys if k[1] == t]
    print(f"  turn {t}: {sum(a.get(k) == b.get(k) for k in kk)}/{len(kk)}")
if diff: print("  first differing (uid, turn):", diff[:10])
