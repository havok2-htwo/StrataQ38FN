# cmp.py new.json ref.json : byte compare + per-key token compare of two dumps
import json, sys
a, b = sys.argv[1], sys.argv[2]
ra, rb = open(a, 'rb').read(), open(b, 'rb').read()
print(f"{a.split('/')[-1]} vs {b}: bytes {'IDENTICAL' if ra == rb else 'differ'}")
A, B = json.loads(ra), json.loads(rb)
def walk(x, y, path):
    if type(x) != type(y): print(f"  {path}: type differs"); return 1
    if isinstance(x, dict):
        n = 0
        for k in sorted(set(x) | set(y)):
            if k not in x or k not in y: print(f"  {path}/{k}: missing on one side"); n += 1
            else: n += walk(x[k], y[k], f"{path}/{k}")
        return n
    if isinstance(x, list) and x and all(isinstance(v, int) for v in x + y):
        if x != y:
            i = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
            print(f"  {path}: differ at token {i} (len {len(x)} vs {len(y)})"); return 1
        return 0
    if isinstance(x, list):
        if len(x) != len(y): print(f"  {path}: len {len(x)} vs {len(y)}"); return 1
        return sum(walk(p, q, f"{path}[{i}]") for i, (p, q) in enumerate(zip(x, y)))
    if x != y: print(f"  {path}: {x!r} vs {y!r}"); return 1
    return 0
n = walk(A, B, "")
print(f"  {n} differing entries")
