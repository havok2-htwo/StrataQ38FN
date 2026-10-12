#!/usr/bin/env python3
"""Compare the SASS of every kernel in a 'before' dump with the same kernel in an 'after' dump (cuobjdump -sass).
Addresses and encodings are dropped; the instruction text must be identical."""
import re, sys
def funcs(path):
    out, cur, name = {}, None, None
    arch = None
    for l in open(path):
        m = re.match(r"\s*arch = (sm_\d+)", l)
        if m: arch = m.group(1)
        m = re.match(r"\s*Function : (\S+)", l)
        if m:
            name = (arch, re.sub(r"_GLOBAL__N__[0-9a-f]+_\d+_(\w+?)_cu_[0-9a-f]{8}", r"ANON_\1", m.group(1))); cur = out.setdefault(name, []); continue
        if cur is None: continue
        m = re.match(r"\s*/\*[0-9a-f]{4,}\*/\s*(.*?)\s*;?\s*(/\*.*)?$", l)
        if m and m.group(1): cur.append(m.group(1))
    return out
b, a = funcs(sys.argv[1]), funcs(sys.argv[2])
same = diff = 0
for k, v in b.items():
    if k not in a: print("MISSING", k); diff += 1; continue
    if a[k] == v: same += 1
    else: diff += 1; print("DIFF", k, len(v), len(a[k]))
new = [k for k in a if k not in b]
print(f"{sys.argv[1].split('/')[-1]}: {same} kernels identical, {diff} differ/missing; {len(new)} new: " +
      ", ".join(sorted(set(re.sub(r'I.*', '', k[1]) for k in new))))
