import sqlite3, sys
for f in sys.argv[1:]:
    db = sqlite3.connect(f)
    t0 = db.execute("select min(k.start) from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id where s.value like 'mul_mat_q%'").fetchone()[0]
    mc = db.execute("select deviceId, copyKind, bytes, start, end from CUPTI_ACTIVITY_KIND_MEMCPY where bytes >= 500000 and copyKind in (1,2) order by start").fetchall()
    # group consecutive D2H pieces per device (gap < 0.1 ms) = one hand-off out; H2D likewise
    def groups(dev, kind):
        g = []
        for d, k, b, s, e in mc:
            if d != dev or k != kind: continue
            if g and s - g[-1][1] < 100000: g[-1][1] = e; g[-1][2] += b
            else: g.append([s, e, b])
        return g
    out = []
    for d in range(3):
        for s, e, b in groups(d, 2):
            if b < 5e6: continue
            # receiver: first H2D group on d+1 starting after s with same bytes
            r = [x for x in groups(d + 1, 1) if x[0] >= s and abs(x[2] - b) < 1e5]
            if not r: continue
            r = r[0]
            k = db.execute("select min(start) from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? and start>=?", (d + 1, r[1])).fetchone()[0]
            ke = db.execute("select max(end) from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? and end<=? and streamId=(select streamId from CUPTI_ACTIVITY_KIND_MEMCPY where deviceId=? and start=?)", (d, s, d, s)).fetchone()[0]
            out.append(f"G{d}->G{d+1} {b/1e6:4.1f}MB @{(ke-t0)/1e6:6.1f}: send end->recv compute {(k-ke)/1e6:5.2f} ms (D2H {(e-s)/1e6:.2f}, H2D start +{(r[0]-ke)/1e6:.2f}, H2D end +{(r[1]-ke)/1e6:.2f})")
    print("==", f); print("\n".join(sorted(out, key=lambda x: float(x.split('@')[1].split(':')[0]))))
