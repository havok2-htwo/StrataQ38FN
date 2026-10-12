import sqlite3, sys
db = sqlite3.connect(sys.argv[1])
# first prompt MMQ kernel as t0
t0 = db.execute("select min(k.start) from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id where s.value like 'mul_mat_q%'").fetchone()[0]
rows = db.execute("select deviceId, copyKind, bytes, start, end, streamId from CUPTI_ACTIVITY_KIND_MEMCPY where bytes > 500000 and copyKind in (1,2) order by start").fetchall()
for d, k, b, s, e, st in rows:
    print(f"GPU{d} {'H2D' if k==1 else 'D2H'} {b/1e6:6.2f} MB  {(s-t0)/1e6:8.3f} - {(e-t0)/1e6:8.3f}  ({(e-s)/1e3:6.0f} us) stream {st}")
# kernel spans per device: last kernel end before each H2D of the hand-off, first kernel after
