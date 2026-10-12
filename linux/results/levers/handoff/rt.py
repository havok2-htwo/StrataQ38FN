import sqlite3, sys
db = sqlite3.connect(sys.argv[1]); a=float(sys.argv[2]); b=float(sys.argv[3])
t0 = db.execute("select min(k.start) from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id where s.value like 'mul_mat_q%'").fetchone()[0]
for s,e,n,tid in db.execute("select r.start,r.end,s.value,r.globalTid from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id order by r.start"):
    rs=(s-t0)/1e6
    if a<=rs<=b: print(f"{rs:8.3f} {(e-s)/1e3:8.1f}us tid{tid%100000} {n}")
