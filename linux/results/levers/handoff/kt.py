import sqlite3, sys
db = sqlite3.connect(sys.argv[1]); dev=int(sys.argv[2]); a=float(sys.argv[3]); b=float(sys.argv[4])
t0 = db.execute("select min(k.start) from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id where s.value like 'mul_mat_q%'").fetchone()[0]
for s,e,n,st in db.execute("select k.start,k.end,s.value,k.streamId from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id where k.deviceId=? order by k.start",(dev,)):
    rs=(s-t0)/1e6
    if a<=rs<=b: print(f"{rs:8.3f} {(e-s)/1e3:7.1f}us st{st} {n[:60]}")
for s,e,k,by,st in db.execute("select start,end,copyKind,bytes,streamId from CUPTI_ACTIVITY_KIND_MEMCPY where deviceId=? order by start",(dev,)):
    rs=(s-t0)/1e6
    if a<=rs<=b: print(f"{rs:8.3f} {(e-s)/1e3:7.1f}us st{st} MEMCPY kind{k} {by}")
for s,e,by,st in db.execute("select start,end,bytes,streamId from CUPTI_ACTIVITY_KIND_MEMSET where deviceId=? order by start",(dev,)):
    rs=(s-t0)/1e6
    if a<=rs<=b: print(f"{rs:8.3f} {(e-s)/1e3:7.1f}us st{st} MEMSET {by}")
