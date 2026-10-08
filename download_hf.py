"""Resumable download of a whole HF repo revision (all files) with size check.
usage: python download_hf.py REPO REVISION OUTDIR"""
import json, os, sys, time, urllib.request
repo, rev, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
tree = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true"))
files = [(e["path"], e.get("size") or 0) for e in tree if e["type"] == "file"]
files.sort(key=lambda x: x[1])
print(f"{len(files)} files, {sum(s for _, s in files)/1e9:.1f} GB", flush=True)
for name, size in files:
    path = os.path.join(out, name.replace("/", os.sep)); os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) == size:
        continue
    tmp = path + ".part"
    while True:
        try:
            done = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            req = urllib.request.Request(f"https://huggingface.co/{repo}/resolve/{rev}/{name}",
                                         headers={"Range": f"bytes={done}-"} if done else {})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "ab" if done and r.status == 206 else "wb") as f:
                t0 = last = time.time(); got = 0
                while chunk := r.read(4 << 20):
                    f.write(chunk); got += len(chunk)
                    if time.time() - last > 60:
                        last = time.time()
                        print(f"{name}: {(done+got)/1e9:.1f}/{size/1e9:.1f} GB {got/(last-t0)/1e6:.0f} MB/s", flush=True)
            if os.path.getsize(tmp) == size: break
            print(name, "size mismatch, retry", flush=True)
        except Exception as e:
            print(name, "ERROR", e, "- retry in 20s", flush=True); time.sleep(20)
    os.replace(tmp, path); print(name, "OK", flush=True)
print("ALL DONE", flush=True)
