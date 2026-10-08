"""Resumable download of Unsloth UD-Q4_K_XL shards (pinned revision) + sha256 check."""
import hashlib, os, time, urllib.request
REV = "38bb39ee97821de2c9009abb7e93950eec396e66"
BASE = f"https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/{REV}/UD-Q4_K_XL/"
DIR = r"C:\StrataQ38FN-models\unsloth-UD-Q4_K_XL"
SHARDS = {
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf": (10946624, "4448186216b3af4cc558bbce2c3213f01608f8f8b2e5267a9767971dd3ec8082"),
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf": (49859583136, "3f342f1c1580473f1ee94ddd5b28206e8c07a70fa1a366f59d1d6c922919a6c9"),
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf": (49376141504, "56758f40269cad5cd9b0d3d6fbae0f40f6d5be6de49e4ab392dbe83157d9cbd3"),
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf": (12087983520, "753bda48b98ba4f1636134a90a967de1b2d3908a236c026e464777342e53510a"),
}
def dl(name, size):
    path = os.path.join(DIR, name); tmp = path + ".part"
    if os.path.exists(path) and os.path.getsize(path) == size:
        print(name, "already done", flush=True); return
    while True:
        try:
            done = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            req = urllib.request.Request(BASE + name, headers={"Range": f"bytes={done}-"} if done else {})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "ab" if done and r.status == 206 else "wb") as f:
                if r.status != 206: done = 0
                t0 = last = time.time(); got = 0
                while chunk := r.read(4 << 20):
                    f.write(chunk); got += len(chunk)
                    if time.time() - last > 60:
                        last = time.time()
                        print(f"{name}: {(done+got)/1e9:.1f}/{size/1e9:.1f} GB  {got/(last-t0)/1e6:.0f} MB/s", flush=True)
            if os.path.getsize(tmp) == size: break
        except Exception as e:
            print(name, "ERROR", e, "- retry in 20s", flush=True); time.sleep(20)
    os.replace(tmp, path); print(name, "OK", flush=True)
for n, (s, _) in SHARDS.items(): dl(n, s)
for n, (s, sha) in SHARDS.items():
    h = hashlib.sha256()
    with open(os.path.join(DIR, n), "rb") as f:
        while b := f.read(64 << 20): h.update(b)
    print(n, "SHA OK" if h.hexdigest() == sha else "SHA MISMATCH " + h.hexdigest(), flush=True)
print("ALL DONE", flush=True)
