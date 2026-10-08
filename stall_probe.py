"""Find decode stalls: stream several requests, log token gaps, sample disk/CPU/GPU each 0.5 s.

Assumes a Strata server is already running on --url. Writes logs/stall_probe.jsonl and prints
every gap > --gap-ms together with the system samples around it.
"""
import argparse
import json
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
samples = []
stop = False

PS = (
    "$ErrorActionPreference='SilentlyContinue';"
    "while($true){"
    "$d=Get-CimInstance Win32_PerfFormattedData_PerfDisk_PhysicalDisk | ? Name -ne '_Total';"
    "$c=Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor -Filter \"Name='_Total'\";"
    "$p=Get-CimInstance Win32_PerfFormattedData_PerfProc_Process | ? {$_.PercentProcessorTime -gt 25 -and $_.Name -notin '_Total','Idle'} | Sort PercentProcessorTime -Desc | Select -First 4;"
    "$o=@{t=[DateTimeOffset]::Now.ToUnixTimeMilliseconds()/1000.0; cpu=$c.PercentProcessorTime;"
    "disk=@($d|%{@{n=$_.Name;rd=[int]($_.DiskReadBytesPersec/1MB);wr=[int]($_.DiskWriteBytesPersec/1MB);lat=$_.AvgDisksecPerRead;q=$_.CurrentDiskQueueLength}});"
    "procs=@($p|%{\"$($_.Name)=$($_.PercentProcessorTime)\"})};"
    "($o|ConvertTo-Json -Compress -Depth 4); [Console]::Out.Flush() }"
)


def sampler():
    proc = subprocess.Popen(["powershell", "-NoProfile", "-Command", PS], stdout=subprocess.PIPE, text=True)
    gpu = subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm,power.draw,pcie.link.gen.current",
                            "--format=csv,noheader,nounits", "-lms", "500"], stdout=subprocess.PIPE, text=True)

    def read_gpu():
        for line in gpu.stdout:
            if stop:
                break
            samples.append({"t": time.time(), "gpu": line.strip()})

    threading.Thread(target=read_gpu, daemon=True).start()
    for line in proc.stdout:
        if stop:
            break
        try:
            samples.append(json.loads(line))
        except Exception:
            pass
    proc.kill()
    gpu.kill()


def stream(url, prompt, max_tokens):
    body = json.dumps({"model": "strata", "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0, "reasoning_effort": "none", "max_tokens": max_tokens,
                       "stream": True}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    stamps = []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("data:") and line[5:].strip() != "[DONE]":
                obj = json.loads(line[5:])
                for ch in obj.get("choices", []):
                    if ch.get("delta", {}).get("content"):
                        stamps.append(time.time())
    return stamps


def main():
    global stop
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--gap-ms", type=float, default=150)
    a = ap.parse_args()
    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    time.sleep(3)
    topics = ["the history of the printing press", "how vaccines train the immune system",
              "Rust ownership and borrowing", "the causes of the 2008 financial crisis",
              "how a jet engine works", "die Geschichte der Hanse", "quicksort vs mergesort",
              "how GPS determines position"]
    gaps = []
    for i in range(a.n):
        p = f"Explain {topics[i % len(topics)]} in detail."
        st = stream(a.url, p, 512)
        tps = (len(st) - 1) / (st[-1] - st[0]) if len(st) > 2 else 0
        g = [(st[k - 1], st[k] - st[k - 1]) for k in range(1, len(st)) if st[k] - st[k - 1] > a.gap_ms / 1000]
        print(f"req {i}: {len(st)} chunks, {tps:.1f} chunks/s, gaps>{a.gap_ms:.0f}ms: "
              f"{len(g)} total {sum(x[1] for x in g):.2f}s", flush=True)
        gaps += g
    time.sleep(1)
    stop = True
    with open(HERE / "logs" / "stall_probe.jsonl", "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    for t0, dt in sorted(gaps, key=lambda x: -x[1])[:12]:
        near = [s for s in samples if t0 - 1.0 <= s["t"] <= t0 + dt + 0.5]
        print(f"\nGAP {dt * 1000:.0f} ms at {time.strftime('%H:%M:%S', time.localtime(t0))}")
        for s in near:
            print("   ", json.dumps(s)[:260])


if __name__ == "__main__":
    main()
