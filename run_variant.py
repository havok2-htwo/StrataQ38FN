"""A/B runner: start Strata with a config variant, benchmark it, stop it.

usage: python run_variant.py NAME [--base strata-unsloth-ud-iq4_xs.json]
         [--set flag=VALUE ...] [--add flag[=VALUE] ...] [--drop flag ...]  (flags without leading --) [--env K=V ...]
         [--runs 2] [--keep]   (--keep leaves the server running afterwards)
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
STRATA = HERE / "Strata"
PY = STRATA / ".venv" / "Scripts" / "python.exe"
sys.path.insert(0, str(HERE))
import bench_strata  # noqa: E402


def edit_args(args, sets, adds, drops):
    out = list(args)

    def find(flag):
        return out.index(flag) if flag in out else -1

    fix = lambda x: x if x.startswith("--") else "--" + x
    drops = [fix(x) for x in drops]
    sets = [fix(x) for x in sets]
    adds = [fix(x) for x in adds]
    for flag in drops:
        i = find(flag)
        if i >= 0:
            # drop the flag and its value (if the next token is not a flag)
            n = 2 if i + 1 < len(out) and not out[i + 1].startswith("--") else 1
            del out[i:i + n]
    for item in sets:
        flag, val = item.split("=", 1)
        i = find(flag)
        if i >= 0:
            out[i + 1] = val
        else:
            out += [flag, val]
    for item in adds:
        flag, _, val = item.partition("=")
        out += [flag] + ([val] if val else [])
    return out


def kill_port(port):
    r = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True)
    pids = {l.split()[-1] for l in r.stdout.splitlines() if f":{port} " in l and ("LISTENING" in l or "ABH" in l)}
    for pid in pids:
        subprocess.run(["taskkill", "/PID", pid, "/T", "/F"], capture_output=True)
    # the engine itself (in case it outlived the server)
    subprocess.run(["taskkill", "/IM", "strata.exe", "/T", "/F"], capture_output=True)
    time.sleep(3)


def wait_ready(url, proc, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url + "/health", timeout=3) as r:
                if json.load(r).get("loaded"):
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--base", default="strata-unsloth-ud-iq4_xs.json")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--add", action="append", default=[])
    ap.add_argument("--drop", action="append", default=[])
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()

    cfg = json.loads((STRATA / a.base).read_text(encoding="utf-8"))
    cfg["args"] = edit_args(cfg["args"], a.set, a.add, a.drop)
    log = HERE / "logs" / f"engine-{a.name}.log"
    log.parent.mkdir(exist_ok=True)
    if log.exists():
        log.unlink()
    cfg["log"] = str(log)
    cfg_path = HERE / "configs" / f"strata-{a.name}.json"
    cfg_path.parent.mkdir(exist_ok=True)
    cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")

    env = dict(os.environ)
    for kv in a.env:
        k, v = kv.split("=", 1)
        env[k] = v
    env["PYTHONUNBUFFERED"] = "1"

    kill_port(a.port)
    url = f"http://127.0.0.1:{a.port}"
    out = open(HERE / "logs" / f"server-{a.name}.out", "w", encoding="utf-8")
    t0 = time.time()
    proc = subprocess.Popen([str(PY), str(STRATA / "serve" / "server.py"), "--engine", "strata",
                             "--config", str(cfg_path), "--port", str(a.port)],
                            cwd=str(STRATA), env=env, stdout=out, stderr=subprocess.STDOUT)
    print(f"[{a.name}] args: {' '.join(x for x in cfg['args'] if not x.startswith(('X:', 'C:')))}", flush=True)
    print(f"[{a.name}] env: {a.env}", flush=True)
    ok = wait_ready(url, proc)
    print(f"[{a.name}] ready={ok} after {time.time() - t0:.0f}s", flush=True)
    if log.exists():
        keys = re.compile(r"expert cache auto|expert cache \d+ slots|RAM budget|resident RAM mode|"
                          r"file tier reads|borrows|pool workers|WARNING|warning|error|failed", re.I)
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if keys.search(line):
                print("   log:", line[:300], flush=True)
    if not ok:
        print(open(HERE / "logs" / f"server-{a.name}.out", encoding="utf-8", errors="replace").read()[-3000:])
        kill_port(a.port)
        sys.exit(1)
    sys.argv = ["bench_strata.py", a.name, "--url", url, "--runs", str(a.runs)] + (["--only", a.only] if a.only else [])
    bench_strata.main()
    # per-request engine summary lines
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if "generated in" in line or "hit rate" in line or "blob reads" in line:
            print("   eng:", line.replace("strata serve: ", "")[:220], flush=True)
    if not a.keep:
        kill_port(a.port)


if __name__ == "__main__":
    main()
