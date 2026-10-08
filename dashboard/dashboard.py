"""Strata LAN dashboard: one page with the engine's speed, caches, memory, power/energy/cost and SSD traffic.

Power at the wall is estimated (no meter): rest of the system (idle measurement minus GPU idle) + GPU power (NVML)
+ an optional CPU load share; the price and these values are set on the page (settings.json).

Reads the Strata server's /metrics (127.0.0.1:8080 with the API key, else a bench engine on 8090), the disks (psutil),
Windows commit (GetPerformanceInfo) and the engine log's decode timing.  Keeps running totals and peaks across
engine restarts in state.json next to this file.  Read-only towards Strata: it never sends a request that runs the
model.  The API key stays on this PC; the page only gets numbers.

usage: python dashboard.py [--port 8082] [--host 0.0.0.0]
"""
import argparse
import collections
import ctypes
import json
import os
import re
import subprocess
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import psutil

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Strata" / "serve"))
try:
    from telemetry import gpu_reader, _cpu_name     # NVML through ctypes, the same reader Strata's Monitor uses
except Exception:                                   # the dashboard still runs without it
    gpu_reader, _cpu_name = None, lambda: None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # X:\dev\StrataQ38FN
STRATA = ROOT / "Strata"
STATE_FILE = HERE / "state.json"
SETTINGS_FILE = HERE / "settings.json"
# power model (all editable on the page): wall = rest + gpu * psu_factor + cpu_extra * cpu_load,
# rest = wall_idle - gpu_idle * psu_factor  (measured 2026-10-08: 123 W at the socket with the GPU at 28 W)
SETTINGS_DEFAULT = {"price_eur_kwh": 0.30, "wall_idle_w": 123.0, "gpu_idle_w": 28.0, "psu_factor": 1.0,
                    "cpu_extra_w": 0.0}
SETTINGS_RANGE = {"price_eur_kwh": (0, 10), "wall_idle_w": (0, 3000), "gpu_idle_w": (0, 1000), "psu_factor": (0.5, 2),
                  "cpu_extra_w": (0, 1000)}
BUCKETS = ("prefill", "decode", "idle", "offline")   # what the time the dashboard ran was spent on
MAX_TICK_GAP = 15                       # a longer gap between two ticks = sleep/hang; it counts as 1 s
HISTORY = 600                           # 10 minutes at 1 s
SESSION_DIRS = [ROOT / "sessions-x", Path(r"C:\StrataQ38FN-sessions")]
SOURCES = [("http://127.0.0.1:8080", True), ("http://127.0.0.1:8090", False)]


def api_key():
    try:
        return (ROOT / "api_key.txt").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


# ------------------------------------------------------------------------------------------------ system readings
class PERFINFO(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("CommitTotal", ctypes.c_size_t), ("CommitLimit", ctypes.c_size_t),
                ("CommitPeak", ctypes.c_size_t), ("PhysicalTotal", ctypes.c_size_t),
                ("PhysicalAvailable", ctypes.c_size_t), ("SystemCache", ctypes.c_size_t),
                ("KernelTotal", ctypes.c_size_t), ("KernelPaged", ctypes.c_size_t),
                ("KernelNonpaged", ctypes.c_size_t), ("PageSize", ctypes.c_size_t),
                ("HandleCount", ctypes.c_ulong), ("ProcessCount", ctypes.c_ulong), ("ThreadCount", ctypes.c_ulong)]


def commit_info():
    pi = PERFINFO()
    pi.cb = ctypes.sizeof(pi)
    ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(pi), pi.cb)
    p = pi.PageSize
    return {"commit_used": pi.CommitTotal * p, "commit_limit": pi.CommitLimit * p,
            "phys_total": pi.PhysicalTotal * p, "phys_avail": pi.PhysicalAvailable * p,
            "system_cache": pi.SystemCache * p}


def disk_letters():
    """PhysicalDriveN -> 'C:' etc. (one PowerShell call at start)."""
    out = {}
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command",
                            "Get-Partition | Where-Object DriveLetter | ForEach-Object { \"$($_.DiskNumber) $($_.DriveLetter)\" }"],
                           capture_output=True, text=True, timeout=20)
        for line in r.stdout.split("\n"):
            parts = line.split()
            if len(parts) == 2:
                out.setdefault(f"PhysicalDrive{parts[0]}", []).append(parts[1] + ":")
    except Exception:
        pass
    return {k: "+".join(sorted(v)) for k, v in out.items()}


def strata_processes():
    """The biggest strata.exe (memory), and the I/O counters of every Strata process: 'engine' = strata.exe (expert
    and PLE reads go through ReadFile + NO_BUFFERING, so they are real SSD reads), 'server' = the Python server and the
    vLLM facade (mostly pipe, socket and log traffic)."""
    best, io = None, {}
    for p in psutil.process_iter(["name", "cmdline"]):
        name = (p.info["name"] or "").lower()
        try:
            if name == "strata.exe":
                mi = p.memory_info()
                if best is None or mi.private > best["private"]:
                    best = {"pid": p.pid, "private": mi.private, "working_set": mi.rss}
                kind = "engine"
            elif name in ("python.exe", "pythonw.exe"):
                cmd = " ".join(p.info["cmdline"] or [])
                if not (("server.py" in cmd and "--engine" in cmd) or "vllm_facade.py" in cmd):
                    continue
                kind = "server"
            else:
                continue
            c = p.io_counters()
            io[(p.pid, p.create_time())] = (kind, c.read_bytes, c.write_bytes)
        except (psutil.Error, OSError):
            pass
    return best, io


def session_files():
    files, total = 0, 0
    for d in SESSION_DIRS:
        try:
            for f in d.iterdir():
                if f.is_file():
                    files += 1
                    total += f.stat().st_size
        except OSError:
            pass
    return {"files": files, "bytes": total, "dirs": [str(d) for d in SESSION_DIRS if d.exists()]}


RE_TIMING = re.compile(r"decode timing: (\d+) windows, avg T ([\d.]+), ([\d.]+) tokens/window, ([\d.]+) ms/window = "
                       r"verify ([\d.]+) \(GPU-reach wait ([\d.]+) \+ per-layer host ([\d.]+) \[.*?CPU ([\d.]+)\] \+ stage ([\d.]+)\)"
                       r".*?draft ([\d.]+)")
RE_SUFFIX = re.compile(r"suffix drafts: (\d+) windows, (\d+) of (\d+) drafts accepted")


def newest_engine_log():
    cands = list(STRATA.glob("strata-*.log")) + list((ROOT / "logs").glob("engine-*.log"))
    cands = [c for c in cands if c.exists()]
    return max(cands, key=lambda c: c.stat().st_mtime) if cands else None


def tail_lines(path, max_bytes=200_000):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            n = f.tell()
            f.seek(max(0, n - max_bytes))
            return f.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return []


# ------------------------------------------------------------------------------------------------ collector
class Collector(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        self.key = api_key()
        self.letters = disk_letters()
        self.hist = collections.defaultdict(lambda: collections.deque(maxlen=HISTORY))
        self.snapshot = {}
        self.state = self.load_state()
        self.prev_disk = None
        self.prev_t = None
        self.seen_requests = set(self.state.get("seen", []))
        self.last_save = 0
        self.log_cache = {"path": None, "mtime": 0, "timing": None, "suffix": None}
        self.last_m = None
        self.settings = self.load_settings()
        self.reset_box = None
        self.started = time.time()
        self.prev_io = {}                   # (pid, create_time) -> (kind, read, write) of the last tick
        self.prev_io_t = None
        self.gpu = None
        try:
            self.gpu = gpu_reader(0) if gpu_reader else None
            self.gpu_name = self.gpu.name() if self.gpu and self.gpu.ok() else None
        except Exception:
            self.gpu, self.gpu_name = None, None
        self.cpu_name = _cpu_name()
        psutil.cpu_percent(None)

    # ---- persistent totals
    @staticmethod
    def base_state(now):
        return {"since": now, "requests": 0, "prompt_tokens": 0, "reused": 0, "read": 0, "output_tokens": 0,
                "prompt_ms": 0.0, "decode_ms": 0.0, "drafts_offered": 0, "drafts_accepted": 0,
                "hit_weighted": 0.0, "hit_weight": 0.0, "pcie_weighted": 0.0, "engine_file_mb": 0.0,
                "peak_decode": 0.0, "peak_decode_at": None, "peak_prefill": 0.0, "peak_prefill_at": None,
                "peak_power": 0.0, "peak_power_at": None, "longest_prompt": 0,
                "disk": {}, "seen": [],
                # time + energy meter (added 2026-10-08; older state files start it at the first run of this version).
                # Per bucket: seconds, GPU joules, CPU load-seconds (0..1 * s) - the wall energy is a linear function of
                # these, so a changed power setting applies to the whole period.
                "meter_since": now, "meter": {b: {"t": 0.0, "gpu_j": 0.0, "cpu_s": 0.0} for b in BUCKETS},
                # the tokens counted since meter_since (the energy per token must use the same period)
                "meter_tokens": {"prompt": 0, "read": 0, "out": 0, "requests": 0, "prompt_s": 0.0, "decode_s": 0.0},
                # Strata process I/O since meter_since, and the drive totals at meter_since (for the share)
                "strata_io": {"engine": {"read": 0, "write": 0}, "server": {"read": 0, "write": 0}},
                "disk_at_meter": None}

    def load_state(self):
        now = time.time()
        base = self.base_state(now)
        try:
            s = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            base.update(s)
        except (OSError, ValueError):
            pass
        if base.get("disk_at_meter") is None:      # older state: the share counts from now on
            base["disk_at_meter"] = {k: dict(v) for k, v in base["disk"].items()}
        return base

    @staticmethod
    def _key_time(key):
        try:
            return float(key.split("-", 1)[0])
        except ValueError:
            return 0.0

    def save_state(self):
        # the newest keys by request time (the engine lists up to 500 recent requests; a set has no order, and a
        # dropped key that the engine still lists would be counted again after a restart)
        newest = sorted(self.seen_requests, key=self._key_time)[-1000:]
        self.seen_requests = set(newest)
        self.state["seen"] = newest
        tmp = STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state), encoding="utf-8")
        os.replace(tmp, STATE_FILE)

    def request_reset(self, timeout=10):
        """Called from an HTTP thread: the collector thread does the reset between two ticks (it owns self.state)."""
        box = {"event": threading.Event(), "result": None}
        self.reset_box = box
        return box["result"] if box["event"].wait(timeout) else None

    def do_reset(self):
        """'new period': archive state.json as state.<date>_<time>.json and start every total at zero.  The requests
        already seen stay seen, so the engine's list of recent requests is not counted again."""
        self.save_state()
        archive = HERE / f"state.{time.strftime('%Y-%m-%d_%H%M%S')}.json"
        archive.write_bytes(STATE_FILE.read_bytes())
        seen = self.state.get("seen", [])
        self.state = self.base_state(time.time())
        self.state["seen"] = seen
        self.state["disk_at_meter"] = {}
        self.save_state()
        self.last_save = time.time()
        return archive.name

    def load_settings(self):
        s = dict(SETTINGS_DEFAULT)
        try:
            for k, v in json.loads(SETTINGS_FILE.read_text(encoding="utf-8")).items():
                if k in SETTINGS_DEFAULT:
                    try:
                        v = float(v)
                    except (ValueError, TypeError):
                        continue
                    lo, hi = SETTINGS_RANGE[k]
                    if lo <= v <= hi:
                        s[k] = v
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        return s

    def update_settings(self, new):
        """Validates and stores the power/price settings; returns the stored settings or raises ValueError."""
        s = dict(self.settings)
        for k, v in new.items():
            if k not in SETTINGS_DEFAULT:
                continue
            v = float(v)
            lo, hi = SETTINGS_RANGE[k]
            if not (lo <= v <= hi):
                raise ValueError(f"{k} must be between {lo} and {hi}")
            s[k] = v
        tmp = SETTINGS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(s, indent=1), encoding="utf-8")
        os.replace(tmp, SETTINGS_FILE)
        self.settings = s
        return s

    # ---- sources
    def fetch_metrics(self):
        for base, needs_key in SOURCES:
            req = urllib.request.Request(base + "/metrics")
            if needs_key and self.key:
                req.add_header("Authorization", "Bearer " + self.key)
            try:
                with urllib.request.urlopen(req, timeout=2) as r:
                    m = json.load(r)
                m["_source"] = base
                return m
            except Exception:
                continue
        return None

    def account_requests(self, m):
        st = self.state
        for r in m.get("requests") or []:
            key = f"{r.get('time')}-{r.get('prompt_tokens')}-{r.get('output_tokens')}"
            if key in self.seen_requests:
                continue
            self.seen_requests.add(key)
            st["requests"] += 1
            pt, reused = r.get("prompt_tokens") or 0, r.get("reused") or 0
            read = r.get("prompt_read") if r.get("prompt_read") is not None else max(0, pt - reused)
            st["prompt_tokens"] += pt
            st["reused"] += reused
            st["read"] += read or 0
            st["output_tokens"] += r.get("output_tokens") or 0
            # the meter only counts requests that finished while it was measuring (not ones from before meter_since or
            # from while this dashboard was not running), so tokens and energy cover the same time
            if (r.get("time") or 0) >= max(st["meter_since"], self.started):
                mt = st["meter_tokens"]
                mt["requests"] += 1
                mt["prompt"] += pt
                mt["read"] += read or 0
                mt["out"] += r.get("output_tokens") or 0
                mt["prompt_s"] = mt.get("prompt_s", 0.0) + (r.get("prompt_ms") or 0) / 1000
                mt["decode_s"] = mt.get("decode_s", 0.0) + (r.get("decode_ms") or 0) / 1000
            st["prompt_ms"] += r.get("prompt_ms") or 0
            st["decode_ms"] += r.get("decode_ms") or 0
            st["drafts_offered"] += r.get("drafts_offered") or 0
            st["drafts_accepted"] += r.get("drafts_accepted") or 0
            st["engine_file_mb"] += r.get("file_mb") or 0
            st["longest_prompt"] = max(st["longest_prompt"], pt)
            w = r.get("engine_generated") or r.get("output_tokens") or 0
            if r.get("hit_rate") is not None and w:
                st["hit_weighted"] += r["hit_rate"] * w
                st["pcie_weighted"] += (r.get("pcie_share") or 0) * w
                st["hit_weight"] += w
            dts = r.get("decode_tok_s")
            if dts and (r.get("output_tokens") or 0) >= 64 and dts > st["peak_decode"]:
                st["peak_decode"], st["peak_decode_at"] = dts, r.get("time")
            t = r.get("time") or 0
            if dts and (r.get("output_tokens") or 0) >= 16 and t >= (st.get("last_decode") or {}).get("at", 0):
                st["last_decode"] = {"tps": dts, "tokens": r.get("output_tokens"), "at": t}
            if read and read >= 256 and r.get("prompt_ms") and t >= (st.get("last_prefill") or {}).get("at", 0):
                st["last_prefill"] = {"tps": read / (r["prompt_ms"] / 1000), "tokens": read, "at": t}
            if read and read >= 1024 and r.get("prompt_ms"):
                pps = read / (r["prompt_ms"] / 1000)
                if pps > st["peak_prefill"]:
                    st["peak_prefill"], st["peak_prefill_at"] = pps, r.get("time")

    def disk_sample(self, now):
        cur = psutil.disk_io_counters(perdisk=True) or {}
        drives = {}
        if self.prev_disk is not None:
            dt = max(1e-3, now - self.prev_t)
            for name, c in cur.items():
                p = self.prev_disk.get(name)
                if p is None:
                    continue
                dr, dw = max(0, c.read_bytes - p.read_bytes), max(0, c.write_bytes - p.write_bytes)
                label = self.letters.get(name, name)
                tot = self.state["disk"].setdefault(label, {"read": 0, "write": 0})
                tot["read"] += dr
                tot["write"] += dw
                drives[label] = {"read_mbs": dr / dt / 1e6, "write_mbs": dw / dt / 1e6,
                                 "boot_read": c.read_bytes, "boot_write": c.write_bytes,
                                 "total_read": tot["read"], "total_write": tot["write"]}
        self.prev_disk, self.prev_t = cur, now
        return drives

    def log_sample(self):
        path = newest_engine_log()
        lc = self.log_cache
        if path is None:
            return lc
        mt = path.stat().st_mtime
        if path != lc["path"] or mt != lc["mtime"]:
            lc.update(path=path, mtime=mt)
            for line in reversed(tail_lines(path)):
                m = RE_TIMING.search(line)
                if m:
                    g = [float(x) for x in m.groups()]
                    lc["timing"] = {"windows": g[0], "tokens_per_window": g[2], "ms_per_window": g[3],
                                    "verify": g[4], "gpu_wait": g[5], "host": g[6], "cpu_experts": g[7],
                                    "stage": g[8], "draft": g[9]}
                    break
            for line in reversed(tail_lines(path, 50_000)):
                m = RE_SUFFIX.search(line)
                if m:
                    lc["suffix"] = {"windows": int(m.group(1)), "accepted": int(m.group(2)), "offered": int(m.group(3))}
                    break
        return lc

    def io_sample(self, io, now):
        """Strata process I/O: deltas per process (restarts and several processes add up), MB/s now + the totals."""
        tot = self.state["strata_io"]
        rate = {"engine": [0.0, 0.0], "server": [0.0, 0.0]}
        if self.prev_io_t is not None:
            dt = max(1e-3, now - self.prev_io_t)
            for key, (kind, rd, wr) in io.items():
                p = self.prev_io.get(key)
                dr, dw = (rd - p[1], wr - p[2]) if p else (rd, wr)   # a process started since the last tick: all new
                dr, dw = max(0, dr), max(0, dw)
                tot[kind]["read"] += dr
                tot[kind]["write"] += dw
                rate[kind][0] += dr / dt / 1e6
                rate[kind][1] += dw / dt / 1e6
        self.prev_io, self.prev_io_t = io, now
        return {k: {"read_mbs": v[0], "write_mbs": v[1], "total_read": tot[k]["read"], "total_write": tot[k]["write"]}
                for k, v in rate.items()}

    def meter_sample(self, now, bucket, gpu_w, cpu_pct):
        """Adds the time since the last tick to the bucket the engine is in now (1 Hz sampling of its state)."""
        prev = getattr(self, "prev_meter_t", None)
        self.prev_meter_t = now
        if prev is None:
            return
        dt = now - prev
        if dt <= 0:
            return
        if dt > MAX_TICK_GAP:                       # the PC slept or the dashboard hung: not running time
            dt = 1.0
        b = self.state["meter"][bucket]
        b["t"] += dt
        b["gpu_j"] += (gpu_w or 0.0) * dt
        b["cpu_s"] += (cpu_pct or 0.0) / 100 * dt

    def energy_view(self, now, bucket, gpu_w, cpu_pct):
        """Wall power now, energy and cost per bucket, the cost per 1M tokens and the 24 h projection."""
        s, st = self.settings, self.state
        psu, cpu_x = s["psu_factor"], s["cpu_extra_w"]
        rest = max(0.0, s["wall_idle_w"] - s["gpu_idle_w"] * psu)
        wall_j = lambda b: rest * b["t"] + psu * b["gpu_j"] + cpu_x * b["cpu_s"]
        kwh = lambda j: j / 3.6e6
        price = s["price_eur_kwh"]
        mb = st["meter"]
        mt = st["meter_tokens"]
        # The busy time (and its energy) comes from the 1 Hz state sampling, which is unbiased for the total but
        # misses most sub-second prefills.  So the split between prefill and decode is taken from the requests' own
        # timings (prompt_ms / decode_ms, same period), and the sampled busy buckets are shared out by that ratio.
        busy = {f: mb["prefill"][f] + mb["decode"][f] for f in ("t", "gpu_j", "cpu_s")}
        ps, ds = mt.get("prompt_s", 0.0), mt.get("decode_s", 0.0)
        if ps + ds > 0:
            r = ps / (ps + ds)
            split = {"prefill": {f: v * r for f, v in busy.items()}, "decode": {f: v * (1 - r) for f, v in busy.items()}}
        else:
            split = {"prefill": mb["prefill"], "decode": mb["decode"]}
        src = {"prefill": split["prefill"], "decode": split["decode"], "idle": mb["idle"], "offline": mb["offline"]}
        per = {}
        for k in BUCKETS:
            b = src[k]
            j = wall_j(b)
            per[k] = {"t": b["t"], "kwh": kwh(j), "gpu_kwh": kwh(b["gpu_j"]), "eur": kwh(j) * price,
                      "avg_w": j / b["t"] if b["t"] else None, "avg_gpu_w": b["gpu_j"] / b["t"] if b["t"] else None}
        run_t = sum(per[k]["t"] for k in BUCKETS)
        online_t = run_t - per["offline"]["t"]
        busy_t = busy["t"]
        all_j = sum(wall_j(mb[k]) for k in BUCKETS)
        busy_j = wall_j(busy)
        idle_j = wall_j(mb["idle"])             # engine online but not computing
        off_j = wall_j(mb["offline"])           # engine not reachable (stopped, loading, restarting)
        per_m = lambda eur, n: eur / n * 1e6 if n else None
        # online idle energy shared out by the busy-time split (prefill vs decode); offline time is not Strata's
        idle_eur = kwh(idle_j) * price
        share_p = per["prefill"]["t"] / busy_t if busy_t else 0.5
        cpu_now = (cpu_pct or 0.0) / 100
        wall_now = rest + psu * (gpu_w or 0.0) + cpu_x * cpu_now
        # 24 h, all from the same meter period: tokens per running second (= the utilisation so far) and tokens per
        # busy second (= 100 % busy with the same mix of reading and writing); now = full * util
        day = 86400
        proj = {"util": busy_t / run_t if run_t else None,
                "now_in": mt["read"] / run_t * day if run_t else None,
                "now_in_prompt": mt["prompt"] / run_t * day if run_t else None,
                "now_out": mt["out"] / run_t * day if run_t else None,
                "full_in": mt["read"] / busy_t * day if busy_t else None,
                "full_out": mt["out"] / busy_t * day if busy_t else None,
                "decode_only": st["output_tokens"] / (st["decode_ms"] / 1000) * day if st["decode_ms"] else None,
                "kwh_now": kwh(all_j) / run_t * day if run_t else None,
                "kwh_full": kwh(busy_j) / busy_t * day if busy_t else None}
        proj["eur_now"] = proj["kwh_now"] * price if proj["kwh_now"] is not None else None
        proj["eur_full"] = proj["kwh_full"] * price if proj["kwh_full"] is not None else None
        return {
            "settings": s, "rest_w": rest, "bucket": bucket,
            "now": {"wall": wall_now, "gpu": gpu_w, "rest": rest, "cpu_extra": cpu_x * cpu_now},
            "since": st["meter_since"], "tokens_since": st["since"],
            "run_t": run_t, "online_t": online_t, "busy_t": busy_t, "per": per,
            "kwh": kwh(all_j), "gpu_kwh": sum(per[k]["gpu_kwh"] for k in BUCKETS), "eur": kwh(all_j) * price,
            "busy_kwh": kwh(busy_j), "idle_kwh": kwh(idle_j), "offline_kwh": kwh(off_j),
            "avg_w": all_j / run_t if run_t else None,
            "avg_gpu_w": sum(mb[k]["gpu_j"] for k in BUCKETS) / run_t if run_t else None,
            "avg_busy_w": busy_j / busy_t if busy_t else None,
            "avg_idle_w": idle_j / mb["idle"]["t"] if mb["idle"]["t"] else None,
            "avg_offline_w": off_j / mb["offline"]["t"] if mb["offline"]["t"] else None,
            "tokens": dict(mt),
            # active only: the energy while reading / writing per token read / written
            "eur_per_m_in": per_m(per["prefill"]["eur"], mt["read"]),
            "eur_per_m_out": per_m(per["decode"]["eur"], mt["out"]),
            "eur_per_m_prompt": per_m(per["prefill"]["eur"], mt["prompt"]),
            # including the online idle time (not engine-offline time), shared out by the busy-time split
            "eur_per_m_in_all": per_m(per["prefill"]["eur"] + idle_eur * share_p, mt["read"]),
            "eur_per_m_out_all": per_m(per["decode"]["eur"] + idle_eur * (1 - share_p), mt["out"]),
            "projection": proj,
        }

    # ---- loop
    def run(self):
        while True:
            t0 = time.time()
            box = self.reset_box
            if box is not None:
                self.reset_box = None
                try:
                    box["result"] = self.do_reset()
                except Exception as e:
                    box["result"] = None
                    with self.lock:
                        self.snapshot["error"] = "reset: " + repr(e)
                box["event"].set()
            try:
                self.tick(t0)
            except Exception as e:  # never die
                with self.lock:
                    self.snapshot["error"] = repr(e)
            time.sleep(max(0.2, 1.0 - (time.time() - t0)))

    def tick(self, now):
        m = self.fetch_metrics()
        drives = self.disk_sample(now)
        com = commit_info()
        proc, io = strata_processes()
        sio = self.io_sample(io, now)
        lc = self.log_sample()
        if m:
            self.account_requests(m)
            self.last_m = m
        shown = m or self.last_m or {}             # offline: keep the last engine facts and requests, marked stale
        hw = {k: v for k, v in (shown.get("hardware") or {}).items() if not k.startswith(("gpu_", "cpu"))}
        g = {}
        try:
            g = self.gpu.read() if self.gpu else {}
        except Exception:
            g = {}
        hw.update(gpu_util=g.get("util"), gpu_mem_used=g.get("mem_used"), gpu_mem_total=g.get("mem_total"),
                  gpu_temp=g.get("temp"), gpu_power=g.get("power"), gpu_power_limit=g.get("power_limit"),
                  gpu_pcie_gen=g.get("pcie_gen"), gpu_pcie_width=g.get("pcie_width"),
                  gpu_pcie_rx_mb=g.get("pcie_rx_mb"), gpu_pcie_tx_mb=g.get("pcie_tx_mb"),
                  cpu=psutil.cpu_percent(None), gpu_name=self.gpu_name, cpu_name=self.cpu_name,
                  cores=psutil.cpu_count(False), threads=psutil.cpu_count(True))
        live = (m or {}).get("live") or {"state": "offline"}
        eng = shown.get("engine") or {}
        st = self.state
        power = hw.get("gpu_power")
        if power and power > st["peak_power"]:
            st["peak_power"], st["peak_power_at"] = power, now
        bucket = ("offline" if m is None else "prefill" if live.get("state") == "reading"
                  else "decode" if live.get("state") == "generating" else "idle")
        self.meter_sample(now, bucket, power, hw.get("cpu"))
        energy = self.energy_view(now, bucket, power, hw.get("cpu"))
        wall = energy["now"]["wall"]
        if wall > st.get("peak_wall", 0):
            st["peak_wall"], st["peak_wall_at"] = wall, now
        h = self.hist
        h["t"].append(now)
        h["decode"].append(live.get("tok_s") if live.get("state") == "generating" else 0)
        h["prefill"].append(live.get("prefill_tok_s_mean") if live.get("state") == "reading" else 0)
        h["power"].append(power)
        h["wall"].append(wall)
        h["strata_r"].append(sio["engine"]["read_mbs"])
        h["strata_w"].append(sio["engine"]["write_mbs"])
        h["gpu_util"].append(hw.get("gpu_util"))
        h["vram_used"].append(hw.get("gpu_mem_used"))
        h["ram_used"].append(com["phys_total"] - com["phys_avail"])
        h["commit_used"].append(com["commit_used"])
        h["cpu"].append(hw.get("cpu"))
        for label, d in drives.items():
            h[f"disk_r_{label}"].append(d["read_mbs"])
            h[f"disk_w_{label}"].append(d["write_mbs"])
        # Strata's share of all SSD traffic since the meter started (both counted only while the dashboard runs)
        base = st.get("disk_at_meter") or {}
        disk_r = sum(v["read"] - (base.get(k) or {}).get("read", 0) for k, v in st["disk"].items())
        disk_w = sum(v["write"] - (base.get(k) or {}).get("write", 0) for k, v in st["disk"].items())
        share_r = min(1.0, sio["engine"]["total_read"] / disk_r) if disk_r > 0 else None
        share_w = min(1.0, sio["engine"]["total_write"] / disk_w) if disk_w > 0 else None
        hit = st["hit_weighted"] / st["hit_weight"] if st["hit_weight"] else None
        pcie = st["pcie_weighted"] / st["hit_weight"] if st["hit_weight"] else None
        reqs = []
        for r in shown.get("requests") or []:
            pt, reused = r.get("prompt_tokens") or 0, r.get("reused") or 0
            read = r.get("prompt_read") if r.get("prompt_read") is not None else max(0, pt - reused)
            reqs.append({"time": r.get("time"), "prompt": pt, "reused": reused, "read": read,
                         "out": r.get("output_tokens"), "prompt_ms": r.get("prompt_ms"),
                         "prefill_tps": (read / (r["prompt_ms"] / 1000)) if read and r.get("prompt_ms") else None,
                         "decode_tps": r.get("decode_tok_s"), "hit": r.get("hit_rate"), "pcie": r.get("pcie_share"),
                         "file_mb": r.get("file_mb"), "finish": r.get("finish"),
                         "accept": (r["drafts_accepted"] / r["drafts_offered"]) if r.get("drafts_offered") else None,
                         "duration": r.get("duration_s")})
        # the newest values from the engine's list too (requests counted before a dashboard restart are skipped above)
        for r in reqs:                              # newest first
            if r["decode_tps"] and (r["out"] or 0) >= 16 and (r["time"] or 0) > (st.get("last_decode") or {}).get("at", 0):
                st["last_decode"] = {"tps": r["decode_tps"], "tokens": r["out"], "at": r["time"]}
            if r["prefill_tps"] and (r["read"] or 0) >= 256 and (r["time"] or 0) > (st.get("last_prefill") or {}).get("at", 0):
                st["last_prefill"] = {"tps": r["prefill_tps"], "tokens": r["read"], "at": r["time"]}
        snap = {
            "now": now,
            "source": (m or {}).get("_source"),
            "online": m is not None,
            "stale_at": None if m else (self.last_m or {}).get("time"),
            "engine": {k: eng.get(k) for k in ("model", "version", "max_context", "kv", "kv_resident", "expert_slots",
                                               "expert_cache_mib", "spec", "mtp_max", "pool_workers", "pcie_frac",
                                               "spec_min_p", "vram_free_mib", "conversation_cache_mib",
                                               "conversation_cache_slots", "arena_mib")},
            "live": live,
            "hardware": hw,
            "conversation_cache": shown.get("conversation_cache"),
            "server_totals": (m or {}).get("totals"),
            "totals": {k: st[k] for k in ("since", "requests", "prompt_tokens", "reused", "read", "output_tokens",
                                          "prompt_ms", "decode_ms", "drafts_offered", "drafts_accepted",
                                          "engine_file_mb", "peak_decode", "peak_decode_at", "peak_prefill",
                                          "peak_prefill_at", "peak_power", "peak_power_at", "longest_prompt")}
                      | {k: st.get(k) for k in ("last_prefill", "last_decode", "peak_wall", "peak_wall_at")},
            "hit_rate": hit, "pcie_share": pcie,
            "memory": {**com, "process": proc},
            "disks": drives,
            "energy": energy,
            "strata_io": {**sio, "share_read": share_r, "share_write": share_w, "procs": len(io)},
            "sessions": session_files(),
            "timing": lc.get("timing"), "suffix": lc.get("suffix"),
            "log": str(lc.get("path")) if lc.get("path") else None,
            "requests": reqs[:12],
        }
        with self.lock:
            self.snapshot = snap
        if now - self.last_save > 15:
            self.save_state()
            self.last_save = now

    def history(self, seconds):
        with self.lock:
            n = min(len(self.hist["t"]), seconds)
            return {k: list(v)[-n:] for k, v in self.hist.items()}

    def get(self):
        with self.lock:
            return dict(self.snapshot)


# ------------------------------------------------------------------------------------------------ HTTP
def make_handler(col):
    page = HERE / "index.html"

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self._send(200, page.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/snapshot":
                self._send(200, json.dumps(col.get()).encode(), "application/json")
            elif path == "/api/history":
                self._send(200, json.dumps(col.history(HISTORY)).encode(), "application/json")
            elif path == "/api/stream":
                # Server-Sent Events: one snapshot per second (+ the newest history point)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                try:
                    while True:
                        data = {"snap": col.get(), "point": col.history(1)}
                        self.wfile.write(b"data: " + json.dumps(data).encode() + b"\n\n")
                        self.wfile.flush()
                        time.sleep(1)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    return
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            # the only writes: the power/price settings and starting a new counting period (the old one is archived)
            path = self.path.split("?")[0]
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = -1
            if not 0 <= n <= 10_000:
                return self._send(413, b"too large", "text/plain")
            body = self.rfile.read(n) if n else b"{}"
            try:
                if path == "/api/settings":
                    data = json.loads(body or b"{}")
                    if not isinstance(data, dict):
                        raise ValueError("expected an object")
                    out = {"settings": col.update_settings(data)}
                elif path == "/api/reset":
                    archive = col.request_reset()
                    if archive is None:
                        return self._send(503, b'{"error": "reset timed out"}', "application/json")
                    out = {"archived": archive}
                else:
                    return self._send(404, b"not found", "text/plain")
            except (ValueError, TypeError) as e:
                return self._send(400, json.dumps({"error": str(e)}).encode(), "application/json")
            except OSError as e:
                return self._send(500, json.dumps({"error": str(e)}).encode(), "application/json")
            self._send(200, json.dumps(out).encode(), "application/json")

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    col = Collector()
    col.start()
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(col))
    srv.daemon_threads = True
    print(f"Strata dashboard on http://{a.host}:{a.port}/  (state: {STATE_FILE})", flush=True)
    try:
        srv.serve_forever()
    finally:
        col.save_state()


if __name__ == "__main__":
    main()
