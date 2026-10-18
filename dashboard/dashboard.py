"""Strata LAN dashboard: one page with the engine's live requests (answering progress, batch slots), speed, caches,
memory, every GPU (load, VRAM, power, temperature, clocks, PCIe link and bus traffic), power/energy/cost and disk
traffic.  Runs on Windows (RTX 5090) and Linux (4x RTX 3090) with Python's standard library + psutil.

It always shows a page: without a Strata server the GPU, CPU, RAM, disk and PCIe readings stay live and the engine
panels say offline / last seen.  When a server answers, its GET /metrics (configurable sources, by default
127.0.0.1:8080 with the API key from api_key.txt, then 127.0.0.1:8090, plus every running Strata server.py found in
the process list) supplies the live requests, slots and speeds.

Power at the wall is estimated (no meter): rest of the system (idle measurement minus GPU idle) + GPU power (NVML,
all cards) + an optional CPU load share; the price and these values are set on the page (settings file).

GPUs: NVML through ctypes - Strata's serve/telemetry.py gpu_reader when a Strata checkout is importable (--strata),
else a small reader of its own; every card is read on its own, an error on one card does not hide the others.

Read-only towards Strata: the only request it sends is GET /metrics (never one that runs the model).  The API key stays
on this machine; the page only gets numbers.  Everyone sees everything; changes (power/price, new period, GPU power,
fans, the Strata settings and restart - strata_admin.py) need the admin login (the G3 admin password, g3control.py).  Keeps running totals and peaks across engine restarts in a state file.

usage: python dashboard.py [--port 8082] [--host 0.0.0.0] [--source URL[,KEYFILE]]... [--strata PATH]
                           [--log ENGINE_LOG | --engine-config CONFIG.json] [--state FILE] [--settings FILE]
"""
import argparse
import collections
import ctypes
import importlib.util
import json
import os
import platform
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import psutil

import g3control
import strata_admin

IS_WIN = os.name == "nt"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # X:\dev\StrataQ38FN  or  /media/raid0/StrataQ38FN
STRATA = ROOT / "Strata"
STATE_FILE = HERE / "state.json"        # main() may point these elsewhere (--state / --settings)
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
SESSION_DIRS = [ROOT / "sessions-x"] + ([Path(r"C:\StrataQ38FN-sessions")] if IS_WIN else [])
FETCH_TIMEOUT = 1.5                     # per source and tick (refused connections return at once)


def read_key(path):
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def default_sources():
    """(url, key file or None): Strata on 8080 with the API key (the Windows service), then the bench port 8090."""
    return [("http://127.0.0.1:8080", ROOT / "api_key.txt"), ("http://127.0.0.1:8090", None)]


# ------------------------------------------------------------------------------------------------ GPUs (NVML)
class _OwnNvml:
    """NVML through ctypes, the same readings as Strata's serve/telemetry.py _Nvml (used when no Strata checkout is
    importable).  Anything that cannot be read is None."""

    class Util(ctypes.Structure):
        _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

    class Mem(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

    def __init__(self, index=0):
        self.lib = self.dev = None
        names = ["nvml.dll", os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                                          "NVIDIA Corporation", "NVSMI", "nvml.dll")] if IS_WIN \
            else ["libnvidia-ml.so.1", "libnvidia-ml.so"]
        for n in names:
            try:
                self.lib = ctypes.CDLL(n)
                break
            except OSError:
                continue
        if self.lib is None:
            return
        try:
            init = getattr(self.lib, "nvmlInit_v2", None) or self.lib.nvmlInit
            if init() != 0:
                self.lib = None
                return
            h = ctypes.c_void_p()
            get = getattr(self.lib, "nvmlDeviceGetHandleByIndex_v2", None) or self.lib.nvmlDeviceGetHandleByIndex
            if get(ctypes.c_uint(index), ctypes.byref(h)) != 0:
                self.dev = None
                return
            self.dev = h
        except (AttributeError, OSError):
            self.lib = None

    def ok(self):
        return self.lib is not None and self.dev is not None

    def _uint(self, fn, *args):
        v = ctypes.c_uint()
        try:
            return v.value if getattr(self.lib, fn)(self.dev, *args, ctypes.byref(v)) == 0 else None
        except (AttributeError, OSError):
            return None

    def name(self):
        buf = ctypes.create_string_buffer(96)
        try:
            if self.lib.nvmlDeviceGetName(self.dev, buf, ctypes.c_uint(96)) == 0:
                return buf.value.decode(errors="replace")
        except (AttributeError, OSError):
            pass
        return None

    def read(self):
        out = {}
        u = self.Util()
        try:
            if self.lib.nvmlDeviceGetUtilizationRates(self.dev, ctypes.byref(u)) == 0:
                out["util"] = u.gpu
        except (AttributeError, OSError):
            pass
        m = self.Mem()
        try:
            if self.lib.nvmlDeviceGetMemoryInfo(self.dev, ctypes.byref(m)) == 0:
                out["mem_used"], out["mem_total"] = m.used, m.total
        except (AttributeError, OSError):
            pass
        out["temp"] = self._uint("nvmlDeviceGetTemperature", ctypes.c_uint(0))          # NVML_TEMPERATURE_GPU
        mw = self._uint("nvmlDeviceGetPowerUsage")
        out["power"] = mw / 1000.0 if mw is not None else None
        lim = self._uint("nvmlDeviceGetEnforcedPowerLimit")
        out["power_limit"] = lim / 1000.0 if lim is not None else None
        out["pcie_gen"] = self._uint("nvmlDeviceGetCurrPcieLinkGeneration")        # drops at idle (power saving)
        out["pcie_gen_max"] = self._uint("nvmlDeviceGetMaxPcieLinkGeneration")
        out["pcie_width"] = self._uint("nvmlDeviceGetCurrPcieLinkWidth")
        rx = self._uint("nvmlDeviceGetPcieThroughput", ctypes.c_uint(1))                 # NVML_PCIE_UTIL_RX_BYTES, KB/s
        tx = self._uint("nvmlDeviceGetPcieThroughput", ctypes.c_uint(0))
        out["pcie_rx_mb"] = rx / 1024.0 if rx is not None else None                      # telemetry.py's unit
        out["pcie_tx_mb"] = tx / 1024.0 if tx is not None else None
        return out


class _PciInfo(ctypes.Structure):           # nvmlPciInfo_t (v3)
    _fields_ = [("busIdLegacy", ctypes.c_char * 16), ("domain", ctypes.c_uint), ("bus", ctypes.c_uint),
                ("device", ctypes.c_uint), ("pciDeviceId", ctypes.c_uint), ("pciSubSystemId", ctypes.c_uint),
                ("busId", ctypes.c_char * 32)]


def strata_candidates(given):
    """Where Strata's serve/telemetry.py may be: --strata first, then the checkouts next to this repo."""
    out = [Path(given)] if given else []
    return out + [ROOT / "Strata" / "serve", ROOT / "Strata-142" / "serve"]


def load_telemetry(paths):
    """Strata's serve/telemetry.py as a module of its own (no sys.path change), or (None, None)."""
    for p in paths:
        for d in (p, p / "serve"):
            f = d / "telemetry.py"
            if not f.is_file():
                continue
            try:
                spec = importlib.util.spec_from_file_location("strata_telemetry", f)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "gpu_reader"):
                    return mod, str(d)
            except Exception:
                continue
    return None, None


def _cpu_name_own():
    if IS_WIN:
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif os.path.exists("/proc/cpuinfo"):
        try:
            for line in open("/proc/cpuinfo", encoding="utf-8", errors="replace"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or None


class GpuSet:
    """Every NVIDIA card: static facts once (name, PCI bus id, maximum link width and SM clock), the readings each
    tick.  Only read-only NVML queries - safe while benchmarks run on the cards."""

    def __init__(self, strata_path=None):
        self.mod, self.mod_path = load_telemetry(strata_candidates(strata_path))
        self.cards, self.error = [], None
        self.backend = f"Strata telemetry.py ({self.mod_path})" if self.mod else "eigener NVML-Leser"
        make = (lambda i: self.mod.gpu_reader(i)) if self.mod else _OwnNvml
        try:
            first = make(0)
            if getattr(first, "lib", None) is None and self.mod:     # the Strata reader failed: try our own
                first, make, self.backend = _OwnNvml(0), _OwnNvml, "eigener NVML-Leser"
        except Exception as e:
            first, self.error = None, f"NVML: {e!r}"
        lib = getattr(first, "lib", None)
        if lib is None:
            self.error = self.error or "NVML nicht verfügbar (kein NVIDIA-Treiber?)"
            return
        n = ctypes.c_uint()
        try:
            cnt = getattr(lib, "nvmlDeviceGetCount_v2", None) or lib.nvmlDeviceGetCount
            count = n.value if cnt(ctypes.byref(n)) == 0 else 1
        except (AttributeError, OSError):
            count = 1
        for i in range(count):
            card = {"index": i, "reader": None, "name": None, "bus": None, "pcie_width_max": None,
                    "sm_clock_max": None, "error": None}
            try:
                r = first if i == 0 else make(i)
                card["reader"] = r
                if not r.ok():
                    card["error"] = "NVML: kein Handle für diese Karte"
                else:
                    card["name"] = r.name()
                    card["pcie_width_max"] = r._uint("nvmlDeviceGetMaxPcieLinkWidth")
                    card["sm_clock_max"] = r._uint("nvmlDeviceGetMaxClockInfo", ctypes.c_uint(1))   # NVML_CLOCK_SM
                    pi = _PciInfo()
                    try:                            # the bus id is a nice-to-have: never an error for the card
                        fn = getattr(r.lib, "nvmlDeviceGetPciInfo_v3", None) or getattr(r.lib, "nvmlDeviceGetPciInfo_v2")
                        if fn(r.dev, ctypes.byref(pi)) == 0:
                            card["bus"] = pi.busId.decode(errors="replace").lower()
                    except (AttributeError, OSError):
                        pass
            except Exception as e:                  # one card's error does not hide the others
                card["error"] = card["error"] or f"NVML: {e!r}"
            self.cards.append(card)

    def names(self):
        return [c["name"] for c in self.cards if c["name"]]

    def read(self):
        out = []
        for c in self.cards:
            g = {"index": c["index"], "name": c["name"], "bus": c["bus"], "error": c["error"],
                 "pcie_width_max": c["pcie_width_max"], "sm_clock_max": c["sm_clock_max"]}
            r = c["reader"]
            if r is not None and c["error"] is None:
                try:
                    d = r.read() or {}
                    for k in ("util", "mem_used", "mem_total", "temp", "power", "power_limit", "pcie_gen",
                              "pcie_gen_max", "pcie_width"):
                        g[k] = d.get(k)
                    # NVML gives KB/s; telemetry.py divides by 1024 - here decimal MB/s (KB/s / 1000)
                    for k in ("rx", "tx"):
                        v = d.get(f"pcie_{k}_mb")
                        g[f"pcie_{k}_mb"] = v * 1024 / 1000 if v is not None else None
                    g["sm_clock"] = r._uint("nvmlDeviceGetClockInfo", ctypes.c_uint(1))       # NVML_CLOCK_SM, MHz
                    g["mem_clock"] = r._uint("nvmlDeviceGetClockInfo", ctypes.c_uint(2))      # NVML_CLOCK_MEM
                    if all(g.get(k) is None for k in ("util", "mem_used", "temp", "power")):
                        g["error"] = "NVML liefert keine Werte (Karte verloren?)"
                except Exception as e:
                    g["error"] = f"NVML: {e!r}"
            out.append(g)
        return out


def gpu_totals(gpus):
    """The single-card fields the page always had: sums (power, VRAM, PCIe), mean load, hottest card."""
    ok = [g for g in gpus if not g.get("error")]

    def tot(k):
        v = [g[k] for g in ok if g.get(k) is not None]
        return sum(v) if v else None

    util = [g["util"] for g in ok if g.get("util") is not None]
    temps = [g["temp"] for g in ok if g.get("temp") is not None]
    names = collections.Counter(g["name"] for g in gpus if g.get("name"))
    name = " + ".join(f"{n}× {k}" if n > 1 else k for k, n in names.items()) or None
    first = ok[0] if ok else {}
    return {"gpu_util": sum(util) / len(util) if util else None, "gpu_mem_used": tot("mem_used"),
            "gpu_mem_total": tot("mem_total"), "gpu_temp": max(temps) if temps else None, "gpu_power": tot("power"),
            "gpu_power_limit": tot("power_limit"), "gpu_pcie_gen": first.get("pcie_gen"),
            "gpu_pcie_gen_max": first.get("pcie_gen_max"), "gpu_pcie_width": first.get("pcie_width"),
            "gpu_pcie_rx_mb": tot("pcie_rx_mb"), "gpu_pcie_tx_mb": tot("pcie_tx_mb"), "gpu_name": name,
            "gpu_count": len(gpus), "gpu_ok": len(ok)}


# ------------------------------------------------------------------------------------------------ system readings
class PERFINFO(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("CommitTotal", ctypes.c_size_t), ("CommitLimit", ctypes.c_size_t),
                ("CommitPeak", ctypes.c_size_t), ("PhysicalTotal", ctypes.c_size_t),
                ("PhysicalAvailable", ctypes.c_size_t), ("SystemCache", ctypes.c_size_t),
                ("KernelTotal", ctypes.c_size_t), ("KernelPaged", ctypes.c_size_t),
                ("KernelNonpaged", ctypes.c_size_t), ("PageSize", ctypes.c_size_t),
                ("HandleCount", ctypes.c_ulong), ("ProcessCount", ctypes.c_ulong), ("ThreadCount", ctypes.c_ulong)]


def _overcommit_mode():
    try:
        return int(Path("/proc/sys/vm/overcommit_memory").read_text().strip())
    except (OSError, ValueError):
        return None


OVERCOMMIT = None if IS_WIN else _overcommit_mode()


def commit_info():
    """Windows: GetPerformanceInfo.  Linux: /proc/meminfo (Committed_AS / CommitLimit - only a limit with
    vm.overcommit_memory = 2) and psutil's available memory; the page labels it accordingly."""
    if IS_WIN:
        pi = PERFINFO()
        pi.cb = ctypes.sizeof(pi)
        ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(pi), pi.cb)
        p = pi.PageSize
        return {"commit_used": pi.CommitTotal * p, "commit_limit": pi.CommitLimit * p,
                "phys_total": pi.PhysicalTotal * p, "phys_avail": pi.PhysicalAvailable * p,
                "system_cache": pi.SystemCache * p, "os": "windows", "commit_enforced": True}
    mi = {}
    try:
        for line in open("/proc/meminfo", encoding="ascii", errors="replace"):
            k, _, v = line.partition(":")
            parts = v.split()
            if parts:
                mi[k] = int(parts[0]) * 1024
    except (OSError, ValueError):
        pass
    vm = psutil.virtual_memory()
    return {"commit_used": mi.get("Committed_AS"), "commit_limit": mi.get("CommitLimit"),
            "phys_total": vm.total, "phys_avail": vm.available,
            "system_cache": (mi.get("Cached") or 0) + (mi.get("Buffers") or 0),
            "swap_used": (mi.get("SwapTotal") or 0) - (mi.get("SwapFree") or 0), "swap_total": mi.get("SwapTotal"),
            "os": "linux", "commit_enforced": OVERCOMMIT == 2}


def disk_letters():
    """Windows: PhysicalDriveN -> 'C:' etc. (one PowerShell call at start).
    Linux: the whole block devices worth showing -> 'nvme0n1 /data': loop/ram/optical devices are skipped, and so
    is every disk that belongs to a RAID / device-mapper volume (the md/dm device is shown instead), so no byte is
    counted twice."""
    out = {}
    if IS_WIN:
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
    mounts = {}
    try:
        for p in psutil.disk_partitions(all=False):
            mounts.setdefault(os.path.basename(os.path.realpath(p.device)), []).append(p.mountpoint)
    except Exception:
        pass
    sb = Path("/sys/block")
    try:
        devs = sorted(os.listdir(sb))
    except OSError:
        return {}
    for dev in devs:
        if dev.startswith(("loop", "ram", "zram", "sr", "fd", "nbd")):
            continue
        try:
            subs = [s for s in os.listdir(sb / dev) if s.startswith(dev)]          # its partitions
            if any(os.listdir(sb / dev / x / "holders") for x in ["."] + subs):     # part of md/dm: shown there
                continue
        except OSError:
            continue
        mps = sorted(set(mounts.get(dev, []) + [m for s in subs for m in mounts.get(s, [])]))
        out[dev] = dev + (" " + ",".join(mps) if mps else "")
    return out


_PROC_CACHE = {"at": 0.0, "procs": []}       # [(psutil.Process, kind, server info or None)]
PROC_RESCAN_S = 5                               # a full process scan costs ~0.1 s CPU with ~1,500 processes


def _scan_processes():
    """Every Strata engine, server and facade process (name first: the command line only of Python processes)."""
    found = []
    for p in psutil.process_iter(["name"]):
        name = (p.info["name"] or "").lower()
        try:
            if name == "strata.exe" or (not IS_WIN and name.startswith("strata")):
                found.append((p, "engine", None))
            elif name.startswith("python"):
                cmdl = p.cmdline()
                cmd = " ".join(cmdl)
                if "server.py" in cmd and "--engine" in cmd:
                    try:
                        cwd = p.cwd()
                    except (psutil.Error, OSError):
                        cwd = None
                    found.append((p, "server", {"pid": p.pid, "cmdline": list(cmdl), "cwd": cwd}))
                elif "vllm_facade.py" in cmd:
                    found.append((p, "server", None))
        except (psutil.Error, OSError):
            pass
    return found


def strata_processes(now=None):
    """The biggest engine process (memory), the I/O counters of every Strata process, and the running Strata servers.
    'engine' = strata.exe on Windows / strata* on Linux (expert and PLE reads bypass the cache, so they are real SSD
    reads), 'server' = the Python server and the vLLM facade (mostly pipe, socket and log traffic).  The process list
    is scanned every PROC_RESCAN_S seconds; in between only the known processes are read."""
    now = time.time() if now is None else now
    pc = _PROC_CACHE
    if now - pc["at"] >= PROC_RESCAN_S or any(not p.is_running() for p, _, _ in pc["procs"]):
        pc["procs"], pc["at"] = _scan_processes(), now
    best, io, servers = None, {}, []
    for p, kind, info in pc["procs"]:
        try:
            if kind == "engine":
                mi = p.memory_info()
                private = getattr(mi, "private", None)
                if private is None:                         # Linux: resident minus shared (file-backed) pages
                    private = mi.rss - getattr(mi, "shared", 0)
                if best is None or private > best["private"]:
                    best = {"pid": p.pid, "name": p.name(), "private": private, "working_set": mi.rss}
            elif info:
                servers.append(info)
            c = p.io_counters()
            io[(p.pid, p.create_time())] = (kind, c.read_bytes, c.write_bytes)
        except (psutil.Error, OSError):
            pass
    return best, io, servers


def _arg(argv, flag):
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


_CONF_CACHE = {}


def read_config(path):
    """A Strata run config (JSON), cached by mtime; {} when unreadable."""
    try:
        mt = path.stat().st_mtime
    except OSError:
        return {}
    c = _CONF_CACHE.get(str(path))
    if c and c[0] == mt:
        return c[1]
    try:
        j = json.loads(path.read_text(encoding="utf-8"))
        j = j if isinstance(j, dict) else {}
    except (OSError, ValueError):
        j = {}
    _CONF_CACHE[str(path)] = (mt, j)
    return j


def config_log(conf_path, base=None):
    """The engine log a run config names ("log"), absolute."""
    j = read_config(conf_path)
    log = j.get("log")
    if not log:
        return None
    p = Path(log)
    if not p.is_absolute():
        p = Path(j.get("cwd") or base or conf_path.parent) / p
    return p


def server_info(servers):
    """The running Strata servers from their command lines: the URL to ask and the engine log of their config."""
    out = []
    for s in servers:
        argv, cwd = s["cmdline"], s["cwd"]
        conf = _arg(argv, "--config")
        cp = None
        if conf:
            cp = Path(conf)
            if not cp.is_absolute() and cwd:
                cp = Path(cwd) / cp
        port = _arg(argv, "--port") or (read_config(cp).get("port") if cp else None) or 8080
        host = _arg(argv, "--host") or "127.0.0.1"
        if host in ("0.0.0.0", "::", ""):
            host = "127.0.0.1"
        try:
            url = f"http://{host}:{int(port)}"
        except (TypeError, ValueError):
            url = None
        out.append({"pid": s["pid"], "url": url, "config": str(cp) if cp else None,
                    "log": config_log(cp, cwd) if cp else None})
    return out


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
RE_PARK = re.compile(r"conversation cache: parked (\d+) tokens .*snapshot_bytes=(\d+)")


def newest_glob_log():
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
    def __init__(self, sources=None, strata_path=None, log_path=None, engine_config=None, discover=True):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        # configured sources: (url, api key or "") - the key never leaves this process
        self.sources = [(u.rstrip("/"), read_key(k) if k else "") for u, k in (sources or default_sources())]
        self.discover = discover
        self.good_source = None                 # the source that answered last: asked first
        self.source_status = []
        self.servers = []
        self.log_path = Path(log_path) if log_path else None
        self.engine_config = Path(engine_config) if engine_config else None
        self.letters = disk_letters()
        self.hist = collections.defaultdict(lambda: collections.deque(maxlen=HISTORY))
        self.snapshot = {}
        self.state = self.load_state()
        self.prev_disk = None
        self.prev_t = None
        self.seen_requests = set(self.state.get("seen", []))
        self.last_save = 0
        self.log_cache = {"path": None, "mtime": 0, "parsed_at": 0, "timing": None, "suffix": None}
        self.last_m = None
        self.last_seen = None                   # when a Strata server last answered
        self.settings = self.load_settings()
        self.reset_box = None
        self.started = time.time()
        self.prev_io = {}                   # (pid, create_time) -> (kind, read, write) of the last tick
        self.prev_io_t = None
        try:
            self.gpus = GpuSet(strata_path)
        except Exception as e:              # never without a page
            self.gpus = None
            self.gpu_error = repr(e)
        else:
            self.gpu_error = self.gpus.error
        mod = self.gpus.mod if self.gpus else None
        try:
            self.cpu_name = mod._cpu_name() if mod and hasattr(mod, "_cpu_name") else _cpu_name_own()
        except Exception:
            self.cpu_name = _cpu_name_own()
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
        archive = STATE_FILE.with_name(f"{STATE_FILE.stem}.{time.strftime('%Y-%m-%d_%H%M%S')}.json")
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
    def candidates(self):
        """The configured sources, then the running servers found in the process list (no key: a server that needs
        one shows up as 'needs API key'); the one that answered last goes first."""
        seen, out = set(), []
        for url, key in self.sources:
            if url not in seen:
                seen.add(url)
                out.append({"url": url, "key": key, "origin": "config"})
        if self.discover:
            for s in self.servers:
                if s.get("url") and s["url"] not in seen:
                    seen.add(s["url"])
                    out.append({"url": s["url"], "key": "", "origin": "process"})
        out.sort(key=lambda c: c["url"] != self.good_source)
        return out

    def fetch_metrics(self):
        """GET /metrics of the first source that is a Strata server (JSON with "live" and "engine"); read-only.
        Returns (metrics or None, the status of every source tried - without keys)."""
        status = []
        for c in self.candidates():
            st = {"url": c["url"], "origin": c["origin"], "key": bool(c["key"]), "state": "offline"}
            status.append(st)
            req = urllib.request.Request(c["url"] + "/metrics", headers={"Accept": "application/json"})
            if c["key"]:
                req.add_header("Authorization", "Bearer " + c["key"])
            try:
                with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as r:
                    body = r.read(16_000_000)
                m = json.loads(body)
                if not isinstance(m, dict) or "live" not in m or "engine" not in m:
                    st["state"] = "kein Strata-Server"
                    continue
            except urllib.error.HTTPError as e:
                st["state"] = "API-Key nötig" if e.code == 401 else f"HTTP {e.code}"
                continue
            except (ValueError, UnicodeDecodeError):
                st["state"] = "kein Strata-Server"
                continue
            except Exception:
                continue
            st["state"] = "online"
            m["_source"] = c["url"]
            self.good_source = c["url"]
            return m, status
        return None, status

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
        if not IS_WIN:                              # whole disks and md/dm volumes only (no partitions, no members)
            cur = {k: v for k, v in cur.items() if k in self.letters}
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

    def engine_log(self):
        """--log, else the log of a running server's config, else --engine-config's, else the newest Strata log."""
        if self.log_path:
            return self.log_path
        logs = [s["log"] for s in self.servers if s.get("log") and s["log"].exists()]
        if logs:
            return max(logs, key=lambda p: p.stat().st_mtime)
        if self.engine_config:
            p = config_log(self.engine_config)
            if p and p.exists():
                return p
        return newest_glob_log()

    def log_sample(self, now):
        path = self.engine_log()
        lc = self.log_cache
        if path is None:
            return lc
        try:
            mt = path.stat().st_mtime
        except OSError:
            return lc
        # a busy engine writes its log all the time: parse at most every 5 s
        if path != lc["path"]:                      # another engine log: nothing of the old one carries over
            lc.update(timing=None, suffix=None, park_bpt=None)
        if path != lc["path"] or (mt != lc["mtime"] and now - lc["parsed_at"] >= 5):
            lc.update(path=path, mtime=mt, parsed_at=now)
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
            # parked conversations: bytes per token of the last 200 parks (Σ bytes / Σ tokens) -> token estimate in the UI
            parks = [m.groups() for m in map(RE_PARK.search, tail_lines(path, 2_000_000)) if m][-200:]
            tok = sum(int(t) for t, _ in parks)
            if tok:
                lc["park_bpt"] = {"bytes_per_token": sum(int(b) for _, b in parks) / tok, "samples": len(parks)}
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

    @staticmethod
    def batch_view(live):
        """--batch: the slots summed up (decode rate of all slots, how many read / decode / idle)."""
        slots = live.get("slots") or []
        if not slots:
            return None
        busy = [s for s in slots if s.get("state") != "idle"]
        return {"slots": len(slots), "busy": len(busy),
                "reading": sum(1 for s in busy if s.get("state") == "reading"),
                "decoding": sum(1 for s in busy if s.get("state") != "reading"),
                "tok_s_sum": sum(s.get("tok_s") or 0 for s in busy),
                "generated_sum": sum(s.get("generated") or 0 for s in busy),
                "held_tokens": sum(s.get("held_tokens") or 0 for s in slots if s.get("state") == "idle")}

    def tick(self, now):
        proc, io, servers = strata_processes(now)
        self.servers = server_info(servers)
        m, status = self.fetch_metrics()
        self.source_status = status
        drives = self.disk_sample(now)
        com = commit_info()
        sio = self.io_sample(io, now)
        lc = self.log_sample(now)
        if m:
            self.account_requests(m)
            self.last_m = m
            self.last_seen = now
        shown = m or self.last_m or {}             # offline: keep the last engine facts and requests, marked stale
        hw = {k: v for k, v in (shown.get("hardware") or {}).items()
              if not k.startswith(("gpu", "cpu")) and k != "gpus"}
        try:
            gpus = self.gpus.read() if self.gpus else []
        except Exception as e:
            gpus, self.gpu_error = [], repr(e)
        hw.update(gpu_totals(gpus))
        hw.update(cpu=psutil.cpu_percent(None), cpu_name=self.cpu_name,
                  cores=psutil.cpu_count(False), threads=psutil.cpu_count(True))
        live = (m or {}).get("live") or {"state": "offline"}
        batch = self.batch_view(live) if m else None
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
        # decode: with --batch all slots together, else the running request
        if batch:
            h["decode"].append(batch["tok_s_sum"])
        else:
            h["decode"].append(live.get("tok_s") if live.get("state") == "generating" else 0)
        h["prefill"].append(live.get("prefill_tok_s_mean") if live.get("state") == "reading" else 0)
        h["slots_busy"].append(batch["busy"] if batch else None)
        h["power"].append(power)
        h["wall"].append(wall)
        h["strata_r"].append(sio["engine"]["read_mbs"])
        h["strata_w"].append(sio["engine"]["write_mbs"])
        h["gpu_util"].append(hw.get("gpu_util"))
        h["vram_used"].append(hw.get("gpu_mem_used"))
        h["pcie_rx"].append(hw.get("gpu_pcie_rx_mb"))
        h["pcie_tx"].append(hw.get("gpu_pcie_tx_mb"))
        for g in gpus:
            i = g["index"]
            h[f"gpu{i}_util"].append(g.get("util"))
            h[f"gpu{i}_power"].append(g.get("power"))
            h[f"gpu{i}_rx"].append(g.get("pcie_rx_mb"))
            h[f"gpu{i}_tx"].append(g.get("pcie_tx_mb"))
            h[f"gpu{i}_temp"].append(g.get("temp"))
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
            "platform": "windows" if IS_WIN else "linux",
            "source": (m or {}).get("_source"),
            "sources": status,
            "servers": [{k: (str(v) if isinstance(v, Path) else v) for k, v in s.items()} for s in self.servers],
            "online": m is not None,
            "stale_at": None if m else (self.last_m or {}).get("time"),
            "last_seen": self.last_seen,
            "engine": {k: eng.get(k) for k in ("model", "version", "max_context", "kv", "kv_resident", "expert_slots",
                                               "expert_cache_mib", "spec", "mtp_max", "pool_workers", "pcie_frac",
                                               "spec_min_p", "vram_free_mib", "conversation_cache_mib",
                                               "conversation_cache_slots", "arena_mib", "batch_slots")},
            "live": live,
            "batch": batch,
            "hardware": hw,
            "gpus": gpus,
            "gpu_backend": self.gpus.backend if self.gpus else None,
            "gpu_error": self.gpu_error,
            "conversation_cache": ({**shown["conversation_cache"], "park_bpt": lc.get("park_bpt")}
                                   if isinstance(shown.get("conversation_cache"), dict) else shown.get("conversation_cache")),
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
def make_handler(col, ctl=None, adm=None):
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

        def _ctl_token(self):
            for part in (self.headers.get("Cookie") or "").split(";"):
                k, _, v = part.strip().partition("=")
                if k == g3control.COOKIE:
                    return v
            return None

        def _same_origin(self):
            """Writes to G3 only from this page: no other site may ride the login (and JSON forces a CORS preflight)."""
            site = (self.headers.get("Sec-Fetch-Site") or "").lower()
            if site and site not in ("same-origin", "none"):
                return False
            origin = self.headers.get("Origin")
            if origin and origin.split("://", 1)[-1].rstrip("/").lower() != (self.headers.get("Host") or "").lower():
                return False
            return (self.headers.get("Content-Type") or "").startswith("application/json")

        def _control(self, path, body):
            if not self._same_origin():
                return self._send(403, b'{"error": "nur von der Monitor-Seite selbst"}', "application/json")
            tok = self._ctl_token()
            try:
                data = json.loads(body or b"{}")
                if path == "/api/control/login":
                    tok = ctl.login(str((data or {}).get("password") or ""))
                    raw = json.dumps({"ok": True}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Set-Cookie", f"{g3control.COOKIE}={tok}; Path=/; HttpOnly; SameSite=Strict; "
                                                   f"Max-Age={g3control.SESSION_S}")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    return self.wfile.write(raw)
                if path == "/api/control/logout":
                    ctl.logout(tok)
                    out = {"ok": True}
                elif path == "/api/control/power":
                    out = ctl.save_power(tok, data)
                elif path == "/api/control/fan":
                    out = ctl.save_fan(tok, data)
                elif path == "/api/control/strata/config" and adm:
                    ctl._need(tok)
                    out = {"saved": adm.save(data)}
                    if data.get("restart"):
                        out["restart"] = adm.restart()
                elif path == "/api/control/strata/restart" and adm:
                    ctl._need(tok)
                    out = adm.restart()
                else:
                    return self._send(404, b"not found", "text/plain")
            except PermissionError as e:
                return self._send(401, json.dumps({"error": str(e)}).encode(), "application/json")
            except ConnectionError as e:
                return self._send(502, json.dumps({"error": str(e)}).encode(), "application/json")
            except (ValueError, TypeError) as e:
                return self._send(400, json.dumps({"error": str(e)}).encode(), "application/json")
            except OSError as e:
                return self._send(500, json.dumps({"error": str(e)}).encode(), "application/json")
            self._send(200, json.dumps(out).encode(), "application/json")

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self._send(200, page.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/snapshot":
                self._send(200, json.dumps(col.get()).encode(), "application/json")
            elif path == "/api/history":
                self._send(200, json.dumps(col.history(HISTORY)).encode(), "application/json")
            elif path == "/api/control" and ctl:
                self._send(200, json.dumps(ctl.view(self._ctl_token())).encode(), "application/json")
            elif path == "/api/strata" and adm:
                # read by everyone; the engine log's tail only with the login
                admin = bool(ctl and ctl.session(self._ctl_token()))
                self._send(200, json.dumps(adm.view(logged_in=admin)).encode(), "application/json")
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
            # the writes: the power/price settings, a new counting period (the old one is archived) and /api/control/*
            # (G3 power/fans, Strata settings/restart) - all of them only with the admin login when there is one (ctl)
            path = self.path.split("?")[0]
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = -1
            if not 0 <= n <= 10_000:
                return self._send(413, b"too large", "text/plain")
            body = self.rfile.read(n) if n else b"{}"
            if path.startswith("/api/control/") and ctl:
                return self._control(path, body)
            if ctl and path in ("/api/settings", "/api/reset"):
                if not self._same_origin():
                    return self._send(403, b'{"error": "nur von der Monitor-Seite selbst"}', "application/json")
                if not ctl.session(self._ctl_token()):
                    return self._send(401, '{"error": "nur nach Anmeldung"}'.encode(), "application/json")
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
    global STATE_FILE, SETTINGS_FILE
    ap = argparse.ArgumentParser(description="Strata LAN dashboard (Windows + Linux)")
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--source", action="append", metavar="URL[,KEYFILE]",
                    help="a Strata server to read GET /metrics from (repeatable, asked in this order); KEYFILE holds "
                         "its API key.  Default: http://127.0.0.1:8080 with ../api_key.txt, then http://127.0.0.1:8090")
    ap.add_argument("--no-discover", action="store_true",
                    help="do not also ask the Strata servers found in the process list")
    ap.add_argument("--strata", help="Strata checkout (or its serve/ folder) whose telemetry.py reads the GPUs; "
                                     "default ../Strata, then ../Strata-142; without one an own NVML reader is used")
    ap.add_argument("--log", help="engine log to read the decode timing from (default: the running server's config "
                                  "\"log\", else --engine-config's, else the newest Strata/strata-*.log)")
    ap.add_argument("--engine-config", help="Strata run config whose \"log\" is the engine log")
    ap.add_argument("--state", help=f"running totals file (default {STATE_FILE})")
    ap.add_argument("--settings", help=f"power/price settings file (default {SETTINGS_FILE})")
    ap.add_argument("--g3", default="http://127.0.0.1:8000", help="G3 (Genesis3) whose admin API sets the GPU power "
                    "limits and fan curves (settings menu; writes need the G3 admin password); 'off' = no menu")
    ap.add_argument("--strata-unit", default=strata_admin.UNIT, help="systemd user unit of the Strata server that "
                    "the settings menu restarts (Linux; the settings are written to --engine-config)")
    ap.add_argument("--g3-dir", default="/home/administrator/G3_VLLM", help="G3 folder (its logs/ settings are read "
                    "without a login)")
    a = ap.parse_args()
    if a.state:
        STATE_FILE = Path(a.state).resolve()
    if a.settings:
        SETTINGS_FILE = Path(a.settings).resolve()
    sources = None
    if a.source:
        sources = []
        for s in a.source:
            url, _, key = s.partition(",")
            sources.append((url.strip(), key.strip() or None))
    col = Collector(sources=sources, strata_path=a.strata, log_path=a.log, engine_config=a.engine_config,
                    discover=not a.no_discover)
    col.start()
    ctl = None if a.g3 == "off" else g3control.G3Control(a.g3, a.g3_dir)
    # the Strata settings need the login, so only with the G3 login and a run config to write to
    adm = strata_admin.StrataAdmin(a.engine_config, a.strata_unit) if ctl and a.engine_config and not IS_WIN else None
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(col, ctl, adm))
    srv.daemon_threads = True
    print(f"Strata dashboard on http://{a.host}:{a.port}/  (state: {STATE_FILE}, settings: {SETTINGS_FILE}, "
          f"GPUs: {len(col.gpus.cards) if col.gpus else 0} via {col.gpus.backend if col.gpus else col.gpu_error}, "
          f"sources: {', '.join(u for u, _ in col.sources)}{' + discovered' if col.discover else ''})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        col.save_state()


if __name__ == "__main__":
    main()
