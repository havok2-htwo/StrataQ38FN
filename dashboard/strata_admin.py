"""Strata server settings for the monitor (Linux): read by everyone, changed only after the admin login.

What it covers (the knobs that are safe to turn without a new build or a deploy):
  - the conversation cache in RAM (--conversation-cache-mib / -slots / -min-free-mib / -min-tokens)
  - its SSD tier (STRATA_CONV_TIER_*: the day's write budget, the disk cap, min tokens, time to live, free space)
  - the server itself: status, restart (systemctl --user restart strata.service), the engine log's tail
Not covered on purpose: switching the config, the binary or the model - that stays a deploy.

The values live in the production run config (--engine-config, normally linux/configs/production.json, a symlink: the
file it points at is written, the link stays).  The engine reads them at its start, so a change is "pending" until the
next restart; the running values come from the engine process itself (/proc cmdline + environ), not from the file.
"""
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import psutil

UNIT = "strata.service"
LOG_TAIL_BYTES = 96 * 1024

# key -> where it lives, its name there, unit factor (UI value * factor = config value), range in UI units, the
# engine's default (UI units) when the config does not set it, label
FIELDS = {
    "cc_gib":           ("arg", "--conversation-cache-mib", 1024, (0, 160), None, "RAM-Cache GiB"),
    "cc_slots":         ("arg", "--conversation-cache-slots", 1, (16, 2048), None, "RAM-Cache Slots"),
    "cc_min_free_gib":  ("arg", "--conversation-cache-min-free-mib", 1024, (4, 64), None, "RAM frei halten GiB"),
    "cc_min_tokens":    ("arg", "--conversation-cache-min-tokens", 1, (0, 65536), None, "RAM-Cache ab Tokens"),
    "tier_write_gb_day": ("env", "STRATA_CONV_TIER_WRITE_GB_DAY", 1, (0, 5000), 200, "SSD schreiben GB/Tag"),
    "tier_cap_gb":      ("env", "STRATA_CONV_TIER_CAP_GB", 1, (10, 5000), 600, "SSD belegt max GB"),
    "tier_min_tokens":  ("env", "STRATA_CONV_TIER_MIN_TOKENS", 1, (1024, 262144), 16384, "SSD ab Tokens"),
    "tier_ttl_h":       ("env", "STRATA_CONV_TIER_TTL_H", 1, (1, 8760), 168, "SSD Aufbewahrungszeit h"),
    "tier_min_free_gb": ("env", "STRATA_CONV_TIER_MIN_FREE_GB", 1, (8, 2000), 64, "SSD frei halten GiB"),
}


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return int(f) if f == int(f) else f


def _arg(args, name):
    try:
        return args[args.index(name) + 1]
    except (ValueError, IndexError):
        return None


def values_from(args, env):
    """The fields in UI units from an argument list and an environment (None = not set and no known default)."""
    out = {}
    for key, (where, name, factor, _rng, default, _label) in FIELDS.items():
        raw = _arg(args, name) if where == "arg" else env.get(name)
        v = _num(raw)
        out[key] = (_num(round(v / factor, 3)) if v is not None else default)
    return out


class StrataAdmin:
    def __init__(self, config_path, unit=UNIT):
        self.config_path = Path(config_path) if config_path else None
        self.unit = unit
        self.lock = threading.Lock()            # one write / restart at a time
        self.last_restart = None                # (epoch, who-ish message) of the last restart from here

    # ---- read
    def _config(self):
        return json.loads(self.config_path.read_text(encoding="utf-8"))

    def service(self):
        props = ["ActiveState", "SubState", "MainPID", "ActiveEnterTimestamp", "NRestarts", "ExecMainStartTimestamp"]
        try:
            out = subprocess.run(["systemctl", "--user", "show", self.unit] + [f"-p{p}" for p in props],
                                 capture_output=True, text=True, timeout=10).stdout
        except (OSError, subprocess.SubprocessError) as e:
            return {"error": str(e)}
        s = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
        pid = int(s.get("MainPID") or 0)
        since = None
        if pid:
            try:
                since = psutil.Process(pid).create_time()
            except psutil.Error:
                pass
        return {"unit": self.unit, "active": s.get("ActiveState"), "sub": s.get("SubState"), "pid": pid or None,
                "since": since, "restarts": _num(s.get("NRestarts"))}

    @staticmethod
    def engine_process(main_pid):
        """The engine (the child that has --conversation-cache-mib or --serve) of the server process."""
        if not main_pid:
            return None
        try:
            kids = psutil.Process(main_pid).children(recursive=True)
        except psutil.Error:
            return None
        for c in kids:
            try:
                cl = c.cmdline()
            except psutil.Error:
                continue
            if "--serve" in cl or "--conversation-cache-mib" in cl:
                return c
        return None

    def view(self, logged_in=False):
        """Everything the settings tab shows; the log only with a login."""
        out = {"config": str(self.config_path) if self.config_path else None, "fields": {}, "error": None}
        out["fields"] = {k: {"label": f[5], "min": f[3][0], "max": f[3][1], "where": f[0], "name": f[1]}
                         for k, f in FIELDS.items()}
        svc = self.service()
        out["service"] = svc
        try:
            cfg = self._config()
            out["config_real"] = str(self.config_path.resolve())
            out["exe"] = cfg.get("exe")
            out["saved"] = values_from(cfg.get("args") or [], cfg.get("env") or {})
            tier_dir = (cfg.get("env") or {}).get("STRATA_CONV_TIER_DIR")
            log = cfg.get("log")
        except (OSError, ValueError, AttributeError) as e:
            out["error"] = f"Config nicht lesbar: {e}"
            cfg, tier_dir, log = {}, None, None
        eng = self.engine_process(svc.get("pid"))
        if eng is not None:
            try:
                out["running"] = values_from(eng.cmdline(), eng.environ())
                out["engine_pid"] = eng.pid
                tier_dir = eng.environ().get("STRATA_CONV_TIER_DIR", tier_dir)
            except psutil.Error:
                out["running"] = None
        else:
            out["running"] = None
        if tier_dir:
            try:
                du = psutil.disk_usage(tier_dir)
                used = sum(p.stat().st_size for p in Path(tier_dir).iterdir() if p.is_file())
                out["disk"] = {"dir": tier_dir, "total": du.total, "free": du.free, "tier_bytes": used}
            except OSError as e:
                out["disk"] = {"dir": tier_dir, "error": str(e)}
        try:
            out["ram_total"] = psutil.virtual_memory().total
        except Exception:
            pass
        out["last_restart"] = self.last_restart
        if logged_in and log:
            out["log"] = self.log_tail(log)
        return out

    @staticmethod
    def log_tail(path, nbytes=LOG_TAIL_BYTES):
        try:
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - nbytes))
                data = f.read()
        except OSError as e:
            return f"(Log nicht lesbar: {e})"
        text = data.decode("utf-8", "replace")
        return text.split("\n", 1)[1] if size > nbytes and "\n" in text else text

    # ---- write
    def save(self, data):
        """Validates the given fields and writes them into the config; returns the stored values or raises ValueError."""
        if not isinstance(data, dict):
            raise ValueError("expected an object")
        with self.lock:
            cfg = self._config()
            args = list(cfg.get("args") or [])
            env = dict(cfg.get("env") or {})
            new = {}
            for key, v in data.items():
                if key not in FIELDS:
                    continue
                where, name, factor, (lo, hi), _d, label = FIELDS[key]
                v = _num(v)
                if v is None or not lo <= v <= hi:
                    raise ValueError(f"{label}: erlaubt {lo} bis {hi}")
                new[key] = v
            # the disk cap must fit on the disk next to what it keeps free
            tier_dir = env.get("STRATA_CONV_TIER_DIR")
            if tier_dir and ("tier_cap_gb" in new or "tier_min_free_gb" in new):
                cur = values_from(args, env)
                cap = new.get("tier_cap_gb", cur["tier_cap_gb"]) * 1e9
                keep = new.get("tier_min_free_gb", cur["tier_min_free_gb"]) * 2 ** 30
                du = psutil.disk_usage(tier_dir)
                used = sum(p.stat().st_size for p in Path(tier_dir).iterdir() if p.is_file())
                room = du.free + used - keep
                if cap > room:
                    raise ValueError(f"SSD belegt max: höchstens {int(room // 1e9)} GB passen auf {tier_dir} "
                                     f"(frei {du.free / 1e9:.0f} GB + Stufe {used / 1e9:.0f} GB − frei halten "
                                     f"{keep / 2 ** 30:.0f} GiB)")
            for key, v in new.items():
                where, name, factor = FIELDS[key][:3]
                raw = str(int(round(v * factor)))
                if where == "env":
                    env[name] = raw
                elif name in args:
                    args[args.index(name) + 1] = raw
                else:
                    args += [name, raw]
            cfg["args"], cfg["env"] = args, env
            real = self.config_path.resolve()    # write the file the link points at; the link stays
            tmp = real.with_name(real.name + ".tmp")
            tmp.write_text(json.dumps(cfg, indent=1, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, real)
            return values_from(args, env)

    def restart(self):
        with self.lock:
            r = subprocess.run(["systemctl", "--user", "--no-block", "restart", self.unit],
                               capture_output=True, text=True, timeout=20)
            if r.returncode != 0:
                raise ValueError(f"Neustart abgelehnt: {(r.stderr or r.stdout).strip()}")
            self.last_restart = time.time()
            return {"ok": True, "at": self.last_restart}
