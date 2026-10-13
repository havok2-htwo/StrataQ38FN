"""GPU power limits (day/night) and fan curves for the Strata Monitor's settings menu.

G3 (Genesis3, G3_VLLM) owns both: its power manager applies the day/night limits, its fan controller runs the curves.
The monitor never writes G3's files and never calls nvidia-smi itself:
  - reading:  G3's settings files directly (no login needed, same machine), or G3's admin API when logged in (+ status);
  - writing:  only through G3's admin API (POST /api/admin/power|fan/settings) with a G3 admin login, which G3
              validates and applies exactly like its own UI.
The login happens in the monitor (G3 admin password), the G3 session cookie stays on the server: the browser only gets
an HttpOnly monitor cookie.  Writes must come from the monitor page itself (same origin, JSON body).

Fan rule of the monitor: more cooling only.  A curve may never run below FLOOR_CURVE (G3's default curve) at any
temperature, the minimum speed may not go below G3's default, the 100 % hard limit may not move up, and the
hysteresis (fans slow down later) may not shrink.  "auto" (the driver's own control) stays allowed: it is what every
GPU runs without fan settings.  Less cooling is still possible in G3's own admin UI.
"""
import json
import secrets
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

FLOOR_CURVE = [[40, 30], [60, 45], [70, 65], [78, 85], [83, 100]]   # = G3 server_fan.DEFAULT_CURVE
FLOOR_MIN_PERCENT = 30                                               # = G3 DEFAULT_MIN_PERCENT
FLOOR_HYSTERESIS_C = 3                                               # = G3 DEFAULT_HYSTERESIS_C
MAX_HARD_LIMIT_C = 83                                                # = G3 DEFAULT_HARD_LIMIT_C
SESSION_S = 8 * 3600
COOKIE = "strata_monitor_ctl"


def interpolate(curve, temp):
    """Fan % at `temp`: linear between the points, flat outside them (as G3's controller does)."""
    pts = sorted(curve)
    if temp <= pts[0][0]:
        return float(pts[0][1])
    for (t0, p0), (t1, p1) in zip(pts, pts[1:]):
        if temp <= t1:
            return p0 + (p1 - p0) * (temp - t0) / (t1 - t0)
    return float(pts[-1][1])


def check_more_cooling(fan):
    """ValueError unless the fan settings cool at least as much as the floor (see the module doc)."""
    if not isinstance(fan, dict):
        raise ValueError("Lüfter-Einstellungen müssen ein Objekt sein.")
    def num(key, default):
        v = fan.get(key, default)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{key} muss eine Zahl sein.")
        return v
    if num("min_percent", FLOOR_MIN_PERCENT) < FLOOR_MIN_PERCENT:
        raise ValueError(f"Mindestdrehzahl unter {FLOOR_MIN_PERCENT} % ist hier nicht erlaubt (nur mehr Lüftung).")
    if num("hysteresis_c", FLOOR_HYSTERESIS_C) < FLOOR_HYSTERESIS_C:
        raise ValueError(f"Hysterese unter {FLOOR_HYSTERESIS_C} °C ist hier nicht erlaubt.")
    if num("hard_limit_c", MAX_HARD_LIMIT_C) > MAX_HARD_LIMIT_C:
        raise ValueError(f"Die 100-%-Grenze darf hier nicht über {MAX_HARD_LIMIT_C} °C liegen.")
    gpus = fan.get("gpus") or []
    if not isinstance(gpus, list):
        raise ValueError("gpus muss eine Liste sein.")
    for i, g in enumerate(gpus):
        for mode in ("day", "night"):
            cfg = (g or {}).get(mode) or {}
            if cfg.get("mode", "auto") != "curve":
                continue
            curve = cfg.get("curve")
            if not isinstance(curve, list) or len(curve) < 2:
                raise ValueError(f"GPU {i} {mode}: die Kurve braucht mindestens 2 Punkte.")
            try:
                pts = [[float(p[0]), float(p[1])] for p in curve]
            except (TypeError, ValueError, IndexError, KeyError):
                raise ValueError(f"GPU {i} {mode}: jeder Punkt ist [°C, %].")
            for t in range(0, 101):
                have, need = interpolate(pts, t), interpolate(FLOOR_CURVE, t)
                if have + 1e-9 < need:
                    raise ValueError(f"GPU {i} {'Tag' if mode == 'day' else 'Nacht'}: bei {t} °C {have:.0f} %, "
                                     f"die Grundkurve verlangt mindestens {need:.0f} % (nur mehr Lüftung).")


def expected_mode(power, now=None):
    """'day' or 'night' from G3's schedule (same rule as G3: night from start to end, across midnight)."""
    if not power or not power.get("night_enabled", power.get("start_time") != power.get("end_time")):
        return "day"
    now = now or datetime.now()
    m = now.hour * 60 + now.minute
    def mins(s):
        h, mm = s.split(":")
        return int(h) * 60 + int(mm)
    try:
        a, b = mins(power["start_time"]), mins(power["end_time"])
    except (KeyError, ValueError, AttributeError):
        return "day"
    if a == b:
        return "day"
    return "night" if (a <= m < b if a < b else m >= a or m < b) else "day"


class G3Control:
    def __init__(self, url="http://127.0.0.1:8000", g3_dir="/home/administrator/G3_VLLM"):
        self.url = url.rstrip("/")
        self.dir = Path(g3_dir)
        self.sessions = {}                       # monitor token -> {"g3": G3 cookie value, "until": epoch}
        self.lock = threading.Lock()

    # ---- sessions
    def session(self, token):
        with self.lock:
            s = self.sessions.get(token or "")
            if s and s["until"] < time.time():
                self.sessions.pop(token, None)
                s = None
            return s

    def login(self, password):
        """-> monitor token; ValueError with G3's message on a wrong password / lockout."""
        body = json.dumps({"username": self._username(), "password": password}).encode()
        req = urllib.request.Request(self.url + "/api/admin/login", body, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                cookie = r.headers.get("Set-Cookie", "")
        except urllib.error.HTTPError as e:
            raise ValueError(self._detail(e))
        except OSError as e:
            raise ConnectionError(f"G3 nicht erreichbar ({self.url}): {e}")
        g3 = next((c.split("=", 1)[1] for c in cookie.split(";") if c.strip().startswith("genesis_vllm_session=")), None)
        if not g3:
            raise ValueError("G3 hat keine Sitzung zurückgegeben.")
        token = secrets.token_urlsafe(32)
        with self.lock:
            self.sessions[token] = {"g3": g3.strip(), "until": time.time() + SESSION_S}
        return token

    def logout(self, token):
        with self.lock:
            s = self.sessions.pop(token or "", None)
        if s:
            try:
                self._call("POST", "/api/admin/logout", s, {})
            except Exception:
                pass

    # ---- read / write
    def view(self, token):
        """Settings for the menu: from G3's API with status when logged in, else from G3's files (read-only)."""
        s = self.session(token)
        out = {"g3_url": self.url, "logged_in": bool(s), "floor": {"curve": FLOOR_CURVE, "min_percent": FLOOR_MIN_PERCENT,
               "hysteresis_c": FLOOR_HYSTERESIS_C, "hard_limit_c": MAX_HARD_LIMIT_C}, "error": None}
        if s:
            try:
                p = self._call("GET", "/api/admin/power/settings", s)
                f = self._call("GET", "/api/admin/fan/settings", s)
                out["power"] = {k: v for k, v in p.items() if k != "settings"} | {"settings": p.get("settings")}
                out["fan"] = {k: v for k, v in f.items() if k != "settings"} | {"settings": f.get("settings")}
                out["mode"] = expected_mode(p.get("settings"))
                return out
            except PermissionError:
                out["logged_in"] = False
            except Exception as e:
                out["error"] = str(e)
        power = self._file("genesis_power_settings.json")
        fan = self._file("genesis_fan_settings.json")
        out["power"] = {"settings": power, "configured": power is not None}
        out["fan"] = {"settings": fan, "configured": fan is not None}
        out["mode"] = expected_mode(power)
        return out

    def save_power(self, token, data):
        return self._call("POST", "/api/admin/power/settings", self._need(token), data)

    def save_fan(self, token, data):
        check_more_cooling(data)
        return self._call("POST", "/api/admin/fan/settings", self._need(token), data)

    # ---- helpers
    def _need(self, token):
        s = self.session(token)
        if not s:
            raise PermissionError("nicht angemeldet")
        return s

    def _call(self, method, path, s, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data, method=method,
                                     headers={"Content-Type": "application/json", "Cookie": f"genesis_vllm_session={s['g3']}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            if e.code == 401:
                with self.lock:
                    for k, v in list(self.sessions.items()):
                        if v is s:
                            self.sessions.pop(k, None)
                raise PermissionError("G3-Sitzung abgelaufen, bitte neu anmelden")
            raise ValueError(self._detail(e))
        except OSError as e:
            raise ConnectionError(f"G3 nicht erreichbar ({self.url}): {e}")

    @staticmethod
    def _detail(e):
        try:
            d = json.loads(e.read() or b"{}").get("detail")
            return d if isinstance(d, str) else json.dumps(d)
        except Exception:
            return f"G3 antwortet {e.code}"

    def _file(self, name):
        try:
            d = json.loads((self.dir / "logs" / name).read_text())
            return d if isinstance(d, dict) else None
        except (OSError, ValueError):
            return None

    def _username(self):
        try:
            for line in (self.dir / ".env").read_text().splitlines():
                if line.startswith("GENESIS_ADMIN_USERNAME="):
                    return line.split("=", 1)[1].strip().strip('"\'') or "admin"
        except OSError:
            pass
        return "admin"
