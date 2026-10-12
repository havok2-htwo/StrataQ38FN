#!/usr/bin/env bash
# Switch production from G3/vLLM (Qwen3.8-27B-FP8) to Strata (Qwen3.8-Flash-Next) behind G3's proxy on port 8000.
#   cutover.sh          do it (G3 must be on branch feat/external-backends, >= 2.5.0)
# Undo: rollback.sh
set -euo pipefail
G3=/home/administrator/G3_VLLM; HERE="$(cd "$(dirname "$0")" && pwd)"; TS=$(date +%Y%m%d-%H%M%S)
[ "$(git -C $G3 branch --show-current)" = "feat/external-backends" ] || { echo "G3 is not on feat/external-backends"; exit 1; }
# 1. stop a running G3 (it stops the vLLM servers it manages)
systemctl --user stop g3.service 2>/dev/null || true
if ss -ltn | grep -q ":8000 "; then echo "port 8000 still in use (G3 started by hand?) - stop it first"; exit 1; fi
# 2. no vLLM autostart on the GPUs Strata uses (backup first)
cp $G3/logs/autostart.json $G3/logs/autostart.json.bak-$TS
python3 - "$G3/logs/autostart.json" <<'PY'
import json, sys
p = sys.argv[1]; d = json.load(open(p))
for k, v in d.items():
    v["autostart"] = False; v["auto_restart"] = False
    if isinstance(v.get("request"), dict):
        v["request"]["autostart"] = False; v["request"]["auto_restart"] = False
json.dump(d, open(p, "w"), indent=4); print("autostart off for:", ", ".join(d))
PY
# 3. register Strata as G3's external backend
[ -f $G3/logs/external_backends.json ] && cp $G3/logs/external_backends.json $G3/logs/external_backends.json.bak-$TS
cat > $G3/logs/external_backends.json <<'JSON'
{"backends": [
  {"name": "Qwen3.8-Flash-Next", "url": "http://127.0.0.1:8090", "kind": "chat", "occupies_gpus": [0, 1, 2, 3]}
]}
JSON
# 4. services: install, enable (boot/login), start; old test scopes out of the way
for u in strata-dashboard strata-bench; do systemctl --user stop $u.scope 2>/dev/null || true; done
"$HERE/systemd/install-user-services.sh" --start
# 5. wait for Strata, then check every service
for i in $(seq 120); do curl -s -m 3 http://127.0.0.1:8090/health | grep -q '"loaded": *true' && break; sleep 5; done
echo "strata:    $(curl -s -m 3 http://127.0.0.1:8090/health | head -c 120)"
sleep 10
echo "g3 models: $(curl -s -m 5 http://127.0.0.1:8000/v1/models | head -c 300)"
echo "webui:     HTTP $(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/)"
echo "dashboard: HTTP $(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8082/)"
echo "gpu power:  $(nvidia-smi --query-gpu=power.limit --format=csv,noheader | tr '\n' ' ') (G3 power manager: day 380 W, night 280 W 22:00-07:00)"
echo "boot without login: $(loginctl show-user "$USER" -p Linger) (needs: sudo loginctl enable-linger $USER)"
