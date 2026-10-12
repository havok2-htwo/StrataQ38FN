#!/usr/bin/env bash
# Desktop: start the production stack (Strata 8090, G3 8000, Open WebUI 8080, dashboard 8082) and show the status.
systemctl --user start strata.service strata-dashboard.service g3.service open-webui.service
echo "Starte Strata ... (das Modell braucht ca. 1,5 Minuten)"
for i in $(seq 90); do curl -s -m 3 http://127.0.0.1:8090/health | grep -q '"loaded": *true' && break; sleep 2; done
sleep 5
printf '\n%-14s %s\n' "Dienst" "Status"
for u in strata g3 open-webui strata-dashboard; do printf '%-14s %s\n' "$u" "$(systemctl --user is-active $u.service)"; done
echo
echo "Strata:     $(curl -s -m 3 http://127.0.0.1:8090/health | grep -o '"status": *"[a-z]*"')"
echo "G3 Modelle: $(curl -s -m 5 http://127.0.0.1:8000/v1/models | grep -o '"id":"[^"]*"' | tr '\n' ' ')"
echo "GPU Power:  $(nvidia-smi --query-gpu=power.limit --format=csv,noheader | tr '\n' ' ')"
echo
echo "G3:         http://192.168.20.126:8000/"
echo "Open WebUI: http://192.168.20.126:8080/"
echo "Dashboard:  http://192.168.20.126:8082/"
xdg-open http://127.0.0.1:8082/ >/dev/null 2>&1 &
echo; read -rp "Enter zum Schliessen ..." _
