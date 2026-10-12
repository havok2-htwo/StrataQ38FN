#!/usr/bin/env bash
# Desktop: stop the production stack (frees the GPUs).
systemctl --user stop open-webui.service g3.service strata.service strata-dashboard.service
for u in strata g3 open-webui strata-dashboard; do printf '%-14s %s\n' "$u" "$(systemctl --user is-active $u.service)"; done
echo "GPUs: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"
echo; read -rp "Enter zum Schliessen ..." _
