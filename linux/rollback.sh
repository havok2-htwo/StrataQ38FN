#!/usr/bin/env bash
# Back to G3/vLLM: stop Strata, unregister it, restore the newest autostart.json backup, restart G3.
set -euo pipefail
G3=/home/administrator/G3_VLLM
systemctl --user stop strata.service || true
systemctl --user disable strata.service || true
[ -f $G3/logs/external_backends.json ] && mv $G3/logs/external_backends.json $G3/logs/external_backends.json.off-$(date +%Y%m%d-%H%M%S)
B=$(ls -t $G3/logs/autostart.json.bak-* 2>/dev/null | head -1); [ -n "$B" ] && cp "$B" $G3/logs/autostart.json && echo "restored $B"
systemctl --user restart g3.service
echo "G3 restarted; its autostart brings the vLLM models back (or start them in the dashboard)."
