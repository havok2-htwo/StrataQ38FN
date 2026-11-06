#!/usr/bin/env bash
# Night run of window-borrow.sh (2026-11-07 02:00 CET, owner's go-ahead 2026-11-06): production already runs ll10-borrow, so the
# arms are built from the previous production (1441j-reads2 + server 90f73293 in Strata-143-base); N arms use Strata-srv-1106.
# Posts a line to the AITalk channel before and after (the Genesis agent pauses its counting for the window).
set -uo pipefail
L=/media/raid0/StrataQ38FN/linux; AT=http://10.0.0.1:7420/api/channels/strata-genesis3-cache-langer-cha/messages
post() { python3 -I -c 'import json,sys;print(json.dumps({"sender":"Claude-Strata","type":"chat","content":sys.argv[1]}))' "$1" |
         curl -sS -m 10 -X POST -H 'Content-Type: application/json' --data-binary @- -o /dev/null "$AT" || true; }
post "@Claude-Genesis3-Cache @Supervisor (Human) **GPU-Messfenster startet jetzt ($(date -u +%H:%M) UTC):** Produktion pausiert ~95 min (max. 110), danach automatisch zurück mit demselben Stand wie vorher (ll10-borrow). Bitte die Aufrufe in dieser Zeit nicht in die Statistik zählen."
cd "$L"
BASE_CONFIG=configs/T16-1441j-fair-g8-mmap4m-prod262r-vision-cpu.json BASE_SRV=/media/raid0/StrataQ38FN/Strata-143-base \
    ./window-borrow.sh --server-tree /media/raid0/StrataQ38FN/Strata-srv-1106; rc=$?
up=$(systemctl --user is-active strata.service)
verdicts=$(grep -E '==> .*(PASS|FAIL)' results/borrow/window.log 2>/dev/null | sed 's/^\[[0-9:]*\] *//' | cut -c1-220 | head -14)
post "@Claude-Genesis3-Cache @Supervisor (Human) **GPU-Messfenster beendet ($(date -u +%H:%M) UTC), Exit $rc, Produktion: $up.** Auswertung folgt am Morgen. Rohurteile:
\`\`\`
${verdicts:-keine Urteilszeilen gefunden - siehe linux/results/borrow/window.log}
\`\`\`"
exit $rc
