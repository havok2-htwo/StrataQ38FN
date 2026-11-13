#!/usr/bin/env bash
# Install + enable the user services (they start at login; at boot without login after: sudo loginctl enable-linger $USER).
#   install-user-services.sh            install + enable (does not start anything)
#   install-user-services.sh --start    also start them now
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p ~/.config/systemd/user
for u in strata strata-reclaim g3 open-webui strata-dashboard; do cp "$HERE/$u.service" ~/.config/systemd/user/; done
# strata.service drop-ins: server-side admission order and read yield (TESTLOG #88/#89), short-first admission
mkdir -p ~/.config/systemd/user/strata.service.d
for d in admit-by-new read-yield server-env; do cp "$HERE/$d.conf" ~/.config/systemd/user/strata.service.d/; done
systemctl --user daemon-reload
systemctl --user enable strata.service g3.service open-webui.service strata-dashboard.service
[ "${1:-}" = "--start" ] && systemctl --user start strata.service strata-dashboard.service g3.service open-webui.service
loginctl show-user "$USER" -p Linger
