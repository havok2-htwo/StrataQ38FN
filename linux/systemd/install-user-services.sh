#!/usr/bin/env bash
# Install + enable the user services (they start at login; at boot without login after: sudo loginctl enable-linger $USER).
#   install-user-services.sh            install + enable (does not start anything)
#   install-user-services.sh --start    also start them now
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p ~/.config/systemd/user
for u in strata g3 open-webui strata-dashboard; do cp "$HERE/$u.service" ~/.config/systemd/user/; done
systemctl --user daemon-reload
systemctl --user enable strata.service g3.service open-webui.service strata-dashboard.service
[ "${1:-}" = "--start" ] && systemctl --user start strata.service strata-dashboard.service g3.service open-webui.service
loginctl show-user "$USER" -p Linger
