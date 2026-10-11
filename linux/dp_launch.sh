#!/usr/bin/env bash
# Start several Strata servers (data-parallel) as children of this script: dp_launch.sh cfg1:port1 cfg2:port2 ...
# Meant to run inside one systemd scope; stopping the scope stops all of them.
STRATA=/media/raid0/StrataQ38FN/Strata
cd "$STRATA"
pids=()
for spec in "$@"; do
  cfg=${spec%%:*}; port=${spec##*:}
  "$STRATA/.venv/bin/python" "$STRATA/serve/server.py" --engine strata --config "$cfg" --port "$port" \
    > "/media/raid0/StrataQ38FN/linux/results/server-dp-$port.out" 2>&1 < /dev/null &
  pids+=($!)
done
wait "${pids[@]}"
