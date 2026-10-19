#!/usr/bin/env bash
# Short GPU window (~10 min, production down): prefix fan-out on production's engine + config + server + server env,
# test server on 127.0.0.1:8091 (own log, own empty tier dir).  prefix_fanout.py: 16 requests sharing a ~20K document,
# warm (x sent once before) and cold (all 16 at once, x new), then warm with a 60K document.  Production is started
# again on exit, abort and Ctrl-C (and only stopped once it reports loaded and idle).  Results: results/fanout/.
#   run: cd /media/raid0/StrataQ38FN/linux && ./window-fanout.sh
set -uo pipefail
cd "$(dirname "$0")" || exit 1
for v in $(compgen -e | grep '^STRATA_'); do unset "$v"; done
R=/media/raid0/StrataQ38FN; L=$R/linux; C=$L/configs; OUT=$L/results/fanout; mkdir -p "$OUT"
PY=$R/Strata/.venv/bin/python; SRV=$R/Strata-143-prod
SP=8091; SURL=http://127.0.0.1:$SP; PROD_URL=http://127.0.0.1:8090; TIER=/media/raid0/strata-kvspill-bench/fanout
STOPPED=0; BGPID=""
exec 3>&1
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/window.log" >&3; }
gpus_free() { for i in $(seq 90); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && return 0; sleep 2; done; return 1; }
prod_loaded() { for i in $(seq "${1:-120}"); do curl -s -m 3 $PROD_URL/health | grep -q '"loaded": *true' && return 0; sleep 5; done; return 1; }
prod_idle() { curl -s -m 3 $PROD_URL/v1/load | python3 -I -c 'import json,sys; s=json.load(sys.stdin)["slots"]; sys.exit(0 if s["busy"] == 0 and s["waiting"] == 0 else 1)' 2>/dev/null; }
stop_bench() { systemctl --user stop strata-bench.scope 2>/dev/null; sleep 3; gpus_free || log "GPUs busy after the bench server stopped"; }
bgw() { "$@" 3>&- & BGPID=$!; wait $BGPID; local rc=$?; BGPID=""; return $rc; }
back() { trap - EXIT INT TERM HUP; trap '' PIPE
  [ -n "$BGPID" ] && kill "$BGPID" 2>/dev/null
  stop_bench; rm -rf "$TIER"
  if [ "$STOPPED" = 1 ]; then
    systemctl --user start strata.service; log "production start requested (production.json -> $(readlink $C/production.json))"
    if prod_loaded 180; then log "production loaded again"; else log "PRODUCTION DID NOT REPORT LOADED - check strata.service now"; fi
  fi; }

PRODSRC=$(readlink -f "$C/production.json")
for f in "$PRODSRC" "$SRV/serve/server.py" prefix_fanout.py; do [ -f "$f" ] || { echo "missing: $f"; exit 1; }; done
if systemctl --user is-active -q strata-bench.scope || curl -s -m 2 -o /dev/null $SURL/health; then
  echo "strata-bench.scope is active or port $SP answers - another bench is running; window NOT started"; exit 1; fi
SRVENV=$(systemctl --user show strata.service -p Environment --value | tr ' ' '\n' | grep '^STRATA_' | tr '\n' ' ')
rm -rf "$TIER"; mkdir -p "$TIER"
python3 -I - "$PRODSRC" "$OUT/config.json" "$TIER" "$OUT/engine.log" <<'PYEOF' || exit 1
import json, sys
c = json.load(open(sys.argv[1])); c["env"]["STRATA_CONV_TIER_DIR"] = sys.argv[3]; c["log"] = sys.argv[4]
json.dump(c, open(sys.argv[2], "w"), indent=1)
PYEOF
: > "$OUT/engine.log"
log "prep: production $(basename $PRODSRC), exe $(basename "$(python3 -c "import json;print(json.load(open('$PRODSRC'))['exe'])")"), server $SRV @$(git -C $SRV rev-parse --short HEAD), server env: $SRVENV"
[ -n "${DRY:-}" ] && { log "dry run: ok, production untouched"; exit 0; }

log "waiting until production reports loaded and idle"
prod_loaded 120 || { log "production not loaded - window NOT started"; exit 1; }
for i in $(seq 36); do prod_idle && break; sleep 5; done
trap back EXIT
trap 'trap "" PIPE; log "aborted by signal"; exit 1' INT TERM HUP
log "window start: stopping strata.service"; STOPPED=1; systemctl --user stop strata.service
gpus_free || { log "GPUs not free"; exit 1; }

systemctl --user reset-failed strata-bench.scope 2>/dev/null
( cd "$SRV" && env $SRVENV systemd-run --user --scope --unit=strata-bench -p MemoryMax=224G -p MemorySwapMax=0 \
    $PY "$SRV/serve/server.py" --engine strata --config "$OUT/config.json" --port $SP --host 127.0.0.1 \
    > "$OUT/server.out" 2>&1 < /dev/null & )
t0=$(date +%s)
until curl -s -m 3 $SURL/health | grep -q '"loaded": *true'; do
  [ $(( $(date +%s) - t0 )) -gt 10 ] && ! systemctl --user is-active -q strata-bench.scope && { log "server died during start"; tail -5 "$OUT/server.out" | tee -a "$OUT/window.log"; exit 1; }
  [ $(( $(date +%s) - t0 )) -gt 600 ] && { log "start timeout"; exit 1; }; sleep 5; done
log "test server ready after $(( $(date +%s) - t0 )) s"
for run in "warm 20000 w20" "cold 20000 c20" "warm 60000 w60"; do
  set -- $run
  bgw timeout 600 $PY -I prefix_fanout.py --url $SURL --n 16 --doc-tok $2 --mode $1 --tag "$3-$(date +%s)" --label $3 \
      --out "$OUT/$3.json" > "$OUT/$3.txt" 2>&1
  log "$3: $(grep '^==>' "$OUT/$3.txt" | cut -c1-400)"
done
grep -a "prompt .* tokens = .* reused" "$OUT/engine.log" | tail -60 > "$OUT/engine-reuse.txt"
log "engine reuse lines: $(wc -l < "$OUT/engine-reuse.txt") (results/fanout/engine-reuse.txt)"
back
log "window done"
