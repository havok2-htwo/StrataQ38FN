#!/usr/bin/env bash
# STRATA_SLOT_CKPT_MOVE check: upstream v0.1.45's tools/batch_groups_cache_test.py (git 477ac13c) against the production
# engine binary + production config, CKPT_MOVE=1 vs unset.  NEEDS ALL 4 GPUs: run only in a GPU window with production
# stopped by the user (this script never stops, restarts or talks to strata.service / port 8090).
#   ./run_ckpt_move_check.sh            # U B16 B16P, each moveon + moveoff (6 engine starts, ~20 min)
#   ./run_ckpt_move_check.sh U          # only upstream's own args (2 starts, ~6-7 min)
set -uo pipefail
D=$(cd "$(dirname "$0")" && pwd)
PY=/media/raid0/StrataQ38FN/Strata/.venv/bin/python
export STRATA_REPO=/media/raid0/StrataQ38FN/Strata-143-prod     # production's server code (90f73293)
VARIANTS=("${@:-U B16 B16P}"); VARIANTS=(${VARIANTS[*]})
declare -A EXTRA=(
  [U]="--batch 2 --batch-groups 2 --prompt-cache 4 --conversation-cache-mib 4096 --conversation-cache-slots 3"  # upstream's
  [B16]=""                                                                  # production args as they are (--batch 16 via "parallel")
  [B16P]="--prompt-cache 4 --conversation-cache-mib 4096 --conversation-cache-slots 3"   # --batch 16 + upstream's small cache
)

# --- guards: production must not be running and the GPUs must be free ---
if systemctl --user is-active --quiet strata.service; then
  echo "strata.service is active: production holds the GPUs. Not running (stop it only in an agreed GPU window)." >&2; exit 2; fi
if ss -ltn 2>/dev/null | grep -q ':8090 '; then echo "port 8090 is listening: not running." >&2; exit 2; fi
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
if [ "${used:-99999}" -gt 1024 ]; then echo "a GPU has ${used} MiB in use: not running." >&2; exit 2; fi

"$PY" "$D/make_configs.py" >/dev/null || exit 3
# production's server-side drop-in env (strata.service.d); STRATA_SLOT_CKPT_MOVE must not leak in from this shell
export STRATA_ADMIT_BY_NEW=1 STRATA_YIELD_SHORT=2048 STRATA_YIELDS_MAX=4 STRATA_ADMIT_SHORT_FIRST=8192 STRATA_PREFIX_DIAG=1
unset STRATA_SLOT_CKPT_MOVE STRATA_EXTRA_ARGS SMOKE_LOG STRATA_EXE GPUS SPLIT

mkdir -p "$D/out"
summary="$D/out/summary.txt"; : > "$summary"
for v in "${VARIANTS[@]}"; do
  for m in on off; do
    tag="$v-move$m"; cfg="$D/configs/$tag.json"; log="$D/logs/$tag.log"
    rm -f "$log"; rm -rf "$D/tier/$tag"; mkdir -p "$D/tier/$tag"      # the test reads the WHOLE log: fresh each run
    echo "=== $tag  extra: ${EXTRA[$v]:-(none)}  $(date +%T)"
    t0=$(date +%s)
    STRATA_EXTRA_ARGS="${EXTRA[$v]}" timeout 1200 "$PY" -u "$D/batch_groups_cache_test.py" "$cfg" "$D/out/$tag.json" \
      2>&1 | tee "$D/out/$tag.stdout"
    rc=${PIPESTATUS[0]}
    res=$(grep -E '^(PASS|FAIL)$' "$D/out/$tag.stdout" | tail -1)
    echo "$tag rc=$rc ${res:-NO-RESULT} $(( $(date +%s) - t0 )) s | $(grep '^requests' "$D/out/$tag.stdout") | $(grep '^slot gave back' "$D/out/$tag.stdout")" | tee -a "$summary"
    grep -aE 'batch-groups auto|WARNING|CKPT|error|Error' "$log" | head -20 > "$D/out/$tag.notes"
    sleep 5   # let the engine's VRAM go before the next start
  done
done
# exactness: CKPT_MOVE on vs off must give the same tokens at temperature 0
for v in "${VARIANTS[@]}"; do
  "$PY" -I - "$D/out/$v-moveon.json" "$D/out/$v-moveoff.json" <<'PY' | tee -a "$summary"
import json, sys
try:
    a, b = (json.load(open(p)) for p in sys.argv[1:3])
except Exception as e:
    print(sys.argv[1], "compare: missing", e); sys.exit()
same = [ra["answers"] == rb["answers"] for ra, rb in zip(a, b)]
print(sys.argv[1].rsplit("/", 1)[-1].replace("-moveon.json", ""), "move on/off tokens identical per round:", same)
PY
done
echo; cat "$summary"
