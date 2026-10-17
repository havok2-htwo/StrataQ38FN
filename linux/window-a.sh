#!/usr/bin/env bash
# GPU window A (upstream plan, scratchpad upstream2/plan.md "Window A" steps 1-4; ~2.6 h expected, production down).
# Stock upstream v0.1.45 (477ac13c, Strata-145-stock, setup.py release flags, CUDA 13.0) unless an arm says "ours".
#   1. HC_Q8 boot gate: tools/batch_test.py (greedy, all experts resident, --pcie-frac 0 --adapt-every 1000000,
#      16 prompts in 16 slots) with STRATA_HC_Q8=1 on stock + --trim-stage-weights, the aa8c85d7 build
#      (Strata-145-hcq8 = 477ac13c + 6d25c8c6) + trim, and the hcq8 build without trim. Per card: the engine's
#      "STRATA_HC_Q8=1: ... GiB" line, "VRAM free with everything loaded", the lowest nvidia-smi memory.free, expert cache
#      slots; tokens stock vs hcq8 (and each slot against its solo run). HARD GATE: hc_q8 GiB per card must drop
#      under trim. Also says whether the production engine log ever had the hc_q8 line (TESTLOG note).
#      The stock-trim vs hcq8-trim token comparison counts as an identity gate only when both arms hold the same expert
#      cache slots on every card (--expert-cache auto sizes from free VRAM, and the fix frees VRAM); else "not comparable".
#   2. STRATA_SLOT_CKPT_MOVE: upstream's tools/batch_groups_cache_test.py (mirrors session 9f's
#      run_ckpt_move_check.sh: same U / B16P / B16 extras, fresh log and conversation tier per run, MOVE on/off token
#      diff). First our production build (strata-g3-1441j-reads2, Strata-143-prod server code, drop-in env) MOVE on and
#      off, B16 (production-shaped) first: it decides whether production.json keeps MOVE; then stock: merged switches
#      without MOVE / + MOVE without CKPT_POOL / + MOVE + CKPT_POOL. A variant's on/off pair (or stock triple) starts
#      only when the step cap and DEADLINE leave room for all of its runs.
#   3. docs/BATCHING.md arms on stock: 8 slots / 16 slots / 16 slots + every merged switch except MOVE
#      (+ STRATA_ADMIT_SHORT_FIRST=8192 in the server's env). bench_concurrency 1/8/16 short + 16 x 1.2K, then
#      bench_multiturn 16x3 greedy on the same server; reply identity across arms (compare_replies.py + multiturn sha).
#      1 round by default (S3_ROUNDS; more rounds rotate the arm order). STRATA_MMQ_DIRECT / STRATA_MMQ_GROUP_SORT are
#      inert in setup.py's CUDA release build (STRATA_MMQ_TAB=OFF); the window records the engine's own lines for it.
#   4. #2326 on 4 stages, stock, 16 slots: STRATA_PIPED_CONCURRENT unset / =1 / STRATA_PIPED_READ_PIECE=16384 (no
#      STRATA_SPLIT_MTP_BATCH: it turns CONCURRENT off; STRATA_IQ_MT_MIN=1 in all three so CPU expert rows are
#      deterministic). tools/piped_read_join_bench.py cold32k,two_joiners,followup, 3 interleaved rounds (rotated
#      order, a fresh server each), --determinism per arm in round 1 (sequential prompts: checks the plain path only);
#      the engine's "read N tokens on a thread of its own" lines; a fresh server per arm for bench_multiturn 16x3 greedy
#      + bench_agentic (unset vs conc/piece replies are informational: lent experts run on the CPU); last, 8 repeats
#      with CONCURRENT=1 (each its own timeout: a timeout = a hang, rc != 0 = an error; the server is restarted after
#      any rc != 0, the repeats stop after S4_MAX_HANGS hangs or at the step cap).
# Isolation as window-stock-cmp.sh: production (strata.service :8090) is stopped once it is loaded, every bench server
# runs in the strata-bench scope on 127.0.0.1:8095 (MemoryMax 224G), and strata.service is started again on every exit
# path (normal end, error, INT/TERM, the HARD_STOP watchdog), plus an independent transient timer (wa-restore.timer,
# HARD_STOP + 10 min) for SIGKILL/OOM of this script (a reboot starts the enabled strata.service anyway, Linger=yes).
# DEADLINE is when production starts again: every run's timeout is cut to end by DEADLINE, no engine starts with less
# than MIN_LEFT s before it, and production is not stopped at all unless the first step fits before it. Production is
# loaded again ~DEADLINE + 3 min (bench stop + ~100 s load). HARD_STOP (default DEADLINE + 10 min) is only a backstop.
# Every step has a time cap. No STRATA_* from the calling shell reaches an arm.
# Results: results/wa/ (window.log, s1-*, s2-*, wa-s3-*, wa-s4-*, sweep.jsonl(+requests), multiturn.jsonl,
# agentic.jsonl, engine logs <label>.engine.log). Configs: configs/WA-*.json.
#   DRY=1 ./window-a.sh                 # the plan: preflight + every command and engine line, touches nothing
#   ./window-a.sh                       # the window (needs Matthias' approval of the GPU window)
#   STEPS="1 2" DEADLINE=09:20 HARD_STOP=09:30 ./window-a.sh
set -uo pipefail
cd "$(dirname "$0")" || exit 1
DRY=${DRY:-0}; STEPS=${STEPS:-1 5 2 3 4}
DEADLINE=${DEADLINE:-09:20}; HARD_STOP=${HARD_STOP:-}
S2_STOCK_VARIANTS=${S2_STOCK_VARIANTS:-U B16P}; S2_OURS_VARIANTS=${S2_OURS_VARIANTS:-B16 B16P U}
S3_ROUNDS=${S3_ROUNDS:-1}; S4_ROUNDS=${S4_ROUNDS:-3}; S4_REPEATS=${S4_REPEATS:-8}; S4_MAX_HANGS=${S4_MAX_HANGS:-2}
MIN_LEFT=${MIN_LEFT:-300}      # s before DEADLINE: no engine start with less (a start takes ~75-95 s)
S2_RUN_EST=240                 # s per batch_groups_cache_test run (room check for a variant's pair/triple)
S4_SCEN=${S4_SCEN:-cold32k,two_joiners,followup}
L=/media/raid0/StrataQ38FN/linux; R=$L/results/wa; CF=$L/configs
PY=/media/raid0/StrataQ38FN/Strata/.venv/bin/python; SP=8095; SURL=http://127.0.0.1:$SP
STOCK=/media/raid0/StrataQ38FN/Strata-145-stock; HCQ8=/media/raid0/StrataQ38FN/Strata-145-hcq8
PROD=/media/raid0/StrataQ38FN/Strata-143-prod
STOCK_SHA=477ac13cbe8e582c20484871ce459d20d82df09d; HCQ8_SHA=6d25c8c6aad4591b3117f1767441c66ad597e339
PROD_SHA=90f73293f85a4b8908414f8da2eeb6a7da3e16f5
OURS_EXE=/media/raid0/StrataQ38FN/Strata/engine/strata-g3-1441j-reads2
TOOLS=$STOCK/tools
# production's server-side drop-in env (~/.config/systemd/user/strata.service.d/*.conf) for the "ours" arms
OURS_ENV=(STRATA_ADMIT_SHORT_FIRST=8192 STRATA_PREFIX_DIAG=1 STRATA_YIELD_SHORT=2048 STRATA_YIELDS_MAX=4 STRATA_ADMIT_BY_NEW=1)
S3_SW_ENV=(STRATA_ADMIT_SHORT_FIRST=8192)          # the one server-side merged switch (stock serve/server.py)
declare -A EXTRA=(   # STRATA_EXTRA_ARGS of batch_groups_cache_test.py (as 9f's run_ckpt_move_check.sh)
  [U]="--batch 2 --batch-groups 2 --prompt-cache 4 --conversation-cache-mib 4096 --conversation-cache-slots 3"
  [B16P]="--batch 16 --prompt-cache 4 --conversation-cache-mib 4096 --conversation-cache-slots 3"
  [B16]="--batch 16 --conversation-cache-mib 131072 --conversation-cache-slots 448")   # production's own cache
BT_EXTRA="--pcie-frac 0 --adapt-every 1000000"       # docs/BATCHING.md: exact comparisons
SCOPE=(systemd-run --user --scope --quiet --unit=strata-bench -p MemoryMax=224G -p MemorySwapMax=0)
CAP1=1800; CAP2=3600; CAP3=3600; CAP4=6000; CAP5=2400           # step caps (s): no new run starts in a step past its cap
EST1=10; EST2=37; EST3=25; EST4=82; EST5=20                   # expected minutes (engine start ~75-95 s each; see the header)

for v in $(compgen -e | grep '^STRATA_'); do unset "$v"; done   # nothing of ours from the calling shell
log() { if [ "$DRY" = 1 ]; then echo "[plan] $*"; else echo "[$(date +%T)] $*" | tee -a "$R/window.log"; fi; }
die() { log "ABORT before the window: $*"; exit 1; }
gpus_free() { for _ in $(seq 90); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && return 0; sleep 2; done; return 1; }
prod_loaded() { for _ in $(seq "${1:-120}"); do curl -s -m 3 http://127.0.0.1:8090/health | grep -q '"loaded": *true' && return 0; sleep 5; done; return 1; }
cfgget() { python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get(sys.argv[2],''))" "$1" "$2"; }
cfgenv() { python3 -c "import json,sys;[print(f'{k}={v}') for k,v in json.load(open(sys.argv[1])).get('env',{}).items()]" "$1"; }
DL_TS=$(date -d "$DEADLINE" +%s) || exit 1
if [ -n "$HARD_STOP" ]; then HS_TS=$(date -d "$HARD_STOP" +%s) || exit 1; else HS_TS=$(( DL_TS + 600 )); HARD_STOP=$(date -d "@$HS_TS" +%H:%M); fi
RESTORE_TS=$(( HS_TS + 600 ))   # wa-restore.timer: starts strata.service even if this script was SIGKILLed
past_deadline() { [ "$(date +%s)" -ge "$DL_TS" ]; }
STEP_END=0
ok_to_run() {   # before every engine start: MIN_LEFT s before DEADLINE, not past the current step's cap
  [ "$DRY" = 1 ] && return 0
  if [ $(( DL_TS - $(date +%s) )) -lt "$MIN_LEFT" ]; then log "DEADLINE $DEADLINE (less than $MIN_LEFT s left): no further runs"; ABORT=1; return 1; fi
  if [ "$(date +%s)" -ge "$STEP_END" ]; then log "step cap reached: the rest of this step is skipped"; return 1; fi
  return 0; }
room_for() {    # N: the step cap and DEADLINE leave room for N batch_groups_cache_test runs
  [ "$DRY" = 1 ] && return 0
  local now need; now=$(date +%s); need=$(( $1 * S2_RUN_EST ))
  [ $(( STEP_END - now )) -ge "$need" ] && [ $(( DL_TS - now )) -ge "$need" ]; }
ABORT=0

CUR=; SMI=; WD=; BACK_DONE=0
stop_bench() { systemctl --user stop strata-bench.scope 2>/dev/null; gpus_free || log "warning: GPUs still busy after stop"
  systemctl --user reset-failed strata-bench.scope 2>/dev/null; }
back() {
  [ "$DRY" = 1 ] && return; [ "$BACK_DONE" = 1 ] && return; BACK_DONE=1; trap '' INT TERM
  [ -n "$WD" ] && { pkill -P "$WD" 2>/dev/null; kill "$WD" 2>/dev/null; }; [ -n "$SMI" ] && kill "$SMI" 2>/dev/null
  stop_bench; systemctl --user start strata.service; log "production started again (strata.service)"
  systemctl --user stop wa-restore.timer 2>/dev/null; systemctl --user reset-failed wa-restore.timer wa-restore.service 2>/dev/null
  if prod_loaded 120; then log "production loaded again"; else log "WARNING: production not loaded 10 min after the start"; fi; }
on_sig() { trap '' INT TERM; log "interrupted (INT/TERM or HARD_STOP $HARD_STOP)"
  [ -n "$CUR" ] && { kill -TERM "$CUR"; wait "$CUR"; } 2>/dev/null; exit 130; }
CUT=0
step() {   # step SECS OUTFILE CMD...: CMD under a timeout cut to end by DEADLINE, in the background (so INT/TERM reach
           # the trap at once). rc 99: not run (under 60 s left); CUT=1: the timeout was cut to DEADLINE.
  local t=$1 out=$2; shift 2; CUT=0
  if [ "$DRY" = 1 ]; then local c="$*"; c=${c//"${SCOPE[*]}"/[strata-bench scope]}; c=${c//$PY/python}; c=${c//$TOOLS\//tools145/}
    c=${c//$R\//wa/}; c=${c//$CF\//configs/}; echo "    + [cap ${t}s] $c > ${out#$L/}"; return 0; fi
  local left=$(( DL_TS - $(date +%s) ))
  if [ "$left" -lt 60 ]; then log "DEADLINE $DEADLINE: $(basename "$out") not run"; ABORT=1; return 99; fi
  if [ "$t" -gt "$left" ]; then log "$(basename "$out"): timeout cut from $t s to $left s (DEADLINE $DEADLINE)"; t=$left; CUT=1; fi
  timeout "$t" "$@" > "$out" 2>&1 & CUR=$!; wait "$CUR"; local rc=$?; CUR=
  [ "$CUT" = 1 ] && [ "$rc" = 124 ] && { log "$(basename "$out") stopped at DEADLINE $DEADLINE"; ABORT=1; }
  return $rc; }

declare -A SHOWN=()
show_engine() {   # DRY: MODE REPO CFG [EXTRA]: the engine's command line and the config's env, as the tools build them
  local k="$1|$3|${4:-}"; [ -n "${SHOWN[$k]:-}" ] && { echo "      engine + env: as above for $(basename "$3") ${4:+[$4]}"; return; }; SHOWN[$k]=1
  STRATA_REPO=$2 "$PY" -I - "$1" "$3" "${4:-}" <<'PY'
import json, os, sys
from pathlib import Path
mode, cfgp, extra = sys.argv[1], sys.argv[2], sys.argv[3].split()
cfg = json.load(open(cfgp))
if mode == "bt":     # tools/batch_test.py: cfg args + --layer-split + --batch, the env from the caller (window-a.sh)
    args = list(cfg["args"]) + ["--layer-split", str(cfg.get("layer_split") or "auto")] + extra
else:                # serve/server.py's own engine_args (server runs, batch_groups_cache_test.py)
    REPO = Path(os.environ["STRATA_REPO"]); sys.path.insert(0, str(REPO / "tools")); sys.path.insert(0, str(REPO))
    import contextlib, io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        from serve.server import engine_args
        args = engine_args(cfg) + extra
    for ln in buf.getvalue().splitlines():
        print(f"      (server.py says: {ln.strip()})")
for f in ("--batch", "--batch-groups", "--conversation-cache-mib", "--conversation-cache-slots", "--prompt-cache"):
    if args.count(f) > 1:
        print(f"      DUPLICATE {f} x{args.count(f)}")
print(f"      engine: {Path(cfg['exe']).parent.parent.name}/{Path(cfg['exe']).name} --serve "
      f"{' '.join(a if '/' not in a else Path(a).name for a in args)}")
env = {k: v for k, v in (cfg.get("env") or {}).items() if not k.startswith("MALLOC_")}
KEY = ("STRATA_SLOT_CKPT_MOVE", "STRATA_CKPT_POOL", "STRATA_SLOT_PARK", "STRATA_SLOT_CHAIN_REUSE", "STRATA_GIVEBACK_CHECKS",
       "STRATA_CONV_TIER_DIR")
shown = env if len(env) <= 16 else {k: env.get(k, "(unset)") for k in KEY}
more = "" if len(env) <= 16 else f" (+{len(env) - sum(k in env for k in KEY)} more, as production.json)"
print(f"      env: {' '.join(f'{k}={v}' for k, v in shown.items()) or '-'}{more}  log: {Path(cfg['log']).name}")
PY
}

start_srv() {   # LABEL CFG SERVER_DIR [VAR=VAL ...]: a bench server in the strata-bench scope; 0 once it is loaded
  local lab=$1 cfg=$2 sdir=$3; shift 3
  if [ "$DRY" = 1 ]; then
    echo "    + server $lab: $(basename "$sdir")/serve/server.py --config $(basename "$cfg") :$SP${*:+  server env: $*}"
    show_engine srv "$sdir" "$cfg"; return 0; fi
  ok_to_run || return 1
  stop_bench; local englog; englog=$(cfgget "$cfg" log); : > "$englog"   # this start's own engine log
  ( cd "$sdir" && env "$@" "${SCOPE[@]}" "$PY" "$sdir/serve/server.py" --engine strata --config "$cfg" --port $SP \
      --host 127.0.0.1 > "$R/$lab.server.out" 2>&1 < /dev/null & )
  sleep 5; local t0; t0=$(date +%s)
  until curl -s -m 3 $SURL/health | grep -q '"loaded": *true'; do
    systemctl --user is-active -q strata-bench.scope || { log "$lab: server died during start"; srv_done "$lab" "$cfg"; return 1; }
    [ $(( $(date +%s) - t0 )) -gt 900 ] && { log "$lab: start timeout (900 s)"; srv_done "$lab" "$cfg"; return 1; }
    past_deadline && { log "$lab: DEADLINE $DEADLINE during the start"; ABORT=1; srv_done "$lab" "$cfg"; return 1; }
    sleep 5; done
  log "$lab ready after $(( $(date +%s) - t0 )) s"
  curl -s -m 10 $SURL/v1/status > "$R/$lab.status.json"; }
srv_done() {    # LABEL CFG: stop the bench server, keep its engine log
  [ "$DRY" = 1 ] && return 0
  stop_bench; local englog; englog=$(cfgget "$2" log); [ -f "$englog" ] || return 0; cp "$englog" "$R/$1.engine.log"
  log "$1 engine: $(grep -a -m1 -o -- '--batch: [0-9]* slot sessions' "$englog") | $(grep -a -c 'engine started' "$englog") start(s)$(inert "$englog" "$2")"; }
inert() {       # ENGINE_LOG CFG: switches the engine ignores (MMQ_DIRECT without STRATA_MMQ_TAB); GROUP_SORT has no "ignored"
                # line, so a config that sets it without the engine's "prompt MMQ sorts the full expert groups" line = inert
  local f=$1 s=""
  s=$(grep -a -o 'STRATA_[A-Z_]*=[0-9]* is ignored' "$f" 2>/dev/null | sort -u | tr '\n' ';')
  if grep -q '"STRATA_MMQ_GROUP_SORT"' "$2" 2>/dev/null; then
    local gs; gs=$(grep -a -m1 -o 'STRATA_MMQ_GROUP_SORT=1): [0-9]* of [0-9]* layers' "$f" 2>/dev/null | cut -d' ' -f2-)
    [ -n "$gs" ] && s="$s MMQ_GROUP_SORT sorts $gs;" || s="$s MMQ_GROUP_SORT set, no 'prompt MMQ sorts' line (inert);"; fi
  [ -n "$s" ] && echo " | $s"; }
healthy() { curl -s -m 5 $SURL/health | grep -q '"loaded": *true'; }
conc() {   # LABEL CAP bench_concurrency args...
  local lab=$1; shift
  step "$1" "$R/$lab.txt" "$PY" -I bench_concurrency.py --url $SURL --label "$lab" --out "$R/sweep.jsonl" "${@:2}"
  local rc=$?; [ "$DRY" = 1 ] || log "$lab done (rc $rc)$(grep -E '^ +[0-9]+ ' "$R/$lab.txt" | awk '{printf " | %s users %s tok/s TTFT p50 %s", $1, $4, $7}')"; }
mt() { step 900 "$R/$1.txt" "$PY" -I bench_multiturn.py --url $SURL --users 16 --turns 3 --temperature 0 --label "$1" \
         --out "$R/multiturn.jsonl"; local rc=$?; [ "$DRY" = 1 ] || log "$1 done (rc $rc): $(tail -1 "$R/$1.txt")"; }
ag() { step 1500 "$R/$1.txt" "$PY" -I bench_agentic.py --url $SURL --label "$1" --out "$R/agentic.jsonl"
       local rc=$?; [ "$DRY" = 1 ] || log "$1 done (rc $rc): $(tail -1 "$R/$1.txt" | cut -c1-110)"; }
mt_same() {   # multiturn reply identity: LABEL_A LABEL_B (greedy; sha per uid and turn)
  [ "$DRY" = 1 ] && { echo "    + multiturn sha identity $1 vs $2"; return; }
  "$PY" -I - "$R/multiturn.jsonl" "$1" "$2" <<'PY'
import json, sys
rows = {}
try:
    lines = open(sys.argv[1]).readlines()
except OSError:
    lines = []
for line in lines:
    try:
        r = json.loads(line)
    except ValueError:
        continue
    rows[r.get("label")] = {(x[0], x[1]): x[3] for x in r.get("replies", [])}
a, b = rows.get(sys.argv[2]), rows.get(sys.argv[3])
if not a or not b:
    print(f"multiturn {sys.argv[2]} vs {sys.argv[3]}: missing"); sys.exit()
k = sorted(set(a) & set(b)); same = sum(a[x] == b[x] for x in k)
print(f"multiturn {sys.argv[2]} vs {sys.argv[3]}: {same}/{len(k)} replies identical"
      + ("" if same == len(k) else f"; first differing (uid, turn): {[x for x in k if a[x] != b[x]][:5]}"))
PY
}

# ---------------------------------------------------------------- preflight (read-only; before production goes down)
preflight() {
  local bad=0 c
  [ -x $STOCK/build/strata ] || { echo "  MISSING $STOCK/build/strata"; bad=1; }
  [ -x $HCQ8/build/strata ] || { echo "  MISSING $HCQ8/build/strata"; bad=1; }
  [ -x $OURS_EXE ] || { echo "  MISSING $OURS_EXE"; bad=1; }
  [ "$(git -C $STOCK rev-parse HEAD 2>/dev/null)" = $STOCK_SHA ] || { echo "  $STOCK is not at v0.1.45 $STOCK_SHA"; bad=1; }
  [ "$(git -C $HCQ8 rev-parse HEAD 2>/dev/null)" = $HCQ8_SHA ] || { echo "  $HCQ8 is not at $HCQ8_SHA"; bad=1; }
  [ "$(git -C $HCQ8 rev-parse HEAD~1 2>/dev/null)" = $STOCK_SHA ] || { echo "  $HCQ8 is not 477ac13c + one commit"; bad=1; }
  [ "$(git -C $PROD rev-parse HEAD 2>/dev/null)" = $PROD_SHA ] || { echo "  $PROD is not at $PROD_SHA (production server)"; bad=1; }
  for d in $STOCK $HCQ8; do git -C $d diff --quiet HEAD -- src include serve tools CMakeLists.txt || { echo "  $d has local changes"; bad=1; }; done
  for b in $STOCK $HCQ8; do [ "$b/build/strata" -nt "$b/build/build.ninja" ] || { echo "  $b/build/strata is older than its build.ninja"; bad=1; }; done
  for f in batch_test.py batch_groups_cache_test.py piped_read_join_bench.py strata_tokenizer.py; do
    [ -f $TOOLS/$f ] || { echo "  MISSING $TOOLS/$f"; bad=1; }; done
  for f in bench_concurrency.py bench_multiturn.py bench_agentic.py compare_replies.py; do [ -f $L/$f ] || { echo "  MISSING $L/$f"; bad=1; }; done
  for c in $CF/WA-*.json; do python3 -c "import json,sys;json.load(open(sys.argv[1]))" "$c" || { echo "  bad JSON $c"; bad=1; }; done
  [ "$(ls $CF/WA-*.json | wc -l)" = 16 ] || { echo "  expected 16 configs/WA-*.json"; bad=1; }
  # every env var and CLI flag of the stock arms exists in v0.1.45 (git grep for the quoted name in the tree)
  local names
  names=$("$PY" -I - $CF/WA-S[1-5]-stock-*.json $CF/WA-S1-hcq8-*.json <<'PY'
import json, sys
n = set()
for p in sys.argv[1:]:
    c = json.load(open(p))
    n |= {k for k in c.get("env", {}) if not k.startswith("MALLOC_")}
    n |= {a for a in c["args"] if a.startswith("--")}
print(" ".join(sorted(n)))
PY
)
  names="$names STRATA_ADMIT_SHORT_FIRST $(echo "${EXTRA[*]} $BT_EXTRA" | tr ' ' '\n' | grep '^--' | sort -u | tr '\n' ' ') --layer-split --batch"
  local miss=""
  for n in $names; do git -C $STOCK grep -q -F "\"$n\"" $STOCK_SHA -- src serve include || miss="$miss $n"; done
  [ -z "$miss" ] && echo "  v0.1.45 has every env var / flag used ($(echo $names | wc -w))" || { echo "  NOT in v0.1.45:$miss"; bad=1; }
  # the upstream tools' options used here
  miss=""
  for o in --exe --config --batch --n --max-new --keys --extra --dump; do grep -q -- "\"$o\"" $TOOLS/batch_test.py || miss="$miss batch_test$o"; done
  for o in --port --scenarios --repeat --out --determinism; do grep -q -- "\"$o\"" $TOOLS/piped_read_join_bench.py || miss="$miss piped$o"; done
  for s in ${S4_SCEN//,/ }; do grep -q "\"$s\":" $TOOLS/piped_read_join_bench.py || miss="$miss scenario:$s"; done
  grep -q 'STRATA_EXTRA_ARGS' $TOOLS/batch_groups_cache_test.py && grep -q 'SMOKE_LOG' $TOOLS/batch_groups_cache_test.py || miss="$miss cache_test-env"
  [ -z "$miss" ] && echo "  upstream tools take every option used" || { echo "  tool options missing:$miss"; bad=1; }
  # nothing else on the bench port or running a GPU window / bench driver (other than this script's own process group)
  ss -ltn 2>/dev/null | grep -q ":$SP " && { echo "  port $SP in use: $(ss -ltnp 2>/dev/null | grep ":$SP " | grep -o 'users:.*' | head -1)"; bad=1; }
  local mypg others p anc=" " q=$$
  mypg=$(ps -o pgid= -p $$ | tr -d ' ')
  while [ -n "$q" ] && [ "$q" != 0 ] && [ "$q" != 1 ]; do anc="$anc$q "; q=$(ps -o ppid= -p "$q" 2>/dev/null | tr -d ' '); done
  # a script run as "[ba]sh [path/]NAME.sh" or "[path/]NAME.sh" (not a shell line that only mentions one), not ours/ancestors
  others=$(for p in $(pgrep -f '^(/usr)?(/bin/)?((ba)?sh )?([^ ]*/)?(window-[A-Za-z0-9_-]+|run_strata_variant|run_ckpt_move_check|soak)\.sh( |$)' 2>/dev/null); do
    case "$anc" in *" $p "*) continue;; esac
    [ "$(ps -o pgid= -p "$p" 2>/dev/null | tr -d ' ')" = "$mypg" ] || ps -o pid=,args= -p "$p" 2>/dev/null; done)
  [ -n "$others" ] && { echo "  another window/bench script is running:"; echo "$others" | sed 's/^/    /'; bad=1; }
  systemctl --user is-active -q strata-bench.scope && { echo "  strata-bench.scope is in use (another bench?)"; bad=1; }
  # time budget
  local now; now=$(date +%s)
  [ "$now" -lt "$DL_TS" ] || { echo "  DEADLINE $DEADLINE ($(date -d @$DL_TS '+%F %T')) has passed"; bad=1; }
  [ "$HS_TS" -gt "$DL_TS" ] || { echo "  HARD_STOP must be after DEADLINE"; bad=1; }
  echo "  deadline $(date -d @$DL_TS '+%F %T') ($(( (DL_TS - now) / 60 )) min from now; every run ends by it, production"\
" loaded again ~$(date -d @$(( DL_TS + 180 )) +%H:%M)), hard stop (backstop) $(date -d @$HS_TS '+%T'), restore timer $(date -d @$RESTORE_TS '+%T')"
  systemctl --user is-enabled -q strata.service || { echo "  strata.service is not enabled (a reboot would not start production)"; bad=1; }
  return $bad; }

# ---------------------------------------------------------------- step 1: HC_Q8 boot gate
step1() {
  log "step 1: HC_Q8 boot gate (STRATA_HC_Q8=1; stock+trim, hcq8+trim, hcq8 without trim; batch_test 16 slots greedy)"
  local arm cfg exe
  for arm in stock-trim hcq8-trim hcq8-notrim; do
    cfg=$CF/WA-S1-$arm.json; exe=$(cfgget "$cfg" exe)
    if [ "$DRY" = 1 ]; then show_engine bt $STOCK "$cfg" "--batch 16 $BT_EXTRA"; fi
    ok_to_run || break
    [ "$DRY" = 1 ] || { stop_bench; rm -f "$R/s1-$arm.engine.log"
      nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free --format=csv,noheader,nounits -l 2 > "$R/s1-$arm.vram.csv" 2>/dev/null & SMI=$!; }
    step 900 "$R/s1-$arm.out" "${SCOPE[@]}" env CUDA_DEVICE_ORDER=PCI_BUS_ID BATCH_TEST_LOG="$R/s1-$arm.engine.log" $(cfgenv "$cfg") \
      "$PY" -I $TOOLS/batch_test.py --exe "$exe" --config "$cfg" --batch 16 --n 16 --max-new 128 --keys temperature=0 \
      --extra "$BT_EXTRA" --dump "$R/s1-$arm.dump.json"
    local rc=$?
    [ "$DRY" = 1 ] || { kill "$SMI" 2>/dev/null; wait "$SMI" 2>/dev/null; SMI=; stop_bench
      log "s1 $arm rc $rc: $(grep -a 'STRATA_HC_Q8=1:' "$R/s1-$arm.engine.log" 2>/dev/null | grep -o '[0-9.]* GiB' | tr '\n' ' ')| $(grep -c IDENTICAL "$R/s1-$arm.out") of 16 slots = solo | cache slots CUDA0 $(grep -a -m1 -o -E '^strata generate: expert cache [0-9]+' "$R/s1-$arm.engine.log" 2>/dev/null | grep -o '[0-9]*$') $(grep -a -o -E 'CUDA[1-3] runs layers [0-9-]+, expert cache [0-9]+' "$R/s1-$arm.engine.log" 2>/dev/null | sort -u | sed -E 's/ runs layers [0-9-]+, expert cache / /' | tr '\n' ' ')"; }
  done
  [ "$DRY" = 1 ] && { echo "    + summary + HARD GATE -> results/wa/s1-summary.txt"; return; }
  "$PY" -I - "$R" > "$R/s1-summary.txt" 2>&1 <<'PY'
import csv, json, re, sys
from pathlib import Path
R = Path(sys.argv[1]); arms = ["stock-trim", "hcq8-trim", "hcq8-notrim"]; S = {}
for a in arms:
    p = R / f"s1-{a}.engine.log"
    text = p.read_text(errors="replace").splitlines() if p.exists() else []
    card, hc, slots, free, cover = None, [], {}, {}, {}
    for ln in text:
        m = re.search(r"layer split: CUDA(\d) loads the dense weights", ln)
        if m: card = int(m.group(1))
        m = re.search(r"STRATA_HC_Q8=1: ([\d.]+) GiB", ln)
        if m: hc.append((card if card is not None else len(hc), float(m.group(1))))
        m = re.search(r"layer split: CUDA(\d) runs layers [\d-]+, expert cache (\d+) slots .*?, (\d+) of its (\d+) profiled pairs", ln)
        if m: slots[int(m.group(1))] = int(m.group(2)); cover[int(m.group(1))] = (int(m.group(3)), int(m.group(4)))
        m = re.search(r"^strata generate: expert cache (\d+) slots, ([\d.]+) GiB of VRAM", ln)
        if m and 0 not in slots: slots[0] = int(m.group(1))
        m = re.search(r"strata serve: (?:CUDA(\d): )?(\d+) MiB of VRAM free with everything loaded", ln)
        if m: free[int(m.group(1) or 0)] = int(m.group(2))
    smin = {}
    try:
        for r in csv.reader(open(R / f"s1-{a}.vram.csv")):
            i, f = int(r[1]), int(r[3]); smin[i] = min(f, smin.get(i, f))
    except (OSError, ValueError, IndexError):
        pass
    out = (R / f"s1-{a}.out").read_text(errors="replace") if (R / f"s1-{a}.out").exists() else ""
    dump = json.loads((R / f"s1-{a}.dump.json").read_text()) if (R / f"s1-{a}.dump.json").exists() else None
    S[a] = dict(hc=hc, slots=slots, free=free, smin=smin, ident=out.count("IDENTICAL"), dump=dump, cover=cover,
                batch=next((l for l in out.splitlines() if l.startswith("batch:")), "batch: -"))
    print(f"== {a}")
    print("  hc_q8 GiB per card (engine line, card = the last 'CUDAn loads the dense weights' before it):",
          ", ".join(f"CUDA{c} {g:.2f}" for c, g in hc) or "NO hc_q8 LINE")
    print("  VRAM free with everything loaded (engine, MiB):", ", ".join(f"CUDA{c} {v}" for c, v in sorted(free.items())) or "-")
    print("  lowest memory.free during the run (nvidia-smi, MiB):", ", ".join(f"CUDA{c} {v}" for c, v in sorted(smin.items())) or "-")
    print("  expert cache slots:", ", ".join(f"CUDA{c} {v}" for c, v in sorted(slots.items())) or "-",
          "| profiled pairs held (CUDA1-3):", ", ".join(f"CUDA{c} {h} of {n}" for c, (h, n) in sorted(cover.items())) or "-")
    print(f"  {S[a]['ident']}/16 slots identical to their solo run; {S[a]['batch']}")
def same(x, y):
    if not (S[x]["dump"] and S[y]["dump"]): return "missing dump"
    a, b = S[x]["dump"], S[y]["dump"]
    s = sum(p == q for p, q in zip(a["solo"], b["solo"])); t = sum(p == q for p, q in zip(a["batch"], b["batch"]))
    return f"solo {s}/{len(a['solo'])}, batch {t}/{len(a['batch'])} identical"
def gate(x, y):   # an identity gate only when both arms hold the same expert cache on every card (auto sizes from free VRAM)
    sx, sy = S[x]["slots"], S[y]["slots"]
    if len(sx) != 4 or sx != sy:
        return f"NOT COMPARABLE: cache differs ({x} {dict(sorted(sx.items()))} vs {y} {dict(sorted(sy.items()))}); informational: {same(x, y)}"
    full = all(h == n for h, n in S[x]["cover"].values()) and len(S[x]["cover"]) == 3
    return f"identity gate (same cache on every card{', full complement' if full else ', NOT the full complement'}): {same(x, y)}"
print("tokens stock-trim vs hcq8-trim:", gate("stock-trim", "hcq8-trim"))
print("tokens hcq8-trim vs hcq8-notrim:", gate("hcq8-trim", "hcq8-notrim"))
st, hq, nt = (dict(S[a]["hc"]) for a in arms)
cards = sorted(st)
drop = len(cards) == 4 and all(c in hq and hq[c] < st[c] - 0.01 for c in cards)
print("HARD GATE (hc_q8 GiB drops on every card under trim):", "PASS" if drop else "FAIL",
      "|", ", ".join(f"CUDA{c} {st[c]:.2f} -> {hq.get(c, float('nan')):.2f}" for c in cards))
print("no-trim check (hcq8 without trim = stock with trim, all layers):",
      "same" if nt and sorted(nt.values()) == sorted(st.values()) else f"differs/missing: {nt}")
PY
  cat "$R/s1-summary.txt" | tee -a "$R/window.log"
  # TESTLOG note: HC_Q8 never on in production (no config sets it, no engine log line)
  { echo "production configs setting STRATA_HC_Q8: $(grep -l STRATA_HC_Q8 $CF/production.json $CF/T16-1441*.json 2>/dev/null | wc -l)"
    echo "hc_q8 lines in the production engine log: $(grep -a -c 'STRATA_HC_Q8=1:' /media/raid0/StrataQ38FN/Strata/strata-unsloth-ud-iq4_xs.log)"; } \
    | tee -a "$R/s1-summary.txt" "$R/window.log"; }

# ---------------------------------------------------------------- step 2: SLOT_CKPT_MOVE (batch_groups_cache_test.py)
s2_run() {   # TAG CFG REPO VARIANT [VAR=VAL ...]
  local tag=$1 cfg=$2 repo=$3 v=$4; shift 4
  if [ "$DRY" = 1 ]; then show_engine srv "$repo" "$cfg" "${EXTRA[$v]}"; fi
  ok_to_run || return 1
  [ "$DRY" = 1 ] || { stop_bench; rm -f "$R/$tag.engine.log"; rm -rf "$R/tier-s2"; mkdir -p "$R/tier-s2"; }   # fresh log + tier
  local t0; t0=$(date +%s)
  step 900 "$R/$tag.out" "${SCOPE[@]}" env STRATA_REPO="$repo" STRATA_EXTRA_ARGS="${EXTRA[$v]}" SMOKE_LOG="$R/$tag.engine.log" "$@" \
    "$PY" -I -u $TOOLS/batch_groups_cache_test.py "$cfg" "$R/$tag.json"
  local rc=$?
  [ "$DRY" = 1 ] && return 0
  stop_bench
  log "$tag rc $rc $(grep -E '^(PASS|FAIL)$' "$R/$tag.out" | tail -1) $(( $(date +%s) - t0 )) s | $(grep '^requests' "$R/$tag.out") | $(grep '^slot gave back' "$R/$tag.out")"
  { grep -aE 'batch-groups auto|INFO batch|WARNING|CKPT|error|Error|is ignored' "$R/$tag.engine.log" 2>/dev/null | head -20
    inert "$R/$tag.engine.log" "$cfg"; } > "$R/$tag.notes"
  return 0; }
step2() {
  log "step 2: STRATA_SLOT_CKPT_MOVE, upstream batch_groups_cache_test.py (ours first: $S2_OURS_VARIANTS; stock: $S2_STOCK_VARIANTS)"
  local v arm
  # ours first (production-shaped B16 first: it decides whether production.json keeps MOVE); a pair starts only whole
  for v in $S2_OURS_VARIANTS; do
    room_for 2 || { log "step 2: no room for the ours $v on/off pair (cap/DEADLINE): ours $v and later skipped"; break; }
    for arm in moveon moveoff; do
      [ "$DRY" = 1 ] && echo "  s2-ours-$arm-$v (extra: ${EXTRA[$v]}; drop-in env ${OURS_ENV[*]})"
      s2_run "s2-ours-$arm-$v" $CF/WA-S2-ours-$arm.json $PROD "$v" "${OURS_ENV[@]}" || break 2; done; done
  for v in $S2_STOCK_VARIANTS; do
    room_for 3 || { log "step 2: no room for the stock $v triple (cap/DEADLINE): stock $v and later skipped"; break; }
    for arm in nomove move movepool; do
      [ "$DRY" = 1 ] && echo "  s2-stock-$arm-$v (extra: ${EXTRA[$v]})"
      s2_run "s2-stock-$arm-$v" $CF/WA-S2-stock-$arm.json $STOCK "$v" || break 2; done; done
  [ "$DRY" = 1 ] && { echo "    + summary (PASS/FAIL, reuse, tokens vs s2-stock-nomove-<v> and ours on vs off) -> results/wa/s2-summary.txt"; return; }
  "$PY" -I - "$R" > "$R/s2-summary.txt" 2>&1 <<'PY'
import json, re, sys
from pathlib import Path
R = Path(sys.argv[1])
for p in sorted(R.glob("s2-*.out")):
    t = p.read_text(errors="replace")
    res = (re.findall(r"^(PASS|FAIL)$", t, re.M) or ["NO-RESULT"])[-1]
    req = next((l for l in t.splitlines() if l.startswith("requests")), "")
    gb = next((l for l in t.splitlines() if l.startswith("slot gave back")), "")
    print(f"{p.stem:28s} {res:9s} {req} | {gb}")
def load(n):
    p = R / f"{n}.json"
    return json.loads(p.read_text()) if p.exists() else None
for p in sorted(R.glob("s2-*.json")):
    tag = p.stem; v = tag.rsplit("-", 1)[1]
    for ref in (f"s2-stock-nomove-{v}", f"s2-ours-moveoff-{v}"):
        a, b = load(ref), load(tag)
        if ref == tag or not a or not b or (ref.startswith("s2-ours") != tag.startswith("s2-ours")):
            continue
        print(f"tokens {tag} vs {ref}: per round identical {[ra['answers'] == rb['answers'] for ra, rb in zip(a, b)]}")
for p in sorted(R.glob("s2-stock-*.notes")):
    s = [l.strip() for l in p.read_text(errors="replace").splitlines() if "is ignored" in l or "MMQ_GROUP_SORT" in l]
    if s: print(f"{p.stem}: inert switches: {' '.join(sorted(set(s)))}")
PY
  cat "$R/s2-summary.txt" | tee -a "$R/window.log"; }

# ---------------------------------------------------------------- step 3: merged switches on stock (BATCHING.md arms)
step3() {
  log "step 3: stock 8 slots / 16 slots / 16 slots + merged switches without MOVE ($S3_ROUNDS round(s))"
  local r arm lab cfg order
  local -a ORD3=("p8 p16 p16sw" "p16 p16sw p8" "p16sw p8 p16")   # rounds > 1 rotate the order (drift, warm-up)
  for r in $(seq "$S3_ROUNDS"); do order=${ORD3[$(( (r - 1) % 3 ))]}
    [ "$DRY" = 1 ] && [ "$S3_ROUNDS" -gt 1 ] && echo "  round $r: $order"
    for arm in $order; do
    lab=wa-s3-$arm-r$r; cfg=$CF/WA-S3-stock-$arm.json
    if [ "$arm" = p16sw ]; then start_srv $lab $cfg $STOCK "${S3_SW_ENV[@]}" || continue
    else start_srv $lab $cfg $STOCK || continue; fi
    conc $lab-short 900 --levels 1,8,16 --rounds 2 --max-tok 512 --prompt-tok 100 --temperature 0
    conc $lab-long 600 --levels 16 --rounds 2 --max-tok 512 --temperature 0 --warmup 0
    mt $lab-mt
    srv_done $lab $cfg; done; done
  [ "$DRY" = 1 ] && { echo "    + identity: compare_replies.py short/long + multiturn sha, p8 vs p16, p16 vs p16sw -> results/wa/s3-identity.txt"; return; }
  { for k in short long; do for pr in "p8 p16" "p16 p16sw"; do set -- $pr
      echo "$k $1 vs $2: $("$PY" -I compare_replies.py "$R/sweep.requests.jsonl" wa-s3-$1-r1-$k wa-s3-$2-r1-$k --show 3 | tail -3 | tr '\n' ' ')"; done; done
    mt_same wa-s3-p8-r1-mt wa-s3-p16-r1-mt; mt_same wa-s3-p16-r1-mt wa-s3-p16sw-r1-mt
    echo "p16sw switches the engine did not apply (setup.py release build: STRATA_MMQ_TAB=OFF):$(inert "$R/wa-s3-p16sw-r1.engine.log" $CF/WA-S3-stock-p16sw.json || echo ' none reported')"; } > "$R/s3-identity.txt" 2>&1
  cat "$R/s3-identity.txt" | tee -a "$R/window.log"; }

# ---------------------------------------------------------------- step 4: #2326 STRATA_PIPED_CONCURRENT on 4 stages
s4_reads() { [ "$DRY" = 1 ] || grep -a -E 'read [0-9]+ tokens on a thread of its own' "$R/$1.engine.log" 2>/dev/null | sed "s/^/$1: /" >> "$R/s4-reads.txt"; }
step4() {
  log "step 4: #2326 stock 16 slots, PIPED_CONCURRENT unset / =1 / PIPED_READ_PIECE=16384, $S4_ROUNDS interleaved rounds of $S4_SCEN"
  local r arm lab cfg order rc
  local -a ORD=("unset piece conc" "piece conc unset" "conc unset piece")
  for r in $(seq "$S4_ROUNDS"); do order=${ORD[$(( (r - 1) % 3 ))]}
    [ "$DRY" = 1 ] && echo "  round $r: $order"
    for arm in $order; do
      lab=wa-s4-$arm-r$r; cfg=$CF/WA-S4-stock-$arm.json
      start_srv $lab $cfg $STOCK || continue
      [ "$r" = 1 ] && step 300 "$R/wa-s4-$arm-det.json" "$PY" -I $TOOLS/piped_read_join_bench.py --port $SP --determinism
      step 1200 "$R/$lab.out" "$PY" -I $TOOLS/piped_read_join_bench.py --port $SP --scenarios "$S4_SCEN" --out "$R/$lab.json"
      rc=$?; [ "$DRY" = 1 ] || log "$lab piped bench rc $rc: $(grep -c '"scenario"' "$R/$lab.out") scenario lines"
      srv_done $lab $cfg; s4_reads $lab; done; done
  # bench_multiturn + bench_agentic per arm, a fresh server each (before the repeats: a hang there cannot eat them)
  for arm in unset conc piece; do
    lab=wa-s4-$arm-mtag; cfg=$CF/WA-S4-stock-$arm.json
    start_srv $lab $cfg $STOCK || continue
    mt $lab-mt; ag $lab-agentic
    srv_done $lab $cfg; s4_reads $lab; done
  # S4_REPEATS repeats with CONCURRENT=1: each its own timeout (a timeout = a hang, other rc != 0 = an error). After any
  # rc != 0 the server is restarted (a wedged engine still answers /health "loaded": the process is alive), its engine
  # log kept as <lab>-<i>.engine.log (the stall dump); the repeats stop after S4_MAX_HANGS hangs, at the step cap or DEADLINE.
  local i rc hang=0 err=0 okc=0 rest=0 up=0 hung="" errs=""
  lab=wa-s4-conc-rep; cfg=$CF/WA-S4-stock-conc.json
  [ "$DRY" = 1 ] && echo "  $S4_REPEATS repeats with STRATA_PIPED_CONCURRENT=1 (one server, restarted after any rc != 0; stop after $S4_MAX_HANGS hangs or at the step cap; shown once)"
  if start_srv $lab $cfg $STOCK; then up=1
    for i in $(seq "$S4_REPEATS"); do
      if [ "$DRY" != 1 ]; then
        [ "$up" = 1 ] || { start_srv $lab $cfg $STOCK && up=1; } || break
        [ "$(date +%s)" -ge "$STEP_END" ] && { log "step cap: repeats stop at $((i - 1))"; break; }
      fi
      step 600 "$R/$lab$i.out" "$PY" -I $TOOLS/piped_read_join_bench.py --port $SP --scenarios "$S4_SCEN" --out "$R/$lab$i.json"
      rc=$?; [ "$DRY" = 1 ] && break
      [ "$rc" = 99 ] && break                                           # not run: DEADLINE
      if [ "$rc" = 124 ] && [ "$CUT" = 1 ]; then log "$lab$i cut at DEADLINE (not counted)"; break; fi
      if [ "$rc" = 0 ]; then okc=$((okc + 1)); log "$lab$i ok"
      elif [ "$rc" = 124 ]; then hang=$((hang + 1)); hung="$hung $i"; log "$lab: repeat $i hung (600 s timeout)"
      else err=$((err + 1)); errs="$errs $i"; log "$lab: repeat $i error (rc $rc): $(tail -1 "$R/$lab$i.out" | cut -c1-120)"; fi
      if [ "$rc" != 0 ] || ! healthy; then
        log "$lab: restart after repeat $i"; rest=$((rest + 1)); srv_done $lab-$i $cfg; s4_reads $lab-$i; up=0
        [ "$hang" -ge "$S4_MAX_HANGS" ] && { log "$lab: $hang hangs: the remaining repeats are skipped"; break; }
      fi
    done
    [ "$up" = 1 ] && { srv_done $lab $cfg; s4_reads $lab; }
    [ "$DRY" = 1 ] || log "s4 CONCURRENT=1 repeats: $okc ok, $hang hang(s)${hung:+ (repeat$hung)}, $err error(s)${errs:+ (repeat$errs)}," \
      "$rest server restart(s); stall/no-progress lines: $(cat "$R"/$lab*.engine.log 2>/dev/null | grep -a -c -E 'no progress for|no step finished|stalled|stall report')," \
      "engine starts: $(cat "$R"/$lab*.engine.log 2>/dev/null | grep -a -c 'engine started')"
  fi
  [ "$DRY" = 1 ] && { echo "    + summary: scenarios per arm/round, determinism sha, multiturn sha, read lines -> results/wa/s4-summary.txt"; return; }
  { "$PY" -I - "$R" <<'PY'
import json, sys
from pathlib import Path
R = Path(sys.argv[1])
print(f"{'run':22s} {'cold32k stop':>12s} {'cold32k TTFT':>12s} {'2j stop':>8s} {'2j TTFT':>14s} {'fup TTFT':>8s} {'45K TTFT':>8s} {'agg tok/s':>18s}")
import re
runs = sorted(p for p in R.glob("wa-s4-*.json") if re.search(r"-r\d+$", p.stem))
for p in runs + sorted(R.glob("wa-s4-conc-rep*.json")):
    try:
        d = json.loads(p.read_text())
    except ValueError:
        print(p.stem, "unreadable"); continue
    c, t, f = d.get("cold32k", {}), d.get("two_joiners", {}), d.get("followup", {})
    print(f"{p.stem:22s} {c.get('worst_max_gap_s', '-'):>12} {str(c.get('joiner_ttft_s', '-')):>12} {t.get('worst_max_gap_s', '-'):>8} "
          f"{str(t.get('joiner_ttft_s', '-')):>14} {f.get('followup_ttft_s', '-'):>8} {f.get('long_ttft_s', '-'):>8} "
          f"{str([c.get('aggregate_tok_s'), t.get('aggregate_tok_s')]):>18}")
det = {}
for arm in ("unset", "piece", "conc"):
    try:
        det[arm] = [r["answer_sha256"] for r in json.loads((R / f"wa-s4-{arm}-det.json").read_text())]
    except (OSError, ValueError, KeyError):
        det[arm] = None
for arm in ("piece", "conc"):
    a, b = det["unset"], det[arm]
    print(f"determinism unset vs {arm} (6 sequential prompts, nothing decoding beside them: the plain path only, not a concurrent read):", "missing" if not a or not b else f"{sum(x == y for x, y in zip(a, b))}/{len(a)} identical")
PY
    echo "informational (STRATA_IQ_MT_MIN=1 in all arms; under PIECE/CONCURRENT the slots compute the experts a read borrows on the CPU, so a difference is expected, not a failure):"
    mt_same wa-s4-unset-mtag-mt wa-s4-piece-mtag-mt; mt_same wa-s4-unset-mtag-mt wa-s4-conc-mtag-mt
    echo "read-on-a-thread lines: $(wc -l < "$R/s4-reads.txt" 2>/dev/null || echo 0) (results/wa/s4-reads.txt)"; } > "$R/s4-summary.txt" 2>&1
  cat "$R/s4-summary.txt" | tee -a "$R/window.log"; }

# ---------------------------------------------------------------- step 5: ours (production, text only) vs best stock 0.1.45
# The user's gate for posting upstream: our production build must stay ahead of the best stock 0.1.45 setup
# (16 slots + every merged switch + STRATA_PIPED_CONCURRENT=1; falls back to the S3 p16sw config if that does not start).
# Same benches and requests for both arms: bench_concurrency short 1/2/4/8/16 + 1.2K 4/16 (greedy), bench_multiturn 16x3
# (greedy), bench_agentic. Results: wa-s5-*.txt, s5-summary.txt.
step5() {
  log "step 5: ours (production FE, text only) vs best stock 0.1.45 (16 slots + merged switches + PIPED_CONCURRENT)"
  local arm lab cfg
  for arm in ours stock-best; do
    lab=wa-s5-$arm; cfg=$CF/WA-S5-$arm.json
    if [ "$arm" = ours ]; then start_srv $lab $cfg $PROD "${OURS_ENV[@]}" || continue
    else
      if ! start_srv $lab $cfg $STOCK "${S3_SW_ENV[@]}"; then
        log "$lab: did not start, falling back to WA-S3-stock-p16sw (no PIPED_CONCURRENT)"
        lab=wa-s5-stock-p16sw; cfg=$CF/WA-S3-stock-p16sw.json; start_srv $lab $cfg $STOCK "${S3_SW_ENV[@]}" || continue
      fi
    fi
    conc $lab-short 900 --levels 1,2,4,8,16 --rounds 2 --max-tok 512 --prompt-tok 100 --temperature 0
    conc $lab-long 600 --levels 4,16 --rounds 2 --max-tok 512 --temperature 0 --warmup 0
    mt $lab-mt
    ag $lab-ag
    srv_done $lab $cfg; done
  [ "$DRY" = 1 ] && { echo "    + summary: ours vs stock-best per level (agg tok/s, TTFT p50/p95/max), multiturn follow-up TTFT, agentic wall/TTFT -> results/wa/s5-summary.txt"; return; }
  { for k in short long; do echo "== $k"; for a in ours stock-best stock-p16sw; do [ -f "$R/wa-s5-$a-$k.txt" ] && grep -E '^ +[0-9]+ ' "$R/wa-s5-$a-$k.txt" | awk -v a=$a '{printf "%-11s %3s users agg %7s  TTFT p50 %6s p95 %6s max %6s\n", a, $1, $4, $7, $8, $9}'; done; done
    echo "== multiturn"; for a in ours stock-best stock-p16sw; do [ -f "$R/wa-s5-$a-mt.txt" ] && echo "$a: $(tail -1 "$R/wa-s5-$a-mt.txt")"; done
    echo "== agentic"; for a in ours stock-best stock-p16sw; do [ -f "$R/wa-s5-$a-ag.txt" ] && { echo "$a:"; tail -5 "$R/wa-s5-$a-ag.txt"; }; done; } > "$R/s5-summary.txt" 2>&1
  cat "$R/s5-summary.txt" | tee -a "$R/window.log"; }

# ---------------------------------------------------------------- main
FIRST=${STEPS%% *}; FE=EST$FIRST; NEED_FIRST=$(( (${!FE} + 5) * 60 ))   # production is stopped only if the first step fits
arm_restore() {   # an independent backstop outside this process: starts production at RESTORE_TS even after a SIGKILL
  systemctl --user stop wa-restore.timer 2>/dev/null; systemctl --user reset-failed wa-restore.timer wa-restore.service 2>/dev/null
  systemd-run --user --quiet --unit=wa-restore --on-calendar="$(date -d "@$RESTORE_TS" '+%F %T')" \
    --timer-property=AccuracySec=1s /bin/sh -c 'systemctl --user stop strata-bench.scope; systemctl --user start strata.service' \
    && systemctl --user is-active -q wa-restore.timer; }
if [ "$DRY" = 1 ]; then
  echo "=== window A, DRY RUN: nothing is started or stopped; steps: $STEPS"
  echo "preflight:"; preflight || echo "  (preflight would ABORT the real run before production is stopped)"
  echo "would: wait for production loaded; ABORT without touching production unless >= $(( NEED_FIRST / 60 )) min (step $FIRST + 5) remain before $DEADLINE;"
  echo "       arm wa-restore.timer ($(date -d @$RESTORE_TS +%H:%M): stop strata-bench.scope, start strata.service; removed by the normal restore),"
  echo "       start the HARD_STOP watchdog ($HARD_STOP), systemctl --user stop strata.service, check GPUs free"
  tot=0; for s in $STEPS; do e=EST$s; tot=$(( tot + ${!e} )); done
  echo "expected: ~$tot min for steps '$STEPS' + ~5 min production stop/start -> start by $(date -d "@$(( DL_TS - (tot + 5) * 60 ))" +%H:%M) for the full window;"
  echo "          later starts lose the tail (runs are cut at $DEADLINE). Production loaded again by ~$(date -d @$(( DL_TS + 180 )) +%H:%M) at the latest"
  echo "          (DEADLINE + bench stop + ~100 s load); HARD_STOP $HARD_STOP and the restore timer $(date -d @$RESTORE_TS +%H:%M) are backstops only"
else
  mkdir -p "$R/logs"
  log "=== window A start (steps: $STEPS; deadline $DEADLINE, hard stop $HARD_STOP, restore timer $(date -d @$RESTORE_TS +%H:%M))"
  preflight 2>&1 | tee -a "$R/window.log"; [ "${PIPESTATUS[0]}" = 0 ] || die "preflight failed"
  prod_loaded || log "warning: production was not loaded before the window"
  left=$(( DL_TS - $(date +%s) ))
  [ "$left" -ge "$NEED_FIRST" ] || die "only $(( left / 60 )) min left before $DEADLINE, step $FIRST needs ~$(( NEED_FIRST / 60 )): production left running"
  ss -ltn 2>/dev/null | grep -q ":$SP " && die "port $SP in use"
  systemctl --user is-active -q strata-bench.scope && die "strata-bench.scope is in use (another bench?)"
  arm_restore || die "could not arm wa-restore.timer: production left running"
  log "wa-restore.timer armed for $(date -d @$RESTORE_TS '+%F %T')"
  trap back EXIT; trap on_sig INT TERM
  ( sleep $(( HS_TS - $(date +%s) )); kill -TERM $$ ) 2>/dev/null & WD=$!   # HARD_STOP: ends the window, restores production
  log "stopping strata.service"; systemctl --user stop strata.service; gpus_free || { log "GPUs not free"; exit 1; }
fi
T0=$(date +%s); DUR=""
for s in $STEPS; do
  [ "$ABORT" = 1 ] && { log "step $s skipped (deadline)"; continue; }
  [ "$DRY" = 1 ] || { past_deadline && { log "DEADLINE $DEADLINE passed: step $s and the rest skipped"; ABORT=1; continue; }; }
  cap=CAP$s; est=EST$s; STEP_END=$(( $(date +%s) + ${!cap} )); ts=$(date +%s)
  if [ "$DRY" = 1 ]; then echo; echo "--- step $s (expected ~${!est} min, cap $(( ${!cap} / 60 )) min)"
  else log "step $s: expected ~${!est} min, cap $(( ${!cap} / 60 )) min"; fi
  step$s
  DUR="$DUR step$s=$(( ($(date +%s) - ts) / 60 ))min"
done
[ "$DRY" = 1 ] && { echo; echo "would: stop strata-bench.scope, check GPUs free, systemctl --user start strata.service, wait for it to load (on every exit path)"; exit 0; }
log "=== window A done after $(( ($(date +%s) - T0) / 60 )) min:$DUR"
