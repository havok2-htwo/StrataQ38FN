#!/usr/bin/env bash
# GPU window "borrow" (~110 min of production outage planned, cap WINDOW_S, default 7800 s = 130 min): the ll10-borrow engine patch set
# (Strata-ll10-borrow/build/strata, a copy pinned by sha256 at prep) against production as it runs, arm by arm, on the
# Genesis ping-pong that loses production's cache (main agent ~136k + check call + a ~29.5k sub-agent sharing the 13k
# tool head; repro_park_ram.py pingpong) and on the RAM-gate interleave case (repro_park_ram.py interleave).
#   P   production as it runs: production.json read now (exe, args, env), server Strata-143-prod + the drop-in's STRATA_*
#   N0  new binary, every new switch off: STRATA_SERVE_STOP_BOUND=0, no STRATA_PARK_BORROW / CONV_MMAP / GATE_EVICT,
#       ALWAYS on Strata-143-prod (P's server: P and N0 differ in the binary only; must equal P: greedy identity PASS/FAIL)
#   S0  (opt-in, ARMS="P N0 S0 ..."; needs --server-tree) N0 on the --server-tree server: S0 vs N0 = the server alone
#   N1  new binary, STRATA_SERVE_STOP_BOUND as its default (on: a request commits only the rows it can emit)
#   N2  N1 + STRATA_PARK_BORROW=1
#   N3  N2 + STRATA_CONV_MMAP=1 STRATA_GATE_EVICT=1 MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=1048576
#   N4  N2 + STRATA_READS=$READS_N (stage 2 background reads, TESTLOG #88, with its starvation fix: ll7-reads2 063af1fd =
#       ll10-borrow 23588120) + server STRATA_SERVER_READS=1 STRATA_READS_BG_MIN=$READS_BG_MIN (#88's agent-mix READS arm,
#       where it failed; w1/w2's long reads are >= 105K, its shorts and streams < 4096: there 4096 = #88's 8192).  Needs
#       --server-tree with STRATA_SERVER_READS (Strata-srv-1106 >= 22f4781f).  FAIR (STRATA_READ_YIELD=1 + the drop-in's
#       STRATA_YIELD_SHORT) stays on as in N2 (#88 measured READS without it); STRATA_READS_SOLO stays 0
# N1..N4 run the server from Strata-143-prod too, or from --server-tree DIR (e.g. Strata-srv-1106; P and N0 always run
# Strata-143-prod, so with --server-tree N1 vs N0 is the stop bound + the server; S0 separates the two).  Per arm, one fresh server start (empty tier dir, its own engine log), in this order:
#   1 repro_park_ram.py pingpong @136000, 2 cycles (A2, A3: reuse target >= 95 %; SUB/SUB2 reuse; borrowed / restored /
#     superseded lines; park capture_ms / wait_ms), --pressure $PP_PRESSURE                                  ~3-5 min
#   2 repro_park_ram.py interleave @136000 (16 fillers, B in A's slot, A2), --pressure $IL_PRESSURE (cgroup: the test
#     scope's MemoryMax at usage + 18102 MiB, production's RAM situation of 2026-11-06)                      ~3-4 min
#   3 bench_multiturn.py 16 users x 3 turns greedy (--doc $MT_DOC): reply sha per (uid, turn) = the identity check over
#     the parked/restored path (wall 86-96 s in #88)                                                         ~2 min
#   4 bench_concurrency.py 1/8/16 users short (2 rounds, 128-512 tokens, greedy): throughput + reply sha per uid
#     (compare_replies.py vs P = the identity check over the batch path; 5 + ~15 + 22 s in #88/#89)             ~1.5 min
#   + engine start ~2 min; the engine log's conversation-cache lines (counts, capture_ms/wait_ms) and the scope's peak RSS
# Then the reads step for each arm of READS_ARMS (default "N2 N4", those in ARMS), after all arms, back to back: one
# more fresh server start (empty tier dir, engine-rd-<arm>.log) with
#   R1 fair_tools.py --mode scen --scenario win --rep $WIN_REP (TESTLOG #88's w1: a cold ~125K read beside 2 decoding
#      streams, 4 shorts at +3/6/9/12 s; w2: a cold ~105K read, a burst of 16 shorts at +4 s; one rep: 59 s in #88)
#   R2 bench_agentic.py greedy (the agent mix where READS failed in #88: wall 116.8 -> 142-148 s)  ~6 min incl. the start
# The summary judges N4 vs N2 as #88 did (win_vs_fair --gain 1.0 --long-tol 0.05 + the agent mix): every w1 short's
# TTFT (max) and the time to all 16 w2 first tokens <= 1.00 x N2's, the long reads' totals <= 1.05 x, bench_agentic's
# and bench_multiturn's wall <= 1.05 x, every reply identical (win, bench_agentic per request, bench_multiturn), N4's
# reads went to the background -> PASS / FAIL.  When the two reads steps no longer fit, the last one (N4) runs alone.
# Per arm ~11-12 min by those parts (ARM_EST 15 min is the skip threshold), the reads step ~6 min (READS_EST 7): 6 arms
# + 2 reads steps ~80 min, by the thresholds ~104 min, + production stop/start ~4 min (cap WINDOW_S 130 min).  A step
# that no longer fits the window is skipped (logged); ARMS="P N0 N2" runs a subset (P is needed for the identity verdicts).
# Isolation: test servers only on 127.0.0.1:8091 in strata-bench.scope (MemoryMax 224G as production's unit), tier dirs
# under /media/raid0/strata-kvspill-bench/borrow/<run> (never production's), results in results/borrow (an existing one
# moves to results/borrow.prev-<time>), production.json / the unit / its drop-in are only read.  Production is stopped
# only once it reports loaded and started again on EVERY exit path: the EXIT trap (normal end, error, INT/TERM/HUP, the
# HARD_STOP watchdog at WINDOW_S + 5 min); for a SIGKILL / OOM kill of this script the transient unit borrow-guard.service
# (waits for this script's PID to end: kills the running step's process group, stops strata-bench.scope, starts
# strata.service - within seconds) and, independent of both, the transient timer borrow-restore.timer (WINDOW_S + 15 min
# after the stop: stops strata-bench.scope, starts strata.service); the normal restore removes both.  While production
# is stopped the script sends nothing to port 8090 except the restore's own health polls.  The summary table is printed
# after production is back.
#   DRY=1 ./window-borrow.sh [--server-tree DIR]   # prep checks + configs + the plan; production untouched
#   ./window-borrow.sh [--server-tree DIR]         # the window (needs the user's GPU-window approval)
#   ARMS="P N0 N1 N2 N3 N4" READS_ARMS="N2 N4" WINDOW_S=7800 PP_PRESSURE=none IL_PRESSURE=cgroup ./window-borrow.sh
#   stage 2 on the default (TESTLOG #96): ARMS="N3 N4" READS_ARMS="N3 N4" READS_BASE=N3 READS_BG_MIN=32768 WINDOW_S=3300 \
#       BASE_SRV=$R/Strata-143-base ./window-borrow.sh --server-tree $R/Strata-srv-1106   (N4 = N3 + reads, judged vs N3)
#   verdicts: grep -E 'PASS|FAIL|SKIP' results/borrow/window.log; table: results/borrow/summary.txt
set -u
R=/media/raid0/StrataQ38FN; L=$R/linux; C=$L/configs
URL=http://127.0.0.1:8090; TPORT=8091; PY=$R/Strata/.venv/bin/python
NEWSRC=$R/Strata-ll10-borrow/build/strata                  # being built in parallel: pinned by a copy at prep
SRV=${BASE_SRV:-$R/Strata-143-prod}                         # P's and N0's server checkout (default: production's; and N by default)
# BASE_CONFIG / BASE_SRV: the baseline the arms are built from when production already runs the new build (2026-11-06:
# BASE_CONFIG=configs/T16-1441j-fair-g8-mmap4m-prod262r-vision-cpu.json BASE_SRV=$R/Strata-143-base @90f73293); the
# safety checks (which engine runs now, what is restored) always read production.json
NSRV=$SRV
DRY=${DRY:-0}
while [ $# -gt 0 ]; do
    case $1 in
        --server-tree) NSRV=$(readlink -f "${2:?--server-tree needs a directory}"); shift 2 ;;
        --server-tree=*) NSRV=$(readlink -f "${1#*=}"); shift ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) sed -n '2,/^set -u/p' "$0" | sed '$d'; exit 0 ;;
        *) echo "unknown argument $1 (--server-tree DIR, --dry-run)"; exit 2 ;;
    esac
done
ARMS=${ARMS:-P N0 N1 N2 N3 N4}
WINDOW_S=${WINDOW_S:-7800}; RESERVE=480                     # outage cap (stop -> restart), room kept for the restart
ARM_EST=${ARM_EST:-900}; ARM_CAP=${ARM_CAP:-2100}           # per arm: expected / hard cap (s)
READS_ARMS=${READS_ARMS-N2 N4}                              # the arms that also get the reads step (those in ARMS)
READS_EST=${READS_EST:-420}; READS_CAP=${READS_CAP:-1200}   # the reads step: expected / hard cap (s)
READS_N=${READS_N:-2}; READS_BG_MIN=${READS_BG_MIN:-4096}   # N4: engine STRATA_READS, server STRATA_READS_BG_MIN
READS_BASE=${READS_BASE:-N2}                                # N4 = READS_BASE + stage 2 (N2: night of 11-07; N3: the default since)
WIN_REP=${WIN_REP:-1}                                       # fair_tools win --rep (#88's rep 1 prompts)
FT=$L/fair_tools.py; NEWTREE=$R/Strata-ll10-borrow
# the starvation fix (#88: a held read waited 4 x (reads) picks; 063af1fd) adds no string to the binary: its source
# line must be in NEWTREE's HEAD and the binary built after the commit that brought it
READS_FIX_SRC='rs::next(view, bg_pick, (int64_t) bg_reads.size())'

SIZE=${SIZE:-136000}; CYCLES=${CYCLES:-2}
PP_PRESSURE=${PP_PRESSURE:-none}; IL_PRESSURE=${IL_PRESSURE:-cgroup}
PP_FILLERS=${PP_FILLERS:-8}; IL_FILLERS=${IL_FILLERS:-16}
MT_DOC=${MT_DOC:-8000}
MEMMAX=${MEMMAX:-224G}
ALLOW_MISSING=${ALLOW_MISSING:-0}                           # 1: drop the arms whose switch the binary lacks
if [ "$DRY" = 1 ]; then OUT=$L/results/borrow-dry; else OUT=$L/results/borrow; fi
CF=$OUT/cfg; BIN=$OUT/bin; TIER=/media/raid0/strata-kvspill-bench/borrow
RP=$L/repro_park_ram.py
TAG=bw$(date +%m%d%H%M)                                     # label prefix (results/sweep.jsonl is shared)
RUNIT=borrow-restore; GUNIT=borrow-guard
for v in $(compgen -e | grep '^STRATA_'); do unset "$v"; done   # nothing of ours from the calling shell
# until production is stopped a signal only ends the script (nothing to clean up; production keeps serving)
trap 'echo "window aborted before production was stopped"; exit 1' INT TERM HUP

log() { echo "[$(date +%T)] $*" | tee -a "$OUT/window.log"; }
gpu_apps() { nvidia-smi --query-compute-apps=pid --format=csv,noheader; }
vram() { nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | tr '\n' ';'; }
# kills only engines of $R/Strata* and the pinned copy under $BIN (production is stopped when this runs)
free_gpus() { local i p e; for i in $(seq 45); do [ -z "$(gpu_apps)" ] && return 0; sleep 2; done
    for p in $(gpu_apps); do e=$(readlink -f "/proc/$p/exe" 2>/dev/null)
        case "$e" in "$R"/Strata*|"$BIN"/*) kill "$p" ;; esac; done
    for i in $(seq 30); do [ -z "$(gpu_apps)" ] && return 0; sleep 2; done; return 1; }
prod_loaded() { curl -s -m 3 "$URL/health" | grep -q '"loaded": *true'; }
start_prod() { log "starting production as before (production.json -> $(readlink "$C/production.json"))"
    systemctl --user start strata.service; local t0; t0=$(date +%s)
    until prod_loaded; do
        systemctl --user is-active -q strata.service || { log "strata.service died during the start FAIL"; return 1; }
        [ $(( $(date +%s) - t0 )) -gt 900 ] && { log "production start timeout FAIL"; return 1; }; sleep 5; done
    log "production loaded after $(( $(date +%s) - t0 )) s: $(curl -s -m 5 "$URL/health" | cut -c1-200)"; }
bench_down() { systemctl --user stop strata-bench.scope 2>/dev/null; systemctl --user reset-failed strata-bench.scope 2>/dev/null
    free_gpus || log "GPUs busy after $1"; }
disarm_restore() { systemctl --user stop "$RUNIT.timer" 2>/dev/null
    systemctl --user reset-failed "$RUNIT.timer" "$RUNIT.service" 2>/dev/null; }
arm_restore() {   # an independent backstop outside this process: starts production at $1 even after a SIGKILL
    disarm_restore
    systemd-run --user --quiet --unit="$RUNIT" --on-calendar="$(date -d "@$1" '+%F %T')" \
        --timer-property=AccuracySec=1s /bin/sh -c 'systemctl --user stop strata-bench.scope; systemctl --user start strata.service' \
        && systemctl --user is-active -q "$RUNIT.timer"; }
disarm_guard() { systemctl --user stop "$GUNIT.service" 2>/dev/null; systemctl --user reset-failed "$GUNIT.service" 2>/dev/null; }
arm_guard() {   # restores production within seconds when this script dies without its EXIT trap (SIGKILL, OOM kill):
    # waits for this PID, kills the step's process group (a 'timeout' leader, $OUT/step.pgid), the bench scope, starts
    # production.  Checked to be still waiting after 2 s (a PID it cannot see would fire it at once: then no window)
    disarm_guard
    systemd-run --user --quiet --collect --unit="$GUNIT" /bin/sh -c 'tail --pid="$1" -f /dev/null
        g=$(cat "$2" 2>/dev/null); [ -n "$g" ] && [ "$(ps -o comm= -p "$g" 2>/dev/null)" = timeout ] && /bin/kill -s KILL -- "-$g"
        systemctl --user stop strata-bench.scope; systemctl --user start strata.service' sh "$$" "$OUT/step.pgid" || return 1
    sleep 2; systemctl --user is-active -q "$GUNIT.service"; }

STOPPED=0; STEP_PID=; WD=; BACK_DONE=0; T0=$(date +%s)
# back: production again (idempotent; every exit path).  The step's timeout leads its own process group: INT to the
# group, a short wait, KILL; then the bench scope, the GPUs, strata.service; signals are ignored from here on
back() { [ "$BACK_DONE" = 1 ] && return 0; BACK_DONE=1; trap '' INT TERM HUP
    [ -n "$WD" ] && { pkill -P "$WD" 2>/dev/null; kill "$WD" 2>/dev/null; }
    if [ -n "$STEP_PID" ] && kill -0 "$STEP_PID" 2>/dev/null; then
        kill -INT -- -"$STEP_PID" 2>/dev/null || kill -INT "$STEP_PID" 2>/dev/null
        for _ in $(seq 30); do kill -0 "$STEP_PID" 2>/dev/null || break; sleep 1; done
        kill -KILL -- -"$STEP_PID" 2>/dev/null; fi
    [ "$STOPPED" = 1 ] || return 0
    bench_down end
    if start_prod || { log "production start failed: one more try"; systemctl --user restart strata.service; start_prod; }; then
        disarm_restore; disarm_guard; log "$RUNIT.timer and $GUNIT.service removed"
    else log "PRODUCTION DID NOT COME BACK - check strata.service now ($RUNIT.timer and $GUNIT.service left armed)"; fi
    local P PE NOWCFG; P=$(systemctl --user show -p MainPID --value strata.service)
    PE=$(for c in $(pgrep -P "$P" 2>/dev/null); do readlink "/proc/$c/exe"; done | tr '\n' ' ')
    NOWCFG=$(sha256sum "$(readlink -f "$C/production.json")" | cut -c1-16)
    log "    [prod] ==> production.json sha256 $NOWCFG (at the start $PRODCFG_SHA), engine running: ${PE:-?}: $([ "$NOWCFG" = "$PRODCFG_SHA" ] && echo "$PE" | grep -q "$(basename "$PEXE")" && echo PASS || echo FAIL)"
    rm -rf "$TIER"
    log "window done after $(( ($(date +%s) - T0) / 60 )) min of production outage"; }
final() { back; [ "$STOPPED" = 1 ] && { summary 2>&1 | tee "$OUT/summary.txt" | tee -a "$OUT/window.log"
    grep -hE 'PASS|FAIL|SKIP' "$OUT/window.log" | grep -v '^\s*$' | tail -80; }; }
on_sig() { log "window aborted: ${1:-signal}"; exit 130; }    # -> EXIT trap -> final (back + summary)
left() { echo $(( T0 + WINDOW_S - RESERVE - $(date +%s) )); }
# step <name> <expected s> <cap s> <cmd...>: skipped when the expected time no longer fits the window; the cap is cut
# to what is left; the output goes to $OUT/<name>.txt and its '==>' verdict lines into window.log
step() { local name=$1 est=$2 limit=$3; shift 3; local l rc; l=$(left)
    if [ "$l" -lt "$est" ]; then log "step $name: SKIP (needs ~${est} s, ${l} s left in the window)"; echo skip > "$OUT/rc_$name"; return 1; fi
    [ "$limit" -gt "$l" ] && limit=$l
    log "step $name: start (expected ~${est} s, cap ${limit} s); VRAM (index, used, free): $(vram)"
    timeout -s INT -k 120 "$limit" bash -c '"$@"' "$name" "$@" > "$OUT/$name.txt" 2>&1 < /dev/null &
    STEP_PID=$!; echo "$STEP_PID" > "$OUT/step.pgid"; wait "$STEP_PID"; rc=$?; STEP_PID=; : > "$OUT/step.pgid"
    echo $rc > "$OUT/rc_$name"
    [ $rc -eq 124 ] && log "step $name: TIMEOUT after ${limit} s FAIL"
    grep -aE '^==>' "$OUT/$name.txt" | cut -c1-700 | sed "s/^/    [$name] /" | tee -a "$OUT/window.log" >/dev/null
    log "step $name: exit $rc"
    bench_down "$name"; return $rc; }

# ---- helpers the arm runs (exported: step runs them under timeout through bash -c) ------------------------------------
tup() { curl -s -m 3 "http://127.0.0.1:$TPORT/health" | grep -q '"loaded": *true'; }
# runcfg <NAME> <run>: a copy of $CF/NAME.json for one server start - its own engine log $OUT/engine-<run>.log and a
# fresh, empty tier dir (STRATA_CONV_TIER_WARM would otherwise preload an earlier start's conversations); the .srv
# sidecar (SERVER_DIR + the server env) goes with it
runcfg() { local d=$TIER/$2 f=$CF/run-$2.json; rm -rf "$d"; mkdir -p "$d"; rm -f "$OUT/engine-$2.log"
    python3 -c "import json,sys; c=json.load(open(sys.argv[1])); c['env']['STRATA_CONV_TIER_DIR']=sys.argv[2]; c['log']=sys.argv[3]; json.dump(c,open(sys.argv[4],'w'),indent=1)" \
        "$CF/$1.json" "$d" "$OUT/engine-$2.log" "$f" && cp "$CF/$1.srv" "$CF/run-$2.srv" && echo "$f"; }
# srv <cfg> <cmd...>: the command with the config's server checkout and server env, on 127.0.0.1:$TPORT
srv() { local s=${1%.json}.srv; shift; ( cd "$L" && env $(cat "$s") PORT=$TPORT BIND=127.0.0.1 MEMMAX=$MEMMAX "$@" ); }
# up <label> <cfg>: a test server with a 1-request warm-up (KEEP=1 leaves it running)
up() { srv "$2" env KEEP=1 ./run_strata_variant.sh "$1" "$2" 1 --rounds 1 --max-tok 16 --prompt-tok 100 --temperature 0 2>&1 | tail -3
    tup || { echo "==> $1: the test server did not come up FAIL"; systemctl --user stop strata-bench.scope; return 1; }
    echo "server $(head -1 "$L/results/$1.txt" 2>/dev/null | cut -c1-250)"; }
rss_peak() { local cg; cg=$(systemctl --user show -p ControlGroup --value strata-bench.scope 2>/dev/null)
    if [ -n "$cg" ] && [ -r "/sys/fs/cgroup$cg/memory.peak" ]; then
        awk '{printf "%.2f", $1 / 1073741824}' "/sys/fs/cgroup$cg/memory.peak"; else echo "?"; fi; }
# arm_run <ARM>: one server start: pingpong, interleave, multiturn, short sweep; engine log lines; peak RSS
arm_run() { local a=$1 cfg rc=0 u=http://127.0.0.1:$TPORT elog lab=$TAG-$1
    cfg=$(runcfg "$a" "$a") || return 1; elog=$OUT/engine-$a.log
    up "$lab-up" "$cfg" || return 1
    timeout -s INT 1500 "$PY" -I "$RP" --url "$u" --label "$lab" --scenarios pingpong --sizes "$SIZE" --cycles "$CYCLES" \
        --fillers "$PP_FILLERS" --pressure "$PP_PRESSURE" --unit strata-bench.scope --engine-log "$elog" \
        --out "$OUT/repro.jsonl" > "$OUT/$a-pingpong.txt" 2>&1 || rc=1
    grep -aE '^==>' "$OUT/$a-pingpong.txt" || echo "==> $lab pingpong: no verdict (see $OUT/$a-pingpong.txt) FAIL"
    if tup; then
        timeout -s INT 1200 "$PY" -I "$RP" --url "$u" --label "$lab" --scenarios interleave --sizes "$SIZE" \
            --fillers "$IL_FILLERS" --pressure "$IL_PRESSURE" --unit strata-bench.scope --engine-log "$elog" \
            --out "$OUT/repro.jsonl" > "$OUT/$a-interleave.txt" 2>&1 || rc=1
        grep -aE '^==>' "$OUT/$a-interleave.txt" || echo "==> $lab interleave: no verdict (see $OUT/$a-interleave.txt) FAIL"
    else echo "==> $lab: the server is gone after pingpong FAIL"; return 1; fi
    if tup; then
        timeout -s INT 900 "$PY" -I "$L/bench_multiturn.py" --url "$u" --users 16 --turns 3 --doc "$MT_DOC" --temperature 0 \
            --label "$lab-mt" --out "$OUT/multiturn.jsonl" > "$OUT/$a-mt.txt" 2>&1 || rc=1
        echo "==> $lab multiturn: $(tail -1 "$OUT/$a-mt.txt" | cut -c1-200) INFO"
        srv "$cfg" env ATTACH=1 KEEP=1 ./run_strata_variant.sh "$lab-short" "$cfg" 1,8,16 --rounds 2 --max-tok 512 \
            --prompt-tok 100 --temperature 0 --max-tok-min 128 2>&1 | tail -4
    else echo "==> $lab: the server is gone after interleave FAIL"; rc=1; fi
    echo "$(rss_peak)" > "$OUT/rss-$a.txt"; echo "==> rss $a: peak $(cat "$OUT/rss-$a.txt") GiB (memory.peak) INFO"
    grep -a 'conversation cache' "$elog" > "$OUT/$a-cache-lines.txt" 2>/dev/null
    echo "==> $a engine: $(grep -ac 'conversation cache: restored' "$elog" 2>/dev/null) restored, $(grep -a 'conversation cache' "$elog" 2>/dev/null | grep -c borrow) borrow, $(grep -ac superseded "$elog" 2>/dev/null) superseded, $(grep -ac 'skip parking' "$elog" 2>/dev/null) skip-parking, $(grep -ac 'physical RAM admission' "$elog" 2>/dev/null) RAM refusals, $(grep -ac 'the prompt read goes to the background at' "$elog" 2>/dev/null) background reads INFO"
    sleep 2; systemctl --user stop strata-bench.scope; return $rc; }
# reads_run <ARM>: the stage 2 reads step - one more server start (fresh tier dir, engine-rd-<ARM>.log): fair_tools win
# (w1, w2: rep $WIN_REP -> win-<ARM>.json), then bench_agentic greedy (-> agentic.jsonl, label $TAG-<ARM>-agentic)
reads_run() { local a=$1 cfg rc=0 u=http://127.0.0.1:$TPORT elog lab=$TAG-$1
    cfg=$(runcfg "$a" "rd-$a") || return 1; elog=$OUT/engine-rd-$a.log
    up "$lab-rdup" "$cfg" || return 1
    timeout -s INT 900 "$PY" -I "$FT" --mode scen --config "$cfg" --url "$u" --scenario win --rep "$WIN_REP" \
        --label "$lab-win" --out "$OUT/win-$a.json" > "$OUT/$a-win.txt" 2>&1 || rc=1
    grep -a '^round ' "$OUT/$a-win.txt" | cut -c1-400 | sed "s/^/==> $lab win /; s/\$/ INFO/"
    grep -aE '^==>' "$OUT/$a-win.txt" || echo "==> $lab win: no verdict (see $OUT/$a-win.txt) FAIL"
    if tup; then
        timeout -s INT 900 "$PY" -I "$L/bench_agentic.py" --url "$u" --temperature 0 --label "$lab-agentic" \
            --out "$OUT/agentic.jsonl" > "$OUT/$a-agentic.txt" 2>&1 || rc=1
        echo "==> $lab agentic: $(tail -1 "$OUT/$a-agentic.txt" | cut -c1-200) INFO"
    else echo "==> $lab: the server is gone after the win scenario FAIL"; rc=1; fi
    echo "$(rss_peak)" > "$OUT/rss-rd-$a.txt"; echo "==> rss rd-$a: peak $(cat "$OUT/rss-rd-$a.txt") GiB (memory.peak) INFO"
    echo "==> rd-$a engine: $(grep -ac 'the prompt read goes to the background at' "$elog" 2>/dev/null) background reads, $(grep -ac "background read of slot .*'s request is ready" "$elog" 2>/dev/null) ready, $(grep -ac 'gives way at' "$elog" 2>/dev/null) give-ways, $(grep -ac 'engine started' "$elog" 2>/dev/null) engine starts INFO"
    sleep 2; systemctl --user stop strata-bench.scope; return $rc; }
export -f tup runcfg srv up rss_peak arm_run reads_run
export PY L CF OUT TIER TPORT TAG RP SIZE CYCLES PP_PRESSURE IL_PRESSURE PP_FILLERS IL_FILLERS MT_DOC MEMMAX FT WIN_REP
cfg_exe() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['exe'])" "$1"; }

# ---- summary (no GPU; after production is back) -----------------------------------------------------------------------
summary() { READS_N=$READS_N READS_BG_MIN=$READS_BG_MIN READS_BASE=$READS_BASE "$PY" -I - "$OUT" "$L/results/sweep.jsonl" "$L/results/sweep.requests.jsonl" "$L/compare_replies.py" "$TAG" $ARMS <<'PYEOF'
import argparse, contextlib, io, json, os, re, statistics, subprocess, sys
from pathlib import Path
out, sweep, sreq, cmp, tag, arms = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6:]
def jl(p):
    try:
        return [json.loads(x) for x in open(p) if x.strip()]
    except OSError:
        return []
sys.path.insert(0, str(Path(cmp).parent))      # (linux/: fair_tools for the reads step's win facts and engine lines)
try:
    import fair_tools as FTM
except Exception as e:  # noqa: BLE001 - the main table does not need it
    FTM = None
    print(f"(fair_tools.py not importable: {e!r}; the reads columns stay empty)")
def win_of(a):
    try:
        return FTM.win_facts(json.load(open(out / f"win-{a}.json"))) if FTM else None
    except (OSError, ValueError):
        return None
ag = {}
for r in jl(out / "agentic.jsonl"):
    lab = r.get("label", "")
    if lab.startswith(tag + "-") and lab.endswith("-agentic"):
        ag[lab[len(tag) + 1:-len("-agentic")]] = r
mtrow = {r.get("label"): r for r in jl(out / "multiturn.jsonl")}
scen = {}
for r in jl(out / "repro.jsonl"):
    if r.get("kind") == "scenario":
        scen[(r["label"], r["scenario"])] = r
rows = {}
for r in jl(sweep):
    if r.get("label", "").endswith("-short") and r["label"].startswith(tag + "-"):
        rows.setdefault(r["label"], {})[r["users"]] = r
mt = {r.get("label"): {(x[0], x[1]): x[3] for x in r.get("replies", [])} for r in jl(out / "multiturn.jsonl")}
def ident_conc(a):
    if a == "P":
        return "ref"
    try:
        p = subprocess.run([sys.executable, "-I", cmp, sreq, f"{tag}-P-short", f"{tag}-{a}-short"], capture_output=True,
                           text=True, timeout=120)
    except Exception:
        return "?"
    if "Traceback" in p.stderr:
        return "err"
    return {0: "same", 1: "DIFF", 2: "-"}.get(p.returncode, "?")
def ident_mt(a):
    if a == "P":
        return "ref"
    x, y = mt.get(f"{tag}-P-mt"), mt.get(f"{tag}-{a}-mt")
    if not x or not y:
        return "-"
    k = set(x) & set(y)
    return f"{sum(x[i] == y[i] for i in k)}/{len(k)}"
def ident_a(a):
    if a == "P":
        return "ref"
    res = []
    for s in ("pingpong", "interleave"):
        p, q = scen.get((f"{tag}-P", s)), scen.get((f"{tag}-{a}", s))
        if p and q:
            res.append("=" if p.get("a_reply_sha") == q.get("a_reply_sha") else "x")
    return "".join(res) or "-"
def eng(a):
    try:
        lines = [ln for ln in open(out / f"engine-{a}.log", errors="replace") if "conversation cache" in ln]
    except OSError:
        return None
    cap = sorted(float(m.group(1)) for ln in lines for m in [re.search(r"capture_ms=([\d.]+)", ln)] if m)
    wt = sorted(float(m.group(1)) for ln in lines for m in [re.search(r"wait_ms=([\d.]+)", ln)] if m)
    return {"borrow": sum("borrow" in ln for ln in lines), "restored": sum("restored" in ln for ln in lines),
            "supers": sum("superseded" in ln for ln in lines), "refuse": sum("physical RAM admission" in ln for ln in lines),
            "cap50": statistics.median(cap) if cap else None, "cap_max": cap[-1] if cap else None,
            "wait_max": wt[-1] if wt else None, "parks": len(cap)}
def server(a):
    try:
        d = open(out / "cfg" / f"{a}.srv").read().split()[0].split("=", 1)[1]
    except (OSError, IndexError):
        return "?"
    h = subprocess.run(["git", "-C", d, "rev-parse", "--short=8", "HEAD"], capture_output=True, text=True).stdout.strip()
    return f"{Path(d).name}@{h or '?'}"
def pct(x):
    return f"{100 * x:.1f}" if isinstance(x, (int, float)) else "-"
def f(x, d=0):
    return "-" if x is None else f"{x:.{d}f}"
print(f"# window borrow {tag}: per arm (one server start each); reuse % = reused / prompt; ident = vs P (greedy)")
hdr = (f"{'arm':4s} {'A2%':>6s} {'A3%':>6s} {'SUB':>13s} {'SUB2':>13s} {'IL A2%':>6s} {'IL ttft':>7s} {'borrow':>6s} "
       f"{'restor':>6s} {'supers':>6s} {'refuse':>6s} {'parks':>5s} {'cap p50':>7s} {'cap max':>7s} {'wait max':>8s} "
       f"{'tps@1':>6s} {'tps@8':>6s} {'tps@16':>6s} {'ttft16':>6s} {'id conc':>7s} {'id mt':>7s} {'id A':>4s} {'RSS GiB':>7s} "
       f"{'w1 max':>6s} {'w2 all':>6s} {'ag wall':>7s} pp   server")
print(hdr); print("-" * len(hdr))
for a in arms:
    pp, il = scen.get((f"{tag}-{a}", "pingpong")), scen.get((f"{tag}-{a}", "interleave"))
    tr = (pp or {}).get("turn_reuse") or {}
    sub = (pp or {}).get("sub_reuse") or {}
    def s(k):
        v = sub.get(k)
        return f"{v[0]}/{v[1]}" if v and v[0] is not None else "-"
    e = eng(a) or {}
    sw = rows.get(f"{tag}-{a}-short", {})
    try:
        rss = open(out / f"rss-{a}.txt").read().strip()
    except OSError:
        rss = "-"
    wf = win_of(a) or {}
    print(f"{a:4s} {pct(tr.get('A2')):>6s} {pct(tr.get('A3')):>6s} {s('SUB'):>13s} {s('SUB2'):>13s} "
          f"{pct((il or {}).get('a2_reuse')):>6s} {f((il or {}).get('a2_ttft_s'), 2):>7s} {e.get('borrow', '-'):>6} "
          f"{e.get('restored', '-'):>6} {e.get('supers', '-'):>6} {e.get('refuse', '-'):>6} {e.get('parks', '-'):>5} "
          f"{f(e.get('cap50'), 1):>7s} {f(e.get('cap_max'), 1):>7s} {f(e.get('wait_max'), 1):>8s} "
          f"{f((sw.get(1) or {}).get('agg_out_tps'), 1):>6s} {f((sw.get(8) or {}).get('agg_out_tps'), 1):>6s} "
          f"{f((sw.get(16) or {}).get('agg_out_tps'), 1):>6s} {f((sw.get(16) or {}).get('ttft_p50'), 2):>6s} "
          f"{ident_conc(a):>7s} {ident_mt(a):>7s} {ident_a(a):>4s} {rss:>7s} "
          f"{f((wf.get('w1') or {}).get('s_max'), 2):>6s} {f((wf.get('w2') or {}).get('all_first'), 2):>6s} "
          f"{f((ag.get(a) or {}).get('wall_s'), 1):>7s} "
          f"{'PASS' if pp and pp.get('hit') else 'FAIL' if pp else '-':4s} {server(a)}")
if "N0" in arms and "P" in arms:
    ok = ident_conc("N0") == "same" and (lambda m: m != "-" and m.split("/")[0] == m.split("/")[1])(ident_mt("N0"))
    print(f"==> identity N0 (all new switches off, STRATA_SERVE_STOP_BOUND=0) vs P: conc {ident_conc('N0')}, multiturn "
          f"{ident_mt('N0')}, A replies {ident_a('N0')} {'PASS' if ok else 'FAIL'}")
for a in arms:
    if a not in ("P", "N0"):
        print(f"==> identity {a} vs P (expected exact): conc {ident_conc(a)}, multiturn {ident_mt(a)}, A replies {ident_a(a)} INFO")
print("(pp = pingpong PASS: A2 and A3 >= 95 % reused and above the grid point below the branch; SUB = reused/prompt; IL = interleave; cap = park capture_ms; "
      "id A: '=' per scenario when A's greedy reply equals P's; w1 max = every w1 short's TTFT, w2 all = time to all 16 "
      "w2 first tokens, ag wall = bench_agentic's wall (s): the reads step, arms of READS_ARMS only)")

# ---- stage 2 (background reads): N4 vs N2, TESTLOG #88's rule ---------------------------------------------------------
RB, RN = os.environ.get("READS_BASE", "N2"), "N4"
RA = [a for a in arms if (out / f"win-{a}.json").exists() or a in ag]
def bgstat(name):
    try:
        lines = open(out / name, errors="replace").read().splitlines()
    except OSError:
        return None
    if FTM is None:
        return None
    bg = sum(bool(FTM.BG_RE.search(ln)) for ln in lines)
    rdy = sum(bool(FTM.RDY_RE.search(ln)) for ln in lines)
    errd = sum("background read of slot" in ln and " ends at " in ln and ": err" in ln for ln in lines)
    unp = sum("holds no read of this prompt that gave way" in ln for ln in lines)
    return {"bg": bg, "ready": rdy, "err": errd, "unpinned": unp, "give": sum("gives way at" in ln for ln in lines)}
def bgs(name):
    b = bgstat(name)
    return "-" if b is None else f"{b['bg']}/{b['ready']}"
def ag_keyed(r):            # per request: (who, its k-th call) -> reply sha (None: an error or no sha in the row)
    d = {}
    for row in sorted((r or {}).get("rows") or [], key=lambda z: z.get("at", 0)):
        d.setdefault(row.get("who"), []).append(None if row.get("err") else row.get("sha"))
    return {(w, i): x for w, v in d.items() for i, x in enumerate(v)}
def same_ag(x, y):          # main and sub-agent requests all; the chat user's: those both runs made (timing decides how many)
    a, b = ag_keyed(x), ag_keyed(y)
    keys = {k for k in set(a) | set(b) if not str(k[0]).startswith("chat")} | (set(a) & set(b))
    return sum(1 for k in keys if a.get(k) is not None and a.get(k) == b.get(k)), len(keys)
def same_win(x, y):
    keys = sorted(k for k in set(x) | set(y) if not k.startswith("_") and not k.endswith("-meta"))
    ok = sum(1 for k in keys if isinstance(x.get(k), dict) and isinstance(y.get(k), dict) and not x[k].get("error")
             and not y[k].get("error") and x[k].get("text") == y[k].get("text"))
    return ok, len(keys)
def same_mt(a, b):
    x, y = mt.get(f"{tag}-{a}-mt"), mt.get(f"{tag}-{b}-mt")
    if not x or not y:
        return None
    k = set(x) | set(y)
    return sum(x.get(i) is not None and x.get(i) == y.get(i) for i in k), len(k)
def g2(d, *ks):
    for k in ks:
        d = (d or {}).get(k)
    return d
if RA:
    print()
    print(f"# stage 2 reads step (one more server start per arm: fair_tools win --rep, then bench_agentic greedy); mt = the "
          f"arm's own bench_multiturn run; bg rd = background reads handed / ready (reads step, arm run)")
    h2 = (f"{'arm':4s} {'w1 p50':>6s} {'w1 max':>6s} {'w1 long':>7s} {'w2 all':>6s} {'w2 p50':>6s} {'w2 p95':>6s} "
          f"{'w2 long':>7s} {'ag wall':>7s} {'main':>5s} {'sub':>5s} {'chat':>5s} {'ag p95':>6s} {'ag tok':>6s} "
          f"{'mt wall':>7s} {'mt fol':>6s} {'bg rd':>7s} {'bg arm':>7s} {'RSS':>6s}")
    print(h2); print("-" * len(h2))
    for a in RA:
        w = win_of(a) or {}
        w1, w2, g, m = w.get("w1") or {}, w.get("w2") or {}, ag.get(a) or {}, mtrow.get(f"{tag}-{a}-mt") or {}
        try:
            rr = open(out / f"rss-rd-{a}.txt").read().strip()
        except OSError:
            rr = "-"
        print(f"{a:4s} {f(w1.get('s_p50'), 2):>6s} {f(w1.get('s_max'), 2):>6s} {f(w1.get('long_total'), 2):>7s} "
              f"{f(w2.get('all_first'), 2):>6s} {f(w2.get('s_p50'), 2):>6s} {f(w2.get('s_p95'), 2):>6s} "
              f"{f(w2.get('long_total'), 2):>7s} {f(g.get('wall_s'), 1):>7s} {f(g2(g, 'main', 'ttft_p50'), 2):>5s} "
              f"{f(g2(g, 'sub', 'ttft_p50'), 2):>5s} {f(g2(g, 'chat', 'ttft_p50'), 2):>5s} {f(g2(g, 'main', 'ttft_p95'), 2):>6s} "
              f"{str(g.get('out_tok', '-')):>6s} {f(m.get('wall_s'), 1):>7s} {f(m.get('follow_ttft_p50'), 2):>6s} "
              f"{bgs(f'engine-rd-{a}.log'):>7s} {bgs(f'engine-{a}.log'):>7s} {rr:>6s}")
    print("(#88 for reference, rep 1-2 medians, production of then: w1 short TTFT max PROD 20.20 / FAIR 1.19 / READS "
          "0.88 s, w2 all 16 first 17.08 / 4.87 / 17.03 s; bench_agentic wall PROD 116.8-116.9 / READS 142.1-148.3 s)")
    wb, wn = win_of(RB), win_of(RN)
    gb, gn = ag.get(RB), ag.get(RN)
    mb, mn = mtrow.get(f"{tag}-{RB}-mt"), mtrow.get(f"{tag}-{RN}-mt")
    what = (f"{RN} (N2 + STRATA_READS={os.environ.get('READS_N', '?')}, server STRATA_SERVER_READS=1 "
            f"STRATA_READS_BG_MIN={os.environ.get('READS_BG_MIN', '?')}, the starvation fix 063af1fd)")
    if FTM and wb and wn:          # #88's own tool on the same files (median of one run = the run)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                FTM.mode_win(argparse.Namespace(arms=[f"{RB}={out}/win-{RB}.json", f"{RN}={out}/win-{RN}.json"],
                                                gain=1.0, long_tol=0.05))
                FTM.mode_readstat(argparse.Namespace(
                    arms=[f"{RB}={out}/engine-rd-{RB}.log", f"{RN}={out}/engine-rd-{RN}.log",
                          f"{RN}arm={out}/engine-{RN}.log"], want_bg=f"{RN},{RN}arm", want_none=RB,
                    want_clean=f"{RB},{RN},{RN}arm", max_starts=1))
            except Exception as e:  # noqa: BLE001
                print(f"==> fair_tools win / readstat failed: {e!r} FAIL")
        for ln in buf.getvalue().splitlines():
            print(f"    [fair_tools] {ln.strip()[:400]}")
    if not (wb and wn and gb and gn):
        print(f"==> stage 2 {RN} vs {RB}: not measured (reads step ran for {', '.join(RA) or 'none'}; want both win "
              f"files and both bench_agentic runs) INFO")
    else:
        res = []
        def judge(name, ok, line):
            res.append((name, ok))
            print(f"==> stage 2 {RN} vs {RB} {line} {'PASS' if ok else 'FAIL'}")
        b1, n1, b2, n2 = wb.get("w1") or {}, wn.get("w1") or {}, wb.get("w2") or {}, wn.get("w2") or {}
        ok = None not in (b1.get("s_max"), n1.get("s_max")) and n1["s_max"] <= b1["s_max"] and n1.get("s_miss") == 0
        judge("w1", ok, f"w1 (a cold ~125K read beside 2 streams, 4 shorts): every short's TTFT (max) {f(b1.get('s_max'), 2)} "
              f"-> {f(n1.get('s_max'), 2)} s, p50 {f(b1.get('s_p50'), 2)} -> {f(n1.get('s_p50'), 2)} s (want <= 1.00 x, "
              f"#88's rule; shorts without a token {n1.get('s_miss')})")
        ok = None not in (b2.get("all_first"), n2.get("all_first")) and n2["all_first"] <= b2["all_first"] and n2.get("s_miss") == 0
        judge("w2", ok, f"w2 (a cold ~105K read, 16 shorts at once): all 16 first tokens after {f(b2.get('all_first'), 2)} -> "
              f"{f(n2.get('all_first'), 2)} s, p50 {f(b2.get('s_p50'), 2)} -> {f(n2.get('s_p50'), 2)} s (want <= 1.00 x)")
        ok = all(None not in (x.get("long_total"), y.get("long_total")) and y["long_total"] <= 1.05 * x["long_total"]
                 for x, y in ((b1, n1), (b2, n2)))
        judge("long", ok, f"long reads: w1 total {f(b1.get('long_total'), 2)} -> {f(n1.get('long_total'), 2)} s, w2 "
              f"{f(b2.get('long_total'), 2)} -> {f(n2.get('long_total'), 2)} s (want <= 1.05 x)")
        aw = (gb.get("wall_s"), gn.get("wall_s"))
        mw = ((mb or {}).get("wall_s"), (mn or {}).get("wall_s"))
        rel = lambda p: f"{100 * (p[1] / p[0] - 1):+.1f} %" if None not in p and p[0] else "-"
        ok = None not in aw and aw[1] <= 1.05 * aw[0] and None not in mw and mw[1] <= 1.05 * mw[0]
        errs = sum(int(g2(gn, r, "err") or 0) for r in ("main", "sub", "chat"))
        judge("mix", ok and errs == 0, f"agent mix: bench_agentic wall {f(aw[0], 1)} -> {f(aw[1], 1)} s ({rel(aw)}), main TTFT "
              f"p50 {f(g2(gb, 'main', 'ttft_p50'), 2)} -> {f(g2(gn, 'main', 'ttft_p50'), 2)} s, {gb.get('out_tok')} -> "
              f"{gn.get('out_tok')} output tokens, {errs} request errors; bench_multiturn wall {f(mw[0], 1)} -> "
              f"{f(mw[1], 1)} s ({rel(mw)}), follow-up TTFT p50 {f((mb or {}).get('follow_ttft_p50'), 2)} -> "
              f"{f((mn or {}).get('follow_ttft_p50'), 2)} s (want both walls <= +5 %, no errors)")
        sw, sa, sm = same_win(json.load(open(out / f"win-{RB}.json")), json.load(open(out / f"win-{RN}.json"))), \
            same_ag(gb, gn), same_mt(RB, RN)
        ok = sw[0] == sw[1] > 0 and sa[0] == sa[1] > 0 and sm is not None and sm[0] == sm[1] > 0
        judge("replies", ok, f"replies (greedy): win {sw[0]}/{sw[1]}, bench_agentic {sa[0]}/{sa[1]} (per request), "
              f"bench_multiturn {'-' if sm is None else f'{sm[0]}/{sm[1]}'} identical")
        st_rd, st_arm = bgstat(f"engine-rd-{RN}.log"), bgstat(f"engine-{RN}.log")
        ok = bool(st_rd) and st_rd["bg"] > 0 and st_rd["ready"] == st_rd["bg"] and st_rd["err"] == 0 and st_rd["unpinned"] == 0
        judge("bg", ok, f"background reads: reads step {st_rd and st_rd['bg']} handed / {st_rd and st_rd['ready']} ready / "
              f"{st_rd and st_rd['err']} dropped with err / {st_rd and st_rd['unpinned']} unpinned (want > 0, all ready, "
              f"none err / unpinned); arm run {st_arm and st_arm['bg']} / {st_arm and st_arm['ready']}; {RB}: "
              f"{(bgstat(f'engine-rd-{RB}.log') or {}).get('bg')} (want 0)")
        bad = [n for n, ok in res if not ok]
        print(f"==> stage 2 {what} vs {RB}: {'PASS (all ' + str(len(res)) + ')' if not bad else 'FAIL (' + ', '.join(bad) + ')'}")
PYEOF
}

# ---- prep (no GPU; production still serving) --------------------------------------------------------------------------
PRODSRC=$(readlink -f "$C/production.json")
BASESRC=$(readlink -f "${BASE_CONFIG:-$C/production.json}")
for f in "$RP" "$L/bench_multiturn.py" "$L/bench_concurrency.py" "$L/compare_replies.py" "$L/run_strata_variant.sh" \
         "$SRV/serve/server.py" "$NSRV/serve/server.py" "$PRODSRC" "$BASESRC" "$NEWSRC"; do [ -e "$f" ] || { echo "missing $f"; exit 1; }; done
[ -x "$L/run_strata_variant.sh" ] || { echo "$L/run_strata_variant.sh is not executable"; exit 1; }
[ -x "$NEWSRC" ] || { echo "$NEWSRC is not executable"; exit 1; }
for s in "$SRV" "$NSRV"; do
    git -C "$s" diff --quiet HEAD -- serve || { echo "$s has uncommitted changes under serve/: the server is not HEAD"; exit 1; }
    # parse only (py_compile would write __pycache__ into the checkout)
    "$PY" -I -c 'import ast,sys; ast.parse(open(sys.argv[1]).read(), sys.argv[1])' "$s/serve/server.py" \
        || { echo "$s/serve/server.py does not parse"; exit 1; }
done
[ "$(git -C "$SRV" rev-parse --short=8 HEAD)" = 90f73293 ] || { echo "$SRV is not at 90f73293 (production's server)"; exit 1; }
git -C "$NSRV" merge-base --is-ancestor 90f73293 HEAD 2>/dev/null || { echo "$NSRV does not contain 90f73293"; exit 1; }
for f in "$RP" "$L/bench_multiturn.py" "$L/bench_concurrency.py" "$L/compare_replies.py" "$FT" "$L/bench_agentic.py"; do
    [ -e "$f" ] || { echo "missing $f"; exit 1; }
    "$PY" -I "$f" --help > /dev/null || { echo "$f does not load"; exit 1; }; done
grep -q '"sha": hashlib' "$L/bench_agentic.py" || { echo "$L/bench_agentic.py writes no reply sha (the agent mix's identity)"; exit 1; }
PPCHK=$("$PY" -I "$RP" --dry-run --exact --scenarios pingpong,interleave --sizes "$SIZE" --cycles "$CYCLES" 2>&1) \
    || { echo "$PPCHK"; echo "repro_park_ram.py --dry-run --exact failed"; exit 1; }
# the switches the arms set must exist in the binary (an unknown STRATA_* is silently ignored: the arm would be N0/N1)
declare -A NEED=([N0]=STRATA_SERVE_STOP_BOUND [S0]=STRATA_SERVE_STOP_BOUND [N1]=STRATA_SERVE_STOP_BOUND [N2]=STRATA_PARK_BORROW [N3]="STRATA_CONV_MMAP STRATA_GATE_EVICT" [N4]=STRATA_READS)
MISSING=; KEEP_ARMS=
for a in $ARMS; do
    case $a in P|N0|S0|N1|N2|N3|N4) ;; *) echo "unknown arm $a (P N0 S0 N1 N2 N3 N4)"; exit 1 ;; esac
    [ "$a" = S0 ] && [ "$NSRV" = "$(readlink -f "$SRV")" ] && { echo "arm S0 needs --server-tree DIR other than $SRV (else it is N0)"; exit 1; }
    miss=; for s in ${NEED[$a]:-}; do grep -qa "$s" "$NEWSRC" || miss="$miss $s"; done
    case $a in N3|N4) grep -qa STRATA_PARK_BORROW "$NEWSRC" || miss="$miss STRATA_PARK_BORROW" ;; esac
    if [ "$a" = N4 ]; then   # the server half, the starvation fix in the binary, nothing of it in the baseline already
        grep -q 'STRATA_SERVER_READS' "$NSRV/serve/server.py" || miss="$miss STRATA_SERVER_READS(server $NSRV: --server-tree)"
        fixt=$(git -C "$NEWTREE" log -1 --format=%ct -S"$READS_FIX_SRC" HEAD -- src/program/generate.cpp 2>/dev/null)
        if [ -z "$fixt" ] || ! git -C "$NEWTREE" grep -qF "$READS_FIX_SRC" HEAD -- src/program/generate.cpp; then
            miss="$miss reads-fix-063af1fd(not in $NEWTREE HEAD)"
        elif [ "$(stat -c %Y "$NEWSRC")" -lt "$fixt" ]; then miss="$miss reads-fix-063af1fd(binary older than the fix commit)"; fi
        python3 -c "import json,sys; sys.exit(any(k.startswith('STRATA_READS') for k in json.load(open(sys.argv[1]))['env']))" "$BASESRC" \
            || miss="$miss STRATA_READS*(already in the baseline config: N2 would not be without it)"
        case " $(systemctl --user show strata.service -p Environment --value) " in *" STRATA_SERVER_READS="*|*" STRATA_READS_"*)
            miss="$miss STRATA_SERVER_READS(already in production's server env: N2 would get it too)" ;; esac
    fi
    if [ -n "$miss" ]; then MISSING="$MISSING $a:${miss# }"; [ "$ALLOW_MISSING" = 1 ] && continue; fi
    KEEP_ARMS="$KEEP_ARMS $a"
done
if [ -n "$MISSING" ]; then
    [ "$ALLOW_MISSING" = 1 ] || { echo "$NEWSRC (or the N server) lacks what the arms need:$MISSING (rebuild / --server-tree, or ALLOW_MISSING=1 drops those arms)"; exit 1; }
    echo "dropping arms (switch missing in the binary or the server):$MISSING"; fi
ARMS=${KEEP_ARMS# }; [ -n "$ARMS" ] || { echo "no arm left"; exit 1; }
RARMS=; for a in $ARMS; do case " $READS_ARMS " in *" $a "*) RARMS="$RARMS $a" ;; esac; done; RARMS=${RARMS# }
TOKCHK=
if [ -n "$RARMS" ]; then   # the win scenario's prompts measured with the tokenizer (CPU only, low priority)
    TOKCHK=$(nice -n 19 taskset -c 0-47 "$PY" -I "$FT" --mode tokcheck --config "$BASESRC" --short 2048 --reps-list "$WIN_REP" 2>&1) \
        || { echo "$TOKCHK"; echo "fair_tools.py tokcheck failed: the reads step's scenarios do not fit"; exit 1; }
fi
curl -s -m 2 "http://127.0.0.1:$TPORT/health" > /dev/null && { echo "port $TPORT already serves something: the test servers need it"; exit 1; }
systemctl --user is-active -q strata-bench.scope && { echo "strata-bench.scope is active: another bench is running"; exit 1; }
systemctl --user is-active -q "$RUNIT.timer" && { echo "$RUNIT.timer is active: another window of this kind?"; exit 1; }
systemctl --user is-active -q "$GUNIT.service" && { echo "$GUNIT.service is active: another window of this kind?"; exit 1; }
PEXE=$(cfg_exe "$PRODSRC"); [ -x "$PEXE" ] || { echo "production's exe $PEXE missing"; exit 1; }
if [ "$DRY" != 1 ]; then
    systemctl --user is-active -q strata.service || { echo "strata.service is not active: start it first"; exit 1; }
    # every GPU process now is production's engine (free_gpus later kills what is left of $R/Strata* and $BIN only)
    for p in $(gpu_apps); do [ "$(readlink -f "/proc/$p/exe" 2>/dev/null)" = "$(readlink -f "$PEXE")" ] \
        || { echo "GPU process $p ($(readlink -f "/proc/$p/exe" 2>/dev/null)) is not production's engine: not starting"; exit 1; }; done
fi
# production's server env (the drop-in's STRATA_* keys, read now): every test server gets it
SRVENV=$(systemctl --user show strata.service -p Environment --value | tr ' ' '\n' | grep '^STRATA_' | tr '\n' ' ')
PROD_SHA=$(sha256sum "$PEXE" | cut -c1-16); PRODCFG_SHA=$(sha256sum "$PRODSRC" | cut -c1-16)
BEXE=$(cfg_exe "$BASESRC"); [ -x "$BEXE" ] || { echo "the baseline's exe $BEXE missing"; exit 1; }
if [ "$DRY" = 1 ]; then rm -rf "$OUT"                    # (the dry run's own scratch dir)
elif [ -e "$OUT" ]; then PREV=$OUT.prev-$(date +%Y%m%d-%H%M%S); mv "$OUT" "$PREV" || { echo "cannot move $OUT aside"; exit 1; }; fi
mkdir -p "$OUT" "$CF" "$BIN"
NEW_SHA=$(sha256sum "$NEWSRC" | cut -c1-16); NEW=$BIN/strata-ll10-borrow-$NEW_SHA
cp "$NEWSRC" "$NEW" && chmod +x "$NEW" || { echo "cannot copy $NEWSRC"; exit 1; }
[ "$(sha256sum "$NEW" | cut -c1-16)" = "$NEW_SHA" ] || { echo "$NEWSRC changed while it was copied (a build running?)"; exit 1; }
echo "$PPCHK" | sed 's/^/    [dry-run --exact] /' >> "$OUT/window.log"
[ -n "$TOKCHK" ] && echo "$TOKCHK" | sed 's/^/    [tokcheck] /' >> "$OUT/window.log"
log "window prep ($TAG): arms $ARMS; reads step for '${RARMS:-none}' (fair_tools win --rep $WIN_REP + bench_agentic; N4: STRATA_READS=$READS_N, server STRATA_SERVER_READS=1 STRATA_READS_BG_MIN=$READS_BG_MIN; $NEWTREE $(git -C "$NEWTREE" status --porcelain -- src include CMakeLists.txt 2>/dev/null | grep -q . && echo 'has UNCOMMITTED source changes' || echo 'sources clean')); production $(basename "$PRODSRC") sha256 $PRODCFG_SHA, exe $(basename "$PEXE") sha256 $PROD_SHA, server $SRV @$(git -C "$SRV" rev-parse --short HEAD); new $NEWSRC sha256 $NEW_SHA ($(git -C "$R/Strata-ll10-borrow" rev-parse --short HEAD 2>/dev/null) $(git -C "$R/Strata-ll10-borrow" log -1 --format=%s 2>/dev/null | cut -c1-70)) -> $NEW; N server $NSRV @$(git -C "$NSRV" rev-parse --short HEAD); server env: $SRVENV; pingpong @$SIZE x$CYCLES pressure $PP_PRESSURE fillers $PP_FILLERS, interleave pressure $IL_PRESSURE fillers $IL_FILLERS; baseline (P, and the arms' config) $(basename "$BASESRC"), exe $(basename "$BEXE")"

# configs, all from the baseline config (production's unless BASE_CONFIG): exe, then the arm's env changes (K=V sets, -K drops); the sidecar <name>.srv:
# the server checkout + production's server env (+ SRVX[name]: the arm's own server env)
declare -A SRVX=([N4]="STRATA_SERVER_READS=1 STRATA_READS_BG_MIN=$READS_BG_MIN")
mkcfg() {   # $1 name, $2 exe, $3 server dir, then changes ...
    local name=$1 exe=$2 sdir=$3; shift 3
    python3 - "$BASESRC" "$CF/$name.json" "$exe" "$@" <<'PYEOF' || exit 1
import json, sys
src, dst, exe = sys.argv[1:4]
c = json.load(open(src)); c["exe"] = exe; e = c["env"]
for kv in sys.argv[4:]:
    if kv.startswith("-"):
        e.pop(kv[1:], None)
    else:
        k, v = kv.split("=", 1); e[k] = v
json.dump(c, open(dst, "w"), indent=1)
PYEOF
    echo "SERVER_DIR=$sdir $SRVENV${SRVX[$name]:+ ${SRVX[$name]}}" > "$CF/$name.srv"; }
OFF=(-STRATA_PARK_BORROW -STRATA_CONV_MMAP -STRATA_GATE_EVICT)
mkcfg P  "$BEXE" "$SRV"
mkcfg N0 "$NEW"  "$SRV"  "${OFF[@]}" STRATA_SERVE_STOP_BOUND=0     # P's server: N0 vs P = the binary only
mkcfg S0 "$NEW"  "$NSRV" "${OFF[@]}" STRATA_SERVE_STOP_BOUND=0     # N0 on the --server-tree server
mkcfg N1 "$NEW"  "$NSRV" "${OFF[@]}" -STRATA_SERVE_STOP_BOUND
mkcfg N2 "$NEW"  "$NSRV" "${OFF[@]}" -STRATA_SERVE_STOP_BOUND STRATA_PARK_BORROW=1
mkcfg N3 "$NEW"  "$NSRV" -STRATA_SERVE_STOP_BOUND STRATA_PARK_BORROW=1 STRATA_CONV_MMAP=1 STRATA_GATE_EVICT=1 \
    MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=1048576
case $READS_BASE in
  N2) mkcfg N4 "$NEW" "$NSRV" "${OFF[@]}" -STRATA_SERVE_STOP_BOUND STRATA_PARK_BORROW=1 STRATA_READS="$READS_N" ;;   # N2 + stage 2
  N3) mkcfg N4 "$NEW" "$NSRV" -STRATA_SERVE_STOP_BOUND STRATA_PARK_BORROW=1 STRATA_CONV_MMAP=1 STRATA_GATE_EVICT=1 \
          MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=1048576 STRATA_READS="$READS_N" ;;                      # N3 + stage 2
  *) echo "READS_BASE must be N2 or N3"; exit 1 ;;
esac
for a in P N0 S0 N1 N2 N3 N4; do
    log "    [cfg] $a: exe $(basename "$(cfg_exe "$CF/$a.json")"), server $(sed 's/ .*//' "$CF/$a.srv")${SRVX[$a]:+ + server env ${SRVX[$a]}}, env vs production: $(python3 -c "
import json,sys; p=json.load(open(sys.argv[1]))['env']; c=json.load(open(sys.argv[2]))['env']
d=[f'{k}={c[k]}' for k in c if p.get(k)!=c[k]]+[f'-{k}' for k in p if k not in c]; print(' '.join(d) or 'none')" "$BASESRC" "$CF/$a.json")"
done
NEED_S=$(( $(echo $ARMS | wc -w) * ARM_EST + $(echo $RARMS | wc -w) * READS_EST + 300 ))
if [ "$DRY" = 1 ]; then
    log "dry run: prep and configs ok, production untouched ($CF); expected outage ~$(( NEED_S / 60 )) min for arms '$ARMS' + reads steps '${RARMS:-none}' (cap $(( WINDOW_S / 60 )) min, steps until $(( (WINDOW_S - RESERVE) / 60 )))"
    for a in $ARMS; do echo "  step arm_$a (expected ~$(( ARM_EST / 60 )) min, cap $(( ARM_CAP / 60 )) min): arm_run $a -> $OUT/arm_$a.txt"
        echo "    server: $(cat "$CF/$a.srv" | cut -c1-160) ... on 127.0.0.1:$TPORT, MemoryMax $MEMMAX"
        [ -n "${SRVX[$a]:-}" ] && echo "    server env of the arm: ${SRVX[$a]} (engine: $(python3 -c "import json,sys; e=json.load(open(sys.argv[1]))['env']; print(' '.join(f'{k}={e[k]}' for k in sorted(e) if k.startswith(('STRATA_READ', 'STRATA_PARK_BORROW', 'STRATA_SERVE_STOP'))))" "$CF/$a.json"))"
        echo "    + repro_park_ram.py --label $TAG-$a --scenarios pingpong --sizes $SIZE --cycles $CYCLES --fillers $PP_FILLERS --pressure $PP_PRESSURE"
        echo "    + repro_park_ram.py --label $TAG-$a --scenarios interleave --sizes $SIZE --fillers $IL_FILLERS --pressure $IL_PRESSURE"
        echo "    + bench_multiturn.py --users 16 --turns 3 --doc $MT_DOC --temperature 0 --label $TAG-$a-mt"
        echo "    + run_strata_variant.sh (ATTACH) $TAG-$a-short 1,8,16 --rounds 2 --max-tok 512 --max-tok-min 128 --prompt-tok 100 --temperature 0"; done
    for a in $RARMS; do echo "  step reads_$a (expected ~$(( READS_EST / 60 )) min, cap $(( READS_CAP / 60 )) min): reads_run $a -> $OUT/reads_$a.txt (one more fresh server start, after all arms)"
        echo "    + fair_tools.py --mode scen --scenario win --rep $WIN_REP --label $TAG-$a-win -> win-$a.json (w1 cold ~125K beside 2 streams + 4 shorts, w2 cold ~105K + 16 shorts)"
        echo "    + bench_agentic.py --temperature 0 --label $TAG-$a-agentic -> agentic.jsonl"; done
    [ "$(echo $RARMS | wc -w)" -gt 1 ] && echo "  (the reads steps run back to back; when they no longer all fit, only the last, ${RARMS##* }, runs)"
    echo "  then: production restarted (EXIT trap; $GUNIT.service on a SIGKILL; $RUNIT.timer backstop), summary table -> $OUT/summary.txt"
    grep -E '^(what|A |  )' <<< "$PPCHK" | sed 's/^/  [dry-run --exact] /'
    [ -n "$TOKCHK" ] && grep -E '^==> tokcheck win' <<< "$TOKCHK" | sed 's/^/  [tokcheck] /'
    exit 0
fi
[ "$NEED_S" -le $(( WINDOW_S - RESERVE )) ] || log "note: arms '$ARMS' + reads steps '$RARMS' need ~$(( NEED_S / 60 )) min, the window allows $(( (WINDOW_S - RESERVE) / 60 )): the last ones may be skipped"

# ---- the window -------------------------------------------------------------------------------------------------------
for _ in $(seq 180); do prod_loaded && break; sleep 5; done
prod_loaded || { echo "production does not report loaded: not stopping it"; exit 1; }
RESTORE_TS=$(( $(date +%s) + WINDOW_S + 900 ))
arm_restore "$RESTORE_TS" || { echo "could not arm $RUNIT.timer: production left running"; exit 1; }
: > "$OUT/step.pgid"
arm_guard || { disarm_restore; disarm_guard; echo "could not arm $GUNIT.service: production left running"; exit 1; }
log "window start ($(( WINDOW_S / 60 )) min cap; $GUNIT.service waits for PID $$; $RUNIT.timer armed for $(date -d "@$RESTORE_TS" '+%F %T')): stopping strata.service"
trap '' INT TERM HUP                             # (not while it stops)
trap final EXIT
systemctl --user stop strata.service
STOPPED=1; T0=$(date +%s)                       # (WINDOW_S: production's outage, from its stop to its restart)
trap 'on_sig signal' INT TERM HUP               # from here on every exit restores production (EXIT -> final)
# HARD_STOP: ends the window, restores production (only this script: after a SIGKILL its PID may belong to another)
( sleep $(( WINDOW_S + 300 )); tr '\0' ' ' < /proc/$$/cmdline 2>/dev/null | grep -q window-borrow && kill -TERM $$ ) 2>/dev/null & WD=$!
# production's port closed (a socket check: nothing is sent to 8090 while production is stopped)
for _ in $(seq 60); do ss -Hltn 'sport = :8090' | grep -q . || break; sleep 2; done
free_gpus || on_sig "GPUs not free"

for a in $ARMS; do step "arm_$a" "$ARM_EST" "$ARM_CAP" arm_run "$a"; done
# the reads steps (stage 2: N2 then N4, back to back); when the pair no longer fits, the last one alone (INFO only)
RUN_R=$RARMS; nr=$(echo $RUN_R | wc -w)
if [ "$nr" -gt 1 ] && [ "$(left)" -lt $(( nr * READS_EST )) ]; then
    for a in $RUN_R; do [ "$a" = "${RUN_R##* }" ] || { log "step reads_$a: SKIP (the $nr reads steps need ~$(( nr * READS_EST )) s, $(left) s left: only reads_${RUN_R##* }, no baseline)"; echo skip > "$OUT/rc_reads_$a"; }; done
    RUN_R=${RUN_R##* }
fi
for a in $RUN_R; do step "reads_$a" "$READS_EST" "$READS_CAP" reads_run "$a"; done
exit 0                                           # -> EXIT trap: production back, then the summary
