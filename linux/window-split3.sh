#!/usr/bin/env bash
# GPU window (approved 2026-10-16): loan-free rebalance - 11,24,37 + --kv-resident 28672 / 24576 + STRATA_SPLIT_OWN_RESERVE_MIB=768
# (split2 showed the smaller --kv-resident alone frees only ~0.2 GiB: CUDA1/CUDA2 still borrowed).
# One binary throughout: build-ll13rs/strata (g3-1441-ll + the opt-in STRATA_SPLIT_OWN_RESERVE_MIB; unset = the same
# room test as strata-g3-1441b, host code only, kernels SASS-identical).  Arms (configs written to results/split/):
#   prod       T16-1441b-prod262r-vision-cpu.json (split 12,25,37), exe swapped, nothing else changed
#   11-24-37   the same with layer_split 11,24,37 (the default 1.5 GiB room test: loans expected on CUDA1/CUDA2)
#   11-24-37r  11,24,37 + STRATA_SPLIT_OWN_RESERVE_MIB=768 (no loan expected: ~0.8-1.0 GiB still free once loaded)
# per arm: (1) startup facts (residency, loans, prompt chunk, VRAM free) from the engine's own lines,
#          (2) pure decode, 16 slots, 30 s: rows/s (prof_driver), GPU clocks/power/throttle reasons logged beside it,
#          (3) 11-24-37r and prod once more with STRATA_VERIFY_PROFILE (stage table),
#          (4) exactness: batch_test16 greedy --skip-solo --dump: 15 short prompts decode while the 16th reads ~40K
#              tokens of filler FRESH (no solo pass before it, a new conversation-tier dir per run, so neither the
#              prompt cache nor a parked conversation can serve it) - the case where a loan streams experts.
#              The batch tokens of every arm are cmp'd against prod's; admissions must take longer than a cached
#              admission (>= 4 s; a cache hit admits all 16 in ~2.5 s) or the dump is flagged as not meaningful.
#              If an arm differs, the same again without the long prompt (split itself vs loan).
#   then (5) HTTP A/B (ab_quick, ABBA x2) only for an arm that is >= +1.5 % in (2) AND identical in (4),
#   and production is started again unchanged (production.json is never touched).
# Test engines park into results/split/tier/* (not production's /media/raid0/strata-kvspill, not its write budget).
# Expected length: 25-35 min without (5) (more if an arm differs), ~50 min with one HTTP arm.  Results: results/split/.
set -u
R=/media/raid0/StrataQ38FN; OUT=$R/linux/results/split3; mkdir -p "$OUT/tier"; URL=http://127.0.0.1:8090; C=$R/linux/configs
P=$R/linux/results/profile2; EXE=$R/Strata/engine/strata-g3-1441c
PROD=$C/T16-1441b-prod262r-vision-cpu.json
ARMS=(s28r s24r)
[ -x "$EXE" ] || { echo "missing $EXE"; exit 1; }
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/window.log"; }
gpu_apps() { nvidia-smi --query-compute-apps=pid --format=csv,noheader; }
free_gpus() { for i in $(seq 45); do [ -z "$(gpu_apps)" ] && return 0; sleep 2; done
    for p in $(gpu_apps); do case "$(readlink -f /proc/$p/exe 2>/dev/null)" in $R/Strata*) kill "$p" ;; esac; done
    for i in $(seq 30); do [ -z "$(gpu_apps)" ] && return 0; sleep 2; done; return 1; }
start_prod() { log "starting production as before (production.json -> $(readlink $C/production.json))"
    systemctl --user start strata.service; local t0; t0=$(date +%s)
    until curl -s -m 3 $URL/health | grep -q '"loaded": *true'; do
        systemctl --user is-active -q strata.service || { log "strata.service died during the start"; return 1; }
        [ $(( $(date +%s) - t0 )) -gt 900 ] && { log "start timeout"; return 1; }; sleep 5; done
    log "production loaded after $(( $(date +%s) - t0 )) s"; }
SMI=""
smi_start() { nvidia-smi --query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,clocks_throttle_reasons.active \
                  --format=csv -lms 500 > "$OUT/gpu-$1.csv" 2>/dev/null & SMI=$!; }
smi_stop() { [ -n "$SMI" ] && kill "$SMI" 2>/dev/null; wait "$SMI" 2>/dev/null; SMI=""; }
trap 'log "window aborted"; smi_stop; systemctl --user stop strata-bench.scope 2>/dev/null; free_gpus; start_prod; exit 1' INT TERM

# arm config: the production config with exe = $EXE, layer_split and extra env per arm (prof_driver and ab_quick read
# exe/env from the config; batch_test16 takes env from its environment, see cfg_env)
mkcfg() {   # $1 arm, $2 split, $3 extra env "K=V ..." (may be empty), $4 --kv-resident (default: the config's)
    python3 - "$PROD" "$OUT/cfg-$1.json" "$EXE" "$2" "$3" "${4:-}" <<'EOF'
import json, sys
src, dst, exe, split, extra, kvr = sys.argv[1:]
c = json.load(open(src)); c["exe"] = exe; c["layer_split"] = split
if kvr:
    a = c["args"]; a[a.index("--kv-resident") + 1] = kvr
for kv in extra.split():
    k, v = kv.split("=", 1); c["env"][k] = v
json.dump(c, open(dst, "w"), indent=2)
EOF
}
mkcfg prod 12,25,37 ""
mkcfg s28r 11,24,37 "STRATA_SPLIT_OWN_RESERVE_MIB=768" 28672
mkcfg s24r 11,24,37 "STRATA_SPLIT_OWN_RESERVE_MIB=768" 24576
cfg_of() { echo "$OUT/cfg-$1.json"; }
# a fresh, empty conversation-tier dir per engine run (overrides the config's production dir)
tier() { local d=$OUT/tier/$1; rm -rf "$d"; mkdir -p "$d"; echo "$d"; }
tiercfg() {   # $1 config, $2 tier dir -> a copy with STRATA_CONV_TIER_DIR replaced
    python3 - "$1" "$2" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1])); c["env"]["STRATA_CONV_TIER_DIR"] = sys.argv[2]
out = sys.argv[2] + ".json"; json.dump(c, open(out, "w"), indent=2); print(out)
EOF
}
cfg_env() { python3 -c "import json,sys,shlex; print(' '.join(k+'='+shlex.quote(v) for k,v in json.load(open(sys.argv[1]))['env'].items()))" "$1"; }
facts() {   # the engine's startup lines that decide whether the split is exact and loan-free
    grep -aE "runs layers|profiled pairs|keeps its own prompt buffers|borrows|experts resident|have experts out of VRAM|prompt chunk|VRAM free with everything loaded|expert cache auto|SPLIT_OWN_RESERVE" "$1" \
        | sed 's/^[0-9.]* E //' | cut -c1-190; }
decode() {  # $1 arm, $2 "p" for the profiled run
    local d=$OUT/dec_$1${2:+_p} a="--mode decode --batch 16 --seconds 30" cfg
    cfg=$(tiercfg "$(cfg_of $1)" "$(tier dec_$1${2:+_p})")
    [ -n "${2:-}" ] && a="$a --env STRATA_VERIFY_PROFILE=1"
    free_gpus || log "GPUs busy before decode $1${2:+ (profiled)}"
    log "decode $1${2:+ (profiled)}: start"; smi_start "dec_$1${2:+_p}"
    timeout -s INT 900 python3 $P/prof_driver.py --config "$cfg" --out "$d" $a > "$d.driver.out" 2>&1; local rc=$?; smi_stop
    local line; line=$(python3 $P/dec_rows.py "$d/events.log" 2>&1 | tail -1)
    log "decode $1${2:+ (profiled)}: rc=$rc ${line#*events.log: }"
    [ -z "${2:-}" ] && { facts "$d/events.log" > "$OUT/facts-$1.txt"; sed 's/^/    /' "$OUT/facts-$1.txt" | tee -a "$OUT/window.log" >/dev/null; }
    echo "$line" | grep -oE '[0-9.]+ rows/s' | cut -d' ' -f1 > "$OUT/rows_$1${2:+_p}"; }
dump() {    # $1 arm, $2 long-prompt tokens (0: none)
    local f=$OUT/bt_$1_L$2 cfg; cfg=$(tiercfg "$(cfg_of $1)" "$(tier bt_$1_L$2)")
    free_gpus || log "GPUs busy before dump $1"
    log "dump $1 (long prompt $2): start"
    timeout -s INT -k 120 1500 bash -c "cd $R/Strata-1441 && env $(cfg_env "$cfg") BATCH_TEST_LOG=$f.engine.log \
        python3 tools/batch_test16.py --exe $EXE --config $cfg --batch 16 --n 16 --max-new 160 --skip-solo \
        $( [ "$2" -gt 0 ] && echo "--long-tokens $2" ) --dump $f.json" > "$f.txt" 2>&1
    local rc=$? adm; adm=$(grep -aoE 'admissions [0-9.]+s' "$f.txt" | grep -oE '[0-9.]+' | head -1)
    log "dump $1 (long prompt $2): exit $rc, admissions ${adm:-?} s"
    if [ "$2" -gt 0 ] && ! python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) >= 4.0 else 1)" "${adm:-0}"; then
        log "  WARNING: admissions < 4 s - the long prompt was not read beside the decoding slots; dump $1 not meaningful"
        echo NOREAD > "$f.flag"; fi
    grep -aE "borrows|no loan|have experts out of VRAM|lend|SPLIT_OWN_RESERVE" "$f.engine.log" | sed 's/^[0-9.]* E //' | cut -c1-170 \
        | sed 's/^/    /' | tee -a "$OUT/window.log" >/dev/null; }
same() {    # batch tokens only (no solo pass)
    [ -f "$OUT/bt_$1_L$2.flag" ] || [ -f "$OUT/bt_prod_L$2.flag" ] && { echo NOREAD; return; }
    python3 - "$OUT/bt_prod_L$2.json" "$OUT/bt_$1_L$2.json" <<'EOF'
import json, sys
try:
    a, b = (json.load(open(p))["batch"] for p in sys.argv[1:])
except Exception:
    print("MISSING"); sys.exit()
d = [i for i in range(max(len(a), len(b))) if i >= len(a) or i >= len(b) or a[i] != b[i]]
print("IDENTICAL" if not d else "DIFFER (slots " + ",".join(map(str, d)) + ")")
EOF
}

log "window start: stopping strata.service"; systemctl --user stop strata.service
for i in $(seq 60); do curl -s -m 2 $URL/health >/dev/null || break; sleep 2; done
free_gpus || { log "GPUs not free"; start_prod; exit 1; }

# 1. production reference (same window, same temperatures): rows/s, stage profile, greedy token dump
decode prod
decode prod p
dump prod 40000
# 2. arms
for c in "${ARMS[@]}"; do
  decode "$c"
  [ "$c" = s28r ] && decode "$c" p
  dump "$c" 40000
  v=$(same "$c" 40000); log "exactness $c vs prod (greedy, 40K prompt read beside decode): $v"
  case "$v" in DIFFER*)   # the split itself, or only the loan's streamed experts during the long read?
    [ -f "$OUT/bt_prod_L0.json" ] || dump prod 0
    dump "$c" 0; log "exactness $c vs prod (greedy, no long prompt): $(same "$c" 0)" ;;
  esac
done
# stage tables (ms per stage-window; NB vprof.py's GROUP predates the "router+plan+bell"/"hc1 *" names: those rows
# read 0.00 but are in "total stamped")
( cd "$OUT" && python3 $P/vprof.py dec_prod_p/events.log dec_s28r_p/events.log ) > "$OUT/vprof.txt" 2>&1
sed 's/^/    /' "$OUT/vprof.txt" >> "$OUT/window.log"
# 3. HTTP only for an arm that is clearly faster in pure decode and exact
ref=$(cat "$OUT/rows_prod" 2>/dev/null); HTTP=()
for c in "${ARMS[@]}"; do
  r=$(cat "$OUT/rows_$c" 2>/dev/null)
  g=$(python3 -c "import sys; a,b=sys.argv[1:]; print(f'{(float(b)/float(a)-1)*100:+.1f}')" "${ref:-0}" "${r:-0}" 2>/dev/null || echo "n/a")
  v=$(same "$c" 40000); log "decode $c vs prod: ${r:-?} vs ${ref:-?} rows/s ($g %), dump $v"
  if [ "$v" = IDENTICAL ] && python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) >= 1.5 else 1)" "$g" 2>/dev/null; then
    HTTP+=("$(tiercfg "$(cfg_of $c)" "$(tier ab_$c)")"); fi
done
if [ ${#HTTP[@]} -gt 0 ]; then
  log "step ab: start (${HTTP[*]##*/})"
  ( cd $R/linux && SERVER_DIR=$R/Strata-143-prod REPS=2 ORDER=abba timeout -s INT 3600 \
      ./ab_quick.sh split3 "$(tiercfg "$(cfg_of prod)" "$(tier ab_prod)")" "${HTTP[@]}" ) > "$OUT/ab.txt" 2>&1
  log "step ab: exit $?"; systemctl --user stop strata-bench.scope 2>/dev/null; free_gpus || log "GPUs busy after ab"
else
  log "step ab: skipped (no arm >= +1.5 % in pure decode with an identical dump)"
fi
trap - INT TERM
start_prod; log "window done"
rm -rf "$OUT/tier"
