#!/usr/bin/env bash
# GPU night window "212" (port plan section 4, 2026-11-07/08): production (ll10b + production.json) against upstream
# v0.2.12 (strata-212-stocktab, stock serve/server.py of Strata-212) and the partial port (strata-212m-p2 = v0.2.12 +
# P1/P2: QSA_SLOTS_STREAM, GR_DOWN_PF, EARLY_SAMPLE; server Strata-212m/serve).  Text only (the stock binaries still
# have the batch M-RoPE bug #1242).  Production is restored on every exit path; a deploy is never part of this.
#
# Phases (times for the default 23:45 start; every step has a cap, no step starts past DEADLINE):
#   0 prep, no GPU: binaries pinned by sha256, configs/T212-*.json checked (every STRATA_* / --flag exists in the arm's
#     source tree), production recorded (binary + config sha, /metrics, GET /v1/load, RSS, VRAM, power limits)
#   1 CPU host tests while production still serves (23:45-00:00): CUDA_VISIBLE_DEVICES= ctest on Strata-212m/build,
#     serve unittests of Strata-212m (scopes with a hard MemoryMax, nice 19, CPUs 0-63).  A failure of one of OUR
#     tests (the ones the port added or changed) skips the H arms; the stock arms still run
#   2 production stopped (the reminder to coordinate with the docs/bench session is logged first), GPUs checked free
#   3 GPU ctest subset on one card (212m build): a failing parity test drops its lever from tonight's arms
#   4 identity: tools/batch_test.py (Strata-212's, greedy, 16 slots, a 40K prompt, solo + batch dumps) with production,
#     stocktab (B env) and 212m-p2 (H env); then batch_interleave / batch_groups_cache (SLOT_CKPT_MOVE 1 and 0) /
#     cancel_park on 212m-p2.  IDENT_GATE=strict (the plan): B != A skips the stocktab arms, H != A skips the 212m arms;
#     IDENT_GATE=port: only H != B skips the 212m arms (B vs A INFO); IDENT_GATE=off: nothing is skipped
#   5 performance arms, one fresh server each on 127.0.0.1:8091: warm-up + 4-user probe (the arm's INFO lines checked:
#     a path that did not engage stops the arm), bench_concurrency short 1/2/4/8/16 users (greedy; per-user decode at
#     1/2/4), long 16 users, bench_multiturn 16x3 greedy, piped_read_join_bench cold32k (a cold 32K join beside 4
#     streams); /metrics, peak memory, VRAM per card.  Exact arms (C D H H2 H3) must return B's short replies
#     (HTTP_GATE=1: a difference stops the arm).  Then r2 repeats of close pairs while time is left
#   6 restore (DEADLINE, default 04:00, or earlier when done): strata.service with the unchanged production.json, checks
#     (/v1/models, one chat, one image, GET /v1/load, /metrics, the log's INFO lines), "window done", summary table
#
# Arms (configs/T212-<arm>.json, made by --gen-configs from production.json; T212.manifest.json lists every change):
#   A   production: ll10b, production.json as is, server Strata-143-prod + the drop-in's server env
#   B   stocktab: production env minus STRATA_PARK_ASYNC (another feature upstream) and minus every switch v0.2.12 lacks,
#       plus parity pins (SM86_PERF GDN_TC GDN_CONVL2 PROMPT_ATTN_IMMA AUTO_NOBORROW PF_W16_CACHE_MIB MMQ_KQ = 0, stack
#       BATCH_GDN_GROUPS BATCH_ATTN_ROWS CKPT_POOL_READY = 0); "tool_call_recovery": false, "drafting": "mtp"
#   C   B + the upstream stack (BATCH_GDN_GROUPS=1 BATCH_ATTN_ROWS=1 CKPT_POOL_READY=4) + the sm_86 table (SM86_PERF unset)
#   D   C + STRATA_IQ4_W4A8=1 + STRATA_SELECT_TWO_PASS=1
#   E   C + STRATA_SPLIT_MTP_BATCH=0 + STRATA_PIPED_CONCURRENT=1 (stops after the probe when the engine says it is off)
#   E2  E + STRATA_PIPED_READ_PIECE=16384
#   F   C + STRATA_PW_WITH_BATCH=1 + --pipeline-windows 2 (on CUDA the pipeline-windows auto default is HIP-only)
#   G   C + --batch-mtp + STRATA_PIPED_MTP=1 + STRATA_BATCH_SELF_COMMIT=0 (PIPED_MTP refuses SELF_COMMIT)
#   H   212m-p2: production env minus PARK_ASYNC and what 212m lacks (keeps QSA_SLOTS_STREAM GR_DOWN_PF=2 EARLY_SAMPLE)
#       + B's parity pins + STRATA_ATTN_DAILY=0
#   H2  H - QSA_SLOTS_STREAM + STRATA_ATTN_DAILY=1 (DAILY alone; the rest of the sm_86 table stays off)
#   H3  the better of H / H2 (short agg at 16 users; T212-H3 or T212-H3d) + IQ4_W4A8=1 + the upstream stack
#   I   212i (strata-212i, server Strata-212i/serve: v0.2.12 + our ports, the candidate production): production env with
#       PARK_ASYNC -> PARK_CAPTURE=1, minus what is not ported (PIPE_OVERLAP(_KVS) SPEC_LANES SOLO_PIPELINE READ_YIELD
#       ATTN_VBATCH READS*), + SERVE_STOP_BOUND=1 + H's parity pins + ATTN_DAILY=0; production's whole server env.  An
#       exact arm against H (B when H has no replies); its own bt identity (I vs H) and INFO steps (never counted as
#       crashes in a row): GPU ctest conversation_* on Strata-212i/build, batch_groups_cache with PARK_CAPTURE_VERIFY=1
#       SLOT_PARK_VERIFY=1, and tools/vision_batch_check.py (server mode) against I's server.  Problems of the 212i
#       tree / build skip only I (production and the other arms run).  Not in ARMS: none of I's steps run
#   The 212p arms (binary 'pipe' = strata-212p, server Strata-212p/serve = 212i's server; g3-212-pipe = 212i + P9
#   PIPE_OVERLAP(_KVS), P10 SOLO_PIPELINE + SPEC_LANES, P11 READ_YIELD; run "w212b", 2026-11-08):
#   P   I's env + production's PIPE_OVERLAP=1 PIPE_OVERLAP_KVS=1 SPEC_LANES=4 SOLO_PIPELINE=1 READ_YIELD=1 (all of
#       production's features).  Exact: short replies = I's (last night's w212-I-short from P_REF, I's own when I runs
#       too), bt-P = bt-A (16/16 solo + batch)
#   PT  P + the sm_86 exact table (SM86_PERF unset, ATTN_DAILY=0 kept, QSA_SLOTS(_STREAM)=1) + the upstream stack
#       (BATCH_GDN_GROUPS=1 BATCH_ATTN_ROWS=1 CKPT_POOL_READY=4); the bit-changing pins stay.  THE candidate for
#       production: exact against P (short replies) and A (bt-PT = bt-A, bt-PT = bt-P).  INFO steps: interleave,
#       batch_groups_cache with the VERIFY switches, cancel_park, batch_overlap_test (engine, phase 4), early_close,
#       stop_resend and vision_batch_check (its server, after its benches)
#   PTD PT with ATTN_DAILY=1 QSA_SLOTS_STREAM=0 (the table's own attention).  Exact against P
#   PT4 PT + IQ4_W4A8=1 + SELECT_TWO_PASS=1 (D's switches).  Exact against P
#   PTX PT without the pins of the bit-changing upstream defaults (GDN_TC GDN_CONVL2 PROMPT_ATTN_IMMA MMQ_KQ
#       PF_W16_CACHE_MIB AUTO_NOBORROW: upstream's defaults apply).  NOT exact: INFO only ("how much speed is on the
#       table"), never gated, never a candidate; its identity is reported as DIFF-expected
#   With pipe arms in ARMS the host and GPU ctests run on Strata-212p/build (212m's only when 212m arms are listed too)
#
#   ./window-212.sh --dry-run        # checks + the plan; touches no service, no GPU, writes nothing
#   ./window-212.sh                  # the window (refuses before WINDOW_START unless FORCE=1)
#   ./window-212.sh --summary        # the summary table again from results/$RUN
#   ./window-212.sh --gen-configs    # (re)write configs/T212-*.json + T212.manifest.json from production.json
#   env: ARMS="A B C D E E2 F G H H2 H3 I" WINDOW_START="YYYY-MM-DD 23:45" DEADLINE="YYYY-MM-DD 04:00" (or HH:MM, default 04:00)
#        (WINDOW_START unset: tonight's 23:45 - last night's before noon -, or a full-date DEADLINE - 4 h 15 min)
#        RUN=w212 (results/$RUN, units $RUN-*, bench tier dir $RUN; RUN=w212b defaults ARMS to "A P PT PTD PT4 PTX")
#        NOW=1 (WINDOW_START = now, DEADLINE = now + DEADLINE_H hours, default 3; an explicit DEADLINE wins)
#        P_REF=results/w212/sweep.requests.jsonl (I's short replies for P when I is not in ARMS) R2_FIRST=auto ("A PT P" with pipe arms)
#        PIPE_SHA=74702164 (strata-212p's expected sha256 prefix; empty: any) SERVE_UNIT_CAP=600 ARM_TYP=450 (estimate only)
#        IDENT_GATE=strict|port|off HTTP_GATE=1 HOST_TESTS=1 GPU_CTEST=1 IDENTITY=1 SPARE=1 FORCE=0 MEMMAX=224G
#        RSS_MAX_GIB=200 PRESS_MAX=40 VRAM_MIN=300 HANG_S=120 RESTORE_SMOKE=1
#   verdicts: grep -E 'PASS|FAIL|SKIP|ABORT' results/$RUN/window.log; table: results/$RUN/summary.txt
set -u
export LC_ALL=C
R=/media/raid0/StrataQ38FN; L=$R/linux; C=$L/configs
PY=$R/Strata/.venv/bin/python
UP=$R/Strata-212; PWT=$R/Strata-212m; IWT=$R/Strata-212i; PPT=$R/Strata-212p; PRODSRV=$R/Strata-143-prod; OURS=$R/Strata-ll10-borrow; ENG=$R/Strata/engine
PPORT=8090; TPORT=${TPORT:-8091}; URL=http://127.0.0.1:$PPORT; TURL=http://127.0.0.1:$TPORT
RUN=${RUN:-w212}   # the run's name: results dir, bench tier dir, unit names and lock (never a leftover of another run)
[[ $RUN =~ ^w212[A-Za-z0-9_-]{0,20}$ ]] || { echo "RUN must be w212 or w212<suffix> ([A-Za-z0-9_-])"; exit 2; }
OUT=${OUT:-$L/results/$RUN}; TIER=/media/raid0/strata-kvspill-bench/$RUN
BUNIT=$RUN-bench; HUNIT=$RUN-host; RUNIT=$RUN-restore; GUNIT=$RUN-guard
LOCK=/run/user/$(id -u)/window-212$( [ "$RUN" = w212 ] || echo "-$RUN").lock
PRESS=/sys/fs/cgroup/user.slice/user-$(id -u).slice/user@$(id -u).service/memory.pressure
MODE=run
while [ $# -gt 0 ]; do
    case $1 in
        --dry-run) MODE=dry ;;
        --summary) MODE=summary ;;
        --gen-configs) MODE=gen ;;
        -h|--help) sed -n '2,/^set -u/p' "$0" | sed '$d'; exit 0 ;;
        *) echo "unknown argument $1 (--dry-run, --summary, --gen-configs)"; exit 2 ;;
    esac; shift
done
[ "${DRY:-0}" = 1 ] && MODE=dry
if [ "$RUN" = w212b ]; then ARMS=${ARMS:-A P PT PTD PT4 PTX}; else ARMS=${ARMS:-A B C D E E2 F G H I H2 H3}; fi
P_REF=${P_REF:-$L/results/w212/sweep.requests.jsonl}   # last night's w212-I-short: P's reference without an I run
R2_FIRST=${R2_FIRST-auto}   # r2 repeats run first (those in ARMS that ran ok), then the close pairs; auto: "A PT P" with pipe arms
IDENT_GATE=${IDENT_GATE:-strict}; HTTP_GATE=${HTTP_GATE:-1}; HOST_TESTS=${HOST_TESTS:-1}; GPU_CTEST=${GPU_CTEST:-1}
IDENTITY=${IDENTITY:-1}; SPARE=${SPARE:-1}; FORCE=${FORCE:-0}; RESTORE_SMOKE=${RESTORE_SMOKE:-1}
MEMMAX=${MEMMAX:-224G}; RSS_MAX_GIB=${RSS_MAX_GIB:-200}; PRESS_MAX=${PRESS_MAX:-40}; VRAM_MIN=${VRAM_MIN:-300}
HANG_S=${HANG_S:-120}; BOOT_CAP=${BOOT_CAP:-900}; BOOT_HANG=${BOOT_HANG:-300}
SERVE_UNIT_CAP=${SERVE_UNIT_CAP:-600}    # phase 1's serve unittests (production serving; stock test_stream_ownership hangs)
ARM_EST=${ARM_EST:-720}                  # an arm starts only with this much time left before DEADLINE
HARD_GRACE=420; RESTORE_GRACE=720        # HARD_STOP = DEADLINE + 7 min (TERM to this script), restore timer + 12 min
if [ "${NOW:-0}" = 1 ]; then   # a daytime window from now: DEADLINE = now + DEADLINE_H hours unless given
    [[ ${DEADLINE_H:-3} =~ ^[1-9][0-9]?$ ]] || { echo "DEADLINE_H must be whole hours (1-99)"; exit 2; }
    WINDOW_START=${WINDOW_START:-$(date '+%F %T')}
    DEADLINE=${DEADLINE:-$(date -d "@$(( $(date -d "$WINDOW_START" +%s) + ${DEADLINE_H:-3} * 3600 ))" '+%F %T')}
fi
if [ -z "${WINDOW_START:-}" ]; then   # no fixed date (a past one refuses nothing): a full-date DEADLINE's 23:45 the night
    if [ -n "${DEADLINE:-}" ] && ! [[ $DEADLINE =~ ^[0-9]{1,2}:[0-9]{2}$ ]]; then   # before = DEADLINE - 4 h 15 min;
        _d=$(date -d "$DEADLINE" +%s) || exit 2; WINDOW_START=$(date -d "@$(( _d - 15300 ))" '+%F %T')
    elif [ "$(date +%H)" -lt 12 ]; then WINDOW_START="$(date -d yesterday +%F) 23:45"   # else tonight's 23:45 (before
    else WINDOW_START="$(date +%F) 23:45"; fi                                         # noon: last night's)
fi
DEADLINE=${DEADLINE:-04:00}
case $IDENT_GATE in strict|port|off) ;; *) echo "IDENT_GATE must be strict, port or off"; exit 2 ;; esac
[ "$TPORT" = "$PPORT" ] && { echo "TPORT $TPORT is production's port: refused"; exit 2; }
WS_TS=$(date -d "$WINDOW_START" +%s) || exit 2
if [[ $DEADLINE =~ ^[0-9]{1,2}:[0-9]{2}$ ]]; then
    DL_TS=$(date -d "$(date -d "@$WS_TS" +%F) $DEADLINE" +%s); [ "$DL_TS" -le "$WS_TS" ] && DL_TS=$(( DL_TS + 86400 ))
else DL_TS=$(date -d "$DEADLINE" +%s) || exit 2; fi
[ "$DL_TS" -gt "$WS_TS" ] || { echo "DEADLINE must be after WINDOW_START"; exit 2; }
for v in $(compgen -e | grep '^STRATA_'); do unset "$v"; done   # nothing of ours from the calling shell

# arm -> binary key, server tree, engine source tree (for the name checks)
declare -A BIN_SRC=([prod]=$ENG/strata-g3-ll10b [stocktab]=$ENG/strata-212-stocktab [port]=$ENG/strata-212m-p2
                    [int]=$ENG/strata-212i [pipe]=$ENG/strata-212p)
declare -A BIN_TREE=([prod]=$OURS [stocktab]=$UP [port]=$PWT [int]=$IWT [pipe]=$PPT)
declare -A ARM_BIN=([A]=prod [B]=stocktab [C]=stocktab [D]=stocktab [E]=stocktab [E2]=stocktab [F]=stocktab [G]=stocktab
                    [H]=port [H2]=port [H3]=port [H3d]=port [I]=int [P]=pipe [PT]=pipe [PTD]=pipe [PT4]=pipe [PTX]=pipe)
declare -A ARM_SRV=([A]=$PRODSRV [B]=$UP [C]=$UP [D]=$UP [E]=$UP [E2]=$UP [F]=$UP [G]=$UP [H]=$PWT [H2]=$PWT [H3]=$PWT [H3d]=$PWT
                    [I]=$IWT [P]=$PPT [PT]=$PPT [PTD]=$PPT [PT4]=$PPT [PTX]=$PPT)
EXACT_ARMS=" C D H H2 H3 H3d I P PT PTD PT4 "   # greedy replies must equal B's (I: H's; P: I's; PT*: P's); PTX is not exact
PIPE_ARMS="P PT PTD PT4 PTX"
ICB=$IWT/build                           # 212i's ctest build (phase 3's INFO conversation_* tests)
PCB=$PPT/build                           # 212p's ctest build (host + GPU ctest when pipe arms run)
I_CTESTS="conversation_snapshot_test conversation_snapshot_test_ckpt_async conversation_snapshot_test_ckpt_vram_slots
 conversation_cache_test conversation_alloc_test conversation_alloc_test_off conversation_tier_test"
# 212p's GPU tests beside GPU_TESTS: P9's bitwise overlap test (serial vs concurrent prompt read beside streamed decode)
# and the plain streamed-K/V test it builds on; P9-P11's other tests (read_yield, overlap_kvs, solo_pipeline,
# spec_lanes, arena_carve) are CPU tests of the host phase
P_GPU_TESTS="kv_stream_overlap_parity kv_stream_parity"
PIPE_SHA=${PIPE_SHA-74702164}            # strata-212p's sha256 prefix (g3-212-pipe a099d295, 2026-11-08 04:22)
in_arms() { case " $ARMS " in *" $1 "*) return 0 ;; esac; return 1; }
any_arm() { local a; for a in "$@"; do in_arms "$a" && return 0; done; return 1; }
pipe_on() { any_arm $PIPE_ARMS; }
old_on() { any_arm B C D E E2 F G H H2 H3 H3d I; }   # the arms of the first night (212m's host / GPU ctest, bt-B/H)
# INFO lines: WANT_BOOT must be in the engine log after the probe (else the path did not engage: the arm stops),
# FAILPAT stops the arm when it appears, WANT_END is checked at the arm's end (INFO: proven / not seen); ';;' separates
SM86='sm_86 \(Ampere\): [0-9]+ exact speed switches on by default'
MMQD='prompt MMQ reads resident experts in place \(STRATA_MMQ_DIRECT=1\)'
ES='STRATA_EARLY_SAMPLE=1: sampled rows'
IQ4='prompt experts: the IQ4 W4A8 kernel runs'
# 212m's told-once line; with --kv-resident every slot's K/V is streamed (kv_mode 1), so H's rows take slots-stream, and
# H2 (stream off) the per-row path: its DAILY shows as the line's STRATA_ATTN_DAILY=1
SLOTS='batch decode attention: slots(-stream)? \(.*STRATA_ATTN_DAILY=0'
DAILY='batch decode attention: .*STRATA_ATTN_DAILY=1'
# 212i's told-once start lines (generate.cpp; all printed at the engine's start, before the probe): park capture, slot park,
# disk tier, mmap buffers, evicting gates, MSG_CKPT_TAIL.  PARK_BORROW, SLOT_CHAIN_REUSE, GIVEBACK_CHECKS and
# SERVE_STOP_BOUND print nothing; MROPE_SLOTS says mrope_slots=1 on the INFO protocol line (stdout, not the log) and its
# regions line only when the regions fit: those are end-of-arm INFO
IPARK='STRATA_PARK_CAPTURE: a finished request.s conversation is captured beside the slots'
ISLOT='STRATA_SLOT_PARK: a pipelined slot.s conversation is captured when it ends'
ITIER='conversation tier: on: /media/raid0/strata-kvspill-bench/'
IMMAP='STRATA_CONV_MMAP: parked K/V segments'; IGATE='STRATA_GATE_EVICT: the physical-RAM gates evict only what can help'
ITAIL='STRATA_MSG_CKPT_TAIL=[0-9]+: a prompt read also checkpoints'
# 212p's told-once start lines (generate.cpp): the lanes, the solo pipeline (both printed at the engine's start); the
# overlap prints its first streamed read and one stats line per overlapped read (end of arm); READ_YIELD says only
# when it is off.  The sm_86 table: 9 switches, ATTN_DAILY=0 is set by us ("a switch you set is kept"), so 8
LANES='STRATA_SPEC_LANES=4: lanes on slots'; SOLO='STRATA_SOLO_PIPELINE: --pipeline-windows 2 beside --batch'
SM86_8='sm_86 \(Ampere\): 8 exact speed switches on by default'
SLOTSS='batch decode attention: slots-stream \(.*STRATA_ATTN_DAILY=0'
P_BOOT="$MMQD;;$ES;;$SLOTS;;$IPARK;;$ISLOT;;$ITIER;;$IMMAP;;$IGATE;;$ITAIL;;$LANES;;$SOLO"
P_FAIL="STRATA_PARK_CAPTURE is off|STRATA_SLOT_PARK is off|STRATA_PARK_ASYNC=1 is ignored|conversation tier: off|STRATA_SPEC_LANES is off|STRATA_SOLO_PIPELINE is off|STRATA_READ_YIELD is off beside"
P_END="overlapped read of [0-9]+ tokens;;STRATA_PIPE_OVERLAP_KVS=1: first overlapped read with a streamed;;conversation tier: on;;conversation cache: restored [0-9]+"
declare -A WANT_BOOT=([A]="STRATA_SPEC_LANES=4: lanes on slots;;STRATA_SOLO_PIPELINE;;$MMQD" [B]="$MMQD" [C]="$MMQD;;$SM86"
    [D]="$SM86" [E]="$SM86" [E2]="$SM86" [F]="$SM86;;--pipeline-windows 2 stays on for the solo path"
    [G]="$SM86;;STRATA_PIPED_MTP: [0-9]+ pipelined groups?" [H]="$MMQD;;$ES;;$SLOTS"
    [H2]="$ES;;$DAILY" [H3]="$ES" [H3d]="$ES;;$DAILY"
    [I]="$MMQD;;$ES;;$SLOTS;;$IPARK;;$ISLOT;;$ITIER;;$IMMAP;;$IGATE;;$ITAIL"
    [P]="$P_BOOT" [PT]="$P_BOOT;;$SM86_8;;$SLOTSS" [PTD]="${P_BOOT//";;$SLOTS"/};;$SM86_8;;$DAILY" [PT4]="$P_BOOT;;$SM86_8;;$SLOTSS"
    [PTX]="$P_BOOT;;$SM86_8")
declare -A FAILPAT=([B]="$SM86|STRATA_MMQ_DIRECT=[0-9] is ignored" [H]="$SM86" [H2]="$SM86" [H3]="$SM86" [H3d]="$SM86"
    [E]="STRATA_PIPED_CONCURRENT is off beside" [F]="--pipeline-windows 2 is off"
    [G]="WARNING: STRATA_PIPED_MTP is off|WARNING: --batch-mtp is off"
    [I]="$SM86|STRATA_PARK_CAPTURE is off|STRATA_SLOT_PARK is off|STRATA_PARK_ASYNC=1 is ignored|conversation tier: off"
    [P]="$SM86|$P_FAIL" [PT]="$P_FAIL" [PTD]="$P_FAIL" [PT4]="$P_FAIL" [PTX]="$P_FAIL")
declare -A WANT_END=([A]="overlapped read of [0-9]+ tokens;;conversation tier: on" [D]="$IQ4;;two-pass scores on"
    [E]="read [0-9]+ tokens on a thread of its own" [E2]="piped read [0-9]+ tokens in;;read [0-9]+ tokens on a thread of its own"
    [H]="batch decode attention: slots-stream" [H3]="$IQ4;;batch decode attention: slots-stream" [H3d]="$IQ4"
    [I]="batch decode attention: slots-stream;;the image position table holds [0-9]+ regions;;conversation cache: borrowed [0-9]+ of a parked;;conversation cache: restored [0-9]+"
    [P]="$P_END;;batch decode attention: slots-stream" [PT]="$P_END;;batch decode attention: slots-stream;;the image position table holds [0-9]+ regions"
    [PTD]="$P_END" [PT4]="$P_END;;$IQ4;;two-pass scores on" [PTX]="$P_END")
# a dropped lever's INFO line is not wanted any more (STREAM off: with --kv-resident the rows take the per-row path, so
# neither slots line can appear - $SLOTS too)
declare -A DROP_PAT=([STRATA_MMQ_DIRECT]="$MMQD" [STRATA_IQ4_W4A8]="$IQ4" [STRATA_EARLY_SAMPLE]="$ES"
    [STRATA_ATTN_DAILY]="$DAILY" [STRATA_QSA_SLOTS_STREAM]="$SLOTSS|||$SLOTS|||batch decode attention: slots-stream"
    [STRATA_PIPE_OVERLAP_KVS]="STRATA_PIPE_OVERLAP_KVS=1: first overlapped read with a streamed")
wants() {   # <pattern list>: without the patterns of dropped levers (DROP_PAT: '|||' separates several, longest first)
    local w=$1 k p rest; for k in $DROPS; do rest=${DROP_PAT[$k]:-}
        while [ -n "$rest" ]; do p=${rest%%"|||"*}; [ "$p" = "$rest" ] && rest="" || rest=${rest#*"|||"}
            w=${w//"$p"/}; done; done; echo "$w"; }
# the parity tests that guard a lever (a FAILED test, host or GPU phase, drops the lever: set to 0 in every v0.2.12 arm;
# sampler_same_test is a host test)
declare -A LEVER_TESTS=([STRATA_QSA_SLOTS_STREAM]="qsa_slots_stream_parity qsa_slots_stream_parity_lanecell"
    [STRATA_GR_DOWN_PF]="gr_parity_down_pf2" [STRATA_EARLY_SAMPLE]="sampler_same_test" [STRATA_IQ4_W4A8]="prefill_iq4_w4a8_test"
    [STRATA_SELECT_TWO_PASS]="qsa_two_pass_parity" [STRATA_ATTN_DAILY]="qsa_merge_daily_parity_daily"
    [STRATA_MMQ_DIRECT]="prefill_mmq_tab_test expert_cache_slot_guard_test" [STRATA_PIPE_OVERLAP_KVS]="kv_stream_overlap_parity")
declare -A ARM_POINT=([D]="STRATA_IQ4_W4A8 STRATA_SELECT_TWO_PASS" [H2]="STRATA_ATTN_DAILY"   # all dropped: the arm is skipped
    [PTD]="STRATA_ATTN_DAILY" [PT4]="STRATA_IQ4_W4A8 STRATA_SELECT_TWO_PASS")
GPU_TESTS="prefill_mmq_tab_test expert_cache_slot_guard_test qsa_slots_parity qsa_slots_parity_lanecell qsa_slots_stream_parity
 qsa_slots_stream_parity_lanecell gr_parity gr_parity_down_pf2 gr_parity_down_pf3 conversation_snapshot_test
 conversation_snapshot_test_ckpt_async conversation_snapshot_test_ckpt_vram_slots conversation_copy_streams_test batch_rows_parity
 prefill_iq4_w4a8_test qsa_two_pass_parity gemm_w16_parity"
DAILY_TESTS="qsa_merge_daily_parity_default qsa_merge_daily_parity_daily"
OUR_TESTS_FIXED="gr_parity conversation_snapshot_test serve_stop_bound_test prefill_mmq_tab_test expert_cache_slot_guard_test"
CTB=$PWT/build; CTK=port; CTT=$PWT       # the ctest build of phases 1 and 3, its binary key and tree: 212m's, or
if pipe_on && ! old_on; then CTB=$PCB; CTK=pipe; CTT=$PPT; GPU_TESTS="$GPU_TESTS $P_GPU_TESTS"; fi   # 212p's (pipe arms only)
BT_EXTRA="--pcie-frac 0 --adapt-every 1000000"     # docs/BATCHING.md: exact comparisons
CRASH_RE='CUDA error|illegal memory access|an illegal (instruction|address)|unspecified launch failure|misaligned address|non-finite logits|terminate called|Segmentation fault|core dumped|: out of memory|cudaErrorMemoryAllocation'
SCOPE=(systemd-run --user --scope --quiet --unit=$BUNIT -p MemoryMax=$MEMMAX -p MemorySwapMax=0)
GENV=(CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 PATH=/usr/local/cuda-13.0/bin:/usr/bin:/bin)

# ---- configs ------------------------------------------------------------------------------------------------------
gen_configs() {   # configs/T212-<arm>.json + T212.manifest.json from production.json (production.json is only read)
    "$PY" -I - "$C/production.json" "$C" "$OUT" "$TIER" "$UP" "$PWT" "$OURS" "$ENG" "$TPORT" "$IWT" "$PPT" <<'PYEOF'
import copy, hashlib, json, os, subprocess, sys, time
prod_path, cdir, outdir, tier, up, port, ours, eng, tport, intw, pipew = sys.argv[1:12]
src = os.path.realpath(prod_path)
raw = open(src, "rb").read()
prod = json.loads(raw)
def has(tree, name):        # the quoted name in the tree's tracked files (the source the binary is built from)
    return subprocess.run(["git", "-C", tree, "grep", "-q", "-F", f'"{name}"', "--", "src", "include", "serve",
                           "strata_control"], capture_output=True).returncode == 0
def flag_ok(tree, flag):
    return subprocess.run(["git", "-C", tree, "grep", "-q", "-F", f'"{flag}"', "--", "src/program/generate.cpp"],
                          capture_output=True).returncode == 0
PINS = {"STRATA_SM86_PERF": "0", "STRATA_GDN_TC": "0", "STRATA_GDN_CONVL2": "0", "STRATA_PROMPT_ATTN_IMMA": "0",
        "STRATA_AUTO_NOBORROW": "0", "STRATA_PF_W16_CACHE_MIB": "0", "STRATA_MMQ_KQ": "0",
        "STRATA_BATCH_GDN_GROUPS": "0", "STRATA_BATCH_ATTN_ROWS": "0", "STRATA_CKPT_POOL_READY": "0"}
STACK = {"STRATA_BATCH_GDN_GROUPS": "1", "STRATA_BATCH_ATTN_ROWS": "1", "STRATA_CKPT_POOL_READY": "4"}
penv = prod["env"]
def base(tree):
    keep, gone = {}, []
    for k, v in penv.items():
        if k == "STRATA_PARK_ASYNC":              # upstream f73b3c86: a different feature (VRAM-stage park) by that name
            gone.append(k + " (another feature in v0.2.12)")
        elif k.startswith("MALLOC_") or has(tree, k):
            keep[k] = v
        else:
            gone.append(k)
    return keep, gone
envs, args_add, notes, dropped = {}, {}, {}, {}
up_env, up_gone = base(up)
port_env, port_gone = base(port)
envs["B"] = {**up_env, **PINS}
c = {k: v for k, v in envs["B"].items() if k != "STRATA_SM86_PERF"}
envs["C"] = {**c, **STACK}
envs["D"] = {**envs["C"], "STRATA_IQ4_W4A8": "1", "STRATA_SELECT_TWO_PASS": "1"}
envs["E"] = {**envs["C"], "STRATA_SPLIT_MTP_BATCH": "0", "STRATA_PIPED_CONCURRENT": "1"}
envs["E2"] = {**envs["E"], "STRATA_PIPED_READ_PIECE": "16384"}
envs["F"] = {**envs["C"], "STRATA_PW_WITH_BATCH": "1"}; args_add["F"] = ["--pipeline-windows", "2"]
envs["G"] = {**envs["C"], "STRATA_PIPED_MTP": "1", "STRATA_BATCH_SELF_COMMIT": "0"}; args_add["G"] = ["--batch-mtp"]
envs["H"] = {**port_env, **PINS, "STRATA_ATTN_DAILY": "0"}
for k, v in (("STRATA_QSA_SLOTS_STREAM", "1"), ("STRATA_GR_DOWN_PF", "2"), ("STRATA_EARLY_SAMPLE", "1")):
    envs["H"].setdefault(k, v)
envs["H2"] = {k: v for k, v in envs["H"].items() if k != "STRATA_QSA_SLOTS_STREAM"}; envs["H2"]["STRATA_ATTN_DAILY"] = "1"
envs["H3"] = {**envs["H"], "STRATA_IQ4_W4A8": "1", **STACK}
envs["H3d"] = {**envs["H2"], "STRATA_IQ4_W4A8": "1", **STACK}
# I: 212i = v0.2.12 + our ports.  Our park capture is STRATA_PARK_CAPTURE there (STRATA_PARK_ASYNC is upstream's other
# park: never set); not ported: dropped by name even if a later 212i grows a switch of that name
NOT_PORTED = ("STRATA_PIPE_OVERLAP", "STRATA_PIPE_OVERLAP_KVS", "STRATA_SPEC_LANES", "STRATA_SOLO_PIPELINE",
              "STRATA_READ_YIELD", "STRATA_ATTN_VBATCH")
int_env, int_gone = base(intw)
for k in list(int_env):
    if k in NOT_PORTED or k.startswith("STRATA_READS"):
        int_env.pop(k); int_gone.append(k + " (not ported)")
if penv.get("STRATA_PARK_ASYNC") == "1":
    int_env["STRATA_PARK_CAPTURE"] = "1"
int_env["STRATA_SERVE_STOP_BOUND"] = "1"                 # 212i: opt-in on CUDA (on by default on HIP only)
int_env["STRATA_CONV_TIER_DIR"] = f"{tier}/I"            # never production's tier dir (a fresh dir per start at run time)
envs["I"] = {**int_env, **PINS, "STRATA_ATTN_DAILY": "0"}
for k, v in (("STRATA_QSA_SLOTS_STREAM", "1"), ("STRATA_GR_DOWN_PF", "2"), ("STRATA_EARLY_SAMPLE", "1")):
    envs["I"].setdefault(k, v)
# P: 212p = 212i + P9 (PIPE_OVERLAP, _KVS) + P10 (SOLO_PIPELINE, SPEC_LANES) + P11 (READ_YIELD): I's env with
# production's values of the five put back (a switch production does not set stays unset)
PIPE_BACK = ("STRATA_PIPE_OVERLAP", "STRATA_PIPE_OVERLAP_KVS", "STRATA_SPEC_LANES", "STRATA_SOLO_PIPELINE", "STRATA_READ_YIELD")
envs["P"] = dict(envs["I"])
envs["P"]["STRATA_CONV_TIER_DIR"] = f"{tier}/P"
pipe_back = {k: penv[k] for k in PIPE_BACK if k in penv}
envs["P"].update(pipe_back)
pipe_gone = [x for x in int_gone if x.split(" ")[0] not in pipe_back]
# PT: P + the sm_86 exact table (SM86_PERF unset; ATTN_DAILY=0 kept, so slots-stream stays) + the upstream stack;
# the bit-changing pins stay; STRATA_SPEC_ADAPT_SM86 stays unset
envs["PT"] = {k: v for k, v in envs["P"].items() if k not in ("STRATA_SM86_PERF", "STRATA_SPEC_ADAPT_SM86")}
envs["PT"].update(STACK)
envs["PTD"] = {**envs["PT"], "STRATA_ATTN_DAILY": "1", "STRATA_QSA_SLOTS_STREAM": "0"}
envs["PT4"] = {**envs["PT"], "STRATA_IQ4_W4A8": "1", "STRATA_SELECT_TWO_PASS": "1"}       # arm D's switches
BITPINS = ("STRATA_GDN_TC", "STRATA_GDN_CONVL2", "STRATA_PROMPT_ATTN_IMMA", "STRATA_MMQ_KQ", "STRATA_PF_W16_CACHE_MIB",
           "STRATA_AUTO_NOBORROW")
envs["PTX"] = {k: v for k, v in envs["PT"].items() if k not in BITPINS}
for a in ("PT", "PTD", "PT4", "PTX"):
    envs[a]["STRATA_CONV_TIER_DIR"] = f"{tier}/{a}"
notes.update(P="212p (212i + PIPE_OVERLAP(_KVS) SOLO_PIPELINE SPEC_LANES READ_YIELD): I's env + production's values of those five",
             PT="P + sm_86 exact table (SM86_PERF unset, ATTN_DAILY=0 kept) + upstream stack: the production candidate",
             PTD="PT with ATTN_DAILY=1 QSA_SLOTS_STREAM=0 (the table's own attention)",
             PT4="PT + IQ4_W4A8 + SELECT_TWO_PASS (arm D's switches)",
             PTX="PT without the pins of the bit-changing upstream defaults: NOT exact, INFO only, never a candidate")
notes.update(B="stock arm: production env (upstream names only) + parity pins", C="B + upstream stack + sm_86 table",
             D="C + IQ4_W4A8 + SELECT_TWO_PASS", E="C + SPLIT_MTP_BATCH=0 + PIPED_CONCURRENT=1", E2="E + PIPED_READ_PIECE=16384",
             F="C + PW_WITH_BATCH=1 + --pipeline-windows 2 (CUDA's auto is off)",
             G="C + --batch-mtp + PIPED_MTP=1 + BATCH_SELF_COMMIT=0 (PIPED_MTP refuses SELF_COMMIT)",
             H="212m-p2: production env (212m names) + parity pins + ATTN_DAILY=0", H2="H - QSA_SLOTS_STREAM + ATTN_DAILY=1",
             H3="H + IQ4_W4A8 + upstream stack (H won)", H3d="H2 + IQ4_W4A8 + upstream stack (H2 won)", A="production as is",
             I="212i (v0.2.12 + our ports): production env, PARK_ASYNC -> PARK_CAPTURE, minus the not-ported switches, "
               "+ SERVE_STOP_BOUND=1 + H's parity pins + ATTN_DAILY=0")
exe = {"A": prod["exe"]}
for a in ("B", "C", "D", "E", "E2", "F", "G"):
    exe[a] = f"{eng}/strata-212-stocktab"
for a in ("H", "H2", "H3", "H3d"):
    exe[a] = f"{eng}/strata-212m-p2"
exe["I"] = f"{eng}/strata-212i"
PIPE = ("P", "PT", "PTD", "PT4", "PTX")
for a in PIPE:
    exe[a] = f"{eng}/strata-212p"
tree = {a: (ours if a == "A" else intw if a == "I" else pipew if a in PIPE else port if a.startswith("H") else up) for a in exe}
bad, man = [], {"production": src, "production_sha256": hashlib.sha256(raw).hexdigest(), "generated": time.strftime("%F %T"),
                "trees": {t: subprocess.run(["git", "-C", t, "rev-parse", "--short=8", "HEAD"], capture_output=True,
                                            text=True).stdout.strip() for t in (up, port, ours, intw, pipew)},
                "dropped_upstream": up_gone, "dropped_port": port_gone, "dropped_int": int_gone, "dropped_pipe": pipe_gone,
                "exact": {"P": "vs I (short replies), bt vs A", "PT": "vs P (short replies), bt vs A and P",
                          "PTD": "vs P", "PT4": "vs P", "PTX": "NOT exact (INFO, DIFF expected), never a candidate"},
                "arms": {}}
for a in exe:
    c = copy.deepcopy(prod)
    c["exe"] = exe[a]; c["port"] = int(tport); c["log"] = f"{outdir}/engine-{a}.log"
    if a == "A":
        c["env"]["STRATA_CONV_TIER_DIR"] = f"{tier}/A"     # never production's tier dir (a fresh dir per start at run time)
    else:
        c["env"] = envs[a]
        c["args"] = list(c["args"]) + args_add.get(a, [])
        c["tool_call_recovery"] = False                    # upstream flipped the default to true (0.2.8)
        c["drafting"] = "mtp"                              # the MTP draft layer, as production (inert with slots anyway)
    for k in c["env"]:
        if not k.startswith("MALLOC_") and not has(tree[a], k):
            bad.append(f"{a}: {k} not in {tree[a]}")
    for f in c["args"]:
        if f.startswith("--") and not flag_ok(tree[a], f):
            bad.append(f"{a}: flag {f} not in {tree[a]}")
    json.dump(c, open(f"{cdir}/T212-{a}.json", "w"), indent=1)
    pe = prod["env"]
    diff = [f"{k}={v}" for k, v in c["env"].items() if pe.get(k) != v] + [f"-{k}" for k in pe if k not in c["env"]]
    man["arms"][a] = {"exe": exe[a], "note": notes[a], "env_vs_production": diff,
                      "args_added": [x for x in c["args"] if x not in prod["args"]]}
json.dump(man, open(f"{cdir}/T212.manifest.json", "w"), indent=1)
print(f"wrote {len(exe)} configs T212-*.json + T212.manifest.json from {os.path.basename(src)} "
      f"(sha256 {man['production_sha256'][:16]})")
for b in bad:
    print("NOT IN SOURCE:", b)
sys.exit(1 if bad else 0)
PYEOF
}
if [ "$MODE" = gen ]; then mkdir -p "$C"; gen_configs; exit $?; fi

# ---- helpers ------------------------------------------------------------------------------------------------------
log() { if [ "$MODE" = dry ]; then echo "[plan] $*"; else echo "[$(date +%T)] $*" | tee -a "$OUT/window.log"; fi; }
fsize() { stat -c %s "$1" 2>/dev/null || echo 0; }
left() { echo $(( DL_TS - $(date +%s) )); }
cfgget() { python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ''))" "$1" "$2"; }
cfgenv() { python3 -c "import json,sys; [print(f'{k}={v}') for k, v in json.load(open(sys.argv[1])).get('env', {}).items()]" "$1"; }
gpu_apps() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null; }
vram() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | tr '\n' ' '; }
free_gpus() { local i p e; for i in $(seq 45); do [ -z "$(gpu_apps)" ] && return 0; sleep 2; done
    for p in $(gpu_apps); do grep -qs '/strata\.service$' "/proc/$p/cgroup" && continue   # never production's engine
        e=$(readlink -f "/proc/$p/exe" 2>/dev/null)
        case "$e" in "$R"/Strata*|"$OUT"/bin/*) kill "$p" ;; esac; done
    for i in $(seq 30); do [ -z "$(gpu_apps)" ] && return 0; sleep 2; done; return 1; }
prod_loaded() { curl -s -m 3 "$URL/health" | grep -q '"loaded": *true'; }
tup() { curl -s -m 3 "$TURL/health" | grep -q '"loaded": *true'; }
scope_cg() { systemctl --user show -p ControlGroup --value "$BUNIT.scope" 2>/dev/null; }
stop_bench() { systemctl --user stop "$BUNIT.scope" 2>/dev/null; systemctl --user reset-failed "$BUNIT.scope" 2>/dev/null
    free_gpus || log "warning: GPUs still busy after stopping $BUNIT.scope"; }
port_busy() { ss -Hltn "sport = :$1" 2>/dev/null | grep -q .; }
# a binary pinned by a copy (sha256 checked twice: a build writing it now is refused); empty when it is missing
declare -A PINNED=()
pin_bin() { local k=$1 s=${BIN_SRC[$1]} h d
    [ -n "${PINNED[$k]:-}" ] && { echo "${PINNED[$k]}"; return 0; }
    [ -x "$s" ] || return 1
    h=$(sha256sum "$s" | cut -c1-16); d=$OUT/bin/$(basename "$s")-$h
    mkdir -p "$OUT/bin"; cp "$s" "$d" && chmod +x "$d" || return 1
    [ "$(sha256sum "$d" | cut -c1-16)" = "$h" ] || { rm -f "$d"; return 1; }
    PINNED[$k]=$d; echo "$d"; }
# the server env of an arm: production's drop-in STRATA_* (read now); a v0.2.12 server gets only the names it reads
# (in serve/*.py, its tests left out: 212i reads STRATA_PREFIX_DIAG in serve/prefix_diag.py)
SRVENV_ALL=$(systemctl --user show strata.service -p Environment --value 2>/dev/null | tr ' ' '\n' | grep '^STRATA_' | tr '\n' ' ')
srv_reads() { local f; for f in "$1"/serve/*.py; do case $f in */test_*) continue ;; esac
    grep -qF "\"$2\"" "$f" && return 0; done; return 1; }
srv_env() { local a=$1 kv k out=
    for kv in $SRVENV_ALL; do k=${kv%%=*}
        if [ "$a" = A ] || srv_reads "${ARM_SRV[$a]}" "$k"; then out="$out $kv"; fi; done
    echo "${out# }"; }
srv_env_dropped() { local a=$1 kv k out=
    for kv in $SRVENV_ALL; do k=${kv%%=*}; [ "$a" = A ] || srv_reads "${ARM_SRV[$a]}" "$k" || out="$out $k"; done
    echo "${out# }"; }

# ---- gates / drops ------------------------------------------------------------------------------------------------
declare -A SKIP_BIN=() SKIP_ARM=()
DROPS=""                                   # levers set to 0 in every v0.2.12 arm (a failed parity test)
drop_lever() { case " $DROPS " in *" $1 "*) ;; *) DROPS="$DROPS $1"; log "lever $1 dropped from tonight's v0.2.12 arms: $2" ;; esac; }
# mkrun <ARM> <LABEL>: the arm's config for one start: pinned exe, its own engine log, a fresh tier dir, dropped levers
mkrun() { local a=$1 lab=$2 exe f=$OUT/cfg/run-$2.json
    exe=${PINNED[${ARM_BIN[$a]}]:-}; [ -n "$exe" ] || return 1      # (pin_bin runs in the main shell before)
    mkdir -p "$OUT/cfg"; rm -rf "$TIER/$lab"; mkdir -p "$TIER/$lab"; rm -f "$OUT/engine-$lab.log"
    W212_PTIER=$PTIER python3 - "$C/T212-$a.json" "$f" "$exe" "$OUT/engine-$lab.log" "$TIER/$lab" "$a" $DROPS <<'PYEOF' || return 1
import json, os, sys
src, dst, exe, log, tier, arm = sys.argv[1:7]
c = json.load(open(src)); e = c["env"]; c["exe"] = exe; c["log"] = log
if "STRATA_CONV_TIER_DIR" in e:
    e["STRATA_CONV_TIER_DIR"] = tier
pt = os.environ.get("W212_PTIER", "").rstrip("/")     # production's SSD tier: never in a bench config
if not pt or not tier.startswith("/media/raid0/strata-kvspill-bench/") or \
        any(str(v).rstrip("/") == pt or str(v).startswith(pt + "/") for v in e.values()):
    sys.exit(f"mkrun: production's SSD tier dir {pt or '(unknown)'} in the run config, or no bench tier: refused")
if arm != "A":
    for k in sys.argv[7:]:
        if k in e and e[k] != "0":
            e[k] = "0"
        elif k == "STRATA_ATTN_DAILY" and e.get("STRATA_SM86_PERF") != "0":
            e[k] = "0"                                  # the sm_86 table would set it
json.dump(c, open(dst, "w"), indent=1)
PYEOF
    echo "$f"; }

# ---- monitor: one check every 5 s while a step or a boot runs (MON_WHY: why it must stop, empty while fine) --------
WATCH=""; VRAM_ON=0; MON_HANG=$HANG_S; MON_LBL=x; ENGSTART_BASE=-1; ENGLOG=""
declare -A WOFF=(); declare -A VMIN=(); RSS_MAX=0; LOWV=0; LAST_PROG=0; PROG_SIG=""
mon_reset() { local f; LOWV=0; LAST_PROG=$(date +%s); PROG_SIG=""; WOFF=()
    for f in $WATCH; do WOFF[$f]=$(fsize "$f"); done; }
arm_mon_reset() { VMIN=(); RSS_MAX=0; }
mon_check() {
    MON_WHY=""; local now p q i fr u us=0 low=0 cg r f n sig="" ln
    now=$(date +%s)
    if port_busy $PPORT; then MON_WHY="port$PPORT"; return; fi              # production's port must stay closed
    p=$(awk '/^some/ {split($2, a, "="); print a[2]}' "$PRESS" 2>/dev/null)
    if awk -v p="${p:-0}" -v m="$PRESS_MAX" 'BEGIN { exit !(p >= m) }'; then MON_WHY="pressure:${p}%"; return; fi
    cg=$(scope_cg)
    if [ -n "$cg" ] && [ -r "/sys/fs/cgroup$cg/memory.stat" ]; then
        r=$(awk '$1 == "anon" || $1 == "shmem" { s += $2 } END { printf "%d", s / 1073741824 }' "/sys/fs/cgroup$cg/memory.stat")
        [ "$r" -gt "$RSS_MAX" ] && RSS_MAX=$r
        if [ "$r" -gt "$RSS_MAX_GIB" ]; then MON_WHY="rss:${r}GiB"; return; fi
    fi
    q=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    [ -n "$q" ] && echo "$(date +%s),$(echo "$q" | tr '\n' ';')" >> "$OUT/vram-$MON_LBL.csv"
    while IFS=, read -r i fr u; do
        [ -z "$i" ] && continue
        case $fr in ''|*[!0-9]*) continue ;; esac; case $u in ''|*[!0-9]*) u=0 ;; esac   # ("[N/A]" / an error line: no fatal $(( )))
        us=$(( us + u ))
        if [ "$VRAM_ON" = 1 ]; then
            { [ -z "${VMIN[$i]:-}" ] || [ "$fr" -lt "${VMIN[$i]}" ]; } && VMIN[$i]=$fr
            [ "$fr" -lt "$VRAM_MIN" ] && low=1
        fi; done <<< "$q"
    if [ "$low" = 1 ]; then LOWV=$(( LOWV + 1 )); else LOWV=0; fi
    if [ "$LOWV" -ge 2 ]; then MON_WHY="vram:<${VRAM_MIN}MiB($(echo "$q" | tr '\n' ' '))"; return; fi
    for f in $WATCH; do
        [ -f "$f" ] || continue
        ln=$(tail -c +$(( ${WOFF[$f]:-0} + 1 )) "$f" 2>/dev/null | grep -a -m1 -E "$CRASH_RE")
        if [ -n "$ln" ]; then MON_WHY="crash:$(echo "$ln" | cut -c1-160)"; return; fi
        sig="$sig$(fsize "$f")."; done
    if [ -n "$ENGLOG" ] && [ "$ENGSTART_BASE" -ge 0 ]; then
        n=$(grep -a -c 'engine started' "$ENGLOG" 2>/dev/null)
        if [ "${n:-0}" -gt "$ENGSTART_BASE" ]; then MON_WHY="crash:the engine restarted (${n} starts)"; return; fi
    fi
    if [ "$sig" != "$PROG_SIG" ] || [ "$us" -gt 20 ]; then LAST_PROG=$now; PROG_SIG=$sig; fi
    if [ $(( now - LAST_PROG )) -ge "$MON_HANG" ]; then MON_WHY="hang:no log growth and GPUs idle for ${MON_HANG}s"; return; fi
}
# ABORT_ALL: stop at once and restore production (memory, port 8090, two crashes in a row)
ABORT_ALL=0; CONSEC_CRASH=0; INFO_STEP=0     # INFO_STEP=1: phase 4's INFO engine tests (not counted as crashes in a row)
judge_why() {    # after a step/boot that stopped for MON_WHY
    case $1 in
        rss:*|pressure:*|port$PPORT) ABORT_ALL=1; log "ABORT: $1 - stopping at once, production is restored" ;;
        crash:*|hang:*) [ "$INFO_STEP" = 1 ] || CONSEC_CRASH=$(( CONSEC_CRASH + 1 )); log "crash/hang ($1); nvidia-smi: $(nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null | tr '\n' ';')"
            [ "$CONSEC_CRASH" -ge 2 ] && { ABORT_ALL=1; log "ABORT: two crashes in a row - production is restored"; } ;;
    esac; }
STEP_PID=; STEP_RC=0; STEP_WHY=""
kill_step() { [ -n "$STEP_PID" ] || return 0; kill -INT -- -"$STEP_PID" 2>/dev/null || kill -INT "$STEP_PID" 2>/dev/null
    local i; for i in $(seq 20); do kill -0 "$STEP_PID" 2>/dev/null || return 0; sleep 1; done
    kill -KILL -- -"$STEP_PID" 2>/dev/null; }
# run_step <NAME> <CAP s> <cmd...>: under timeout (its own process group), cut to DEADLINE, output $OUT/<NAME>.txt;
# the monitor runs meanwhile (WATCH / VRAM_ON set by the caller).  STEP_RC (99 = not run), STEP_WHY
run_step() { local name=$1 cap=$2 l; shift 2; STEP_WHY=""; STEP_RC=0; l=$(left)
    if [ "$ABORT_ALL" = 1 ]; then STEP_RC=99; STEP_WHY=aborted; return 99; fi
    if [ "$l" -lt 60 ]; then STEP_RC=99; STEP_WHY=deadline; log "step $name: SKIP (DEADLINE)"; return 99; fi
    [ "$cap" -gt "$l" ] && cap=$l
    log "step $name: start (cap ${cap} s); VRAM free $(vram)"
    timeout -s INT -k 60 "$cap" "$@" > "$OUT/$name.txt" 2>&1 < /dev/null 9>&- &
    STEP_PID=$!; echo "$STEP_PID" > "$OUT/step.pgid"; mon_reset
    while kill -0 "$STEP_PID" 2>/dev/null; do
        sleep 5; mon_check; [ -n "$MON_WHY" ] && { STEP_WHY=$MON_WHY; log "step $name: stopped: $MON_WHY FAIL"; kill_step; break; }
    done
    wait "$STEP_PID"; STEP_RC=$?; STEP_PID=; : > "$OUT/step.pgid"
    [ "$STEP_RC" = 124 ] && [ -z "$STEP_WHY" ] && STEP_WHY=timeout
    [ "$STEP_RC" = 0 ] && [ -z "$STEP_WHY" ] && CONSEC_CRASH=0            # ("two crashes in a row")
    log "step $name: exit $STEP_RC${STEP_WHY:+ ($STEP_WHY)}"
    [ -n "$STEP_WHY" ] && judge_why "$STEP_WHY"
    return "$STEP_RC"; }

# ---- bench server ----------------------------------------------------------------------------------------------------
# srv_up <LABEL> <CFG> <ARM>: the arm's server in $BUNIT.scope on 127.0.0.1:$TPORT, 0 once loaded
srv_up() { local lab=$1 cfg=$2 a=$3 sdir=${ARM_SRV[$3]} senv t0 env_
    stop_bench
    port_busy "$TPORT" && { log "$lab: port $TPORT is in use: not started FAIL"; return 1; }
    senv=$(srv_env "$a")
    ( exec 9>&-; cd "$sdir" && exec env "${GENV[@]:2}" $senv "${SCOPE[@]}" "$PY" "$sdir/serve/server.py" --engine strata --config "$cfg" \
        --port "$TPORT" --host 127.0.0.1 ) > "$OUT/server-$lab.out" 2>&1 < /dev/null &
    log "$lab: server $(basename "$sdir")@$(git -C "$sdir" rev-parse --short=8 HEAD) env [$senv]$( [ -n "$(srv_env_dropped "$a")" ] && echo " (not in this server: $(srv_env_dropped "$a"))"), exe $(basename "$(cfgget "$cfg" exe)")"
    WATCH="$OUT/engine-$lab.log $OUT/server-$lab.out"; ENGLOG=$OUT/engine-$lab.log; ENGSTART_BASE=-1
    VRAM_ON=0; MON_HANG=$BOOT_HANG; MON_LBL=$lab; mon_reset; t0=$(date +%s); sleep 5
    until tup; do
        systemctl --user is-active -q "$BUNIT.scope" || { log "$lab: the server died during its start FAIL"; tail -5 "$OUT/server-$lab.out" | sed 's/^/    /' | tee -a "$OUT/window.log"; STEP_WHY="crash:boot"; judge_why "$STEP_WHY"; return 1; }
        [ $(( $(date +%s) - t0 )) -gt "$BOOT_CAP" ] && { log "$lab: start timeout ($BOOT_CAP s) FAIL"; STEP_WHY="hang:boot"; judge_why "$STEP_WHY"; return 1; }
        [ "$(left)" -lt 30 ] && { log "$lab: DEADLINE during the start"; STEP_WHY=deadline; return 1; }
        mon_check; [ -n "$MON_WHY" ] && { log "$lab: start stopped: $MON_WHY FAIL"; STEP_WHY=$MON_WHY; judge_why "$MON_WHY"; return 1; }
        sleep 5; done
    BOOT_S=$(( $(date +%s) - t0 )); ENGSTART_BASE=$(grep -a -c 'engine started' "$ENGLOG" 2>/dev/null)
    VRAM_ON=1; MON_HANG=$HANG_S
    log "$lab: ready after $BOOT_S s; VRAM free $(vram)"
    mon_check; [ -n "$MON_WHY" ] && { log "$lab: after the start: $MON_WHY FAIL"; STEP_WHY=$MON_WHY; judge_why "$MON_WHY"; return 1; }
    curl -s -m 10 "$TURL/v1/status" > "$OUT/status-$lab.json"; return 0; }
grep_all() {   # <file> <pattern;;pattern...>: prints the patterns NOT found
    local f=$1 p miss=; local IFS=$'\n'
    for p in $(echo "$2" | sed 's/;;/\n/g'); do [ -n "$p" ] && ! grep -a -q -E -- "$p" "$f" 2>/dev/null && miss="$miss [$p]"; done
    echo "${miss# }"; }

# ---- one arm --------------------------------------------------------------------------------------------------------
ST=""   # the status file of the arm running now
stw() { echo "$1=$2" >> "$ST"; }
arm_run() { local a=$1 lab=$2 cfg el miss fp ref r cmp ok=1
    ST=$OUT/arm-$lab.st; : > "$ST"; stw arm "$a"; stw label "$lab"; arm_mon_reset; pin_bin "${ARM_BIN[$a]}" > /dev/null
    stw bin "${ARM_BIN[$a]}"; stw server "$(basename "${ARM_SRV[$a]}")@$(git -C "${ARM_SRV[$a]}" rev-parse --short=8 HEAD)"
    if ! cfg=$(mkrun "$a" "$lab"); then stw status skip; stw why "binary ${BIN_SRC[${ARM_BIN[$a]}]} missing"; log "arm $lab: SKIP (binary ${BIN_SRC[${ARM_BIN[$a]}]} missing)"; return 0; fi
    stw sha "$(basename "$(cfgget "$cfg" exe)" | sed 's/.*-//')"
    log "arm $lab: env vs production: $(python3 -c "
import json,sys; p=json.load(open(sys.argv[1]))['env']; c=json.load(open(sys.argv[2]))
e=c['env']; d=[f'{k}={e[k]}' for k in e if p.get(k)!=e[k] and k!='STRATA_CONV_TIER_DIR']+[f'-{k}' for k in p if k not in e]
x=[a for a in c['args'] if a not in json.load(open(sys.argv[3]))['args']]
print(' '.join(d) or 'none', '| args +', ' '.join(x) or '-')" "$C/production.json" "$cfg" "$C/production.json")"
    el=$OUT/engine-$lab.log
    if ! srv_up "$lab" "$cfg" "$a"; then stw status "$( [ "$STEP_WHY" = deadline ] && echo skip || echo fail)"; stw why "start: $STEP_WHY"; arm_end "$lab"; return 1; fi
    stw boot_s "$BOOT_S"; stw vram_boot "$(vram)"
    local B="$PY -I -B $L/bench_concurrency.py --url $TURL --temperature 0"
    run_step "$lab-warm" 300 $B --label "w212-$lab-warm" --levels 1 --rounds 1 --max-tok 16 --prompt-tok 100 --out "$OUT/warm.jsonl"
    run_step "$lab-probe" 300 $B --label "w212-$lab-probe" --levels 4 --rounds 1 --max-tok 64 --prompt-tok 100 --warmup 0 --out "$OUT/warm.jsonl"
    [ -n "$STEP_WHY" ] && { stw status fail; stw why "probe: $STEP_WHY"; arm_end "$lab"; return 1; }
    # the arm's path: the boot / probe lines it needs, the lines that say it is off
    sleep 2; miss=$(grep_all "$el" "$(wants "${WANT_BOOT[$a]:-}")"); fp=""
    [ -n "${FAILPAT[$a]:-}" ] && fp=$(grep -a -m1 -E -- "${FAILPAT[$a]}" "$el" | cut -c1-200)
    if [ -n "$miss" ] || [ -n "$fp" ]; then
        log "arm $lab: NOT ENGAGED FAIL: ${miss:+missing $miss}${fp:+ / says: $fp} - the arm stops"
        stw status notengaged; stw why "${miss:+missing $miss}${fp:+ says: $fp}"; arm_end "$lab"; return 1; fi
    stw engaged_boot yes; log "arm $lab: INFO lines present: $(echo "${WANT_BOOT[$a]:-none}" | sed 's/;;/ + /g' | tr -d '\\')"
    run_step "$lab-short" 900 $B --label "w212-$lab-short" --levels 1,2,4,8,16 --rounds 2 --max-tok 512 --prompt-tok 100 --out "$OUT/sweep.jsonl"
    [ -n "$STEP_WHY" ] && { stw status fail; stw why "short: $STEP_WHY"; arm_end "$lab"; return 1; }
    ref=w212-B-short   # I: against H (same kernels), B when H has no short replies tonight
    [ "$a" = I ] && grep -qF '"label":"w212-H-short"' "$OUT/sweep.requests.jsonl" 2>/dev/null && ref=w212-H-short
    # the pipe arms: P against I (this run's, else last night's as ref-I-short); PT PTD PT4 against P (its first run),
    # I when P has none or P's were not I's; PTX against PT (INFO: DIFF expected).  The compared file: this run's replies + the reference
    local rf=$OUT/sweep.requests.jsonl
    case $a in
        P|PT|PTD|PT4|PTX)
            rf=$OUT/cmp.requests.jsonl; cat "$OUT/ref.requests.jsonl" "$OUT/sweep.requests.jsonl" > "$rf" 2>/dev/null
            has_lab() { grep -qF "\"label\":\"$1\"" "$rf" 2>/dev/null || grep -qF "\"label\": \"$1\"" "$rf" 2>/dev/null; }
            ref=""; has_lab w212-I-short && ref=w212-I-short || { has_lab ref-I-short && ref=ref-I-short; }
            case $a in PT|PTD|PT4) has_lab w212-P-short && grep -qx 'ident=same' "$OUT/arm-P.st" 2>/dev/null && ref=w212-P-short ;;
                       PTX) ref=""; has_lab w212-PT-short && ref=w212-PT-short ;; esac
            [ -n "$ref" ] || ref=none ;;
    esac
    local rn=${ref#w212-}; rn=${rn%-short}
    if [ "$ref" = none ]; then stw ident "-"; log "arm $lab: short replies: no reference reply set (INFO '-')"
    elif [ "$a" = PTX ]; then
        "$PY" -I "$L/compare_replies.py" "$rf" "$ref" "w212-$lab-short" --show 3 > "$OUT/$lab-ident.txt" 2>&1; r=$?
        cmp=$( { [ $r = 0 ] && echo same; } || { [ $r = 1 ] && echo DIFF; } || echo "-"); stw ident "DIFF-expected:$cmp"
        log "arm $lab: short replies vs $rn: $cmp (non-exact arm: DIFF expected, INFO, never gated) $(tail -1 "$OUT/$lab-ident.txt" | cut -c1-160)"
    elif [ "$a" != A ] && [ "$a" != B ]; then
        "$PY" -I "$L/compare_replies.py" "$rf" "$ref" "w212-$lab-short" --show 3 > "$OUT/$lab-ident.txt" 2>&1; r=$?
        cmp=$( { [ $r = 0 ] && echo same; } || { [ $r = 1 ] && echo DIFF; } || echo "-"); stw ident "$cmp"
        log "arm $lab: short replies vs $rn: $cmp $(tail -1 "$OUT/$lab-ident.txt" | cut -c1-160)"
        if [ "$cmp" = DIFF ] && case "$EXACT_ARMS" in *" $a "*) true ;; *) false ;; esac; then
            if [ "$HTTP_GATE" = 1 ]; then log "arm $lab: an exact arm with other replies than $rn FAIL - the arm stops (log kept)"
                stw status identfail; arm_end "$lab"; return 1
            else log "arm $lab: exact arm with other replies than $rn (HTTP_GATE=0: goes on) FAIL"; fi; fi
    elif [ "$a" = B ]; then
        "$PY" -I "$L/compare_replies.py" "$OUT/sweep.requests.jsonl" w212-A-short "w212-$lab-short" --show 3 > "$OUT/$lab-ident.txt" 2>&1; r=$?
        cmp=$( { [ $r = 0 ] && echo same; } || { [ $r = 1 ] && echo DIFF; } || echo "-"); stw ident "vsA:$cmp"
        log "arm $lab: short replies vs A (other server code: INFO): $cmp"
    fi
    run_step "$lab-long" 600 $B --label "w212-$lab-long" --levels 16 --rounds 2 --max-tok 512 --warmup 0 --out "$OUT/sweep.jsonl"
    [ -n "$STEP_WHY" ] && { stw status fail; stw why "long: $STEP_WHY"; arm_end "$lab"; return 1; }
    run_step "$lab-mt" 900 "$PY" -I -B "$L/bench_multiturn.py" --url "$TURL" --users 16 --turns 3 --doc 8000 --temperature 0 \
        --label "w212-$lab-mt" --out "$OUT/multiturn.jsonl"
    [ -n "$STEP_WHY" ] && { stw status fail; stw why "multiturn: $STEP_WHY"; arm_end "$lab"; return 1; }
    run_step "$lab-join" 600 "$PY" -I -B "$UP/tools/piped_read_join_bench.py" --port "$TPORT" --scenarios cold32k --out "$OUT/join-$lab.json"
    [ -n "$STEP_WHY" ] && { stw status fail; stw why "join: $STEP_WHY"; arm_end "$lab"; return 1; }
    if [ "$a" = H ] || { [ "$a" = PT ] && [ "$lab" = PT ]; }; then   # HTTP robustness on the arm's server (H: stock serve/ of
        local rt=$UP; [ "$a" = PT ] && rt=$PPT                       # v0.2.12; PT: 212p's = 212i's, lanes + overlap on)
        run_step "$lab-early_close" 300 "$PY" -I -B "$rt/tools/early_close_test.py" "$TURL"
        log "    [early_close $lab] ==> $([ "$STEP_RC" = 0 ] && echo PASS || echo "FAIL (rc $STEP_RC)")"
        run_step "$lab-stop_resend" 600 "$PY" -I -B "$rt/tools/stop_resend_test.py" "$TURL" 40 1
        log "    [stop_resend $lab] ==> $([ "$STEP_RC" = 0 ] && echo PASS || echo "FAIL (rc $STEP_RC)")"
        # a crash / hang here is a finding of its own field: the benches are in, status stays ok (the summary marks it;
        # the r2 repeat still runs - its benches only); the vision step is not tried on a crashed server
        [ -n "$STEP_WHY" ] && [ "$STEP_WHY" != timeout ] && { stw robust "fail: $STEP_WHY"; stw status ok
            log "arm $lab: robustness FAIL ($STEP_WHY): recorded as robust=, the arm's benches stay ok"; arm_end "$lab"; return 0; }
    fi
    if [ "$a" = I ] && [ "$lab" = I ]; then   # INFO: image and text slots side by side on I's server (not in r2 repeats)
        if [ -n "${I_VISION_SKIP:-}" ]; then log "    [vision I] ==> SKIP ($I_VISION_SKIP)"
        else INFO_STEP=1   # (its crash does not count for "two crashes in a row"; I's numbers are in already)
            run_step "$lab-vision" 900 "$PY" -I -B "$IWT/tools/vision_batch_check.py" --url "$TURL/v1" --out "$OUT/vision-$lab.json"
            log "    [vision I] ==> rc $STEP_RC${STEP_WHY:+ ($STEP_WHY)} $(grep -a -m1 -E '^  ==> ' "$OUT/$lab-vision.txt" | sed 's/^ *//') (0 GO, 2 NO-GO, 3 INCONCLUSIVE) INFO"
            stw vision "rc $STEP_RC${STEP_WHY:+ $STEP_WHY}"; INFO_STEP=0
            [ -n "$STEP_WHY" ] && { stw status ok; stw why "vision (INFO, after the benches): $STEP_WHY"; arm_end "$lab"; return 0; }
        fi
    fi
    if [ "$a" = PT ] && [ "$lab" = PT ]; then   # INFO: image and text slots side by side, lanes + overlap on (not in r2)
        if [ -n "${PT_VISION_SKIP:-}" ]; then log "    [vision PT] ==> SKIP ($PT_VISION_SKIP)"
        else INFO_STEP=1
            run_step "$lab-vision" 900 "$PY" -I -B "$PPT/tools/vision_batch_check.py" --url "$TURL/v1" --out "$OUT/vision-$lab.json"
            log "    [vision PT] ==> rc $STEP_RC${STEP_WHY:+ ($STEP_WHY)} $(grep -a -m1 -E '^  ==> ' "$OUT/$lab-vision.txt" | sed 's/^ *//') (0 GO, 2 NO-GO, 3 INCONCLUSIVE) INFO"
            stw vision "rc $STEP_RC${STEP_WHY:+ $STEP_WHY}"; INFO_STEP=0
            [ -n "$STEP_WHY" ] && { stw status ok; stw why "vision (INFO, after the benches): $STEP_WHY"; arm_end "$lab"; return 0; }
        fi
    fi
    stw status ok; arm_end "$lab"; CONSEC_CRASH=0; return 0; }
arm_end() { local lab=$1 a cg miss el=$OUT/engine-$1.log
    a=$(sed -n 's/^arm=//p' "$ST" | head -1)
    curl -s -m 10 "$TURL/metrics" > "$OUT/metrics-$lab.txt" 2>/dev/null
    cg=$(scope_cg); [ -n "$cg" ] && [ -r "/sys/fs/cgroup$cg/memory.peak" ] && stw mem_peak_gib "$(awk '{printf "%.1f", $1 / 1073741824}' "/sys/fs/cgroup$cg/memory.peak")"
    stw rss_max_gib "$RSS_MAX"; stw vram_min "$(for i in $(echo "${!VMIN[@]}" | tr ' ' '\n' | sort -n); do printf '%s:%s ' "$i" "${VMIN[$i]}"; done)"
    if [ -n "${WANT_END[$a]:-}" ]; then miss=$(grep_all "$el" "$(wants "${WANT_END[$a]}")"); stw engaged_end "${miss:-all seen}"
        log "arm $lab: end INFO lines: ${miss:+not seen $miss}${miss:-all seen ($(echo "${WANT_END[$a]}" | sed 's/;;/ + /g' | tr -d '\\'))}"; fi
    stw warn "$(grep -a -m3 -E 'is ignored|WARNING' "$el" 2>/dev/null | cut -c1-140 | tr '\n' '|')"
    stw engine_starts "$(grep -a -c 'engine started' "$el" 2>/dev/null)"
    sleep 1; stop_bench; ENGLOG=""; ENGSTART_BASE=-1; VRAM_ON=0; WATCH=""; }

# ---- identity (engine-level, no server) -------------------------------------------------------------------------------
bt_run() {   # <A|B|H>: tools/batch_test.py (Strata-212's) greedy, 16 slots, a 40K prompt, solo + batch dumps
    local a=$1 cfg exe; pin_bin "${ARM_BIN[$a]}" > /dev/null; cfg=$(mkrun "$a" "bt-$a") || { log "identity bt-$a: SKIP (binary missing)"; echo skip > "$OUT/rc_bt-$a"; return 1; }
    exe=$(cfgget "$cfg" exe); WATCH="$OUT/engine-bt-$a.log"; ENGLOG=""; VRAM_ON=0; MON_HANG=$BOOT_HANG; MON_LBL=bt-$a
    stop_bench; VRAM_ON=1
    run_step "bt-$a" 1200 "${SCOPE[@]}" env "${GENV[@]}" $(cfgenv "$cfg") BATCH_TEST_LOG="$OUT/engine-bt-$a.log" \
        "$PY" -I -B "$UP/tools/batch_test.py" --exe "$exe" --config "$cfg" --batch 16 --n 16 --max-new 128 \
        --long-tokens 40000 --extra "$BT_EXTRA" --dump "$OUT/bt-$a.json"
    echo "$STEP_RC" > "$OUT/rc_bt-$a"; stop_bench
    log "    [bt-$a] ==> $(grep -c IDENTICAL "$OUT/bt-$a.txt") of 16 slots = their solo run; $(grep -m1 '^batch:' "$OUT/bt-$a.txt" | cut -c1-160)"; }
bt_same() {  # <base> <arm>: solo + batch tokens identical -> 0, differ -> 1, missing -> 2
    python3 - "$OUT/bt-$1.json" "$OUT/bt-$2.json" <<'PYEOF'
import json, sys
try:
    a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
except (OSError, ValueError):
    print("missing dump"); sys.exit(2)
out, ok = [], True
for k in ("solo", "batch"):
    x, y = a.get(k) or [], b.get(k) or []
    bad = [i for i in range(max(len(x), len(y))) if i >= len(x) or i >= len(y) or x[i] != y[i]]
    first = {i: next((j for j in range(min(len(x[i]), len(y[i]))) if x[i][j] != y[i][j]), min(len(x[i]), len(y[i])))
             for i in bad if i < len(x) and i < len(y)}
    out.append(f"{k} {max(len(x), len(y)) - len(bad)}/{max(len(x), len(y))} identical" + (f" (first diff {first})" if bad else ""))
    ok = ok and not bad
print("; ".join(out))
sys.exit(0 if ok and (a.get("batch") or []) else 1)
PYEOF
}
eng_test() {  # <name> <cap> <cfg> <cmd...>: an upstream engine test on 212m-p2 in the bench scope (STRATA_REPO = 212m,
    local name=$1 cap=$2 cfg=$3; shift 3   # or ET_REPO: 212i's tests)
    WATCH="$OUT/engine-$name.log"; ENGLOG=""; VRAM_ON=1; MON_HANG=$BOOT_HANG; MON_LBL=$name; stop_bench
    run_step "$name" "$cap" "${SCOPE[@]}" env "${GENV[@]}" $(cfgenv "$cfg") STRATA_REPO="${ET_REPO:-$PWT}" STRATA_EXE="$(cfgget "$cfg" exe)" \
        SMOKE_LOG="$OUT/engine-$name.log" BATCH_TEST_LOG="$OUT/engine-$name.log" "$@"
    local rc=$STEP_RC; stop_bench; return "$rc"; }

# ---- ctest parsing ------------------------------------------------------------------------------------------------------
ctest_status() {  # <ctest output>: "name status" per test
    sed -nE 's/.*Test +#[0-9]+: ([A-Za-z0-9_.-]+) \.*[ *]*([A-Za-z][A-Za-z: ()]*[A-Za-z)]) +[0-9.]+ sec.*/\1 \2/p' "$1" | sed 's/ (.*//; s/  */ /g'; }

# ---- summary (no GPU) ---------------------------------------------------------------------------------------------------
summary() { "$PY" -I - "$OUT" "$L/compare_replies.py" <<'PYEOF'
import json, os, re, subprocess, sys
from pathlib import Path
out, cmp = Path(sys.argv[1]), sys.argv[2]
def jl(p):
    try:
        return [json.loads(x) for x in open(p) if x.strip()]
    except (OSError, ValueError):
        return []
sw = {}
for r in jl(out / "sweep.jsonl"):
    sw[(r.get("label"), r.get("users"))] = r
mt = {r.get("label"): r for r in jl(out / "multiturn.jsonl")}
def st(lab):
    d = {}
    try:
        for ln in open(out / f"arm-{lab}.st"):
            if "=" in ln:
                k, v = ln.rstrip("\n").split("=", 1)
                d.setdefault(k, v)
    except OSError:
        pass
    return d
labs = sorted({p.name[4:-3] for p in out.glob("arm-*.st")},
              key=lambda s: (os.path.getmtime(out / f"arm-{s}.st")))
def f(x, d=1):
    if isinstance(x, list):   # bench_join reports one value per joiner
        return "-" if not x else "|".join(f(v, d) for v in x)
    return "-" if x is None else f"{x:.{d}f}"
def g(lab, kind, users, key):
    r = sw.get((f"w212-{lab}-{kind}", users))
    return None if r is None else r.get(key)
def join(lab):
    try:
        return json.load(open(out / f"join-{lab}.json")).get("cold32k") or {}
    except (OSError, ValueError):
        return {}
def prefix(lab):
    try:
        t = open(out / f"metrics-{lab}.txt").read()
    except OSError:
        return None
    q = re.search(r"^vllm:prefix_cache_queries_total\S*\s+([\d.e+]+)", t, re.M)
    h = re.search(r"^vllm:prefix_cache_hits_total\S*\s+([\d.e+]+)", t, re.M)
    return 100 * float(h.group(1)) / float(q.group(1)) if q and h and float(q.group(1)) > 0 else None
print("# window 212: one fresh server per arm on 127.0.0.1:8091, greedy; agg = aggregate output tok/s, TTFT in s, "
      "dec = per-user decode p50 tok/s at 1/2/4 users, mt = bench_multiturn 16x3 (wall s / follow-up TTFT p50 s), "
      "join = cold 32K joiner beside 4 streams (joiner TTFT / worst stream gap s), pfx = prefix reuse %, "
      "VRAM = lowest free MiB per card during the arm, mem = memory.peak GiB of the scope")
hdr = (f"{'arm':6s} {'status':10s} {'boot':>4s} {'agg16s':>7s} {'vsA':>6s} {'vsB':>6s} {'TTFT s p50/p95/max':>18s} {'agg16l':>7s} "
       f"{'vsA':>6s} {'TTFT l p50/p95':>14s} {'dec 1/2/4':>16s} {'mt wall/fu':>11s} {'join ttft/gap':>13s} {'pfx':>5s} "
       f"{'mem':>5s} {'ident':>8s}  VRAM min")
print(hdr); print("-" * len(hdr))
base = {}
for lab in labs:
    s = st(lab)
    a16, l16 = g(lab, "short", 16, "agg_out_tps"), g(lab, "long", 16, "agg_out_tps")
    base.setdefault(s.get("arm"), (a16, l16))
for lab in labs:
    s = st(lab)
    a16, l16 = g(lab, "short", 16, "agg_out_tps"), g(lab, "long", 16, "agg_out_tps")
    def rel(x, ref):
        return "-" if x is None or not ref else f"{100 * (x / ref - 1):+.1f}"
    A, B = base.get("A", (None, None)), base.get("B", (None, None))
    ts = "/".join(f(g(lab, "short", 16, k), 2) for k in ("ttft_p50", "ttft_p95", "ttft_max"))
    tl = "/".join(f(g(lab, "long", 16, k), 2) for k in ("ttft_p50", "ttft_p95"))
    dec = "/".join(f(g(lab, "short", u, "dec_tps_p50"), 1) for u in (1, 2, 4))
    m = mt.get(f"w212-{lab}-mt") or {}
    j = join(lab)
    print(f"{lab:6s} {s.get('status', '?'):10s} {s.get('boot_s', '-'):>4s} {f(a16):>7s} {rel(a16, A[0]):>6s} {rel(a16, B[0]):>6s} "
          f"{ts:>18s} {f(l16):>7s} {rel(l16, A[1]):>6s} {tl:>14s} {dec:>16s} "
          f"{f(m.get('wall_s')) + '/' + f(m.get('follow_ttft_p50'), 2):>11s} "
          f"{f(j.get('joiner_ttft_s'), 2) + '/' + f(j.get('worst_max_gap_s'), 2):>13s} {f(prefix(lab)):>5s} "
          f"{s.get('mem_peak_gib', '-'):>5s} {s.get('ident', '-'):>8s}  {s.get('vram_min', '-')}")
print()
for lab in labs:
    s = st(lab)
    extra = [f"{k}={s[k]}" for k in ("why", "robust", "engaged_end", "warn") if s.get(k)]
    if extra:
        print(f"  {lab}: " + "; ".join(x[:300] for x in extra))
# identity and tests (window.log verdict lines)
try:
    wl = open(out / "window.log").read().splitlines()
except OSError:
    wl = []
print("\n# identity / tests / gates (window.log)")
for ln in wl:
    if re.search(r"==>|lever .* dropped|SKIP|NOT ENGAGED|ABORT|IDENT|host tests|gpu ctest|\[prod\]", ln) and "step " not in ln[:20]:
        print("  " + re.sub(r"^\[[0-9:]+\] *", "", ln)[:300])
# what the night decides (plan section 4), from the table above
def v(lab, kind, users, key):
    return g(lab, kind, users, key)
print("\n# decisions (INFO; read the table)")
def pair(x, y, what, key, kind="short", users=16, lower=False):
    a, b = v(x, kind, users, key), v(y, kind, users, key)
    if not a or b is None:
        print(f"  {what}: {x} vs {y}: not measured"); return
    print(f"  {what}: {x} {a:.2f} vs {y} {b:.2f} ({100 * (b / a - 1):+.1f} %{', lower is better' if lower else ''})")
OLD = ("B", "C", "D", "E", "E2", "F", "G", "H", "H2", "H3", "I")
PIPE = ("P", "PT", "PTD", "PT4", "PTX")
if any(st(l).get("arm") in OLD for l in labs):
    pair("H", "H2", "F14b DAILY vs slots(+stream), short agg 16", "agg_out_tps")
    for u in (1, 2, 4):
        pair("A", "F", f"F16 PW_WITH_BATCH vs SOLO_PIPELINE/lanes, dec p50 at {u} user(s)", "dec_tps_p50", users=u)
        pair("A", "G", f"F16 PIPED_MTP vs lanes, dec p50 at {u} user(s)", "dec_tps_p50", users=u)
    for x in ("E", "E2"):
        ja, jx = join("A"), join(x)
        print(f"  F15 join pause A vs {x}: worst stream gap {f(ja.get('worst_max_gap_s'), 2)} vs {f(jx.get('worst_max_gap_s'), 2)} s, "
              f"joiner TTFT {f(ja.get('joiner_ttft_s'), 2)} vs {f(jx.get('joiner_ttft_s'), 2)} s")
    pair("B", "C", "sm_86 table + stack, short agg 16", "agg_out_tps")
    pair("C", "D", "IQ4_W4A8 + TWO_PASS, long agg 16", "agg_out_tps", kind="long")
    pair("A", "H3", "H3 (candidate) vs production, short agg 16", "agg_out_tps")
    pair("A", "I", "I (212i integration, the candidate) vs production, short agg 16", "agg_out_tps")
    pair("A", "I", "I vs production, long agg 16", "agg_out_tps", kind="long")
    pair("H", "I", "I vs H (the ported cache features on the same kernels), short agg 16", "agg_out_tps")
# the 212p arms: single runs, and the mean over the arm's runs (r1 + r2) where an r2 exists
def runs(arm):
    return [l for l in labs if st(l).get("arm") == arm and st(l).get("status") == "ok"]
def mean(arm, kind, users, key):
    xs = [g(l, kind, users, key) for l in runs(arm)]
    xs = [x for x in xs if isinstance(x, (int, float))]
    return (sum(xs) / len(xs), len(xs)) if xs else (None, 0)
def mpair(x, y, what, key, kind="short", users=16, lower=False):
    (a, na), (b, nb) = mean(x, kind, users, key), mean(y, kind, users, key)
    if not a or b is None:
        print(f"  {what}: {x} vs {y}: not measured"); return
    print(f"  {what}: {x} {a:.2f} vs {y} {b:.2f} ({100 * (b / a - 1):+.1f} %{', lower is better' if lower else ''}; "
          f"mean of {na} / {nb} run(s))")
def jpair(x, y, what):
    jx, jy = join(x), join(y)
    if not jx and not jy:
        print(f"  {what}: {x} vs {y}: not measured"); return
    print(f"  {what}: cold 32K joiner beside 4 streams, worst stream gap {x} {f(jx.get('worst_max_gap_s'), 2)} vs {y} "
          f"{f(jy.get('worst_max_gap_s'), 2)} s, joiner TTFT {f(jx.get('joiner_ttft_s'), 2)} vs {f(jy.get('joiner_ttft_s'), 2)} s")
def block(x, y, title, dec=True, join_=False):
    print(f"  -- {title}")
    mpair(x, y, "short agg 16", "agg_out_tps")
    mpair(x, y, "long agg 16", "agg_out_tps", kind="long")
    if dec:
        for u in (1, 2, 4):
            mpair(x, y, f"decode p50 per user at {u} user(s)", "dec_tps_p50", users=u)
    if join_:
        jpair(x, y, "join")
    sx, sy, rb = st(y).get("ident"), st(y).get("status"), st(y).get("robust")
    print(f"     {y}: status {sy or '-'}, short replies {sx or '-'}" + (f", ROBUSTNESS {rb}" if rb else ""))
if any(st(l).get("arm") in PIPE for l in labs):
    idl = [re.sub(r"^\[[0-9:]+\] *", "", ln).strip() for ln in wl if re.search(r"\[IDENT bt (P|PT|PTX) vs ", ln)]
    print("  engine identity (batch_test, 16 slots, solo + batch): " + ("; ".join(idl) if idl else "not run"))
    block("A", "PT", "PT (candidate) vs production", join_=True)
    block("P", "PT", "PT vs P (the sm_86 table + upstream stack: the table gain)")
    block("A", "P", "P vs production (212p with production's switches, no table)", join_=True)
    block("PT", "PTD", "PTD vs PT (the table's DAILY attention against slots-stream)", dec=False)
    block("PT", "PT4", "PT4 vs PT (IQ4_W4A8 + SELECT_TWO_PASS)", dec=False)
    block("PT", "PTX", "PTX vs PT (non-exact: the bit-changing upstream defaults on - how much speed is on the table; "
                       "never a candidate)")
PYEOF
}
if [ "$MODE" = summary ]; then summary; exit 0; fi

# ---- prep (no GPU, production untouched) ---------------------------------------------------------------------------
PRODSRC=$(readlink -f "$C/production.json")
BAD=0; bad() { echo "  PROBLEM: $*"; BAD=1; }
note() { echo "  note: $*"; }
prep() {
    local f a k t
    for f in "$PRODSRC" "$PY" "$L/bench_concurrency.py" "$L/bench_multiturn.py" "$L/compare_replies.py" "$UP/tools/batch_test.py" \
             "$UP/tools/piped_read_join_bench.py" "$UP/tools/early_close_test.py" "$UP/tools/stop_resend_test.py" \
             "$PWT/tools/batch_interleave_test.py" "$PWT/tools/batch_groups_cache_test.py" "$PWT/tools/cancel_park_test.py" \
             "$PRODSRV/serve/server.py" "$UP/serve/server.py" "$PWT/serve/server.py" "$C/T212.manifest.json"; do
        [ -e "$f" ] || bad "missing $f"; done
    if pipe_on; then for f in "$PPT/tools/batch_interleave_test.py" "$PPT/tools/batch_groups_cache_test.py" "$PPT/tools/cancel_park_test.py" \
                              "$PPT/tools/batch_overlap_test.py" "$PPT/tools/early_close_test.py" "$PPT/tools/stop_resend_test.py"; do
        [ -e "$f" ] || bad "missing $f"; done
        "$PY" -I -B "$PPT/tools/batch_overlap_test.py" --help > /dev/null 2>&1 || bad "$PPT/tools/batch_overlap_test.py --help fails"
        # P's short replies: I's of this run, else last night's (the same bench, prompts by uid; A/A-r2, H/H-r2 were 62/62)
        in_arms I || grep -qF '"label":"w212-I-short"' "$P_REF" 2>/dev/null || note "no w212-I-short in P_REF $P_REF and I not in ARMS: P's short replies have no reference (INFO '-')"
    fi
    for t in "$PRODSRV" "$UP" "$PWT"; do
        git -C "$t" diff --quiet HEAD -- serve 2>/dev/null || bad "$t has uncommitted changes under serve/"
        "$PY" -I -c 'import ast,sys; ast.parse(open(sys.argv[1]).read(), sys.argv[1])' "$t/serve/server.py" 2>/dev/null || bad "$t/serve/server.py does not parse"
    done
    [ "$(git -C "$PRODSRV" rev-parse --short=8 HEAD)" = 22f4781f ] || note "$PRODSRV is at $(git -C "$PRODSRV" rev-parse --short=8 HEAD), not 22f4781f"
    [ "$(git -C "$UP" rev-parse --short=8 HEAD)" = 146df3c4 ] || bad "$UP is not at v0.2.12 146df3c4"
    git -C "$PWT" merge-base --is-ancestor 146df3c4 HEAD 2>/dev/null || bad "$PWT does not contain 146df3c4"
    for f in "$L/bench_concurrency.py" "$L/bench_multiturn.py" "$L/compare_replies.py" "$UP/tools/batch_test.py" "$UP/tools/piped_read_join_bench.py"; do
        "$PY" -I -B "$f" --help > /dev/null 2>&1 || bad "$f --help fails"; done
    # the configs: made from production.json as it is now, every name in the arm's tree (gen_configs checks it too)
    local msha psha; msha=$(python3 -c "import json;print(json.load(open('$C/T212.manifest.json'))['production_sha256'][:16])" 2>/dev/null)
    psha=$(sha256sum "$PRODSRC" | cut -c1-16)
    [ "$msha" = "$psha" ] || { [ "${ALLOW_PROD_CHANGED:-0}" = 1 ] && note "production.json changed since --gen-configs (ALLOW_PROD_CHANGED=1)" \
        || bad "production.json (sha $psha) is not what configs/T212-*.json were made from ($msha): run --gen-configs again"; }
    for a in A B C D E E2 F G H H2 H3 H3d; do
        f=$C/T212-$a.json; [ -f "$f" ] || { bad "missing $f"; continue; }
        python3 -c "import json,sys; c=json.load(open(sys.argv[1])); assert c['port']==int(sys.argv[2]) and c['log'].startswith(sys.argv[3])" "$f" "$TPORT" "$OUT/engine-" 2>/dev/null \
            || bad "$f: port/log not as expected (port $TPORT, log under $OUT)"
        for k in $(python3 -c "import json,sys; print(' '.join(k for k in json.load(open(sys.argv[1]))['env'] if not k.startswith('MALLOC_')))" "$f"); do
            git -C "${BIN_TREE[${ARM_BIN[$a]}]}" grep -q -F "\"$k\"" -- src include serve strata_control || bad "T212-$a: $k is not in ${BIN_TREE[${ARM_BIN[$a]}]}"; done
        case $a in A) ;; *) grep -q '"STRATA_PARK_ASYNC"' "$f" && bad "T212-$a sets STRATA_PARK_ASYNC (another feature in v0.2.12)" ;; esac
    done
    for a in $ARMS; do [ -n "${ARM_BIN[$a]:-}" ] || bad "unknown arm $a"; done
    for k in prod stocktab port $(in_arms I && echo int) $(pipe_on && echo pipe); do
        if [ -x "${BIN_SRC[$k]}" ]; then note "binary $k: ${BIN_SRC[$k]} sha256 $(sha256sum "${BIN_SRC[$k]}" | cut -c1-16), $(date -r "${BIN_SRC[$k]}" '+%F %T')"
        else note "binary $k: ${BIN_SRC[$k]} MISSING now - its arms/identity steps are skipped unless it exists at run time"; fi; done
    [ -x "${BIN_SRC[prod]}" ] || bad "production's binary ${BIN_SRC[prod]} is missing"
    [ "$(cfgget "$PRODSRC" exe)" = "${BIN_SRC[prod]}" ] || bad "production.json's exe is $(cfgget "$PRODSRC" exe), not ${BIN_SRC[prod]}"
    if [ -x "${BIN_SRC[port]}" ]; then
        for k in STRATA_QSA_SLOTS_STREAM STRATA_GR_DOWN_PF STRATA_EARLY_SAMPLE; do grep -qa "$k" "${BIN_SRC[port]}" || bad "${BIN_SRC[port]} has no $k"; done; fi
    if [ -x "${BIN_SRC[stocktab]}" ]; then grep -qa STRATA_PIPED_MTP "${BIN_SRC[stocktab]}" || bad "${BIN_SRC[stocktab]} is no v0.2.12 build (no STRATA_PIPED_MTP)"; fi
    [ -f "$UP/build/CMakeCache.txt" ] && note "Strata-212/build: $(grep -E '^STRATA_MMQ_TAB:' "$UP/build/CMakeCache.txt" || echo 'STRATA_MMQ_TAB not in the cache (OFF)')"
    if [ -f "$CTB/CTestTestfile.cmake" ]; then note "ctest build $CTB present ($(grep -E '^STRATA_BUILD_TESTS:' "$CTB/CMakeCache.txt" 2>/dev/null))"
    else note "no ctest build at $CTB yet: host tests and GPU ctest will be skipped unless it exists at run time"; fi
    port_busy "$TPORT" && bad "port $TPORT is in use"
    systemctl --user is-active -q "$BUNIT.scope" && bad "$BUNIT.scope is active"
    systemctl --user is-active -q "$RUNIT.timer" && bad "$RUNIT.timer is active (another window of this kind?)"
    systemctl --user is-active -q "$GUNIT.service" && bad "$GUNIT.service is active"
    systemctl --user is-active -q strata-bench.scope && bad "strata-bench.scope is active (another bench running?)"
    # another run of this script (any RUN): its lock held, or its units still active (a leftover w212-* never blocks)
    local lk u; for lk in /run/user/$(id -u)/window-212*.lock; do [ -e "$lk" ] && [ "$lk" != "$LOCK" ] || continue
        flock -n "$lk" true 2>/dev/null || bad "another window-212.sh holds $lk"; done
    u=$(systemctl --user list-units --state=active --no-legend --plain 'w212*' 2>/dev/null | awk '{print $1}' | tr '\n' ' ')
    [ -n "${u// /}" ] && bad "active units of a window-212 run: $u"
    local others; others=$(pgrep -af '(window-[A-Za-z0-9_-]+|run_strata_variant|soak|ab_quick)\.sh' 2>/dev/null | grep -v -e "$$" -e pgrep | grep -v 'window-212.sh')
    [ -n "$others" ] && bad "another window/bench script runs: $(echo "$others" | head -2 | tr '\n' ';')"
    systemctl --user is-enabled -q strata.service || bad "strata.service is not enabled"
    [ -r "$PRESS" ] || bad "cannot read $PRESS (memory pressure of user@$(id -u))"
    local now; now=$(date +%s)
    [ "$now" -lt "$DL_TS" ] || bad "DEADLINE $(date -d "@$DL_TS" '+%F %T') has passed"
    return $BAD; }
# arm I's own checks (main shell, after prep): a problem of the 212i tree / build / config skips I only, never the window.
# int_problems prints one line per problem; vision_problems the reasons the INFO vision step cannot run.  The same for
# the pipe arms (212p): tree_problems <arm> <tree> <binary key> checks one arm's config against its tree
tree_problems() { local a=$1 t=$2 bk=$3 f=$C/T212-$1.json k
    [ -f "$f" ] || { echo "missing $f (run --gen-configs)"; return; }
    python3 -c "import json,sys; c=json.load(open(sys.argv[1])); assert c['port']==int(sys.argv[2]) and c['log'].startswith(sys.argv[3]) and c['env']['STRATA_CONV_TIER_DIR'].startswith('/media/raid0/strata-kvspill-bench/') and c['exe']==sys.argv[4]" "$f" "$TPORT" "$OUT/engine-" "${BIN_SRC[$bk]}" 2>/dev/null \
        || echo "$f: port/log/tier dir/exe not as expected (port $TPORT, log under $OUT, tier under the bench dir, exe ${BIN_SRC[$bk]})"
    [ -f "$t/serve/server.py" ] || { echo "missing $t/serve/server.py"; return; }
    git -C "$t" diff --quiet HEAD -- serve 2>/dev/null || echo "$t has uncommitted changes under serve/"
    "$PY" -I -c 'import ast,sys; ast.parse(open(sys.argv[1]).read(), sys.argv[1])' "$t/serve/server.py" 2>/dev/null || echo "$t/serve/server.py does not parse"
    git -C "$t" merge-base --is-ancestor 146df3c4 HEAD 2>/dev/null || echo "$t does not contain 146df3c4"
    for k in $(python3 -c "import json,sys; print(' '.join(k for k in json.load(open(sys.argv[1]))['env'] if not k.startswith('MALLOC_')))" "$f"); do
        git -C "$t" grep -q -F "\"$k\"" -- src include serve strata_control || echo "T212-$a: $k is not in $t"; done
    grep -q '"STRATA_PARK_ASYNC"' "$f" && echo "T212-$a sets STRATA_PARK_ASYNC (upstream's other park in $(basename "$t"))"
    [ "$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['env'].get('STRATA_PARK_ASYNC', ''))" "$PRODSRC")" = 1 ] \
        && ! grep -q '"STRATA_PARK_CAPTURE": "1"' "$f" && echo "T212-$a lacks STRATA_PARK_CAPTURE=1 (production's STRATA_PARK_ASYNC=1)"; }
int_problems() { local k
    tree_problems I "$IWT" int
    if [ -x "${BIN_SRC[int]}" ]; then for k in STRATA_PARK_CAPTURE STRATA_SLOT_PARK STRATA_QSA_SLOTS_STREAM STRATA_GATE_EVICT STRATA_SERVE_STOP_BOUND; do
        grep -qa "$k" "${BIN_SRC[int]}" || echo "${BIN_SRC[int]} has no $k"; done; fi; }
# the pipe arms: every listed one's config (a problem skips the pipe arms, never production or the other arms); the
# five switches P puts back must be production's values, PTX must be the only one without the pins
pipe_problems() { local a k
    for a in $PIPE_ARMS; do in_arms "$a" || [ "$a" = P ] || continue; tree_problems "$a" "$PPT" pipe; done
    [ "$(git -C "$PPT" rev-parse --abbrev-ref HEAD 2>/dev/null)" = g3-212-pipe ] || echo "$PPT is not on branch g3-212-pipe"
    git -C "$PPT" merge-base --is-ancestor 98001378 HEAD 2>/dev/null || echo "$PPT does not contain 212i's 98001378"
    python3 - "$C" "$PRODSRC" $PIPE_ARMS <<'PYEOF' || echo "the pipe configs cannot be checked"
import json, os, sys
cdir, prod = sys.argv[1:3]; pe = json.load(open(prod))["env"]
def env(a):
    p = f"{cdir}/T212-{a}.json"
    return json.load(open(p))["env"] if os.path.exists(p) else None
P, PT = env("P"), env("PT")
if P is None or PT is None:
    print("T212-P.json / T212-PT.json missing"); sys.exit(0)
for k in ("STRATA_PIPE_OVERLAP", "STRATA_PIPE_OVERLAP_KVS", "STRATA_SPEC_LANES", "STRATA_SOLO_PIPELINE", "STRATA_READ_YIELD"):
    if P.get(k) != pe.get(k):
        print(f"T212-P: {k}={P.get(k)} but production has {pe.get(k)}")
pins = ("STRATA_GDN_TC", "STRATA_GDN_CONVL2", "STRATA_PROMPT_ATTN_IMMA", "STRATA_AUTO_NOBORROW", "STRATA_PF_W16_CACHE_MIB", "STRATA_MMQ_KQ")
for a in ("P", "PT", "PTD", "PT4"):
    e = env(a)
    if e is not None and any(e.get(k) != "0" for k in pins):
        print(f"T212-{a}: a bit-changing default is not pinned to 0 (an exact arm)")
if P.get("STRATA_SM86_PERF") != "0":
    print("T212-P: STRATA_SM86_PERF is not 0")
for a in ("PT", "PTD", "PT4", "PTX"):
    e = env(a)
    if e is None:
        continue
    if "STRATA_SM86_PERF" in e or "STRATA_SPEC_ADAPT_SM86" in e:
        print(f"T212-{a}: STRATA_SM86_PERF / STRATA_SPEC_ADAPT_SM86 set (the sm_86 table must apply)")
    want = {"STRATA_BATCH_GDN_GROUPS": "1", "STRATA_BATCH_ATTN_ROWS": "1", "STRATA_CKPT_POOL_READY": "4", "STRATA_QSA_SLOTS": "1"}
    want.update({"STRATA_ATTN_DAILY": "1", "STRATA_QSA_SLOTS_STREAM": "0"} if a == "PTD" else
                {"STRATA_ATTN_DAILY": "0", "STRATA_QSA_SLOTS_STREAM": "1"})
    for k, v in want.items():
        if e.get(k) != v:
            print(f"T212-{a}: {k}={e.get(k)}, want {v}")
X = env("PTX")
if X is not None and any(k in X for k in pins):
    print("T212-PTX still pins a bit-changing default")
PYEOF
    # the build the owner was told about (PIPE_SHA= empty: any build; a rebuild since is refused, not measured silently)
    [ -n "$PIPE_SHA" ] && [ -x "${BIN_SRC[pipe]}" ] && ! sha256sum "${BIN_SRC[pipe]}" | grep -q "^$PIPE_SHA" \
        && echo "${BIN_SRC[pipe]} sha256 $(sha256sum "${BIN_SRC[pipe]}" | cut -c1-16) is not the expected $PIPE_SHA (PIPE_SHA=)"
    if [ -x "${BIN_SRC[pipe]}" ]; then for k in STRATA_PIPE_OVERLAP_KVS STRATA_SPEC_LANES STRATA_SOLO_PIPELINE STRATA_READ_YIELD STRATA_PARK_CAPTURE STRATA_SLOT_PARK STRATA_QSA_SLOTS_STREAM; do
        grep -qa "$k" "${BIN_SRC[pipe]}" || echo "${BIN_SRC[pipe]} has no $k"; done
        [ "$(sha256sum "${BIN_SRC[pipe]}" | cut -c1-16)" = "$(sha256sum "$PCB/strata" 2>/dev/null | cut -c1-16)" ] \
            || echo "${BIN_SRC[pipe]} is not $PCB/strata (the ctests would test another build)"; fi; }
vision_problems() { local a=${1:-I} t=${2:-$IWT} f=$C/T212-${1:-I}.json
    [ -f "$t/tools/vision_batch_check.py" ] || { echo "no $t/tools/vision_batch_check.py"; return; }
    "$PY" -I -c 'import PIL.Image' 2>/dev/null || echo "no Pillow in $PY"
    # 212i's server starts the encoder from the config's "vision" block as before (Vision.reader_args: exe, mmproj, model,
    # gpu, threads, max_tokens; the same ENC / OK / READY lines), so production's block is used as it is (212p: the same
    # server)
    python3 - "$f" <<'PYEOF' 2>/dev/null || echo "T212-$a.json cannot be read"
import json, os, sys
c = json.load(open(sys.argv[1])); v = c.get("vision") or {}
for k in ("exe", "mmproj", "model"):
    if not v.get(k) or not os.path.exists(v[k]):
        print(f'the config\'s "vision" block has no usable {k} ({v.get(k)})')
if "--vision" not in c.get("args", []):
    print("no --vision in the args")
if c.get("env", {}).get("STRATA_MROPE_SLOTS") != "1":
    print("STRATA_MROPE_SLOTS is not 1 in the env (the server would run every image request alone)")
PYEOF
}

# ---- restore --------------------------------------------------------------------------------------------------------
STOPPED=0; BACK_DONE=0; WD=; T_STOP=0; PEXE=$(cfgget "$PRODSRC" exe); PRODCFG_SHA=$(sha256sum "$PRODSRC" | cut -c1-16)
PTIER=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['env'].get('STRATA_CONV_TIER_DIR', ''))" "$PRODSRC" 2>/dev/null)
PTIER=${PTIER:-/media/raid0/strata-kvspill}
start_prod() { local t0; log "starting production as before (production.json -> $(readlink "$C/production.json"))"
    systemctl --user start strata.service; t0=$(date +%s)
    until prod_loaded; do
        systemctl --user is-active -q strata.service || { log "strata.service died during the start FAIL"; return 1; }
        [ $(( $(date +%s) - t0 )) -gt 900 ] && { log "production start timeout FAIL"; return 1; }; sleep 5; done
    log "production loaded after $(( $(date +%s) - t0 )) s"; }
disarm() { systemctl --user stop "$RUNIT.timer" 2>/dev/null; systemctl --user reset-failed "$RUNIT.timer" "$RUNIT.service" 2>/dev/null
    systemctl --user stop "$GUNIT.service" 2>/dev/null; systemctl --user reset-failed "$GUNIT.service" 2>/dev/null; }
restore_checks() { local P PE NOWCFG m
    P=$(systemctl --user show -p MainPID --value strata.service)
    PE=$(for c in $(pgrep -P "$P" 2>/dev/null); do readlink "/proc/$c/exe"; done | tr '\n' ' ')
    NOWCFG=$(sha256sum "$(readlink -f "$C/production.json")" | cut -c1-16)
    log "    [prod] ==> production.json sha256 $NOWCFG (at the start $PRODCFG_SHA), engine running: ${PE:-?}: $([ "$NOWCFG" = "$PRODCFG_SHA" ] && echo "$PE" | grep -q "$(basename "$PEXE")" && echo PASS || echo FAIL)"
    m=$(curl -s -m 10 "$URL/v1/models" | head -c 200); log "    [prod] /v1/models: ${m:-no answer}"
    log "    [prod] GET /v1/load: $(curl -s -m 10 "$URL/v1/load" | head -c 240)"
    log "    [prod] /metrics: $(curl -s -m 10 "$URL/metrics" | grep -a -c .) lines, $(curl -s -m 10 "$URL/metrics" | grep -a -m2 -iE 'tier' | cut -c1-120 | tr '\n' ';')"
    local el; el=$(cfgget "$PRODSRC" log)
    n=$(grep -an 'engine started' "$el" | tail -1 | cut -d: -f1)
    log "    [prod] engine log since its start: $(tail -n +"${n:-1}" "$el" | grep -a -c -E 'STRATA_SPEC_LANES=4: lanes|conversation tier: on|STRATA_PIPE_OVERLAP_KVS|STRATA_CONV_MMAP|STRATA_GATE_EVICT') of the lanes / tier / overlap / mmap / gate-evict lines (want >= 5)"
    if [ "$RESTORE_SMOKE" = 1 ]; then
        log "    [prod] chat: $(curl -s -m 120 "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d '{"model":"x","max_tokens":12,"temperature":0,"messages":[{"role":"user","content":"Say OK."}]}' | python3 -c 'import json,sys
try: d=json.load(sys.stdin); m=d["choices"][0]["message"]; print("ok:", repr((m.get("content") or m.get("reasoning_content") or "")[:60]))
except Exception as e: print("FAIL", e)')"
        log "    [prod] image: $(python3 - "$URL" <<'PYEOF'
import base64, json, struct, sys, urllib.request, zlib
def png(w, h):
    rows = b"".join(b"\0" + b"".join(bytes([(x * 8) % 256, (y * 8) % 256, 160]) for x in range(w)) for y in range(h))
    def ch(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + ch(b"IDAT", zlib.compress(rows)) + ch(b"IEND", b"")
url = "data:image/png;base64," + base64.b64encode(png(64, 64)).decode()
body = {"model": "x", "max_tokens": 16, "temperature": 0, "messages": [{"role": "user", "content": [
    {"type": "text", "text": "Describe the colours in one short sentence."}, {"type": "image_url", "image_url": {"url": url}}]}]}
try:
    r = urllib.request.urlopen(urllib.request.Request(sys.argv[1] + "/v1/chat/completions", json.dumps(body).encode(),
                                                      {"Content-Type": "application/json"}), timeout=300)
    print("ok:", repr(json.load(r)["choices"][0]["message"].get("content", "")[:60]))
except Exception as e:  # noqa: BLE001
    print("FAIL", repr(e)[:120])
PYEOF
)"
    fi; }
back() {   # production again (idempotent; every exit path once production was stopped); signals ignored from here on
    [ "$BACK_DONE" = 1 ] && return 0; BACK_DONE=1; trap '' INT TERM HUP
    [ -n "$WD" ] && { pkill -P "$WD" 2>/dev/null; kill "$WD" 2>/dev/null; }
    kill_step; systemctl --user stop "$HUNIT.scope" 2>/dev/null
    [ "$STOPPED" = 1 ] || { disarm; return 0; }
    stop_bench
    if start_prod || { log "production start failed: one more try"; systemctl --user restart strata.service; start_prod; }; then
        disarm; log "$RUNIT.timer and $GUNIT.service removed"
    else log "PRODUCTION DID NOT COME BACK - check strata.service now ($RUNIT.timer and $GUNIT.service left armed)"; fi
    restore_checks
    rm -rf "$TIER"
    log "production back after $(( ($(date +%s) - T_STOP) / 60 )) min of outage"; }
final() { back; [ "$STOPPED" = 1 ] || return 0
    summary > "$OUT/summary.txt" 2>&1; cat "$OUT/summary.txt" >> "$OUT/window.log"
    log "window done (production $(systemctl --user is-active strata.service), summary $OUT/summary.txt)"; }
on_sig() { log "window aborted: ${1:-signal}"; exit 130; }

# ---- dry run ----------------------------------------------------------------------------------------------------------
# (read-only: prep / *_problems only read files, git and systemctl state; nothing is pinned, started, stopped or written)
if [ "$MODE" = dry ]; then
    echo "=== window 212 ($RUN), DRY RUN: nothing is started, stopped or written ($(date '+%F %T'))"
    echo "run: RUN=$RUN OUT=$OUT tier $TIER units $BUNIT.scope $HUNIT.scope $RUNIT.timer $GUNIT.service lock $LOCK"
    echo "prep checks:"; prep; PREP_RC=$?
    [ "$PREP_RC" = 0 ] && echo "  prep: ok" || echo "  prep: PROBLEMS above - the real run would refuse to start"
    if in_arms I; then
        ip=$(int_problems); vp=$(vision_problems)
        echo "arm I checks (a problem skips I only): ${ip:-ok}$( [ -x "${BIN_SRC[int]}" ] && echo ", binary sha256 $(sha256sum "${BIN_SRC[int]}" | cut -c1-16) $(date -r "${BIN_SRC[int]}" '+%F %T')" || echo ", binary ${BIN_SRC[int]} MISSING now (I and its steps are skipped unless it exists at run time)")"
        echo "  vision step: ${vp:-ok}; 212i ctest build $ICB: $([ -f "$ICB/CTestTestfile.cmake" ] && echo present || echo missing)$(for t in conversation_snapshot_test conversation_cache_test conversation_alloc_test conversation_tier_test; do [ -x "$ICB/$t" ] || printf ' (%s not built yet)' "$t"; done)"
    fi
    if pipe_on; then
        pp=$(pipe_problems); vp=$(in_arms PT && vision_problems PT "$PPT")
        echo "pipe arms checks (a problem skips the pipe arms only): ${pp:-ok}$( [ -x "${BIN_SRC[pipe]}" ] && echo ", binary sha256 $(sha256sum "${BIN_SRC[pipe]}" | cut -c1-16) $(date -r "${BIN_SRC[pipe]}" '+%F %T') (expected ${PIPE_SHA:-any}), $PPT@$(git -C "$PPT" rev-parse --short=8 HEAD) $(git -C "$PPT" rev-parse --abbrev-ref HEAD)" || echo ", binary ${BIN_SRC[pipe]} MISSING now")"
        in_arms PT && echo "  PT vision step: ${vp:-ok}"
        ctn=$(grep -rhoE 'add_test\(\[=\[[A-Za-z0-9_.-]+' --include=CTestTestfile.cmake "$PCB" 2>/dev/null | sed 's/^add_test(\[=\[//' | tr '\n' ' ')
        echo "  212p ctest build $PCB: $([ -f "$PCB/CTestTestfile.cmake" ] && echo present || echo missing), $(echo $ctn | wc -w) tests$(for t in $GPU_TESTS $DAILY_TESTS $I_CTESTS $P_GPU_TESTS read_yield_test overlap_kvs_test solo_pipeline_test spec_lanes_test arena_carve_test; do case " $ctn " in *" $t "*) ;; *) printf ' (no test %s)' "$t" ;; esac; done)"
        if in_arms I; then echo "  P's short-reply reference: this run's arm I (w212-I-short)"
        else echo "  P's short-reply reference: $(python3 -c "import json,sys; print(sum(1 for l in open(sys.argv[1]) if l.strip() and json.loads(l).get('label')=='w212-I-short'))" "$P_REF" 2>/dev/null || echo 0) levels of w212-I-short in $P_REF (copied as ref-I-short at the start; PT PTD PT4 use P's replies once P = I)"; fi
    fi
    echo "window: start $(date -d "@$WS_TS" '+%F %T') (refused earlier unless FORCE=1), DEADLINE $(date -d "@$DL_TS" '+%F %T') (no step starts later; restore then), HARD_STOP $(date -d "@$(( DL_TS + HARD_GRACE ))" '+%F %T'), $RUNIT.timer $(date -d "@$(( DL_TS + RESTORE_GRACE ))" '+%F %T')"
    echo "gates: IDENT_GATE=$IDENT_GATE HTTP_GATE=$HTTP_GATE; aborts: free VRAM < $VRAM_MIN MiB (2 samples), scope anon+shmem > $RSS_MAX_GIB GiB, user@$(id -u) memory pressure >= $PRESS_MAX %, hang $HANG_S s, crash lines, two crashes in a row, anything on :$PPORT"
    echo "server env (production's drop-in, read now): $SRVENV_ALL"
    echo
    echo "phase 1 (production serving, ~15 min cap): host ctest on $CTB (CUDA_VISIBLE_DEVICES= , -j4, scope MemoryMax 8G), serve unittests of $CTT (MemoryMax 4G, cap ${SERVE_UNIT_CAP} s)$( [ "$CTK" != pipe ] && pipe_on && echo "; then the same for $PCB / $PPT")"
    echo "phase 3 (~6 min): GPU ctest on card 0 of $CTB: $(echo $GPU_TESTS | wc -w) tests + $DAILY_TESTS"
    in_arms I && echo "         + INFO (~5 min): GPU ctest on card 0 of $ICB: $(echo $I_CTESTS)"
    pipe_on && echo "         + INFO (~2 min): GPU ctest on card 0 of $PCB: $(echo $I_CTESTS)$( [ "$CTK" != pipe ] && echo " $P_GPU_TESTS")"
    BT_PLAN="A $(old_on && echo B H) $(in_arms I && echo I) $(any_arm P PT PTD PT4 && echo P) $(any_arm PT PTD PT4 && echo PT) $(in_arms PTX && echo PTX)"
    echo "phase 4: batch_test $(echo $BT_PLAN) (~2.5 min each; 16 slots, 40K prompt, solo + batch)$(old_on && echo ', interleave / groups_cache MOVE 1+0 / cancel_park on 212m-p2')"
    in_arms I && echo "         + (~15 min): batch_test I (I vs H, B when H has no dump: IDENT_GATE=$IDENT_GATE), INFO groups_cache on 212i with STRATA_PARK_CAPTURE_VERIFY=1 STRATA_SLOT_PARK_VERIFY=1$( [ -f "$IWT/tools/cache_reuse_check.py" ] && echo ' + cache_reuse_check' || echo ' (no tools/cache_reuse_check.py in 212i: not run)')"
    pipe_on && echo "         pipe identity: bt P vs A, PT vs A, PT vs P must be 16/16 (IDENT_GATE=$IDENT_GATE: P != A skips P PT PTD PT4, PT != A/P skips PT PTD PT4), PTX vs A INFO (DIFF expected, never gated)"
    in_arms PT && echo "         + INFO (~8 min) on PT's env with 212p's tools: interleave, batch_overlap_test, groups_cache with the VERIFY switches, cancel_park"
    echo "phase 5 arms (~${ARM_EST}s cap each, ~$(( ${ARM_TYP:-450} / 60 )) min typical):"
    printf '  %-4s %-9s %-10s %-26s %s\n' arm binary present server "env vs reference / args"
    for a in $ARMS; do
        k=${ARM_BIN[$a]:-?}; [ "$a" = H3 ] && k=port
        pres=$([ -x "${BIN_SRC[$k]:-/nonexistent}" ] && echo yes || echo MISSING)
        diffs=$(python3 - "$C" "$a" <<'PYEOF' 2>/dev/null
import json, sys
cdir, a = sys.argv[1:3]
ref = {"A": "production", "B": "production", "H": "production", "I": "production", "P": "I", "PT": "P", "PTD": "PT",
       "PT4": "PT", "PTX": "PT"}.get(a, "H" if a.startswith("H") else "B")
c = json.load(open(f"{cdir}/T212-{a}.json"))
r = json.load(open(f"{cdir}/production.json" if ref == "production" else f"{cdir}/T212-{ref}.json"))
e, p = c["env"], r["env"]
d = [f"{k}={e[k]}" for k in e if p.get(k) != e[k] and k != "STRATA_CONV_TIER_DIR"]
gone = [k for k in p if k not in e]
x = [v for v in c["args"] if v not in r["args"]]
print(f"vs {ref}: " + (" ".join(d) or "-") + (f" | minus {len(gone)} switches (T212.manifest.json)" if len(gone) > 6 else
      (" | minus " + " ".join(gone) if gone else "")) + " | args + " + (" ".join(x) or "-"))
PYEOF
)
        printf '  %-4s %-9s %-10s %-26s %s\n' "$a" "$k" "$pres" "$(basename "${ARM_SRV[$a]}")@$(git -C "${ARM_SRV[$a]}" rev-parse --short=8 HEAD)" "$diffs"
        echo "       server env: $(srv_env "$a")$( [ -n "$(srv_env_dropped "$a")" ] && echo " (dropped, not in this server: $(srv_env_dropped "$a"))")"
        echo "       INFO lines after the probe: $(echo "${WANT_BOOT[$a]:-none}" | sed 's/;;/ + /g' | tr -d '\\')${FAILPAT[$a]:+; stops on: $(echo "${FAILPAT[$a]}" | tr -d '\\')}${WANT_END[$a]:+; at the end (INFO): $(echo "${WANT_END[$a]}" | sed 's/;;/ + /g' | tr -d '\\')}"
        [ "$a" = H3 ] && echo "       (H3 = T212-H3 when H's short agg at 16 users >= H2's, else T212-H3d)"
        [ "$a" = I ] && echo "       (short replies vs H, B without H; after the join: INFO vision_batch_check --url $TURL/v1, cap 900 s)"
        case $a in
            P) echo "       (exact: short replies vs I$(in_arms I || echo " (last night's, ref-I-short)"), bt-P vs bt-A)" ;;
            PT) echo "       (THE candidate; exact: short replies vs P, bt-PT vs bt-A and bt-P; after the join: early_close, stop_resend, INFO vision_batch_check)" ;;
            PTD|PT4) echo "       (exact: short replies vs P)" ;;
            PTX) echo "       (NOT exact: INFO only, never a candidate; replies vs PT reported as DIFF-expected)" ;;
        esac
    done
    n=$(echo $ARMS | wc -w)
    echo "  steps per arm: boot (cap $BOOT_CAP s), warm-up, 4-user probe + INFO check, short 1/2/4/8/16 (greedy, identity for$EXACT_ARMS), long 16, multiturn 16x3, join cold32k$( echo " $ARMS " | grep -q ' H ' && echo '; H also early_close + stop_resend')$(in_arms PT && echo '; PT also early_close + stop_resend + vision')"
    if pipe_on; then R2P=""; for a in A PT P; do in_arms "$a" && R2P="$R2P $a"; done
        [ "$R2_FIRST" = auto ] || R2P=$R2_FIRST
        echo "  then r2 repeats:$R2P (always, those that ran ok), then the close pairs (|delta| < 3 %) while >= $ARM_EST s are left"
    else echo "  then r2 repeats of close pairs (|delta| < 3 %) while >= $ARM_EST s are left"; fi
    # the outage from the stop (after phase 1, ~15 min after the start): typical times of the w212 night
    nb=$(echo $BT_PLAN | wc -w); typ=${ARM_TYP:-450}
    iex=0; in_arms I && iex=$(( iex + 5 + 15 + 10 )); old_on && iex=$(( iex + 7 )); in_arms PT && iex=$(( iex + 8 + 6 ))   # PT: engine INFO tests, HTTP + vision
    nr2=0; pipe_on && nr2=$(echo $R2P | wc -w)
    est=$(( 6 + (nb * 150 + 59) / 60 + (n + nr2) * typ / 60 + iex + 6 ))
    echo "expected outage: ~$est min (GPU ctest ~6, identity ~$(( (nb * 150 + 59) / 60 )), $n arms + $nr2 r2 x ~$(( typ / 60 ))$( [ $iex -gt 0 ] && echo ", extra steps ~$iex"), restore ~6) from the stop at ~$(date -d "@$(( WS_TS + 900 ))" +%H:%M): production back ~$(date -d "@$(( WS_TS + 900 + est * 60 ))" +%H:%M); at most until DEADLINE $(date -d "@$DL_TS" +%H:%M) + restore (more close-pair r2 runs fill up to DEADLINE)"
    [ $(( WS_TS + 900 + est * 60 )) -gt $(( DL_TS - ARM_EST )) ] && echo "  note: that is past DEADLINE - $ARM_EST s: the last arms in ARMS order may be skipped; a shorter ARMS list keeps them"
    exit 0
fi

# ---- the window -------------------------------------------------------------------------------------------------------
now=$(date +%s)
if [ "$now" -lt "$WS_TS" ] && [ "$FORCE" != 1 ]; then echo "too early: GPU use from $(date -d "@$WS_TS" '+%F %T') only (FORCE=1 overrides)"; exit 1; fi
[ "$now" -ge "$DL_TS" ] && { echo "DEADLINE $(date -d "@$DL_TS" '+%F %T') has passed"; exit 1; }
# one window-212 at a time (prep's pgrep check leaves out window-212.sh itself, and a second one would stop this one's
# restore timer and fight over the GPUs); long-lived children close fd 9
# (the lock carries the run's name; prep refuses while another run's lock is held)
exec 9> "$LOCK" && flock -n 9 || { echo "another window-212.sh holds $LOCK: refused"; exit 1; }
trap 'echo "window aborted before production was stopped"; systemctl --user stop "$HUNIT.scope" 2>/dev/null; disarm; exit 1' INT TERM HUP
[ -e "$OUT" ] && { mv "$OUT" "$OUT.prev-$(date +%Y%m%d-%H%M%S)" || { echo "cannot move $OUT aside"; exit 1; }; }
mkdir -p "$OUT" "$OUT/cfg" "$OUT/bin"
log "=== window 212 prep (deadline $(date -d "@$DL_TS" '+%F %T'); arms $ARMS; IDENT_GATE=$IDENT_GATE HTTP_GATE=$HTTP_GATE)"
prep 2>&1 | tee -a "$OUT/window.log"; [ "${PIPESTATUS[0]}" = 0 ] || { log "ABORT before the window: prep failed (production untouched)"; exit 1; }
systemctl --user is-active -q strata.service || { log "ABORT: strata.service is not active: start it first"; exit 1; }
# the binaries of the listed arms (+ 212m's for the first night's identity / engine tests)
PIN_KEYS=$(for a in A $ARMS; do echo "${ARM_BIN[$a]}"; done | sort -u | tr '\n' ' '); old_on && PIN_KEYS="$PIN_KEYS stocktab port"
for k in $(echo $PIN_KEYS | tr ' ' '\n' | sort -u); do pin_bin "$k" > /dev/null && log "pinned $k: ${PINNED[$k]}" || log "binary $k missing now (pinned at first use if it appears)"; done
I_VISION_SKIP=""
if in_arms I; then   # arm I's own checks: a problem skips I (and its steps) only
    ip=$(int_problems); [ -n "$ip" ] && { SKIP_BIN[int]="212i checks: $(echo "$ip" | head -3 | tr '\n' ';')"; log "arm I: ${SKIP_BIN[int]}"; }
    I_VISION_SKIP=$(vision_problems | head -3 | tr '\n' ';'); [ -n "$I_VISION_SKIP" ] && log "arm I: the vision step will be skipped: $I_VISION_SKIP"
fi
PT_VISION_SKIP=""
if pipe_on; then     # the pipe arms' own checks: a problem skips them (and their steps) only
    pp=$(pipe_problems); [ -n "$pp" ] && { SKIP_BIN[pipe]="212p checks: $(echo "$pp" | head -3 | tr '\n' ';')"; log "pipe arms: ${SKIP_BIN[pipe]}"; }
    in_arms PT && { PT_VISION_SKIP=$(vision_problems PT "$PPT" | head -3 | tr '\n' ';'); [ -n "$PT_VISION_SKIP" ] && log "arm PT: the vision step will be skipped: $PT_VISION_SKIP"; }
    # P's reference short replies: I's of this run when I runs, else last night's w212-I-short (relabelled ref-I-short;
    # the same bench_concurrency.py, levels, rounds and prompts by uid)
    if ! in_arms I && [ -f "$P_REF" ]; then
        python3 - "$P_REF" "$OUT/ref.requests.jsonl" <<'PYEOF' && log "P's reference: $(wc -l < "$OUT/ref.requests.jsonl") levels of w212-I-short from $P_REF (sha256 $(sha256sum "$P_REF" | cut -c1-16)) as ref-I-short"
import json, sys
n = 0
with open(sys.argv[2], "w") as o:
    for ln in open(sys.argv[1]):
        if ln.strip():
            r = json.loads(ln)
            if r.get("label") == "w212-I-short":
                r["label"] = "ref-I-short"; o.write(json.dumps(r) + "\n"); n += 1
sys.exit(0 if n else 1)
PYEOF
    fi
    [ -s "$OUT/ref.requests.jsonl" ] || in_arms I || log "P's reference: none (no w212-I-short in $P_REF): P's short identity is INFO '-'"
fi
# ---- phase 0: production as it is
mkdir -p "$OUT/prod-before"
{ echo "production.json -> $PRODSRC sha256 $PRODCFG_SHA"; echo "exe $PEXE sha256 $(sha256sum "$PEXE" | cut -c1-16)"
  echo "server $PRODSRV @$(git -C "$PRODSRV" rev-parse --short=8 HEAD)"; echo "server env: $SRVENV_ALL"
  P=$(systemctl --user show -p MainPID --value strata.service); for c in $P $(pgrep -P "$P"); do echo "pid $c $(readlink /proc/$c/exe) $(grep -E 'VmRSS|RssAnon|RssFile|RssShmem' /proc/$c/status | tr -s ' \t' ' ' | tr '\n' ' ')"; done
  cg=$(systemctl --user show -p ControlGroup --value strata.service); echo "cgroup memory.current $(cat /sys/fs/cgroup$cg/memory.current) peak $(cat /sys/fs/cgroup$cg/memory.peak 2>/dev/null)"
  nvidia-smi --query-gpu=index,name,memory.used,memory.free,power.limit,power.draw,temperature.gpu --format=csv
  free -g; cat "$PRESS"; } > "$OUT/prod-before/state.txt" 2>&1
curl -s -m 10 "$URL/metrics" > "$OUT/prod-before/metrics.txt"; curl -s -m 10 "$URL/v1/load" > "$OUT/prod-before/load.json"
log "production recorded ($OUT/prod-before): $(grep -m1 '^exe' "$OUT/prod-before/state.txt"); power limits $(nvidia-smi --query-gpu=power.limit --format=csv,noheader | tr '\n' ' ') (owner's night cap is 280 W, day 320 W - not changed here)"
# ---- phase 1: CPU host tests (production serving)
# host_phase <binary key> <build> <tree> <file suffix>: ctest without a card + the tree's serve unittests; a failure of
# one of OUR tests (added since v0.2.12 in the tree's CMakeLists, or OUR_TESTS_FIXED) skips that binary's arms
host_phase() { local k=$1 b=$2 t=$3 x=$4 ours rc fails ourf tt
    ours="$OUR_TESTS_FIXED $(git -C "$t" diff 146df3c4..HEAD -- CMakeLists.txt | sed -nE 's/^\+.*add_test\(NAME ([A-Za-z0-9_]+).*/\1/p' | tr '\n' ' ')"
    log "host tests $(basename "$t") (production serving; ours: $ours)"
    WATCH=""; VRAM_ON=0; MON_HANG=900; MON_LBL=host
    timeout -k 30 900 systemd-run --user --scope --quiet --unit=$HUNIT -p MemoryMax=8G -p MemorySwapMax=0 nice -n 19 taskset -c 0-63 \
        env CUDA_VISIBLE_DEVICES= ctest --test-dir "$b" --output-on-failure -j4 --timeout 600 > "$OUT/host_ctest$x.txt" 2>&1 < /dev/null
    rc=$?; systemctl --user stop "$HUNIT.scope" 2>/dev/null; systemctl --user reset-failed "$HUNIT.scope" 2>/dev/null
    ctest_status "$OUT/host_ctest$x.txt" > "$OUT/host_ctest_status$x.txt"
    fails=$(awk '$2 !~ /^(Passed|Skipped|Not)/ {print $1}' "$OUT/host_ctest_status$x.txt" | tr '\n' ' ')
    ourf=; for tt in $fails; do case " $GPU_TESTS $DAILY_TESTS $P_GPU_TESTS " in *" $tt "*) continue ;; esac   # (judged on a card in phase 3)
        case " $ours " in *" $tt "*) ourf="$ourf $tt" ;; esac; done
    log "    [host tests $(basename "$t")] ==> ctest rc $rc: $(grep -E 'tests passed|tests failed' "$OUT/host_ctest$x.txt" | tail -1); failed: ${fails:-none}; ours failed: ${ourf:-none} $([ -z "$ourf" ] && echo PASS || echo FAIL)"
    [ -n "$ourf" ] && SKIP_BIN[$k]="our host tests failed:$ourf"
    timeout -k 30 "$SERVE_UNIT_CAP" systemd-run --user --scope --quiet --unit=$HUNIT -p MemoryMax=4G -p MemorySwapMax=0 nice -n 19 taskset -c 0-63 \
        env -C "$t" APPDATA="$OUT/xdg" XDG_CONFIG_HOME="$OUT/xdg/c" XDG_STATE_HOME="$OUT/xdg/s" XDG_DATA_HOME="$OUT/xdg/d" LOCALAPPDATA="$OUT/xdg/l" \
        "$PY" -B -m unittest discover -s serve -p 'test_*.py' -t . > "$OUT/serve_unit$x.txt" 2>&1 < /dev/null
    rc=$?; systemctl --user stop "$HUNIT.scope" 2>/dev/null; systemctl --user reset-failed "$HUNIT.scope" 2>/dev/null
    log "    [serve unittests $(basename "$t")] ==> rc $rc: $(grep -E '^(Ran|OK|FAILED)' "$OUT/serve_unit$x.txt" | tr '\n' ' ') INFO (rc 124 = cap ${SERVE_UNIT_CAP} s; stock test_stream_ownership hangs)"; }
if [ "$HOST_TESTS" = 1 ] && [ -f "$CTB/CTestTestfile.cmake" ]; then host_phase "$CTK" "$CTB" "$CTT" ""
else log "host tests: SKIP (HOST_TESTS=$HOST_TESTS, $CTB/CTestTestfile.cmake $([ -f "$CTB/CTestTestfile.cmake" ] && echo present || echo missing))"; fi
if [ "$HOST_TESTS" = 1 ] && [ "$CTK" != pipe ] && pipe_on; then   # both kinds of arms: 212p's build too
    if [ -f "$PCB/CTestTestfile.cmake" ]; then host_phase pipe "$PCB" "$PPT" _pipe
    else log "host tests 212p: SKIP ($PCB/CTestTestfile.cmake missing)"; fi; fi
# ---- phase 2: stop production
# runnable [arm...]: the arms of ARMS other than A (and the given ones) with no skip reason now
runnable() { local a x; for a in $ARMS; do [ "$a" = A ] && continue; x=$a; [ "$a" = H3 ] && x=H3d
    case " $* " in *" $a "*) continue ;; esac
    [ -z "${SKIP_ARM[$a]:-}${SKIP_BIN[${ARM_BIN[$x]:-}]:-}" ] && echo "$a"; done | tr '\n' ' '; }
if [ "$(echo $ARMS)" != A ] && [ -z "$(runnable)" ]; then
    log "ABORT: every arm of ARMS other than A is skipped already ($(for a in $ARMS; do [ "$a" = A ] || printf '%s: %s; ' "$a" "${SKIP_ARM[$a]:-${SKIP_BIN[${ARM_BIN[$a]:-}]:-}}"; done)): production left running"; exit 1; fi
for _ in $(seq 180); do prod_loaded && break; sleep 5; done
prod_loaded || { log "ABORT: production does not report loaded: not stopping it"; exit 1; }
for p in $(gpu_apps); do [ "$(readlink -f "/proc/$p/exe" 2>/dev/null)" = "$(readlink -f "$PEXE")" ] \
    || { log "ABORT: GPU process $p ($(readlink -f "/proc/$p/exe" 2>/dev/null)) is not production's engine: not starting"; exit 1; }; done
[ "$(left)" -gt 1800 ] || { log "ABORT: less than 30 min before DEADLINE: production left running"; exit 1; }
systemctl --user stop "$RUNIT.timer" 2>/dev/null; systemctl --user reset-failed "$RUNIT.timer" "$RUNIT.service" 2>/dev/null
systemd-run --user --quiet --unit="$RUNIT" --on-calendar="$(date -d "@$(( DL_TS + RESTORE_GRACE ))" '+%F %T')" --timer-property=AccuracySec=1s \
    /bin/sh -c "systemctl --user stop $BUNIT.scope; systemctl --user start strata.service" && systemctl --user is-active -q "$RUNIT.timer" \
    || { log "ABORT: could not arm $RUNIT.timer: production left running"; exit 1; }
: > "$OUT/step.pgid"
systemd-run --user --quiet --collect --unit="$GUNIT" /bin/sh -c 'tail --pid="$1" -f /dev/null
    g=$(cat "$2" 2>/dev/null); [ -n "$g" ] && [ "$(ps -o comm= -p "$g" 2>/dev/null)" = timeout ] && /bin/kill -s KILL -- "-$g"
    systemctl --user stop "$3.scope"; systemctl --user start strata.service' sh "$$" "$OUT/step.pgid" "$BUNIT" \
    && sleep 2 && systemctl --user is-active -q "$GUNIT.service" || { disarm; log "ABORT: could not arm $GUNIT.service: production left running"; exit 1; }
log "REMINDER: coordinate with the docs/bench Claude session (ListAgents; e.g. strataq38fn-e8) before production stops - agreed by hand before this start"
log "window start: $RUNIT.timer armed for $(date -d "@$(( DL_TS + RESTORE_GRACE ))" '+%F %T'), $GUNIT.service waits for PID $$; stopping strata.service"
trap '' INT TERM HUP; trap final EXIT
systemctl --user stop strata.service
STOPPED=1; T_STOP=$(date +%s)
trap 'on_sig signal' INT TERM HUP
( exec 9>&-; sleep $(( DL_TS + HARD_GRACE - $(date +%s) )); tr '\0' ' ' < /proc/$$/cmdline 2>/dev/null | grep -q window-212 && kill -TERM $$ ) 2>/dev/null & WD=$!
for _ in $(seq 60); do port_busy $PPORT || break; sleep 2; done
port_busy $PPORT && { log "ABORT: port $PPORT still open after the stop"; exit 1; }
free_gpus || { log "ABORT: GPUs not free after the stop"; exit 1; }
log "production stopped; VRAM free $(vram)"

# ---- phase 3: GPU ctest subset (one card, 212m build; 212p's with only pipe arms, + P_GPU_TESTS)
if [ "$GPU_CTEST" = 1 ] && [ -f "$CTB/CTestTestfile.cmake" ] && [ "$ABORT_ALL" = 0 ]; then
    pin_bin "$CTK" > /dev/null && p=${PINNED[$CTK]} && [ -x "$CTB/strata" ] && log "$(basename "$CTT") build/strata sha256 $(sha256sum "$CTB/strata" | cut -c1-16) vs pinned $(basename "${BIN_SRC[$CTK]}") $(basename "$p" | sed 's/.*-//') $([ "$(sha256sum "$CTB/strata" | cut -c1-16)" = "$(basename "$p" | sed 's/.*-//')" ] && echo '(same build)' || echo '(OTHER build: ctest results are of the build dir)')"
    RE="^($(echo $GPU_TESTS | tr ' ' '|'))\$"; RD="^($(echo $DAILY_TESTS | tr ' ' '|'))\$"
    # (a failing test is a result here, not a crash: no crash-line scan; ctest's own --timeout cuts a hung test)
    WATCH=""; ENGLOG=""; VRAM_ON=0; MON_HANG=99999; MON_LBL=ctest
    run_step gpu_ctest 1500 "${SCOPE[@]}" env CUDA_VISIBLE_DEVICES=0 STRATA_ATTN_DAILY=0 ctest --test-dir "$CTB" -R "$RE" -j1 --timeout 600 --output-on-failure
    stop_bench
    run_step gpu_ctest_daily 400 "${SCOPE[@]}" env CUDA_VISIBLE_DEVICES=0 ctest --test-dir "$CTB" -R "$RD" -j1 --timeout 300 --output-on-failure
    stop_bench
    cat "$OUT/gpu_ctest.txt" "$OUT/gpu_ctest_daily.txt" > "$OUT/gpu_ctest_all.txt" 2>/dev/null
    ctest_status "$OUT/gpu_ctest_all.txt" > "$OUT/gpu_ctest_status.txt"
    log "    [gpu ctest] ==> $(awk '{c[$2]++} END {for (k in c) printf "%s %d, ", k, c[k]}' "$OUT/gpu_ctest_status.txt")missing: $(for t in $GPU_TESTS $DAILY_TESTS; do grep -q "^$t " "$OUT/gpu_ctest_status.txt" || printf '%s ' "$t"; done)"
else log "gpu ctest: SKIP (GPU_CTEST=$GPU_CTEST, build $([ -f "$CTB/CTestTestfile.cmake" ] && echo present || echo missing))"; fi
# INFO: 212i's conversation tests on one card (never a lever drop, never "two crashes in a row"; the tier test's files go
# under the bench tier, removed at the end)
if in_arms I && [ "$GPU_CTEST" = 1 ] && [ -f "$ICB/CTestTestfile.cmake" ] && [ "$ABORT_ALL" = 0 ] && [ -z "${SKIP_BIN[int]:-}" ]; then
    WATCH=""; ENGLOG=""; VRAM_ON=0; MON_HANG=99999; MON_LBL=ctest-212i; INFO_STEP=1; mkdir -p "$TIER/ctest-212i"
    run_step gpu_ctest_212i 900 "${SCOPE[@]}" env CUDA_VISIBLE_DEVICES=0 TMPDIR="$TIER/ctest-212i" ctest --test-dir "$ICB" \
        -R "^($(echo $I_CTESTS | tr ' ' '|'))\$" -j1 --timeout 300 --output-on-failure
    INFO_STEP=0; stop_bench
    ctest_status "$OUT/gpu_ctest_212i.txt" > "$OUT/gpu_ctest_212i_status.txt"
    log "    [gpu ctest 212i] ==> $(awk '{printf "%s %s, ", $1, $2}' "$OUT/gpu_ctest_212i_status.txt")missing: $(for t in $I_CTESTS; do grep -q "^$t " "$OUT/gpu_ctest_212i_status.txt" || printf '%s ' "$t"; done)INFO"
elif in_arms I; then log "gpu ctest 212i: SKIP (GPU_CTEST=$GPU_CTEST, build $ICB $([ -f "$ICB/CTestTestfile.cmake" ] && echo present || echo missing)${SKIP_BIN[int]:+, ${SKIP_BIN[int]}})"; fi
# INFO: 212p's conversation tests (as 212i's), and P_GPU_TESTS when the judged run above was 212m's (with only pipe arms
# they are in GPU_TESTS: a failing kv_stream_overlap_parity drops STRATA_PIPE_OVERLAP_KVS)
if pipe_on && [ "$GPU_CTEST" = 1 ] && [ -f "$PCB/CTestTestfile.cmake" ] && [ "$ABORT_ALL" = 0 ] && [ -z "${SKIP_BIN[pipe]:-}" ]; then
    PT_CT="$I_CTESTS"; [ "$CTK" = pipe ] || PT_CT="$PT_CT $P_GPU_TESTS"
    WATCH=""; ENGLOG=""; VRAM_ON=0; MON_HANG=99999; MON_LBL=ctest-212p; INFO_STEP=1; mkdir -p "$TIER/ctest-212p"
    run_step gpu_ctest_212p 900 "${SCOPE[@]}" env CUDA_VISIBLE_DEVICES=0 TMPDIR="$TIER/ctest-212p" ctest --test-dir "$PCB" \
        -R "^($(echo $PT_CT | tr ' ' '|'))\$" -j1 --timeout 300 --output-on-failure
    INFO_STEP=0; stop_bench
    ctest_status "$OUT/gpu_ctest_212p.txt" > "$OUT/gpu_ctest_212p_status.txt"
    log "    [gpu ctest 212p] ==> $(awk '{printf "%s %s, ", $1, $2}' "$OUT/gpu_ctest_212p_status.txt")missing: $(for t in $PT_CT; do grep -q "^$t " "$OUT/gpu_ctest_212p_status.txt" || printf '%s ' "$t"; done)INFO"
elif pipe_on; then log "gpu ctest 212p: SKIP (GPU_CTEST=$GPU_CTEST, build $PCB $([ -f "$PCB/CTestTestfile.cmake" ] && echo present || echo missing)${SKIP_BIN[pipe]:+, ${SKIP_BIN[pipe]}})"; fi
# the levers whose parity test failed (host: sampler_same_test; GPU: the rest; a GPU test's host-phase result is not used)
for k in "${!LEVER_TESTS[@]}"; do for t in ${LEVER_TESTS[$k]}; do
    case " $GPU_TESTS $DAILY_TESTS " in *" $t "*) sf=$OUT/gpu_ctest_status.txt ;;
        *) case " $P_GPU_TESTS " in *" $t "*) sf=$OUT/gpu_ctest_212p_status.txt ;;   # (212m's run above: 212p's INFO run)
                                     *) sf="$OUT/host_ctest_status.txt $OUT/host_ctest_status_pipe.txt" ;; esac ;; esac
    s=$(cat $sf 2>/dev/null | awk -v t="$t" '$1 == t && $2 !~ /^(Passed|Skipped|Not)/ {print $2; exit}')
    case $s in Passed|Skipped|Not|'') ;; *) drop_lever "$k" "ctest $t $s" ;; esac; done; done
for a in "${!ARM_POINT[@]}"; do n=0; m=0; for k in ${ARM_POINT[$a]}; do m=$((m + 1)); case " $DROPS " in *" $k "*) n=$((n + 1)) ;; esac; done
    [ "$n" = "$m" ] && SKIP_ARM[$a]="its lever(s) dropped: ${ARM_POINT[$a]}"; done
[ -n "$DROPS" ] && echo "$DROPS" > "$OUT/drops.txt"

# ---- phase 4: identity
if [ "$IDENTITY" = 1 ] && [ "$ABORT_ALL" = 0 ]; then
    # bt-A always; B and H for the first night's arms; I; P (PT's reference too), PT, PTX (INFO) for the pipe arms
    BT_LIST="A $(old_on && echo B H) $(in_arms I && echo I) $(any_arm P PT PTD PT4 && echo P) $(any_arm PT PTD PT4 && echo PT) $(in_arms PTX && echo PTX)"
    log "identity: batch_test for $BT_LIST"
    for a in $BT_LIST; do [ "$ABORT_ALL" = 1 ] && break
        [ -n "${SKIP_BIN[${ARM_BIN[$a]}]:-}" ] && { log "identity bt-$a: SKIP (${SKIP_BIN[${ARM_BIN[$a]}]})"; continue; }
        bt_run "$a"; done
    c_ba=2; c_hb=2; c_ha=2
    if old_on; then
    r_ba=$(bt_same A B); c_ba=$?; r_hb=$(bt_same B H); c_hb=$?; r_ha=$(bt_same A H); c_ha=$?
    log "    [IDENT bt B vs A] ==> $r_ba $([ $c_ba = 0 ] && echo PASS || { [ $c_ba = 1 ] && echo FAIL || echo SKIP; })"
    log "    [IDENT bt H vs B] ==> $r_hb $([ $c_hb = 0 ] && echo PASS || { [ $c_hb = 1 ] && echo FAIL || echo SKIP; })"
    log "    [IDENT bt H vs A] ==> $r_ha $([ $c_ha = 0 ] && echo PASS || { [ $c_ha = 1 ] && echo FAIL || echo SKIP; })"
    fi
    case $IDENT_GATE in
        strict) [ $c_ba = 1 ] && SKIP_BIN[stocktab]="batch_test B != A (IDENT_GATE=strict)"
                [ $c_ha = 1 ] && SKIP_BIN[port]="batch_test H != A (IDENT_GATE=strict)" ;;
        port)   [ $c_hb = 1 ] && SKIP_BIN[port]="batch_test H != B (IDENT_GATE=port)" ;;
    esac
    for k in stocktab port; do [ -n "${SKIP_BIN[$k]:-}" ] && log "IDENT gate: the $k arms are skipped: ${SKIP_BIN[$k]}"; done
    if in_arms I && [ -z "${SKIP_BIN[int]:-}" ]; then   # I is exact against H (same kernels + cache features); B without H's dump
        iref=H; r_i=$(bt_same H I); c_i=$?; [ $c_i = 2 ] && [ -f "$OUT/bt-I.json" ] && { iref=B; r_i=$(bt_same B I); c_i=$?; }
        log "    [IDENT bt I vs $iref] ==> $r_i $([ $c_i = 0 ] && echo PASS || { [ $c_i = 1 ] && echo FAIL || echo SKIP; })"
        [ "$IDENT_GATE" != off ] && [ $c_i = 1 ] && { SKIP_BIN[int]="batch_test I != $iref (IDENT_GATE=$IDENT_GATE)"; log "IDENT gate: arm I is skipped: ${SKIP_BIN[int]}"; }
    fi
    # the pipe arms: P and PT must be A's tokens (solo + batch, 16/16), PT also P's; PTX is INFO (DIFF expected).
    # IDENT_GATE strict / port: P != A skips P PT PTD PT4, PT != A or P skips PT PTD PT4 (the table's arms); PTX never
    if pipe_on && [ -z "${SKIP_BIN[pipe]:-}" ]; then
        vd() { [ "$1" = 0 ] && echo PASS || { [ "$1" = 1 ] && echo FAIL || echo SKIP; }; }
        if any_arm P PT PTD PT4; then
            r_pa=$(bt_same A P); c_pa=$?; log "    [IDENT bt P vs A] ==> $r_pa $(vd $c_pa)"
            if [ "$IDENT_GATE" != off ] && [ $c_pa = 1 ]; then
                for x in P PT PTD PT4; do SKIP_ARM[$x]="batch_test P != A (IDENT_GATE=$IDENT_GATE)"; done
                log "IDENT gate: arms P PT PTD PT4 are skipped: batch_test P != A"; fi
        fi
        if any_arm PT PTD PT4; then
            r_pta=$(bt_same A PT); c_pta=$?; r_ptp=$(bt_same P PT); c_ptp=$?
            log "    [IDENT bt PT vs A] ==> $r_pta $(vd $c_pta)"; log "    [IDENT bt PT vs P] ==> $r_ptp $(vd $c_ptp)"
            if [ "$IDENT_GATE" != off ] && { [ $c_pta = 1 ] || [ $c_ptp = 1 ]; }; then
                for x in PT PTD PT4; do [ -n "${SKIP_ARM[$x]:-}" ] || SKIP_ARM[$x]="batch_test PT != A or P (IDENT_GATE=$IDENT_GATE)"; done
                log "IDENT gate: arms PT PTD PT4 are skipped: batch_test PT != A or != P"; fi
        fi
        if in_arms PTX; then r_x=$(bt_same A PTX); c_x=$?
            log "    [IDENT bt PTX vs A] ==> $r_x $([ $c_x = 1 ] && echo 'DIFF-expected' || { [ $c_x = 0 ] && echo 'identical (not expected)' || echo SKIP; }) INFO (non-exact arm, never gated)"; fi
    fi
    old_on && pin_bin port > /dev/null
    if old_on && [ -z "${SKIP_BIN[port]:-}" ] && [ "$ABORT_ALL" = 0 ] && cfg=$(mkrun H et-H); then
        INFO_STEP=1     # (INFO tests: a crash here does not count for "two crashes in a row"; the plan means arms)
        eng_test interleave 900 "$cfg" "$PY" -I -B "$PWT/tools/batch_interleave_test.py" --exe "$(cfgget "$cfg" exe)" --config "$cfg" \
            --extra "$BT_EXTRA --no-prefill-borrow" --dump "$OUT/interleave.json"
        log "    [interleave] ==> $([ "$STEP_RC" = 0 ] && echo PASS || echo "FAIL (rc $STEP_RC)") $(grep -aE 'PASS|FAIL' "$OUT/interleave.txt" | tail -1 | cut -c1-140)"
        for mv in 1 0; do [ "$ABORT_ALL" = 1 ] && break
            c2=$OUT/cfg/run-bgc-move$mv.json
            python3 -c "import json,sys; c=json.load(open(sys.argv[1])); c['env']['STRATA_SLOT_CKPT_MOVE']=sys.argv[2]; c['log']=sys.argv[3]; json.dump(c,open(sys.argv[4],'w'),indent=1)" "$cfg" "$mv" "$OUT/engine-bgc-move$mv.log" "$c2"
            eng_test "bgc-move$mv" 900 "$c2" env STRATA_EXTRA_ARGS="" "$PY" -I -B -u "$PWT/tools/batch_groups_cache_test.py" "$c2" "$OUT/bgc-move$mv.json"
            log "    [bgc-move$mv] ==> $(grep -E '^(PASS|FAIL)$' "$OUT/bgc-move$mv.txt" | tail -1) (INFO: the 0.1.45 window found this test's FAIL a counting artifact) $(grep '^requests' "$OUT/bgc-move$mv.txt" | cut -c1-120)"; done
        log "    [bgc MOVE 1 vs 0] ==> tokens per round identical: $(python3 -c "import json,sys
try: a,b=json.load(open(sys.argv[1])),json.load(open(sys.argv[2])); print([x.get('answers')==y.get('answers') for x,y in zip(a,b)])
except Exception as e: print('missing', e)" "$OUT/bgc-move1.json" "$OUT/bgc-move0.json") INFO"
        [ "$ABORT_ALL" = 0 ] && eng_test cancel_park 600 "$cfg" env STRATA_EXTRA_ARGS="" "$PY" -I -B "$PWT/tools/cancel_park_test.py" "$cfg"
        log "    [cancel_park] ==> $(grep -E '^(PASS|FAIL)' "$OUT/cancel_park.txt" | tail -1) INFO"
        INFO_STEP=0
    fi
    # INFO: 212i's cache paths with their byte checks (a capture beside the slots / a slot's capture against a plain one;
    # the engine stops with ERR on a mismatch), production's SLOT_CKPT_MOVE=1
    in_arms I && pin_bin int > /dev/null
    if in_arms I && [ -z "${SKIP_BIN[int]:-}" ] && [ "$ABORT_ALL" = 0 ] && cfg=$(mkrun I et-I); then
        INFO_STEP=1; ET_REPO=$IWT
        c2=$OUT/cfg/run-bgc-I.json
        python3 -c "import json,sys; c=json.load(open(sys.argv[1])); c['env'].update(STRATA_PARK_CAPTURE_VERIFY='1', STRATA_SLOT_PARK_VERIFY='1'); c['log']=sys.argv[2]; json.dump(c,open(sys.argv[3],'w'),indent=1)" "$cfg" "$OUT/engine-bgc-I.log" "$c2"
        eng_test bgc-I 900 "$c2" env STRATA_EXTRA_ARGS="" "$PY" -I -B -u "$IWT/tools/batch_groups_cache_test.py" "$c2" "$OUT/bgc-I.json"
        log "    [bgc-I verify] ==> $(grep -E '^(PASS|FAIL)$' "$OUT/bgc-I.txt" | tail -1) rc $STEP_RC; verified: $(grep -a -c 'STRATA_PARK_CAPTURE_VERIFY: [0-9]* tokens, [0-9]* bytes identical' "$OUT/engine-bgc-I.log" 2>/dev/null) park captures, $(grep -a -c 'STRATA_SLOT_PARK_VERIFY: slot [0-9]*, [0-9]* tokens, [0-9]* bytes identical' "$OUT/engine-bgc-I.log" 2>/dev/null) slot captures identical; mismatches: $(grep -a -c -E 'VERIFY: (the capture beside the slots differs|slot [0-9]+.s capture \(stage|slot [0-9]+ was written before)' "$OUT/engine-bgc-I.log" 2>/dev/null) INFO"
        # (tools/cache_reuse_check.py is not in 212i as of 98001378; should it appear, it is not run: its arguments were not checked)
        [ -f "$IWT/tools/cache_reuse_check.py" ] && log "    [cache_reuse-I] ==> SKIP (tools/cache_reuse_check.py appeared in $IWT; not wired up tonight)" \
            || log "    [cache_reuse-I] ==> SKIP (no tools/cache_reuse_check.py in $IWT)"
        ET_REPO=""; INFO_STEP=0
    fi
    # INFO: PT's engine tests on 212p's tools (as H's / I's: interleave, groups_cache with the VERIFY byte checks,
    # cancel_park) + P9's batch_overlap_test (every admission read beside the decoding groups = its solo tokens), all
    # with the lanes, the solo pipeline and the overlap on (PT's env).  Not counted as crashes in a row
    pin_bin pipe > /dev/null
    if in_arms PT && [ -z "${SKIP_BIN[pipe]:-}" ] && [ -z "${SKIP_ARM[PT]:-}" ] && [ "$ABORT_ALL" = 0 ] && cfg=$(mkrun PT et-PT); then
        INFO_STEP=1; ET_REPO=$PPT
        eng_test interleave-PT 900 "$cfg" "$PY" -I -B "$PPT/tools/batch_interleave_test.py" --exe "$(cfgget "$cfg" exe)" --config "$cfg" \
            --extra "$BT_EXTRA --no-prefill-borrow" --dump "$OUT/interleave-PT.json"
        log "    [interleave PT] ==> $([ "$STEP_RC" = 0 ] && echo PASS || echo "FAIL (rc $STEP_RC)") $(grep -aE 'PASS|FAIL' "$OUT/interleave-PT.txt" | tail -1 | cut -c1-140) INFO"
        [ "$ABORT_ALL" = 0 ] && eng_test overlap-PT 900 "$cfg" "$PY" -I -B "$PPT/tools/batch_overlap_test.py" --exe "$(cfgget "$cfg" exe)" --config "$cfg" \
            --extra "$BT_EXTRA --no-prefill-borrow" --dump "$OUT/overlap-PT.json"
        log "    [batch_overlap PT] ==> $([ "$STEP_RC" = 0 ] && echo PASS || echo "FAIL (rc $STEP_RC)") $(grep -aE 'PASS|FAIL' "$OUT/overlap-PT.txt" 2>/dev/null | tail -1 | cut -c1-140); overlapped reads: $(grep -a -c 'overlapped read of [0-9]* tokens' "$OUT/engine-overlap-PT.log" 2>/dev/null) INFO"
        c2=$OUT/cfg/run-bgc-PT.json
        python3 -c "import json,sys; c=json.load(open(sys.argv[1])); c['env'].update(STRATA_PARK_CAPTURE_VERIFY='1', STRATA_SLOT_PARK_VERIFY='1'); c['log']=sys.argv[2]; json.dump(c,open(sys.argv[3],'w'),indent=1)" "$cfg" "$OUT/engine-bgc-PT.log" "$c2"
        [ "$ABORT_ALL" = 0 ] && eng_test bgc-PT 900 "$c2" env STRATA_EXTRA_ARGS="" "$PY" -I -B -u "$PPT/tools/batch_groups_cache_test.py" "$c2" "$OUT/bgc-PT.json"
        log "    [bgc-PT verify] ==> $(grep -E '^(PASS|FAIL)$' "$OUT/bgc-PT.txt" 2>/dev/null | tail -1) rc $STEP_RC; verified: $(grep -a -c 'STRATA_PARK_CAPTURE_VERIFY: [0-9]* tokens, [0-9]* bytes identical' "$OUT/engine-bgc-PT.log" 2>/dev/null) park captures, $(grep -a -c 'STRATA_SLOT_PARK_VERIFY: slot [0-9]*, [0-9]* tokens, [0-9]* bytes identical' "$OUT/engine-bgc-PT.log" 2>/dev/null) slot captures identical; mismatches: $(grep -a -c -E 'VERIFY: (the capture beside the slots differs|slot [0-9]+.s capture \(stage|slot [0-9]+ was written before)' "$OUT/engine-bgc-PT.log" 2>/dev/null) INFO (FAIL = the known counting artifact when mismatches are 0)"
        [ "$ABORT_ALL" = 0 ] && eng_test cancel_park-PT 600 "$cfg" env STRATA_EXTRA_ARGS="" "$PY" -I -B "$PPT/tools/cancel_park_test.py" "$cfg"
        log "    [cancel_park PT] ==> $(grep -E '^(PASS|FAIL)' "$OUT/cancel_park-PT.txt" 2>/dev/null | tail -1) INFO"
        ET_REPO=""; INFO_STEP=0
    fi
else log "identity: SKIP (IDENTITY=$IDENTITY)"; fi
# after the gates: with exact arms listed but none left (only A and the INFO arm PTX), the outage buys nothing to decide
if [ "$ABORT_ALL" = 0 ] && [ -n "$(for a in $ARMS; do [ "$a" = A ] || [ "$a" = PTX ] || echo "$a"; done)" ] && [ -z "$(runnable PTX)" ]; then
    log "ABORT: no arm besides A and the INFO arm PTX is left after the gates ($(for a in $ARMS; do [ "$a" = A ] || printf '%s: %s; ' "$a" "${SKIP_ARM[$a]:-${SKIP_BIN[${ARM_BIN[$a]:-}]:-}}"; done)) - production is restored"
    exit 1; fi

# ---- phase 5: performance arms
run_arm() { local a=$1 lab=${2:-$1}
    [ "$ABORT_ALL" = 1 ] && return 1
    [ "$(left)" -lt "$ARM_EST" ] && { log "arm $lab: SKIP (needs ~$ARM_EST s, $(left) s left before DEADLINE)"; echo "arm=$a" > "$OUT/arm-$lab.st"; echo "status=skip" >> "$OUT/arm-$lab.st"; echo "why=deadline" >> "$OUT/arm-$lab.st"; return 1; }
    local why=${SKIP_ARM[$a]:-${SKIP_BIN[${ARM_BIN[$a]}]:-}}
    [ -n "$why" ] && { log "arm $lab: SKIP ($why)"; printf 'arm=%s\nstatus=skip\nwhy=%s\n' "$a" "$why" > "$OUT/arm-$lab.st"; return 1; }
    arm_run "$a" "$lab"; }
agg16() { python3 - "$OUT/sweep.jsonl" "w212-$1-short" <<'PYEOF'
import json, sys
v = None
try:
    for ln in open(sys.argv[1]):
        r = json.loads(ln)
        if r.get("label") == sys.argv[2] and r.get("users") == 16:
            v = r.get("agg_out_tps")
except OSError:
    pass
print(v if v is not None else "")
PYEOF
}
for a in $ARMS; do
    [ "$ABORT_ALL" = 1 ] && break
    [ "$(left)" -lt "$ARM_EST" ] && { log "DEADLINE: arm $a and the rest are not started"; break; }
    if [ "$a" = H3 ]; then
        h=$(agg16 H); h2=$(agg16 H2); pick=H3
        [ -n "$h2" ] && { [ -z "$h" ] || awk -v x="$h" -v y="$h2" 'BEGIN { exit !(y > x) }'; } && pick=H3d
        log "H3: H short agg 16 = ${h:--}, H2 = ${h2:--}: H3 runs as $pick"; run_arm "$pick" H3
    else run_arm "$a"; fi
done
# r2: the close pairs (|delta| < 3 % on the pair's metric) while time is left
if [ "$SPARE" = 1 ] && [ "$ABORT_ALL" = 0 ]; then
    [ "$R2_FIRST" = auto ] && { pipe_on && R2_FIRST="A PT P" || R2_FIRST=""; }
    R2_FIRST_RUN=""; for a in $R2_FIRST; do in_arms "$a" && grep -qx 'status=ok' "$OUT/arm-$a.st" 2>/dev/null && R2_FIRST_RUN="$R2_FIRST_RUN $a"; done
    R2=$(python3 - "$OUT" "$R2_FIRST_RUN" <<'PYEOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1]); rows = {}
first = sys.argv[2].split()   # R2_FIRST's arms that ran ok: always repeated, first
try:
    for ln in open(out / "sweep.jsonl"):
        r = json.loads(ln); rows[(r["label"], r["users"])] = r
except OSError:
    pass
def v(a, kind, u, k):
    r = rows.get((f"w212-{a}-{kind}", u)); return None if r is None else r.get(k)
cand = []
for x, y, kind, u, k in (("B", "C", "short", 16, "agg_out_tps"), ("C", "D", "long", 16, "agg_out_tps"),
                          ("H", "H2", "short", 16, "agg_out_tps"), ("A", "F", "short", 1, "dec_tps_p50"),
                          ("A", "G", "short", 2, "dec_tps_p50"), ("A", "G", "short", 4, "dec_tps_p50"), ("A", "H3", "short", 16, "agg_out_tps"),
                          ("P", "PT", "short", 16, "agg_out_tps"), ("PT", "PTD", "short", 16, "agg_out_tps"),
                          ("PT", "PT4", "long", 16, "agg_out_tps")):
    p, q = v(x, kind, u, k), v(y, kind, u, k)
    if p and q and abs(q / p - 1) < 0.03:
        cand.append((abs(q / p - 1), x, y))
seen = list(first)
for _, x, y in sorted(cand):
    for a in (x, y):
        if a not in seen:
            seen.append(a)
print(" ".join(seen))
PYEOF
)
    log "r2 runs: ${R2:-none} (first: ${R2_FIRST_RUN:-none} of R2_FIRST=$R2_FIRST, then the close pairs)"
    for a in $R2; do [ "$ABORT_ALL" = 1 ] && break; [ "$(left)" -lt "$ARM_EST" ] && break
        aa=$a; [ "$a" = H3 ] && aa=$(sed -n 's/^arm=//p' "$OUT/arm-H3.st" | head -1)
        run_arm "$aa" "$a-r2"; done
fi
log "arms done ($(left) s before DEADLINE); restoring production"
exit 0                                           # -> EXIT trap: production back, then the summary
