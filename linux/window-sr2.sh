#!/usr/bin/env bash
# confirm --short-read 32 vs 64 on production's binary + config (zero code): agent mix (bench_multiturn) ABBA x2 and
# 16 users short/long MIX=128 with reply identity.  Test servers 127.0.0.1:8091 only, own tier dirs, own log.
set -u
R=/media/raid0/StrataQ38FN; L=$R/linux; C=$L/configs; OUT=$L/results/sr2; mkdir -p "$OUT"; URL=http://127.0.0.1:8090
PY=$R/Strata/.venv/bin/python; MT_ARGS="--users 16 --turns 4 --doc 2000 --add 8 --max-tok 192 --think 0.5"
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/window.log"; }
free_gpus() { for i in $(seq 60); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && return 0; sleep 2; done; return 1; }
start_prod() { log "starting production (production.json -> $(readlink $C/production.json))"; systemctl --user start strata.service; local t0; t0=$(date +%s)
    until curl -s -m 3 $URL/health | grep -q '"loaded": *true'; do [ $(( $(date +%s) - t0 )) -gt 900 ] && break; sleep 5; done
    log "production up after $(( $(date +%s) - t0 )) s"; }
STEP=
abort() { trap '' INT TERM HUP; log aborted; [ -n "$STEP" ] && kill -INT -- -"$STEP" 2>/dev/null; sleep 5; systemctl --user stop strata-bench.scope 2>/dev/null; free_gpus; start_prod; exit 1; }
trap 'echo "aborted before production was stopped"; exit 1' INT TERM HUP
arm() { local n=$1 cfg=$C/$1.json k=$2
  ( cd $L && export SERVER_DIR=$R/Strata-143-prod PORT=8091 BIND=127.0.0.1 KEEP=1; ./run_strata_variant.sh "sr2-$k-$n-short" $cfg 16 --rounds 2 --max-tok 512 --prompt-tok 100 --temperature 0 --max-tok-min 128 2>&1 | tail -3
    ATTACH=1 ./run_strata_variant.sh "sr2-$k-$n-long" $cfg 16 --rounds 2 --max-tok 512 --temperature 0 --warmup 0 --max-tok-min 128 2>&1 | tail -2
    timeout -s INT 600 $PY -I bench_multiturn.py --url http://127.0.0.1:8091 --label "sr2-$k-$n-mt" $MT_ARGS --temperature 0 --out $OUT/multiturn.jsonl 2>&1 | tail -6 ) >> $OUT/$k-$n.txt 2>&1
  systemctl --user stop strata-bench.scope 2>/dev/null; free_gpus; }
log "stopping strata.service"; trap abort INT TERM HUP; systemctl --user stop strata.service; free_gpus
k=0; for n in SR2-64 SR2-32 SR2-32 SR2-64; do k=$((k+1)); log "arm $k $n: start"; timeout -s INT 1500 bash -c "$(declare -f arm log free_gpus); R=$R L=$L C=$C OUT=$OUT PY=$PY MT_ARGS='$MT_ARGS'; arm $n $k" & STEP=$!; wait $STEP; STEP=; log "arm $k $n: done"; tail -1 $OUT/$k-$n.txt | tee -a $OUT/window.log; done
trap '' INT TERM HUP
start_prod; log "window done"
