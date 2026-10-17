#!/usr/bin/env bash
# Window (~40-45 min, production down): STRATA_ADMIT_BY_NEW=1 (server branch ll6-admit, Strata-ll6-admit @9db053af)
# against the multiturn follow-up TTFT regression under STRATA_PIPE_OVERLAP(_KVS) (results/mt-regression-2026-10-17.md,
# TESTLOG #86, labels mtr-*).  Same engine and config in A and E; only the HTTP server differs:
#   A  = production minus vision (configs/MTR-A.json: overlap + KVS on), server Strata-143-prod (= production today)
#   E  = A's engine/config, server Strata-ll6-admit with STRATA_ADMIT_BY_NEW=1 (max-wait default 45 s in that mode)
#   E60= optional arm E' (E + STRATA_ADMIT_MAX_WAIT_S=60), only if listed in ARMS
#   B  = overlap off (configs/MTR-B.json), server Strata-143-prod: the reference without the regression
# Every arm gets the production drop-in env STRATA_ADMIT_SHORT_FIRST=8192 STRATA_PREFIX_DIAG=1.
# Default order A E E A B (ARMS="A E E A B"; e.g. ARMS="A E E A B E60" adds E').  Per run: a fresh server in the
# strata-bench scope on 127.0.0.1:8091 with its OWN, EMPTY tier dir /media/raid0/strata-kvspill-bench/admit/<label>
# (STRATA_CONV_TIER_WARM=1 would otherwise preload the previous run's conversations) and its own engine log, then
#   bench_multiturn 16 users x 3 turns --temperature 0, bench_agentic (greedy by default),
#   bench_concurrency 16 users short (100-token prompts) then long (~1.2K) greedy, ab_quick-style.
# Exactness (order only => greedy replies identical): multiturn reply sha256 per (uid, turn) and the short/long reply
# sha256 per uid (compare_replies.py) of every run against the first A run; PASS/FAIL for A and E runs, B informational
# (overlap off is a different engine path; in TESTLOG #86 its multiturn replies matched A's too).  bench_agentic's
# chat user is timing dependent (A and B differed in output tokens on 17.10), so it is compared on speed only.
# Waits until production reports loaded before stopping it; starts production again on exit, abort and Ctrl-C and waits
# for it to load.  Never writes production.json, the production configs or any binary (the run configs are generated
# into results/admit/).  Results: results/admit/ (window.log, summary.txt, *-server.out, engine-*.log, *.jsonl).
#   run: cd /media/raid0/StrataQ38FN/linux && ./window-admit.sh 2>&1 | tee results/admit-window.out
set -uo pipefail
cd "$(dirname "$0")" || exit 1
# nothing STRATA_* from the calling shell reaches a bench server or its engine (the engine inherits the server's
# environment, and the config's "env" only overrides the keys it names): every arm's settings come from this script
for v in $(compgen -e | grep '^STRATA_'); do unset "$v"; done
R=/media/raid0/StrataQ38FN; L=$R/linux; C=$L/configs; OUT=$L/results/admit; mkdir -p "$OUT"
PY=$R/Strata/.venv/bin/python; STRATA=$R/Strata
SRV_PROD=$R/Strata-143-prod; SRV_NEW=$R/Strata-ll6-admit
SP=8091; BIND=127.0.0.1; SURL=http://127.0.0.1:$SP; PROD_URL=http://127.0.0.1:8090
TIER=/media/raid0/strata-kvspill-bench/admit
ARMS=${ARMS:-"A E E A B"}
STOPPED=0; BGPID=""
# fd 3 = the script's own stdout: log lines from the signal trap and back() (which can run inside a bench call whose
# stdout goes to that bench's .txt) still reach the terminal
exec 3>&1
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/window.log" >&3; }
gpus_free() { for i in $(seq 90); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && return 0; sleep 2; done; return 1; }
prod_loaded() { for i in $(seq "${1:-120}"); do curl -s -m 3 $PROD_URL/health | grep -q '"loaded": *true' && return 0; sleep 5; done; return 1; }
stop_bench() { systemctl --user stop strata-bench.scope 2>/dev/null; sleep 3; gpus_free || log "GPUs busy after the bench server stopped"; }
# a bench client runs in the background and the script waits for it, so INT/TERM/HUP act at once (a foreground
# `timeout` sits in its own process group: Ctrl-C would not reach it and bash would defer the trap until it ended)
bgw() { "$@" 3>&- & BGPID=$!; wait $BGPID; local rc=$?; BGPID=""; return $rc; }
back() { trap - EXIT INT TERM HUP; trap '' PIPE     # (tee may be gone after Ctrl-C: never die on a write)
  [ -n "$BGPID" ] && kill "$BGPID" 2>/dev/null
  stop_bench
  if [ "$STOPPED" = 1 ]; then
    systemctl --user start strata.service; log "production start requested (production.json -> $(readlink $C/production.json))"
    if prod_loaded 180; then log "production loaded again"; else log "PRODUCTION DID NOT REPORT LOADED - check strata.service now"; fi
  fi; }

# --- before the window (no GPU): checks and the run configs
for f in "$C/MTR-A.json" "$C/MTR-B.json" "$SRV_PROD/serve/server.py" "$SRV_NEW/serve/server.py" "$SRV_NEW/serve/admit_est.py" \
         bench_multiturn.py bench_agentic.py bench_concurrency.py compare_replies.py; do
  [ -f "$f" ] || { echo "missing: $f"; exit 1; }; done
grep -q 'STRATA_ADMIT_BY_NEW' "$SRV_NEW/serve/server.py" || { echo "$SRV_NEW has no STRATA_ADMIT_BY_NEW"; exit 1; }
for a in $ARMS; do case $a in A|E|E60|B) ;; *) echo "unknown arm $a (A E E60 B)"; exit 1 ;; esac; done
log "server A/B: $SRV_PROD @$(git -C $SRV_PROD rev-parse --short HEAD)$(git -C $SRV_PROD diff --quiet HEAD -- serve || echo +dirty);" \
    "server E: $SRV_NEW @$(git -C $SRV_NEW rev-parse --short HEAD)$(git -C $SRV_NEW diff --quiet HEAD -- serve || echo +dirty); arms: $ARMS"
mkcfg() { # <base.json> <out.json> <tier dir> <engine log>
  python3 -I - "$@" <<'PYEOF'
import json, sys
c = json.load(open(sys.argv[1]))
c["env"]["STRATA_CONV_TIER_DIR"] = sys.argv[3]
c["log"] = sys.argv[4]
json.dump(c, open(sys.argv[2], "w"), indent=1)
PYEOF
}

# the server's own process in the bench scope: its script path and STRATA_ADMIT_*/STRATA_PREFIX_DIAG environment
srv_ident() { local cg p
  cg=$(systemctl --user show -p ControlGroup --value strata-bench.scope 2>/dev/null)
  for p in $(cat "/sys/fs/cgroup$cg/cgroup.procs" 2>/dev/null); do
    tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q '/serve/server.py ' || continue
    echo "$(tr '\0' '\n' < /proc/$p/cmdline | sed -n 2p) $(tr '\0' '\n' < /proc/$p/environ | grep -E '^STRATA_(ADMIT_|PREFIX_DIAG)' | sort | tr '\n' ' ')"
    return 0; done; return 1; }
prod_idle() { curl -s -m 3 $PROD_URL/v1/load | python3 -I -c 'import json,sys; s=json.load(sys.stdin)["slots"]; sys.exit(0 if s["busy"] == 0 and s["waiting"] == 0 else 1)' 2>/dev/null; }

# --- the window
if systemctl --user is-active -q strata-bench.scope || curl -s -m 2 -o /dev/null $SURL/health; then
  echo "strata-bench.scope is active or port $SP answers - another bench is running; window NOT started"; exit 1; fi
log "waiting until production reports loaded"
prod_loaded 120 || { log "production not loaded after 10 min - window NOT started, production untouched"; exit 1; }
for i in $(seq 36); do prod_idle && break; [ "$i" = 1 ] && log "production busy - waiting up to 3 min for it to go idle"; sleep 5; done
trap back EXIT
trap 'trap "" PIPE; log "aborted by signal"; exit 1' INT TERM HUP
log "window start: stopping strata.service"; STOPPED=1; systemctl --user stop strata.service
gpus_free || { log "GPUs not free"; exit 1; }

declare -A REP
run_arm() { local arm=$1 base srv extra
  REP[$arm]=$(( ${REP[$arm]:-0} + 1 )); local lab=admit-$arm-${REP[$arm]}
  case $arm in
    A) base=MTR-A; srv=$SRV_PROD; extra="" ;;
    B) base=MTR-B; srv=$SRV_PROD; extra="" ;;
    E) base=MTR-A; srv=$SRV_NEW; extra="STRATA_ADMIT_BY_NEW=1" ;;
    E60) base=MTR-A; srv=$SRV_NEW; extra="STRATA_ADMIT_BY_NEW=1 STRATA_ADMIT_MAX_WAIT_S=60" ;;
  esac
  local want="$srv/serve/server.py $(printf '%s\n' STRATA_ADMIT_SHORT_FIRST=8192 STRATA_PREFIX_DIAG=1 $extra | sort | tr '\n' ' ')"
  local cfg=$OUT/config-$lab.json elog=$OUT/engine-$lab.log slog=$OUT/$lab-server.out td=$TIER/$lab
  rm -rf "$td"; mkdir -p "$td"; : > "$elog"
  mkcfg "$C/$base.json" "$cfg" "$td" "$elog" || { log "$lab: config failed"; return 1; }
  systemctl --user stop strata-bench.scope 2>/dev/null; systemctl --user reset-failed strata-bench.scope 2>/dev/null
  curl -s -m 2 -o /dev/null $SURL/health && { log "$lab: port $SP still answers after the scope stopped - run skipped"; return 1; }
  # env -u: nothing from the calling shell may change the admission order of an arm
  ( cd "$srv" && env -u STRATA_ADMIT_BY_NEW -u STRATA_ADMIT_MAX_WAIT_S -u STRATA_ADMIT_EST_BLOCKS \
      STRATA_ADMIT_SHORT_FIRST=8192 STRATA_PREFIX_DIAG=1 $extra \
      systemd-run --user --scope --unit=strata-bench -p MemoryMax=224G -p MemorySwapMax=0 \
      $STRATA/.venv/bin/python "$srv/serve/server.py" --engine strata --config "$cfg" --port $SP --host $BIND \
      > "$slog" 2>&1 < /dev/null & )
  local t0; t0=$(date +%s)
  until curl -s -m 3 $SURL/health | grep -q '"loaded": *true'; do
    [ $(( $(date +%s) - t0 )) -gt 10 ] && ! systemctl --user is-active -q strata-bench.scope && { log "$lab: server died during start"; tail -5 "$slog" | tee -a "$OUT/window.log"; stop_bench; return 1; }
    [ $(( $(date +%s) - t0 )) -gt 600 ] && { log "$lab start timeout"; stop_bench; return 1; }; sleep 5; done
  log "$lab ready after $(( $(date +%s) - t0 )) s (server $(basename $srv), config $base, env $extra)"
  local got; got=$(srv_ident)
  if [ "$got" != "$want" ]; then
    log "$lab ARM CHECK FAILED - run skipped: want [$want] got [$got]"; stop_bench; return 1; fi
  log "$lab arm check ok: $got"
  curl -s -m 5 $SURL/v1/load > "$OUT/$lab-load0.json"
  wc -l < "$slog" > "$OUT/$lab.mark-mt"
  bgw timeout 900 $PY -I bench_multiturn.py --url $SURL --users 16 --turns 3 --temperature 0 --label $lab-mt \
    --out "$OUT/multiturn.jsonl" > "$OUT/$lab-mt.txt" 2>&1; log "$lab multiturn: $(tail -1 "$OUT/$lab-mt.txt")"
  wc -l < "$slog" > "$OUT/$lab.mark-agentic"
  bgw timeout 1500 $PY -I bench_agentic.py --url $SURL --label $lab-agentic --out "$OUT/agentic.jsonl" > "$OUT/$lab-agentic.txt" 2>&1
  log "$lab agentic: $(tail -1 "$OUT/$lab-agentic.txt" | cut -c1-110)"
  wc -l < "$slog" > "$OUT/$lab.mark-ident"
  bgw timeout 900 $PY -I bench_concurrency.py --url $SURL --label $lab-short --levels 16 --rounds 2 --max-tok 512 --prompt-tok 100 \
    --temperature 0 --out "$OUT/sweep.jsonl" > "$OUT/$lab-short.txt" 2>&1
  bgw timeout 900 $PY -I bench_concurrency.py --url $SURL --label $lab-long --levels 16 --rounds 2 --max-tok 512 \
    --temperature 0 --warmup 0 --out "$OUT/sweep.jsonl" > "$OUT/$lab-long.txt" 2>&1
  log "$lab short/long done"
  wc -l < "$slog" > "$OUT/$lab.mark-end"
  stop_bench
  log "$lab engine: $(grep -c 'overlapped read of' "$elog") overlapped reads, $(grep -c 'BYIELD' "$elog") BYIELD mentions (give-way, if the engine logs it);" \
      "server: $(grep -c '\[strata\] admit: total' "$slog") admit lines, $(grep -c 'has waited' "$slog") waited-notes"
  rm -rf "$td"
  echo "$lab" >> "$OUT/runs.txt"; }

: > "$OUT/runs.txt"
for a in $ARMS; do run_arm "$a"; done
stop_bench
log "benches done; starting production"
back

# --- summary (no GPU)
"$PY" -I - "$OUT" <<'EOF' | tee "$OUT/summary.txt" | tee -a "$OUT/window.log"
import json, os, re, statistics as st, subprocess, sys
out = sys.argv[1]
runs = [l.strip() for l in open(f"{out}/runs.txt") if l.strip()]
arm = lambda lab: lab.split("-")[1]
mt = {}
for line in open(f"{out}/multiturn.jsonl") if os.path.exists(f"{out}/multiturn.jsonl") else []:
    r = json.loads(line); mt[r["label"][:-3]] = r
ag = {}
for line in open(f"{out}/agentic.jsonl") if os.path.exists(f"{out}/agentic.jsonl") else []:
    r = json.loads(line); ag[r["label"][:-8]] = r
sw = {}
for line in open(f"{out}/sweep.jsonl") if os.path.exists(f"{out}/sweep.jsonl") else []:
    r = json.loads(line); sw[r["label"]] = r

print("== multiturn 16 users x 3 turns, greedy: TTFT p50/p95 (s) per turn, follow-up p50, wall, output tok/s")
print(f"{'run':<12} {'turn1':>12} {'turn2':>12} {'turn3':>12} {'follow p50':>10} {'wall':>5} {'tok/s':>6}")
for lab in runs:
    r = mt.get(lab)
    if not r: print(f"{lab:<12} (no multiturn row)"); continue
    t = [f"{r[f'turn{k}']['ttft_p50']:.2f}/{r[f'turn{k}']['ttft_p95']:.2f}" for k in (1, 2, 3)]
    print(f"{lab:<12} {t[0]:>12} {t[1]:>12} {t[2]:>12} {r['follow_ttft_p50']:>10.2f} {r['wall_s']:>5.0f} {r['agg_out_tps']:>6.1f}")
print("per arm (mean over runs):")
for a in dict.fromkeys(arm(l) for l in runs):
    rs = [mt[l] for l in runs if arm(l) == a and l in mt]
    if not rs: continue
    m = lambda f: st.mean(f(r) for r in rs)
    print(f"  {a:<4} n={len(rs)} turn1 p50 {m(lambda r: r['turn1']['ttft_p50']):.2f}  turn2 p50/p95 "
          f"{m(lambda r: r['turn2']['ttft_p50']):.2f}/{m(lambda r: r['turn2']['ttft_p95']):.2f}  turn3 p50/p95 "
          f"{m(lambda r: r['turn3']['ttft_p50']):.2f}/{m(lambda r: r['turn3']['ttft_p95']):.2f}  follow p50 "
          f"{m(lambda r: r['follow_ttft_p50']):.2f}  wall {m(lambda r: r['wall_s']):.0f}  tok/s {m(lambda r: r['agg_out_tps']):.1f}")

print("\n== agentic (greedy; chat user timing dependent): TTFT p50/p95/max per role, wall, tok/s")
for lab in runs:
    r = ag.get(lab)
    if not r: print(f"{lab:<12} (no agentic row)"); continue
    roles = "  ".join(f"{k} {r[k]['ttft_p50']}/{r[k]['ttft_p95']}/{r[k]['ttft_max']}" for k in ("main", "sub", "chat") if k in r)
    print(f"{lab:<12} {roles}  wall {r['wall_s']:.0f}  {r['agg_out_tps']} tok/s")

print("\n== short/long 16 users (ab_quick style): agg tok/s, TTFT p50")
for lab in runs:
    print(f"{lab:<12} " + "  ".join(f"{k} {sw[f'{lab}-{k}']['agg_out_tps']:.1f} tok/s TTFT {sw[f'{lab}-{k}']['ttft_p50']:.2f}"
                                    if f"{lab}-{k}" in sw else f"{k} (no row)" for k in ("short", "long")))

print("\n== reply identity vs the first A run (greedy; order only => identical)")
refs = [l for l in runs if arm(l) == "A"]
fails = 0
if not refs:
    print("no A run - nothing to compare against")
else:
    ref = refs[0]
    for lab in runs:
        if lab == ref: continue
        must = arm(lab) != "B"
        parts = []
        a, b = mt.get(ref), mt.get(lab)
        if a and b:
            ra = {(u, t): (g, h) for u, t, g, h in a["replies"]}; rb = {(u, t): (g, h) for u, t, g, h in b["replies"]}
            diff = sorted(k for k in set(ra) | set(rb) if ra.get(k) != rb.get(k))
            parts.append(f"mt {len(ra) - len([k for k in diff if k in ra])}/{len(ra)} identical" + (f" (first differ uid/turn {diff[:4]})" if diff else ""))
            bad = bool(diff)
        else:
            parts.append("mt: missing"); bad = True
        for k in ("short", "long"):
            p = subprocess.run([sys.executable, "-I", "compare_replies.py", f"{out}/sweep.requests.jsonl", f"{ref}-{k}",
                                f"{lab}-{k}", "--users", "16", "--show", "0"], capture_output=True, text=True)
            hit = [x for x in p.stdout.splitlines() if "common uids" in x or x.startswith("no requests")]
            parts.append(f"{k}: {(hit[0].split(': ', 1)[-1] if hit else (p.stdout + p.stderr).strip()[-80:])}")
            bad |= p.returncode != 0
        verdict = ("FAIL" if bad else "PASS") if must else "(info)"
        fails += must and bad
        print(f"{lab:<12} vs {ref}: " + "; ".join(parts) + f"  {verdict}")
print(f"identity: {'PASS' if not fails else f'FAIL ({fails} run(s) differ)'}")

print("\n== admission diag (server log '[strata] admit:' lines, STRATA_ADMIT_BY_NEW runs only), per section")
rx = re.compile(r"\[strata\] admit: total (\d+), est\. new (\d+) \(cached ~(-?\d+)\), waited ([\d.]+) s, ([a-z, -]+), (\d+) waiting")
for lab in runs:
    try:
        lines = open(f"{out}/{lab}-server.out", errors="replace").read().splitlines()
        mk = [int(open(f"{out}/{lab}.mark-{m}").read()) for m in ("mt", "agentic", "ident", "end")]
    except (OSError, ValueError):
        print(f"{lab:<12} (no server log / marks)"); continue
    for sec, (lo, hi) in zip(("mt", "agentic", "short/long"), zip(mk, mk[1:])):
        ms = [m for m in (rx.search(x) for x in lines[lo:hi]) if m]
        if not ms:
            if arm(lab).startswith("E"): print(f"{lab:<12} {sec:<10} no admit lines")
            continue
        kinds = {}
        for m in ms: kinds[m.group(5)] = kinds.get(m.group(5), 0) + 1
        tot = [int(m.group(1)) for m in ms]; new = [int(m.group(2)) for m in ms]; w = [float(m.group(4)) for m in ms]
        big = [(t, n) for t, n in zip(tot, new) if t > 8192]
        print(f"{lab:<12} {sec:<10} {len(ms)} admits, {', '.join(f'{k} {v}' for k, v in sorted(kinds.items()))}; "
              f"total p50 {st.median(tot):.0f}, est.new p50 {st.median(new):.0f}; waited p50 {st.median(w):.1f} max {max(w):.1f} s; "
              f"total>8192 but est.new<=8192: {sum(1 for t, n in big if n <= 8192)}/{len(big)}")
    other = sum(1 for x in lines if "has waited" in x)
    if other: print(f"{lab:<12} 'has waited' notes: {other}")
EOF
log "window done (production was started again above); summary: $OUT/summary.txt"
