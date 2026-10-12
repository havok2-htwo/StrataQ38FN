#!/usr/bin/env bash
# Quick A/B (~2-3 min GPU per arm incl. the ~30 s load): one server start per config and repetition, 16 users,
# greedy (temperature 0), short (100-token) then 1.2K prompts on the same server.  The prompts depend only on the
# request id and prompt kind, so every arm gets byte-identical requests and an exact change must return identical
# replies: the summary at the end checks that per uid against the first arm (compare_replies.py) next to the speed.
#
# Noise (REVIEW-2026-10-12, recomputed from sweep.jsonl at 16 users): pooled run-to-run sd 0.41 %, so one run per
# arm resolves only ~1.2 % (one-run A-B threshold 1.24 %; repeated T16-142 arms differed by -1.5 % / -1.3 %).  The
# old "+-0.3 %, one run is enough" (TESTLOG #41) held for the overlap-series configs only.  Rule: |delta| < ~1.3 %
# from a single run is "not resolved"; for effects below ~1.5-2 % run REPS=3 (alternating arms, ABABAB) or compare
# engine-side timing (the engine's stat lines in engine-<label>.run.log, rows/s), and always use REPS=3 to confirm a
# new default.  agg_out_tps_steady (sweep.jsonl) excludes the admission ramp and the closed-loop drain tail.
#
#   ab_quick.sh <tag> <config.json>...
# Env: SERVER_DIR, PORT, LEVELS=16, ROUNDS=2 as in run_strata_variant.sh;
#   REPS=n    n repetitions, arms alternate (A B A B ...); labels get -r<k>; the summary prints per-arm means and the
#             paired per-repetition deltas vs the first arm.  ORDER=abba reverses every second repetition (A B B A ..).
#   MIX=128   mixed reply lengths: ROUNDS lengths evenly spaced over 128..512, every user gets all of them in a rotated
#             order (equal work per user, no closed-loop tail), so the slots do not finish in lockstep.
cd "$(dirname "$0")" || exit 1
TAG=$1; shift
REPS=${REPS:-1}
PY=/media/raid0/StrataQ38FN/Strata/.venv/bin/python
for rep in $(seq 1 "$REPS"); do
  arms=("$@")
  if [ "${ORDER:-abab}" = "abba" ] && [ $(( rep % 2 )) -eq 0 ]; then
    arms=(); for (( k=$#; k>=1; k-- )); do arms+=("${!k}"); done
  fi
  for cfg in "${arms[@]}"; do
    n=$(basename "$cfg" .json)
    L=ab-$TAG-$n; [ "$REPS" -gt 1 ] && L=$L-r$rep
    KEEP=1 ./run_strata_variant.sh "$L-short" "$cfg" "${LEVELS:-16}" --rounds "${ROUNDS:-2}" --max-tok 512 --prompt-tok 100 \
      --temperature 0 ${MIX:+--max-tok-min $MIX} 2>&1 | tail -3
    ATTACH=1 KEEP=1 ./run_strata_variant.sh "$L-long" "$cfg" "${LEVELS:-16}" --rounds "${ROUNDS:-2}" --max-tok 512 \
      --temperature 0 --warmup 0 ${MIX:+--max-tok-min $MIX} 2>&1 | tail -2
    systemctl --user stop strata-bench.scope
  done
done

# summary: speed per arm (mean over repetitions), paired deltas vs the first arm, reply identity vs the first arm
"$PY" -I - "$TAG" "$REPS" "$@" <<'EOF'
import json, os, statistics as st, subprocess, sys
tag, reps, cfgs = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
arms = [os.path.basename(c)[:-5] if c.endswith(".json") else os.path.basename(c) for c in cfgs]
rows = {}
for line in open("results/sweep.jsonl"):
    r = json.loads(line)
    rows[(r["label"], r["users"])] = r            # a repeated label: the last row wins
def lab(arm, rep, kind):
    return f"ab-{tag}-{arm}" + (f"-r{rep}" if reps > 1 else "") + f"-{kind}"
levels = sorted({u for (l, u) in rows if l.startswith(f"ab-{tag}-")})
for kind in ("short", "long"):
    for u in levels:
        base = {}
        for arm in arms:
            v = [rows.get((lab(arm, k, kind), u)) for k in range(1, reps + 1)]
            if not all(v):
                continue
            tps = [x["agg_out_tps"] for x in v]
            std = [x.get("agg_out_tps_steady") or float("nan") for x in v]
            line = (f"{kind:>5} {u:>3} users {arm:<24} agg {st.mean(tps):7.1f}"
                    + (f" (sd {st.stdev(tps) / st.mean(tps) * 100:.2f} %)" if reps > 1 else "")
                    + f"  steady {st.mean(std):7.1f}  TTFT p50 {st.mean(x['ttft_p50'] for x in v):.2f}")
            if not base:
                base = {"tps": tps, "arm": arm}
            else:
                d = [(b / a - 1) * 100 for a, b in zip(base["tps"], tps)]
                line += f"  vs {base['arm']}: {st.mean(d):+.2f} %" + (f" (pairs {', '.join(f'{x:+.2f}' for x in d)})" if reps > 1 else "")
                for k in range(1, reps + 1):
                    p = subprocess.run([sys.executable, "-I", "compare_replies.py", "results/sweep.requests.jsonl",
                                        lab(base["arm"], k, kind), lab(arm, k, kind), "--users", str(u), "--show", "0"],
                                       capture_output=True, text=True)
                    line += f"  [r{k}: {p.stdout.split(' - ')[-1].strip() if ' - ' in p.stdout else p.stdout.strip()}]"
            print(line)
EOF
