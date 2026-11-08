#!/usr/bin/env bash
# Production: PT + STRATA_PIPE_OVERLAP_MIN (windows w212f/w212g, 2026-11-08, TESTLOG #101/#102): engine strata-212q =
# strata-212p + e55fa457 (a segment of fewer than MIN new tokens is read serially: a burst of short admissions overlapped
# one by one only admits the later ones later) and STRATA_PIPE_OVERLAP_PRIO=0; everything else as deploy-212pt.sh.
#   MIN=<tokens> deploy-212q.sh   (the measured winner; rollback: deploy-212pt.sh)
# deploy-212pt.sh's notes: arm PT of window w212c (2026-11-08, TESTLOG #99): upstream v0.2.12 + our series + P9-P11 + the sm_86 table.
# Engine strata-212p (g3-212-pipe @a099d295, sha256 74702164d1e5e37d) pinned as strata-212p-74702164; server Strata-143-prod
# checked out at a099d295 (v0.2.12 server + our serve patches); config T16-212pt-prod262r-vision-cpu.json = configs/T212-PT.json
# with production's exe/log/port/tier and sampling {temperature 0, presence_penalty 0}: the v0.2.12 server's model-card
# defaults (#2685) would sample plain clients (production was greedy for them) and add presence_penalty 1.5 with thinking
# off, which turns off the solo pipeline, lanes and merge-first; "answer_room": false keeps thinking replies as before.  Rollback: deploy-ll10b.sh (ll10b + server 22f4781f).
set -euo pipefail
R=/media/raid0/StrataQ38FN; URL=http://127.0.0.1:8090
MIN=${MIN:?MIN=<tokens> (STRATA_PIPE_OVERLAP_MIN)}; [[ $MIN =~ ^[0-9]+$ ]] || { echo "MIN must be a number"; exit 1; }
SERVER=a099d2953b6b70084d92da0f3ac7b015719a4d5e; SRC=$R/linux/results/w212g/bin/strata-212q-9743c131b36ed22b; EXE=$R/Strata/engine/strata-212q-9743c131
TARGET=T16-212q-prod262r-vision-cpu.json
[ -z "$(git -C $R/Strata-143-prod status --porcelain)" ] || { echo "Strata-143-prod has local changes, not touching it"; exit 1; }
git -C $R/Strata-143-prod cat-file -e "$SERVER^{commit}" || { echo "server commit $SERVER unknown"; exit 1; }
[ -x "$EXE" ] || install -m 755 "$SRC" "$EXE"
[ "$(sha256sum "$EXE" | cut -c1-16)" = 9743c131b36ed22b ] || { echo "engine sha mismatch"; exit 1; }
python3 - "$R/linux/configs/T212-PT.json" "$R/linux/configs/production.json" "$R/linux/configs/$TARGET" "$EXE" "$MIN" <<'PY'
import json, sys
src, prod, dst, exe, mn = sys.argv[1:6]
c = json.load(open(src)); p = json.load(open(prod))
c["exe"] = exe; c["log"] = p["log"]; c["port"] = p["port"]
c["env"]["STRATA_CONV_TIER_DIR"] = p["env"]["STRATA_CONV_TIER_DIR"]
c["env"]["STRATA_PIPE_OVERLAP_MIN"] = mn; c["env"]["STRATA_PIPE_OVERLAP_PRIO"] = "0"
c["sampling"] = {"temperature": 0, "presence_penalty": 0}
c["answer_room"] = False                     # 0.1.45 implicit thinking budget at max_tokens >= 1024: production (22f4781f) had none
bad = [v for v in c["env"].values() if "bench" in str(v) or "/results/" in str(v)]
assert not bad, bad
assert "STRATA_PARK_ASYNC" not in c["env"], "PARK_ASYNC is a different feature on v0.2.12"
json.dump(c, open(dst, "w"), indent=1)
PY
echo "production.json: $(readlink $R/linux/configs/production.json) -> $TARGET"
echo "server: $(git -C $R/Strata-143-prod rev-parse --short HEAD) -> ${SERVER:0:8}"
git -C $R/Strata-143-prod checkout -q --detach $SERVER
ln -sfn "$TARGET" $R/linux/configs/production.json
systemctl --user restart strata.service; t0=$(date +%s)
until curl -s -m 3 $URL/health | grep -q '"loaded": *true'; do
  systemctl --user is-active -q strata.service || { echo "died"; journalctl --user -u strata.service -n 30 --no-pager; exit 1; }
  [ $(( $(date +%s) - t0 )) -gt 900 ] && { echo "timeout"; exit 1; }; sleep 5; done
echo "loaded after $(( $(date +%s) - t0 )) s"
P=$(systemctl --user show -p MainPID --value strata.service)
for c in $(pgrep -P $P); do
  tr '\0' ' ' < /proc/$c/cmdline | awk '{print $1}'
  tr '\0' '\n' < /proc/$c/environ | grep -E '^(STRATA_PARK_BORROW|STRATA_CONV_MMAP|STRATA_GATE_EVICT|STRATA_PARK_CAPTURE|STRATA_SERVE_STOP_BOUND|STRATA_SM86_PERF|STRATA_ATTN_DAILY|STRATA_SOLO_PIPELINE|STRATA_SPEC_LANES|STRATA_PIPE_OVERLAP|STRATA_PIPE_OVERLAP_MIN|STRATA_PIPE_OVERLAP_PRIO)=' | tr '\n' ' '; echo
done
