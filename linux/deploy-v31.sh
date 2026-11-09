#!/usr/bin/env bash
# Production: Unsloth Dynamic v3.1 UD-IQ4_XS (window v31, 2026-11-09, arm VR) on production's Q8k engine, server and env
# (deploy-212q.sh): config T16-212q-v31-prod262r-vision-cpu.json = T16-212q-prod262r-vision-cpu.json with the v3.1 files
# (--pack Strata-data/packs/unsloth-ud-iq4_xs-v31, --native the 4-shard GGUF, tokenizer, vision.model), layer_split
# 13,25,36 and STRATA_SPLIT_OWN_RESERVE_MIB=1024: with v3.1's experts (+0.86 GiB) production's 12,25,37 made CUDA1/CUDA2 lend
# expert cache slots to the prompt path (CPU experts, no overlapped read: multiturn 171 s vs ~91 s); 13,25,36 alone still
# left CUDA3 short by ~0.2 GiB of the default 1.5 GiB reserve.  model_name stays production's (G3 / Genesis send it).
#   ./deploy-v31.sh               # switch production.json, restart, check the engine's placement and the 6 request shapes;
#                                 # rolls back to v3.0 by itself when v3.1 does not load, an expert is not in VRAM, a card
#                                 # lends slots to the prompt path (seen at the smoke's first reads) or the smoke fails
#   ./deploy-v31.sh --rollback    # back to T16-212q-prod262r-vision-cpu.json (v3.0, Q8k)
set -euo pipefail
R=/media/raid0/StrataQ38FN; L=$R/linux; C=$L/configs; URL=http://127.0.0.1:8090
BASE=T16-212q-prod262r-vision-cpu.json; TARGET=T16-212q-v31-prod262r-vision-cpu.json
V31_PACK=$R/Strata-data/packs/unsloth-ud-iq4_xs-v31
V31_GGUF=/media/raid0/Models/Strata-GGUF/unsloth-UD-IQ4_XS-v31/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf
SPLIT=13,25,36; RESERVE=1024
case "${1:-}" in
  --rollback) NEW=$BASE ;;
  "") NEW=$TARGET
      [ -s "$V31_PACK/native_experts.txt" ] && [ -s "$V31_GGUF.done" ] || { echo "v3.1 pack or GGUF missing"; exit 1; }
      python3 - "$C/$BASE" "$C/$TARGET" "$V31_PACK" "$V31_GGUF" "$SPLIT" "$RESERVE" <<'PY'
import json, sys
src, dst, pack, gguf, split, reserve = sys.argv[1:7]
c = json.load(open(src)); a = c["args"]
old_pack = a[a.index("--pack") + 1]; old_gguf = a[a.index("--native") + 1]
a[a.index("--pack") + 1] = pack; a[a.index("--native") + 1] = gguf
assert c["tokenizer"].startswith(old_pack + "/"); c["tokenizer"] = pack + c["tokenizer"][len(old_pack):]
if c.get("vision", {}).get("model") == old_gguf: c["vision"]["model"] = gguf
c["layer_split"] = split; c["env"]["STRATA_SPLIT_OWN_RESERVE_MIB"] = reserve
assert not any(x == old_pack or x.startswith(old_pack + "/") or x == old_gguf for x in a)
json.dump(c, open(dst, "w"), indent=1)
PY
      ;;
  *) echo "usage: $0 [--rollback]"; exit 2 ;;
esac
# never beside a GPU window (the window's own restore would leave production on whatever this start placed)
pgrep -f 'window-v31\.sh|window-212\.sh|window-borrow\.sh' >/dev/null && { echo "a window script is running: not deploying"; exit 1; }
for u in strata-bench.scope v31-restore.timer v31-guard.service; do
  systemctl --user is-active -q "$u" && { echo "$u is active (a window?): not deploying"; exit 1; }; done
start() {   # start <config>: relink, restart, wait for loaded (0) or died / 900 s (1)
  ln -sfn "$1" $C/production.json
  systemctl --user restart strata.service; local t0; t0=$(date +%s)
  until curl -s -m 3 $URL/health | grep -q '"loaded": *true'; do
    systemctl --user is-active -q strata.service || { echo "died"; journalctl --user -u strata.service -n 30 --no-pager; return 1; }
    [ $(( $(date +%s) - t0 )) -gt 900 ] && { echo "timeout"; return 1; }; sleep 5; done
  echo "loaded after $(( $(date +%s) - t0 )) s"; }
rollback() { echo "ROLLING BACK to $BASE: $1"; start "$BASE" || echo "ROLLBACK START FAILED - check strata.service now"; exit 1; }
echo "production.json: $(readlink $C/production.json) -> $NEW"
if ! start "$NEW"; then [ "$NEW" = "$BASE" ] && exit 1; rollback "v3.1 did not load"; fi
[ "$NEW" = "$BASE" ] && exit 0
LOG=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['log'])" "$C/production.json")
n=$(grep -an 'engine started' "$LOG" | tail -1 | cut -d: -f1)
tail -n +"$n" "$LOG" | grep -aE 'native pack:|layer split: CUDA[0-9] runs|every expert is in the GPU caches' | cut -c1-160 || true
tail -n +"$n" "$LOG" | grep -aq 'every expert is in the GPU caches' || rollback "not every expert in the GPU caches"
src=0; "$R/Strata/.venv/bin/python" -I "$L/v31_smoke.py" --url "$URL" --model "$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['model_name'])" "$C/production.json")" --label "deploy-$NEW" || src=$?
tail -n +"$n" "$LOG" | grep -aE 'keeps its own prompt buffers|prompt path borrows' | sort | uniq -c | cut -c1-120 || true
[ "$src" = 0 ] || rollback "smoke FAIL"
tail -n +"$n" "$LOG" | grep -aq 'prompt path borrows [0-9]* of' && rollback "a card lends expert cache slots to the prompt path"
echo "v3.1 live: PASS"
