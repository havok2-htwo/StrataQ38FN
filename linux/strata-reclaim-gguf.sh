#!/bin/sh
# After Strata has loaded, push the GGUF expert pages out of the strata.service cgroup's page cache.
# With --resident-experts every expert is uploaded to VRAM at start and the mapping is never read again
# ("files 0 blobs", "0.0 MB of expert reads" per request), but the clean pages stay charged to the cgroup as
# active_file, which the conversation cache's RAM guard counts as used. memory.swap.max=0, so memory.reclaim can
# only take clean file pages: anon (parked conversations), shmem (pinned K/V) and the mlocked PLE stay.
# Run by strata-reclaim.service (oneshot, started --no-block from strata.service), never inside the start job, and
# by soak.sh for its strata-bench.scope. The cgroup is the running engine's (strata.service or strata-bench.scope).
KEEP=$((8 * 1024 * 1024 * 1024))     # leave 8 GiB of file LRU (binary, CUDA libs, pack, MTP)
STEP=$((8 * 1024 * 1024 * 1024))

loaded=0
for i in $(seq 1 180); do            # up to ~15 min for the load
  curl -sf -m 2 http://127.0.0.1:8090/health | grep -q '"loaded": *true' && { loaded=1; break; }
  sleep 5
done
[ "$loaded" = 1 ] || { echo "strata not loaded after 15 min; nothing reclaimed"; exit 0; }
sleep 30
CG=
for u in strata.service strata-bench.scope; do
  systemctl --user is-active -q $u && { CG=/sys/fs/cgroup$(systemctl --user show -P ControlGroup $u); break; }
done
[ -n "$CG" ] && [ -f "$CG/memory.reclaim" ] || { echo "neither strata.service nor strata-bench.scope is active; nothing reclaimed"; exit 0; }

file() { awk '/^(active|inactive)_file /{s+=$2} END{print s+0}' "$CG/memory.stat"; }
before=$(file)
for n in $(seq 1 16); do
  f=$(file); [ "$f" -le "$KEEP" ] && break
  amt=$((f - KEEP)); [ "$amt" -gt "$STEP" ] && amt=$STEP
  echo "$amt" > "$CG/memory.reclaim" 2>/dev/null   # EAGAIN when a pass falls short (MGLRU aging): retry
done
after=$(file)
echo "file LRU $((before >> 30)) -> $((after >> 30)) GiB;" \
     $(grep -E '^(workingset_refault_file|pgmajfault) ' "$CG/memory.stat" | tr '\n' ' ')
