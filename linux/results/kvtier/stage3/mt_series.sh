#!/bin/bash
# the forced-spill series: 48 users x 4 turns, doc 10000, RAM tier 6 GiB (tier off = re-read / on / on + verify),
# and the never-evicting 128 GiB / 128-slot reference (reply hashes)
cd /media/raid0/StrataQ38FN/linux/results/kvtier/stage3
S="--conversation-cache-mib 6144"
./mt.sh m6off  mt-off.json   48 4 10000 $S > series-m6off.log 2>&1
./mt.sh m6tier mt-tier.json  48 4 10000 $S > series-m6tier.log 2>&1
./mt.sh m6tierv mt-tierv.json 48 4 10000 $S > series-m6tierv.log 2>&1
./mt.sh m128off mt-off.json  48 4 10000 --conversation-cache-mib 131072 --conversation-cache-slots 128 > series-m128off.log 2>&1
echo SERIES DONE > series-done.log
