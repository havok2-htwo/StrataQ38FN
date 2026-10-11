#!/usr/bin/env bash
# M4 of upstream/DESIGN-overlap.md: knob arms of the overlapped admission, 16/24 users, short + 1.2K prompts, 1 rep.
cd "$(dirname "$0")"
R=./run_strata_variant.sh
S="--rounds 2 --max-tok 512 --prompt-tok 100"; L="--rounds 2 --max-tok 512"
for cfg in ${ARMS:-T16-ovM4-prio0 T16-ovM4-sleep500 T16-ovM4-prio0-sleep500 T16-ovM4-conn32}; do
  for cls in short long; do
    a=$S; [ $cls = long ] && a=$L
    $R strata-$cfg-$cls configs/$cfg.json 16,24 $a 2>&1 | tail -3
    n=$(grep -n "engine started" results/engine-strata-$cfg-$cls.log | tail -1 | cut -d: -f1)
    tail -n +${n:-1} results/engine-strata-$cfg-$cls.log | grep "overlapped read of" | python3 -c "
import sys,re,statistics as st
r=[];l=[]
for x in sys.stdin:
  m=re.search(r'in (\d+) ms.*?lost (\S+) ms',x)
  if m: r.append(int(m[1])); l.append(float(m[2])) if m[2]!='n/a' else None
print('reads',len(r),'read med',st.median(r) if r else '-','lost med',st.median(l) if l else '-')"
  done
done
echo SERIES OVM4 DONE
