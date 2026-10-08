BASE="--add mtp-q4=all --add host-core=last --set vram-reserve-mib=1005"
run() { name=$1; shift
  python run_variant.py $name $BASE "$@" --env STRATA_DECODE_TIMING=1 --runs 1 --only chat --keep > logs/F-$name.txt 2>&1
  python stall_probe.py --n 16 >> logs/F-$name.txt 2>&1
  echo "== $name"; grep -E "^req " logs/F-$name.txt | awk '{print $5}' | tr '\n' ' '; echo
  grep "decode timing" logs/engine-$name.log | sed -E 's/.*ms\/window = verify ([0-9.]+).*\+ stage ([0-9.]+)\).*/\1 \2/' | awk '{s+=$2; if($2>m)m=$2} END {printf "stage avg %.1f max %.1f ms/window\n", s/NR, m}'
}
run F1-plecache16m --set ple-row-cache=16777216
run F2-plemmap --set ple-io=mmap
run F3-both --set ple-row-cache=16777216 --set ple-io=mmap
python -c "import run_variant; run_variant.kill_port(8080)"
echo SERIES-DONE
