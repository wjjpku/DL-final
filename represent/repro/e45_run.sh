#!/bin/bash
# E4 (horizon ladders, mandatory per A3) + E5 (sign-flip control replication).
# Wave 1: E5 arms (P=3, ~16GB incl b192 control) + both E4 trunks (~4.4GB).
# Wave 2: E4 rungs 8 rungs x 2 trunk lengths at P=8.
PY=/root/miniconda3/bin/python
cd /root/dlf/repro
printf 'b96_e10_s1338\nb96_enodrop_s1338\nb192_enodrop_s1338\n' | \
  xargs -P 3 -I{} bash -c "$PY train_bladder.py --only {} --data_dir /root/dlf/data >> /root/dlf/e5.log 2>&1" &
W1=$!
for L in 12000 24000; do
  $PY train_floor2.py --scale m --trunk_len $L --trunk_only --data_dir /root/dlf/data >> /root/dlf/e4.log 2>&1 &
done
wait
JOBS=""
for L in 12000 24000; do
  for r in floor_5 floor_10 floor_20 floor_30 floor_40 floor_60 floor_80 floor_150; do
    JOBS="$JOBS $L:$r"
  done
done
echo $JOBS | tr ' ' '\n' | xargs -P 8 -I{} bash -c 'IFS=: read L r <<< "{}"; /root/miniconda3/bin/python train_floor2.py --scale m --trunk_len $L --only $r --data_dir /root/dlf/data >> /root/dlf/e4.log 2>&1'
echo "E45 DONE" >> /root/dlf/e4.log
