#!/bin/bash
# Stage-1 refill: original chain's ladder arms died to dash's lack of <<< and
# the suite arms OOM'd against the b192 bladder refill.  Re-dispatch the full
# job list (skip-if-exists makes this idempotent); wsdcon_40 excluded because
# its original process is still running.
PY=/root/miniconda3/bin/python
cd /root/dlf/repro
while pgrep -f train_bladder >/dev/null; do sleep 60; done
JOBS=""
for r in floor_30 floor_60; do JOBS="$JOBS F:m:1337:$r"; done
for r in floor_5 floor_10 floor_20 floor_30 floor_40 floor_60 floor_80 floor_150; do
  JOBS="$JOBS F:m:1338:$r F:l:1337:$r F:l:1338:$r"
done
for s in constant cosine wsd wsdld wsdcon_5 wsdcon_10 wsdcon_20 wsdcon_80; do
  JOBS="$JOBS S:l:1337:$s"
done
echo $JOBS | tr ' ' '\n' | xargs -P 7 -I{} bash -c 'IFS=: read ty sc sd rg <<< "{}"; if [ "$ty" = "F" ]; then /root/miniconda3/bin/python train_floor2.py --scale $sc --seed $sd --only $rg --data_dir /root/dlf/data; else /root/miniconda3/bin/python train_suite.py --scale $sc --only $rg --data_dir /root/dlf/data; fi >> /root/dlf/refill_stage1.log 2>&1'
echo "REFILL STAGE1 DONE" >> /root/dlf/refill_stage1.log
