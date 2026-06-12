#!/bin/bash
# Attempt 3 wave 2 (memory-safe re-plan): fp32 double-backward measured at
# 9-11.5GB per l arm -- wave A: remaining m arms P=3 alongside the live
# b12/b192 arms; wave B: l arms P=2 after all m arms exit.
cd /root/dlf/repro
MJ="m:spec_e10 m:spec_e40 m:spec_nodrop"
echo $MJ | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'IFS=: read sc tag <<< "{}"; /root/miniconda3/bin/python train_spectrum.py --scale $sc --only $tag --data_dir /root/dlf/data >> /root/dlf/spectrum.log 2>&1'
while pgrep -f 'train_spectrum.py --scale m' >/dev/null; do sleep 60; done
LJ="l:spec_e10 l:spec_e40 l:spec_nodrop"
echo $LJ | tr ' ' '\n' | xargs -P 2 -I{} bash -c 'IFS=: read sc tag <<< "{}"; /root/miniconda3/bin/python train_spectrum.py --scale $sc --only $tag --data_dir /root/dlf/data >> /root/dlf/spectrum.log 2>&1'
echo "SPECTRUM2 DONE" >> /root/dlf/spectrum.log
