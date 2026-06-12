#!/bin/bash
# Attempt 3 spectroscopy wave: scale slot gate-filled m+l (DECISION_TABLE).
# l arms first (longest); bash -c (dash lesson from stage1_chain).
cd /root/dlf/repro
JOBS="l:spec_e10 l:spec_e40 l:spec_nodrop m:spec_e10_b12 m:spec_e10 m:spec_e40 m:spec_nodrop m:spec_e10_b192"
echo $JOBS | tr ' ' '\n' | xargs -P 4 -I{} bash -c 'IFS=: read sc tag <<< "{}"; /root/miniconda3/bin/python train_spectrum.py --scale $sc --only $tag --data_dir /root/dlf/data >> /root/dlf/spectrum.log 2>&1'
echo "SPECTRUM ALL DONE" >> /root/dlf/spectrum.log
