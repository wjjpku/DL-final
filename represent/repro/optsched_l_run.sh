#!/bin/bash
# Attempt 2 appendix Phase B: 25M derby, auto-starts after spectroscopy.
# 12 arms (4 ds x 3 seeds) at P=6 (~3GB each bf16 training).
cd /root/dlf/repro
while ! grep -q 'SPECTRUM3 DONE' /root/dlf/spectrum.log 2>/dev/null; do sleep 120; done
JOBS=""
for s in 1337 1338 1339; do for d in 1300 3000 5000 5700; do JOBS="$JOBS ds${d}_s${s}"; done; done
echo $JOBS | tr ' ' '\n' | xargs -P 6 -I{} bash -c '/root/miniconda3/bin/python train_optsched.py --scale l --only {} --data_dir /root/dlf/data >> /root/dlf/optsched_l.log 2>&1'
echo "OPTSCHED_L DONE" >> /root/dlf/optsched_l.log
