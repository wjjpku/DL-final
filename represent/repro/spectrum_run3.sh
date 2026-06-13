#!/bin/bash
# Attempt 3 chain v3 (true memory account: 12-15GB retained per unsplit arm).
# Driver now half-batches HVPs (identical operator, ~half peak), so P=3.
# Waits for the in-flight B-axis arms (old driver copy) to exit first.
cd /root/dlf/repro
while pgrep -f 'only spec_e10_b1' >/dev/null; do sleep 30; done
echo "m:spec_e10 m:spec_e40 m:spec_nodrop" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'IFS=: read sc tag <<< "{}"; /root/miniconda3/bin/python train_spectrum.py --scale $sc --only $tag --data_dir /root/dlf/data >> /root/dlf/spectrum.log 2>&1'
echo "l:spec_e10 l:spec_e40 l:spec_nodrop" | tr ' ' '\n' | xargs -P 3 -I{} bash -c 'IFS=: read sc tag <<< "{}"; /root/miniconda3/bin/python train_spectrum.py --scale $sc --only $tag --data_dir /root/dlf/data >> /root/dlf/spectrum.log 2>&1'
echo "SPECTRUM3 DONE" >> /root/dlf/spectrum.log
