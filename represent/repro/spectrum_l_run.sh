#!/bin/bash
# Attempt 3 l wave, take 2: micro-12 probe batches (~5-6GB/arm), P=1
# sequential so it coexists with the Phase-B derby (~18GB) on the 32GB card.
cd /root/dlf/repro
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for tag in spec_e10 spec_e40 spec_nodrop; do
  /root/miniconda3/bin/python train_spectrum.py --scale l --only $tag --data_dir /root/dlf/data >> /root/dlf/spectrum.log 2>&1
done
echo "SPECTRUM_L DONE" >> /root/dlf/spectrum.log
