"""Attempt 2: 9-schedule calibration suite at an arbitrary scale, bs=48 (G1
override of train.py's per-scale bs).  Mirrors train.py's build_schedules.

Output: results/curves_suite_<scale>/<sched>.csv
Usage: python train_suite.py --scale l --data_dir /root/dlf/data [--only wsd]
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T
from train_floor2 import SCALES, step_once

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PEAK, ENDF, WARM, TOTAL = 1.5e-3, 0.1, 400, 6000
WSDCON_DROP = 3000
BS = 48


def build_schedules():
    end = PEAK * ENDF
    S = {}
    S["cosine"] = T.sched_cosine(TOTAL, PEAK, end, WARM)
    S["constant"] = T.sched_const(TOTAL, PEAK, WARM)
    S["wsd"] = T.sched_wsd(TOTAL, 4000, PEAK, end, WARM)
    S["wsdld"] = T.sched_wsdld(TOTAL, 4000, PEAK, end, WARM)
    for s2 in [0.5e-4, 1.0e-4, 2.0e-4, 4.0e-4, 8.0e-4]:
        S[f"wsdcon_{int(round(s2*1e5))}"] = T.sched_wsdcon(
            TOTAL, WSDCON_DROP, PEAK, s2, WARM)
    return S


def train_one(scale, name, etas, seed, trd, vad, outdir):
    out = os.path.join(outdir, f"{name}.csv")
    if os.path.exists(out):
        print(f"[skip] {scale}/{name}", flush=True)
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)
    eval_gen = torch.Generator().manual_seed(12345)
    model = T.GPT(vocab=T.VOCAB, **SCALES[scale]).to(T.DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0, betas=(0.9, 0.95),
                            weight_decay=0.1, eps=1e-8)
    block = SCALES[scale]["block"]
    rows = []
    ema = None
    t0 = time.time()
    fine_after = (WSDCON_DROP - 40) if name.startswith("wsdcon") else None
    for step in range(TOTAL):
        loss = step_once(model, opt, gen, trd, float(etas[step]), BS, block)
        fine = fine_after is not None and step >= fine_after and step % 4 == 0 \
            and step <= fine_after + 1640
        if step % 20 == 0 or fine or step == TOTAL - 1:
            lv = loss.item()
            ema = lv if ema is None else 0.9 * ema + 0.1 * lv
            ev = T.eval_loss(model, vad, block, BS, 10, eval_gen)
            rows.append((step, float(etas[step]), ema, ev))
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "lr", "train_loss", "eval_loss"])
        w.writerows(rows)
    print(f"[done] {scale}/{name} ({time.time()-t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", required=True, choices=list(SCALES))
    ap.add_argument("--data_dir", default="/root/dlf/data")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    trd = np.memmap(os.path.join(a.data_dir, "wiki_train.u8"), dtype=np.uint8,
                    mode="r")
    vad = np.memmap(os.path.join(a.data_dir, "wiki_val.u8"), dtype=np.uint8,
                    mode="r")
    outdir = os.path.join(ROOT, "results", f"curves_suite_{a.scale}")
    os.makedirs(outdir, exist_ok=True)
    S = build_schedules()
    import json
    sj = os.path.join(outdir, "schedules.json")
    merged = json.load(open(sj)) if os.path.exists(sj) else {}
    merged.update({k: np.asarray(v).tolist() for k, v in S.items()})
    json.dump(merged, open(sj, "w"))
    names = [a.only] if a.only else list(S)
    for name in names:
        train_one(a.scale, name, S[name], a.seed, trd, vad, outdir)
    print(f"SUITE {a.scale} DONE", flush=True)


if __name__ == "__main__":
    main()
