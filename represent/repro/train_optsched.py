"""Attempt 1F: final-loss-level cooldown derby at 10.7M (pre-registered in
results/formula_lab/optsched_predictions_m.json -- gap predictions and the
verdict rule were committed before this launch).

Arms: linear cooldown start ds in {1300, 3000, 5000, 5700} -> 0.1*peak then
hold, total 6000, seeds {1337, 1338, 1339}; plus wsdld at seeds 1338/1339
(1337 reuses results/curves/m_wsdld.csv).  Same seed = bitwise-identical
trunks until each arm's ds.  Metric: mean raw eval loss over [5800, 6000).

Output: results/curves_optsched/ds<ds>_s<seed>.csv, wsdld_s<seed>.csv
Usage:  python train_optsched.py --data_dir /root/dlf/data
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CDIR = os.path.join(ROOT, "results", "curves_optsched")

CFG = dict(d=384, nh=6, nl=6, block=256)
PEAK, WARM, TOTAL, ENDF = 1.5e-3, 400, 6000, 0.1
DS = [1300, 3000, 5000, 5700]
SEEDS = [1337, 1338, 1339]


def lin_sched(ds):
    e = np.full(TOTAL, PEAK)
    e[:WARM] = PEAK * (np.arange(1, WARM + 1) / WARM)
    t = np.arange(ds, TOTAL)
    fr = (t - ds) / max(TOTAL - ds, 1)
    e[ds:] = PEAK * (1 - fr) + ENDF * PEAK * fr
    return e


def wsdld_sched():
    e = np.full(TOTAL, PEAK)
    e[:WARM] = PEAK * (np.arange(1, WARM + 1) / WARM)
    ds = 4000
    t = np.arange(ds, TOTAL)
    fr = (t - ds) / max(TOTAL - ds, 1)
    e[ds:] = PEAK * (1 - fr) + ENDF * PEAK * fr
    return e


def train_one(tag, etas, seed, trd, vad):
    out = os.path.join(CDIR, tag + ".csv")
    if os.path.exists(out):
        print(f"[skip] {tag}", flush=True)
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)
    eval_gen = torch.Generator().manual_seed(12345)
    model = T.GPT(vocab=T.VOCAB, **CFG).to(T.DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0, betas=(0.9, 0.95),
                            weight_decay=0.1, eps=1e-8)
    rows = []
    ema = None
    t0 = time.time()
    for step in range(TOTAL):
        for g in opt.param_groups:
            g["lr"] = float(etas[step])
        x, y = T.get_batch(trd, CFG["block"], 48, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        dense_tail = step >= 5600 and step % 10 == 0
        if step % 20 == 0 or dense_tail or step == TOTAL - 1:
            lv = loss.item()
            ema = lv if ema is None else 0.9 * ema + 0.1 * lv
            ne = 40 if dense_tail else 20
            ev = T.eval_loss(model, vad, CFG["block"], 48, ne, eval_gen)
            rows.append((step, float(etas[step]), ema, ev))
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "lr", "train_loss", "eval_loss"])
        w.writerows(rows)
    print(f"[done] {tag} ({time.time()-t0:.0f}s)", flush=True)


def main():
    global CFG, CDIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/root/dlf/data")
    ap.add_argument("--only", default=None)
    ap.add_argument("--scale", default="m",
                    help="m (default, original bed) or ml/l/xl per "
                         "train_floor2.SCALES; non-m outputs go to "
                         "curves_optsched_<scale> (12 ds-arms, no wsdld)")
    a = ap.parse_args()
    if a.scale != "m":
        from train_floor2 import SCALES
        CFG = dict(SCALES[a.scale])
        CDIR = os.path.join(ROOT, "results", f"curves_optsched_{a.scale}")
    os.makedirs(CDIR, exist_ok=True)
    trd = np.memmap(os.path.join(a.data_dir, "wiki_train.u8"), dtype=np.uint8, mode="r")
    vad = np.memmap(os.path.join(a.data_dir, "wiki_val.u8"), dtype=np.uint8, mode="r")
    arms = []
    for seed in SEEDS:
        for ds in DS:
            arms.append((f"ds{ds}_s{seed}", lin_sched(ds), seed))
    if a.scale == "m":
        for seed in [1338, 1339]:
            arms.append((f"wsdld_s{seed}", wsdld_sched(), seed))
    if a.only:
        tag, etas, seed = next(x for x in arms if x[0] == a.only)
        train_one(tag, etas, seed, trd, vad)
        return
    for tag, etas, seed in arms:
        train_one(tag, etas, seed, trd, vad)
    print("OPTSCHED DONE", flush=True)


if __name__ == "__main__":
    main()
