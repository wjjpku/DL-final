"""g2d3 decisive m-bed test (prereg=lamrho_prereg.json): rho-ladder of
cooldown widths from a shared trunk, to test whether the lag-kernel rate
lam reads off the local decrement concentration rho.

From the G3-replayed m trunk (const 1.5e-3 to step 3000), cool
1.5e-3 -> 1e-4 over WIDTH W steps, then HOLD 1e-4 for HOLD steps and watch
the relaxation.  Output: results/curves_lamrho/W<W>[_s<seed>].csv
Usage: python train_lamrho.py --seed 1337 --only W40 --data_dir /root/dlf/data
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
from train_floor2 import SCALES, make_trunk, restore, step_once, TRUNK

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PEAK, ETA2, HOLD, BS = 1.5e-3, 1e-4, 4000, 48
WIDTHS = {"W1": 1, "W10": 10, "W40": 40, "W160": 160, "W640": 640,
          # g2d3b wider+denser rho ladder (prereg=lamrho_b_prereg.json)
          "W4": 4, "W80": 80, "W320": 320, "W1280": 1280,
          "W2560": 2560, "W5120": 5120}


def cdir():
    d = os.path.join(ROOT, "results", "curves_lamrho")
    os.makedirs(d, exist_ok=True)
    return d


def schedule(W):
    """linear cool PEAK->ETA2 over W steps, then hold ETA2 for HOLD."""
    if W <= 1:
        cool = np.array([ETA2])
    else:
        cool = np.linspace(PEAK, ETA2, W + 1)[1:]
    return np.concatenate([cool, np.full(HOLD, ETA2)])


def run_arm(seed, blob, tag, trd, vad):
    sfx = "" if seed == 1337 else f"_s{seed}"
    out = os.path.join(cdir(), f"{tag}{sfx}.csv")
    if os.path.exists(out):
        print(f"[skip] {tag}{sfx}", flush=True)
        return
    W = WIDTHS[tag]
    etas = schedule(W)
    model, opt, gen = restore("m", seed, blob)
    eval_gen = torch.Generator().manual_seed(12345)
    block = SCALES["m"]["block"]
    rows = []
    ema = None
    t0 = time.time()
    n2 = len(etas)
    for i in range(n2):
        step = TRUNK + i
        loss = step_once(model, opt, gen, trd, float(etas[i]), BS, block)
        post = i - W                       # steps since cooldown end
        dense = 0 <= post <= 1600 and post % 4 == 0
        coarse = step % 100 == 0
        if dense or coarse or i == n2 - 1 or i < W:
            lv = loss.item()
            ema = lv if ema is None else 0.9 * ema + 0.1 * lv
            ev = T.eval_loss(model, vad, block, BS, 20, eval_gen)
            rows.append((step, float(etas[i]), ema, ev))
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "lr", "train_loss", "eval_loss"])
        w.writerows(rows)
    print(f"[done] {tag}{sfx} ({time.time()-t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/root/dlf/data")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    trd = np.memmap(os.path.join(a.data_dir, "wiki_train.u8"), dtype=np.uint8,
                    mode="r")
    vad = np.memmap(os.path.join(a.data_dir, "wiki_val.u8"), dtype=np.uint8,
                    mode="r")
    blob = make_trunk("m", a.seed, trd)
    tags = [a.only] if a.only else list(WIDTHS)
    for tag in tags:
        run_arm(a.seed, blob, tag, trd, vad)
    print(f"LAMRHO s{a.seed} DONE", flush=True)


if __name__ == "__main__":
    main()
