"""Attempt 2: generalized equal-S floor ladder across scales (pre-registered
in results/formula_lab/scaleladder_prereg.json).

Scales (single recipe, depth/width co-scaled; bs=48 ALWAYS per G1):
  m  : d=384 nh=6 nl=6   (~10.7M)   [existing seed-1337 6-rung ladder reused]
  ml : d=448 nh=7 nl=7   (~16M)
  l  : d=512 nh=8 nl=8   (~25M)
  xl : d=576 nh=9 nl=9   (~36M)

8 rungs: eta2 in {0.5,1,2,3,4,6,8,15}e-4, trunk const peak 1.5e-3 to 3000,
stage-2 length T2 = round(S2STAR/eta2), S2STAR=1.2 (equal final cumulative
LR).  Trunk checkpointed once per (scale, seed) with G3 bitwise replay.
N_EVAL=40 in the last 25% of each stage 2, else 10.

Output: results/curves_floor_<scale>/floor_<tag>[_s<seed>].csv
Usage: python train_floor2.py --scale l --data_dir /root/dlf/data
         [--seed 1337] [--only floor_20] [--trunk_only]
"""
import argparse
import csv
import io
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCALES = {
    "m": dict(d=384, nh=6, nl=6, block=256),
    "ml": dict(d=448, nh=7, nl=7, block=256),
    "l": dict(d=512, nh=8, nl=8, block=256),
    "xl": dict(d=576, nh=9, nl=9, block=256),
}
PEAK, WARM, TRUNK, S2STAR, BS = 1.5e-3, 400, 3000, 1.2, 48
DIRSFX = ""   # set to f"_b{bs}" by --bs for non-48 beds (bs192 prereg)
RUNGS = {
    "floor_5": 0.5e-4, "floor_10": 1.0e-4, "floor_20": 2.0e-4,
    "floor_30": 3.0e-4, "floor_40": 4.0e-4, "floor_60": 6.0e-4,
    "floor_80": 8.0e-4, "floor_150": 1.5e-3,
}


def build_model(scale, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = T.GPT(vocab=T.VOCAB, **SCALES[scale]).to(T.DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0, betas=(0.9, 0.95),
                            weight_decay=0.1, eps=1e-8)
    return model, opt


def cdir(scale):
    d = os.path.join(ROOT, "results", f"curves_floor_{scale}{DIRSFX}")
    os.makedirs(d, exist_ok=True)
    return d


def step_once(model, opt, gen, trd, lr, bs, block):
    for g in opt.param_groups:
        g["lr"] = lr
    x, y = T.get_batch(trd, block, bs, gen)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss


def make_trunk(scale, seed, trd):
    ck = os.path.join(cdir(scale), f"trunk_s{seed}.pt")
    if os.path.exists(ck):
        return open(ck, "rb").read()
    model, opt = build_model(scale, seed)
    gen = torch.Generator().manual_seed(seed)
    t0 = time.time()
    block = SCALES[scale]["block"]
    for step in range(TRUNK):
        lr = PEAK * (step + 1) / WARM if step < WARM else PEAK
        step_once(model, opt, gen, trd, lr, BS, block)
    buf = io.BytesIO()
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "gen": gen.get_state()}, buf)
    blob = buf.getvalue()
    open(ck, "wb").write(blob)
    print(f"[trunk] {scale} s{seed} built ({time.time()-t0:.0f}s)", flush=True)
    # G3 bitwise replay
    losses = []
    for _ in range(2):
        m2, o2, g2 = restore(scale, seed, blob)
        ls = [float(step_once(m2, o2, g2, trd, PEAK, BS, block).item())
              for _ in range(200)]
        losses.append(ls)
    ok = losses[0] == losses[1]
    print(f"[G3] {scale} s{seed} replay: {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        raise SystemExit("G3 FAILED")
    return blob


def restore(scale, seed, blob):
    model, opt = build_model(scale, seed)
    st = torch.load(io.BytesIO(blob), map_location=T.DEV, weights_only=False)
    model.load_state_dict(st["model"])
    opt.load_state_dict(st["opt"])
    gen = torch.Generator()
    gen.set_state(st["gen"].cpu().to(torch.uint8))
    return model, opt, gen


def run_rung(scale, seed, blob, tag, trd, vad):
    sfx = "" if seed == 1337 else f"_s{seed}"
    out = os.path.join(cdir(scale), f"{tag}{sfx}.csv")
    if os.path.exists(out):
        print(f"[skip] {scale}/{tag}{sfx}", flush=True)
        return
    eta2 = RUNGS[tag]
    n2 = int(round(S2STAR / eta2))
    model, opt, gen = restore(scale, seed, blob)
    eval_gen = torch.Generator().manual_seed(12345)
    block = SCALES[scale]["block"]
    rows = []
    ema = None
    t0 = time.time()
    for i in range(n2):
        step = TRUNK + i
        loss = step_once(model, opt, gen, trd, eta2, BS, block)
        dense = step <= TRUNK + 1600 and step % 4 == 0
        tail = i >= n2 * 0.75 and step % max(n2 // 80, 10) == 0
        coarse = step % 200 == 0
        if dense or tail or coarse or i == n2 - 1:
            lv = loss.item()
            ema = lv if ema is None else 0.9 * ema + 0.1 * lv
            ne = 40 if tail else 10
            ev = T.eval_loss(model, vad, block, BS, ne, eval_gen)
            rows.append((step, eta2, ema, ev))
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "lr", "train_loss", "eval_loss"])
        w.writerows(rows)
    print(f"[done] {scale}/{tag}{sfx} ({time.time()-t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", required=True, choices=list(SCALES))
    ap.add_argument("--data_dir", default="/root/dlf/data")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--only", default=None)
    ap.add_argument("--trunk_only", action="store_true")
    ap.add_argument("--bs", type=int, default=48)
    a = ap.parse_args()
    if a.bs != 48:
        global BS, DIRSFX
        BS = a.bs
        DIRSFX = f"_b{a.bs}"
    trd = np.memmap(os.path.join(a.data_dir, "wiki_train.u8"), dtype=np.uint8,
                    mode="r")
    vad = np.memmap(os.path.join(a.data_dir, "wiki_val.u8"), dtype=np.uint8,
                    mode="r")
    blob = make_trunk(a.scale, a.seed, trd)
    if a.trunk_only:
        print("TRUNK READY", flush=True)
        return
    tags = [a.only] if a.only else list(RUNGS)
    for tag in tags:
        run_rung(a.scale, a.seed, blob, tag, trd, vad)
    print(f"LADDER {a.scale} s{a.seed} DONE", flush=True)


if __name__ == "__main__":
    main()
