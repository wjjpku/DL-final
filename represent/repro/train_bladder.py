"""Attempt 1A: batch-size clock ladder (pre-registered in
results/formula_lab/bladder_prereg.json -- read it; this driver implements it
verbatim).

Shared trunk (seed, bs=48, constant peak to step 3000) is checkpointed once
and must pass a 200-step bitwise replay test (G3) before arms run.  Arms fork
from the checkpoint with stage-2 batch size B2 and stage-2 LR eta2; eval is
always bs=48 with the fixed eval generator (G2).

Outputs: results/curves_bladder/<tag>.csv  (step,lr,train_loss,eval_loss)
  tag = b<B2>_e<eta2 tag>_s<seed>, eta2 tag: 10/40 (x1e-5) or 'nodrop'.
Usage:  python train_bladder.py --data_dir /root/dlf/data [--seed 1337]
        [--arms all|main|replicates]
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
CDIR = os.path.join(ROOT, "results", "curves_bladder")
os.makedirs(CDIR, exist_ok=True)

CFG = dict(d=384, nh=6, nl=6, block=256)
PEAK, WARM, TRUNK = 1.5e-3, 400, 3000
STAGE2_STEPS = {12: 8000, 24: 5600, 48: 4000, 96: 4000, 192: 4000}
EVAL_BS, N_EVAL, N_EVAL_TAIL = 48, 10, 40


def build_model(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = T.GPT(vocab=T.VOCAB, **CFG).to(T.DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0, betas=(0.9, 0.95),
                            weight_decay=0.1, eps=1e-8)
    return model, opt


def train_steps(model, opt, gen, trd, etas, bs, t0_step, log_rows=None,
                eval_cfg=None):
    """Run len(etas) steps; optionally log eval per eval_cfg."""
    for i in range(len(etas)):
        step = t0_step + i
        lr = float(etas[i])
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = T.get_batch(trd, CFG["block"], bs, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if eval_cfg is not None:
            dense_lo, dense_hi, total_end, vad, eval_gen, ema_box = eval_cfg
            dense = dense_lo <= step <= dense_hi and step % 4 == 0
            tail = step >= total_end - 1000 and step % 25 == 0
            coarse = step % 100 == 0
            if dense or tail or coarse or step == total_end - 1:
                lv = loss.item()
                ema_box[0] = lv if ema_box[0] is None else 0.9 * ema_box[0] + 0.1 * lv
                ne = N_EVAL_TAIL if tail else N_EVAL
                ev = T.eval_loss(model, vad, CFG["block"], EVAL_BS, ne, eval_gen)
                log_rows.append((step, lr, ema_box[0], ev))
    return model, opt


def checkpoint_state(model, opt, gen):
    buf = io.BytesIO()
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "gen": gen.get_state()}, buf)
    return buf.getvalue()


def restore(blob, seed):
    model, opt = build_model(seed)
    state = torch.load(io.BytesIO(blob), map_location=T.DEV,
                       weights_only=False)
    model.load_state_dict(state["model"])
    opt.load_state_dict(state["opt"])
    gen = torch.Generator()
    gen.set_state(state["gen"])
    return model, opt, gen


def replay_test(blob, trd, seed):
    """G3: 200-step bitwise replay -- two restores must match exactly."""
    losses = []
    for _ in range(2):
        model, opt, gen = restore(blob, seed)
        etas = np.full(200, PEAK)
        ls = []
        for i in range(200):
            for g in opt.param_groups:
                g["lr"] = PEAK
            x, y = T.get_batch(trd, CFG["block"], 48, gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ls.append(loss.item())
        losses.append(ls)
    ok = losses[0] == losses[1]
    print(f"[G3] bitwise replay: {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def run_arm(blob, trd, vad, seed, B2, eta2, tag):
    out = os.path.join(CDIR, tag + ".csv")
    if os.path.exists(out):
        print(f"[skip] {tag}", flush=True)
        return
    model, opt, gen = restore(blob, seed)
    eval_gen = torch.Generator().manual_seed(12345)
    n2 = STAGE2_STEPS[B2]
    total_end = TRUNK + n2
    dense_hi = 6200 if B2 == 12 else 4600
    rows = [];
    ema = [None]
    t0 = time.time()
    etas = np.full(n2, eta2)
    train_steps(model, opt, gen, trd, etas, B2, TRUNK, rows,
                (2900, dense_hi, total_end, vad, eval_gen, ema))
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "lr", "train_loss", "eval_loss"])
        w.writerows(rows)
    print(f"[done] {tag} ({time.time()-t0:.0f}s, {len(rows)} pts)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/root/dlf/data")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--arms", default="all", choices=["all", "main", "replicates"])
    a = ap.parse_args()

    trd = np.memmap(os.path.join(a.data_dir, "wiki_train.u8"), dtype=np.uint8, mode="r")
    vad = np.memmap(os.path.join(a.data_dir, "wiki_val.u8"), dtype=np.uint8, mode="r")

    def make_trunk(seed):
        ck = os.path.join(CDIR, f"trunk_s{seed}.pt")
        if os.path.exists(ck):
            print(f"[trunk] cached s{seed}", flush=True)
            return open(ck, "rb").read()
        model, opt = build_model(seed)
        gen = torch.Generator().manual_seed(seed)
        etas = np.concatenate([PEAK * np.arange(1, WARM + 1) / WARM,
                               np.full(TRUNK - WARM, PEAK)])
        t0 = time.time()
        train_steps(model, opt, gen, trd, etas, 48, 0)
        blob = checkpoint_state(model, opt, gen)
        open(ck, "wb").write(blob)
        print(f"[trunk] s{seed} built ({time.time()-t0:.0f}s)", flush=True)
        if not replay_test(blob, trd, seed):
            raise SystemExit("G3 replay FAILED -- aborting")
        return blob

    B2S = [12, 24, 48, 96, 192]
    if a.arms in ("all", "main"):
        blob = make_trunk(1337)
        for B2 in B2S:
            for eta2, et in [(1e-4, "10"), (4e-4, "40"), (PEAK, "nodrop")]:
                run_arm(blob, trd, vad, 1337, B2, eta2, f"b{B2}_e{et}_s1337")
    if a.arms in ("all", "replicates"):
        blob = make_trunk(1338)
        for B2 in [12, 192]:
            for eta2, et in [(1e-4, "10"), (4e-4, "40")]:
                run_arm(blob, trd, vad, 1338, B2, eta2, f"b{B2}_e{et}_s1338")
    print("BLADDER DONE", flush=True)


if __name__ == "__main__":
    main()
