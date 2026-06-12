"""Seed replicates for the paired deposit-ratio test (A2/A1).

Single-seed verdict was 'unmeasurable' (eval noise + smoothing).  Train
{constant, onedrop, twodrop} at seeds 1338 and 1339 (suffix _s<seed>);
averaging paired differences across 3 seeds (incl. existing 1337) should
cut the noise ~sqrt(3)x and tighten the specification spread.

Curves -> results/curves_gen/<name>_s<seed>.csv
"""
import os, sys, time, csv
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T
import train_gen as G

CDIR = G.CDIR
SEEDS = [1338, 1339]


def build3():
    e = G.const(G.TOTAL, G.PEAK, G.WARM)
    S = {"constant": (e.copy(), [])}
    e1 = e.copy(); e1[2500:] = 0.5 * G.PEAK
    S["onedrop"] = (e1, [2500])
    e2 = e1.copy(); e2[4500:] = 0.15 * G.PEAK
    S["twodrop"] = (e2, [2500, 4500])
    return S


def train_one(name, etas, drops, trd, vad, seed):
    out = os.path.join(CDIR, f"{name}_s{seed}.csv")
    if os.path.exists(out):
        print(f"  [skip] {name}_s{seed}", flush=True)
        return
    torch.manual_seed(seed); np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)
    eval_gen = torch.Generator().manual_seed(12345)
    model = T.GPT(vocab=T.VOCAB, d=G.CFG["d"], nh=G.CFG["nh"], nl=G.CFG["nl"],
                  block=G.CFG["block"]).to(T.DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0, betas=(0.9, 0.95),
                            weight_decay=0.1, eps=1e-8)
    bs, blk = G.CFG["bs"], G.CFG["block"]
    fm = G.fine_mask(len(etas), drops)
    rows = []; ema = None; t0 = time.time()
    for step in range(len(etas)):
        lr = float(etas[step])
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = T.get_batch(trd, blk, bs, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        fine = bool(fm[step]) and step % 4 == 0
        if step % G.LOG_EVERY == 0 or fine or step == len(etas) - 1:
            lv = loss.item()
            ema = lv if ema is None else 0.9 * ema + 0.1 * lv
            ev = T.eval_loss(model, vad, blk, bs, G.N_EVAL, eval_gen)
            rows.append((step, lr, ema, ev))
            if step % (G.LOG_EVERY * 30) == 0:
                print(f"  {name}_s{seed} {step}/{len(etas)} eval={ev:.4f} "
                      f"({(step+1)/max(time.time()-t0,1e-9):.0f} it/s)", flush=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "lr", "train_loss", "eval_loss"])
        w.writerows(rows)
    print(f"  [done] {name}_s{seed} ({time.time()-t0:.0f}s)", flush=True)


def main():
    print("device", T.DEV, flush=True)
    trd, vad = T.get_data(G.DATA_DIR)
    S = build3()
    for seed in SEEDS:
        for name, (etas, drops) in S.items():
            train_one(name, etas, drops, trd, vad, seed)
    print("SEED REPLICATES DONE", flush=True)


if __name__ == "__main__":
    main()
