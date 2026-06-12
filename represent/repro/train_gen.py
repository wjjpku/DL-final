"""Out-of-family generality suite for the upgraded non-adiabatic law.

Four NEW schedule shapes at the m scale (10.7M), matched to the existing
results/curves suite (same model cfg, peak 1.5e-3, warm 400, total 6000,
seed 1337) so the existing m_* curves serve as calibration partners:

  twodrop  : two separated instant drops (2500: ->0.5 peak, 4500: ->0.15 peak)
             -- tests convolution superposition of the kernel.
  cyclic   : drop to 0.3 peak @2500, re-warm to peak @3500-3700, drop to
             0.1 peak @4800 -- tests the one-sided (.)_+ response.
  invsqrt  : eta = peak/sqrt(1+(t-warm)/800) -- smooth unseen family,
             adiabatic regime (correction should stay ~0; no-harm check).
  sharp600 : 600-step linear decay @4000->4600 to 0.1 peak, then hold
             -- sharper than m_wsd's 2000-step decay, long settle window.

Curves -> results/curves_gen/.  Data dir passed explicitly (DL-final copy
has no represent/data).
"""
import os, sys, time, csv, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CDIR = os.path.join(ROOT, "results", "curves_gen")
os.makedirs(CDIR, exist_ok=True)

DATA_DIR = r"C:\Users\21100\Desktop\represent\data"
CFG = dict(d=384, nh=6, nl=6, block=256, bs=48)   # m scale, matches train.py
PEAK = 1.5e-3
END = 0.1 * PEAK
WARM = 400
TOTAL = 6000
SEED = 1337
N_EVAL = 10
LOG_EVERY = 20


def const(total, peak, warm):
    e = np.full(total, peak)
    e[:warm] = peak * (np.arange(1, warm + 1) / warm)
    return e


def build():
    S = {}
    e = const(TOTAL, PEAK, WARM)
    e[2500:] = 0.5 * PEAK
    e[4500:] = 0.15 * PEAK
    S["twodrop"] = (e, [2500, 4500])

    # counterfactual: first drop only.  Same seed + identical LR sequence up
    # to 4500 -> bitwise-identical state at 4500; the difference curve after
    # 4500 isolates the second drop's response.
    e = const(TOTAL, PEAK, WARM)
    e[2500:] = 0.5 * PEAK
    S["onedrop"] = (e, [2500])

    e = const(TOTAL, PEAK, WARM)
    e[2500:] = 0.3 * PEAK
    e[3500:3700] = np.linspace(0.3 * PEAK, PEAK, 200)
    e[3700:] = PEAK
    e[4800:] = 0.1 * PEAK
    S["cyclic"] = (e, [2500, 4800])

    t = np.arange(TOTAL, dtype=np.float64)
    e = PEAK / np.sqrt(1.0 + np.maximum(t - WARM, 0.0) / 800.0)
    e[:WARM] = PEAK * (np.arange(1, WARM + 1) / WARM)
    S["invsqrt"] = (e, [])

    e = const(TOTAL, PEAK, WARM)
    e[4000:4600] = np.linspace(PEAK, END, 600)
    e[4600:] = END
    S["sharp600"] = (e, [4000])
    return S


def fine_mask(total, drops):
    m = np.zeros(total, dtype=bool)
    for d in drops:
        lo, hi = max(d - 100, 0), min(d + 1600, total)
        m[lo:hi] = True
    return m


def train_one(name, etas, drops, trd, vad):
    out = os.path.join(CDIR, name + ".csv")
    if os.path.exists(out):
        print(f"  [skip] {name}", flush=True)
        return
    torch.manual_seed(SEED); np.random.seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    eval_gen = torch.Generator().manual_seed(12345)
    model = T.GPT(vocab=T.VOCAB, d=CFG["d"], nh=CFG["nh"], nl=CFG["nl"],
                  block=CFG["block"]).to(T.DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0, betas=(0.9, 0.95),
                            weight_decay=0.1, eps=1e-8)
    bs, blk = CFG["bs"], CFG["block"]
    fm = fine_mask(len(etas), drops)
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
        if step % LOG_EVERY == 0 or fine or step == len(etas) - 1:
            lv = loss.item()
            ema = lv if ema is None else 0.9 * ema + 0.1 * lv
            ev = T.eval_loss(model, vad, blk, bs, N_EVAL, eval_gen)
            rows.append((step, lr, ema, ev))
            if step % (LOG_EVERY * 30) == 0:
                print(f"  {name} {step}/{len(etas)} lr={lr:.2e} eval={ev:.4f} "
                      f"({(step+1)/max(time.time()-t0,1e-9):.0f} it/s)", flush=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "lr", "train_loss", "eval_loss"])
        w.writerows(rows)
    print(f"  [done] {name} ({time.time()-t0:.0f}s)", flush=True)


def main():
    print("device", T.DEV, flush=True)
    trd, vad = T.get_data(DATA_DIR)
    S = build()
    json.dump({k: v[0].tolist() for k, v in S.items()},
              open(os.path.join(CDIR, "schedules.json"), "w"))
    for name, (etas, drops) in S.items():
        train_one(name, etas, drops, trd, vad)
    print("GEN RUN DONE", flush=True)


if __name__ == "__main__":
    main()
