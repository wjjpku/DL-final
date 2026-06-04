"""
train_showcase.py -- a focused real-transformer experiment DESIGNED to expose the
non-adiabatic lag, addressing why the MPL-protocol replica was marginal at small scale:
  * higher peak LR (2.5e-3)  -> bigger SGD noise floor -> bigger lag
  * a SHARP wsd decay (300 steps) vs a GRADUAL decay to the SAME final LR
    -> the paper's clean rate-dependence signature ("same destination, faster sweep, bigger lag")
  * wsdcon two-stage probes with a long relaxation window for tau ~ 1/eta.
Reuses the model from train.py. Curves -> results/curves_show/<name>.csv.
"""
import os, sys, time, csv, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T   # GPT, get_data, get_batch, eval_loss, DEV, VOCAB

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CDIR = os.path.join(ROOT, "results", "curves_show")
os.makedirs(CDIR, exist_ok=True)

CFG = dict(d=384, nh=6, nl=6, block=256, bs=48)   # ~10M, fast
PEAK = 2.5e-3
END = 2.5e-4
WARM = 400
TOTAL = 6000
GRAD_START = 2000          # gradual decay starts early
SHARP_START = 5000         # sharp decay starts late
SHARP_LEN = 300
DROP = 2500                # wsdcon drop step
STAGE2 = [0.5e-4, 1.0e-4, 2.0e-4, 4.0e-4, 8.0e-4]
SEED = 1234
N_EVAL = 12
LOG_EVERY = 15


def const(total, peak, warm):
    e = np.full(total, peak); e[:warm] = peak * (np.arange(1, warm + 1) / warm); return e

def cosine(total, peak, end, warm):
    e = np.empty(total)
    for t in range(total):
        e[t] = peak * (t + 1) / warm if t < warm else end + 0.5 * (peak - end) * (1 + np.cos(np.pi * (t - warm) / max(total - warm, 1)))
    return e

def wsd_grad(total, start, peak, end, warm):
    e = const(total, peak, warm); dec = np.arange(start, total); fr = (dec - start) / max(total - start, 1)
    e[start:] = (np.sqrt(peak) * (1 - fr) + np.sqrt(end) * fr) ** 2; return e

def wsd_sharp(total, start, length, peak, end, warm):
    e = const(total, peak, warm)
    for i, t in enumerate(range(start, min(start + length, total))):
        fr = i / max(length, 1); e[t] = peak * (1 - fr) + end * fr
    e[start + length:] = end; return e

def wsdcon(total, drop, peak, s2, warm):
    e = const(total, peak, warm); e[drop:] = s2; return e


def build():
    S = {}
    S["constant"] = const(TOTAL, PEAK, WARM)
    S["cosine"] = cosine(TOTAL, PEAK, END, WARM)
    S["wsd_grad"] = wsd_grad(TOTAL, GRAD_START, PEAK, END, WARM)
    S["wsd_sharp"] = wsd_sharp(TOTAL, SHARP_START, SHARP_LEN, PEAK, END, WARM)
    for s2 in STAGE2:
        S[f"wsdcon_{int(round(s2*1e5))}"] = wsdcon(TOTAL, DROP, PEAK, s2, WARM)
    return S


def train_one(name, etas, fine_after=None, fine_every=4):
    out = os.path.join(CDIR, name + ".csv")
    if os.path.exists(out):
        print(f"  [skip] {name}"); return
    torch.manual_seed(SEED); np.random.seed(SEED)
    gen = torch.Generator().manual_seed(SEED); eval_gen = torch.Generator().manual_seed(999)
    model = T.GPT(vocab=T.VOCAB, d=CFG["d"], nh=CFG["nh"], nl=CFG["nl"], block=CFG["block"]).to(T.DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8)
    trd, vad = DATA
    bs, blk = CFG["bs"], CFG["block"]; rows = []; ema = None; t0 = time.time()
    for step in range(len(etas)):
        lr = float(etas[step])
        for g in opt.param_groups: g["lr"] = lr
        x, y = T.get_batch(trd, blk, bs, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        fine = (fine_after is not None and step >= fine_after and step % fine_every == 0)
        if step % LOG_EVERY == 0 or fine or step == len(etas) - 1:
            lv = loss.item(); ema = lv if ema is None else 0.9 * ema + 0.1 * lv
            ev = T.eval_loss(model, vad, blk, bs, N_EVAL, eval_gen)
            rows.append((step, lr, ema, ev))
            if step % (LOG_EVERY * 30) == 0:
                print(f"  {name} {step}/{len(etas)} lr={lr:.2e} eval={ev:.4f} ({(step+1)/max(time.time()-t0,1e-9):.0f} it/s)", flush=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["step", "lr", "train_loss", "eval_loss"]); w.writerows(rows)
    print(f"  [done] {name} ({time.time()-t0:.0f}s)", flush=True)


DATA = None
def main():
    global DATA
    print("device", T.DEV, flush=True)
    DATA = T.get_data()
    S = build()
    json.dump({k: v.tolist() for k, v in S.items()}, open(os.path.join(CDIR, "schedules.json"), "w"))
    for name, etas in S.items():
        fa = (DROP - 40) if name.startswith("wsdcon") else (SHARP_START - 40 if name == "wsd_sharp" else None)
        train_one(name, etas, fine_after=fa, fine_every=4)
    print("SHOWCASE DONE", flush=True)


if __name__ == "__main__":
    main()
