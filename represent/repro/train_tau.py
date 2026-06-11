"""Focused tau~1/eta run: wsdcon two-stage probes with a LONG relaxation window so the
slow small-eta relaxations are not window-truncated (the showcase had only a 3500-step window,
which truncates tau for small eta). Drop early (step 1500), train to 7000 -> 5500-step window.
Plus 'constant' for the MPL backbone. Curves -> results/curves_tau/."""
import os, sys, time, csv, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CDIR = os.path.join(ROOT, "results", "curves_tau")
os.makedirs(CDIR, exist_ok=True)

CFG = dict(d=384, nh=6, nl=6, block=256, bs=48)   # 10M, fast
PEAK = 2.0e-3
WARM = 400
TOTAL = 7000
DROP = 1500                       # early drop -> 5500-step relaxation window
STAGE2 = [1.0e-4, 2.0e-4, 4.0e-4, 8.0e-4, 16.0e-4]   # 16x range, all < PEAK
SEED = 7
N_EVAL = 10
LOG_EVERY = 20


def const(total, peak, warm):
    e = np.full(total, peak); e[:warm] = peak * (np.arange(1, warm + 1) / warm); return e

def wsdcon(total, drop, peak, s2, warm):
    e = const(total, peak, warm); e[drop:] = s2; return e


def build():
    S = {"constant": const(TOTAL, PEAK, WARM)}
    for s2 in STAGE2:
        S[f"wsdcon_{int(round(s2*1e5))}"] = wsdcon(TOTAL, DROP, PEAK, s2, WARM)
    return S


def train_one(name, etas, trd, vad, fine_after=None, fine_every=3):
    out = os.path.join(CDIR, name + ".csv")
    if os.path.exists(out):
        print(f"  [skip] {name}"); return
    torch.manual_seed(SEED); np.random.seed(SEED)
    gen = torch.Generator().manual_seed(SEED); eval_gen = torch.Generator().manual_seed(999)
    model = T.GPT(vocab=T.VOCAB, d=CFG["d"], nh=CFG["nh"], nl=CFG["nl"], block=CFG["block"]).to(T.DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8)
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


DATA_DIR = r"C:\Users\21100\Desktop\represent\data"


def main():
    print("device", T.DEV, flush=True)
    trd, vad = T.get_data(DATA_DIR)
    S = build()
    json.dump({k: v.tolist() for k, v in S.items()}, open(os.path.join(CDIR, "schedules.json"), "w"))
    for name, etas in S.items():
        fa = (DROP - 40) if name.startswith("wsdcon") else None
        train_one(name, etas, trd, vad, fine_after=fa, fine_every=3)
    print("TAU RUN DONE", flush=True)


if __name__ == "__main__":
    main()
