"""T-C (final adjudication): one probe whose stage-2 LR exactly matches the
10.7M bed targets' terminal LR (sharp600/wsd/wsdld end at 0.1*peak=1.5e-4),
so the shipped matched-probe rule can fire on this bed.
Curve -> results/curves/m_wsdcon_15.csv (same seed/cfg as the m-suite)."""
import os, sys, time, csv
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "curves", "m_wsdcon_15.csv")
DATA_DIR = r"C:\Users\21100\Desktop\represent\data"
CFG = dict(d=384, nh=6, nl=6, block=256, bs=48)
PEAK, WARM, TOTAL, DROP, STAGE2, SEED = 1.5e-3, 400, 6000, 3000, 1.5e-4, 1337


def main():
    if os.path.exists(OUT):
        print("[skip] exists"); return
    trd, vad = T.get_data(DATA_DIR)
    e = np.full(TOTAL, PEAK)
    e[:WARM] = PEAK * (np.arange(1, WARM + 1) / WARM)
    e[DROP:] = STAGE2
    torch.manual_seed(SEED); np.random.seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    eval_gen = torch.Generator().manual_seed(12345)
    model = T.GPT(vocab=T.VOCAB, **{k: CFG[k] for k in ["d", "nh", "nl", "block"]}).to(T.DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0, betas=(0.9, 0.95),
                            weight_decay=0.1, eps=1e-8)
    rows = []; ema = None; t0 = time.time()
    for step in range(TOTAL):
        for g in opt.param_groups:
            g["lr"] = float(e[step])
        x, y = T.get_batch(trd, CFG["block"], CFG["bs"], gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        fine = step >= DROP - 40 and step % 4 == 0
        if step % 20 == 0 or fine or step == TOTAL - 1:
            lv = loss.item(); ema = lv if ema is None else 0.9 * ema + 0.1 * lv
            ev = T.eval_loss(model, vad, CFG["block"], CFG["bs"], 10, eval_gen)
            rows.append((step, float(e[step]), ema, ev))
            if step % 600 == 0:
                print(f"  {step}/{TOTAL} eval={ev:.4f} "
                      f"({(step+1)/max(time.time()-t0,1e-9):.0f} it/s)", flush=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["step", "lr", "train_loss", "eval_loss"])
        w.writerows(rows)
    # merge into schedules.json (do NOT clobber)
    import json
    sj = os.path.join(ROOT, "results", "curves", "schedules.json")
    sched = json.load(open(sj))
    sched["wsdcon_15"] = e.tolist()
    json.dump(sched, open(sj, "w"))
    print(f"[done] ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
