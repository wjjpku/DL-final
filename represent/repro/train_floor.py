"""Equal-S constant-LR floor ladder at the m scale (10.7M).

Purpose: identify the equilibrium floor F(eta) (and its log-log exponent p)
on a REAL transformer, free of the backbone confound that makes the existing
m_wsdcon_{5..80} final losses non-monotonic in stage-2 LR (their final
cumulative LR S differs, so A*S^-alpha moves opposite to the noise floor).

Design:
  * shared trunk: warmup 400 -> constant peak 1.5e-3 until step 3000
    (fixed seed => identical init AND identical per-step training batches
    across all rungs; trunk state at the drop is the same for every run).
  * stage 2: instant drop to eta2, run T2 = round(S2STAR/eta2) steps so that
    EVERY rung ends at the same total cumulative LR  S* = S_trunk + S2STAR.
    Backbone terms (any function of S) are then identical across rungs at the
    measurement point; floor differences are pure eta physics + eval noise.
  * settle check: tau(10.7M) ~ 475-910 steps (flat in eta; analyze_tau /
    REPORT.md). Smallest stage-2 length is 1500 steps (eta2=8e-4, tau~475)
    = 3.2 tau; all other rungs >= 3000 steps. Floors = last-25%-of-stage-2
    mean of smoothed eval loss.
  * dense post-drop sampling (every 4 steps for 1600 steps) doubles as a
    tau(eta2) measurement, replacing the interrupted curves_tau suite.
  * anchor rung floor_150 (eta2 = peak, no drop) pins the top of the ladder.

Curves -> results/curves_floor/.  schedules.json is MERGED, not clobbered.
Second-seed replicates: pass --seed 1338 --rungs floor_20,floor_80
(curves get an _s1338 suffix).
"""
import os, sys, time, csv, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CDIR = os.path.join(ROOT, "results", "curves_floor")
os.makedirs(CDIR, exist_ok=True)

DATA_DIR = r"C:\Users\21100\Desktop\represent\data"
CFG = dict(d=384, nh=6, nl=6, block=256, bs=48)   # m scale, matches train.py
PEAK = 1.5e-3
WARM = 400
TRUNK = 3000          # drop step (same as m_wsdcon suite)
S2STAR = 1.2          # stage-2 cumulative-LR budget (sum eta) -- equal for all rungs
N_EVAL = 10

# rung name -> stage-2 LR (name = eta2 * 1e5, matching the wsdcon tag convention)
RUNGS = {
    "floor_5":   0.5e-4,
    "floor_10":  1.0e-4,
    "floor_20":  2.0e-4,
    "floor_40":  4.0e-4,
    "floor_80":  8.0e-4,
    "floor_150": 1.5e-3,   # no-drop anchor: constant at peak to the same S*
}


def build(rungs):
    S = {}
    for name in rungs:
        e2 = RUNGS[name]
        T2 = int(round(S2STAR / e2))
        tot = TRUNK + T2
        e = np.full(tot, PEAK)
        e[:WARM] = PEAK * (np.arange(1, WARM + 1) / WARM)
        e[TRUNK:] = e2
        drops = [] if e2 >= PEAK else [TRUNK]
        S[name] = (e, drops)
    return S


def fine_mask(total, drops):
    m = np.zeros(total, dtype=bool)
    for d in drops:
        lo, hi = max(d - 100, 0), min(d + 1600, total)
        m[lo:hi] = True
    return m


def train_one(name, etas, drops, trd, vad, seed):
    out = os.path.join(CDIR, name + ".csv")
    if os.path.exists(out):
        print(f"  [skip] {name}", flush=True)
        return
    torch.manual_seed(seed); np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)
    eval_gen = torch.Generator().manual_seed(12345)
    model = T.GPT(vocab=T.VOCAB, d=CFG["d"], nh=CFG["nh"], nl=CFG["nl"],
                  block=CFG["block"]).to(T.DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0, betas=(0.9, 0.95),
                            weight_decay=0.1, eps=1e-8)
    bs, blk = CFG["bs"], CFG["block"]
    total = len(etas)
    log_every = max(20, (total // 300) // 4 * 4)   # ~300 coarse logs even on 27k-step rungs
    fm = fine_mask(total, drops)
    rows = []; ema = None; t0 = time.time()
    for step in range(total):
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
        if step % log_every == 0 or fine or step == total - 1:
            lv = loss.item()
            ema = lv if ema is None else 0.9 * ema + 0.1 * lv
            ev = T.eval_loss(model, vad, blk, bs, N_EVAL, eval_gen)
            rows.append((step, lr, ema, ev))
            if step % (log_every * 30) == 0:
                print(f"  {name} {step}/{total} lr={lr:.2e} eval={ev:.4f} "
                      f"({(step+1)/max(time.time()-t0,1e-9):.0f} it/s)", flush=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "lr", "train_loss", "eval_loss"])
        w.writerows(rows)
    print(f"  [done] {name} ({time.time()-t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--rungs", default=",".join(RUNGS))  # comma list to restrict
    args = ap.parse_args()
    rungs = [r for r in args.rungs.split(",") if r in RUNGS]
    print("device", T.DEV, "rungs", rungs, "seed", args.seed, flush=True)
    trd, vad = T.get_data(DATA_DIR)
    S = build(rungs)
    suffix = "" if args.seed == 1337 else f"_s{args.seed}"
    # MERGE schedules.json (train.py's main() clobbers; we must not)
    sj = os.path.join(CDIR, "schedules.json")
    sched_db = json.load(open(sj)) if os.path.exists(sj) else {}
    for k, v in S.items():
        sched_db[k + suffix] = v[0].tolist()
    json.dump(sched_db, open(sj, "w"))
    for name, (etas, drops) in S.items():
        train_one(name + suffix, etas, drops, trd, vad, args.seed)
    print("FLOOR LADDER DONE", flush=True)


if __name__ == "__main__":
    main()
