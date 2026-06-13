"""Attempt 1F (promoted): equal-terminal-LR cooldown derby predictions, m-bed.
Adds a backbone ensemble (multi-start x alt split) and emits
results/formula_lab/optsched_predictions_m.json with gap spreads.
All arms: peak 1.5e-3, warmup 400, total 6000, linear cooldown ds -> 0.1*peak, hold.
Committed predictions: J_mpl(arm), lagfeat per closure, inter-arm gaps vs D1(ds=3000)
under (a) MPL alone, (b) MPL + lag per closure."""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "represent", "repro"))
import analyze_curves as AC
from engine import mpl_loss_at, cumS

PEAK, WARM, TOTAL, END = 1.5e-3, 400, 6000, 0.1
ARMS = [("d=0", 0.0, 0.0027610792134732053),
        ("d=0.5", 0.5, 0.013249845363695585),
        ("d=0.75", 0.75, 0.02233834262339137)]
LAM = 1.0
DS = [1300, 3000, 5000, 5700]   # D0..D3


def lagfeat_final(etas, delta, lam=LAM):
    eta = np.asarray(etas, float)
    drop = np.zeros_like(eta)
    drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0)
    if delta > 0:
        drop = drop * np.power(np.maximum(eta / PEAK, 1e-12), delta)
    dec = np.exp(-lam * eta)
    acc = 0.0
    for t in range(len(eta)):
        acc = acc * dec[t] + drop[t]
    return acc / PEAK


def lin_sched(ds):
    e = np.full(TOTAL, PEAK)
    e[:WARM] = PEAK * (np.arange(1, WARM + 1) / WARM)
    t = np.arange(ds, TOTAL)
    fr = (t - ds) / max(TOTAL - ds, 1)
    e[ds:] = PEAK * (1 - fr) + END * PEAK * fr
    return e


def main():
    cv = AC.load_scale("m")
    params, fobj = AC.fit_mpl(cv, ["constant", "cosine", "wsdcon_20"], n_starts=8)
    print(f"MPL fit obj={fobj:.5f}")
    rows = {}
    for ds in DS:
        e = lin_sched(ds)
        J = float(mpl_loss_at(e, np.array([TOTAL - 1]), *params)[0])
        feats = {t: lagfeat_final(e, d) for t, d, _ in ARMS}
        rows[ds] = dict(J_mpl=J, S=float(cumS(e)[-1]),
                        feats={t: float(f) for t, f in feats.items()})
        print(f"ds={ds}: J_mpl={J:.5f} S_T={cumS(e)[-1]:.2f} " +
              " ".join(f"feat({t})={feats[t]:.4f}" for t, _, _ in ARMS))
    ref = 3000
    print(f"\ngaps vs ds={ref} (positive = arm worse), committed predictions:")
    print(f"{'ds':>6} {'dMPL':>9} " +
          " ".join(f"{'d+lag(' + t + ')':>14}" for t, _, _ in ARMS))
    out = dict(mpl_params=[float(p) for p in params], rows=rows, gaps={})
    for ds in DS:
        dm = rows[ds]["J_mpl"] - rows[ref]["J_mpl"]
        line = f"{ds:>6} {dm:>+9.5f} "
        g = {"dMPL": float(dm)}
        for t, d, kappa in ARMS:
            tot = dm + kappa * (rows[ds]["feats"][t] - rows[ref]["feats"][t])
            g[t] = float(tot)
            line += f"{tot:>+14.5f}"
        out["gaps"][str(ds)] = g
        print(line)
    json.dump(out, open(os.path.join(HERE, "opt_schedule_derby.json"), "w"))
    print("saved", os.path.join(HERE, "opt_schedule_derby.json"))


if __name__ == "__main__":
    main()
