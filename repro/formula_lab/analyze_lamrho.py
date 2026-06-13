#!/usr/bin/env python3
"""g2d3 / g2d3b verdict: does the relaxation rate lam read off the local
decrement concentration rho?  Per-arm S-time relaxation fit -> lam(W); then
(g2d3b) out-of-sample log-law test + kernel-MAE predictive margin.
prereg: lamrho_prereg.json (g2d3) + lamrho_b_prereg.json (g2d3b)."""
import glob
import json
import os
import sys

import numpy as np
from scipy.optimize import curve_fit, minimize_scalar
from scipy.stats import spearmanr

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", ".."))
sys.path.insert(0, os.path.join(REPO, "represent", "repro"))
import analyze_curves as AC  # noqa: E402

CDIR = os.path.join(REPO, "represent", "results", "curves_lamrho")
PEAK, ETA2, TRUNK = 1.5e-3, 1e-4, 3000
WIDTHS = {"W1": 1, "W10": 10, "W40": 40, "W160": 160, "W640": 640,
          "W4": 4, "W80": 80, "W320": 320, "W1280": 1280,
          "W2560": 2560, "W5120": 5120}
ORIG_W = {1, 10, 40, 160, 640}


def arm_curve(path, W):
    rows = np.genfromtxt(path, delimiter=",", names=True)
    step = np.atleast_1d(rows["step"]).astype(int)
    loss = np.atleast_1d(rows["eval_loss"]).astype(float)
    lr = np.atleast_1d(rows["lr"]).astype(float)
    sm = AC.smooth_by_step(step, loss)
    order = np.argsort(step)
    step, sm, lr = step[order], sm[order], lr[order]
    S = np.cumsum(lr * np.gradient(step.astype(float)))
    m = step >= (TRUNK + W)
    if m.sum() < 8:
        return None
    return (S[m] - S[m][0]), sm[m]


def fit_lam(x, y):
    def mdl(x, F, A, lam):
        return F + A * np.exp(-lam * x)
    po, _ = curve_fit(mdl, x, y, p0=[y[-1], max(y[0] - y[-1], 1e-3), 10.0],
                      maxfev=60000, bounds=([0, 1e-4, 0.05], [5, 2, 200]))
    pred = mdl(x, *po)
    r2 = 1 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-30)
    return po[2], r2


def fixedlam_mae(x, y, lam):
    """refit F,A with lam fixed; return curve MAE."""
    E = np.exp(-lam * x)
    M = np.vstack([np.ones_like(E), E]).T
    coef, *_ = np.linalg.lstsq(M, y, rcond=None)
    return float(np.mean(np.abs(y - M @ coef)))


def rho_of(W):
    if W <= 1:
        cw = np.array([PEAK]); drop = np.array([PEAK - ETA2])
    else:
        e = np.linspace(PEAK, ETA2, W + 1)[1:]
        cw = np.concatenate([[PEAK], e[:-1]]); drop = np.maximum(cw - e, 0)
    return float(np.mean(drop / np.maximum(cw, 1e-9)))


def main():
    arms = {}     # W -> list of (x,y,lam)
    for tag, W in WIDTHS.items():
        fs = sorted(glob.glob(os.path.join(CDIR, f"{tag}.csv"))
                    + glob.glob(os.path.join(CDIR, f"{tag}_s*.csv")))
        for f in fs:
            c = arm_curve(f, W)
            if c is None:
                continue
            try:
                lam, r2 = fit_lam(*c)
            except Exception:
                continue
            if r2 >= 0.6:
                arms.setdefault(W, []).append((c[0], c[1], lam))
    if len(arms) < 4:
        print("insufficient arms"); return
    Ws = sorted(arms)
    lam_W = {W: float(np.mean([a[2] for a in arms[W]])) for W in Ws}
    print("== per-W relaxation rate (pooled seeds) ==")
    for W in Ws:
        print(f"  W={W:5d} rho={rho_of(W):8.4f}  lam={lam_W[W]:6.2f}  "
              f"(n={len(arms[W])})")
    lamv = np.array([lam_W[W] for W in Ws]); Wv = np.array(Ws, float)
    sp = float(spearmanr(Wv, lamv).statistic)
    span = float(lamv.max() / max(lamv.min(), 1e-9))
    print(f"\nfull-grid Spearman(lam,W)={sp:+.2f}  span={span:.2f}x  "
          f"({len(Ws)} widths)")

    wide = any(W not in ORIG_W for W in Ws)
    report = {"W": Ws, "lam": [lam_W[W] for W in Ws], "rho": [rho_of(W) for W in Ws],
              "spearman": sp, "span": span}
    if not wide:
        # g2d3 only: report exp-form AMBIG as before
        print("g2d3 (original ladder only) -- see lamrho_prereg.json verdict")
        report["verdict"] = "g2d3 ladder only"
    else:
        # g2d3b: out-of-sample log-law + MAE margin
        orig = [W for W in Ws if W in ORIG_W]
        held = [W for W in Ws if W not in ORIG_W]
        a, b = np.polyfit(np.log10([float(W) for W in orig]),
                          [lam_W[W] for W in orig], 1)  # lam = a*log10W + b
        pred_held = a * np.log10([float(W) for W in held]) + b
        true_held = np.array([lam_W[W] for W in held])
        r2_held = 1 - np.sum((true_held - pred_held) ** 2) / max(
            np.sum((true_held - true_held.mean()) ** 2), 1e-30)
        print(f"\nout-of-sample log-law (fit on {orig}, predict {held}):")
        print(f"  lam = {a:.2f}*log10(W) + {b:.2f}; held-out R2 = {r2_held:.2f}")
        # kernel-MAE margin: per-arm fixed global-best lam vs log-law lam(W)
        def loglaw_lam(W):
            return a * np.log10(max(float(W), 1.0)) + b
        # global best single lam minimizing pooled fixed-lam MAE
        allc = [(x, y, W) for W in Ws for (x, y, _) in arms[W]]
        gobj = lambda L: float(np.mean([fixedlam_mae(x, y, L) for x, y, _ in allc]))
        gbest = float(minimize_scalar(gobj, bounds=(0.5, 50), method="bounded").x)
        mae_fixed = np.mean([fixedlam_mae(x, y, gbest) for x, y, _ in allc])
        mae_loglaw = np.mean([fixedlam_mae(x, y, max(loglaw_lam(W), 0.1))
                              for x, y, W in allc])
        margin = (mae_fixed - mae_loglaw) / mae_fixed
        print(f"  pooled curve MAE: best-fixed-lam(={gbest:.1f}) {mae_fixed:.5f} "
              f"vs log-law-lam {mae_loglaw:.5f} -> margin {margin*100:+.1f}%")
        report.update(loglaw_a=float(a), loglaw_b=float(b),
                      heldout_r2=float(r2_held), mae_fixed=float(mae_fixed),
                      mae_loglaw=float(mae_loglaw), mae_margin=float(margin),
                      global_best_lam=gbest)
        if sp <= -0.9 and span >= 3 and r2_held >= 0.8 and margin >= 0.20:
            v = "RATE_READS_RHO_CONFIRMED -> ship concentration-dependent rate"
        elif sp <= -0.9 and span >= 3:
            v = ("RATE_READS_RHO_WEAK: real & >3x but held-out R2<0.8 or "
                 "margin<20% -> keep fixed-lam, report measured")
        elif span < 2 or sp > -0.7:
            v = "FLAT: falsified"
        else:
            v = "AMBIG"
        report["verdict"] = v
        print("VERDICT:", v)
    json.dump(report, open(os.path.join(REPO, "results", "formula_lab",
              "LAMRHO_REPORT.json"), "w"), indent=1)
    print("saved results/formula_lab/LAMRHO_REPORT.json")


if __name__ == "__main__":
    main()
