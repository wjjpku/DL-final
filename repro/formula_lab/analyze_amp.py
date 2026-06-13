#!/usr/bin/env python3
"""g4 T1 (amp_rho_prereg.json): does the VISIBLE lag amplitude A depend on
decrement concentration rho?  Amplitude analog of g2d3's rate result, on the
same 11 fixed-depth varied-width lamrho arms.  ZERO GPU."""
import glob
import json
import os
import sys

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", ".."))
sys.path.insert(0, os.path.join(REPO, "represent", "repro"))
import analyze_curves as AC  # noqa: E402

CDIR = os.path.join(REPO, "represent", "results", "curves_lamrho")
PEAK, ETA2, TRUNK = 1.5e-3, 1e-4, 3000
DEPTH = PEAK - ETA2
WIDTHS = {"W1": 1, "W4": 4, "W10": 10, "W40": 40, "W80": 80, "W160": 160,
          "W320": 320, "W640": 640, "W1280": 1280, "W2560": 2560, "W5120": 5120}


def arm_A(path, W):
    rows = np.genfromtxt(path, delimiter=",", names=True)
    step = np.atleast_1d(rows["step"]).astype(int)
    loss = np.atleast_1d(rows["eval_loss"]).astype(float)
    lr = np.atleast_1d(rows["lr"]).astype(float)
    sm = AC.smooth_by_step(step, loss)
    o = np.argsort(step); step, sm, lr = step[o], sm[o], lr[o]
    S = np.cumsum(lr * np.gradient(step.astype(float)))
    m = step >= (TRUNK + W)
    if m.sum() < 8:
        return None
    x = S[m] - S[m][0]; y = sm[m]

    def mdl(x, F, A, lam):
        return F + A * np.exp(-lam * x)
    try:
        po, _ = curve_fit(mdl, x, y, p0=[y[-1], max(y[0] - y[-1], 1e-3), 10.0],
                          maxfev=60000, bounds=([0, 1e-4, 0.05], [5, 2, 200]))
        pred = mdl(x, *po)
        r2 = 1 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2),
                                               1e-30)
        return po[1], r2          # A (excess amplitude), r2
    except Exception:
        return None


def rho_of(W):
    if W <= 1:
        cw = np.array([PEAK]); drop = np.array([DEPTH])
    else:
        e = np.linspace(PEAK, ETA2, W + 1)[1:]
        cw = np.concatenate([[PEAK], e[:-1]]); drop = np.maximum(cw - e, 0)
    return float(np.mean(drop / np.maximum(cw, 1e-9)))


def main():
    A_W, rho_W, Ws = {}, {}, []
    for tag, W in WIDTHS.items():
        fs = sorted(glob.glob(os.path.join(CDIR, f"{tag}.csv"))
                    + glob.glob(os.path.join(CDIR, f"{tag}_s*.csv")))
        As = []
        for f in fs:
            r = arm_A(f, W)
            if r and r[1] >= 0.6:
                As.append(r[0])
        if As:
            A_W[W] = float(np.mean(As)); rho_W[W] = rho_of(W); Ws.append(W)
    Ws = sorted(Ws)
    print("== visible excess amplitude A(W) (pooled seeds) ==")
    for W in Ws:
        a_norm = A_W[W] / DEPTH
        print(f"  W={W:5d} rho={rho_W[W]:8.4f}  A={A_W[W]:.4f}  "
              f"a=A/depth={a_norm:7.2f}")
    rho = np.array([rho_W[W] for W in Ws])
    a = np.array([A_W[W] / DEPTH for W in Ws])
    Wv = np.array(Ws, float)
    sp_rho = float(spearmanr(rho, a).statistic)
    span = float(a.max() / max(a.min(), 1e-9))
    print(f"\nSpearman(a, rho) = {sp_rho:+.2f}; a span = {span:.2f}x")
    # Hill/Weibull gate phi(rho)=1-exp(-(rho/rc)^q), a = a_max*phi
    gate_r2 = None
    try:
        def gate(rho, amax, rc, q):
            return amax * (1 - np.exp(-(rho / rc) ** q))
        po, _ = curve_fit(gate, rho, a, p0=[a.max(), np.median(rho), 1.0],
                          maxfev=60000,
                          bounds=([a.max() * 0.5, 1e-4, 0.2],
                                  [a.max() * 3, 100, 5]))
        pred = gate(rho, *po)
        gate_r2 = 1 - np.sum((a - pred) ** 2) / max(
            np.sum((a - a.mean()) ** 2), 1e-30)
        print(f"Hill gate a=amax*(1-exp(-(rho/rc)^q)): amax={po[0]:.2f} "
              f"rc={po[1]:.4f} q={po[2]:.2f}  R2={gate_r2:.2f}")
    except Exception as e:
        print("Hill gate fit failed:", e)
    if abs(sp_rho) < 0.5 and span < 2:
        v = ("AMP_FLAT: visible amplitude is concentration-INDEPENDENT on the "
             "width axis -> deficit is depth/eta-bound (public-scale wall)")
    elif gate_r2 and gate_r2 >= 0.8 and abs(sp_rho) >= 0.7:
        v = ("AMP_GATE width-axis supported (pending T2 held-out kappa margin "
             "if a GPU depth ladder is run)")
    elif abs(sp_rho) >= 0.7:
        v = "AMP_GATE_WEAK: varies with rho but no clean gate form"
    else:
        v = "AMBIG"
    print("\nT1 VERDICT:", v)
    json.dump({"W": Ws, "rho": rho.tolist(), "a_norm": a.tolist(),
               "spearman_a_rho": sp_rho, "span": span,
               "hill_r2": gate_r2, "t1_verdict": v},
              open(os.path.join(REPO, "results", "formula_lab",
                                "AMP_RHO_REPORT.json"), "w"), indent=1)
    print("saved results/formula_lab/AMP_RHO_REPORT.json")


if __name__ == "__main__":
    main()
