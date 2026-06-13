#!/usr/bin/env python3
"""g2d3 verdict (lamrho_prereg.json): does the relaxation rate lam read off
the local decrement concentration rho?  Per-arm S-time relaxation fit on the
HOLD window -> lam(W); then test Lambda(rho)=lam_inf+(lam_0-lam_inf)*exp(-rho/rho*)."""
import glob
import json
import os
import sys

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "..", "represent", "repro"))
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", ".."))
sys.path.insert(0, os.path.join(REPO, "represent", "repro"))
import analyze_curves as AC  # noqa: E402

CDIR = os.path.join(REPO, "represent", "results", "curves_lamrho")
PEAK, ETA2, TRUNK = 1.5e-3, 1e-4, 3000
WIDTHS = {"W1": 1, "W10": 10, "W40": 40, "W160": 160, "W640": 640}


def cumS_of(etas):
    return np.cumsum(etas)


def lam_of(path, W):
    rows = np.genfromtxt(path, delimiter=",", names=True)
    step = np.atleast_1d(rows["step"]).astype(int)
    loss = np.atleast_1d(rows["eval_loss"]).astype(float)
    lr = np.atleast_1d(rows["lr"]).astype(float)
    sm = AC.smooth_by_step(step, loss)
    # S-time since cooldown end; hold window only
    wend = TRUNK + W
    # reconstruct cumulative S from step 3000 using the logged lr
    order = np.argsort(step)
    step, sm, lr = step[order], sm[order], lr[order]
    S = np.cumsum(lr * np.gradient(step.astype(float)))  # approx integral
    m = step >= wend
    if m.sum() < 8:
        return None
    x = (S[m] - S[m][0])
    y = sm[m]

    def mdl(x, F, A, lam):
        return F + A * np.exp(-lam * x)
    try:
        # x is in S-units (cumulative eta); lam ~ 1/S-scale
        po, _ = curve_fit(mdl, x, y, p0=[y[-1], max(y[0] - y[-1], 1e-3), 10.0],
                          maxfev=60000,
                          bounds=([0, 1e-4, 0.05], [5, 2, 200]))
        pred = mdl(x, *po)
        r2 = 1 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2),
                                               1e-30)
        return po[2], r2, float(po[1])
    except Exception:
        return None


def rho_of(W):
    if W <= 1:
        e = np.array([ETA2]); drop = np.array([PEAK - ETA2]); cw = np.array([PEAK])
    else:
        e = np.linspace(PEAK, ETA2, W + 1)[1:]
        cw = np.concatenate([[PEAK], e[:-1]])
        drop = np.maximum(cw - e, 0)
    return float(np.mean(drop / np.maximum(cw, 1e-9)))


def main():
    lams, rhos, Ws = [], [], []
    print("== per-W relaxation rate (pooled seeds) ==")
    for tag, W in WIDTHS.items():
        fs = sorted(glob.glob(os.path.join(CDIR, f"{tag}.csv"))
                    + glob.glob(os.path.join(CDIR, f"{tag}_s*.csv")))
        ls = []
        for f in fs:
            r = lam_of(f, W)
            if r and r[1] >= 0.6:
                ls.append(r[0])
        if not ls:
            print(f"  {tag}: no arm passes r2 gate")
            continue
        lam = float(np.mean(ls))
        rho = rho_of(W)
        lams.append(lam); rhos.append(rho); Ws.append(W)
        print(f"  {tag:5s} W={W:4d} rho={rho:7.4f}  lam={lam:6.2f}  (n={len(ls)})")
    if len(lams) < 4:
        print("insufficient arms for verdict"); return
    lams = np.array(lams); rhos = np.array(rhos); Ws = np.array(Ws)
    sp = float(spearmanr(Ws, lams).statistic)
    span = float(lams.max() / max(lams.min(), 1e-9))
    print(f"\nSpearman(lam, W) = {sp:+.2f}  (expect <=-0.8 if lam reads rho)")
    print(f"lam span = {span:.2f}x  (expect >=3x)")

    def Lam(rho, l0, linf, rs):
        return linf + (l0 - linf) * np.exp(-rho / rs)
    fit_ok, l0 = False, None
    try:
        po, _ = curve_fit(Lam, rhos, lams, p0=[15, 4, 1.0], maxfev=60000,
                          bounds=([5, 0.5, 1e-3], [30, 10, 100]))
        pred = Lam(rhos, *po)
        r2 = 1 - np.sum((lams - pred) ** 2) / max(
            np.sum((lams - lams.mean()) ** 2), 1e-30)
        l0, linf, rs = po
        print(f"Lambda(rho) fit: lam_0={l0:.1f} lam_inf={linf:.1f} "
              f"rho*={rs:.3f}  R2={r2:.2f}")
        fit_ok = r2 >= 0.8 and 10 <= l0 <= 25 and 1 <= linf <= 7
    except Exception as e:
        print("Lambda(rho) fit failed:", e)
        r2 = 0.0

    if sp <= -0.8 and span >= 3 and fit_ok:
        v = ("LAMBDA_READS_RHO (pending the >=20%% pooled-MAE kernel margin "
             "check vs best fixed-lam)")
    elif span < 2 or sp > -0.5:
        v = "LAMBDA_FLAT: rate not a smooth function of rho -> redesign FALSIFIED"
    else:
        v = "AMBIG: report as measured"
    print("VERDICT:", v)
    json.dump({"W": Ws.tolist(), "rho": rhos.tolist(), "lam": lams.tolist(),
               "spearman": sp, "span": span, "lam_fit_r2": float(r2),
               "verdict": v},
              open(os.path.join(REPO, "results", "formula_lab",
                                "LAMRHO_REPORT.json"), "w"), indent=1)
    print("saved results/formula_lab/LAMRHO_REPORT.json")


if __name__ == "__main__":
    main()
