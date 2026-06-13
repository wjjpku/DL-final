#!/usr/bin/env python3
"""g2d3c T1 (ml_kernel_prereg.json): is the lam(W) ladder a power law
(fractional-memory signature lam_eff ~ W^-(1-beta)) or merely log-linear?
Reads LAMRHO_REPORT.json (written by analyze_lamrho.py on the full wide grid).
T2 (held-out kernel MAE) is run separately via explore_kernels.py."""
import json
import os

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", ".."))
REP = os.path.join(REPO, "results", "formula_lab", "LAMRHO_REPORT.json")


def aic(y, pred, k):
    n = len(y)
    rss = float(np.sum((y - pred) ** 2))
    return n * np.log(rss / n + 1e-30) + 2 * k, rss


def main():
    d = json.load(open(REP))
    W = np.array(d["W"], float)
    lam = np.array(d["lam"], float)
    if len(W) < 7:
        print(f"only {len(W)} widths in LAMRHO_REPORT -- run g2d3b first")
        return
    lw = np.log10(W)
    # (a) log-linear: lam = A - B*log10 W
    A, B = np.polyfit(lw, lam, 1)[::-1] if False else np.polyfit(lw, lam, 1)
    # np.polyfit returns [slope, intercept]
    slope_ll, icpt_ll = np.polyfit(lw, lam, 1)
    pred_ll = slope_ll * lw + icpt_ll
    aic_ll, rss_ll = aic(lam, pred_ll, 2)
    # (b) power law: log10 lam = C - (1-beta) log10 W
    llam = np.log10(lam)
    s_pl, c_pl = np.polyfit(lw, llam, 1)        # s_pl = -(1-beta)
    pred_pl = 10 ** (s_pl * lw + c_pl)
    aic_pl, rss_pl = aic(lam, pred_pl, 2)
    beta = 1.0 + s_pl                            # s_pl = beta-1
    r2_pl = 1 - rss_pl / max(np.sum((lam - lam.mean()) ** 2), 1e-30)
    r2_ll = 1 - rss_ll / max(np.sum((lam - lam.mean()) ** 2), 1e-30)
    print(f"widths W = {W.tolist()}")
    print(f"lam(W)   = {np.round(lam,2).tolist()}")
    print(f"\nT1 ladder-signature fits (n={len(W)} widths):")
    print(f"  log-linear lam = {icpt_ll:.2f} {slope_ll:+.2f}*log10W   "
          f"R2={r2_ll:.3f} AIC={aic_ll:.1f}")
    print(f"  power law  lam = {10**c_pl:.2f}*W^{s_pl:+.3f}  (beta={beta:.3f})  "
          f"R2={r2_pl:.3f} AIC={aic_pl:.1f}")
    # log-linear extrapolation sanity: where does it cross zero?
    wzero = 10 ** (-icpt_ll / slope_ll) if slope_ll < 0 else np.inf
    print(f"  log-linear hits lam=0 at W={wzero:.0f} (unphysical if within range)")
    daic = aic_ll - aic_pl   # positive => power law preferred
    pl_better = daic > 2 and r2_pl >= 0.9 and 0 < beta < 1
    print(f"\n  delta-AIC (log-linear - power) = {daic:+.1f} "
          f"(>2 favors power law)")
    if pl_better:
        t1 = (f"POWER-LAW supported: scale-free/fractional signature, "
              f"beta={beta:.2f} in (0,1), R2={r2_pl:.2f}, lower AIC")
    elif r2_pl >= 0.9 and r2_ll >= 0.9 and abs(daic) <= 2:
        t1 = ("INDISTINGUISHABLE over this range: both fit; need wider W to "
              "separate -> lean log-linear (simpler), no fractional claim")
    else:
        t1 = "POWER-LAW NOT supported (beta out of (0,1) or worse fit)"
    print("\nT1 VERDICT:", t1)
    print("T2 (held-out kernel MAE exp1 vs heavy-tailed): run "
          "repro/formula_lab/explore_kernels.py; closed direction = collapses.")
    json.dump({"W": W.tolist(), "lam": lam.tolist(), "beta": float(beta),
               "r2_powerlaw": float(r2_pl), "r2_loglinear": float(r2_ll),
               "delta_aic_ll_minus_pl": float(daic), "t1_verdict": t1},
              open(os.path.join(REPO, "results", "formula_lab",
                                "ML_KERNEL_REPORT.json"), "w"), indent=1)
    print("saved results/formula_lab/ML_KERNEL_REPORT.json")


if __name__ == "__main__":
    main()
