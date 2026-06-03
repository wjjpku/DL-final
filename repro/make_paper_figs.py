#!/usr/bin/env python3
"""Generate the two key figures for the non-adiabatic paper."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import load_curve, mpl_predict, MPL_PRECOMPUTED_INIT, SCALES, PEAK_LR
from deep_stime import stime_feature
from nonadiabatic_theory import fit_origin

OUT = REPO.parent / "paper" / "figs"
OUT.mkdir(parents=True, exist_ok=True)
LAM = 10.0

# ---------- Figure A: tau ~ 1/eta (p = 1.00 +- 0.18) ----------
lrb = np.array([3e-5, 9e-5, 18e-5])
tau = {"25M": [2998, 2184, 1161], "100M": [2906, 1616, 394], "400M": [2884, 1598, 496]}
fig, ax = plt.subplots(figsize=(4.3, 3.4))
colors = {"25M": "#4C72B0", "100M": "#55A868", "400M": "#C44E52"}
for k in ["25M", "100M", "400M"]:
    ax.loglog(lrb, tau[k], "o-", color=colors[k], label=f"{k}", ms=6)
# slope -1 guide (tau ∝ 1/eta), anchored at the 100M first point
g = 2906 * (3e-5) / lrb
ax.loglog(lrb, g, "k--", lw=1.2, label=r"slope $-1$ ($\tau\propto1/\eta$)")
ax.set_xlabel(r"stage-2 learning rate $\eta$")
ax.set_ylabel(r"measured relaxation time $\tau$ (steps)")
ax.set_title(r"$\tau\propto\eta^{-p}$, pooled $p=1.00\pm0.18$", fontsize=10)
ax.legend(fontsize=8, frameon=False)
ax.grid(True, which="both", ls=":", alpha=0.4)
fig.tight_layout()
fig.savefig(OUT / "fig_tau_scaling.png", dpi=200)
print("wrote fig_tau_scaling.png")

# ---------- Figure B: residual = non-adiabatic lag ~ kappa * DropRelaxS ----------
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0), sharey=False)
for ax, scale in zip(axes, SCALES):
    p = MPL_PRECOMPUTED_INIT[scale]
    # fit kappa on wsd+wsdld residual
    xs, ys, steps = [], [], {}
    for n in ["wsd_20000_24000.csv", "wsdld_20000_24000.csv"]:
        c = load_curve(scale, n)
        ys.append(c.loss - mpl_predict(p, c)); xs.append(stime_feature(c, LAM))
    kap, r2 = fit_origin(np.concatenate(xs), np.concatenate(ys))
    c = load_curve(scale, "wsd_20000_24000.csv")
    resid = c.loss - mpl_predict(p, c)
    pred = kap * stime_feature(c, LAM)
    ax.plot(c.step, resid * 1e3, color="#444", lw=1.4, label="MPL residual")
    ax.plot(c.step, pred * 1e3, color="#C44E52", lw=1.8, ls="--",
            label=r"$\kappa\,$DropRelaxS")
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_title(f"{scale}M  (wsd),  $R^2$={r2:.2f}", fontsize=10)
    ax.set_xlabel("step")
    if scale == "25":
        ax.set_ylabel(r"loss residual $\times10^{3}$")
        ax.legend(fontsize=8, frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "fig_residual_fit.png", dpi=200)
print("wrote fig_residual_fit.png")
