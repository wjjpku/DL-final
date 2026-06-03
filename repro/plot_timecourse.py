#!/usr/bin/env python3
"""Direct time-course check: does the leaky-bucket model match the measured lag?

On wsdcon the LR steps down at 8000 then holds constant -> the lag should SPIKE
(injection) then decay exponentially (leak), faster for larger stage-2 LR
(tau ~ 1/eta). We overlay the measured MPL residual against floor + kappa*DropRelaxS.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import load_curve, mpl_predict, MPL_PRECOMPUTED_INIT
from deep_stime import stime_feature

OUT = REPO.parent / "paper" / "figs"
LAM = 10.0
SCALE = "400"
curves = [("wsdcon_3.csv", 3e-5, "#4C72B0"), ("wsdcon_9.csv", 9e-5, "#55A868"),
          ("wsdcon_18.csv", 18e-5, "#C44E52")]

p = MPL_PRECOMPUTED_INIT[SCALE]
fig, ax = plt.subplots(figsize=(5.2, 3.6))
for name, lrb, col in curves:
    c = load_curve(SCALE, name)
    resid = c.loss - mpl_predict(p, c)
    feat = stime_feature(c, LAM)
    # fit floor + kappa*feat (floor = separate MPL noise-floor error; kappa = lag amplitude)
    def obj(x):
        return np.sum((resid - (x[0] + x[1] * feat)) ** 2)
    r = minimize(obj, [0.0, 0.05], method="Nelder-Mead")
    floor, kap = r.x
    model = floor + kap * feat
    ax.plot(c.step, resid * 1e3, color=col, lw=1.3, alpha=0.9,
            label=fr"$\eta_2={lrb*1e5:.0f}{{\times}}10^{{-5}}$ (data)")
    ax.plot(c.step, model * 1e3, color=col, lw=2.2, ls="--", alpha=0.9)
ax.axvline(8000, color="gray", lw=0.8, ls=":")
ax.text(8100, ax.get_ylim()[1]*0.92, "LR step-down", fontsize=8, color="gray")
ax.set_xlabel("step"); ax.set_ylabel(r"loss residual $\times10^{3}$")
ax.set_title(f"{SCALE}M wsdcon: lag spikes then leaks (dashed = leaky-bucket model)",
             fontsize=10)
ax.legend(fontsize=8, frameon=False, title="solid=residual, dashed=model")
ax.grid(True, ls=":", alpha=0.35)
fig.tight_layout()
fig.savefig(OUT / "fig_timecourse.png", dpi=200)
print("wrote fig_timecourse.png")

# also report the per-curve decay-shape agreement (R^2 of floor+kappa*feat)
print("\nper-curve fit (floor + kappa*DropRelaxS), post-step R^2:")
for name, lrb, _ in curves:
    c = load_curve(SCALE, name)
    resid = c.loss - mpl_predict(p, c); feat = stime_feature(c, LAM)
    m = c.step > 8050
    def obj(x): return np.sum((resid - (x[0] + x[1]*feat))**2)
    r = minimize(obj, [0.0, 0.05], method="Nelder-Mead"); model = r.x[0]+r.x[1]*feat
    ss = np.sum((resid[m]-resid[m].mean())**2)
    r2 = 1 - np.sum((resid[m]-model[m])**2)/ss
    print(f"  {name:14s} eta2={lrb:.0e}  kappa={r.x[1]:.3f}  post-step R^2={r2:.3f}")
