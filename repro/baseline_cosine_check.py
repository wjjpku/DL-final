#!/usr/bin/env python3
"""Redo the residual / magnitude analysis with a TRUE cosine-only MPL baseline.

Issue caught: MPL_PRECOMPUTED_INIT was fit on the official split
[cosine_24000, constant_24000, wsdcon_9] -- it has seen a decay curve (wsdcon_9).
The paper claims "MPL fit on cosine". Here we fit MPL on cosine ONLY
(cosine_24000 + cosine_72000), recompute everything, and compare.
"""
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import load_curve, mpl_predict, compute_s1, MPL_PRECOMPUTED_INIT, SCALES, PEAK_LR
from validate_theory import fit_mpl, mpl_pred, F_MPL
from deep_stime import stime_feature
from nonadiabatic_theory import fit_origin

LAM = 10.0
DECAY = ["wsd_20000_24000.csv", "wsdld_20000_24000.csv"]
COSINE = ["cosine_24000.csv", "cosine_72000.csv"]


def dLeq_deta(scale, p):
    """Noise-floor slope dL_eq/deta using THIS MPL's backbone (L0,A,alpha)."""
    L0, A, alpha = p[0], p[1], p[2]
    etas, floors = [], []
    for n, mul in [("wsdcon_3.csv", 3), ("wsdcon_9.csv", 9), ("wsdcon_18.csv", 18)]:
        c = load_curve(scale, n)
        backbone = L0 + A * np.power(compute_s1(c), -alpha)
        floors.append(float(np.mean((c.loss - backbone)[-5:])))
        etas.append(mul * 1e-5)
    return np.polyfit(np.array(etas), np.array(floors), 1)[0]


def analyze(label, get_params):
    print(f"\n===== {label} =====")
    print(f"  {'scale':>5s} {'R2(resid~Drop)':>14s} {'kappa_fit':>10s} "
          f"{'kappa_pred':>11s} {'ratio c':>8s}")
    ratios = []
    for s in SCALES:
        p = get_params(s)
        xs, ys = [], []
        for n in DECAY:
            c = load_curve(s, n)
            ys.append(c.loss - mpl_pred(p, c, fast=False)); xs.append(stime_feature(c, LAM))
        X, Y = np.concatenate(xs), np.concatenate(ys)
        kf, r2 = fit_origin(X, Y)
        kp = dLeq_deta(s, p) * PEAK_LR
        ratios.append(kf / kp)
        print(f"  {s:>4s}M {r2:14.3f} {kf:10.4f} {kp:11.4f} {kf/kp:8.2f}")
    ratios = np.array(ratios)
    print(f"  -> ratio c: mean {ratios.mean():.2f}, CV {ratios.std()/ratios.mean()*100:.0f}%")


# Baseline A: the official-split precomputed params (what we used before)
analyze("OFFICIAL-split MPL (incl. wsdcon_9) -- what we used before",
        lambda s: MPL_PRECOMPUTED_INIT[s])

# Baseline B: TRUE cosine-only fit
cos_fits = {}
for s in SCALES:
    train = [load_curve(s, n) for n in COSINE]
    cos_fits[s] = fit_mpl(train, MPL_PRECOMPUTED_INIT[s], F_MPL)
analyze("COSINE-ONLY MPL (cosine_24000 + cosine_72000) -- the paper's claim",
        lambda s: cos_fits[s])

# how big is the WSD residual under each baseline (mean |resid| on decay tails)?
print("\n----- mean residual magnitude on wsd/wsdld decay tail (last 20%) -----")
for label, gp in [("official-split", lambda s: MPL_PRECOMPUTED_INIT[s]),
                  ("cosine-only", lambda s: cos_fits[s])]:
    vals = []
    for s in SCALES:
        for n in DECAY:
            c = load_curve(s, n); r = c.loss - mpl_pred(gp(s), c, fast=False)
            vals.append(np.mean(r[int(0.8*len(r)):]))
    print(f"  {label:16s} mean tail residual = {np.mean(vals):+.4f}")
