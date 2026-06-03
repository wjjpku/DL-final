#!/usr/bin/env python3
"""Extra evidence: the non-adiabatic residual grows with decay sharpness.

For each schedule we report the END-OF-CURVE (tail) residual = how far the loss
ends ABOVE the well-fit MPL. A rate-dependent (non-adiabatic) effect predicts:
  cosine (gradual decay) ~ 0  <  two-stage drops (relax during the long plateau)
  <  WSD / WSDLD (sharpest decay, curve ends right after it -> undamped).
Same final LR, faster sweep => bigger lag: the textbook non-adiabatic signature.
"""
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import load_curve, mpl_predict, MPL_PRECOMPUTED_INIT, SCALES

CURVES = [
    ("cosine_72000.csv", "cosine (gradual, ~22k-step decay)"),
    ("wsdcon_18.csv",    "two-stage -> 18e-5 (then plateau)"),
    ("wsdcon_9.csv",     "two-stage -> 9e-5  (then plateau)"),
    ("wsdcon_3.csv",     "two-stage -> 3e-5  (then plateau)"),
    ("wsdld_20000_24000.csv", "WSDLD (sharp 4k-step linear decay)"),
    ("wsd_20000_24000.csv",   "WSD   (sharp 4k-step decay)"),
]


def tail_resid(scale, name, frac=0.2):
    p = MPL_PRECOMPUTED_INIT[scale]
    c = load_curve(scale, name)
    r = c.loss - mpl_predict(p, c)
    return float(np.mean(r[int((1 - frac) * len(r)):]))


print("=" * 72)
print("End-of-curve (tail) residual = loss minus well-fit MPL, x1e3")
print("ordered by decay sharpness (gradual -> sharp)")
print("=" * 72)
print(f"  {'schedule':36s} {'25M':>7s} {'100M':>7s} {'400M':>7s}")
for name, desc in CURVES:
    vals = [tail_resid(s, name) * 1e3 for s in SCALES]
    print(f"  {desc:36s} {vals[0]:7.2f} {vals[1]:7.2f} {vals[2]:7.2f}")
print("\n(positive = loss ends ABOVE MPL = un-relaxed non-adiabatic lag)")
print("cosine ~ 0; two-stage small (relaxed during plateau); WSD/WSDLD largest.")
