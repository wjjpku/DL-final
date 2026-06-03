#!/usr/bin/env python3
"""S-time kernel residual regression: does kappa = eta_peak * dL_eq/deta with NO fudge?

With the step-time kernel (nonadiabatic_theory.py) the fitted kappa was ~0.3x the
predicted eta_peak*dL_eq/deta, the 0.3 explained by the finite decay duration the
step-time kernel ignored. The S-time kernel exp(-lambda_slow (S-S')) accounts for
the relaxation correctly, so the fudge should vanish: kappa_fit ~ kappa_pred.

Regress cosine-fit MPL residual on the S-time feature (lambda_slow=10) through the
origin, per scale; compare kappa_fit to the independent kappa_pred = eta_peak *
(dL_eq/deta) from the noise-floor slope.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import (  # noqa: E402
    load_curve, compute_s1, mpl_predict, MPL_PRECOMPUTED_INIT, SCALES, PEAK_LR,
)
from deep_stime import stime_feature  # noqa: E402
from nonadiabatic_theory import fit_origin, estimate_dLeq_deta  # noqa: E402

LAM = 10.0
DECAY = ["wsd_20000_24000.csv", "wsdld_20000_24000.csv"]


def main():
    print("=" * 74)
    print(f"S-time kernel (lambda_slow={LAM:.0f}) residual regression vs independent prediction")
    print("=" * 74)
    print(f"  {'scale':>5s} {'R^2(decay)':>11s} {'kappa_fit':>10s} {'kappa_pred':>11s} "
          f"{'ratio':>7s}")
    fits, preds = [], []
    for scale in SCALES:
        p = MPL_PRECOMPUTED_INIT[scale]
        xs, ys = [], []
        for n in DECAY:
            c = load_curve(scale, n)
            ys.append(c.loss - mpl_predict(p, c))
            xs.append(stime_feature(c, LAM))
        X, Y = np.concatenate(xs), np.concatenate(ys)
        kap, r2 = fit_origin(X, Y)
        slope, _, _ = estimate_dLeq_deta(scale)
        kap_pred = slope * PEAK_LR
        fits.append(kap); preds.append(kap_pred)
        print(f"  {scale:>4s}M {r2:11.3f} {kap:10.4f} {kap_pred:11.4f} {kap/kap_pred:7.2f}")
    fits, preds = np.array(fits), np.array(preds)
    print("\n  ratio kappa_fit / kappa_pred across scales:",
          np.round(fits / preds, 2),
          f" (CV={np.std(fits/preds)/np.mean(fits/preds)*100:.0f}%)")
    print("  -> a near-constant ratio means kappa is PREDICTABLE from the noise floor")
    print(f"     (theory wants ratio ~1; step-time kernel gave ~0.3).")
    Ns = np.array([25.0, 100.0, 400.0])
    print(f"\n  kappa_fit ~ N^{np.polyfit(np.log(Ns), np.log(fits), 1)[0]:.2f};"
          f"  dL_eq/deta(=kappa_pred/eta_peak) ~ N^"
          f"{np.polyfit(np.log(Ns), np.log(preds), 1)[0]:.2f}")


if __name__ == "__main__":
    main()
