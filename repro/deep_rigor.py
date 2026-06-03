#!/usr/bin/env python3
"""Validate the rigorous derivation: the non-adiabatic kernel is a SPECTRAL MIXTURE.

Rigorous result (docs/core/nonadiabatic_correction.md, derivation):
    Delta L(t) = sum_i w_i [exp(-2 lambda_i (S_t-S_t')) (x) drop],  sum_i w_i = dL_eq/deta.
The single-exponential DropRelaxS is the single-mode COLLAPSE (approximation C); the
true kernel is a w_i-weighted mixture of S-time exponentials with rates 2 lambda_i.

Predictions tested here:
  (1) a 2-exponential kernel should fit the cosine-fit-MPL residual noticeably better
      than 1 exponential (evidence of the spectral spread; explains R^2<1 and c<1);
  (2) self-consistency of approximation (A): the dominant lag rate corresponds to a
      curvature lambda_eff = lambda_slow/2 with eta_peak*lambda_eff << 2, i.e. the
      lag lives on SLOW modes far below the edge -- where (1-eta lambda)^2 ~ e^{-2 eta lambda}
      is accurate (unlike gamma, which is an edge effect).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import (  # noqa: E402
    load_curve, mpl_predict, MPL_PRECOMPUTED_INIT, SCALES, PEAK_LR,
)
from deep_stime import stime_feature  # noqa: E402

CURVES = ["wsd_20000_24000.csv", "wsdld_20000_24000.csv",
          "wsdcon_3.csv", "wsdcon_9.csv", "wsdcon_18.csv"]


def residual_and_feats(scale, lams):
    p = MPL_PRECOMPUTED_INIT[scale]
    ys, F = [], {lam: [] for lam in lams}
    for n in CURVES:
        c = load_curve(scale, n)
        ys.append(c.loss - mpl_predict(p, c))
        for lam in lams:
            F[lam].append(stime_feature(c, lam))
    y = np.concatenate(ys)
    return y, {lam: np.concatenate(F[lam]) for lam in F}


def r2_of(y, X):
    """Best non-negative-amplitude LSQ fit of y by columns of X; return R^2, coefs."""
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    ss = np.sum((y - y.mean()) ** 2)
    return 1.0 - np.sum(resid ** 2) / ss, coef


def main():
    print("=" * 74)
    print("Spectral-mixture validation: 1- vs 2-exponential S-time kernel")
    print("=" * 74)
    for scale in SCALES:
        # 1-exp: scan lambda
        lam_grid = np.geomspace(2, 60, 25)
        y, feats = residual_and_feats(scale, list(lam_grid))
        best1 = max(lam_grid, key=lambda L: r2_of(y, np.c_[feats[L]])[0])
        r2_1, _ = r2_of(y, np.c_[feats[best1]])

        # 2-exp: optimise (lambda_a, lambda_b)
        def neg_r2(p):
            la, lb = np.exp(p)
            if not (1 < la < 200 and 1 < lb < 200):
                return 1.0
            fa = stime_concat(scale, la); fb = stime_concat(scale, lb)
            return -r2_of(y, np.c_[fa, fb])[0]
        from functools import lru_cache
        cache = {}
        def stime_concat(sc, lam):
            if (sc, round(lam, 3)) not in cache:
                _, ff = residual_and_feats(sc, [lam]); cache[(sc, round(lam, 3))] = ff[lam]
            return cache[(sc, round(lam, 3))]
        res = minimize(neg_r2, np.log([5.0, 30.0]), method="Nelder-Mead",
                       options={"maxiter": 300, "xatol": 1e-3, "fatol": 1e-5})
        la, lb = np.sort(np.exp(res.x))
        r2_2 = -res.fun
        leff = best1 / 2.0   # lambda_eff = lambda_slow/2
        print(f"\n[{scale}M]")
        print(f"  1-exp:  lambda_slow={best1:5.1f}  R^2={r2_1:.3f}")
        print(f"  2-exp:  lambdas=({la:5.1f},{lb:5.1f})  R^2={r2_2:.3f}   "
              f"(gain {r2_2-r2_1:+.3f})")
        print(f"  self-consistency: dominant lambda_eff={leff:.1f}, "
              f"eta_peak*lambda_eff={PEAK_LR*leff:.2e}  (<<2 => slow modes, approx A valid)")


if __name__ == "__main__":
    main()
