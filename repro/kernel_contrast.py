#!/usr/bin/env python3
"""MPL DOES use the LR decrements -- but only through a SATURATING (rising) kernel.

MPL's annealing term is a convolution of the decrements drop_k with the rising kernel
G(x)=1-(1+Cx)^{-beta} (0 -> 1): the permanent, quasi-static annealing benefit. The
non-adiabatic lag is a convolution of the SAME decrements with a DECAYING kernel
e^{-lambda x} (1 -> 0): a transient that rises then relaxes. A monotone-rising kernel
*cannot* reproduce a residual that rises and then falls.

We fit the post-step wsdcon residual (which spikes then decays) with each kernel family,
each given its own free parameters, and compare R^2. If the rising (MPL-class) kernel
loses decisively, MPL's functional class structurally cannot represent the lag.
"""
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import load_curve, mpl_predict, compute_s1, MPL_PRECOMPUTED_INIT, SCALES, PEAK_LR

WSDCON = ["wsdcon_3.csv", "wsdcon_9.csv", "wsdcon_18.csv"]
STEP = 8000


def drops_and_S(curve):
    eta = curve.lrs.astype(np.float64)
    drop = np.zeros_like(eta); drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0)
    return drop, np.cumsum(eta)


def decaying_feat(curve, lam):
    """sum drop * exp(-lam (S_t - S_t'))  -- our kernel (1 -> 0)."""
    eta = curve.lrs.astype(np.float64)
    drop, _ = drops_and_S(curve)
    a = 1.0 - 1.0 / max(lam, 1e-9)   # placeholder; use exact S-time recurrence below
    s = np.empty_like(eta); acc = 0.0
    for t in range(len(eta)):
        acc = acc * np.exp(-lam * eta[t]) + drop[t]
        s[t] = acc
    return s[curve.step]


def rising_feat(curve, C, beta):
    """sum drop * (1-(1+C (S_t-S_t'))^{-beta})  -- MPL's saturating kernel (0 -> 1)."""
    drop, S = drops_and_S(curve)
    idx = curve.step
    St = S[idx]
    # vectorise over decrement steps where drop>0
    kk = np.where(drop > 0)[0]
    dk = drop[kk]; Sk = S[kk]
    gap = St[:, None] - Sk[None, :]
    G = np.where(gap > 0, 1.0 - np.power(1.0 + C * np.maximum(gap, 0.0), -beta), 0.0)
    return (G * dk[None, :]).sum(axis=1)


def best_r2(resid, feat_fn, p0, bounds, mask):
    """Fit floor + kappa*feat(params); maximise R^2 over params (and floor,kappa linear)."""
    def neg_r2(p):
        f = feat_fn(p)
        # linear LSQ for [floor, kappa]
        A = np.c_[np.ones_like(f), f][mask]
        y = resid[mask]
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = y - A @ coef
        ss = np.sum((y - y.mean()) ** 2)
        return -(1 - np.sum(r ** 2) / ss)
    best = None; bf = 1.0
    for x0 in p0:
        rr = minimize(neg_r2, x0, method="Nelder-Mead",
                      options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-6})
        if rr.fun < bf:
            bf, best = rr.fun, rr.x
    return -bf, best


def main():
    print("=" * 70)
    print("Post-step wsdcon residual: DECAYING kernel (ours) vs RISING kernel (MPL class)")
    print("R^2 on steps > 8050 (the spike-then-relax window)")
    print("=" * 70)
    print(f"  {'scale':>5s} {'curve':14s} {'R2 decaying':>12s} {'R2 rising':>10s}")
    dd, rr = [], []
    for scale in SCALES:
        p = MPL_PRECOMPUTED_INIT[scale]
        for n in WSDCON:
            c = load_curve(scale, n)
            resid = c.loss - mpl_predict(p, c)
            mask = c.step > STEP + 50
            r2d, _ = best_r2(resid, lambda q: decaying_feat(c, q[0]),
                             [[8.0], [15.0], [25.0]], None, mask)
            r2r, _ = best_r2(resid, lambda q: rising_feat(c, q[0], q[1]),
                             [[2.0, 0.6], [0.5, 1.0], [10.0, 0.3]], None, mask)
            dd.append(r2d); rr.append(r2r)
            print(f"  {scale:>4s}M {n:14s} {r2d:12.3f} {r2r:10.3f}")
    print("-" * 70)
    print(f"  mean R^2:  decaying (ours) = {np.mean(dd):.3f}   "
          f"rising (MPL class) = {np.mean(rr):.3f}")
    print("\n  A monotone-rising kernel cannot reproduce the relax-back-down of the lag,")
    print("  so MPL's functional class structurally misses it. gamma is its surrogate.")


if __name__ == "__main__":
    main()
