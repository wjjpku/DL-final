#!/usr/bin/env python3
"""Attempt 2 pre-launch: bootstrap rehearsal (zero GPU).

Question: with the equal-S ladder protocol (8 rungs, tail-mean floors, the
measured ~1e-2 raw eval jitter averaged over ~150-tail points -> per-floor
SE ~ 1e-3), what CI half-width on p_hat is achievable, and can adjacent
scales' p be separated?

Method: generate synthetic floors from the measured 10.7M fit
(L=1.0385 + 0.1178 x^p, x=eta2/peak) for p in {0.73, 1.0, 1.25, 1.5};
add per-floor noise sigma_floor in {0.5e-3, 1e-3, 2e-3}; run the exact
3-param NLS + residual bootstrap from analyze_floor.py; report CI half-width
and the implied minimal detectable delta-p between two scales (sqrt2 x
half-width, both-CI criterion).
"""
import sys

import numpy as np
from scipy.optimize import least_squares

PEAK = 1.5e-3
ETAS = np.array([0.5e-4, 1e-4, 2e-4, 3e-4, 4e-4, 6e-4, 8e-4, 1.5e-3])
L0, A0 = 1.0385, 0.1178


def fit_p(etas, fls):
    x = etas / PEAK

    def resid(th):
        L, loga, p = th
        return fls - (L + np.exp(loga) * x ** p)
    best = None
    for p0 in (0.7, 1.0, 1.4, 2.0):
        r = least_squares(resid, x0=[fls.min() - 0.02, np.log(0.05), p0],
                          bounds=([0.5, -10, 0.2], [fls.min(), 3, 3.5]))
        if best is None or r.cost < best.cost:
            best = r
    return best


def rehearse(p_true, sig, n_boot=400, n_rep=30, rng=None):
    rng = rng or np.random.default_rng(0)
    x = ETAS / PEAK
    widths = []
    for _ in range(n_rep):
        fls = L0 + A0 * x ** p_true + rng.normal(0, sig, len(x))
        best = fit_p(ETAS, fls)
        L, loga, p = best.x
        res = fls - (L + np.exp(loga) * x ** p)
        ps = []
        for _ in range(n_boot):
            fb = (L + np.exp(loga) * x ** p) + rng.choice(res, len(res),
                                                          replace=True)
            try:
                r = fit_p(ETAS, fb)
                ps.append(r.x[2])
            except Exception:
                pass
        lo, hi = np.percentile(ps, [5, 95])
        widths.append((hi - lo) / 2)
    return float(np.median(widths))


def main():
    print(f"{'p_true':>7} {'sigma_floor':>12} {'CI half-width':>14} "
          f"{'min detectable dp (2-scale)':>28}")
    for p_true in (0.73, 1.0, 1.25, 1.5):
        for sig in (0.5e-3, 1e-3, 2e-3):
            w = rehearse(p_true, sig)
            print(f"{p_true:7.2f} {sig:12.1e} {w:14.3f} {np.sqrt(2)*w*2:28.3f}")
    print("\nReading: emergence tiers per prereg --")
    print("  middle tier (PRIMARY): successive delta-p > 0 AND "
          "p(top)-p(10.7M) > 2x pooled half-width")
    print("  bonus tier: adjacent pair with disjoint CIs straddling 1")


if __name__ == "__main__":
    main()
