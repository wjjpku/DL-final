#!/usr/bin/env python3
"""Does the eta-weight unify the fitted relaxation rate across families?

Fit lam (origin-LS kappa inside) separately on (a) probes pooled, (b) sharp
pooled, per scale, for delta in {0, 0.25, 0.5, 0.75}.  If the weighted law is
the right structure, lam_probe and lam_sharp should converge.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import (  # noqa: E402
    load_curve, mpl_predict, MPL_PRECOMPUTED_INIT, SCALES,
)
from formula_lab.lab import feature, fit_origin, DECAY, PROBES  # noqa: E402


def fit_lam(scale: str, curves: list[str], delta: float) -> tuple[float, float]:
    p = MPL_PRECOMPUTED_INIT[scale]
    data = []
    for n in curves:
        c = load_curve(scale, n)
        data.append((c, c.loss - mpl_predict(p, c)))

    def sse(loglam):
        lam = float(np.exp(loglam))
        spec = {"form": "pow", "delta": delta, "lam": lam} if delta > 0 else \
               {"form": "lr", "lam": lam}
        xs = [feature(c, spec) for c, _ in data]
        x = np.concatenate(xs)
        y = np.concatenate([r for _, r in data])
        k = max(0.0, fit_origin(x, y)[0])
        return float(np.sum((y - k * x) ** 2))

    res = minimize_scalar(sse, bounds=(np.log(0.5), np.log(80.0)),
                          method="bounded", options={"xatol": 1e-3})
    lam = float(np.exp(res.x))
    return lam, res.fun


def main():
    print(f"{'delta':>6s} " + " ".join(f"{s+'M':>16s}" for s in SCALES)
          + "   (lam_probe / lam_sharp)")
    for delta in [0.0, 0.25, 0.5, 0.75, 1.0]:
        cells = []
        for scale in SCALES:
            lp, _ = fit_lam(scale, PROBES, delta)
            ls, _ = fit_lam(scale, DECAY, delta)
            cells.append(f"{lp:6.1f} /{ls:6.1f}")
        print(f"{delta:6.2f} " + " ".join(f"{c:>16s}" for c in cells))


if __name__ == "__main__":
    main()
