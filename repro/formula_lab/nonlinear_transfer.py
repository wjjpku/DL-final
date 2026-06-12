#!/usr/bin/env python3
"""Direction A, last question: can the nonlinear shape (lam0, rstar) be
transferred leave-one-scale-out (like the c constant), with kappa then
identified from the TARGET's probes alone?

Per target scale:
  (lam0, rstar) = geometric mean of the OTHER scales' joint-family fits
  kappa         = NLS on the target's 3 wsdcon probes with (lam0, rstar) fixed
Evaluate on target wsd+wsdld.  Baselines: linear probes-only -17.1%,
pow d=0.5 -28.6%.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import (  # noqa: E402
    load_curve, mpl_predict, metrics, MPL_PRECOMPUTED_INIT, SCALES,
)
from formula_lab.lab import DECAY, PROBES  # noqa: E402
from formula_lab.nonlinear_ode import ode_response, fit_ode  # noqa: E402


def main():
    # joint-family fits per scale (as in T-A)
    fam = DECAY + PROBES
    shape = {}
    for scale in SCALES:
        th, _ = fit_ode(scale, fam, linear=False)
        shape[scale] = th  # (lam0, rstar, kappa)
        print(f"{scale:>4}M joint fit: lam0={th[0]:.2f} rstar={th[1]:.4f} "
              f"kappa={th[2]:.4f}")

    m0s, m1s, wins = [], [], 0
    for tgt in SCALES:
        others = [s for s in SCALES if s != tgt]
        lam0 = float(np.exp(np.mean([np.log(shape[s][0]) for s in others])))
        rstar = float(np.exp(np.mean([np.log(shape[s][1]) for s in others])))
        p = MPL_PRECOMPUTED_INIT[tgt]
        probes = [(load_curve(tgt, n),) for n in PROBES]
        resids = [(c[0], c[0].loss - mpl_predict(p, c[0])) for c in probes]

        def sse(logk):
            k = float(np.exp(logk))
            tot = 0.0
            for c, r in resids:
                tot += float(np.sum((r - ode_response(c, lam0, rstar, k)) ** 2))
            return tot

        res = minimize_scalar(sse, bounds=(np.log(1e-4), np.log(1.0)),
                              method="bounded", options={"xatol": 1e-4})
        kappa = float(np.exp(res.x))
        row = []
        for n in DECAY:
            c = load_curve(tgt, n)
            base = mpl_predict(p, c)
            pred = base + ode_response(c, lam0, rstar, kappa)
            m0 = metrics(c.loss, base)["mae"]
            m1 = metrics(c.loss, pred)["mae"]
            m0s.append(m0); m1s.append(m1); wins += int(m1 < m0)
            row.append(f"{n.split('_')[0]} {100*(m1/m0-1):+.1f}%")
        print(f"{tgt:>4}M LOO shape (lam0={lam0:.2f}, r*={rstar:.4f}) "
              f"kappa={kappa:.4f}: " + "  ".join(row))
    d = 100.0 * (np.mean(m1s) / np.mean(m0s) - 1.0)
    print(f"\nOVERALL LOO nonlinear transfer: {d:+.1f}% {wins}/6 "
          f"(baselines: linear -17.1%, pow d=0.5 -28.6%)")


if __name__ == "__main__":
    main()
