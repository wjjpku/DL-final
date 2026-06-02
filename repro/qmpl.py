#!/usr/bin/env python3
"""Q-MPL: a discrete-step (S, Q) loss law from a new asymptotic.

Tissue and MPL both use the continuum step approximation (1-eta*lambda)~e^{-eta*lambda},
leaving a single time variable S=sum(eta). Keeping the next discrete-step order
introduces a SECOND time variable Q=sum(eta^2) (from ln(1-x)^2=-2x-x^2-...).
Expanding the bias spectral integral int g(lambda) e^{-2 lambda S - lambda^2 Q} dlambda
gives a derived correction term ~ Q * S^{-(alpha+2)}. We test:

   Q-MPL:  L = L0 + A S^{-alpha} + E * Q * S^{-(alpha+2)} + B * LD(C,beta,gamma)

vs plain MPL (E=0), on the canonical fit-cosine / predict-WSD task.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import (  # noqa: E402
    load_curve, compute_s1, huber_log_residual, metrics, MPL_PRECOMPUTED_INIT,
)
from validate_theory import ld_matrix, _coarse  # noqa: E402

SCALES = ["25", "100", "400"]
TRAIN = ["cosine_24000.csv", "cosine_72000.csv"]
TEST = ["wsd_20000_24000.csv", "wsdld_20000_24000.csv", "wsdcon_3.csv",
        "wsdcon_9.csv", "wsdcon_18.csv"]


def Qvar(curve):
    return np.cumsum(curve.lrs ** 2)[curve.step]


def pred(p, curve, qmpl):
    L0, A, alpha, B, C, beta, gamma, E = p
    s1 = compute_s1(curve)
    ld = ld_matrix(curve, C, beta, gamma, 1500)
    out = L0 + A * np.power(s1, -alpha) + B * ld
    if qmpl:
        out = out + E * Qvar(curve) * np.power(s1, -(alpha + 2.0))
    return out


BOUNDS = [(0, 10), (1e-8, 100), (0.05, 1.5), (-1e6, 1e6),
          (0.3, 6), (0.1, 1.5), (0.1, 1.5), (-1e8, 1e8)]


def fit(curves, qmpl, init, nrestart=6):
    fc = [_coarse(c, 110) for c in curves]
    free = [0, 1, 2, 3, 4, 5, 6] + ([7] if qmpl else [])
    init = np.array(init, float)
    bnd = [BOUNDS[i] for i in free]

    def asm(pf):
        f = init.copy(); f[free] = pf; return f

    def obj(pf):
        prs, ys = [], []
        for c in fc:
            v = pred(asm(pf), c, qmpl)
            if not np.all(np.isfinite(v)) or np.any(v <= 0):
                return 1e18
            prs.append(v); ys.append(c.loss)
        return huber_log_residual(np.concatenate(ys), np.concatenate(prs))

    b = init[free]; best, bf = None, np.inf
    seeds = [b] + [b * f for f in (0.7, 1.3, 0.85, 1.15, 1.5)][:nrestart - 1]
    for x0 in seeds:
        r = minimize(obj, x0, method="L-BFGS-B", bounds=bnd, options={"maxiter": 400})
        if r.fun < bf:
            bf, best = r.fun, r.x
    return asm(best)


def avg_mae(p, scale, names, qmpl):
    return float(np.mean([metrics(load_curve(scale, n).loss, pred(p, load_curve(scale, n), qmpl))["mae"]
                          for n in names]))


def main():
    print(f"{'scale':>6} | {'MPL(7p) test':>13} | {'Q-MPL(8p) test':>15} | {'E*':>10} | winner")
    rows = []
    for s in SCALES:
        cur = [load_curve(s, n) for n in TRAIN]
        init = list(MPL_PRECOMPUTED_INIT[s]) + [0.0]
        mpl = fit(cur, False, init)
        qm = fit(cur, True, list(mpl) if mpl[7] == 0 else mpl)
        m_te = avg_mae(mpl, s, TEST, False)
        q_te = avg_mae(qm, s, TEST, True)
        win = "Q-MPL" if q_te < m_te else "MPL"
        print(f"{s+'M':>6} | {m_te:>13.5f} | {q_te:>15.5f} | {qm[7]:>10.2e} | {win}")
        rows.append((s, m_te, q_te))
    tm = np.mean([r[1] for r in rows]); tq = np.mean([r[2] for r in rows])
    print(f"\n  avg test MAE:  MPL={tm:.5f}   Q-MPL={tq:.5f}   "
          f"{'Q-MPL 更优' if tq < tm else 'MPL 更优'} ({100*(tm-tq)/tm:+.1f}%)")


if __name__ == "__main__":
    main()
