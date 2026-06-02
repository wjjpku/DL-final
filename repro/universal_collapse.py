#!/usr/bin/env python3
"""Universality: do all 27 loss curves (3 scales x 9 schedules) collapse onto a
single master power law under one scale-invariant effective-progress map?

MPL is algebraically a single power law L = L0(N) + A(N) tau^{-alpha} in the
effective progress tau = [S^{-alpha} + (B/A) LD]^{-1/alpha}, which absorbs the
schedule's annealing. Using ONE shared exponent set {alpha*,C*,beta*,gamma*}
(the scale-invariant ones) and only per-scale amplitudes {L0,A,B}, we test
whether (L - L0)/A vs tau collapses to the universal curve y = tau^{-alpha} for
every scale and schedule.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import (  # noqa: E402
    load_curve, compute_s1, huber_log_residual, MPL_PRECOMPUTED_INIT,
)
from validate_theory import ld_matrix, _coarse  # noqa: E402  (vectorised LD)

ROOT = REPO.parent
SCALES = ["25", "100", "400"]
ALL_SCHED = ["cosine_24000.csv", "cosine_72000.csv", "constant_24000.csv",
             "constant_72000.csv", "wsd_20000_24000.csv", "wsdld_20000_24000.csv",
             "wsdcon_3.csv", "wsdcon_9.csv", "wsdcon_18.csv"]


def shared_exponents():
    arr = np.array([MPL_PRECOMPUTED_INIT[s] for s in SCALES])
    m = arr.mean(0)
    return m[2], m[4], m[5], m[6]          # alpha*, C*, beta*, gamma*


def fit_amplitudes(scale, curves, alpha, C, beta, gamma):
    """Fit {L0,A,B} on all curves of a scale, exponents fixed at shared values."""
    fc = [_coarse(c, 100) for c in curves]
    pre = [(compute_s1(c), ld_matrix(c, C, beta, gamma, 1500), c.loss) for c in fc]

    def obj(p):
        L0, A, B = p
        pr, ys = [], []
        for s1, ld, loss in pre:
            v = L0 + A * np.power(s1, -alpha) + B * ld
            if not np.all(np.isfinite(v)) or np.any(v <= 0):
                return 1e18
            pr.append(v); ys.append(loss)
        return huber_log_residual(np.concatenate(ys), np.concatenate(pr))

    init = [MPL_PRECOMPUTED_INIT[scale][i] for i in (0, 1, 3)]
    best, bf = None, np.inf
    for f in (1.0, 0.8, 1.2):
        r = minimize(obj, np.array(init) * f, method="Nelder-Mead",
                     options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-12})
        if r.fun < bf:
            bf, best = r.fun, r.x
    return best


def main():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    alpha, C, beta, gamma = shared_exponents()
    print(f"shared exponents: alpha={alpha:.3f} C={C:.3f} beta={beta:.3f} gamma={gamma:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    colors = {"25": "#4C72B0", "100": "#DD8452", "400": "#C44E52"}
    all_tau, all_y = [], []
    for scale in SCALES:
        curves = [load_curve(scale, n) for n in ALL_SCHED]
        L0, A, B = fit_amplitudes(scale, curves, alpha, C, beta, gamma)
        for c in curves:
            s1 = compute_s1(c)
            ld = ld_matrix(c, C, beta, gamma, 1500)
            y = (c.loss - L0) / A                              # data, normalised by scale amplitude
            tau_arg = np.power(s1, -alpha) + (B / A) * ld      # schedule-predicted effective progress
            m = (y > 0) & (tau_arg > 0)
            tau = np.power(tau_arg[m], -1.0 / alpha)
            ax[0].scatter(c.step[m], c.loss[m], s=3, color=colors[scale], alpha=.25)
            ax[1].scatter(tau, y[m], s=3, color=colors[scale], alpha=.25)
            all_tau.append(tau); all_y.append(y[m])
    # master curve y = tau^{-alpha}
    tt = np.geomspace(min(t.min() for t in all_tau), max(t.max() for t in all_tau), 200)
    ax[1].plot(tt, np.power(tt, -alpha), "k-", lw=1.5, label=r"master: $y=\tau^{-\alpha}$")
    # collapse quality: R^2 of log y vs -alpha log tau across ALL 27 curves' points
    T = np.concatenate(all_tau); Y = np.concatenate(all_y)
    logY, pred = np.log(Y), -alpha * np.log(T)
    r2 = 1 - np.sum((logY - pred) ** 2) / np.sum((logY - logY.mean()) ** 2)

    ax[0].set_xlabel("step"); ax[0].set_ylabel("loss")
    ax[0].set_title("(a) raw: 27 curves (3 scales x 9 schedules)")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel(r"effective progress $\tau$")
    ax[1].set_ylabel(r"$(L-L_0(N))/A(N)$")
    ax[1].set_title(fr"(b) collapse onto one master law ($R^2={r2:.4f}$)")
    ax[1].legend()
    fig.tight_layout()
    out = ROOT / "results" / "universal_collapse.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"collapse R^2 (all 27 curves, shared exponents) = {r2:.5f}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
