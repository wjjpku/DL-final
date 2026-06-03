#!/usr/bin/env python3
"""Empirical search for a 2-parameter correction on FROZEN MPL that beats MPL.

Data-driven (no derivation required), but parsimonious: MPL is frozen at its
published per-scale fit and we add ONE feature with at most (kappa, tau) = 2 new
parameters. We compare candidate features by held-out test MAE AND by whether
(kappa, tau) come out scale-invariant (the anti-overfitting check).

Why the previous attempt (level-lag) failed: it used (EMA(eta) - eta)_+, which
stays large whenever eta stays low -> it keeps "correcting" through wsdcon's long
constant phase and over-shoots. The fix is to key the correction on the *recent
LR decrease*, which returns to ~0 in any constant phase (cosine gradual / wsdcon
plateau) and is large only during a concentrated sharp decay (wsd / wsdld).

Candidate features f_t (then correction = kappa * f[curve.step]):
  A  level-lag        relu( EMA_tau(eta) - eta )           [previous attempt]
  B  drop-relax       s_t = a s_{t-1} + drop_t ,  drop=relu(-d eta)   [a=1-1/tau]
  Bn drop-relax/eta0  B normalised by peak LR (dimensionless)
  C  drop-relax x eta B modulated by current eta (penalty ~ eta * state)

Protocol: frozen MPL = MPL_PRECOMPUTED_INIT (well-fit on cosine). Fit (kappa,tau)
of each candidate on a train split that contains a decay curve, evaluate on the
held-out curves. Two splits are reported to avoid cherry-picking.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import (  # noqa: E402
    Curve, load_curve, huber_log_residual, metrics,
    mpl_predict, MPL_PRECOMPUTED_INIT, SCALES, WARMUP, PEAK_LR,
)

ALL = ["cosine_24000.csv", "cosine_72000.csv", "wsd_20000_24000.csv",
       "wsdld_20000_24000.csv", "wsdcon_3.csv", "wsdcon_9.csv", "wsdcon_18.csv"]

# two honest splits (train must contain >=1 decay so the correction is identifiable)
SPLITS = {
    "trainCosWsd": (["cosine_24000.csv", "cosine_72000.csv", "wsd_20000_24000.csv"],
                    ["wsdld_20000_24000.csv", "wsdcon_3.csv", "wsdcon_9.csv", "wsdcon_18.csv"]),
    "trainCosWsdcon": (["cosine_24000.csv", "cosine_72000.csv", "wsdcon_9.csv"],
                       ["wsd_20000_24000.csv", "wsdld_20000_24000.csv", "wsdcon_3.csv", "wsdcon_18.csv"]),
}


def _ema(x, tau, zi_val):
    a = 1.0 - 1.0 / tau
    y, _ = lfilter([1.0 - a], [1.0, -a], x, zi=[a * zi_val])
    return y


def _drop_relax(drop, tau):
    a = 1.0 - 1.0 / tau
    y, _ = lfilter([1.0], [1.0, -a], drop, zi=[0.0])  # s_t = a s_{t-1} + drop_t
    return y


def feature(curve: Curve, kind: str, tau: float) -> np.ndarray:
    eta = curve.lrs.astype(np.float64)
    drop = np.zeros_like(eta)
    drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0)  # relu(-d eta)
    if kind == "A":
        f = np.clip(_ema(eta, tau, PEAK_LR) - eta, 0.0, None)
    elif kind == "B":
        f = _drop_relax(drop, tau)
    elif kind == "Bn":
        f = _drop_relax(drop, tau) / PEAK_LR
    elif kind == "C":
        f = _drop_relax(drop, tau) * eta / PEAK_LR
    else:
        raise ValueError(kind)
    return f[curve.step]


def fit_corr(p_mpl, train, kind):
    base = [mpl_predict(p_mpl, c) for c in train]
    la = np.concatenate([c.loss for c in train])

    def obj(x):
        kappa, tau = x
        if tau <= 1.0 or kappa < 0:
            return 1e18
        preds = []
        for c, b in zip(train, base):
            pr = b + kappa * feature(c, kind, tau)
            if np.any(~np.isfinite(pr)) or np.any(pr <= 0):
                return 1e18
            preds.append(pr)
        return huber_log_residual(la, np.concatenate(preds))

    best, bf = None, float("inf")
    for k0 in (10.0, 100.0, 1000.0):
        for t0 in (30.0, 150.0, 800.0, 3000.0):
            r = minimize(obj, [k0, t0], method="Nelder-Mead",
                         options={"maxiter": 1500, "xatol": 1e-7, "fatol": 1e-13})
            if r.fun < bf:
                bf, best = float(r.fun), r.x
    return best, bf


def main():
    kinds = ["A", "B", "Bn", "C"]
    for split_name, (tr_names, te_names) in SPLITS.items():
        print("=" * 80)
        print(f"SPLIT {split_name}:  train={tr_names}")
        print(f"{'':18s}  held-out test = {te_names}")
        print("=" * 80)
        # baseline MPL held-out MAE
        mpl_mae, n = [], 0
        per_scale_params = {k: {} for k in kinds}
        corr_mae = {k: [] for k in kinds}
        for scale in SCALES:
            p_mpl = MPL_PRECOMPUTED_INIT[scale]
            train = [load_curve(scale, x) for x in tr_names]
            test = [load_curve(scale, x) for x in te_names]
            for c in test:
                mpl_mae.append(metrics(c.loss, mpl_predict(p_mpl, c))["mae"])
            for kind in kinds:
                (kappa, tau), _ = fit_corr(p_mpl, train, kind)
                per_scale_params[kind][scale] = (kappa, tau)
                for c in test:
                    pr = mpl_predict(p_mpl, c) + kappa * feature(c, kind, tau)
                    corr_mae[kind].append(metrics(c.loss, pr)["mae"])
        mpl_mae = np.array(mpl_mae)
        print(f"\n  baseline MPL held-out mean MAE = {mpl_mae.mean():.5f}\n")
        print(f"  {'feat':4s} {'MAE':>9s} {'vs MPL':>8s} {'wins':>6s}   "
              f"{'kappa (25/100/400)':>22s} {'tau (25/100/400)':>20s}  invariant?")
        for kind in kinds:
            cm = np.array(corr_mae[kind])
            d = (cm.mean() / mpl_mae.mean() - 1) * 100
            ks = [per_scale_params[kind][s][0] for s in SCALES]
            ts = [per_scale_params[kind][s][1] for s in SCALES]
            kcv = np.std(ks) / abs(np.mean(ks)) * 100
            tcv = np.std(ts) / abs(np.mean(ts)) * 100
            inv = "YES" if (kcv < 30 and tcv < 40) else f"no(k{kcv:.0f}/t{tcv:.0f}%)"
            wins = int((cm < mpl_mae).sum())
            tag = "<<" if cm.mean() < mpl_mae.mean() else "  "
            print(f"  {kind:4s} {cm.mean():9.5f} {d:+7.1f}% {wins:3d}/{len(cm):<2d}  "
                  f"[{ks[0]:6.1f} {ks[1]:6.1f} {ks[2]:6.1f}] "
                  f"[{ts[0]:5.0f} {ts[1]:5.0f} {ts[2]:5.0f}]  {inv} {tag}")


if __name__ == "__main__":
    main()
