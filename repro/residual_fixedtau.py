#!/usr/bin/env python3
"""1-parameter correction on frozen MPL: drop-relax feature with FIXED tau.

Lesson from residual_features.py: letting tau float collapses the feature to the
static floor kappa*(peak-eta)_+ (tau->inf), which helps low-eta decays but hurts
wsdcon plateaus and is not scale-invariant. Here we FIX tau to a small physical
sharpening timescale so the feature fires only during a *concentrated* decay and
returns to ~0 in any constant phase, and fit only kappa (1 new parameter).

We scan tau and report, for BOTH splits, the held-out MAE vs MPL and the
scale-invariance of kappa. We also run leave-one-curve-out (LOCO) over the five
non-cosine curves as the most honest generalisation test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import lfilter

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import (  # noqa: E402
    Curve, load_curve, huber_log_residual, metrics,
    mpl_predict, MPL_PRECOMPUTED_INIT, SCALES, PEAK_LR,
)

COS = ["cosine_24000.csv", "cosine_72000.csv"]
NONCOS = ["wsd_20000_24000.csv", "wsdld_20000_24000.csv",
          "wsdcon_3.csv", "wsdcon_9.csv", "wsdcon_18.csv"]


def drop_relax(curve: Curve, tau: float, normalize: bool) -> np.ndarray:
    eta = curve.lrs.astype(np.float64)
    drop = np.zeros_like(eta)
    drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0)
    a = 1.0 - 1.0 / tau
    s, _ = lfilter([1.0], [1.0, -a], drop, zi=[0.0])   # s_t = a s_{t-1} + drop_t
    if normalize:
        s = s / PEAK_LR
    return s[curve.step]


def fit_kappa(p_mpl, train, tau, normalize):
    base = [mpl_predict(p_mpl, c) for c in train]
    la = np.concatenate([c.loss for c in train])
    feats = [drop_relax(c, tau, normalize) for c in train]

    def obj(kappa):
        if kappa < 0:
            return 1e18
        preds = [b + kappa * f for b, f in zip(base, feats)]
        return huber_log_residual(la, np.concatenate(preds))

    r = minimize_scalar(obj, bounds=(0.0, 1e4), method="bounded",
                        options={"xatol": 1e-4})
    return float(r.x)


def held_out(tau, normalize, train_names, test_names):
    """Return (mpl_mae, corr_mae arrays over all scales*test, kappas per scale)."""
    mpl_mae, corr_mae, kappas = [], [], []
    for scale in SCALES:
        p_mpl = MPL_PRECOMPUTED_INIT[scale]
        train = [load_curve(scale, x) for x in train_names]
        test = [load_curve(scale, x) for x in test_names]
        kappa = fit_kappa(p_mpl, train, tau, normalize)
        kappas.append(kappa)
        for c in test:
            mpl_mae.append(metrics(c.loss, mpl_predict(p_mpl, c))["mae"])
            pr = mpl_predict(p_mpl, c) + kappa * drop_relax(c, tau, normalize)
            corr_mae.append(metrics(c.loss, pr)["mae"])
    return np.array(mpl_mae), np.array(corr_mae), kappas


def main():
    normalize = True   # Bn (dimensionless feature); kappa ~ loss units
    taus = [80, 150, 300, 600, 1200, 2400]
    splits = {
        "S1 (test=wsdld+3wsdcon)": (COS + ["wsd_20000_24000.csv"],
                                    ["wsdld_20000_24000.csv", "wsdcon_3.csv", "wsdcon_9.csv", "wsdcon_18.csv"]),
        "S2 (test=wsd+wsdld+2wsdcon)": (COS + ["wsdcon_9.csv"],
                                        ["wsd_20000_24000.csv", "wsdld_20000_24000.csv", "wsdcon_3.csv", "wsdcon_18.csv"]),
    }

    print("=" * 78)
    print("Fixed-tau, 1-parameter (kappa) drop-relax correction on frozen MPL")
    print("=" * 78)
    for sname, (trn, tst) in splits.items():
        print(f"\n[{sname}]  train={trn}")
        print(f"  {'tau':>5s} {'MPL':>8s} {'MPL+corr':>9s} {'delta%':>7s} {'wins':>6s}  "
              f"{'kappa 25/100/400':>20s}  kCV")
        for tau in taus:
            m, c, ks = held_out(tau, normalize, trn, tst)
            d = (c.mean() / m.mean() - 1) * 100
            kcv = np.std(ks) / abs(np.mean(ks)) * 100
            tag = " <<" if c.mean() < m.mean() else ""
            print(f"  {tau:5d} {m.mean():8.5f} {c.mean():9.5f} {d:+7.1f} "
                  f"{int((c<m).sum()):3d}/{len(c):<2d}  "
                  f"[{ks[0]:5.3f} {ks[1]:5.3f} {ks[2]:5.3f}]  {kcv:4.0f}%{tag}")

    # ---- leave-one-curve-out (LOCO): the honest generalisation test ----
    print("\n" + "=" * 78)
    print("Leave-one-curve-out over the 5 non-cosine curves (train=cosine x2 + other 4)")
    print("kappa fit on train, evaluated ONLY on the held-out curve. Avg over 5x3.")
    print("=" * 78)
    print(f"  {'tau':>5s} {'MPL':>8s} {'MPL+corr':>9s} {'delta%':>7s} {'wins':>7s}  kCV(mean)")
    for tau in taus:
        mpl_all, corr_all, kcvs = [], [], []
        for hold in NONCOS:
            trn = COS + [x for x in NONCOS if x != hold]
            ks = []
            for scale in SCALES:
                p_mpl = MPL_PRECOMPUTED_INIT[scale]
                train = [load_curve(scale, x) for x in trn]
                hc = load_curve(scale, hold)
                kappa = fit_kappa(p_mpl, train, tau, normalize)
                ks.append(kappa)
                mpl_all.append(metrics(hc.loss, mpl_predict(p_mpl, hc))["mae"])
                pr = mpl_predict(p_mpl, hc) + kappa * drop_relax(hc, tau, normalize)
                corr_all.append(metrics(hc.loss, pr)["mae"])
            kcvs.append(np.std(ks) / abs(np.mean(ks)) * 100)
        mpl_all, corr_all = np.array(mpl_all), np.array(corr_all)
        d = (corr_all.mean() / mpl_all.mean() - 1) * 100
        tag = " <<" if corr_all.mean() < mpl_all.mean() else ""
        print(f"  {tau:5d} {mpl_all.mean():8.5f} {corr_all.mean():9.5f} {d:+7.1f} "
              f"{int((corr_all<mpl_all).sum()):3d}/{len(corr_all):<2d}  {np.mean(kcvs):4.0f}%{tag}")


if __name__ == "__main__":
    main()
