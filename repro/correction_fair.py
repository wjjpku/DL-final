#!/usr/bin/env python3
"""FAIR test of the 1-parameter drop-relax correction.

Critical fairness fix over residual_fixedtau.py: there the MPL baseline was the
cosine-only precomputed fit while the correction's kappa was fit on cosine + WSD
curves -- the correction had seen more data. Here BOTH arms refit MPL on the SAME
training set; the only difference is the extra kappa*drop_relax term. If the
correction still wins on the held-out curve, it captures structure MPL cannot fit
even when given the same data -> a genuine formula improvement, not an info gap.

Leave-one-curve-out over the 5 non-cosine curves (train = cosine x2 + other 4),
3 scales -> 15 held-out evaluations. tau is FIXED (1 new parameter, kappa).
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
    Curve, load_curve, huber_log_residual, metrics, compute_ld, MPL_PRECOMPUTED_INIT, SCALES, PEAK_LR,
)
from validate_theory import fit_mpl, mpl_pred, F_MPL  # noqa: E402

COS = ["cosine_24000.csv", "cosine_72000.csv"]
NONCOS = ["wsd_20000_24000.csv", "wsdld_20000_24000.csv",
          "wsdcon_3.csv", "wsdcon_9.csv", "wsdcon_18.csv"]


def drop_relax(curve: Curve, tau: float) -> np.ndarray:
    eta = curve.lrs.astype(np.float64)
    drop = np.zeros_like(eta)
    drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0)
    a = 1.0 - 1.0 / tau
    s, _ = lfilter([1.0], [1.0, -a], drop, zi=[0.0])
    return (s / PEAK_LR)[curve.step]


def fit_kappa(p_mpl, train, tau):
    base = [mpl_pred(p_mpl, c, fast=True) for c in train]
    feats = [drop_relax(c, tau) for c in train]
    la = np.concatenate([c.loss for c in train])

    def obj(kappa):
        if kappa < 0:
            return 1e18
        return huber_log_residual(la, np.concatenate([b + kappa * f for b, f in zip(base, feats)]))

    return float(minimize_scalar(obj, bounds=(0.0, 1e4), method="bounded",
                                 options={"xatol": 1e-4}).x)


def main():
    taus = [600, 1200]
    print("=" * 80)
    print("FAIR leave-one-curve-out: BOTH arms refit MPL on the same train (cosine x2 + 4)")
    print("Arm1 = MPL(refit).  Arm2 = MPL(refit) + kappa*drop_relax.  Eval on held-out curve.")
    print("=" * 80)
    for tau in taus:
        rows = []
        kappas = []
        mpl_mae, corr_mae = [], []
        per_curve = {n: [] for n in NONCOS}
        for hold in NONCOS:
            trn = COS + [x for x in NONCOS if x != hold]
            for scale in SCALES:
                train = [load_curve(scale, x) for x in trn]
                hc = load_curve(scale, hold)
                # Arm1: refit MPL (all 7) on this train
                p = fit_mpl(train, MPL_PRECOMPUTED_INIT[scale], F_MPL)
                m_mpl = metrics(hc.loss, mpl_pred(p, hc, fast=False))["mae"]
                # Arm2: add 1-param correction fit on the same train
                kappa = fit_kappa(p, train, tau)
                kappas.append(kappa)
                pr = mpl_pred(p, hc, fast=False) + kappa * drop_relax(hc, tau)
                m_corr = metrics(hc.loss, pr)["mae"]
                mpl_mae.append(m_mpl); corr_mae.append(m_corr)
                per_curve[hold].append((m_mpl, m_corr))
        mpl_mae, corr_mae = np.array(mpl_mae), np.array(corr_mae)
        d = (corr_mae.mean() / mpl_mae.mean() - 1) * 100
        kcv = np.std(kappas) / abs(np.mean(kappas)) * 100
        print(f"\n[tau={tau}]  MPL(refit)={mpl_mae.mean():.5f}  +corr={corr_mae.mean():.5f}  "
              f"delta={d:+.1f}%  wins={int((corr_mae<mpl_mae).sum())}/{len(corr_mae)}  "
              f"kappa CV={kcv:.0f}%")
        print(f"  {'held-out curve':22s} {'MPL':>9s} {'+corr':>9s} {'delta%':>8s}")
        for n in NONCOS:
            mm = np.mean([a for a, _ in per_curve[n]])
            cc = np.mean([b for _, b in per_curve[n]])
            print(f"  {n:22s} {mm:9.5f} {cc:9.5f} {(cc/mm-1)*100:+8.1f}")


def fewshot():
    """FAIR few-shot: train = cosine x2 + ONE WSD curve; both arms refit MPL on
    those 3; test on the OTHER 4 WSD curves. Tests whether the correction's
    inductive bias helps MPL extrapolate from a single decay curve."""
    print("\n" + "=" * 80)
    print("FAIR few-shot: train = cosine x2 + 1 WSD ; test = other 4 WSD (both refit MPL)")
    print("=" * 80)
    for tau in (600, 1200):
        mpl_mae, corr_mae, kappas = [], [], []
        for seed in NONCOS:
            trn = COS + [seed]
            tst = [x for x in NONCOS if x != seed]
            for scale in SCALES:
                train = [load_curve(scale, x) for x in trn]
                test = [load_curve(scale, x) for x in tst]
                p = fit_mpl(train, MPL_PRECOMPUTED_INIT[scale], F_MPL)
                kappa = fit_kappa(p, train, tau)
                kappas.append(kappa)
                for hc in test:
                    mpl_mae.append(metrics(hc.loss, mpl_pred(p, hc, fast=False))["mae"])
                    pr = mpl_pred(p, hc, fast=False) + kappa * drop_relax(hc, tau)
                    corr_mae.append(metrics(hc.loss, pr)["mae"])
        mpl_mae, corr_mae = np.array(mpl_mae), np.array(corr_mae)
        d = (corr_mae.mean() / mpl_mae.mean() - 1) * 100
        kcv = np.std(kappas) / abs(np.mean(kappas)) * 100
        tag = "  <<WIN" if corr_mae.mean() < mpl_mae.mean() else ""
        print(f"[tau={tau}]  MPL(refit)={mpl_mae.mean():.5f}  +corr={corr_mae.mean():.5f}  "
              f"delta={d:+.1f}%  wins={int((corr_mae<mpl_mae).sum())}/{len(corr_mae)}  "
              f"kappa CV={kcv:.0f}%{tag}")


if __name__ == "__main__":
    main()
    fewshot()
