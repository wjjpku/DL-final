#!/usr/bin/env python3
"""Stress-test the few-shot drop-relax win: per-seed breakdown + fixed global kappa.

correction_fair.py showed: in the fair few-shot regime (train = cosine x2 + 1 WSD,
both arms refit MPL, test on the other 4 WSD), MPL + kappa*drop_relax(tau=1200)
beats MPL by ~10% (wins ~42/60) -- but kappa CV ~100%. Two checks here:

  (1) per-seed: which single training curve drives the gain? (Expect: a sharp-decay
      seed calibrates kappa well; a wsdcon seed has little residual signal.)
  (2) FIXED global kappa: replace per-fold fitting by ONE constant kappa, chosen by
      leave-one-SCALE-out transfer (fit kappa on the other two scales' WSD curves,
      apply to the held-out scale). If a transferred constant still wins, the term
      is a usable scale-invariant prior, not fold-specific overfitting.
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
    Curve, load_curve, huber_log_residual, metrics, MPL_PRECOMPUTED_INIT, SCALES, PEAK_LR,
)
from validate_theory import fit_mpl, mpl_pred, F_MPL  # noqa: E402

COS = ["cosine_24000.csv", "cosine_72000.csv"]
NONCOS = ["wsd_20000_24000.csv", "wsdld_20000_24000.csv",
          "wsdcon_3.csv", "wsdcon_9.csv", "wsdcon_18.csv"]
TAU = 1200.0


def drop_relax(curve: Curve, tau: float = TAU) -> np.ndarray:
    eta = curve.lrs.astype(np.float64)
    drop = np.zeros_like(eta); drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0)
    a = 1.0 - 1.0 / tau
    s, _ = lfilter([1.0], [1.0, -a], drop, zi=[0.0])
    return (s / PEAK_LR)[curve.step]


def fit_kappa(p_mpl, train):
    base = [mpl_pred(p_mpl, c, fast=True) for c in train]
    feats = [drop_relax(c) for c in train]
    la = np.concatenate([c.loss for c in train])
    obj = lambda k: 1e18 if k < 0 else huber_log_residual(
        la, np.concatenate([b + k * f for b, f in zip(base, feats)]))
    return float(minimize_scalar(obj, bounds=(0.0, 1e4), method="bounded",
                                 options={"xatol": 1e-4}).x)


def refit(scale, names):
    train = [load_curve(scale, x) for x in names]
    return fit_mpl(train, MPL_PRECOMPUTED_INIT[scale], F_MPL), train


def main():
    # ---------- (1) per-seed breakdown (per-fold kappa) ----------
    print("=" * 78)
    print(f"(1) Few-shot per training-seed breakdown (tau={TAU:.0f}, per-fold kappa)")
    print("    train = cosine x2 + <seed>;  test = other 4 WSD;  both refit MPL")
    print("=" * 78)
    print(f"  {'training seed':22s} {'MPL':>9s} {'+corr':>9s} {'delta%':>8s} {'kappa(25/100/400)':>20s}")
    seed_kappa = {}
    for seed in NONCOS:
        trn = COS + [seed]; tst = [x for x in NONCOS if x != seed]
        mm, cc, ks = [], [], []
        for scale in SCALES:
            p, train = refit(scale, trn)
            k = fit_kappa(p, train); ks.append(k)
            for hc in (load_curve(scale, x) for x in tst):
                mm.append(metrics(hc.loss, mpl_pred(p, hc, fast=False))["mae"])
                cc.append(metrics(hc.loss, mpl_pred(p, hc, fast=False) + k * drop_relax(hc))["mae"])
        seed_kappa[seed] = ks
        mm, cc = np.mean(mm), np.mean(cc)
        print(f"  {seed:22s} {mm:9.5f} {cc:9.5f} {(cc/mm-1)*100:+8.1f} "
              f"[{ks[0]:5.2f} {ks[1]:5.2f} {ks[2]:5.2f}]")

    # ---------- (2) fixed global kappa via leave-one-scale-out ----------
    # Calibrate kappa on a scale using cosine + ALL its WSD curves (well-determined),
    # then TRANSFER that kappa to the OTHER scales' few-shot folds.
    print("\n" + "=" * 78)
    print("(2) Fixed kappa transferred across scales (leave-one-scale-out calibration)")
    print("=" * 78)
    kappa_cal = {}
    for scale in SCALES:
        p_full, train_full = refit(scale, COS + NONCOS)   # MPL on everything (calibration only)
        kappa_cal[scale] = fit_kappa(p_full, train_full)
    print(f"  per-scale calibrated kappa: "
          + " ".join(f"{s}={kappa_cal[s]:.3f}" for s in SCALES)
          + f"  (CV={np.std(list(kappa_cal.values()))/np.mean(list(kappa_cal.values()))*100:.0f}%)")

    mpl_all, corr_all = [], []
    for tgt in SCALES:
        k_use = np.mean([kappa_cal[s] for s in SCALES if s != tgt])  # transferred, excludes target
        for seed in NONCOS:
            trn = COS + [seed]; tst = [x for x in NONCOS if x != seed]
            p, _ = refit(tgt, trn)
            for hc in (load_curve(tgt, x) for x in tst):
                mpl_all.append(metrics(hc.loss, mpl_pred(p, hc, fast=False))["mae"])
                corr_all.append(metrics(hc.loss, mpl_pred(p, hc, fast=False) + k_use * drop_relax(hc))["mae"])
    mpl_all, corr_all = np.array(mpl_all), np.array(corr_all)
    d = (corr_all.mean() / mpl_all.mean() - 1) * 100
    tag = "  <<WIN (fair, transferred constant)" if corr_all.mean() < mpl_all.mean() else "  (no win)"
    print(f"\n  Few-shot with TRANSFERRED constant kappa (no target-scale fitting):")
    print(f"    MPL(refit)={mpl_all.mean():.5f}  +corr={corr_all.mean():.5f}  "
          f"delta={d:+.1f}%  wins={int((corr_all<mpl_all).sum())}/{len(corr_all)}{tag}")


if __name__ == "__main__":
    main()
