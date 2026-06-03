#!/usr/bin/env python3
"""S-time (cumulative-LR) relaxation kernel -- the refined non-adiabatic correction.

deep_tau.py showed the loss relaxes at rate r = eta * lambda_slow (tau ∝ 1/eta,
log-log slope ~ -1), with lambda_slow ~ 10 roughly SCALE-INVARIANT. So the correct
relaxation kernel decays in cumulative-LR S, not in steps:

    DropRelaxS_lam(t) = sum_{t'<=t} exp(-lam (S(t)-S(t'))) relu(eta_{t'-1}-eta_{t'}) / eta_peak
                      = recurrence  s_t = s_{t-1} exp(-lam eta_t) + drop_t.

correction = kappa * DropRelaxS_lam.  Here lam (= lambda_slow) is the scale-invariant
SHAPE constant and kappa (= dL_eq/deta * eta_peak) the N-dependent amplitude -- the
MPL invariance structure.

Tests (FAIR: both arms refit MPL on the same train):
  (1) does the best-fit lam land on the independently measured ~10?
  (2) few-shot improvement vs MPL and vs the step-time kernel;
  (3) is kappa's scale-behaviour the clean amplitude (lam fixed)?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import (  # noqa: E402
    Curve, load_curve, huber_log_residual, metrics, MPL_PRECOMPUTED_INIT, SCALES, PEAK_LR,
)
from validate_theory import fit_mpl, mpl_pred, F_MPL  # noqa: E402

COS = ["cosine_24000.csv", "cosine_72000.csv"]
NONCOS = ["wsd_20000_24000.csv", "wsdld_20000_24000.csv",
          "wsdcon_3.csv", "wsdcon_9.csv", "wsdcon_18.csv"]


def stime_feature(curve: Curve, lam: float) -> np.ndarray:
    eta = curve.lrs.astype(np.float64)
    drop = np.zeros_like(eta); drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0)
    s = np.empty_like(eta)
    acc = 0.0
    for t in range(len(eta)):
        acc = acc * np.exp(-lam * eta[t]) + drop[t]
        s[t] = acc
    return (s / PEAK_LR)[curve.step]


def fit_kappa(p_mpl, train, feats_train):
    base = [mpl_pred(p_mpl, c, fast=True) for c in train]
    la = np.concatenate([c.loss for c in train])
    obj = lambda k: 1e18 if k < 0 else huber_log_residual(
        la, np.concatenate([b + k * f for b, f in zip(base, feats_train)]))
    return float(minimize_scalar(obj, bounds=(0.0, 1e4), method="bounded",
                                 options={"xatol": 1e-4}).x)


def fewshot(lam, feat_cache):
    """Fair few-shot (train cosine x2 + 1 WSD, test other 4, both refit MPL)."""
    mpl_mae, corr_mae, kappas = [], [], []
    for seed in NONCOS:
        trn = COS + [seed]; tst = [x for x in NONCOS if x != seed]
        for scale in SCALES:
            train = [load_curve(scale, x) for x in trn]
            p = fit_mpl(train, MPL_PRECOMPUTED_INIT[scale], F_MPL)
            ftrain = [feat_cache[(scale, x)] for x in trn]
            kappa = fit_kappa(p, train, ftrain)
            kappas.append(kappa)
            for x in tst:
                hc = load_curve(scale, x)
                mpl_mae.append(metrics(hc.loss, mpl_pred(p, hc, fast=False))["mae"])
                pr = mpl_pred(p, hc, fast=False) + kappa * feat_cache[(scale, x)]
                corr_mae.append(metrics(hc.loss, pr)["mae"])
    mpl_mae, corr_mae = np.array(mpl_mae), np.array(corr_mae)
    return mpl_mae, corr_mae, kappas


def main():
    lams = [4.0, 7.0, 10.0, 14.0, 20.0]
    print("=" * 76)
    print("S-time relaxation kernel: fair few-shot vs MPL  (lambda_slow scan)")
    print("  (independently measured lambda_slow ~ 1/(tau*eta) ~ 10)")
    print("=" * 76)
    print(f"  {'lam':>5s} {'MPL':>8s} {'+corr':>8s} {'delta%':>7s} {'wins':>7s}  "
          f"{'kappa 25/100/400':>20s}  kCV")
    best = None
    for lam in lams:
        feat_cache = {(s, n): stime_feature(load_curve(s, n), lam)
                      for s in SCALES for n in COS + NONCOS}
        m, c, ks = fewshot(lam, feat_cache)
        d = (c.mean() / m.mean() - 1) * 100
        # per-scale mean kappa
        ks = np.array(ks).reshape(len(NONCOS), len(SCALES))
        kbar = ks.mean(axis=0)
        kcv = np.std(kbar) / abs(np.mean(kbar)) * 100
        tag = " <<" if c.mean() < m.mean() else ""
        print(f"  {lam:5.0f} {m.mean():8.5f} {c.mean():8.5f} {d:+7.1f} "
              f"{int((c<m).sum()):3d}/{len(c):<2d}  "
              f"[{kbar[0]:5.3f} {kbar[1]:5.3f} {kbar[2]:5.3f}]  {kcv:3.0f}%{tag}")
        if best is None or c.mean() < best[1]:
            best = (lam, c.mean(), d)
    print(f"\n  best lambda_slow = {best[0]:.0f}  (delta {best[2]:+.1f}%); "
          f"compare to measured ~10 from deep_tau.py")


if __name__ == "__main__":
    main()
