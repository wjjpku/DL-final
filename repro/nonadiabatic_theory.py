#!/usr/bin/env python3
"""Verify the non-adiabatic (loss-relaxation-lag) theory of the MPL residual.

Theory (docs/core/nonadiabatic_correction.md): MPL fit on the slow cosine schedule
learns the quasi-static / adiabatic loss L_eq(eta). On a fast WSD decay the loss
lags above L_eq, and linear response gives the residual EXACTLY as

    Delta L(t) = L_true - L_MPL ~ kappa * DropRelax_tau(t),
    DropRelax_tau(t) = sum_{t'<=t} (1-1/tau)^(t-t') * relu(eta_{t'-1}-eta_{t'}),

with kappa = (dL_eq/deta)*PEAK_LR  and  tau = loss relaxation time.

Falsifiable core claim tested here: regress the cosine-fit-MPL residual on
DropRelax (through the origin). The theory predicts:
  (i)  positive slope kappa (right sign),
  (ii) high R^2 on the sharp-decay curves (wsd/wsdld), ~0 feature on cosine,
  (iii) kappa increasing with model size N,
  (iv) kappa ~ PEAK_LR * dL_eq/deta, with dL_eq/deta estimated INDEPENDENTLY from
       the constant/two-stage final losses (noise floor vs LR).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import (  # noqa: E402
    Curve, load_curve, compute_s1, mpl_predict, MPL_PRECOMPUTED_INIT, SCALES, PEAK_LR,
)

TAU = 1200.0
DECAY = ["wsd_20000_24000.csv", "wsdld_20000_24000.csv"]
PLATEAU = ["wsdcon_3.csv", "wsdcon_9.csv", "wsdcon_18.csv"]


def drop_relax(curve: Curve, tau=TAU):
    eta = curve.lrs.astype(np.float64)
    drop = np.zeros_like(eta); drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0)
    a = 1.0 - 1.0 / tau
    s, _ = lfilter([1.0], [1.0, -a], drop, zi=[0.0])
    return (s / PEAK_LR)[curve.step]


def fit_origin(x, y):
    """Least squares y ~ kappa*x through origin; return kappa, R^2."""
    kap = float(np.dot(x, y) / np.dot(x, x))
    resid = y - kap * x
    ss = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - np.sum(resid ** 2) / ss if ss > 0 else float("nan")
    return kap, r2


def estimate_dLeq_deta(scale):
    """Independent dL_eq/deta from two-stage final losses (noise floor vs stage-2 LR).

    wsdcon_{3,9,18} end after 8000 steps held at lr_b in {3,9,18}e-5. Those 8000
    steps >> tau, so the system is settled: final loss ~ L_eq(lr_b) + backbone(S).
    We subtract the MPL backbone A*S^-alpha to isolate the eta-dependent floor, then
    regress floor vs lr_b -> slope dL_eq/deta."""
    p = MPL_PRECOMPUTED_INIT[scale]
    L0, A, alpha = p[0], p[1], p[2]
    etas, floors = [], []
    for n, mul in [("wsdcon_3.csv", 3), ("wsdcon_9.csv", 9), ("wsdcon_18.csv", 18)]:
        c = load_curve(scale, n)
        s1 = compute_s1(c)
        backbone = L0 + A * np.power(s1, -alpha)
        floor_tail = float(np.mean((c.loss - backbone)[-5:]))  # last points, settled
        etas.append(mul * 1e-5); floors.append(floor_tail)
    etas, floors = np.array(etas), np.array(floors)
    slope = np.polyfit(etas, floors, 1)[0]
    return slope, etas, floors


def main():
    print("=" * 78)
    print(f"Non-adiabatic theory check: residual(cosine-fit MPL) ~ kappa*DropRelax (tau={TAU:.0f})")
    print("=" * 78)
    kap_decay = {}
    for scale in SCALES:
        p = MPL_PRECOMPUTED_INIT[scale]
        print(f"\n[{scale}M]")
        # pooled regression over the two sharp-decay curves
        xs, ys = [], []
        for n in DECAY + PLATEAU:
            c = load_curve(scale, n)
            r = c.loss - mpl_predict(p, c)
            f = drop_relax(c)
            kap, r2 = fit_origin(f, r)
            mark = "decay " if n in DECAY else "plateau"
            print(f"   {mark} {n:22s}  kappa={kap:7.4f}  R^2={r2:6.3f}  "
                  f"resid[max]={r.max():+.4f} feat[max]={f.max():.3f}")
            if n in DECAY:
                xs.append(f); ys.append(r)
        X, Y = np.concatenate(xs), np.concatenate(ys)
        kap, r2 = fit_origin(X, Y)
        kap_decay[scale] = kap
        slope, etas, floors = estimate_dLeq_deta(scale)
        kap_pred = slope * PEAK_LR
        print(f"   >> sharp-decay pooled:  kappa={kap:.4f}  R^2={r2:.3f}")
        print(f"   >> independent dL_eq/deta from noise floor = {slope:.1f}  "
              f"=> predicted kappa=PEAK_LR*dLeq/deta = {kap_pred:.4f}")

    ks = np.array([kap_decay[s] for s in SCALES])
    Ns = np.array([25.0, 100.0, 400.0])
    print("\n" + "=" * 78)
    print("kappa vs model size N (theory: kappa = dL_eq/deta grows with N):")
    for s, k in zip(SCALES, ks):
        print(f"   {s:>4s}M  kappa={k:.4f}")
    # fit kappa ~ N^p
    pos = ks > 0
    if pos.sum() >= 2:
        p_exp = np.polyfit(np.log(Ns[pos]), np.log(ks[pos]), 1)[0]
        print(f"   power-law fit: kappa ~ N^{p_exp:.2f}")


if __name__ == "__main__":
    main()
