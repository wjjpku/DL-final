#!/usr/bin/env python3
"""Direction B ('annexation'): fold the lag term INTO the law and fit all
9 parameters jointly on the official train split, instead of freezing MPL
and patching post hoc.

    L(t) = L0 + A S^-a + B*LD(C,beta,gamma) + kappa * DropRelaxS_{lam_s}(t)

Train (official split): cosine_24000, constant_24000, wsdcon_9.
Held-out: wsd, wsdld, cosine_72000, wsdcon_3, wsdcon_18.

Arms:
  A0 frozen MPL (published params)            -- baseline
  A1 frozen MPL + patch (kappa, lam in {7,10,14} fit on train residuals)
  A2 joint 9-param fit (annexation)
Also report the gamma shift under A2 (does the lag term free gamma?).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import (  # noqa: E402
    load_curve, mpl_predict, metrics, huber_log_residual,
    MPL_PRECOMPUTED_INIT, SCALES,
)
import validate_theory as V  # noqa: E402
from deep_stime import stime_feature  # noqa: E402
from formula_lab.lab import fit_origin  # noqa: E402

TRAIN = ["cosine_24000.csv", "constant_24000.csv", "wsdcon_9.csv"]
HELD = ["wsd_20000_24000.csv", "wsdld_20000_24000.csv", "cosine_72000.csv",
        "wsdcon_3.csv", "wsdcon_18.csv"]


def joint_fit(scale: str):
    train = [load_curve(scale, n) for n in TRAIN]
    p0 = np.array(MPL_PRECOMPUTED_INIT[scale], float)

    feats_cache: dict[float, list[np.ndarray]] = {}

    def feats(lam):
        key = round(float(lam), 4)
        if key not in feats_cache:
            feats_cache[key] = [stime_feature(c, lam) for c in train]
        return feats_cache[key]

    def predict_all(theta):
        p7 = np.exp(theta[:7])
        kappa = np.exp(theta[7])
        lam = np.exp(theta[8])
        fs = feats(lam)
        preds, losses = [], []
        for c, f in zip(train, fs):
            pred = V.mpl_pred(p7, c, fast=True) + kappa * f
            preds.append(pred)
            losses.append(c.loss)
        return np.concatenate(losses), np.concatenate(preds)

    def obj(theta):
        try:
            y, yp = predict_all(theta)
        except Exception:
            return 1e18
        if np.any(~np.isfinite(yp)) or np.any(yp <= 0):
            return 1e18
        return huber_log_residual(y, yp)

    best = None
    for k0, l0 in [(0.03, 10.0), (0.01, 10.0), (0.05, 5.0)]:
        x0 = np.concatenate([np.log(p0), [np.log(k0), np.log(l0)]])
        res = minimize(obj, x0, method="Nelder-Mead",
                       options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-12,
                                "adaptive": True})
        if best is None or res.fun < best.fun:
            best = res
    th = best.x
    return np.exp(th[:7]), float(np.exp(th[7])), float(np.exp(th[8])), best.fun


def eval_arm(scale, p7, kappa, lam):
    rows = {}
    for n in HELD:
        c = load_curve(scale, n)
        pred = mpl_predict(np.asarray(p7, float), c)
        if kappa > 0:
            pred = pred + kappa * stime_feature(c, lam)
        rows[n] = metrics(c.loss, pred)["mae"]
    return rows


def main():
    agg = {a: [] for a in ["A0", "A1", "A2"]}
    for scale in SCALES:
        p_pub = np.array(MPL_PRECOMPUTED_INIT[scale], float)

        # A1: patch kappa on train residuals, lam grid
        train = [load_curve(scale, n) for n in TRAIN]
        resid = [c.loss - mpl_predict(p_pub, c) for c in train]
        best = None
        for lam in [7.0, 10.0, 14.0]:
            x = np.concatenate([stime_feature(c, lam) for c in train])
            y = np.concatenate(resid)
            k, _ = fit_origin(x, y)
            k = max(0.0, k)
            sse = float(np.sum((y - k * x) ** 2))
            if best is None or sse < best[2]:
                best = (lam, k, sse)
        lam1, k1, _ = best

        # A2: joint fit
        p7, k2, lam2, fobj = joint_fit(scale)

        r0 = eval_arm(scale, p_pub, 0.0, 10.0)
        r1 = eval_arm(scale, p_pub, k1, lam1)
        r2 = eval_arm(scale, p7, k2, lam2)
        print(f"\n== {scale}M ==")
        print(f"  A1 patch: kappa={k1:.4f} lam={lam1:g}")
        print(f"  A2 joint: kappa={k2:.4f} lam={lam2:.2f} "
              f"gamma {p_pub[6]:.3f}->{p7[6]:.3f}  B {p_pub[3]:.0f}->{p7[3]:.0f} "
              f"beta {p_pub[5]:.3f}->{p7[5]:.3f} C {p_pub[4]:.2f}->{p7[4]:.2f}")
        print(f"  {'curve':22s} {'A0 frozen':>10s} {'A1 patch':>10s} {'A2 joint':>10s}")
        for n in HELD:
            print(f"  {n:22s} {r0[n]:10.5f} {r1[n]:10.5f} {r2[n]:10.5f}")
        for a, r in [("A0", r0), ("A1", r1), ("A2", r2)]:
            agg[a].append(np.mean([r[n] for n in
                                   ["wsd_20000_24000.csv", "wsdld_20000_24000.csv"]]))

    print("\n== sharp-decay held-out MAE (mean over scales) ==")
    m0 = np.mean(agg["A0"])
    for a in ["A0", "A1", "A2"]:
        m = np.mean(agg[a])
        print(f"  {a}: {m:.5f}  ({100*(m/m0-1):+.1f}% vs frozen)")


if __name__ == "__main__":
    main()
