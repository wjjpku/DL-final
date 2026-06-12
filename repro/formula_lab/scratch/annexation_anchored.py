#!/usr/bin/env python3
"""Closure audit T4: anchored annexation.

The closed verdict rests on the UNREGULARIZED joint 9-param fit (A2).  Test the
obvious rescue: ridge-anchor the 7 backbone log-params at the published MPL fit,

    obj(theta) = huber_log(train) + alpha * ||theta[:7] - log(p_pub)||^2,

sweep alpha over a wide log grid, report the full path of held-out MAEs
(sharp wsd/wsdld AND the smooth/probe family cosine_72000, wsdcon_3/18).
Decisive question: does ANY alpha give a uniform win (sharp better than the
A1 patch, smooth family not degraded)?  alpha -> inf recovers A1; alpha -> 0
recovers A2.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import (  # noqa: E402
    load_curve, mpl_predict, metrics, huber_log_residual,
    MPL_PRECOMPUTED_INIT, SCALES,
)
import validate_theory as V  # noqa: E402
from deep_stime import stime_feature  # noqa: E402
from formula_lab.lab import fit_origin  # noqa: E402
from formula_lab.annexation import TRAIN, HELD, eval_arm  # noqa: E402

SHARP = ["wsd_20000_24000.csv", "wsdld_20000_24000.csv"]
SMOOTH = ["cosine_72000.csv", "wsdcon_3.csv", "wsdcon_18.csv"]


def anchored_fit(scale: str, alpha: float):
    train = [load_curve(scale, n) for n in TRAIN]
    p0 = np.array(MPL_PRECOMPUTED_INIT[scale], float)
    logp0 = np.log(p0)
    feats_cache: dict[float, list[np.ndarray]] = {}

    def feats(lam):
        key = round(float(lam), 4)
        if key not in feats_cache:
            feats_cache[key] = [stime_feature(c, lam) for c in train]
        return feats_cache[key]

    def obj(theta):
        try:
            p7 = np.exp(theta[:7])
            kappa = np.exp(theta[7])
            lam = np.exp(theta[8])
            fs = feats(lam)
            ys, yps = [], []
            for c, f in zip(train, fs):
                ys.append(c.loss)
                yps.append(V.mpl_pred(p7, c, fast=True) + kappa * f)
            y = np.concatenate(ys); yp = np.concatenate(yps)
        except Exception:
            return 1e18
        if np.any(~np.isfinite(yp)) or np.any(yp <= 0):
            return 1e18
        return (huber_log_residual(y, yp)
                + alpha * float(np.sum((theta[:7] - logp0) ** 2)))

    best = None
    for k0, l0 in [(0.03, 10.0), (0.01, 10.0)]:
        x0 = np.concatenate([logp0, [np.log(k0), np.log(l0)]])
        res = minimize(obj, x0, method="Nelder-Mead",
                       options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-12,
                                "adaptive": True})
        if best is None or res.fun < best.fun:
            best = res
    th = best.x
    return np.exp(th[:7]), float(np.exp(th[7])), float(np.exp(th[8]))


def main():
    # reference scale of the data term: A0 huber on train
    for scale in SCALES:
        p_pub = np.array(MPL_PRECOMPUTED_INIT[scale], float)
        h = 0.0
        for n in TRAIN:
            c = load_curve(scale, n)
            h += huber_log_residual(c.loss, mpl_predict(p_pub, c))
        print(f"{scale:>4}M A0 train huber = {h:.3e}")

    alphas = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    agg = {}
    for scale in SCALES:
        p_pub = np.array(MPL_PRECOMPUTED_INIT[scale], float)
        # A1 patch baseline
        train = [load_curve(scale, n) for n in TRAIN]
        resid = [c.loss - mpl_predict(p_pub, c) for c in train]
        best = None
        for lam in [7.0, 10.0, 14.0]:
            x = np.concatenate([stime_feature(c, lam) for c in train])
            y = np.concatenate(resid)
            k = max(0.0, fit_origin(x, y)[0])
            sse = float(np.sum((y - k * x) ** 2))
            if best is None or sse < best[2]:
                best = (lam, k, sse)
        lam1, k1, _ = best
        r0 = eval_arm(scale, p_pub, 0.0, 10.0)
        r1 = eval_arm(scale, p_pub, k1, lam1)
        print(f"\n== {scale}M ==  (A0/A1 reference)")
        print(f"  sharp A0 {np.mean([r0[n] for n in SHARP]):.5f} "
              f"A1 {np.mean([r1[n] for n in SHARP]):.5f} | "
              f"cos72 A0 {r0['cosine_72000.csv']:.5f}")
        agg.setdefault("A0", []).append([r0[n] for n in HELD])
        agg.setdefault("A1", []).append([r1[n] for n in HELD])
        for a in alphas:
            p7, k2, lam2 = anchored_fit(scale, a)
            r = eval_arm(scale, p7, k2, lam2)
            agg.setdefault(a, []).append([r[n] for n in HELD])
            print(f"  alpha={a:7.1e}: kappa={k2:.4f} lam={lam2:6.2f} "
                  f"gamma {p_pub[6]:.3f}->{p7[6]:.3f} B {p_pub[3]:.0f}->{p7[3]:.0f} | "
                  f"sharp {np.mean([r[n] for n in SHARP]):.5f} "
                  f"cos72 {r['cosine_72000.csv']:.5f} "
                  f"wsdcon3 {r['wsdcon_3.csv']:.5f} wsdcon18 {r['wsdcon_18.csv']:.5f}",
                  flush=True)

    print("\n== summary over scales (mean MAE; pct vs A0) ==")
    A0 = np.mean(np.array(agg["A0"]), axis=0)
    i_sharp = [HELD.index(n) for n in SHARP]
    i_smooth = [HELD.index(n) for n in SMOOTH]
    for key in ["A0", "A1"] + alphas:
        M = np.mean(np.array(agg[key]), axis=0)
        sh = np.mean(M[i_sharp]); sm = np.mean(M[i_smooth])
        sh0 = np.mean(A0[i_sharp]); sm0 = np.mean(A0[i_smooth])
        cos = M[HELD.index("cosine_72000.csv")]
        cos0 = A0[HELD.index("cosine_72000.csv")]
        print(f"  {str(key):>8s}: sharp {sh:.5f} ({100*(sh/sh0-1):+6.1f}%)  "
              f"smooth {sm:.5f} ({100*(sm/sm0-1):+6.1f}%)  "
              f"cos72 {cos:.5f} ({100*(cos/cos0-1):+6.1f}%)")


if __name__ == "__main__":
    main()
