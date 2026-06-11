#!/usr/bin/env python3
"""FAIR early-masking experiment: re-calibrate the correction on the MASKED baseline.

mask_early.py used a kappa calibrated on the UNMASKED MPL, so it over-corrected once
masking improved the backbone. Here, for every mask level, EVERYTHING is recomputed
consistently with the masked MPL: the backbone (refit on masked cosine), dL_eq/deta
(noise floor minus the masked backbone), and c = kappa_fit/(eta_peak*dL_eq/deta) per
scale; the target's kappa is predicted leave-one-scale-out (no target-curve fitting).
Evaluation is on wsd/wsdld only (the dL_eq/deta probe uses wsdcon -> no leakage).
"""
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import (
    Curve, load_curve, compute_s1, metrics, MPL_PRECOMPUTED_INIT, SCALES, PEAK_LR, TRAIN_CURVES,
)
from validate_theory import fit_mpl, mpl_pred, F_MPL
from deep_stime import stime_feature
from nonadiabatic_theory import fit_origin

LAM = 10.0
EVAL_FROM = 0.40
MASKS = [0.0, 0.10, 0.20, 0.30, 0.40]
DECAY = ["wsd_20000_24000.csv", "wsdld_20000_24000.csv"]
WSDCON = [("wsdcon_3.csv", 3e-5), ("wsdcon_9.csv", 9e-5), ("wsdcon_18.csv", 18e-5)]


def mask_curve(c, mf):
    k = int(mf * len(c.step))
    return Curve(c.name, c.scale, c.step[k:], c.loss[k:], c.lrs)


def dLeq_deta(scale, p):
    L0, A, alpha = p[0], p[1], p[2]
    etas, floors = [], []
    for n, lrb in WSDCON:
        c = load_curve(scale, n)
        backbone = L0 + A * np.power(compute_s1(c), -alpha)
        floors.append(float(np.mean((c.loss - backbone)[-5:]))); etas.append(lrb)
    return np.polyfit(np.array(etas), np.array(floors), 1)[0]


def c_of_masked(scale, p):
    xs, ys = [], []
    for n in DECAY:
        cu = load_curve(scale, n)
        ys.append(cu.loss - mpl_pred(p, cu, fast=False)); xs.append(stime_feature(cu, LAM))
    kfit = fit_origin(np.concatenate(xs), np.concatenate(ys))[0]
    return kfit / (dLeq_deta(scale, p) * PEAK_LR)


def eval_mae(p, kappa, scale, name):
    c = load_curve(scale, name); k = int(EVAL_FROM * len(c.step))
    pred = mpl_pred(p, c, fast=False) + (kappa * stime_feature(c, LAM) if kappa else 0.0)
    return metrics(c.loss[k:], pred[k:])["mae"]


def main():
    print("=" * 70)
    print(f"FAIR masking: kappa re-calibrated on the masked baseline. "
          f"test MAE on wsd/wsdld (step>{EVAL_FROM:.0%}).")
    print("=" * 70)
    print(f"  {'mask':>6s}  {'MPL':>9s}  {'MPL+ours(fair)':>15s}  {'ours delta':>11s}")
    for mf in MASKS:
        P = {s: fit_mpl([mask_curve(load_curve(s, n), mf) for n in TRAIN_CURVES],
                        MPL_PRECOMPUTED_INIT[s], F_MPL) for s in SCALES}
        cvals = {s: c_of_masked(s, P[s]) for s in SCALES}
        mpl_all, our_all = [], []
        for s in SCALES:
            c_loo = np.mean([cvals[o] for o in SCALES if o != s])
            kap = c_loo * PEAK_LR * dLeq_deta(s, P[s])
            for n in DECAY:
                mpl_all.append(eval_mae(P[s], 0.0, s, n))
                our_all.append(eval_mae(P[s], kap, s, n))
        mpl_m, our_m = np.mean(mpl_all), np.mean(our_all)
        d = (our_m / mpl_m - 1) * 100
        print(f"  {mf:6.0%}  {mpl_m:9.5f}  {our_m:15.5f}  {d:+10.1f}%")
    print("\nMPL column: does masking help the (adiabatic) backbone fit?")
    print("ours delta: with kappa re-calibrated on the masked baseline, does the")
    print("            non-adiabatic correction still add value on top of masking?")


if __name__ == "__main__":
    main()
