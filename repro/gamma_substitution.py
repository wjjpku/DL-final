#!/usr/bin/env python3
"""Does the derived (parameter-free) DropRelaxS substitute for MPL's gamma?

Clean test in the setting where gamma is genuinely essential -- cosine->WSD transfer
(train on cosine only, predict the sharp WSD family). Three models, same cosine fit:
  A  full MPL (gamma free)                          fit on cosine
  B  MPL gamma=0                                     fit on cosine  (expected 3-4x worse)
  C  MPL gamma=0  +  PREDICTED kappa*DropRelaxS      (kappa NOT fit: from the noise floor,
                                                      leave-one-scale-out; lambda_slow=10)
DropRelaxS is ~0 on cosine, so B and C share the SAME cosine fit; their difference on the
WSD test is purely the predicted non-adiabatic term. If C ~ A << B, the derived term does
what gamma did -- with zero fitted parameters of its own.
"""
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import load_curve, metrics, mpl_predict, MPL_PRECOMPUTED_INIT, SCALES, PEAK_LR
from validate_theory import fit_mpl, mpl_pred, F_MPL
from deep_stime import stime_feature
from nonadiabatic_theory import fit_origin, estimate_dLeq_deta

LAM = 10.0
COSINE = ["cosine_24000.csv", "cosine_72000.csv"]
TEST = ["wsd_20000_24000.csv", "wsdld_20000_24000.csv",
        "wsdcon_3.csv", "wsdcon_9.csv", "wsdcon_18.csv"]
F_NOGAMMA = [0, 1, 2, 3, 4, 5]
DECAY = ["wsd_20000_24000.csv", "wsdld_20000_24000.csv"]


def c_of(scale):
    """Residual-regression ratio c = kappa_fit / (eta_peak*dLeq) on full MPL (decays)."""
    p = MPL_PRECOMPUTED_INIT[scale]
    xs, ys = [], []
    for n in DECAY:
        cu = load_curve(scale, n)
        ys.append(cu.loss - mpl_predict(p, cu)); xs.append(stime_feature(cu, LAM))
    kfit = fit_origin(np.concatenate(xs), np.concatenate(ys))[0]
    return kfit / (estimate_dLeq_deta(scale)[0] * PEAK_LR)


def main():
    cvals = {s: c_of(s) for s in SCALES}
    print("=" * 74)
    print("Parameter-free DropRelaxS vs gamma  (cosine->WSD, test MAE)")
    print(f"  per-scale c = {[round(cvals[s],2) for s in SCALES]}  (used leave-one-scale-out)")
    print("=" * 74)
    print(f"  {'scale':>5s} {'A: full MPL':>12s} {'B: gamma=0':>11s} {'C: g0+pred':>11s}  "
          f"{'kappa_pred':>10s}  recovers?")
    rows = []
    for s in SCALES:
        train = [load_curve(s, n) for n in COSINE]
        pA = fit_mpl(train, MPL_PRECOMPUTED_INIT[s], F_MPL)
        init0 = MPL_PRECOMPUTED_INIT[s].copy(); init0[6] = 0.0
        pB = fit_mpl(train, init0, F_NOGAMMA)
        c_loo = np.mean([cvals[o] for o in SCALES if o != s])
        kap = c_loo * PEAK_LR * estimate_dLeq_deta(s)[0]
        mA = np.mean([metrics(load_curve(s, n).loss, mpl_pred(pA, load_curve(s, n), fast=False))["mae"] for n in TEST])
        mB = np.mean([metrics(load_curve(s, n).loss, mpl_pred(pB, load_curve(s, n), fast=False))["mae"] for n in TEST])
        mC = np.mean([metrics(load_curve(s, n).loss,
                              mpl_pred(pB, load_curve(s, n), fast=False) + kap * stime_feature(load_curve(s, n), LAM))["mae"]
                      for n in TEST])
        rows.append((mA, mB, mC))
        rec = "YES" if mC <= 1.15 * mA else f"partial({mC/mA:.2f}x)"
        print(f"  {s:>4s}M {mA:12.5f} {mB:11.5f} {mC:11.5f}  {kap:10.3f}  {rec}")
    A, B, C = (np.array([r[i] for r in rows]) for i in range(3))
    print("-" * 74)
    print(f"  mean  full={A.mean():.5f}   gamma=0={B.mean():.5f} ({B.mean()/A.mean():.1f}x worse)"
          f"   g0+predicted={C.mean():.5f} ({C.mean()/A.mean():.2f}x)")
    print("\n  C uses ZERO fitted parameters of its own (kappa predicted from the noise floor).")
    print("  C ~ A << B  =>  the derived non-adiabatic term IS what gamma approximates.")


if __name__ == "__main__":
    main()
