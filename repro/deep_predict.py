#!/usr/bin/env python3
"""Parameter-free non-adiabatic correction: predict it, don't fit it.

Everything is determined WITHOUT touching the target curve's decay residual:
  lambda_slow = 10            universal shape constant (measured in deep_tau.py)
  c           = mean ratio    universal calibration, LEAVE-ONE-SCALE-OUT
  dL_eq/deta  = noise floor    measured on the target scale's two-stage finals
  kappa       = c * eta_peak * dL_eq/deta

Then evaluate, with ZERO fitting, on the held-out sharp-decay curves wsd / wsdld:
  MPL(cosine-fit)            vs   MPL(cosine-fit) + kappa * DropRelaxS_10.
The calibration uses wsdcon (noise floor); the test uses wsd/wsdld -> no leakage.
This is the full cross-scale predictive chain: small-model constants + target-scale
noise floor predict the large-model curve, addressing MPL's no-cross-scale weakness.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import (  # noqa: E402
    load_curve, mpl_predict, metrics, MPL_PRECOMPUTED_INIT, SCALES, PEAK_LR,
)
from deep_stime import stime_feature  # noqa: E402
from nonadiabatic_theory import fit_origin, estimate_dLeq_deta  # noqa: E402

LAM = 10.0
DECAY = ["wsd_20000_24000.csv", "wsdld_20000_24000.csv"]


def kappa_fit(scale):
    """In-sample regression kappa on this scale's decays (used only to calibrate c)."""
    p = MPL_PRECOMPUTED_INIT[scale]
    xs, ys = [], []
    for n in DECAY:
        c = load_curve(scale, n)
        ys.append(c.loss - mpl_predict(p, c)); xs.append(stime_feature(c, LAM))
    return fit_origin(np.concatenate(xs), np.concatenate(ys))[0]


def main():
    # per-scale ratio c_s = kappa_fit / (eta_peak * dL_eq/deta)
    ratio = {}
    for s in SCALES:
        kpred = estimate_dLeq_deta(s)[0] * PEAK_LR
        ratio[s] = kappa_fit(s) / kpred
    print("=" * 72)
    print("Parameter-free cross-scale prediction of the non-adiabatic correction")
    print(f"  lambda_slow={LAM:.0f} (universal); per-scale c=kappa_fit/kappa_pred: "
          + ", ".join(f"{s}={ratio[s]:.2f}" for s in SCALES))
    print("=" * 72)
    print(f"  {'target':>6s} {'c(LOO)':>7s} {'curve':16s} {'MAE_MPL':>9s} "
          f"{'MAE_pred':>9s} {'delta%':>7s}")
    agg_m, agg_c = [], []
    for tgt in SCALES:
        c_loo = np.mean([ratio[s] for s in SCALES if s != tgt])   # leave-one-scale-out
        dLeq = estimate_dLeq_deta(tgt)[0]
        kappa = c_loo * PEAK_LR * dLeq                            # PREDICTED, no fitting
        p = MPL_PRECOMPUTED_INIT[tgt]
        for n in DECAY:
            cu = load_curve(tgt, n)
            m_mpl = metrics(cu.loss, mpl_predict(p, cu))["mae"]
            pred = mpl_predict(p, cu) + kappa * stime_feature(cu, LAM)
            m_pred = metrics(cu.loss, pred)["mae"]
            agg_m.append(m_mpl); agg_c.append(m_pred)
            print(f"  {tgt:>5s}M {c_loo:7.2f} {n:16s} {m_mpl:9.5f} {m_pred:9.5f} "
                  f"{(m_pred/m_mpl-1)*100:+7.1f}")
    agg_m, agg_c = np.array(agg_m), np.array(agg_c)
    print("\n  " + "-" * 60)
    print(f"  OVERALL (zero-fit, cross-scale-transferred constants):")
    print(f"    MPL={agg_m.mean():.5f}  predicted-corr={agg_c.mean():.5f}  "
          f"delta={(agg_c.mean()/agg_m.mean()-1)*100:+.1f}%  "
          f"wins={int((agg_c<agg_m).sum())}/{len(agg_c)}")


if __name__ == "__main__":
    main()
