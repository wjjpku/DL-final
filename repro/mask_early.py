#!/usr/bin/env python3
"""Does masking the early (warmup/transient) part of curves improve the fit?

Hypothesis (user): the early phase carries warmup randomness + a power-law-backbone
transient that can mislead the fit of L0/alpha. We refit on curves with the first
m-fraction of points masked, and evaluate test MAE on a FIXED late region (present at
all mask levels, so the comparison is fair), for MPL and for MPL+DropRelaxS (ours).

Protocol: standard cosine->WSD transfer. Fit MPL on cosine[mask:]; predict the WSD
family; evaluate on the held-out test points beyond the largest mask (late region) and
on the full curve. 'Ours' adds the parameter-free predicted correction
kappa * DropRelaxS_{lambda=10}, kappa = c * eta_peak * dL_eq/deta (leave-one-scale-out c).
"""
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import (
    Curve, load_curve, metrics, MPL_PRECOMPUTED_INIT, SCALES, PEAK_LR,
    TRAIN_CURVES, TEST_CURVES,
)
from validate_theory import fit_mpl, mpl_pred, F_MPL
from deep_stime import stime_feature
from nonadiabatic_theory import fit_origin, estimate_dLeq_deta

LAM = 10.0
EVAL_FROM = 0.40          # fixed late region for fair comparison across mask levels
MASKS = [0.0, 0.10, 0.20, 0.30]
DECAY = ["wsd_20000_24000.csv", "wsdld_20000_24000.csv"]


def mask_curve(c, mf):
    """Keep points from index floor(mf*N) onward (mask the first mf fraction)."""
    n = len(c.step); k = int(mf * n)
    return Curve(c.name, c.scale, c.step[k:], c.loss[k:], c.lrs)


def c_of(scale):
    p = MPL_PRECOMPUTED_INIT[scale]; xs, ys = [], []
    for n in DECAY:
        cu = load_curve(scale, n)
        ys.append(cu.loss - mpl_pred(p, cu, fast=False)); xs.append(stime_feature(cu, LAM))
    return fit_origin(np.concatenate(xs), np.concatenate(ys))[0] / (estimate_dLeq_deta(scale)[0] * PEAK_LR)


def eval_mae(p, kappa, scale, name, region):
    c = load_curve(scale, name)
    k = int(region * len(c.step))
    pred = mpl_pred(p, c, fast=False) + (kappa * stime_feature(c, LAM) if kappa else 0.0)
    return metrics(c.loss[k:], pred[k:])["mae"]


def main():
    cvals = {s: c_of(s) for s in SCALES}
    print("=" * 76)
    print(f"Mask early fraction of curves when FITTING; eval on fixed late region "
          f"(step>{EVAL_FROM:.0%}). cosine->WSD test MAE.")
    print("=" * 76)
    for arm in ["MPL", "MPL+ours"]:
        print(f"\n[{arm}]   mean test MAE on WSD family (lower=better)")
        print(f"  {'mask':>6s}  " + "  ".join(f"{s+'M':>9s}" for s in SCALES) + "   overall")
        for mf in MASKS:
            row, allv = [], []
            for s in SCALES:
                train = [mask_curve(load_curve(s, n), mf) for n in TRAIN_CURVES]
                p = fit_mpl(train, MPL_PRECOMPUTED_INIT[s], F_MPL)
                kap = (np.mean([cvals[o] for o in SCALES if o != s]) * PEAK_LR
                       * estimate_dLeq_deta(s)[0]) if arm == "MPL+ours" else 0.0
                maes = [eval_mae(p, kap, s, n, EVAL_FROM) for n in TEST_CURVES]
                m = float(np.mean(maes)); row.append(m); allv.extend(maes)
            tag = ""
            print(f"  {mf:6.0%}  " + "  ".join(f"{v:9.5f}" for v in row)
                  + f"   {np.mean(allv):.5f}{tag}")
    print("\nIf the masked rows (mask>0) beat the mask=0% row, dropping the early")
    print("transient improves the fit/prediction (user's hypothesis).")


if __name__ == "__main__":
    main()
