#!/usr/bin/env python3
"""g1d1 zero-GPU falsification (convergence review #12 design escape): does a
UNIFIED additive noise-floor term gap(B,depth)=c_N*depth/(1+B/B0)-D_dep0 with
ONE (c_N,B0,D_dep0) fit the existing bladder paired gaps at BOTH drop depths,
and does it beat A4's per-depth free-exponent power law?  A4 already rejected
the eta/B and 1/sqrt(B) families; this checks the unified-one-B0 version on
the committed 10 points before spending any GPU on a 3rd depth rung."""
import numpy as np
from scipy.optimize import curve_fit, minimize_scalar

# common-horizon [6000,7000) paired drop-minus-control gaps (DECISION_TABLE
# AUDIT-A / a4_signflip_pricing.json; bladder_rewindow.py), s1337:
B = np.array([12, 24, 48, 96, 192], float)
GAP = {
    1.4e-3: np.array([-0.1094, -0.0658, -0.0285, +0.0057, +0.0335]),  # e10 (eta2=1e-4)
    1.1e-3: np.array([-0.0906, -0.0614, -0.0383, -0.0169, -0.0003]),  # e40 (eta2=4e-4)
}
DEPTHS = sorted(GAP)


def main():
    # --- g1d1 unified noise-floor: ONE (c_N, B0, D_dep0) across BOTH depths ---
    Ball = np.concatenate([B for _ in DEPTHS])
    dall = np.concatenate([np.full_like(B, d) for d in DEPTHS])
    gall = np.concatenate([GAP[d] for d in DEPTHS])

    def nf(X, c_N, B0, D0):
        b, d = X
        return c_N * d / (1 + b / B0) - D0
    try:
        po, _ = curve_fit(nf, (Ball, dall), gall, p0=[100, 50, 0.05],
                          maxfev=200000, bounds=([0, 1, 0], [1e5, 1e4, 1]))
        pred = nf((Ball, dall), *po)
        mae_nf = float(np.mean(np.abs(gall - pred)))
        print(f"g1d1 unified noise-floor c_N*depth/(1+B/B0)-D0: "
              f"c_N={po[0]:.1f} B0={po[1]:.1f} D0={po[2]:.4f}  "
              f"MAE={mae_nf:.5f} ({len(gall)} pts, 3 params)")
    except Exception as e:
        print("noise-floor fit failed:", e); mae_nf = 9.9

    # --- A4 baseline: per-depth free-exponent law gap = a_d - b_d * B^-0.20 ---
    mae_a4 = []
    for d in DEPTHS:
        def pw(b, a, bb):
            return a - bb * b ** (-0.20)
        po, _ = curve_fit(pw, B, GAP[d], p0=[0.2, 0.5], maxfev=100000)
        mae_a4.append(np.mean(np.abs(GAP[d] - pw(B, *po))))
    mae_a4 = float(np.mean(mae_a4))
    print(f"A4 per-depth free-exponent B^-0.20 law: MAE={mae_a4:.5f} "
          f"(2 params/depth)")

    ratio = mae_nf / max(mae_a4, 1e-9)
    print(f"\nunified-noise-floor MAE / A4 MAE = {ratio:.1f}x")
    if ratio > 2:
        v = ("NOISE_FLOOR_REJECTED: the unified one-B0 eta/B noise-floor form "
             f"underperforms A4 by {ratio:.0f}x on committed data -- the "
             "saturating-eta/B family A4 already rejected does not gain a "
             "batch-axis mechanism; a 3rd-depth GPU rung is NOT warranted.")
    else:
        v = ("NOISE_FLOOR_COMPETITIVE: within 2x of A4 -- a 3rd-depth rung "
             "could be worth ~1-2 GPU-h to test the affine-B* prediction.")
    print("\nVERDICT:", v)


if __name__ == "__main__":
    main()
