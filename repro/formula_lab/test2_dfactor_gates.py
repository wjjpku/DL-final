#!/usr/bin/env python3
"""Adjudication test 2 gates for the backlog-saturation D-factor.

Model: visible amplitude per drop is suppressed by 1/(1 + D_k/D*), where
D_k is the normalized backlog (drop depth + decayed remnant) at deposit time.

Gate (i): a single D* in {0.15, 0.2, 0.3, 0.5} must fit all 9 per-curve
probe/sharp kappa ratios (full-curve, d=0 feature; values from
probe_window.py) with the correct GRADING in drop depth, and beat the
observation-LR null.
Gate (ii): recomputed 10.7M A2/A1 with the factor (D_2 includes the decayed
remnant of deposit 1) must stay inside the measured band [0.86, 0.92].
"""
import numpy as np

# per-probe kappa ratios (kappa_probe/kappa_sharp, full curve, d=0, lam=10)
# rows: scale, cols: wsdcon_3 (D=0.9), wsdcon_9 (D=0.7), wsdcon_18 (D=0.4)
RATIO = {
    "25": [0.36, 0.33, 0.63],
    "100": [0.28, 0.20, 0.24],
    "400": [0.36, 0.28, 0.25],
}
D_PROBE = np.array([0.9, 0.7, 0.4])
# sharp curves deposit gradually: per-step backlog stays small -> reference
# suppression ~1/(1+D_sharp/D*) with effective per-window backlog; the
# simplest version (the proposer's) treats sharp as D~0 (no suppression).


def main():
    print("Gate (i): single D* fit of 9 ratios with correct grading")
    best = None
    for Dstar in [0.15, 0.2, 0.3, 0.5]:
        pred = 1.0 / (1.0 + D_PROBE / Dstar)
        sse, n = 0.0, 0
        for scale, obs in RATIO.items():
            sse += float(np.sum((np.array(obs) - pred) ** 2))
            n += 3
        rmse = np.sqrt(sse / n)
        print(f"  D*={Dstar:4.2f}: pred={np.round(pred,3)} rmse={rmse:.3f}")
        if best is None or rmse < best[1]:
            best = (Dstar, rmse, pred)
    Dstar, rmse, pred = best

    # grading check: model predicts ratio INCREASING in eta2 (decreasing D)
    grading_ok = 0
    for scale, obs in RATIO.items():
        mono = obs[0] <= obs[1] <= obs[2]
        print(f"  {scale:>4}M observed {obs} -> monotone-increasing in eta2: {mono}")
        grading_ok += int(mono)
    print(f"  grading pass: {grading_ok}/3 scales "
          f"(model REQUIRES increasing; 0.86-0.92 of variance must follow D)")

    # observation-LR null: ratio depends on eta2 only via observation window,
    # i.e. any monotone function of eta2 fits equally well -> discrimination
    # comes from the grading + the sharp-curve cells.  With grading failing
    # at >=2 scales the D-factor cannot beat the null.
    print(f"\nbest D*={Dstar} rmse={rmse:.3f}")

    print("\nGate (ii): A2/A1 with the D-factor (10.7M twodrop)")
    lam_S = 1.0
    dS12 = 2000 * 0.75e-3 * lam_S          # S-time gap between drops
    remnant = 0.5 * np.exp(-dS12)
    for Dstar in [0.15, 0.2, 0.3, 0.5]:
        D1 = 0.5
        D2 = 0.35 + remnant
        vis1 = 0.5 / (1.0 + D1 / Dstar)
        vis2 = 0.35 / (1.0 + D2 / Dstar)
        ratio = vis2 / vis1
        ok = 0.86 <= ratio <= 0.92
        print(f"  D*={Dstar:4.2f}: remnant={remnant:.3f} A2/A1={ratio:.3f} "
              f"in [0.86,0.92]: {ok}")


if __name__ == "__main__":
    main()
