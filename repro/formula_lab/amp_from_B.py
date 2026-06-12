#!/usr/bin/env python3
"""Is kappa predictable from MPL's own fitted B (annealing saturation gain)?

MPL: L = L0 + A S^-alpha + B * sum dEta_k G(...).  A drop deta whose G has
saturated reduces loss by B*deta, so MPL's implied equilibrium-floor
sensitivity is dL_eq/deta ~ B.  Compare kappa_fit/(eta_peak*B) stability
across scales vs the probe-slope chain kappa_fit/(eta_peak*dL_eq/deta).
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import MPL_PRECOMPUTED_INIT, PEAK_LR, SCALES  # noqa: E402
from nonadiabatic_theory import estimate_dLeq_deta  # noqa: E402
from deep_predict import kappa_fit  # noqa: E402


def main():
    rows = []
    for s in SCALES:
        kf = kappa_fit(s)
        dl = estimate_dLeq_deta(s)[0]
        B = MPL_PRECOMPUTED_INIT[s][3]
        rows.append((s, kf, dl, B, kf / (dl * PEAK_LR), kf / (B * PEAK_LR)))
    for r in rows:
        print("%5sM kf=%.4f dLdeta=%.1f B=%.1f ratio_probe=%.3f ratio_B=%.4f" % r)
    rp = np.array([r[4] for r in rows])
    rb = np.array([r[5] for r in rows])
    print("CV ratio_probe=%.1f%%  CV ratio_B=%.1f%%"
          % (rp.std() / rp.mean() * 100, rb.std() / rb.mean() * 100))


if __name__ == "__main__":
    main()
