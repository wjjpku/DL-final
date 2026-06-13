#!/usr/bin/env python3
"""bs192 mini-ladder verdict (prereg=bs192_ladder_prereg.json): does batch
size 192 produce the public-bed superlinear floor exponent at the m recipe?
Reference: m bs48 corrected p = 0.647 (90% CI [0.610, 0.683])."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "represent", "repro"))
from analyze_floor2 import tail_floor, fit_p, RUNG_ETA  # noqa: E402

CDIR = os.path.join(REPO, "represent", "results", "curves_floor_m_b192")
REF_P, REF_LO, REF_HI = 0.647, 0.610, 0.683


def main():
    floors = {}
    for r in ["20", "40", "80", "150"]:
        f = os.path.join(CDIR, f"floor_{r}.csv")
        fl, _, _ = tail_floor(f, RUNG_ETA[r])
        floors[r] = fl
        print(f"  floor_{r:<4s} (bs192): {fl:.4f}")
    rungs = sorted(floors, key=lambda r: RUNG_ETA[r])
    etas = np.array([RUNG_ETA[r] for r in rungs])
    fls = np.array([floors[r] for r in rungs])
    mono = bool(np.all(np.diff(fls) > 0))
    p, lo, hi, L0, a0 = fit_p(etas, fls)
    print(f"\n  fit: floor = {L0:.4f} + {a0:.4f}*x^{p:.3f}  "
          f"90% CI [{lo:.3f},{hi:.3f}]  monotone={mono}")
    hw_ref = (REF_HI - REF_LO) / 2
    hw = (hi - lo) / 2
    pooled = float(np.sqrt((hw_ref ** 2 + hw ** 2) / 2))
    dp = p - REF_P
    print(f"  dp(bs48 -> bs192) = {dp:+.3f}; 2x pooled half-width = "
          f"{2*pooled:.3f}")
    if dp > 2 * pooled and lo > 1:
        v = "BS_DRIVES (superlinearity reproduced by batch size)"
    elif dp > 2 * pooled:
        v = "BS_ELEVATES (p raised but not superlinear)"
    else:
        v = "BS_NULL (attribution falls to data/recipe/protocol)"
    print(f"  VERDICT: {v}")


if __name__ == "__main__":
    main()
