#!/usr/bin/env python3
"""E4 verdict (e4_e5_prereg.json): horizon-extended equal-S ladders at m.
p(trunk=12000) and p(trunk=24000) vs the corrected p(3000) = 0.647
[0.610, 0.683].  Design window: step >= trunk + 0.75*T2 per rung.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "represent", "repro"))
import analyze_curves as AC  # noqa: E402
from analyze_floor2 import fit_p, RUNG_ETA  # noqa: E402

CDIR = os.path.join(REPO, "represent", "results", "curves_floor_m")
P3K = (0.647, 0.610, 0.683)


def floor_of(path, trunk, eta2):
    rows = np.genfromtxt(path, delimiter=",", names=True)
    step = np.atleast_1d(rows["step"]).astype(int)
    loss = np.atleast_1d(rows["eval_loss"]).astype(float)
    sm = AC.smooth_by_step(step, loss)
    cut = trunk + 0.75 * int(round(1.2 / eta2))
    m = step >= cut
    assert m.sum() >= 4, f"{path}: only {m.sum()} rows past design cut"
    return float(np.mean(sm[m]))


def main():
    results = {}
    for trunk in [12000, 24000]:
        floors = {}
        for r, eta2 in RUNG_ETA.items():
            f = os.path.join(CDIR, f"floor_{r}_t{trunk}.csv")
            if os.path.exists(f):
                floors[r] = floor_of(f, trunk, eta2)
        if len(floors) < 6:
            print(f"trunk {trunk}: only {len(floors)}/8 rungs -- skip")
            continue
        rungs = sorted(floors, key=lambda r: RUNG_ETA[r])
        etas = np.array([RUNG_ETA[r] for r in rungs])
        fls = np.array([floors[r] for r in rungs])
        mono = bool(np.all(np.diff(fls) > 0))
        p, lo, hi, L0, a0 = fit_p(etas, fls)
        results[trunk] = (p, lo, hi)
        print(f"trunk {trunk}: floors " +
              " ".join(f"{r}={floors[r]:.4f}" for r in rungs))
        print(f"  p = {p:.3f} (90% CI [{lo:.3f},{hi:.3f}]) "
              f"L0={L0:.4f} a={a0:.4f} monotone={mono}")
    if len(results) == 2:
        p3, l3, h3 = P3K
        hw3 = (h3 - l3) / 2
        seq = [(3000, p3, l3, h3)] + [(t, *results[t]) for t in [12000, 24000]]
        print("\np by trunk horizon: " +
              "  ".join(f"{t}: {p:.3f} [{lo:.3f},{hi:.3f}]"
                        for t, p, lo, hi in seq))
        persist = all(
            abs(p - p3) <= 2 * np.sqrt((hw3**2 + ((hi-lo)/2)**2)/2) and hi < 1
            for _, p, lo, hi in seq[1:])
        rising = seq[1][1] > p3 and seq[2][1] > seq[1][1]
        drift = rising and results[24000][1] > h3
        if persist:
            v = "H_PERSIST: sublinearity is horizon-robust"
        elif drift:
            v = "H_DRIFT: budget-indexing load-bearing; p rises with horizon"
        else:
            v = "H_AMBIG: report as measured"
        print("VERDICT:", v)


if __name__ == "__main__":
    main()
