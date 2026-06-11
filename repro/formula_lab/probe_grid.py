#!/usr/bin/env python3
"""Grid (delta, lam) and kernel shapes on the probes-only -> sharp protocol.
Also report the sharp-only oracle (-49% reference upper bound).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import (  # noqa: E402
    load_curve, mpl_predict, metrics, MPL_PRECOMPUTED_INIT, SCALES, PEAK_LR,
)
from formula_lab.lab import fit_origin, DECAY, PROBES  # noqa: E402
from formula_lab.kernels import lomax_feature  # noqa: E402
from formula_lab.lab import feature as one_pole_feature  # noqa: E402


def get_feature(curve, spec):
    if spec.get("kernel") == "lomax":
        # weighted drops via eta_weight_delta in conv kernels
        return lomax_feature(curve, spec["lam"], spec["shape"],
                             eta_weight_delta=spec.get("delta", 0.0))
    return one_pole_feature(curve, spec)


def probes_to_sharp(spec):
    deltas, wins = [], 0
    for scale in SCALES:
        p = MPL_PRECOMPUTED_INIT[scale]
        xs, ys = [], []
        for n in PROBES:
            c = load_curve(scale, n)
            xs.append(get_feature(c, spec))
            ys.append(c.loss - mpl_predict(p, c))
        kappa = max(0.0, fit_origin(np.concatenate(xs), np.concatenate(ys))[0])
        for n in DECAY:
            cu = load_curve(scale, n)
            bp = mpl_predict(p, cu)
            m0 = metrics(cu.loss, bp)["mae"]
            m1 = metrics(cu.loss, bp + kappa * get_feature(cu, spec))["mae"]
            deltas.append(100.0 * (m1 / m0 - 1.0))
            wins += int(m1 < m0)
    return float(np.mean(deltas)), wins


def main():
    print("== (delta, lam) grid, one-pole, probes->sharp mean dMAE% ==")
    lams = [3.0, 5.0, 7.0, 10.0, 14.0]
    print(f"{'':>8s}" + "".join(f" lam={l:<5g}" for l in lams))
    for d in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]:
        row = []
        for lam in lams:
            spec = ({"form": "pow", "delta": d, "lam": lam} if d > 0
                    else {"form": "lr", "lam": lam})
            v, w = probes_to_sharp(spec)
            row.append(f"{v:+6.1f}({w})")
        print(f"d={d:5.2f} " + " ".join(f"{c:>9s}" for c in row))

    print("\n== Lomax kernel x weight, probes->sharp ==")
    for d in [0.0, 0.5, 0.75]:
        for shape in [0.5, 1.0, 2.0]:
            for lam in [5.0, 10.0, 20.0]:
                v, w = probes_to_sharp({"kernel": "lomax", "lam": lam,
                                        "shape": shape, "delta": d})
                print(f"  d={d:4.2f} shape={shape:4.1f} lam={lam:5.1f} {v:+6.1f}% ({w}/6)")


if __name__ == "__main__":
    main()
