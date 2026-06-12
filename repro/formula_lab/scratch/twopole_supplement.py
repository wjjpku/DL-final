#!/usr/bin/env python3
"""T3 supplement: (a) one-pole (delta, lam) rows for matched comparison;
(b) two-pole cells in the high-delta region (where the one-pole optimum is).
Question: does ANY two-pole cell beat the matched one-pole best (-34.5%)?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import (  # noqa: E402
    load_curve, mpl_predict, metrics, MPL_PRECOMPUTED_INIT, SCALES,
)
from formula_lab.lab import feature, fit_origin, DECAY, PROBES  # noqa: E402
from formula_lab.kernels import exp2_feature  # noqa: E402


def probes_to_sharp_feat(featfn):
    deltas, wins = [], 0
    for scale in SCALES:
        p = MPL_PRECOMPUTED_INIT[scale]
        xs, ys = [], []
        for n in PROBES:
            c = load_curve(scale, n)
            xs.append(featfn(c))
            ys.append(c.loss - mpl_predict(p, c))
        kap = max(0.0, fit_origin(np.concatenate(xs), np.concatenate(ys))[0])
        for n in DECAY:
            c = load_curve(scale, n)
            base = mpl_predict(p, c)
            m0 = metrics(c.loss, base)["mae"]
            m1 = metrics(c.loss, base + kap * featfn(c))["mae"]
            deltas.append(100.0 * (m1 / m0 - 1.0))
            wins += int(m1 < m0)
    return float(np.mean(deltas)), wins


def main():
    print("== one-pole (delta, lam) matched rows ==")
    for d in [0.5, 0.75, 1.0]:
        for lam in [3.0, 5.0, 7.0, 10.0]:
            spec = {"form": "pow", "delta": d, "lam": lam}
            v, w = probes_to_sharp_feat(lambda c, s=spec: feature(c, s))
            print(f"  1-pole d={d:4.2f} lam={lam:4.1f}: {v:+6.1f}% ({w}/6)")

    print("\n== two-pole, high-delta region ==")
    rows = []
    for d in [0.5, 0.75, 1.0]:
        for lf in [5.0, 7.0, 10.0, 15.0]:
            for ls in [1.0, 2.0, 5.0]:
                if ls >= lf:
                    continue
                for w in [0.3, 0.5, 0.7, 0.85]:
                    v, wi = probes_to_sharp_feat(
                        lambda c, a=lf, b=ls, ww=w, dd=d:
                        exp2_feature(c, a, b, ww, eta_weight_delta=dd))
                    rows.append((v, wi, d, lf, ls, w))
    rows.sort()
    for v, wi, d, lf, ls, w in rows[:10]:
        print(f"  2-pole d={d:4.2f} lf={lf:4.1f} ls={ls:3.1f} w={w:.2f}: "
              f"{v:+6.1f}% ({wi}/6)")
    print(f"  cells < -34.5%: {sum(1 for r in rows if r[0] < -34.5)}/{len(rows)}")


if __name__ == "__main__":
    main()
