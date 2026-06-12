#!/usr/bin/env python3
"""Hunt 3: concentration-matched probe selection.
Rule: calibrate kappa on the probe whose stage-2 LR is closest to the target
schedule's terminal LR (deepest probe for full decays), instead of pooling.
Leakage-clean: uses only probe curves + the target's *schedule* (known a
priori), never target losses.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import (  # noqa: E402
    load_curve, mpl_predict, metrics, MPL_PRECOMPUTED_INIT, SCALES, PEAK_LR,
)
from formula_lab.lab import feature, fit_origin, DECAY, PROBES  # noqa: E402

# terminal LR of the public sharp decays
for n in DECAY:
    c = load_curve("100", n)
    print(f"{n}: final lr = {c.lrs[-1]:.2e}, peak = {c.lrs.max():.2e}")
for n in PROBES:
    c = load_curve("100", n)
    print(f"{n}: stage2 lr = {c.lrs[-1]:.2e}")

SPECS = [
    ("lr@10", {"form": "lr", "lam": 10}),
    ("pow d=.25@10", {"form": "pow", "delta": 0.25, "lam": 10}),
    ("pow d=.5@10", {"form": "pow", "delta": 0.5, "lam": 10}),
    ("pow d=.5@5", {"form": "pow", "delta": 0.5, "lam": 5}),
    ("pow d=.75@10", {"form": "pow", "delta": 0.75, "lam": 10}),
]


def probes_subset(spec, cal_probes):
    deltas, wins = [], 0
    for scale in SCALES:
        p = MPL_PRECOMPUTED_INIT[scale]
        xs, ys = [], []
        for n in cal_probes:
            c = load_curve(scale, n)
            xs.append(feature(c, spec))
            ys.append(c.loss - mpl_predict(p, c))
        kappa = max(0.0, fit_origin(np.concatenate(xs), np.concatenate(ys))[0])
        for n in DECAY:
            cu = load_curve(scale, n)
            bp = mpl_predict(p, cu)
            m0 = metrics(cu.loss, bp)["mae"]
            m1 = metrics(cu.loss, bp + kappa * feature(cu, spec))["mae"]
            deltas.append(100.0 * (m1 / m0 - 1.0))
            wins += int(m1 < m0)
    return float(np.mean(deltas)), wins


def dilution_subset(spec, cal_probes):
    deltas, wins = [], 0
    for scale in SCALES:
        p = MPL_PRECOMPUTED_INIT[scale]
        for held in DECAY:
            cal = [m for m in DECAY if m != held] + cal_probes
            xs, ys = [], []
            for n in cal:
                c = load_curve(scale, n)
                xs.append(feature(c, spec))
                ys.append(c.loss - mpl_predict(p, c))
            kappa = max(0.0, fit_origin(np.concatenate(xs),
                                        np.concatenate(ys))[0])
            cu = load_curve(scale, held)
            bp = mpl_predict(p, cu)
            m0 = metrics(cu.loss, bp)["mae"]
            m1 = metrics(cu.loss, bp + kappa * feature(cu, spec))["mae"]
            deltas.append(100.0 * (m1 / m0 - 1.0))
            wins += int(m1 < m0)
    return float(np.mean(deltas)), wins


def main():
    subsets = [
        ("pool all 3 [ship]", PROBES),
        ("wsdcon_3 only", ["wsdcon_3.csv"]),
        ("wsdcon_9 only", ["wsdcon_9.csv"]),
        ("wsdcon_18 only", ["wsdcon_18.csv"]),
        ("wsdcon_3+9", ["wsdcon_3.csv", "wsdcon_9.csv"]),
    ]
    print(f"\n{'spec':14s} {'cal set':18s} {'probes-only':>12s} {'dilution':>12s}")
    for tag, spec in SPECS:
        for stag, subset in subsets:
            po, pw = probes_subset(spec, subset)
            di, dw = dilution_subset(spec, subset)
            print(f"{tag:14s} {stag:18s} {po:+8.1f}/{pw} {di:+8.1f}/{dw}")
        print()


if __name__ == "__main__":
    main()
