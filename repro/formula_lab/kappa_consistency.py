#!/usr/bin/env python3
"""Cross-family kappa consistency: does the eta-weighted / floor-drop form make
the per-curve response amplitude UNIVERSAL across schedule families?

For each variant and scale: origin-LS kappa per curve (cosine_72000, wsd,
wsdld, wsdcon_3/9/18), report kappa CV across curves and the probe/sharp
ratio.  Then replicate the more-data-calibration dilution test: calibrate on
[other sharp + 3 probes] -> test held-out sharp (paper: -49% sharp-only
dilutes to -19.5% when probes are added, fixed lam=10).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import (  # noqa: E402
    load_curve, mpl_predict, metrics, MPL_PRECOMPUTED_INIT, SCALES,
)
from formula_lab.lab import feature, fit_origin, DECAY, PROBES  # noqa: E402

CURVES = ["cosine_72000.csv"] + DECAY + PROBES

VARIANTS = [
    ("lr@10 [paper]", {"form": "lr", "lam": 10}),
    ("pow d=0.5@10", {"form": "pow", "delta": 0.5, "lam": 10}),
    ("pow d=0.5@5", {"form": "pow", "delta": 0.5, "lam": 5}),
    ("floor p=1.5@10", {"form": "floor", "p": 1.5, "lam": 10}),
    ("floor p=1.5@5", {"form": "floor", "p": 1.5, "lam": 5}),
    ("floor p=1.25@10", {"form": "floor", "p": 1.25, "lam": 10}),
    ("floor p=2.0@10", {"form": "floor", "p": 2.0, "lam": 10}),
    ("affine rho=0.5@10", {"form": "affine", "rho": 0.5, "lam": 10}),
]


def per_curve_kappa(scale: str, spec: dict) -> dict[str, float]:
    p = MPL_PRECOMPUTED_INIT[scale]
    out = {}
    for n in CURVES:
        c = load_curve(scale, n)
        out[n] = max(0.0, fit_origin(feature(c, spec), c.loss - mpl_predict(p, c))[0])
    return out


def dilution(scale: str, spec: dict) -> dict:
    """Calibrate kappa on pooled [other sharp + 3 probes]; test held-out sharp."""
    p = MPL_PRECOMPUTED_INIT[scale]
    rows = []
    for held in DECAY:
        cal = [n for n in DECAY if n != held] + PROBES
        xs, ys = [], []
        for n in cal:
            c = load_curve(scale, n)
            xs.append(feature(c, spec))
            ys.append(c.loss - mpl_predict(p, c))
        kappa = max(0.0, fit_origin(np.concatenate(xs), np.concatenate(ys))[0])
        cu = load_curve(scale, held)
        base = mpl_predict(p, cu)
        m0 = metrics(cu.loss, base)["mae"]
        m1 = metrics(cu.loss, base + kappa * feature(cu, spec))["mae"]
        rows.append((m0, m1))
    return rows


def main():
    print("== per-curve kappa (origin LS) and cross-family consistency ==")
    for tag, spec in VARIANTS:
        cvs, ratios = [], []
        per_scale = {}
        for scale in SCALES:
            k = per_curve_kappa(scale, spec)
            vals = np.array([k[n] for n in CURVES])
            active = vals[vals > 0]
            cv = float(vals.std() / vals.mean() * 100) if vals.mean() > 0 else float("nan")
            k_sharp = np.mean([k[n] for n in DECAY])
            k_probe = np.mean([k[n] for n in PROBES])
            ratio = k_probe / k_sharp if k_sharp > 0 else float("nan")
            cvs.append(cv); ratios.append(ratio)
            per_scale[scale] = k
        print(f"{tag:20s} kappaCV[{cvs[0]:5.0f}% {cvs[1]:5.0f}% {cvs[2]:5.0f}%]  "
              f"probe/sharp[{ratios[0]:5.2f} {ratios[1]:5.2f} {ratios[2]:5.2f}]")
        if tag.startswith("lr@10") or tag.startswith("floor p=1.5@10"):
            for scale in SCALES:
                k = per_scale[scale]
                detail = " ".join(f"{n.split('.')[0].replace('_20000_24000',''):>10s}={k[n]:.4f}"
                                  for n in CURVES)
                print(f"    {scale:>4}M {detail}")

    print("\n== dilution test: cal=[other sharp + probes] -> held-out sharp ==")
    print(f"{'variant':20s} {'mean dMAE%':>10s} {'wins':>5s}   (paper lr: -19.5%; sharp-only: -49%)")
    for tag, spec in VARIANTS:
        deltas, wins = [], 0
        for scale in SCALES:
            for m0, m1 in dilution(scale, spec):
                deltas.append(100.0 * (m1 / m0 - 1.0))
                wins += int(m1 < m0)
        print(f"{tag:20s} {np.mean(deltas):+10.1f} {wins:>4d}/6")


if __name__ == "__main__":
    main()
