#!/usr/bin/env python3
"""Probes-only calibration: kappa from origin-LS on the 3 wsdcon probes pooled
(per scale), evaluated on wsd+wsdld.  Paper baseline (lr@10): about -17.6%.
Also: Table-1 chain with the mpl-B amplitude for the pow form.
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
from formula_lab import lab  # noqa: E402
from formula_lab.lab import feature, fit_origin, DECAY, PROBES  # noqa: E402

VARIANTS = [
    ("lr@10 [paper]", {"form": "lr", "lam": 10}),
    ("lr@5", {"form": "lr", "lam": 5}),
    ("pow d=0.25@10", {"form": "pow", "delta": 0.25, "lam": 10}),
    ("pow d=0.5@10", {"form": "pow", "delta": 0.5, "lam": 10}),
    ("pow d=0.5@7", {"form": "pow", "delta": 0.5, "lam": 7}),
    ("pow d=0.5@5", {"form": "pow", "delta": 0.5, "lam": 5}),
    ("pow d=0.75@10", {"form": "pow", "delta": 0.75, "lam": 10}),
    ("pow d=0.75@5", {"form": "pow", "delta": 0.75, "lam": 5}),
    ("affine rho=0.5@10", {"form": "affine", "rho": 0.5, "lam": 10}),
    ("affine rho=0.75@5", {"form": "affine", "rho": 0.75, "lam": 5}),
]


def probes_only(spec: dict):
    deltas, wins, rows = [], 0, []
    for scale in SCALES:
        p = MPL_PRECOMPUTED_INIT[scale]
        xs, ys = [], []
        for n in PROBES:
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
            rows.append({"scale": scale, "curve": n, "kappa": kappa,
                         "delta_pct": deltas[-1]})
    return float(np.mean(deltas)), wins, rows


def main():
    print("== probes-only calibration -> sharp decays ==")
    print(f"{'variant':22s} {'mean dMAE%':>10s} {'wins':>5s}")
    for tag, spec in VARIANTS:
        d, w, _ = probes_only(spec)
        print(f"{tag:22s} {d:+10.1f} {w:>4d}/6")

    print("\n== Table-1 chain, mpl-B amplitude ==")
    for tag, spec in [("lr@10 / mpl-B", {"form": "lr", "lam": 10}),
                      ("pow d=0.5@10 / mpl-B", {"form": "pow", "delta": 0.5, "lam": 10}),
                      ("pow d=0.5@5 / mpl-B", {"form": "pow", "delta": 0.5, "lam": 5})]:
        t1 = lab.table1_protocol(spec, "mpl-B")
        rcv = np.array(list(t1["ratios"].values()))
        print(f"{tag:22s} T1 {t1['delta_pct']:+6.1f}% {t1['wins']}/6 "
              f"cCV={rcv.std()/abs(rcv.mean())*100:.0f}%")


if __name__ == "__main__":
    main()
