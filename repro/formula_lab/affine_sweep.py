#!/usr/bin/env python3
"""Full-protocol sweep for the affine weight w = (1-rho) + rho*eta/eta_peak
(the NQM second-order chi form) at lam=10, plus pow d=0.25/0.5 for reference.
Protocols: in-sample R2, LOS, T1 (probe-linear + mpl-B), probes-only,
dilution, 6x6 matrix (final_no_cap).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import (  # noqa: E402
    load_curve, mpl_predict, metrics, MPL_PRECOMPUTED_INIT, SCALES,
)
from formula_lab import lab  # noqa: E402
from formula_lab.lab import feature, fit_origin, DECAY, PROBES  # noqa: E402
from formula_lab.matrix_protocol import run_matrix  # noqa: E402
from formula_lab.kappa_consistency import dilution  # noqa: E402
from formula_lab.probes_only import probes_only  # noqa: E402

VARIANTS = [
    ("lr@10 [paper]", {"form": "lr", "lam": 10}),
    ("pow d=0.25@10", {"form": "pow", "delta": 0.25, "lam": 10}),
    ("pow d=0.5@10", {"form": "pow", "delta": 0.5, "lam": 10}),
    ("affine r=0.5@10", {"form": "affine", "rho": 0.5, "lam": 10}),
    ("affine r=0.65@10", {"form": "affine", "rho": 0.65, "lam": 10}),
    ("affine r=0.75@10", {"form": "affine", "rho": 0.75, "lam": 10}),
    ("affine r=0.9@10", {"form": "affine", "rho": 0.9, "lam": 10}),
]


def main():
    rows = {}
    print(f"{'variant':18s} {'R2(100/400)':>12s} {'LOS':>7s} {'T1lin':>7s} {'T1-B':>7s} "
          f"{'probes':>7s} {'dilut':>7s} {'Mworst':>7s} {'Mmean':>7s} {'McosWsd':>8s} {'maxCosK':>8s}")
    for tag, spec in VARIANTS:
        r2 = lab.insample_r2(spec)
        los = lab.leave_one_sharp_protocol(spec)["delta_pct"]
        t1l = lab.table1_protocol(spec, "probe-linear")["delta_pct"]
        t1b = lab.table1_protocol(spec, "mpl-B")["delta_pct"]
        po, _, _ = probes_only(spec)
        dil = []
        for scale in SCALES:
            for m0, m1 in dilution(scale, spec):
                dil.append(100.0 * (m1 / m0 - 1.0))
        mat = run_matrix(spec)
        rows[tag] = {"r2": r2, "los": los, "t1_lin": t1l, "t1_B": t1b,
                     "probes_only": po, "dilution": float(np.mean(dil)), **{
                         k: mat[k] for k in ["worst_offdiag", "mean_offdiag",
                                             "cosine_to_wsd", "wsdcon9_to_wsd",
                                             "max_cosine_kappa"]}}
        print(f"{tag:18s} {r2['100']:.3f}/{r2['400']:.3f}  {los:+7.1f} {t1l:+7.1f} "
              f"{t1b:+7.1f} {po:+7.1f} {np.mean(dil):+7.1f} {mat['worst_offdiag']:+7.2f} "
              f"{mat['mean_offdiag']:+7.2f} {mat['cosine_to_wsd']:+8.2f} "
              f"{mat['max_cosine_kappa']:8.4f}")
    od = ROOT / "results" / "formula_lab"
    od.mkdir(parents=True, exist_ok=True)
    (od / "affine_sweep.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"wrote {od / 'affine_sweep.json'}")


if __name__ == "__main__":
    main()
