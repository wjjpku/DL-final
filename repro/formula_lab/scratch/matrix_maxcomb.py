#!/usr/bin/env python3
"""6x6 matrix for per-scale spec variants (maxcomb combination)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import current_law_amplitude_kappa_search as amp  # noqa: E402
import current_law_continuous_kappa_search as base  # noqa: E402
import current_law_eb_kappa_search as eb  # noqa: E402
import current_law_orthogonal_kappa_search as orth  # noqa: E402
from current_law_final_kappa import final_kappa  # noqa: E402
from formula_lab.lab import feature, probe_floor_powerlaw  # noqa: E402

P_MEAS = {s: probe_floor_powerlaw(s)[1] for s in base.SCALES}


def run_matrix_perscale(specs: dict) -> dict:
    feats = {(scale, curve): feature(base.load_curve(scale, curve), specs[scale])
             for scale in base.SCALES for curve, _ in base.CURVES}
    base_rows = []
    for curve, label in base.CURVES:
        for scale in base.SCALES:
            stats = amp.enriched_stats(scale, curve, feats)
            base_rows.append({"scale": scale, "train_curve": curve,
                              "train_label": label, **stats})
    orth_stats = {(scale, curve): orth.orthogonal_stats(scale, curve, feats, 2)
                  for curve, _ in base.CURVES for scale in base.SCALES}
    details = []
    for train_curve, _ in base.CURVES:
        pool = [r for r in base_rows if r["train_curve"] != train_curve]
        tau = eb.estimate_tau(pool, "q75")["tau"]
        for scale in base.SCALES:
            kappa = final_kappa(orth_stats[(scale, train_curve)], tau, cap=None)
            for test_curve, _ in base.CURVES:
                scored = base.score(scale, test_curve, kappa, feats)
                details.append({"train_curve": train_curve,
                                "test_curve": test_curve, **scored})
    summary = {}
    for tr, _ in base.CURVES:
        for te, _ in base.CURVES:
            sub = [float(r["delta_pct"]) for r in details
                   if r["train_curve"] == tr and r["test_curve"] == te]
            summary[(tr, te)] = float(np.mean(sub))
    off = [v for (tr, te), v in summary.items() if tr != te]
    return {"worst": float(np.max(off)), "mean": float(np.mean(off)),
            "cos_wsd": summary[("cosine_72000.csv", "wsd_20000_24000.csv")]}


def main():
    variants = {
        "maxcomb": {s: {"form": "pow",
                        "delta": max(0.25, max(P_MEAS[s] - 1.0, 0.0)),
                        "lam": 10} for s in base.SCALES},
    }
    for tag, specs in variants.items():
        r = run_matrix_perscale(specs)
        print(f"{tag}: worst {r['worst']:+.2f} mean {r['mean']:+.2f} "
              f"cos->wsd {r['cos_wsd']:+.2f}")


if __name__ == "__main__":
    main()
