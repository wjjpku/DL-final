#!/usr/bin/env python3
"""6x6 transfer matrix with the paper's final_no_cap estimator, for arbitrary
feature specs.  Baseline to beat (lr@10): worst offdiag -2.7%, median -10.0%,
mean -12.1%, cosine->WSD -4.3%, wsdcon_9->WSD -16.0%, max cosine kappa 0.0089.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
sys.path.insert(0, str(REPO))

import current_law_amplitude_kappa_search as amp  # noqa: E402
import current_law_continuous_kappa_search as base  # noqa: E402
import current_law_eb_kappa_search as eb  # noqa: E402
import current_law_orthogonal_kappa_search as orth  # noqa: E402
from current_law_final_kappa import final_kappa  # noqa: E402
from formula_lab.lab import feature  # noqa: E402


def feats_for(spec: dict):
    return {(scale, curve): feature(base.load_curve(scale, curve), spec)
            for scale in base.SCALES for curve, _ in base.CURVES}


def run_matrix(spec: dict) -> dict:
    feats = feats_for(spec)
    base_rows = []
    for curve, label in base.CURVES:
        for scale in base.SCALES:
            stats = amp.enriched_stats(scale, curve, feats)
            base_rows.append({"scale": scale, "train_curve": curve,
                              "train_label": label, **stats})
    orth_stats = {
        (scale, curve): orth.orthogonal_stats(scale, curve, feats, 2)
        for curve, _ in base.CURVES for scale in base.SCALES
    }
    details = []
    kappas = []
    for train_curve, train_label in base.CURVES:
        pool = [r for r in base_rows if r["train_curve"] != train_curve]
        tau = eb.estimate_tau(pool, "q75")["tau"]
        for scale in base.SCALES:
            kappa = final_kappa(orth_stats[(scale, train_curve)], tau, cap=None)
            kappas.append({"train_curve": train_curve, "scale": scale, "kappa": kappa})
            for test_curve, test_label in base.CURVES:
                scored = base.score(scale, test_curve, kappa, feats)
                details.append({"train_curve": train_curve, "test_curve": test_curve,
                                "kappa": kappa, **scored})
    # summarize: per (train,test) mean of per-scale delta_pct (final_kappa convention)
    summary = {}
    for train_curve, _ in base.CURVES:
        for test_curve, _ in base.CURVES:
            sub = [r for r in details if r["train_curve"] == train_curve
                   and r["test_curve"] == test_curve]
            summary[(train_curve, test_curve)] = float(
                np.mean([float(r["delta_pct"]) for r in sub]))
    off = [v for (tr, te), v in summary.items() if tr != te]
    cos_kappas = [r["kappa"] for r in kappas if r["train_curve"] == "cosine_72000.csv"]
    return {
        "worst_offdiag": float(np.max(off)),
        "median_offdiag": float(np.median(off)),
        "mean_offdiag": float(np.mean(off)),
        "cosine_to_wsd": summary[("cosine_72000.csv", "wsd_20000_24000.csv")],
        "wsdcon9_to_wsd": summary[("wsdcon_9.csv", "wsd_20000_24000.csv")],
        "max_cosine_kappa": float(np.max(cos_kappas)),
        "summary": {f"{tr}->{te}": v for (tr, te), v in summary.items()},
    }


VARIANTS = [
    ("lr@10 [paper]", {"form": "lr", "lam": 10}),
    ("pow d=0.5@10", {"form": "pow", "delta": 0.5, "lam": 10}),
    ("pow d=0.5@7", {"form": "pow", "delta": 0.5, "lam": 7}),
    ("pow d=0.5@5", {"form": "pow", "delta": 0.5, "lam": 5}),
    ("pow d=0.25@10", {"form": "pow", "delta": 0.25, "lam": 10}),
    ("affine rho=0.5@10", {"form": "affine", "rho": 0.5, "lam": 10}),
]


def main():
    out = {}
    print(f"{'variant':20s} {'worst':>7s} {'median':>7s} {'mean':>7s} "
          f"{'cos->wsd':>9s} {'w9->wsd':>9s} {'maxCosK':>8s}")
    for tag, spec in VARIANTS:
        r = run_matrix(spec)
        out[tag] = r
        print(f"{tag:20s} {r['worst_offdiag']:+7.2f} {r['median_offdiag']:+7.2f} "
              f"{r['mean_offdiag']:+7.2f} {r['cosine_to_wsd']:+9.2f} "
              f"{r['wsdcon9_to_wsd']:+9.2f} {r['max_cosine_kappa']:8.4f}")
    od = ROOT / "results" / "formula_lab"
    od.mkdir(parents=True, exist_ok=True)
    (od / "matrix_protocol.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {od / 'matrix_protocol.json'}")


if __name__ == "__main__":
    main()
