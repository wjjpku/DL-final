#!/usr/bin/env python3
"""Component ablation for the next-generation kappa formula.

This audit scores three variants on the same main-plus-extra, all-train-size
matrix:

* soft spectral transfer without posterior-predictive shrinkage,
* the same estimator with rho=0.5 shrinkage, and
* rho=0.5 plus the target-identifiability gate R_target >= 0.01.
"""
from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parent
ROOT = REPO.parent
sys.path.insert(0, str(REPO))

import current_law_continuous_kappa_search as base  # noqa: E402
import current_law_soft_spectral_multicurve_selection_audit as spectral  # noqa: E402
from deep_stime import stime_feature  # noqa: E402


PRED_DIR = ROOT / "results" / "current_law_predictive_shrinkage_audit"
OUT_DIR = ROOT / "results" / "current_law_nextgen_component_ablation_audit"
RETENTION_THRESHOLD = 0.01
EXTRA_CURVES = [
    ("cosine_24000.csv", "Cosine 24k"),
    ("constant_24000.csv", "Constant 24k"),
    ("constant_72000.csv", "Constant 72k"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def target_stats(scale: str, curve_name: str, lam: float) -> dict[str, object]:
    curve = base.load_curve(scale, curve_name)
    phi = stime_feature(curve, base.LAMBDA)
    baseline = base.mpl_predict(base.MPL_PRECOMPUTED_INIT[scale], curve)
    q = spectral.dct_basis(len(curve.step), spectral.MAX_MODE)
    a = spectral.smoother_matrix(q, lam)
    phi_o = spectral.soft_residualize(phi, q, a)
    phi_l2 = float(np.dot(phi, phi))
    retention = 0.0 if phi_l2 <= 1e-18 else float(np.dot(phi_o, phi_o) / phi_l2)
    return {
        "curve": curve,
        "phi": phi,
        "baseline": baseline,
        "base_mae": base.metrics(curve.loss, baseline)["mae"],
        "target_retention": retention,
    }


def score(stats: dict[str, object], kappa: float) -> dict[str, object]:
    curve = stats["curve"]
    pred = stats["baseline"] + kappa * stats["phi"]
    corr_mae = base.metrics(curve.loss, pred)["mae"]
    base_mae = float(stats["base_mae"])
    return {
        "base_mae": base_mae,
        "corr_mae": corr_mae,
        "delta_pct": 100.0 * (corr_mae / base_mae - 1.0),
        "win": int(corr_mae < base_mae),
    }


def summarize(details: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for mode in ["no_predictive_shrinkage", "rho0p5_shrinkage", "rho0p5_plus_Rtarget_gate"]:
        for group in ["main_matrix", "extra_holdout", "all"]:
            sub = [row for row in details if row["mode"] == mode and (group == "all" or row["group"] == group)]
            deltas = [float(row["delta_pct"]) for row in sub]
            rows.append(
                {
                    "mode": mode,
                    "group": group,
                    "tests": len(sub),
                    "mean_delta_pct": float(np.mean(deltas)),
                    "worst_delta_pct": float(max(deltas)),
                    "non_harm_cells": int(sum(delta <= 1e-12 for delta in deltas)),
                    "wins": int(sum(int(row["win"]) for row in sub)),
                    "mean_target_factor": float(np.mean([float(row["target_factor"]) for row in sub])),
                }
            )
    return rows


def summarize_by_train_size(details: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for mode in ["no_predictive_shrinkage", "rho0p5_shrinkage", "rho0p5_plus_Rtarget_gate"]:
        for train_size in sorted({int(row["train_size"]) for row in details}):
            sub = [row for row in details if row["mode"] == mode and int(row["train_size"]) == train_size]
            deltas = [float(row["delta_pct"]) for row in sub]
            rows.append(
                {
                    "mode": mode,
                    "train_size": train_size,
                    "tests": len(sub),
                    "mean_delta_pct": float(np.mean(deltas)),
                    "worst_delta_pct": float(max(deltas)),
                    "non_harm_cells": int(sum(delta <= 1e-12 for delta in deltas)),
                    "wins": int(sum(int(row["win"]) for row in sub)),
                    "mean_target_factor": float(np.mean([float(row["target_factor"]) for row in sub])),
                }
            )
    return rows


def run() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    krows = [
        row
        for row in read_csv(PRED_DIR / "kappa_diagnostics.csv")
        if row["candidate"] in {"none", "train_size_rho0p5"}
    ]
    kappa_by_key = {
        (row["candidate"], row["train_id"], row["train_size"], row["scale"]): row
        for row in krows
    }
    all_curves = [*base.CURVES, *EXTRA_CURVES]
    target_keys = sorted(
        {
            (row["scale"], curve, float(row["selected_lambda"]))
            for row in krows
            for curve, _ in all_curves
        }
    )
    cached = {(scale, curve, lam): target_stats(scale, curve, lam) for scale, curve, lam in target_keys}
    details = []
    for key, none_row in sorted(kappa_by_key.items()):
        candidate, train_id, train_size, scale = key
        if candidate != "none":
            continue
        shrink_row = kappa_by_key[("train_size_rho0p5", train_id, train_size, scale)]
        train_curves = set(train_id.split("|"))
        lam = float(shrink_row["selected_lambda"])
        variants = [
            ("no_predictive_shrinkage", float(none_row["kappa"]), 1.0),
            ("rho0p5_shrinkage", float(shrink_row["kappa"]), 1.0),
            ("rho0p5_plus_Rtarget_gate", float(shrink_row["kappa"]), None),
        ]
        for test_curve, test_label in all_curves:
            if test_curve in train_curves:
                continue
            stats = cached[(scale, test_curve, lam)]
            group = "main_matrix" if test_curve in {curve for curve, _ in base.CURVES} else "extra_holdout"
            for mode, kappa, fixed_factor in variants:
                factor = fixed_factor
                if factor is None:
                    factor = 1.0 if float(stats["target_retention"]) >= RETENTION_THRESHOLD else 0.0
                scored = score(stats, kappa * float(factor))
                details.append(
                    {
                        "mode": mode,
                        "group": group,
                        "scale": scale,
                        "train_id": train_id,
                        "train_label": shrink_row["train_label"],
                        "train_size": int(train_size),
                        "test_curve": test_curve,
                        "test_label": test_label,
                        "selected_lambda": lam,
                        "kappa": kappa,
                        "target_factor": float(factor),
                        "target_retention": float(stats["target_retention"]),
                        **scored,
                    }
                )
    return details, summarize(details), summarize_by_train_size(details)


def write_report(summary: list[dict[str, object]], train_size_summary: list[dict[str, object]]) -> None:
    by_key = {(row["mode"], row["group"]): row for row in summary}
    raw_all = by_key[("no_predictive_shrinkage", "all")]
    shrink_all = by_key[("rho0p5_shrinkage", "all")]
    gated_all = by_key[("rho0p5_plus_Rtarget_gate", "all")]
    lines = [
        "# Next-Gen Component Ablation Audit\n\n",
        "This audit isolates the two stabilizers in the next-generation formula: posterior-predictive shrinkage and the target-identifiability gate.\n\n",
        "## Summary\n\n",
        "| mode | group | mean delta | worst delta | non-harm cells | wins | target factor |\n",
        "|---|---|---:|---:|---:|---:|---:|\n",
    ]
    for row in summary:
        lines.append(
            f"| `{row['mode']}` | {row['group']} | {float(row['mean_delta_pct']):+.1f}% | "
            f"{float(row['worst_delta_pct']):+.1f}% | {int(row['non_harm_cells'])}/{int(row['tests'])} | "
            f"{int(row['wins'])}/{int(row['tests'])} | {float(row['mean_target_factor']):.3f} |\n"
        )
    lines += [
        "\n## Train-Size Breakdown\n\n",
        "| mode | train curves | mean delta | worst delta | non-harm cells | wins | target factor |\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in train_size_summary:
        lines.append(
            f"| `{row['mode']}` | {int(row['train_size'])} | {float(row['mean_delta_pct']):+.1f}% | "
            f"{float(row['worst_delta_pct']):+.1f}% | {int(row['non_harm_cells'])}/{int(row['tests'])} | "
            f"{int(row['wins'])}/{int(row['tests'])} | {float(row['mean_target_factor']):.3f} |\n"
        )
    lines += [
        "\n## Readout\n\n",
        f"Without posterior-predictive shrinkage, the next-gen direction has mean `{float(raw_all['mean_delta_pct']):+.1f}%` but worst `{float(raw_all['worst_delta_pct']):+.1f}%`. "
        f"Adding `rho=0.5` shrinkage improves the worst case to `{float(shrink_all['worst_delta_pct']):+.1f}%` on the combined audit but still leaves diffuse-target failures. "
        f"Adding the `R_target >= 0.01` gate gives `{int(gated_all['non_harm_cells'])}/{int(gated_all['tests'])}` non-harming cells with worst `{float(gated_all['worst_delta_pct']):+.1f}%`. "
        "Thus shrinkage controls finite-calibration amplitude over-transfer, while the target gate controls non-identifiable target directions.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    details, summary, train_size_summary = run()
    write_csv(OUT_DIR / "details.csv", details)
    write_csv(OUT_DIR / "summary.csv", summary)
    write_csv(OUT_DIR / "train_size_summary.csv", train_size_summary)
    write_report(summary, train_size_summary)
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    for row in summary:
        if row["group"] == "all":
            print(
                f"{row['mode']:28s} mean={float(row['mean_delta_pct']):+6.1f}% "
                f"worst={float(row['worst_delta_pct']):+6.1f}% "
                f"nonharm={int(row['non_harm_cells'])}/{int(row['tests'])}"
            )


if __name__ == "__main__":
    main()
