#!/usr/bin/env python3
"""End-to-end audit for the reusable next-generation kappa estimator."""
from __future__ import annotations

import csv
import itertools
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parent
ROOT = REPO.parent
sys.path.insert(0, str(REPO))

import current_law_continuous_kappa_search as base  # noqa: E402
from current_law_nextgen_kappa import NextGenKappaEstimator  # noqa: E402


OUT_DIR = ROOT / "results" / "current_law_nextgen_deployment_audit"
REFERENCE_DIR = ROOT / "results" / "current_law_nextgen_rho_margin_audit"
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


def summarize(details: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for group in ["main_matrix", "extra_holdout", "all"]:
        sub = [row for row in details if group == "all" or row["group"] == group]
        deltas = [float(row["delta_pct"]) for row in sub]
        rows.append(
            {
                "group": group,
                "tests": len(sub),
                "mean_delta_pct": float(np.mean(deltas)),
                "worst_delta_pct": float(max(deltas)),
                "non_harm_cells": int(sum(delta <= 1e-12 for delta in deltas)),
                "wins": int(sum(int(row["win"]) for row in sub)),
                "mean_target_factor": float(np.mean([float(row["target_factor"]) for row in sub])),
                "mean_kappa_safe": float(np.mean([float(row["kappa_safe"]) for row in sub])),
            }
        )
    return rows


def run() -> list[dict[str, object]]:
    estimator = NextGenKappaEstimator()
    labels = {curve: label for curve, label in [*base.CURVES, *EXTRA_CURVES]}
    all_curves = [*base.CURVES, *EXTRA_CURVES]
    details: list[dict[str, object]] = []
    main_curves = tuple(curve for curve, _ in base.CURVES)
    estimate_cache = {}
    for train_size in range(1, len(main_curves)):
        for train_curves in itertools.combinations(main_curves, train_size):
            train_id = "|".join(train_curves)
            train_label = " + ".join(labels[curve] for curve in train_curves)
            for scale in base.SCALES:
                for target_curve, target_label in all_curves:
                    if target_curve in train_curves:
                        continue
                    key = (scale, train_curves, target_curve)
                    if key not in estimate_cache:
                        estimate_cache[key] = estimator.estimate(scale, train_curves, target_curve)
                    estimate = estimate_cache[key]
                    scored = estimator.score(estimate)
                    details.append(
                        {
                            "scale": scale,
                            "train_id": train_id,
                            "train_label": train_label,
                            "train_size": train_size,
                            "test_curve": target_curve,
                            "test_label": target_label,
                            "group": "main_matrix" if target_curve in main_curves else "extra_holdout",
                            "selected_lambda": estimate.selected_lambda,
                            "tau": estimate.tau,
                            "dot_s": estimate.dot_s,
                            "l2_s": estimate.l2_s,
                            "full_l2_s": estimate.full_l2_s,
                            "pooled_retention": estimate.pooled_retention,
                            "raw_map": estimate.raw_map,
                            "kappa_pool": estimate.kappa_pool,
                            "shrink": estimate.shrink,
                            "rho": estimate.rho,
                            "kappa_transfer": estimate.kappa_transfer,
                            "target_retention": estimate.target_retention,
                            "target_factor": estimate.target_factor,
                            "kappa_safe": estimate.kappa_safe,
                            **scored,
                        }
                    )
    return details


def compare_to_reference(details: list[dict[str, object]]) -> list[dict[str, object]]:
    reference = [
        row
        for row in read_csv(REFERENCE_DIR / "details.csv")
        if abs(float(row["rho"]) - 0.5) < 1e-12
    ]
    by_key = {
        (row["scale"], row["train_id"], row["test_curve"]): row
        for row in reference
    }
    rows = []
    for row in details:
        ref = by_key[(row["scale"], row["train_id"], row["test_curve"])]
        rows.append(
            {
                "scale": row["scale"],
                "train_id": row["train_id"],
                "test_curve": row["test_curve"],
                "delta_abs_diff": abs(float(row["delta_pct"]) - float(ref["delta_pct"])),
                "kappa_abs_diff": abs(float(row["kappa_safe"]) - float(ref["kappa"]) * float(ref["target_factor"])),
                "target_retention_abs_diff": abs(float(row["target_retention"]) - float(ref["target_retention"])),
                "lambda_abs_diff": abs(float(row["selected_lambda"]) - float(ref["selected_lambda"])),
                "target_factor_diff": abs(float(row["target_factor"]) - float(ref["target_factor"])),
            }
        )
    return rows


def write_report(summary: list[dict[str, object]], diffs: list[dict[str, object]]) -> None:
    all_row = next(row for row in summary if row["group"] == "all")
    max_delta_diff = max(float(row["delta_abs_diff"]) for row in diffs)
    max_kappa_diff = max(float(row["kappa_abs_diff"]) for row in diffs)
    max_retention_diff = max(float(row["target_retention_abs_diff"]) for row in diffs)
    lines = [
        "# Next-Gen Deployment Estimator Audit\n\n",
        "This audit runs the reusable `NextGenKappaEstimator` end to end on the same all-train-size main-plus-extra matrix. "
        "It verifies that the deployed formula reproduces the established rho-margin audit while not relying on report-specific glue code.\n\n",
        "## Summary\n\n",
        "| group | mean delta | worst delta | non-harm cells | wins | target factor | mean kappa_safe |\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in summary:
        lines.append(
            f"| {row['group']} | {float(row['mean_delta_pct']):+.1f}% | "
            f"{float(row['worst_delta_pct']):+.1f}% | {int(row['non_harm_cells'])}/{int(row['tests'])} | "
            f"{int(row['wins'])}/{int(row['tests'])} | {float(row['mean_target_factor']):.3f} | "
            f"{float(row['mean_kappa_safe']):.5f} |\n"
        )
    lines += [
        "\n## Reference Agreement\n\n",
        "| quantity | max absolute difference |\n",
        "|---|---:|\n",
        f"| delta pct | `{max_delta_diff:.3e}` |\n",
        f"| kappa safe | `{max_kappa_diff:.3e}` |\n",
        f"| target retention | `{max_retention_diff:.3e}` |\n\n",
        "## Readout\n\n",
        f"The deployment estimator reproduces the safe audit with `{int(all_row['non_harm_cells'])}/{int(all_row['tests'])}` non-harming cells, "
        f"mean `{float(all_row['mean_delta_pct']):+.1f}%`, and worst `{float(all_row['worst_delta_pct']):+.1f}%`. "
        f"The maximum absolute delta difference from the rho-margin reference is `{max_delta_diff:.3e}`, and the maximum kappa difference is `{max_kappa_diff:.3e}`. "
        "This makes the formula implementation auditable as a reusable estimator rather than a collection of one-off analysis scripts.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    details = run()
    summary = summarize(details)
    diffs = compare_to_reference(details)
    write_csv(OUT_DIR / "details.csv", details)
    write_csv(OUT_DIR / "summary.csv", summary)
    write_csv(OUT_DIR / "reference_diffs.csv", diffs)
    write_report(summary, diffs)
    all_row = next(row for row in summary if row["group"] == "all")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    print(
        f"deployment mean={float(all_row['mean_delta_pct']):+.1f}% "
        f"worst={float(all_row['worst_delta_pct']):+.1f}% "
        f"nonharm={int(all_row['non_harm_cells'])}/{int(all_row['tests'])}"
    )


if __name__ == "__main__":
    main()
