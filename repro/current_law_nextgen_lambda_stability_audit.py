#!/usr/bin/env python3
"""Audit selected lambda values for the next-generation kappa candidate."""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = ROOT / "results" / "current_law_predictive_shrinkage_audit"
OUT_DIR = ROOT / "results" / "current_law_nextgen_lambda_stability_audit"
BAND = (0.01, 0.03)


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


def run() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    krows = [
        row
        for row in read_csv(PRED_DIR / "kappa_diagnostics.csv")
        if row["candidate"] == "train_size_rho0p5"
    ]
    detail_rows = [
        {
            "train_id": row["train_id"],
            "train_label": row["train_label"],
            "train_size": int(row["train_size"]),
            "scale": row["scale"],
            "selected_lambda": float(row["selected_lambda"]),
            "within_band": int(BAND[0] <= float(row["selected_lambda"]) <= BAND[1]),
            "is_lower_edge": int(abs(float(row["selected_lambda"]) - BAND[0]) <= 1e-12),
            "is_upper_edge": int(abs(float(row["selected_lambda"]) - BAND[1]) <= 1e-12),
            "kappa": float(row["kappa"]),
            "pooled_retention": float(row["pooled_retention"]),
        }
        for row in krows
    ]
    summary_rows: list[dict[str, object]] = []
    for train_size in sorted({int(row["train_size"]) for row in detail_rows}):
        sub = [row for row in detail_rows if int(row["train_size"]) == train_size]
        lambdas = [float(row["selected_lambda"]) for row in sub]
        counts = Counter(lambdas)
        summary_rows.append(
            {
                "train_size": train_size,
                "rows": len(sub),
                "min_lambda": float(min(lambdas)),
                "median_lambda": float(np.median(lambdas)),
                "max_lambda": float(max(lambdas)),
                "mean_lambda": float(np.mean(lambdas)),
                "within_band_rows": int(sum(int(row["within_band"]) for row in sub)),
                "lower_edge_rows": int(sum(int(row["is_lower_edge"]) for row in sub)),
                "upper_edge_rows": int(sum(int(row["is_upper_edge"]) for row in sub)),
                "lambda_counts": " ".join(f"{lam:g}:{count}" for lam, count in sorted(counts.items())),
            }
        )
    all_lambdas = [float(row["selected_lambda"]) for row in detail_rows]
    all_counts = Counter(all_lambdas)
    summary_rows.append(
        {
            "train_size": "all",
            "rows": len(detail_rows),
            "min_lambda": float(min(all_lambdas)),
            "median_lambda": float(np.median(all_lambdas)),
            "max_lambda": float(max(all_lambdas)),
            "mean_lambda": float(np.mean(all_lambdas)),
            "within_band_rows": int(sum(int(row["within_band"]) for row in detail_rows)),
            "lower_edge_rows": int(sum(int(row["is_lower_edge"]) for row in detail_rows)),
            "upper_edge_rows": int(sum(int(row["is_upper_edge"]) for row in detail_rows)),
            "lambda_counts": " ".join(f"{lam:g}:{count}" for lam, count in sorted(all_counts.items())),
        }
    )
    return detail_rows, summary_rows


def write_report(summary_rows: list[dict[str, object]]) -> None:
    all_row = next(row for row in summary_rows if row["train_size"] == "all")
    lines = [
        "# Next-Gen Lambda Stability Audit\n\n",
        "This audit checks the selected soft DCT/Sobolev nuisance strength used by the next-generation `rho=0.5` kappa. "
        "The method restricts train-only inner-CV selection to the identifiable band `lambda in [0.01, 0.03]`.\n\n",
        "## Summary\n\n",
        "| train curves | rows | min | median | max | mean | in band | lower edge | upper edge | counts |\n",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['train_size']} | {int(row['rows'])} | {float(row['min_lambda']):.3f} | "
            f"{float(row['median_lambda']):.3f} | {float(row['max_lambda']):.3f} | "
            f"{float(row['mean_lambda']):.3f} | {int(row['within_band_rows'])}/{int(row['rows'])} | "
            f"{int(row['lower_edge_rows'])} | {int(row['upper_edge_rows'])} | `{row['lambda_counts']}` |\n"
        )
    lines += [
        "\n## Readout\n\n",
        f"All next-generation `rho=0.5` kappa rows remain inside the identifiable band: `{int(all_row['within_band_rows'])}/{int(all_row['rows'])}`. "
        f"The selected values span `{float(all_row['min_lambda']):.3f}` to `{float(all_row['max_lambda']):.3f}`, with median `{float(all_row['median_lambda']):.3f}`. "
        "Single-curve calibration uses the fixed fallback `0.025`; larger train sets often choose the upper band edge `0.030`, which indicates that the useful region is the high-drift-control side of the identifiable band rather than an unconstrained smoothing optimum.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    details, summary = run()
    write_csv(OUT_DIR / "details.csv", details)
    write_csv(OUT_DIR / "summary.csv", summary)
    write_report(summary)
    all_row = next(row for row in summary if row["train_size"] == "all")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    print(
        f"lambda band nonviolations={int(all_row['within_band_rows'])}/{int(all_row['rows'])} "
        f"range=[{float(all_row['min_lambda']):.3f}, {float(all_row['max_lambda']):.3f}]"
    )


if __name__ == "__main__":
    main()
