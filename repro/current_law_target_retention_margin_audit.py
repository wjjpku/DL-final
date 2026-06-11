#!/usr/bin/env python3
"""Margin audit for the next-generation target-retention gate.

The target-identifiability gate uses

    R_target(lambda) = ||M_lambda phi_target||^2 / ||phi_target||^2

and abstains when R_target < threshold. This script asks whether the chosen
threshold 0.01 is a knife-edge or lies inside a stable empirical margin.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = ROOT / "results" / "current_law_target_identifiability_audit"
OUT_DIR = ROOT / "results" / "current_law_target_retention_margin_audit"
CHOSEN_THRESHOLD = 0.01


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


def gated_delta(raw_delta: float, retention: float, threshold: float) -> tuple[float, int, float]:
    factor = 1.0 if retention >= threshold else 0.0
    delta = raw_delta if factor else 0.0
    return delta, int(delta < 0.0), factor


def summarize_threshold(raw_rows: list[dict[str, str]], threshold: float) -> dict[str, object]:
    details = []
    for row in raw_rows:
        delta, win, factor = gated_delta(
            float(row["delta_pct"]),
            float(row["target_retention"]),
            threshold,
        )
        details.append(
            {
                "group": row["group"],
                "delta_pct": delta,
                "win": win,
                "factor": factor,
            }
        )
    out: dict[str, object] = {"threshold": threshold}
    for group in ["main_matrix", "extra_holdout", "all"]:
        sub = [row for row in details if group == "all" or row["group"] == group]
        out[f"{group}_tests"] = len(sub)
        out[f"{group}_mean_delta_pct"] = float(np.mean([row["delta_pct"] for row in sub]))
        out[f"{group}_worst_delta_pct"] = float(max(row["delta_pct"] for row in sub))
        out[f"{group}_non_harm_cells"] = int(sum(row["delta_pct"] <= 1e-12 for row in sub))
        out[f"{group}_wins"] = int(sum(row["win"] for row in sub))
        out[f"{group}_mean_factor"] = float(np.mean([row["factor"] for row in sub]))
    return out


def curve_retention_summary(raw_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows = []
    keys = sorted({(row["test_curve"], row["test_label"], row["group"]) for row in raw_rows})
    for curve, label, group in keys:
        sub = [row for row in raw_rows if row["test_curve"] == curve]
        retentions = [float(row["target_retention"]) for row in sub]
        raw_deltas = [float(row["delta_pct"]) for row in sub]
        rows.append(
            {
                "test_curve": curve,
                "test_label": label,
                "group": group,
                "tests": len(sub),
                "min_retention": float(min(retentions)),
                "median_retention": float(np.median(retentions)),
                "max_retention": float(max(retentions)),
                "raw_mean_delta_pct": float(np.mean(raw_deltas)),
                "raw_worst_delta_pct": float(max(raw_deltas)),
                "raw_harm_cells": int(sum(delta > 1e-12 for delta in raw_deltas)),
            }
        )
    return rows


def margin_summary(raw_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    harmful = [row for row in raw_rows if float(row["delta_pct"]) > 1e-12]
    main = [row for row in raw_rows if row["group"] == "main_matrix"]
    main_wins = [row for row in main if float(row["delta_pct"]) < 0.0]
    max_harmful_retention = max(float(row["target_retention"]) for row in harmful)
    min_main_retention = min(float(row["target_retention"]) for row in main)
    min_main_win_retention = min(float(row["target_retention"]) for row in main_wins)
    midpoint = float((max_harmful_retention * min_main_retention) ** 0.5)
    return [
        {
            "chosen_threshold": CHOSEN_THRESHOLD,
            "max_raw_harmful_retention": max_harmful_retention,
            "min_main_matrix_retention": min_main_retention,
            "min_main_win_retention": min_main_win_retention,
            "geometric_midpoint": midpoint,
            "chosen_over_harmful_max": CHOSEN_THRESHOLD / max_harmful_retention,
            "main_min_over_chosen": min_main_retention / CHOSEN_THRESHOLD,
            "log10_margin_width": float(np.log10(min_main_retention / max_harmful_retention)),
            "harmful_cells": len(harmful),
            "main_cells": len(main),
        }
    ]


def threshold_grid(raw_rows: list[dict[str, str]], margins: dict[str, object]) -> list[float]:
    unique_ret = sorted({float(row["target_retention"]) for row in raw_rows if float(row["target_retention"]) > 0})
    anchors = [
        0.0,
        0.0025,
        0.005,
        float(margins["max_raw_harmful_retention"]),
        float(margins["geometric_midpoint"]),
        0.0075,
        CHOSEN_THRESHOLD,
        float(margins["min_main_matrix_retention"]),
        0.015,
        0.02,
        0.05,
        0.1,
        0.2,
    ]
    eps = 1e-9
    near_events = []
    for value in unique_ret:
        near_events.extend([max(0.0, value - eps), value + eps])
    values = sorted({round(value, 12) for value in [*anchors, *near_events] if value >= 0.0})
    return values


def write_report(
    margins: dict[str, object],
    curve_rows: list[dict[str, object]],
    threshold_rows: list[dict[str, object]],
) -> None:
    chosen = next(row for row in threshold_rows if abs(float(row["threshold"]) - CHOSEN_THRESHOLD) < 1e-12)
    plateau = [
        row
        for row in threshold_rows
        if int(row["all_non_harm_cells"]) == int(row["all_tests"])
        and int(row["main_matrix_wins"]) == 558
    ]
    plateau_min = min(float(row["threshold"]) for row in plateau)
    plateau_max = max(float(row["threshold"]) for row in plateau)
    lines = [
        "# Target-Retention Margin Audit\n\n",
        "This audit checks whether the target-identifiability threshold `R_target(lambda) >= 0.01` is a knife-edge. "
        "It reuses the raw next-generation predictions from the target-identifiability audit and rescans binary retention gates without refitting kappa.\n\n",
        "## Margin\n\n",
        "| quantity | value |\n",
        "|---|---:|\n",
        f"| max raw-harmful target retention | `{float(margins['max_raw_harmful_retention']):.6f}` |\n",
        f"| chosen threshold | `{CHOSEN_THRESHOLD:.6f}` |\n",
        f"| min main-matrix target retention | `{float(margins['min_main_matrix_retention']):.6f}` |\n",
        f"| geometric midpoint | `{float(margins['geometric_midpoint']):.6f}` |\n",
        f"| chosen / harmful max | `{float(margins['chosen_over_harmful_max']):.2f}x` |\n",
        f"| main min / chosen | `{float(margins['main_min_over_chosen']):.2f}x` |\n\n",
        "For a binary gate that admits targets with `R_target >= threshold`, every threshold strictly above the harmful maximum and no larger than the main-matrix minimum blocks the observed harmful diffuse targets while retaining the full main-matrix target set. "
        f"The chosen `0.01` lies inside this interval, not at either boundary.\n\n",
        "## Threshold Sweep\n\n",
        "| threshold | all mean | all worst | all non-harm | main wins | admitted target factor |\n",
        "|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in threshold_rows:
        threshold = float(row["threshold"])
        if threshold not in {0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.05, 0.1, 0.2}:
            continue
        lines.append(
            f"| `{threshold:.4f}` | {float(row['all_mean_delta_pct']):+.1f}% | "
            f"{float(row['all_worst_delta_pct']):+.1f}% | "
            f"{int(row['all_non_harm_cells'])}/{int(row['all_tests'])} | "
            f"{int(row['main_matrix_wins'])}/{int(row['main_matrix_tests'])} | "
            f"{float(row['all_mean_factor']):.3f} |\n"
        )
    lines += [
        "\n## Curve Retention Summary\n\n",
        "| curve | group | min R | max R | raw worst | raw harm cells |\n",
        "|---|---|---:|---:|---:|---:|\n",
    ]
    for row in curve_rows:
        lines.append(
            f"| `{row['test_label']}` | {row['group']} | {float(row['min_retention']):.6f} | "
            f"{float(row['max_retention']):.6f} | {float(row['raw_worst_delta_pct']):+.1f}% | "
            f"{int(row['raw_harm_cells'])}/{int(row['tests'])} |\n"
        )
    lines += [
        "\n## Readout\n\n",
        f"The full-transfer plateau with `1116/1116` non-harming cells and all `558/558` main-matrix wins spans approximately `{plateau_min:.6f}` to `{plateau_max:.6f}` in this event-grid scan. "
        f"At the chosen threshold, the combined audit has mean `{float(chosen['all_mean_delta_pct']):+.1f}%`, worst `{float(chosen['all_worst_delta_pct']):+.1f}%`, and `{int(chosen['all_non_harm_cells'])}/{int(chosen['all_tests'])}` non-harming cells. "
        "Lowering the threshold to `0.005` admits the diffuse cosine target and restores the `+22.5%` failure; increasing it beyond the main cosine retention remains safe but starts dropping useful main-matrix transfers. "
        "Thus `0.01` is best read as a margin-based identifiability floor, not a tuned loss-optimal threshold.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    raw_rows = [
        row
        for row in read_csv(TARGET_DIR / "details.csv")
        if row["mode"] == "raw_nextgen"
    ]
    curves = curve_retention_summary(raw_rows)
    margins = margin_summary(raw_rows)[0]
    thresholds = [summarize_threshold(raw_rows, t) for t in threshold_grid(raw_rows, margins)]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "curve_retention_summary.csv", curves)
    write_csv(OUT_DIR / "margin_summary.csv", [margins])
    write_csv(OUT_DIR / "threshold_sweep.csv", thresholds)
    write_report(margins, curves, thresholds)
    chosen = next(row for row in thresholds if abs(float(row["threshold"]) - CHOSEN_THRESHOLD) < 1e-12)
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    print(
        f"threshold={CHOSEN_THRESHOLD:.3f} mean={float(chosen['all_mean_delta_pct']):+.1f}% "
        f"worst={float(chosen['all_worst_delta_pct']):+.1f}% "
        f"nonharm={int(chosen['all_non_harm_cells'])}/{int(chosen['all_tests'])}"
    )
    print(
        "margin "
        f"{float(margins['max_raw_harmful_retention']):.6f} < {CHOSEN_THRESHOLD:.6f} "
        f"< {float(margins['min_main_matrix_retention']):.6f}"
    )


if __name__ == "__main__":
    main()
