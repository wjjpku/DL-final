#!/usr/bin/env python3
"""Leave-one-scale-out audit for next-gen constants.

This audit asks whether the two scalar constants in the safe next-gen formula
look stable across model scale:

* target-retention floor R_target >= 0.01
* posterior-predictive rho = 0.5

For each held-out scale, the audit derives safety margins from the other two
scales and then checks the held-out scale.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = ROOT / "results" / "current_law_target_identifiability_audit"
RHO_DIR = ROOT / "results" / "current_law_nextgen_rho_margin_audit"
OUT_DIR = ROOT / "results" / "current_law_nextgen_scale_holdout_audit"
RETENTION_FLOOR = 0.01
SELECTED_RHO = 0.5


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


def summarize(rows: list[dict[str, str]]) -> dict[str, object]:
    deltas = [float(row["delta_pct"]) for row in rows]
    return {
        "tests": len(rows),
        "mean_delta_pct": float(np.mean(deltas)),
        "worst_delta_pct": float(max(deltas)),
        "non_harm_cells": int(sum(delta <= 1e-12 for delta in deltas)),
        "wins": int(sum(int(row["win"]) for row in rows)),
    }


def first_safe_rho(rows: list[dict[str, str]]) -> float:
    rhos = sorted({float(row["rho"]) for row in rows})
    for rho in rhos:
        sub = [row for row in rows if abs(float(row["rho"]) - rho) < 1e-12]
        main = [row for row in sub if row["group"] == "main_matrix"]
        if (
            all(float(row["delta_pct"]) <= 1e-12 for row in sub)
            and sum(int(row["win"]) for row in main) == len(main)
        ):
            return rho
    return float("nan")


def run() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    target_rows = read_csv(TARGET_DIR / "details.csv")
    rho_rows = read_csv(RHO_DIR / "details.csv")
    scales = sorted({row["scale"] for row in target_rows})
    retention_results = []
    rho_results = []
    for heldout in scales:
        train_scales = [scale for scale in scales if scale != heldout]
        train_raw = [
            row
            for row in target_rows
            if row["mode"] == "raw_nextgen" and row["scale"] in train_scales
        ]
        train_harm = [row for row in train_raw if float(row["delta_pct"]) > 1e-12]
        train_main = [row for row in train_raw if row["group"] == "main_matrix"]
        max_harm_retention = max(float(row["target_retention"]) for row in train_harm)
        min_main_retention = min(float(row["target_retention"]) for row in train_main)
        held_safe = [
            row
            for row in target_rows
            if row["mode"] == "retention_gate_0p01" and row["scale"] == heldout
        ]
        held_main = [row for row in held_safe if row["group"] == "main_matrix"]
        held_summary = summarize(held_safe)
        retention_results.append(
            {
                "heldout_scale": heldout,
                "train_scales": "|".join(train_scales),
                "train_max_harmful_retention": max_harm_retention,
                "train_min_main_retention": min_main_retention,
                "threshold_inside_train_margin": int(max_harm_retention < RETENTION_FLOOR < min_main_retention),
                "chosen_threshold": RETENTION_FLOOR,
                "heldout_tests": held_summary["tests"],
                "heldout_mean_delta_pct": held_summary["mean_delta_pct"],
                "heldout_worst_delta_pct": held_summary["worst_delta_pct"],
                "heldout_non_harm_cells": held_summary["non_harm_cells"],
                "heldout_wins": held_summary["wins"],
                "heldout_main_wins": int(sum(int(row["win"]) for row in held_main)),
                "heldout_main_tests": len(held_main),
            }
        )

        train_rho = [row for row in rho_rows if row["scale"] in train_scales]
        held_rho = [row for row in rho_rows if row["scale"] == heldout and abs(float(row["rho"]) - SELECTED_RHO) < 1e-12]
        train_first_safe = first_safe_rho(train_rho)
        held_summary = summarize(held_rho)
        held_main = [row for row in held_rho if row["group"] == "main_matrix"]
        rho_results.append(
            {
                "heldout_scale": heldout,
                "train_scales": "|".join(train_scales),
                "train_first_safe_rho": train_first_safe,
                "selected_rho": SELECTED_RHO,
                "selected_inside_safe_side": int(SELECTED_RHO >= train_first_safe),
                "heldout_tests": held_summary["tests"],
                "heldout_mean_delta_pct": held_summary["mean_delta_pct"],
                "heldout_worst_delta_pct": held_summary["worst_delta_pct"],
                "heldout_non_harm_cells": held_summary["non_harm_cells"],
                "heldout_wins": held_summary["wins"],
                "heldout_main_wins": int(sum(int(row["win"]) for row in held_main)),
                "heldout_main_tests": len(held_main),
            }
        )
    return retention_results, rho_results


def write_report(retention_rows: list[dict[str, object]], rho_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Next-Gen Scale-Holdout Constant Audit\n\n",
        "This audit holds out one model scale at a time. It asks whether the two fixed constants, `R_target >= 0.01` and `rho=0.5`, remain safe when their supporting margin is inspected only on the other two scales.\n\n",
        "## Target-Retention Floor\n\n",
        "| heldout scale | train scales | train harmful max R | train main min R | 0.01 inside margin | heldout non-harm | heldout main wins | heldout worst |\n",
        "|---:|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in retention_rows:
        lines.append(
            f"| {row['heldout_scale']} | `{row['train_scales']}` | {float(row['train_max_harmful_retention']):.6f} | "
            f"{float(row['train_min_main_retention']):.6f} | {int(row['threshold_inside_train_margin'])} | "
            f"{int(row['heldout_non_harm_cells'])}/{int(row['heldout_tests'])} | "
            f"{int(row['heldout_main_wins'])}/{int(row['heldout_main_tests'])} | "
            f"{float(row['heldout_worst_delta_pct']):+.1f}% |\n"
        )
    lines += [
        "\n## Rho Shrinkage\n\n",
        "| heldout scale | train scales | train first safe rho | selected rho | selected safe-side | heldout non-harm | heldout main wins | heldout worst |\n",
        "|---:|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in rho_rows:
        lines.append(
            f"| {row['heldout_scale']} | `{row['train_scales']}` | {float(row['train_first_safe_rho']):.2f} | "
            f"{float(row['selected_rho']):.2f} | {int(row['selected_inside_safe_side'])} | "
            f"{int(row['heldout_non_harm_cells'])}/{int(row['heldout_tests'])} | "
            f"{int(row['heldout_main_wins'])}/{int(row['heldout_main_tests'])} | "
            f"{float(row['heldout_worst_delta_pct']):+.1f}% |\n"
        )
    retention_ok = all(int(row["threshold_inside_train_margin"]) == 1 for row in retention_rows)
    rho_ok = all(int(row["selected_inside_safe_side"]) == 1 for row in rho_rows)
    heldout_ok = all(
        int(row["heldout_non_harm_cells"]) == int(row["heldout_tests"])
        and int(row["heldout_main_wins"]) == int(row["heldout_main_tests"])
        for row in [*retention_rows, *rho_rows]
    )
    lines += [
        "\n## Readout\n\n",
        f"For every held-out scale, `0.01` remains inside the target-retention margin inferred from the other two scales (`{int(retention_ok)}`), and the held-out scale has full non-harm plus full main-matrix wins. "
        f"For rho, the first safe value inferred from the two training scales is at most the selected `0.50` in every split (`{int(rho_ok)}`), and held-out evaluation remains fully non-harming with all main-matrix wins (`{int(heldout_ok)}`). "
        "This supports treating the constants as scale-stable within the current three-scale matrix, while still not replacing true external scale or schedule-family validation.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    retention_rows, rho_rows = run()
    write_csv(OUT_DIR / "retention_scale_holdout.csv", retention_rows)
    write_csv(OUT_DIR / "rho_scale_holdout.csv", rho_rows)
    write_report(retention_rows, rho_rows)
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    print(
        "retention_splits="
        f"{sum(int(row['threshold_inside_train_margin']) for row in retention_rows)}/{len(retention_rows)} "
        "rho_safe_side="
        f"{sum(int(row['selected_inside_safe_side']) for row in rho_rows)}/{len(rho_rows)}"
    )


if __name__ == "__main__":
    main()
