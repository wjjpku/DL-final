#!/usr/bin/env python3
"""Compare paper-facing final kappa with the next-gen safe kappa.

This audit uses the common six-schedule off-diagonal matrix for single-curve
calibration. It keeps two readouts:

* scale-level rows, which expose individual-scale failures, and
* train/test cell means across scales, matching the paper-facing matrix
  summary convention.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import TwoSlopeNorm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
FINAL_DIR = ROOT / "results" / "current_law_final_kappa"
TARGET_ID_DIR = ROOT / "results" / "current_law_target_identifiability_audit"
OUT_DIR = ROOT / "results" / "current_law_nextgen_vs_final_audit"
FIG_DIR = OUT_DIR / "figs"


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


def common_rows() -> list[dict[str, object]]:
    final = [
        row
        for row in read_csv(FINAL_DIR / "details.csv")
        if row["estimator"] == "final_no_cap" and row["train_curve"] != row["test_curve"]
    ]
    nextgen = [
        row
        for row in read_csv(TARGET_ID_DIR / "details.csv")
        if row["mode"] == "retention_gate_0p01" and row["train_size"] == "1" and row["group"] == "main_matrix"
    ]
    rows: list[dict[str, object]] = []
    for row in final:
        rows.append(
            {
                "method": "final_no_cap",
                "scale": row["scale"],
                "train_curve": row["train_curve"],
                "train_label": row["train_label"],
                "test_curve": row["test_curve"],
                "test_label": row["test_label"],
                "delta_pct": float(row["delta_pct"]),
                "win": int(row["win"]),
                "kappa": float(row["kappa"]),
            }
        )
    for row in nextgen:
        rows.append(
            {
                "method": "nextgen_safe_rho0p5_Rtarget0p01",
                "scale": row["scale"],
                "train_curve": row["train_id"],
                "train_label": row["train_label"],
                "test_curve": row["test_curve"],
                "test_label": row["test_label"],
                "delta_pct": float(row["delta_pct"]),
                "win": int(row["win"]),
                "kappa": float(row["kappa"]),
                "target_factor": float(row["target_factor"]),
                "target_retention": float(row["target_retention"]),
            }
        )
    return rows


def summarize_scale(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for method in sorted({str(row["method"]) for row in rows}):
        sub = [row for row in rows if row["method"] == method]
        deltas = [float(row["delta_pct"]) for row in sub]
        out.append(
            {
                "method": method,
                "rows": len(sub),
                "mean_delta_pct": float(np.mean(deltas)),
                "worst_delta_pct": float(max(deltas)),
                "wins": int(sum(delta < 0 for delta in deltas)),
                "non_harm_rows": int(sum(delta <= 1e-12 for delta in deltas)),
            }
        )
    return out


def cell_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row["train_curve"]), str(row["test_curve"]))].append(row)
    out = []
    for (method, train_curve, test_curve), sub in sorted(grouped.items()):
        deltas = [float(row["delta_pct"]) for row in sub]
        out.append(
            {
                "method": method,
                "train_curve": train_curve,
                "train_label": sub[0]["train_label"],
                "test_curve": test_curve,
                "test_label": sub[0]["test_label"],
                "mean_delta_pct": float(np.mean(deltas)),
                "worst_scale_delta_pct": float(max(deltas)),
                "wins": int(sum(delta < 0 for delta in deltas)),
                "scale_rows": len(sub),
            }
        )
    return out


def summarize_cells(cells: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for method in sorted({str(row["method"]) for row in cells}):
        sub = [row for row in cells if row["method"] == method]
        means = [float(row["mean_delta_pct"]) for row in sub]
        out.append(
            {
                "method": method,
                "cells": len(sub),
                "mean_cell_delta_pct": float(np.mean(means)),
                "worst_cell_delta_pct": float(max(means)),
                "winning_cells": int(sum(delta < 0 for delta in means)),
            }
        )
    return out


def paired_rows(cells: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key = {
        (row["method"], row["train_curve"], row["test_curve"]): row
        for row in cells
    }
    pairs = []
    for key, final_row in by_key.items():
        method, train_curve, test_curve = key
        if method != "final_no_cap":
            continue
        next_row = by_key[("nextgen_safe_rho0p5_Rtarget0p01", train_curve, test_curve)]
        diff = float(next_row["mean_delta_pct"]) - float(final_row["mean_delta_pct"])
        pairs.append(
            {
                "train_curve": train_curve,
                "train_label": final_row["train_label"],
                "test_curve": test_curve,
                "test_label": final_row["test_label"],
                "final_mean_delta_pct": float(final_row["mean_delta_pct"]),
                "nextgen_mean_delta_pct": float(next_row["mean_delta_pct"]),
                "nextgen_minus_final_pct": diff,
                "nextgen_better": int(diff < 0),
            }
        )
    return sorted(pairs, key=lambda row: float(row["nextgen_minus_final_pct"]))


def key_transfers(cells: list[dict[str, object]]) -> list[dict[str, object]]:
    wanted = {
        ("Cosine", "WSD sharp"),
        ("WSD-con 9e-5", "WSD sharp"),
        ("WSD linear", "WSD sharp"),
    }
    out = []
    for row in cells:
        if (row["train_label"], row["test_label"]) in wanted:
            out.append(row)
    return out


def plot_paired_heatmap(path: Path, pairs: list[dict[str, object]]) -> None:
    labels = sorted({str(row["train_label"]) for row in pairs})
    label_to_idx = {label: i for i, label in enumerate(labels)}
    mat = np.full((len(labels), len(labels)), np.nan)
    for row in pairs:
        i = label_to_idx[str(row["train_label"])]
        j = label_to_idx[str(row["test_label"])]
        mat[i, j] = float(row["nextgen_minus_final_pct"])
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(mat, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-20, vcenter=0, vmax=20))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Next-gen safe minus final_no_cap (cell mean MAE change)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j:
                ax.text(j, i, "-", ha="center", va="center", color="#555555", fontsize=8)
                continue
            value = mat[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("next-gen minus final (percentage points)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def plot_method_matrices(path: Path, cells: list[dict[str, object]]) -> None:
    labels = sorted({str(row["train_label"]) for row in cells})
    methods = ["final_no_cap", "nextgen_safe_rho0p5_Rtarget0p01"]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.7), sharex=True, sharey=True)
    norm = TwoSlopeNorm(vmin=-35, vcenter=0, vmax=15)
    for ax, method in zip(axes, methods):
        mat = np.full((len(labels), len(labels)), np.nan)
        for row in cells:
            if row["method"] != method:
                continue
            i = labels.index(str(row["train_label"]))
            j = labels.index(str(row["test_label"]))
            mat[i, j] = float(row["mean_delta_pct"])
        im = ax.imshow(mat, cmap="RdYlGn_r", norm=norm)
        ax.set_title(method)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                if i == j:
                    ax.text(j, i, "-", ha="center", va="center", color="#555555", fontsize=8)
                elif np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:+.1f}", ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
    cbar.set_label("MAE change vs MPL (%)")
    fig.suptitle("Common single-curve matrix: final vs next-gen safe", y=1.02)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def write_report(
    scale_summary: list[dict[str, object]],
    cell_summary: list[dict[str, object]],
    pairs: list[dict[str, object]],
    keys: list[dict[str, object]],
) -> None:
    by_cell = {row["method"]: row for row in cell_summary}
    by_scale = {row["method"]: row for row in scale_summary}
    final_cell = by_cell["final_no_cap"]
    next_cell = by_cell["nextgen_safe_rho0p5_Rtarget0p01"]
    next_scale = by_scale["nextgen_safe_rho0p5_Rtarget0p01"]
    lines = [
        "# Next-Gen vs Final Kappa Audit\n\n",
        "This audit compares the paper-facing `final_no_cap` estimator with the next-generation safe estimator on the common six-schedule, single-curve off-diagonal matrix.\n\n",
        "## Summary\n\n",
        "| method | cell mean | worst cell | winning cells | scale mean | worst scale | non-harm scale rows |\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in cell_summary:
        scale = by_scale[row["method"]]
        lines.append(
            f"| `{row['method']}` | {float(row['mean_cell_delta_pct']):+.1f}% | "
            f"{float(row['worst_cell_delta_pct']):+.1f}% | {int(row['winning_cells'])}/{int(row['cells'])} | "
            f"{float(scale['mean_delta_pct']):+.1f}% | {float(scale['worst_delta_pct']):+.1f}% | "
        f"{int(scale['non_harm_rows'])}/{int(scale['rows'])} |\n"
        )
    lines += [
        "\n![method matrices](figs/method_matrices.png)\n\n",
        "![paired difference](figs/paired_difference_heatmap.png)\n\n",
        "\n## Key Transfers\n\n",
        "| method | train -> test | mean delta | worst scale | scale wins |\n",
        "|---|---|---:|---:|---:|\n",
    ]
    for row in keys:
        lines.append(
            f"| `{row['method']}` | {row['train_label']} -> {row['test_label']} | "
            f"{float(row['mean_delta_pct']):+.1f}% | {float(row['worst_scale_delta_pct']):+.1f}% | "
            f"{int(row['wins'])}/{int(row['scale_rows'])} |\n"
        )
    better = sum(int(row["nextgen_better"]) for row in pairs)
    lines += [
        "\n## Readout\n\n",
        f"On the cell-mean matrix, next-gen safe is comparable to the final estimator: mean `{float(next_cell['mean_cell_delta_pct']):+.1f}%` versus `{float(final_cell['mean_cell_delta_pct']):+.1f}%`, and both improve all `30/30` off-diagonal cells. "
        f"The final estimator has the stronger worst cell (`{float(final_cell['worst_cell_delta_pct']):+.1f}%` versus `{float(next_cell['worst_cell_delta_pct']):+.1f}%`), while next-gen safe has stronger scale-level non-harm (`{int(next_scale['non_harm_rows'])}/{int(next_scale['rows'])}`). "
        f"Paired by train/test cell, next-gen is better on `{better}/{len(pairs)}` cells, so this is not a strict dominance result.\n\n",
        "The practical difference is directional: next-gen safe is much stronger for Cosine -> WSD sharp, while the final estimator is stronger on some WSD-con -> WSD transfers. This supports presenting next-gen as the best current research extension, not as a replacement for the conservative paper-facing estimator.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = common_rows()
    cells = cell_rows(rows)
    scale_summary = summarize_scale(rows)
    cell_summary = summarize_cells(cells)
    pairs = paired_rows(cells)
    keys = key_transfers(cells)
    write_csv(OUT_DIR / "details.csv", rows)
    write_csv(OUT_DIR / "cell_details.csv", cells)
    write_csv(OUT_DIR / "scale_summary.csv", scale_summary)
    write_csv(OUT_DIR / "cell_summary.csv", cell_summary)
    write_csv(OUT_DIR / "paired_cells.csv", pairs)
    write_csv(OUT_DIR / "key_transfers.csv", keys)
    plot_method_matrices(FIG_DIR / "method_matrices.png", cells)
    plot_paired_heatmap(FIG_DIR / "paired_difference_heatmap.png", pairs)
    write_report(scale_summary, cell_summary, pairs, keys)
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    for row in cell_summary:
        scale = next(r for r in scale_summary if r["method"] == row["method"])
        print(
            f"{row['method']:32s} cell_mean={float(row['mean_cell_delta_pct']):+6.1f}% "
            f"worst_cell={float(row['worst_cell_delta_pct']):+6.1f}% "
            f"scale_nonharm={int(scale['non_harm_rows'])}/{int(scale['rows'])}"
        )


if __name__ == "__main__":
    main()
