#!/usr/bin/env python3
"""Focused report for the assignment goal: cosine calibration -> WSD prediction.

The broader worktree contains many audits with different source schedules.  This
script extracts the part relevant to the original assignment: calibrating from
the long cosine curve and applying the correction to WSD-family targets.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "cosine_to_wsd_focus"

TARGETS = {
    "wsd_20000_24000.csv": "WSD sharp",
    "wsdld_20000_24000.csv": "WSD linear",
    "wsdcon_3.csv": "WSD-con 3e-5",
    "wsdcon_9.csv": "WSD-con 9e-5",
    "wsdcon_18.csv": "WSD-con 18e-5",
}


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


def fmt_pct(value: float) -> str:
    return f"{value:+.1f}%"


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    deltas = [float(row["delta_pct"]) for row in rows]
    return {
        "rows": len(rows),
        "mean_delta": float(np.mean(deltas)),
        "worst_delta": float(np.max(deltas)),
        "wins": int(sum(float(row["delta_pct"]) < 0.0 for row in rows)),
        "nonharm": int(sum(float(row["delta_pct"]) <= 1e-12 for row in rows)),
    }


def raw_family_rows() -> list[dict[str, object]]:
    rows = read_csv(ROOT / "results" / "current_law_decay_matrix" / "summary.csv")
    out = []
    for row in rows:
        if row["train_family"] != "Cosine decay":
            continue
        if row["test_family"] not in {"WSD sharp", "WSD-con step"}:
            continue
        out.append(
            {
                "method": "raw_cosine_kappa",
                "level": "family",
                "test_label": row["test_family"],
                "rows": int(row["tests"]),
                "mean_delta": float(row["delta_pct"]),
                "wins": int(row["wins"]),
                "nonharm": int(row["wins"]),
                "base_mae": float(row["base_mae"]),
                "corr_mae": float(row["corr_mae"]),
            }
        )
    return out


def final_rows() -> list[dict[str, object]]:
    rows = read_csv(ROOT / "results" / "current_law_final_kappa" / "details.csv")
    out = []
    for row in rows:
        if row["estimator"] != "final_no_cap":
            continue
        if row["train_curve"] != "cosine_72000.csv" or row["test_curve"] not in TARGETS:
            continue
        out.append(
            {
                "method": "final_no_cap",
                "level": "scale",
                "scale": row["scale"],
                "train_curve": row["train_curve"],
                "train_label": row["train_label"],
                "test_curve": row["test_curve"],
                "test_label": row["test_label"],
                "base_mae": float(row["base_mae"]),
                "corr_mae": float(row["corr_mae"]),
                "delta_pct": float(row["delta_pct"]),
                "kappa": float(row["kappa"]),
                "win": int(row["win"]),
            }
        )
    return out


def nextgen_rows() -> list[dict[str, object]]:
    rows = read_csv(ROOT / "results" / "current_law_target_identifiability_audit" / "details.csv")
    out = []
    for row in rows:
        if row["mode"] != "retention_gate_0p01" or row["train_size"] != "1":
            continue
        if row["group"] != "main_matrix" or row["train_id"] != "cosine_72000.csv":
            continue
        if row["test_curve"] not in TARGETS:
            continue
        out.append(
            {
                "method": "nextgen_safe_rho0p5_Rtarget0p01",
                "level": "scale",
                "scale": row["scale"],
                "train_curve": row["train_id"],
                "train_label": row["train_label"],
                "test_curve": row["test_curve"],
                "test_label": row["test_label"],
                "base_mae": float(row["base_mae"]),
                "corr_mae": float(row["corr_mae"]),
                "delta_pct": float(row["delta_pct"]),
                "kappa": float(row["kappa"]),
                "target_retention": float(row["target_retention"]),
                "target_factor": float(row["target_factor"]),
                "selected_lambda": float(row["selected_lambda"]),
                "win": int(row["win"]),
            }
        )
    return out


def per_target(rows: list[dict[str, object]], method: str) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["test_curve"])].append(row)
    out = []
    for curve, sub in sorted(grouped.items(), key=lambda kv: TARGETS[kv[0]]):
        summary = summarize(sub)
        out.append(
            {
                "method": method,
                "test_curve": curve,
                "test_label": TARGETS[curve],
                **summary,
                "mean_kappa": float(np.mean([float(row["kappa"]) for row in sub])),
            }
        )
    return out


def write_report(
    raw_rows: list[dict[str, object]],
    final_detail: list[dict[str, object]],
    nextgen_detail: list[dict[str, object]],
    target_summary: list[dict[str, object]],
) -> None:
    final_all = summarize(final_detail)
    nextgen_all = summarize(nextgen_detail)
    lines = [
        "# Cosine-to-WSD Focus Report\n\n",
        "This report isolates the assignment target: learn the correction from `cosine_72000.csv` and apply it to WSD-family targets.  It intentionally excludes WSD/probe-source routes.\n\n",
        "## Formula Direction\n\n",
        "The useful cosine-calibrated estimator is not the raw cosine residual fit.  It decomposes the cosine residual before estimating the transferable amplitude:\n\n",
        "```text\n",
        "r = L_true - L_MPL = kappa * phi + g_lowfreq + eps\n",
        "M_lambda y = y - Q (Q^T Q + lambda D)^(-1) Q^T y\n",
        "kappa = [n/(n+0.5)] * sqrt(||M_lambda phi||^2 / ||phi||^2)\n",
        "        * max(0, <M_lambda phi, M_lambda r> / (||M_lambda phi||^2 + tau_EB^2))\n",
        "```\n\n",
        "Here `Q` is a DCT low-frequency nuisance basis, `lambda` is selected inside the train-only identifiable band `[0.01, 0.03]`, and the WSD target is used only through its LR-derived response feature and target retention gate.\n\n",
        "## Main Result\n\n",
        f"- Conservative `final_no_cap`: mean `{fmt_pct(float(final_all['mean_delta']))}`, worst `{fmt_pct(float(final_all['worst_delta']))}`, wins `{int(final_all['wins'])}/{int(final_all['rows'])}`.\n",
        f"- Focused `nextgen_safe`: mean `{fmt_pct(float(nextgen_all['mean_delta']))}`, worst `{fmt_pct(float(nextgen_all['worst_delta']))}`, wins `{int(nextgen_all['wins'])}/{int(nextgen_all['rows'])}`.\n\n",
        "## Per-Target Summary\n\n",
        "| method | target | mean delta | worst scale | wins |\n",
        "|---|---|---:|---:|---:|\n",
    ]
    for row in target_summary:
        lines.append(
            f"| {row['method']} | {row['test_label']} | {fmt_pct(float(row['mean_delta']))} | "
            f"{fmt_pct(float(row['worst_delta']))} | {int(row['wins'])}/{int(row['rows'])} |\n"
        )
    lines += [
        "\n## Raw Cosine-Kappa Failure\n\n",
        "| raw transfer group | MAE change | wins |\n",
        "|---|---:|---:|\n",
    ]
    for row in raw_rows:
        lines.append(
            f"| {row['test_label']} | {fmt_pct(float(row['mean_delta']))} | "
            f"{int(row['wins'])}/{int(row['rows'])} |\n"
        )
    lines += [
        "\n## Interpretation\n\n",
        "- Raw cosine residual fitting fails because cosine residual is dominated by smooth MPL backbone drift, not only WSD-like decay lag.\n",
        "- The actual solution is a residual decomposition problem: remove low-frequency drift, estimate only the identifiable response component, shrink the amplitude because one cosine curve is limited evidence, then apply it to WSD targets whose response direction is identifiable.\n",
        "- This keeps the assignment goal intact: the calibration source remains cosine; WSD loss curves are used only for evaluation.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    raw = raw_family_rows()
    final_detail = final_rows()
    nextgen_detail = nextgen_rows()
    target_summary = per_target(final_detail, "final_no_cap") + per_target(
        nextgen_detail, "nextgen_safe_rho0p5_Rtarget0p01"
    )
    write_csv(OUT_DIR / "raw_family_summary.csv", raw)
    write_csv(OUT_DIR / "final_no_cap_details.csv", final_detail)
    write_csv(OUT_DIR / "nextgen_safe_details.csv", nextgen_detail)
    write_csv(OUT_DIR / "target_summary.csv", target_summary)
    write_report(raw, final_detail, nextgen_detail, target_summary)


if __name__ == "__main__":
    main()
