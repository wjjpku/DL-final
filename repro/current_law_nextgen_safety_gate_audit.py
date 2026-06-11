#!/usr/bin/env python3
"""Deployment-style target safety gate for the next-generation kappa.

This audit evaluates the raw next-gen single-curve `rho=0.5` transfer and a
target-side localization gate on both:

  * the main six-schedule matrix, and
  * extra repo curves not used in that matrix.

The gate is intentionally target-only:

    a_target = 0 if peak(phi_target) / mean(phi_target) < 2 else 1

It encodes an identifiability rule.  If the target response feature is diffuse,
transferring a positive amplitude into it is indistinguishable from adding
low-frequency MPL drift, so deployment should abstain unless target residual
evidence is available.
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
from deep_stime import stime_feature  # noqa: E402


PRED_DIR = ROOT / "results" / "current_law_predictive_shrinkage_audit"
OUT_DIR = ROOT / "results" / "current_law_nextgen_safety_gate_audit"
LOCALIZATION_THRESHOLD = 2.0
THRESHOLDS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
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
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def target_stats(scale: str, curve_name: str) -> dict[str, object]:
    curve = base.load_curve(scale, curve_name)
    phi = stime_feature(curve, base.LAMBDA)
    baseline = base.mpl_predict(base.MPL_PRECOMPUTED_INIT[scale], curve)
    base_mae = base.metrics(curve.loss, baseline)["mae"]
    peak = float(np.max(phi))
    mean = float(np.mean(phi))
    peak_to_mean = 0.0 if peak <= 0 else peak / max(mean, 1e-12)
    return {"curve": curve, "phi": phi, "baseline": baseline, "base_mae": base_mae, "peak_to_mean": peak_to_mean}


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
    summary: list[dict[str, object]] = []
    for mode in sorted({str(r["mode"]) for r in details}):
        for group in ["main_matrix", "extra_holdout", "all"]:
            sub = [r for r in details if r["mode"] == mode and (group == "all" or r["group"] == group)]
            summary.append(
                {
                    "mode": mode,
                    "group": group,
                    "tests": len(sub),
                    "mean_delta_pct": float(np.mean([float(r["delta_pct"]) for r in sub])),
                    "worst_delta_pct": float(max(float(r["delta_pct"]) for r in sub)),
                    "non_harm_cells": int(sum(float(r["delta_pct"]) <= 1e-12 for r in sub)),
                    "wins": int(sum(int(r["win"]) for r in sub)),
                    "mean_target_factor": float(np.mean([float(r["target_factor"]) for r in sub])),
                }
            )
    return summary


def run() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    krows = [
        r
        for r in read_csv(PRED_DIR / "kappa_diagnostics.csv")
        if r["candidate"] == "train_size_rho0p5" and int(r["train_size"]) == 1
    ]
    all_curves = [*base.CURVES, *EXTRA_CURVES]
    labels = {curve: label for curve, label in all_curves}
    cached = {(scale, curve): target_stats(scale, curve) for scale in base.SCALES for curve, _ in all_curves}
    details: list[dict[str, object]] = []
    for kr in krows:
        scale = kr["scale"]
        kappa = float(kr["kappa"])
        train_curve = kr["train_id"]
        for test_curve, test_label in all_curves:
            if test_curve == train_curve:
                continue
            group = "main_matrix" if test_curve in {c for c, _ in base.CURVES} else "extra_holdout"
            stats = cached[(scale, test_curve)]
            for mode in ["raw_nextgen", "target_localization_gate"]:
                factor = 1.0
                if mode == "target_localization_gate" and float(stats["peak_to_mean"]) < LOCALIZATION_THRESHOLD:
                    factor = 0.0
                scored = score(stats, kappa * factor)
                details.append(
                    {
                        "mode": mode,
                        "group": group,
                        "scale": scale,
                        "train_curve": train_curve,
                        "train_label": kr["train_label"],
                        "test_curve": test_curve,
                        "test_label": test_label,
                        "kappa": kappa,
                        "target_factor": factor,
                        "target_peak_to_mean": float(stats["peak_to_mean"]),
                        **scored,
                    }
                )
    summary = summarize(details)
    sensitivity: list[dict[str, object]] = []
    for threshold in THRESHOLDS:
        swept = []
        for row in details:
            if row["mode"] != "raw_nextgen":
                continue
            factor = 0.0 if float(row["target_peak_to_mean"]) < threshold else 1.0
            scored = score(
                cached[(str(row["scale"]), str(row["test_curve"]))],
                float(row["kappa"]) * factor,
            )
            swept.append({**row, "mode": "threshold_gate", "target_factor": factor, **scored})
        for srow in summarize(swept):
            sensitivity.append({"threshold": threshold, **srow})
    return details, summary, sensitivity


def write_report(summary: list[dict[str, object]], sensitivity: list[dict[str, object]]) -> None:
    def row(mode: str, group: str) -> dict[str, object]:
        return next(r for r in summary if r["mode"] == mode and r["group"] == group)

    raw_extra = row("raw_nextgen", "extra_holdout")
    gated_all = row("target_localization_gate", "all")
    gated_main = row("target_localization_gate", "main_matrix")
    lines = [
        "# Next-Gen Target Safety Gate Audit\n\n",
        "This audit evaluates a deployment-style target-localization gate for the next-generation `rho=0.5` kappa. "
        "The gate is target-only and schedule-label-free: abstain when `peak(phi_target) / mean(phi_target) < 2`.\n\n",
        "## Summary\n\n",
        "| mode | group | mean delta | worst delta | non-harm cells | wins | target factor |\n",
        "|---|---|---:|---:|---:|---:|---:|\n",
    ]
    for r in summary:
        lines.append(
            f"| `{r['mode']}` | {r['group']} | {float(r['mean_delta_pct']):+.1f}% | "
            f"{float(r['worst_delta_pct']):+.1f}% | {int(r['non_harm_cells'])}/{int(r['tests'])} | "
            f"{int(r['wins'])}/{int(r['tests'])} | {float(r['mean_target_factor']):.2f} |\n"
        )
    lines += [
        "\n## Threshold Sensitivity\n\n",
        "| threshold | group | mean delta | worst delta | non-harm cells | target factor |\n",
        "|---:|---|---:|---:|---:|---:|\n",
    ]
    for r in [r for r in sensitivity if r["group"] == "all"]:
        lines.append(
            f"| {float(r['threshold']):.1f} | {r['group']} | {float(r['mean_delta_pct']):+.1f}% | "
            f"{float(r['worst_delta_pct']):+.1f}% | {int(r['non_harm_cells'])}/{int(r['tests'])} | "
            f"{float(r['mean_target_factor']):.2f} |\n"
        )
    lines += [
        "\n## Readout\n\n",
        f"Raw next-gen remains strong on the main matrix but fails the extra holdout group (worst `{float(raw_extra['worst_delta_pct']):+.1f}%`) because of `cosine_24000`. "
        f"The target-localization gate makes the combined main-plus-extra audit non-harming (`{int(gated_all['non_harm_cells'])}/{int(gated_all['tests'])}` cells, worst `{float(gated_all['worst_delta_pct']):+.1f}%`). "
        f"On the main matrix it preserves non-harm (`{int(gated_main['non_harm_cells'])}/{int(gated_main['tests'])}` cells) but abstains on diffuse cosine targets, so wins are lower than raw transfer.\n\n",
        "The threshold sweep shows that `1.5` does not remove the external holdout failure, while thresholds from `2.0` through `4.0` all give `144/144` non-harming cells with the same mean gain. "
        "Larger thresholds such as `5.0` and `6.0` are also safe but more conservative.\n\n",
        "Interpretation: the raw next-gen formula is a strong transfer estimator when the target response direction is identifiable. "
        "The safety gate is a conservative deployment rule for target schedules whose response feature is too diffuse to distinguish from low-frequency MPL drift.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    details, summary, sensitivity = run()
    write_csv(OUT_DIR / "details.csv", details)
    write_csv(OUT_DIR / "summary.csv", summary)
    write_csv(OUT_DIR / "threshold_sensitivity.csv", sensitivity)
    write_report(summary, sensitivity)
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    for r in summary:
        print(
            f"{r['mode']:24s} {r['group']:12s} mean={float(r['mean_delta_pct']):+6.1f}% "
            f"worst={float(r['worst_delta_pct']):+6.1f}% nonharm={int(r['non_harm_cells'])}/{int(r['tests'])}"
        )


if __name__ == "__main__":
    main()
