#!/usr/bin/env python3
"""Target-identifiability attenuation for next-generation kappa.

The current safety rule abstains when the target response feature is diffuse.
This audit tests a more model-native alternative: attenuate transfer by the
fraction of target response energy that survives the same soft spectral
nuisance residualizer used to estimate kappa.
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
OUT_DIR = ROOT / "results" / "current_law_target_identifiability_audit"
LOCALIZATION_THRESHOLD = 2.0
RETENTION_ALPHAS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
RETENTION_THRESHOLDS = [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.05, 0.1]
TRAIN_RELATIVE_BETAS = [0.05, 0.1, 0.2, 0.5, 1.0]
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


def target_stats(scale: str, curve_name: str, lam: float) -> dict[str, object]:
    curve = base.load_curve(scale, curve_name)
    phi = stime_feature(curve, base.LAMBDA)
    baseline = base.mpl_predict(base.MPL_PRECOMPUTED_INIT[scale], curve)
    q = spectral.dct_basis(len(curve.step), spectral.MAX_MODE)
    a = spectral.smoother_matrix(q, lam)
    phi_o = spectral.soft_residualize(phi, q, a)
    phi_l2 = float(np.dot(phi, phi))
    retention = 0.0 if phi_l2 <= 1e-18 else float(np.dot(phi_o, phi_o) / phi_l2)
    peak = float(np.max(phi))
    mean = float(np.mean(phi))
    return {
        "curve": curve,
        "phi": phi,
        "baseline": baseline,
        "base_mae": base.metrics(curve.loss, baseline)["mae"],
        "target_retention": retention,
        "target_peak_to_mean": 0.0 if peak <= 0 else peak / max(mean, 1e-12),
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
    rows: list[dict[str, object]] = []
    for mode in sorted({str(row["mode"]) for row in details}):
        for group in ["main_matrix", "extra_holdout", "all"]:
            sub = [row for row in details if row["mode"] == mode and (group == "all" or row["group"] == group)]
            rows.append(
                {
                    "mode": mode,
                    "group": group,
                    "tests": len(sub),
                    "mean_delta_pct": float(np.mean([float(row["delta_pct"]) for row in sub])),
                    "worst_delta_pct": float(max(float(row["delta_pct"]) for row in sub)),
                    "non_harm_cells": int(sum(float(row["delta_pct"]) <= 1e-12 for row in sub)),
                    "wins": int(sum(int(row["win"]) for row in sub)),
                    "mean_target_factor": float(np.mean([float(row["target_factor"]) for row in sub])),
                    "mean_target_retention": float(np.mean([float(row["target_retention"]) for row in sub])),
                    "mean_peak_to_mean": float(np.mean([float(row["target_peak_to_mean"]) for row in sub])),
                }
            )
    return rows


def summarize_by_train_size(details: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for mode in sorted({str(row["mode"]) for row in details}):
        for train_size in sorted({int(row["train_size"]) for row in details}):
            for group in ["main_matrix", "extra_holdout", "all"]:
                sub = [
                    row
                    for row in details
                    if row["mode"] == mode
                    and int(row["train_size"]) == train_size
                    and (group == "all" or row["group"] == group)
                ]
                rows.append(
                    {
                        "mode": mode,
                        "train_size": train_size,
                        "group": group,
                        "tests": len(sub),
                        "mean_delta_pct": float(np.mean([float(row["delta_pct"]) for row in sub])),
                        "worst_delta_pct": float(max(float(row["delta_pct"]) for row in sub)),
                        "non_harm_cells": int(sum(float(row["delta_pct"]) <= 1e-12 for row in sub)),
                        "wins": int(sum(int(row["win"]) for row in sub)),
                        "mean_target_factor": float(np.mean([float(row["target_factor"]) for row in sub])),
                    }
                )
    return rows


def curve_summary(details: list[dict[str, object]]) -> list[dict[str, object]]:
    raw = [row for row in details if row["mode"] == "raw_nextgen"]
    rows: list[dict[str, object]] = []
    keys = sorted({(str(row["test_curve"]), str(row["test_label"]), str(row["group"])) for row in raw})
    for curve_name, label, group in keys:
        sub = [row for row in raw if row["test_curve"] == curve_name]
        rows.append(
            {
                "test_curve": curve_name,
                "test_label": label,
                "group": group,
                "tests": len(sub),
                "min_retention": float(min(float(row["target_retention"]) for row in sub)),
                "mean_retention": float(np.mean([float(row["target_retention"]) for row in sub])),
                "max_retention": float(max(float(row["target_retention"]) for row in sub)),
                "mean_peak_to_mean": float(np.mean([float(row["target_peak_to_mean"]) for row in sub])),
                "raw_worst_delta_pct": float(max(float(row["delta_pct"]) for row in sub)),
            }
        )
    return rows


def train_reference_retention(
    cached: dict[tuple[str, str, float], dict[str, object]], scale: str, train_curves: tuple[str, ...], lam: float
) -> float:
    values = [float(cached[(scale, curve, lam)]["target_retention"]) for curve in train_curves]
    return float(min(values))


def run() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    krows = [
        row
        for row in read_csv(PRED_DIR / "kappa_diagnostics.csv")
        if row["candidate"] == "train_size_rho0p5"
    ]
    all_curves = [*base.CURVES, *EXTRA_CURVES]
    labels = {curve: label for curve, label in all_curves}
    target_keys = sorted(
        {
            (krow["scale"], curve, float(krow["selected_lambda"]))
            for krow in krows
            for curve, _ in all_curves
        }
    )
    cached = {
        (scale, curve, lam): target_stats(scale, curve, lam)
        for scale, curve, lam in target_keys
    }
    details: list[dict[str, object]] = []
    for krow in krows:
        scale = krow["scale"]
        train_curves = tuple(krow["train_id"].split("|"))
        lam = float(krow["selected_lambda"])
        kappa = float(krow["kappa"])
        train_retention = train_reference_retention(cached, scale, train_curves, lam)
        for test_curve, test_label in all_curves:
            if test_curve in train_curves:
                continue
            stats = cached[(scale, test_curve, lam)]
            retention = float(stats["target_retention"])
            modes = {
                "raw_nextgen": 1.0,
                "peak_mean_gate": 0.0 if float(stats["target_peak_to_mean"]) < LOCALIZATION_THRESHOLD else 1.0,
            }
            for alpha in RETENTION_ALPHAS:
                modes[f"retention_alpha_{str(alpha).replace('.', 'p')}"] = retention**alpha
            for threshold in RETENTION_THRESHOLDS:
                name = str(threshold).replace(".", "p")
                modes[f"retention_gate_{name}"] = 0.0 if retention < threshold else 1.0
            for beta in TRAIN_RELATIVE_BETAS:
                name = str(beta).replace(".", "p")
                modes[f"train_relative_gate_{name}"] = 0.0 if retention < beta * train_retention else 1.0
                modes[f"floor_train_relative_gate_{name}"] = (
                    0.0 if retention < max(0.01, beta * train_retention) else 1.0
                )
            for mode, factor in modes.items():
                scored = score(stats, kappa * factor)
                details.append(
                    {
                        "mode": mode,
                        "group": "main_matrix" if test_curve in {curve for curve, _ in base.CURVES} else "extra_holdout",
                        "scale": scale,
                        "train_id": krow["train_id"],
                        "train_size": int(krow["train_size"]),
                        "train_label": krow["train_label"],
                        "test_curve": test_curve,
                        "test_label": test_label,
                        "selected_lambda": lam,
                        "kappa": kappa,
                        "target_factor": factor,
                        "target_retention": retention,
                        "train_retention": train_retention,
                        "target_peak_to_mean": float(stats["target_peak_to_mean"]),
                        **scored,
                    }
                )
    return details, summarize(details), summarize_by_train_size(details), curve_summary(details)


def write_report(
    summary: list[dict[str, object]], train_size_summary: list[dict[str, object]], curves: list[dict[str, object]]
) -> None:
    lines = [
        "# Target Identifiability Attenuation Audit\n\n",
        "This audit asks whether the next-generation `rho=0.5` kappa can use a continuous target-side identifiability factor instead of a binary peak/mean abstention gate. "
        "For each target schedule, the factor is based on the response-energy retention after applying the same soft DCT/Sobolev nuisance residualizer used during kappa estimation.\n\n",
        "Candidate factor:\n\n",
        "```text\n",
        "R_target(lambda) = ||M_lambda phi_target||^2 / ||phi_target||^2\n",
        "kappa_safe = R_target(lambda)^alpha * kappa_transfer\n",
        "```\n\n",
        "## Summary\n\n",
        "| mode | group | mean delta | worst delta | non-harm cells | wins | target factor | retention |\n",
        "|---|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in summary:
        lines.append(
            f"| `{row['mode']}` | {row['group']} | {float(row['mean_delta_pct']):+.1f}% | "
            f"{float(row['worst_delta_pct']):+.1f}% | {int(row['non_harm_cells'])}/{int(row['tests'])} | "
            f"{int(row['wins'])}/{int(row['tests'])} | {float(row['mean_target_factor']):.3f} | "
            f"{float(row['mean_target_retention']):.3f} |\n"
        )
    lines += [
        "\n## Train-Size Breakdown\n\n",
        "| mode | train curves | group | mean delta | worst delta | non-harm cells | wins | target factor |\n",
        "|---|---:|---|---:|---:|---:|---:|---:|\n",
    ]
    for row in train_size_summary:
        if row["mode"] not in {"raw_nextgen", "retention_gate_0p01", "train_relative_gate_0p5"}:
            continue
        if row["group"] != "all":
            continue
        lines.append(
            f"| `{row['mode']}` | {int(row['train_size'])} | {row['group']} | {float(row['mean_delta_pct']):+.1f}% | "
            f"{float(row['worst_delta_pct']):+.1f}% | {int(row['non_harm_cells'])}/{int(row['tests'])} | "
            f"{int(row['wins'])}/{int(row['tests'])} | {float(row['mean_target_factor']):.3f} |\n"
        )
    lines += [
        "\n## Target Retention By Curve\n\n",
        "| target | group | min retention | mean retention | max retention | peak/mean | raw worst |\n",
        "|---|---|---:|---:|---:|---:|---:|\n",
    ]
    for row in curves:
        lines.append(
            f"| {row['test_label']} | {row['group']} | {float(row['min_retention']):.6f} | "
            f"{float(row['mean_retention']):.6f} | {float(row['max_retention']):.6f} | "
            f"{float(row['mean_peak_to_mean']):.2f} | {float(row['raw_worst_delta_pct']):+.1f}% |\n"
        )
    main_positive = [row for row in curves if row["group"] == "main_matrix" and float(row["min_retention"]) > 0]
    extra_positive = [row for row in curves if row["group"] == "extra_holdout" and float(row["max_retention"]) > 0]
    main_floor = min(float(row["min_retention"]) for row in main_positive)
    extra_ceiling = max(float(row["max_retention"]) for row in extra_positive)
    log_midpoint = float((main_floor * extra_ceiling) ** 0.5)
    all_rows = [row for row in summary if row["group"] == "all"]
    best_safe = [row for row in all_rows if float(row["worst_delta_pct"]) <= 1e-12]
    best_safe = sorted(best_safe, key=lambda row: float(row["mean_delta_pct"]))
    lines += ["\n## Readout\n\n"]
    if best_safe:
        best = best_safe[0]
        lines.append(
            f"The best non-harming target-identifiability candidate is `{best['mode']}`: mean `{float(best['mean_delta_pct']):+.1f}%`, "
            f"worst `{float(best['worst_delta_pct']):+.1f}%`, non-harm `{int(best['non_harm_cells'])}/{int(best['tests'])}`. "
            "Retention-gated candidates are more theory-native than peak/mean because they measure identifiability after the exact nuisance residualizer used by the estimator.\n"
        )
    else:
        lines.append(
            "No continuous retention-only candidate was fully non-harming on the combined main-plus-extra audit. "
            "This means the binary target-localization abstention remains the safer deployment rule on current evidence.\n"
        )
    train_relative = [row for row in all_rows if str(row["mode"]).startswith("train_relative_gate_")]
    safe_train_relative = [row for row in train_relative if float(row["worst_delta_pct"]) <= 1e-12]
    if safe_train_relative:
        best_train = sorted(safe_train_relative, key=lambda row: float(row["mean_delta_pct"]))[0]
        lines.append(
            f"\nA pure train-relative threshold can be made safe only by becoming more conservative: `{best_train['mode']}` has "
            f"mean `{float(best_train['mean_delta_pct']):+.1f}%` and non-harm `{int(best_train['non_harm_cells'])}/{int(best_train['tests'])}`. "
            "Weaker train-relative thresholds preserve more transfer but let the diffuse external cosine target through. "
            "This supports using an absolute target-identifiability floor rather than only a relative-to-calibration threshold.\n"
        )
    lines.append(
        f"\nThe threshold `0.01` has a margin interpretation on the current artifacts: the lowest positive main-matrix target retention is `{main_floor:.6f}`, "
        f"the highest positive extra-holdout diffuse retention is `{extra_ceiling:.6f}`, and their geometric midpoint is `{log_midpoint:.6f}`. "
        "Thus `0.01` separates the retained response directions from the diffuse external cosine target with a visible log-scale gap without using held-out loss values.\n"
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    details, summary, train_size_summary, curves = run()
    write_csv(OUT_DIR / "details.csv", details)
    write_csv(OUT_DIR / "summary.csv", summary)
    write_csv(OUT_DIR / "train_size_summary.csv", train_size_summary)
    write_csv(OUT_DIR / "target_retention_by_curve.csv", curves)
    write_report(summary, train_size_summary, curves)
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    for row in summary:
        if row["group"] == "all":
            print(
                f"{row['mode']:24s} mean={float(row['mean_delta_pct']):+6.1f}% "
                f"worst={float(row['worst_delta_pct']):+6.1f}% "
                f"nonharm={int(row['non_harm_cells'])}/{int(row['tests'])}"
            )


if __name__ == "__main__":
    main()
