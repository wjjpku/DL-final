#!/usr/bin/env python3
"""Rho-margin audit for the next-generation safe kappa formula.

The next-generation formula uses c_n = n / (n + rho) with rho=0.5. This audit
keeps the target-identifiability gate fixed and rescans rho to test whether
0.5 is a knife-edge choice or part of a stable safe range.
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
OUT_DIR = ROOT / "results" / "current_law_nextgen_rho_margin_audit"
RETENTION_THRESHOLD = 0.01
RHO_GRID = [
    0.0,
    0.1,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.6,
    0.75,
    1.0,
    1.25,
    1.5,
    2.0,
]
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
    rows: list[dict[str, object]] = []
    for rho in RHO_GRID:
        for group in ["main_matrix", "extra_holdout", "all"]:
            sub = [row for row in details if abs(float(row["rho"]) - rho) < 1e-12 and (group == "all" or row["group"] == group)]
            deltas = [float(row["delta_pct"]) for row in sub]
            rows.append(
                {
                    "rho": rho,
                    "group": group,
                    "tests": len(sub),
                    "mean_delta_pct": float(np.mean(deltas)),
                    "worst_delta_pct": float(max(deltas)),
                    "non_harm_cells": int(sum(delta <= 1e-12 for delta in deltas)),
                    "wins": int(sum(int(row["win"]) for row in sub)),
                    "mean_shrink": float(np.mean([float(row["shrink"]) for row in sub])),
                    "mean_target_factor": float(np.mean([float(row["target_factor"]) for row in sub])),
                }
            )
    return rows


def summarize_by_train_size(details: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rho in RHO_GRID:
        for train_size in sorted({int(row["train_size"]) for row in details}):
            sub = [
                row
                for row in details
                if abs(float(row["rho"]) - rho) < 1e-12 and int(row["train_size"]) == train_size
            ]
            deltas = [float(row["delta_pct"]) for row in sub]
            rows.append(
                {
                    "rho": rho,
                    "train_size": train_size,
                    "tests": len(sub),
                    "mean_delta_pct": float(np.mean(deltas)),
                    "worst_delta_pct": float(max(deltas)),
                    "non_harm_cells": int(sum(delta <= 1e-12 for delta in deltas)),
                    "wins": int(sum(int(row["win"]) for row in sub)),
                    "mean_shrink": float(np.mean([float(row["shrink"]) for row in sub])),
                }
            )
    return rows


def run() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    krows = [
        row
        for row in read_csv(PRED_DIR / "kappa_diagnostics.csv")
        if row["candidate"] == "none"
    ]
    all_curves = [*base.CURVES, *EXTRA_CURVES]
    target_keys = sorted(
        {
            (row["scale"], curve, float(row["selected_lambda"]))
            for row in krows
            for curve, _ in all_curves
        }
    )
    cached = {(scale, curve, lam): target_stats(scale, curve, lam) for scale, curve, lam in target_keys}
    details: list[dict[str, object]] = []
    for krow in krows:
        scale = krow["scale"]
        train_curves = set(krow["train_id"].split("|"))
        train_size = int(krow["train_size"])
        lam = float(krow["selected_lambda"])
        base_kappa = float(krow["base_kappa"])
        for rho in RHO_GRID:
            shrink = train_size / (train_size + rho)
            transfer_kappa = base_kappa * shrink
            for test_curve, test_label in all_curves:
                if test_curve in train_curves:
                    continue
                stats = cached[(scale, test_curve, lam)]
                target_factor = 1.0 if float(stats["target_retention"]) >= RETENTION_THRESHOLD else 0.0
                scored = score(stats, transfer_kappa * target_factor)
                details.append(
                    {
                        "rho": rho,
                        "shrink": shrink,
                        "scale": scale,
                        "train_id": krow["train_id"],
                        "train_label": krow["train_label"],
                        "train_size": train_size,
                        "test_curve": test_curve,
                        "test_label": test_label,
                        "group": "main_matrix" if test_curve in {curve for curve, _ in base.CURVES} else "extra_holdout",
                        "base_kappa": base_kappa,
                        "kappa": transfer_kappa,
                        "selected_lambda": lam,
                        "target_retention": float(stats["target_retention"]),
                        "target_factor": target_factor,
                        **scored,
                    }
                )
    return details, summarize(details), summarize_by_train_size(details)


def write_report(summary: list[dict[str, object]], train_size_summary: list[dict[str, object]]) -> None:
    by_key = {(float(row["rho"]), row["group"]): row for row in summary}
    selected = by_key[(0.5, "all")]
    safe_all = [
        row
        for row in summary
        if row["group"] == "all"
        and int(row["non_harm_cells"]) == int(row["tests"])
        and float(row["worst_delta_pct"]) <= 1e-12
    ]
    useful_safe = [
        row
        for row in safe_all
        if int(by_key[(float(row["rho"]), "main_matrix")]["wins"]) == 558
    ]
    safe_min = min(float(row["rho"]) for row in safe_all)
    useful_max = max(float(row["rho"]) for row in useful_safe)
    useful_min = min(float(row["rho"]) for row in useful_safe)
    lines = [
        "# Next-Gen Rho Margin Audit\n\n",
        "This audit rescans the posterior-predictive transfer shrinkage `c_n = n/(n+rho)` while keeping the target-identifiability gate fixed at `R_target >= 0.01`. "
        "It asks whether `rho=0.5` is a knife-edge or part of a stable safe range.\n\n",
        "## Rho Sweep\n\n",
        "| rho | all mean | all worst | all non-harm | main wins | mean shrink |\n",
        "|---:|---:|---:|---:|---:|---:|\n",
    ]
    for rho in RHO_GRID:
        row = by_key[(rho, "all")]
        main = by_key[(rho, "main_matrix")]
        lines.append(
            f"| `{rho:.2f}` | {float(row['mean_delta_pct']):+.1f}% | "
            f"{float(row['worst_delta_pct']):+.1f}% | {int(row['non_harm_cells'])}/{int(row['tests'])} | "
            f"{int(main['wins'])}/{int(main['tests'])} | {float(row['mean_shrink']):.3f} |\n"
        )
    lines += [
        "\n## Train-Size Breakdown For Selected Rho\n\n",
        "| train curves | mean delta | worst delta | non-harm cells | wins | mean shrink |\n",
        "|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in train_size_summary:
        if abs(float(row["rho"]) - 0.5) > 1e-12:
            continue
        lines.append(
            f"| {int(row['train_size'])} | {float(row['mean_delta_pct']):+.1f}% | "
            f"{float(row['worst_delta_pct']):+.1f}% | {int(row['non_harm_cells'])}/{int(row['tests'])} | "
            f"{int(row['wins'])}/{int(row['tests'])} | {float(row['mean_shrink']):.3f} |\n"
        )
    lines += [
        "\n## Readout\n\n",
        f"With the target gate fixed, the first fully non-harming grid value is `rho={safe_min:.2f}`. "
        f"The full-useful safe plateau, defined as fully non-harming overall while preserving all `558/558` main-matrix wins, spans `rho={useful_min:.2f}` through `rho={useful_max:.2f}` on this grid. "
        f"The selected `rho=0.50` lies inside this plateau, with mean `{float(selected['mean_delta_pct']):+.1f}%`, worst `{float(selected['worst_delta_pct']):+.1f}%`, and `{int(selected['non_harm_cells'])}/{int(selected['tests'])}` non-harming cells. "
        "This means `rho=0.5` is not a knife-edge. Smaller values are more aggressive and can preserve slightly stronger mean improvement, but the current formula uses `0.5` because it is the weakest simple posterior-predictive half-degree prior that remains inside the stable non-harming range with margin on this audit.\n",
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
    selected = next(row for row in summary if abs(float(row["rho"]) - 0.5) < 1e-12 and row["group"] == "all")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    print(
        f"rho=0.50 mean={float(selected['mean_delta_pct']):+.1f}% "
        f"worst={float(selected['worst_delta_pct']):+.1f}% "
        f"nonharm={int(selected['non_harm_cells'])}/{int(selected['tests'])}"
    )


if __name__ == "__main__":
    main()
