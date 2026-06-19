#!/usr/bin/env python3
"""Compare MPL cosine residual with the DropRelaxS residual estimate.

For each public scale on cosine_72000:
  - MPL residual: r = L_obs - L_MPL
  - Estimated residual: kappa * DropRelaxS(lambda=10), with kappa fit on the
    same full cosine curve through the origin.
  - Remaining error after correction: r - kappa * phi

This is a self-fit diagnostic, not a transfer result.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent
ROOT = REPO.parent
sys.path.insert(0, str(REPO))

from current_law_decay_matrix import LAMBDA  # noqa: E402
from deep_stime import stime_feature  # noqa: E402
from reproduce_cosine_to_wsd import (  # noqa: E402
    MPL_PRECOMPUTED_INIT,
    SCALES,
    load_curve,
    metrics,
    mpl_predict,
)


OUT_DIR = ROOT / "results" / "current_law_decay_matrix" / "error_visualization"
CURVE_NAME = "cosine_72000.csv"


def fit_origin_nonnegative(x: np.ndarray, y: np.ndarray) -> float:
    denom = float(np.dot(x, x))
    if denom <= 1e-18:
        return 0.0
    return max(0.0, float(np.dot(x, y) / denom))


def r2_origin(y: np.ndarray, yhat: np.ndarray) -> float:
    denom = float(np.dot(y, y))
    if denom <= 1e-18:
        return float("nan")
    return 1.0 - float(np.dot(y - yhat, y - yhat) / denom)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - float(np.mean(x))
    yc = y - float(np.mean(y))
    denom = float(np.linalg.norm(xc) * np.linalg.norm(yc))
    if denom <= 1e-18:
        return float("nan")
    return float(np.dot(xc, yc) / denom)


def build_rows() -> list[dict[str, object]]:
    rows = []
    for scale in SCALES:
        curve = load_curve(scale, CURVE_NAME)
        mpl = mpl_predict(MPL_PRECOMPUTED_INIT[scale], curve)
        residual = curve.loss - mpl
        phi = stime_feature(curve, LAMBDA)
        kappa = fit_origin_nonnegative(phi, residual)
        estimate = kappa * phi
        corrected = mpl + estimate
        remaining = curve.loss - corrected
        base_metrics = metrics(curve.loss, mpl)
        corrected_metrics = metrics(curve.loss, corrected)
        rows.append(
            {
                "scale": scale,
                "curve": curve,
                "mpl": mpl,
                "residual": residual,
                "phi": phi,
                "kappa": kappa,
                "estimate": estimate,
                "remaining": remaining,
                "base_mae": base_metrics["mae"],
                "corrected_mae": corrected_metrics["mae"],
                "delta_pct": 100.0 * (corrected_metrics["mae"] / base_metrics["mae"] - 1.0),
                "origin_r2": r2_origin(residual, estimate),
                "pearson": pearson(residual, estimate),
                "mean_residual": float(np.mean(residual)),
                "mean_estimate": float(np.mean(estimate)),
                "mean_remaining": float(np.mean(remaining)),
            }
        )
    return rows


def plot(rows: list[dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(
        2,
        len(rows),
        figsize=(14.6, 6.9),
        sharex=False,
        gridspec_kw={"height_ratios": [1.1, 1.0], "hspace": 0.25, "wspace": 0.25},
    )

    for i, row in enumerate(rows):
        curve = row["curve"]
        steps = curve.step
        residual = row["residual"]
        estimate = row["estimate"]
        remaining = row["remaining"]

        ax_top = axes[0, i]
        ax_bottom = axes[1, i]

        ax_top.axhline(0.0, color="#333333", lw=0.8, alpha=0.7)
        ax_top.plot(steps, residual, color="#111111", lw=1.25, label="MPL residual")
        ax_top.plot(steps, estimate, color="#2563eb", lw=1.25, ls="--", label="our estimated residual")
        ax_top.set_title(
            f"{row['scale']}M cosine self-fit\n"
            f"kappa={float(row['kappa']):.3f}, R2_origin={float(row['origin_r2']):.2f}",
            fontsize=10.5,
        )
        ax_top.set_ylabel("residual")
        ax_top.grid(alpha=0.23, lw=0.5)
        ax_top.tick_params(labelsize=8)
        if i == 0:
            ax_top.legend(frameon=False, fontsize=8.5, loc="best")

        ax_bottom.axhline(0.0, color="#333333", lw=0.8, alpha=0.7)
        ax_bottom.plot(steps, residual, color="#dc2626", lw=1.1, alpha=0.9, label="MPL error")
        ax_bottom.plot(steps, remaining, color="#2563eb", lw=1.15, label="after correction")
        ax_bottom.fill_between(steps, 0.0, remaining, color="#2563eb", alpha=0.12, linewidth=0)
        ax_bottom.set_xlabel("step")
        ax_bottom.set_ylabel("remaining error")
        ax_bottom.grid(alpha=0.23, lw=0.5)
        ax_bottom.tick_params(labelsize=8)
        ax_bottom.text(
            0.03,
            0.95,
            f"MPL MAE={float(row['base_mae']):.5f}\n"
            f"+ours MAE={float(row['corrected_mae']):.5f}\n"
            f"delta={float(row['delta_pct']):+.1f}%",
            transform=ax_bottom.transAxes,
            fontsize=8,
            va="top",
            ha="left",
            bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#cccccc", "alpha": 0.88},
        )
        if i == 0:
            ax_bottom.legend(frameon=False, fontsize=8.5, loc="lower right")

    fig.suptitle(
        "Cosine self-fit: MPL residual vs DropRelaxS estimated residual",
        fontsize=13.5,
        y=0.985,
    )
    fig.text(
        0.5,
        0.012,
        "This is an in-family self-fit diagnostic: kappa is fitted on the same cosine curve. "
        "It should not be interpreted as cross-schedule transfer evidence.",
        ha="center",
        fontsize=9,
        color="#333333",
    )
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def write_outputs(rows: list[dict[str, object]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot(rows, OUT_DIR / "cosine_self_error_comparison.png")

    fields = [
        "scale",
        "kappa",
        "base_mae",
        "corrected_mae",
        "delta_pct",
        "origin_r2",
        "pearson",
        "mean_residual",
        "mean_estimate",
        "mean_remaining",
    ]
    with (OUT_DIR / "cosine_self_error_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})

    lines = [
        "# Cosine Self-Fit Error Comparison\n\n",
        "This compares the MPL residual on `cosine_72000.csv` with `kappa * DropRelaxS(lambda=10)` fitted on the same cosine curve. "
        "It is a self-fit diagnostic, not a transfer result.\n\n",
        "| scale | kappa | MPL MAE | +ours MAE | delta | R2_origin | pearson |\n",
        "|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in rows:
        lines.append(
            f"| {row['scale']}M | {float(row['kappa']):.4f} | "
            f"{float(row['base_mae']):.5f} | {float(row['corrected_mae']):.5f} | "
            f"{float(row['delta_pct']):+.1f}% | {float(row['origin_r2']):.3f} | "
            f"{float(row['pearson']):.3f} |\n"
        )
    lines += [
        "\n## Reading\n\n",
        "- On the cosine curve itself, the correction reduces MPL MAE by about 24--49%, because the fitted DropRelaxS feature tracks a large smooth positive residual component.\n",
        "- The estimated residual is broad and low-frequency, not a localized fast-cooldown transient. This explains why the same raw cosine kappa does not transfer to sharp WSD/WSD-con targets.\n",
    ]
    (OUT_DIR / "COSINE_SELF_ERROR_COMPARISON.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    rows = build_rows()
    write_outputs(rows)
    print(f"wrote {OUT_DIR / 'cosine_self_error_comparison.png'}")
    print(f"wrote {OUT_DIR / 'cosine_self_error_comparison.csv'}")
    print(f"wrote {OUT_DIR / 'COSINE_SELF_ERROR_COMPARISON.md'}")
    for row in rows:
        print(
            f"{row['scale']}M kappa={float(row['kappa']):.4f} "
            f"MPL={float(row['base_mae']):.5f} +ours={float(row['corrected_mae']):.5f} "
            f"delta={float(row['delta_pct']):+.1f}% R2_origin={float(row['origin_r2']):.3f}"
        )


if __name__ == "__main__":
    main()
