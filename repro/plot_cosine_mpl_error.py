#!/usr/bin/env python3
"""Plot MPL prediction errors on the full cosine_72000 curves."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent
ROOT = REPO.parent
sys.path.insert(0, str(REPO))

from reproduce_cosine_to_wsd import (  # noqa: E402
    MPL_PRECOMPUTED_INIT,
    SCALES,
    load_curve,
    metrics,
    mpl_predict,
)


OUT_DIR = ROOT / "results" / "current_law_decay_matrix" / "error_visualization"
CURVE_NAME = "cosine_72000.csv"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        2,
        len(SCALES),
        figsize=(14.0, 6.6),
        sharex=False,
        gridspec_kw={"height_ratios": [1.05, 1.0], "hspace": 0.22, "wspace": 0.25},
    )

    rows = []
    for i, scale in enumerate(SCALES):
        curve = load_curve(scale, CURVE_NAME)
        pred = mpl_predict(MPL_PRECOMPUTED_INIT[scale], curve)
        residual = curve.loss - pred
        scored = metrics(curve.loss, pred)
        rows.append(
            {
                "scale": scale,
                "mae": scored["mae"],
                "rmse": scored["rmse"],
                "mean_residual": float(np.mean(residual)),
                "max_abs_residual": float(np.max(np.abs(residual))),
            }
        )

        ax_top = axes[0, i]
        ax_err = axes[1, i]
        ax_top.plot(curve.step, curve.loss, color="#111111", lw=1.3, label="observed")
        ax_top.plot(curve.step, pred, color="#7a7f87", lw=1.2, ls="--", label="MPL")
        ax_top.set_title(f"{scale}M cosine_72000\nMAE={scored['mae']:.5f}", fontsize=10.5)
        ax_top.set_ylabel("loss")
        ax_top.grid(alpha=0.23, lw=0.5)
        ax_top.tick_params(labelsize=8)
        if i == 0:
            ax_top.legend(frameon=False, fontsize=8.5, loc="best")

        ax_err.axhline(0.0, color="#333333", lw=0.8)
        ax_err.plot(curve.step, residual, color="#dc2626", lw=1.25, label="L_obs - L_MPL")
        ax_err.fill_between(curve.step, 0.0, residual, color="#dc2626", alpha=0.14, linewidth=0)
        ax_err.set_xlabel("step")
        ax_err.set_ylabel("residual")
        ax_err.grid(alpha=0.23, lw=0.5)
        ax_err.tick_params(labelsize=8)
        ax_err.text(
            0.03,
            0.92,
            f"mean={float(np.mean(residual)):+.5f}\nmax |err|={float(np.max(np.abs(residual))):.5f}",
            transform=ax_err.transAxes,
            fontsize=8,
            va="top",
            ha="left",
            bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#cccccc", "alpha": 0.88},
        )

    fig.suptitle("MPL error on full cosine schedule: signed residual L_obs - L_MPL", fontsize=13.5, y=0.985)
    fig.savefig(OUT_DIR / "cosine_mpl_error.png", dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    csv_path = OUT_DIR / "cosine_mpl_error_summary.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("scale,mae,rmse,mean_residual,max_abs_residual\n")
        for row in rows:
            f.write(
                f"{row['scale']},{row['mae']:.10f},{row['rmse']:.10f},"
                f"{row['mean_residual']:.10f},{row['max_abs_residual']:.10f}\n"
            )

    print(f"wrote {OUT_DIR / 'cosine_mpl_error.png'}")
    print(f"wrote {csv_path}")
    for row in rows:
        print(
            f"{row['scale']}M MAE={row['mae']:.5f} "
            f"mean_residual={row['mean_residual']:+.5f} "
            f"max_abs={row['max_abs_residual']:.5f}"
        )


if __name__ == "__main__":
    main()
