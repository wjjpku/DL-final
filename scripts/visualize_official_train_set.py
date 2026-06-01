#!/usr/bin/env python3
"""Visualize the official MPL training set only."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "external" / "MultiPowerLaw" / "loss_curve_repo"
OUT_ROOT = ROOT / "results" / "dataset_overview"

SCALES = ["25", "100", "400"]
TRAIN_CURVES = ["cosine_24000.csv", "constant_24000.csv", "wsdcon_9.csv"]
COLORS = {
    "cosine_24000.csv": "#F58518",
    "constant_24000.csv": "#4C78A8",
    "wsdcon_9.csv": "#B279A2",
}


def load_curve(scale: str, file_name: str) -> dict[str, np.ndarray]:
    raw = np.genfromtxt(DATA_ROOT / f"csv_{scale}" / file_name, delimiter=",", skip_header=1)
    return {
        "step": raw[:, 0].astype(int),
        "lr": raw[:, 1].astype(float),
        "loss": raw[:, 2].astype(float),
    }


def save_summary() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUT_ROOT / "official_train_set_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["scale", "curve", "points", "step_min", "step_max", "lr_min", "lr_max", "loss_start", "loss_end"])
        for scale in SCALES:
            for curve in TRAIN_CURVES:
                item = load_curve(scale, curve)
                writer.writerow(
                    [
                        scale,
                        curve,
                        len(item["step"]),
                        int(item["step"].min()),
                        int(item["step"].max()),
                        float(item["lr"].min()),
                        float(item["lr"].max()),
                        float(item["loss"][0]),
                        float(item["loss"][-1]),
                    ]
                )


def save_figure() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(12, 11))

    for idx, scale in enumerate(SCALES):
        ax_loss = axes[idx, 0]
        ax_lr = axes[idx, 1]
        for curve in TRAIN_CURVES:
            item = load_curve(scale, curve)
            label = curve.replace(".csv", "")
            color = COLORS[curve]
            ax_loss.plot(item["step"], item["loss"], label=label, linewidth=2.2, color=color)
            ax_lr.plot(item["step"], item["lr"], label=label, linewidth=2.2, color=color)

        ax_loss.set_title(f"{scale}M Training Set: Loss")
        ax_loss.set_xlabel("Step")
        ax_loss.set_ylabel("Loss")
        ax_loss.grid(alpha=0.25)

        ax_lr.set_title(f"{scale}M Training Set: LR")
        ax_lr.set_xlabel("Step")
        ax_lr.set_ylabel("LR")
        ax_lr.grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("Official MPL Training Set Only", fontsize=16, y=0.995)
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    fig.savefig(OUT_ROOT / "official_train_set_only.png", dpi=180)
    plt.close(fig)


def main() -> None:
    save_summary()
    save_figure()
    print(OUT_ROOT / "official_train_set_only.png")
    print(OUT_ROOT / "official_train_set_summary.csv")


if __name__ == "__main__":
    main()
