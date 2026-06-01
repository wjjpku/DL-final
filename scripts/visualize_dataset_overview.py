#!/usr/bin/env python3
"""Summarize and visualize the public MultiPowerLaw curve dataset."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "external" / "MultiPowerLaw" / "loss_curve_repo"
OUT_ROOT = ROOT / "results" / "dataset_overview"

SCALES = ["25", "100", "400"]
CURVES = [
    "constant_24000.csv",
    "constant_72000.csv",
    "cosine_24000.csv",
    "cosine_72000.csv",
    "wsd_20000_24000.csv",
    "wsdld_20000_24000.csv",
    "wsdcon_3.csv",
    "wsdcon_9.csv",
    "wsdcon_18.csv",
]

COLORS = {
    "constant_24000.csv": "#4C78A8",
    "constant_72000.csv": "#72B7B2",
    "cosine_24000.csv": "#F58518",
    "cosine_72000.csv": "#E45756",
    "wsd_20000_24000.csv": "#54A24B",
    "wsdld_20000_24000.csv": "#EECA3B",
    "wsdcon_3.csv": "#B279A2",
    "wsdcon_9.csv": "#FF9DA6",
    "wsdcon_18.csv": "#9D755D",
}


def load_curve(scale: str, file_name: str) -> dict[str, np.ndarray]:
    raw = np.genfromtxt(DATA_ROOT / f"csv_{scale}" / file_name, delimiter=",", skip_header=1)
    return {
        "step": raw[:, 0].astype(int),
        "lr": raw[:, 1].astype(float),
        "loss": raw[:, 2].astype(float),
    }


def write_summary(rows: list[dict[str, object]]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUT_ROOT / "dataset_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "scale",
                "curve",
                "points",
                "step_min",
                "step_max",
                "lr_min",
                "lr_max",
                "loss_start",
                "loss_end",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_scale_overview(scale: str) -> None:
    curves = {name: load_curve(scale, name) for name in CURVES}
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=False)

    for name in CURVES:
        item = curves[name]
        label = name.replace(".csv", "")
        color = COLORS[name]
        axes[0].plot(item["step"], item["loss"], label=label, linewidth=2.0, color=color)
        axes[1].plot(item["step"], item["lr"], label=label, linewidth=2.0, color=color)

    axes[0].set_title(f"{scale}M: Loss Curves")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)

    axes[1].set_title(f"{scale}M: Learning-Rate Schedules")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("LR")
    axes[1].grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT_ROOT / f"{scale}M_overview.png", dpi=180)
    plt.close(fig)


def plot_dataset_structure(rows: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.6))
    x = np.arange(len(CURVES))
    width = 0.24

    for idx, scale in enumerate(SCALES):
        subset = [row for row in rows if row["scale"] == scale]
        subset_map = {row["curve"]: row for row in subset}
        values = [int(subset_map[curve]["points"]) for curve in CURVES]
        ax.bar(x + (idx - 1) * width, values, width=width, label=f"{scale}M")

    ax.set_xticks(x)
    ax.set_xticklabels([curve.replace(".csv", "") for curve in CURVES], rotation=30, ha="right")
    ax.set_ylabel("Number of Sampled Points")
    ax.set_title("Public Curve Dataset Structure")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "dataset_structure.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for scale in SCALES:
        for curve in CURVES:
            item = load_curve(scale, curve)
            rows.append(
                {
                    "scale": scale,
                    "curve": curve,
                    "points": int(len(item["step"])),
                    "step_min": int(item["step"].min()),
                    "step_max": int(item["step"].max()),
                    "lr_min": float(item["lr"].min()),
                    "lr_max": float(item["lr"].max()),
                    "loss_start": float(item["loss"][0]),
                    "loss_end": float(item["loss"][-1]),
                }
            )
        plot_scale_overview(scale)

    write_summary(rows)
    plot_dataset_structure(rows)
    print(f"Wrote dataset overview to {OUT_ROOT}")


if __name__ == "__main__":
    main()
