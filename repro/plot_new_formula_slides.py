#!/usr/bin/env python3
"""Generate slide figures for the cosine-to-WSD formula deck."""
from __future__ import annotations

import csv
import os
from pathlib import Path

TMP_CACHE = Path(os.environ.get("TMPDIR", "/tmp")) / "dl-final-cache"
TMP_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(TMP_CACHE / "matplotlib")
os.environ["XDG_CACHE_HOME"] = str(TMP_CACHE / "xdg")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "slides" / "figs"
RESULT_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "joint_curvature"
DATA_DIR = ROOT / "external" / "MultiPowerLaw" / "loss_curve_repo" / "csv_100"
PEAK_LR = 3e-4
SELECTED_JOINT_CONFIG = "15620"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_curve(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = read_rows(DATA_DIR / name)
    step = np.asarray([float(row["step"]) for row in rows], dtype=np.float64)
    lr = np.asarray([float(row["lr"]) for row in rows], dtype=np.float64)
    loss = np.asarray([float(row["loss"]) for row in rows], dtype=np.float64)
    return step, lr, loss


def response_feature(lr: np.ndarray, response_lambda: float) -> np.ndarray:
    a = 0.0
    out = np.zeros_like(lr)
    prev = lr[0]
    for i, eta in enumerate(lr):
        drop = max(prev - eta, 0.0) if i else 0.0
        a = np.exp(-response_lambda * eta) * a + drop
        out[i] = a / PEAK_LR
        prev = eta
    return out


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.04)
    plt.close()


def plot_schedule_features() -> None:
    curves = [
        ("cosine_72000.csv", "cosine"),
        ("wsd_20000_24000.csv", "WSD sharp"),
        ("wsdcon_3.csv", "WSD-con q=0.1"),
        ("wsdcon_9.csv", "WSD-con q=0.3"),
        ("wsdcon_18.csv", "WSD-con q=0.6"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(9.4, 5.0), sharex=False, constrained_layout=True)
    colors = ["#1f2937", "#2563eb", "#dc2626", "#7c3aed", "#059669"]
    for (name, label), color in zip(curves, colors):
        step, lr, _ = read_curve(name)
        axes[0].plot(step / 1000.0, lr / PEAK_LR, lw=1.7, color=color, label=label)
        lam = 4.0 if "wsd_" in name and "con" not in name else 20.0
        if name == "cosine_72000.csv":
            lam = 4.0
        phi = response_feature(lr, lam)
        axes[1].plot(step / 1000.0, phi, lw=1.7, color=color, label=label)
    axes[0].set_ylabel("LR / peak")
    axes[1].set_ylabel("response feature")
    axes[1].set_xlabel("training step (k)")
    axes[0].set_title("Schedules used by the correction")
    axes[1].set_title("Causal LR-drop response features")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(labelsize=8)
    axes[0].legend(ncol=3, fontsize=8, frameon=False)
    savefig(OUT_DIR / "fig_new_formula_schedule_features.png")


def plot_per_target_results() -> None:
    rows = read_rows(RESULT_DIR / "best_target_summary.csv")
    labels = [row["test_label"].replace("WSD-con ", "con ") for row in rows]
    mean = np.asarray([float(row["mean_delta"]) for row in rows])
    worst = np.asarray([float(row["worst_delta"]) for row in rows])
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.6, 3.8), constrained_layout=True)
    ax.barh(y + 0.18, -mean, height=0.32, color="#2563eb", label="mean MAE reduction")
    ax.barh(y - 0.18, -worst, height=0.32, color="#93c5fd", label="worst-scale reduction")
    ax.set_yticks(y, labels)
    ax.set_xlabel("MAE reduction vs MPL (%)")
    ax.set_title("Current core model: per-target improvement")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    for i, value in enumerate(-mean):
        ax.text(value + 0.7, i + 0.18, f"{value:.1f}%", va="center", fontsize=8)
    for i, value in enumerate(-worst):
        ax.text(value + 0.7, i - 0.18, f"{value:.1f}%", va="center", fontsize=8)
    savefig(OUT_DIR / "fig_new_formula_per_target.png")


def plot_delta_heatmap() -> None:
    rows = [
        row
        for row in read_rows(RESULT_DIR / "top_safe_details.csv")
        if row.get("config_id") == SELECTED_JOINT_CONFIG
    ]
    scales = ["25", "100", "400"]
    targets = ["WSD sharp", "WSD linear", "WSD-con 3e-5", "WSD-con 9e-5", "WSD-con 18e-5"]
    data = np.zeros((len(scales), len(targets)))
    for row in rows:
        i = scales.index(row["scale"])
        j = targets.index(row["test_label"])
        data[i, j] = float(row["delta_pct"])
    fig, ax = plt.subplots(figsize=(9.6, 3.0), constrained_layout=True)
    im = ax.imshow(-data, cmap="Blues", vmin=0, vmax=max(65.0, float((-data).max())))
    ax.set_xticks(range(len(targets)), targets, rotation=20, ha="right")
    ax.set_yticks(range(len(scales)), [f"{s}M" for s in scales])
    ax.set_title("Scale-target MAE reduction vs MPL")
    for i in range(len(scales)):
        for j in range(len(targets)):
            ax.text(j, i, f"{-data[i, j]:.1f}%", ha="center", va="center", fontsize=8, color="#111827")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("MAE reduction (%)", fontsize=8)
    savefig(OUT_DIR / "fig_new_formula_heatmap.png")


def plot_ablation_path() -> None:
    names = [
        "fit window",
        "shrink",
        "decouple",
        "curvature",
        "joint curv.",
    ]
    mean = np.array([-34.53, -35.07, -36.18, -37.47, -37.53])
    worst = np.array([-6.08, -6.12, -6.29, -9.43, -10.80])
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9.4, 3.4), constrained_layout=True)
    ax.plot(x, -mean, marker="o", lw=2.0, color="#2563eb", label="mean reduction")
    ax.plot(x, -worst, marker="o", lw=2.0, color="#dc2626", label="worst-row reduction")
    ax.set_xticks(x, names, rotation=18, ha="right")
    ax.set_ylabel("MAE reduction vs MPL (%)")
    ax.set_title("Ablation path: each component improves stability or mean accuracy")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    savefig(OUT_DIR / "fig_new_formula_ablation.png")


def main() -> None:
    plot_schedule_features()
    plot_per_target_results()
    plot_delta_heatmap()
    plot_ablation_path()
    print(f"wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
