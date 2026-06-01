#!/usr/bin/env python3
"""Plot selected windows from continual schedule CSV outputs."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "results" / "continual_schedule_144k"
SCALE = "400M"
CSV_PATH = BASE_DIR / SCALE / "continual_schedule_compare.csv"
WINDOWS = [
    ("57k_72k", 57_000, 72_000),
    ("129k_144k", 129_000, 144_000),
]


def load_schedule_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps, baseline, optimized = [], [], []
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            steps.append(int(row["step"]))
            baseline.append(float(row["baseline_full_lr"]))
            optimized.append(float(row["optimized_full_lr"]))
    return np.asarray(steps), np.asarray(baseline), np.asarray(optimized)


def save_window_plot(steps: np.ndarray, baseline: np.ndarray, optimized: np.ndarray, label: str, start: int, end: int) -> Path:
    mask = (steps >= start) & (steps <= end)
    window_steps = steps[mask]
    window_baseline = baseline[mask]
    window_optimized = optimized[mask]

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(window_steps, window_baseline, label="Baseline all-min suffix", linewidth=2.0, color="#4C78A8")
    ax.plot(window_steps, window_optimized, label="Optimized suffix", linewidth=2.0, color="#E45756")
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title(f"400M LR Schedule Window: {start//1000}k-{end//1000}k")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    out_path = BASE_DIR / SCALE / f"continual_schedule_window_{label}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> None:
    steps, baseline, optimized = load_schedule_csv(CSV_PATH)
    for label, start, end in WINDOWS:
        out_path = save_window_plot(steps, baseline, optimized, label, start, end)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
