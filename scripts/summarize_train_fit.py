#!/usr/bin/env python3
"""Summarize train-set fit quality for official MPL/Tissue comparison."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "paper_reproduction" / "mpl_vs_tissue_compare" / "official_tissue_mpl_metrics.csv"
FIG_PATH = ROOT / "results" / "paper_reproduction" / "mpl_vs_tissue_compare" / "official_avg_train_mae_compare.png"
REPORT_PATH = ROOT / "docs" / "train_fit_check.md"
SCALES = ["25", "100", "400"]


def load_rows() -> list[dict[str, str]]:
    with CSV_PATH.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def avg_metric(rows: list[dict[str, str]], scale: str, split: str, model: str, key: str) -> float:
    subset = [r for r in rows if r["scale"] == scale and r["split"] == split and r["model"] == model]
    return float(np.mean([float(r[key]) for r in subset]))


def save_plot(rows: list[dict[str, str]]) -> None:
    plt.figure(figsize=(8.4, 4.8))
    for model, color in [("mpl", "#F58518"), ("tissue", "#4C78A8")]:
        xs = [int(scale) for scale in SCALES]
        ys = [avg_metric(rows, scale, "train", model, "mae") for scale in SCALES]
        plt.plot(xs, ys, marker="o", label=model.upper(), color=color)
    plt.xticks([25, 100, 400])
    plt.xlabel("Model Size (M)")
    plt.ylabel("Average Train MAE")
    plt.title("Official Split: Train-Set Fit Quality")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=180)
    plt.close()


def write_report(rows: list[dict[str, str]]) -> None:
    lines: list[str] = []
    lines.append("# 训练集拟合能力检查")
    lines.append("")
    lines.append("本报告只看论文口径官方训练集上的拟合效果。训练任务固定为：")
    lines.append("")
    lines.append("- `cosine_24000`")
    lines.append("- `constant_24000`")
    lines.append("- `wsdcon_9`")
    lines.append("")
    lines.append("训练集预测图目录：")
    lines.append("")
    lines.append("- `results/paper_reproduction/mpl_only/{25M,100M,400M}`")
    lines.append("- `results/paper_reproduction/mpl_vs_tissue_compare/` 中的 `*_compare_constant_24000.png`、`*_compare_cosine_24000.png`、`*_compare_wsdcon_9.png`")
    lines.append("")
    lines.append("训练集平均指标如下：")
    lines.append("")
    lines.append("| Scale | MPL Avg MAE | Tissue Avg MAE | MPL Avg RMSE | Tissue Avg RMSE | MPL Avg R2 | Tissue Avg R2 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for scale in SCALES:
        lines.append(
            "| "
            + f"{scale}M | "
            + f"{avg_metric(rows, scale, 'train', 'mpl', 'mae'):.6f} | "
            + f"{avg_metric(rows, scale, 'train', 'tissue', 'mae'):.6f} | "
            + f"{avg_metric(rows, scale, 'train', 'mpl', 'rmse'):.6f} | "
            + f"{avg_metric(rows, scale, 'train', 'tissue', 'rmse'):.6f} | "
            + f"{avg_metric(rows, scale, 'train', 'mpl', 'r2'):.6f} | "
            + f"{avg_metric(rows, scale, 'train', 'tissue', 'r2'):.6f} |"
        )
    lines.append("")
    lines.append("结论：")
    lines.append("")
    lines.append("- 两种方法在训练集上都拟合得很好，`R2` 基本都在 `0.9988` 以上。")
    lines.append("- `25M` 上二者训练拟合接近，`Tissue` 的平均 `MAE` 略低，但 `MPL` 的平均 `RMSE` 略低。")
    lines.append("- `100M` 和 `400M` 上，`Tissue` 在训练集上的平均误差明显更低。")
    lines.append("- 因此，`MPL` 在部分测试任务上更强，并不是因为它单纯更会贴训练集；两者主要差别仍然来自泛化行为。")
    lines.append("")
    lines.append("训练集逐图建议优先看：")
    lines.append("")
    lines.append("- `25_compare_cosine_24000.png`")
    lines.append("- `25_compare_constant_24000.png`")
    lines.append("- `25_compare_wsdcon_9.png`")
    lines.append("- `100_compare_wsdcon_9.png`")
    lines.append("- `400_compare_wsdcon_9.png`")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = load_rows()
    save_plot(rows)
    write_report(rows)
    print(FIG_PATH)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
