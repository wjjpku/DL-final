#!/usr/bin/env python3
"""Package full paper-style reproduction outputs into one unified results directory."""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "paper_reproduction"
MPL_ROOT = ROOT / "external" / "MultiPowerLaw"
COMPARE_ROOT = ROOT / "results" / "official_compare"

SCALES = ["25", "100", "400"]
TRAIN_CURVES = ["cosine_24000", "constant_24000", "wsdcon_9"]
TEST_CURVES = [
    "constant_72000",
    "cosine_72000",
    "wsd_20000_24000",
    "wsdld_20000_24000",
    "wsdcon_3",
    "wsdcon_18",
]


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def package_mpl_only() -> None:
    mpl_out = OUT / "mpl_only"
    for scale in SCALES:
        fit_dir = MPL_ROOT / f"{scale}M" / "fit"
        scale_out = mpl_out / f"{scale}M"
        scale_out.mkdir(parents=True, exist_ok=True)
        for png in fit_dir.glob("*.png"):
            copy_file(png, scale_out / png.name)


def package_compare() -> None:
    compare_out = OUT / "mpl_vs_tissue_compare"
    src_dir = COMPARE_ROOT / "figures" / "compare"
    for png in src_dir.glob("*.png"):
        copy_file(png, compare_out / png.name)
    copy_file(
        COMPARE_ROOT / "figures" / "official_avg_test_mae_compare.png",
        compare_out / "official_avg_test_mae_compare.png",
    )
    copy_file(
        COMPARE_ROOT / "tables" / "official_tissue_mpl_metrics.csv",
        compare_out / "official_tissue_mpl_metrics.csv",
    )
    copy_file(
        COMPARE_ROOT / "tables" / "official_tissue_mpl_params.json",
        compare_out / "official_tissue_mpl_params.json",
    )


def write_index() -> None:
    index = OUT / "TASK_INDEX.md"
    lines: list[str] = []
    lines.append("# 论文口径完整预测任务索引")
    lines.append("")
    lines.append("## 训练任务")
    lines.append("")
    for curve in TRAIN_CURVES:
        lines.append(f"- `{curve}`")
    lines.append("")
    lines.append("## 测试任务")
    lines.append("")
    for curve in TEST_CURVES:
        lines.append(f"- `{curve}`")
    lines.append("")
    lines.append("## 结果目录")
    lines.append("")
    lines.append(f"- `MPL` 官方原始预测图：`{(OUT / 'mpl_only').relative_to(ROOT)}`")
    lines.append(f"- `MPL + Tissue + Ground Truth` 对比图：`{(OUT / 'mpl_vs_tissue_compare').relative_to(ROOT)}`")
    lines.append("")
    lines.append("## 覆盖范围")
    lines.append("")
    lines.append("- 尺度：`25M / 100M / 400M`")
    lines.append("- 每个尺度 9 个任务：3 个训练任务 + 6 个测试任务")
    lines.append("- `MPL` 原始图：共 27 张任务图，另含 3 张 `loss_monitor`")
    lines.append("- `MPL vs Tissue` 对比图：共 27 张任务图，另含 1 张平均测试 MAE 汇总图")
    lines.append("")
    lines.append("## 文件命名")
    lines.append("")
    lines.append("- `mpl_only/25M/constant_72000_mplfit.png`：官方 `MPL` 在 `25M` 上对 `constant_72000` 的预测图")
    lines.append("- `mpl_vs_tissue_compare/25_compare_constant_72000.png`：`25M` 上 `Ground Truth + MPL + Tissue` 对比图")
    index.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    reset_dir(OUT)
    package_mpl_only()
    package_compare()
    write_index()
    print(OUT)


if __name__ == "__main__":
    main()
