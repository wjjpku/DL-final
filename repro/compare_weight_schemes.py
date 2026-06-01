#!/usr/bin/env python3
"""Compare official MPL under multiple train-curve weighting ratios."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from weighted_mpl_official import (  # type: ignore
    ROOT,
    SCHEMES,
    TEST_SET,
    TRAIN_SET,
    FOLDER_PATHS,
    MPL_ROOT,
    load_data,
    generate_init_params,
    initialize_params_weighted,
    mpl_adam_fit_weighted,
    predict_curve,
)


OUT_ROOT = ROOT / "results" / "weighted_scheme_compare"
DOC_PATH = ROOT / "docs" / "weighted_scheme_compare.md"
SCHEME_NAMES = [
    "equal_1_1_1",
    "constant_wsdcon_1_3_3",
    "constant_wsdcon_1_4_4",
    "constant_wsdcon_1_2_2",
    "constant_wsdcon_1_2_4",
    "constant_wsdcon_1_4_2",
]
LABELS = {
    "equal_1_1_1": "111",
    "constant_wsdcon_1_3_3": "133",
    "constant_wsdcon_1_4_4": "144",
    "constant_wsdcon_1_2_2": "122",
    "constant_wsdcon_1_2_4": "124",
    "constant_wsdcon_1_4_2": "142",
}
COLORS = {
    "equal_1_1_1": "#4C78A8",
    "constant_wsdcon_1_3_3": "#E45756",
    "constant_wsdcon_1_4_4": "#72B7B2",
    "constant_wsdcon_1_2_2": "#F58518",
    "constant_wsdcon_1_2_4": "#54A24B",
    "constant_wsdcon_1_4_2": "#B279A2",
}


def evaluate_curve(data: dict, file_name: str, params: list[float]) -> dict[str, float]:
    pred = predict_curve(data, file_name, params)
    loss = data[file_name]["loss"]
    err = loss - pred
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": float(np.mean(np.abs(err) / loss)),
    }


def save_overlay_plot(scale: str, file_name: str, data: dict, params_map: dict[str, list[float]]) -> None:
    step = data[file_name]["step"]
    loss = data[file_name]["loss"]
    out_dir = OUT_ROOT / f"{scale}M" / "curve_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9.0, 5.2))
    plt.plot(step, loss, color="#111111", linewidth=2.4, label="GT")
    for scheme in SCHEME_NAMES:
        pred = predict_curve(data, file_name, params_map[scheme])
        plt.plot(
            step,
            pred,
            color=COLORS[scheme],
            linewidth=1.8,
            linestyle="--" if scheme == "equal_1_1_1" else "-",
            label=LABELS[scheme],
        )
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"{scale}M {file_name.replace('.csv', '')}")
    plt.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / f"{file_name.replace('.csv', '')}_compare.png", dpi=180)
    plt.close()


def save_scale_bar(scale: str, rows: list[dict[str, str | float]]) -> None:
    curves = TEST_SET
    x = np.arange(len(curves))
    width = 0.12
    plt.figure(figsize=(12.0, 5.0))
    for idx, scheme in enumerate(SCHEME_NAMES):
        values = []
        for curve in curves:
            row = next(r for r in rows if r["scale"] == scale and r["scheme"] == scheme and r["curve"] == curve)
            values.append(float(row["mae"]))
        plt.bar(x + (idx - (len(SCHEME_NAMES) - 1) / 2) * width, values, width=width, color=COLORS[scheme], label=LABELS[scheme])
    plt.xticks(x, [c.replace(".csv", "") for c in curves], rotation=22)
    plt.ylabel("MAE")
    plt.title(f"{scale}M Test-Curve Error Compare")
    plt.legend(ncol=3)
    plt.tight_layout()
    out_path = OUT_ROOT / "figures" / f"{scale}M_test_curve_mae_compare.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_overall_bar(summary_rows: list[dict[str, str | float]]) -> None:
    scales = ["25", "100", "400"]
    x = np.arange(len(scales))
    width = 0.12
    plt.figure(figsize=(9.4, 5.0))
    for idx, scheme in enumerate(SCHEME_NAMES):
        values = [float(next(r for r in summary_rows if r["scale"] == s and r["scheme"] == scheme)["avg_test_mae"]) for s in scales]
        plt.bar(x + (idx - (len(SCHEME_NAMES) - 1) / 2) * width, values, width=width, color=COLORS[scheme], label=LABELS[scheme])
    plt.xticks(x, [f"{s}M" for s in scales])
    plt.ylabel("Average Test MAE")
    plt.title("Weight-Scheme Compare Across Scales")
    plt.legend(ncol=3)
    plt.tight_layout()
    out_path = OUT_ROOT / "figures" / "avg_test_mae_all_schemes.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def write_report(summary_rows: list[dict[str, str | float]], ranking_rows: list[dict[str, str | float]]) -> None:
    lines = [
        "# 多比例训练权重方案测试集对比",
        "",
        "本实验比较如下方案，顺序均对应训练集 `[cosine_24000, constant_24000, wsdcon_9]`：",
        "",
        "- `111`: 等权",
        "- `133`: 降低 `cosine_24000`，提高 `constant_24000` 与 `wsdcon_9`",
        "- `144`",
        "- `122`",
        "- `124`",
        "- `142`",
        "",
        "所有方案都做了归一化，总权重保持一致，其余设置完全沿用官方 `MPL`。",
        "",
        "## 跨尺度总排名",
        "",
        "| Rank | Scheme | Avg Test MAE |",
        "| --- | --- | ---: |",
    ]
    for idx, row in enumerate(ranking_rows, start=1):
        lines.append(f"| {idx} | {LABELS[row['scheme']]} | {float(row['overall_avg_test_mae']):.6f} |")
    lines += [
        "",
        "## 分尺度平均测试 MAE",
        "",
        "| Scale | Scheme | Avg Test MAE |",
        "| --- | --- | ---: |",
    ]
    for scale in ["25", "100", "400"]:
        for scheme in SCHEME_NAMES:
            row = next(r for r in summary_rows if r["scale"] == scale and r["scheme"] == scheme)
            lines.append(f"| {scale}M | {LABELS[scheme]} | {float(row['avg_test_mae']):.6f} |")
    lines += [
        "",
        "## 输出目录",
        "",
        "- `results/weighted_scheme_compare/tables/curve_metrics.csv`",
        "- `results/weighted_scheme_compare/tables/summary.csv`",
        "- `results/weighted_scheme_compare/tables/best_params.json`",
        "- `results/weighted_scheme_compare/figures/avg_test_mae_all_schemes.png`",
        "- `results/weighted_scheme_compare/figures/*_test_curve_mae_compare.png`",
        "- `results/weighted_scheme_compare/<scale>M/curve_compare/*.png`",
        "",
    ]
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    scheme_params: dict[str, dict[str, list[float]]] = {scheme: {} for scheme in SCHEME_NAMES}
    curve_rows: list[dict[str, str | float]] = []
    summary_rows: list[dict[str, str | float]] = []

    for scale in ["25", "100", "400"]:
        data = load_data(str(MPL_ROOT / FOLDER_PATHS[scale].replace("./", "")))
        params_map: dict[str, list[float]] = {}
        for scheme in SCHEME_NAMES:
            weights = SCHEMES[scheme]
            init_param = initialize_params_weighted(data, TRAIN_SET, weights)
            best_params, best_loss = mpl_adam_fit_weighted(
                data=data,
                train_set=TRAIN_SET,
                init_params=generate_init_params(init_param),
                weights=weights,
                fig_folder=OUT_ROOT / f"{scale}M" / scheme,
            )
            params_map[scheme] = best_params
            scheme_params[scheme][scale] = best_params
            scheme_rows = []
            for curve in TEST_SET:
                metrics = evaluate_curve(data, curve, best_params)
                row = {
                    "scale": scale,
                    "scheme": scheme,
                    "curve": curve,
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "mape": metrics["mape"],
                    "best_loss": float(best_loss),
                }
                curve_rows.append(row)
                scheme_rows.append(row)
            summary_rows.append(
                {
                    "scale": scale,
                    "scheme": scheme,
                    "avg_test_mae": float(np.mean([float(r["mae"]) for r in scheme_rows])),
                    "avg_test_rmse": float(np.mean([float(r["rmse"]) for r in scheme_rows])),
                    "avg_test_mape": float(np.mean([float(r["mape"]) for r in scheme_rows])),
                }
            )

        for curve in TEST_SET:
            save_overlay_plot(scale, curve, data, params_map)
        save_scale_bar(scale, curve_rows)

    ranking_rows = []
    for scheme in SCHEME_NAMES:
        subset = [r for r in summary_rows if r["scheme"] == scheme]
        ranking_rows.append(
            {
                "scheme": scheme,
                "overall_avg_test_mae": float(np.mean([float(r["avg_test_mae"]) for r in subset])),
            }
        )
    ranking_rows.sort(key=lambda r: float(r["overall_avg_test_mae"]))

    (OUT_ROOT / "tables").mkdir(parents=True, exist_ok=True)
    with (OUT_ROOT / "tables" / "curve_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["scale", "scheme", "curve", "mae", "rmse", "mape", "best_loss"])
        writer.writeheader()
        for row in curve_rows:
            writer.writerow(row)

    with (OUT_ROOT / "tables" / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["scale", "scheme", "avg_test_mae", "avg_test_rmse", "avg_test_mape"])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    with (OUT_ROOT / "tables" / "overall_ranking.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["scheme", "overall_avg_test_mae"])
        writer.writeheader()
        for row in ranking_rows:
            writer.writerow(row)

    with (OUT_ROOT / "tables" / "best_params.json").open("w", encoding="utf-8") as fh:
        json.dump(scheme_params, fh, indent=2)

    save_overall_bar(summary_rows)
    write_report(summary_rows, ranking_rows)
    print(f"Finished weight scheme comparison under {OUT_ROOT}")


if __name__ == "__main__":
    main()
