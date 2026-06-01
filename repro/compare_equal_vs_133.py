#!/usr/bin/env python3
"""Compare equal weighting vs 1-3-3 weighting on official MPL."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from weighted_mpl_official import (  # type: ignore
    OUT_ROOT as WEIGHTED_OUT_ROOT,
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


OUT_ROOT = ROOT / "results" / "weighted_compare_equal_vs_133"
DOC_PATH = ROOT / "docs" / "weighted_compare_equal_vs_133.md"
SCHEME_A = "equal_1_1_1"
SCHEME_B = "constant_wsdcon_1_3_3"


def evaluate_curve(data: dict, file_name: str, params: list[float]) -> dict[str, float]:
    pred = predict_curve(data, file_name, params)
    loss = data[file_name]["loss"]
    err = loss - pred
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": float(np.mean(np.abs(err) / loss)),
    }


def save_overlay_plot(scale: str, file_name: str, data: dict, params_a: list[float], params_b: list[float]) -> None:
    step = data[file_name]["step"]
    loss = data[file_name]["loss"]
    pred_a = predict_curve(data, file_name, params_a)
    pred_b = predict_curve(data, file_name, params_b)
    out_dir = OUT_ROOT / f"{scale}M" / "curve_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.4, 4.8))
    plt.plot(step, loss, color="#222222", linewidth=2.2, label="Ground Truth")
    plt.plot(step, pred_a, color="#4C78A8", linestyle="--", linewidth=2.0, label="Equal 1-1-1")
    plt.plot(step, pred_b, color="#E45756", linestyle="-.", linewidth=2.0, label="1-3-3")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"{scale}M {file_name.replace('.csv', '')}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{file_name.replace('.csv', '')}_compare.png", dpi=180)
    plt.close()


def save_scale_bar(scale: str, rows: list[dict[str, str | float]]) -> None:
    curves = TEST_SET
    mae_equal = []
    mae_133 = []
    for curve in curves:
        row_a = next(r for r in rows if r["scale"] == scale and r["scheme"] == SCHEME_A and r["curve"] == curve)
        row_b = next(r for r in rows if r["scale"] == scale and r["scheme"] == SCHEME_B and r["curve"] == curve)
        mae_equal.append(float(row_a["mae"]))
        mae_133.append(float(row_b["mae"]))
    x = np.arange(len(curves))
    width = 0.38
    plt.figure(figsize=(11.2, 4.8))
    plt.bar(x - width / 2, mae_equal, width=width, color="#4C78A8", label="Equal 1-1-1")
    plt.bar(x + width / 2, mae_133, width=width, color="#E45756", label="1-3-3")
    plt.xticks(x, [c.replace(".csv", "") for c in curves], rotation=22)
    plt.ylabel("MAE")
    plt.title(f"{scale}M Test-Curve Error Compare")
    plt.legend()
    plt.tight_layout()
    out_path = OUT_ROOT / "figures" / f"{scale}M_test_curve_mae_compare.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_overall_bar(summary_rows: list[dict[str, str | float]]) -> None:
    scales = ["25", "100", "400"]
    equal_vals = [float(next(r for r in summary_rows if r["scale"] == s and r["scheme"] == SCHEME_A)["avg_test_mae"]) for s in scales]
    one_three_three_vals = [float(next(r for r in summary_rows if r["scale"] == s and r["scheme"] == SCHEME_B)["avg_test_mae"]) for s in scales]
    x = np.arange(len(scales))
    width = 0.38
    plt.figure(figsize=(8.2, 4.8))
    plt.bar(x - width / 2, equal_vals, width=width, color="#4C78A8", label="Equal 1-1-1")
    plt.bar(x + width / 2, one_three_three_vals, width=width, color="#E45756", label="1-3-3")
    plt.xticks(x, [f"{s}M" for s in scales])
    plt.ylabel("Average Test MAE")
    plt.title("Equal vs 1-3-3 Across Scales")
    plt.legend()
    plt.tight_layout()
    out_path = OUT_ROOT / "figures" / "avg_test_mae_equal_vs_133.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def write_report(summary_rows: list[dict[str, str | float]]) -> None:
    lines = [
        "# 等权与 1-3-3 方案测试集对比",
        "",
        "本实验只对比两种方案：",
        "",
        "- `equal_1_1_1`：训练集三条曲线等权",
        "- `constant_wsdcon_1_3_3`：降低 `cosine_24000` 权重，提高 `constant_24000` 与 `wsdcon_9` 权重",
        "",
        "其余设置全部保持与官方 `MPL` 一致。",
        "",
        "## 平均测试 MAE",
        "",
        "| Scale | Equal 1-1-1 | 1-3-3 | Better |",
        "| --- | ---: | ---: | --- |",
    ]
    for scale in ["25", "100", "400"]:
        equal_val = float(next(r for r in summary_rows if r["scale"] == scale and r["scheme"] == SCHEME_A)["avg_test_mae"])
        one_three_three_val = float(next(r for r in summary_rows if r["scale"] == scale and r["scheme"] == SCHEME_B)["avg_test_mae"])
        better = "1-3-3" if one_three_three_val < equal_val else "Equal 1-1-1"
        lines.append(f"| {scale}M | {equal_val:.6f} | {one_three_three_val:.6f} | {better} |")
    lines += [
        "",
        "## 输出目录",
        "",
        "- `results/weighted_compare_equal_vs_133/tables/equal_vs_133_curve_metrics.csv`",
        "- `results/weighted_compare_equal_vs_133/tables/equal_vs_133_summary.csv`",
        "- `results/weighted_compare_equal_vs_133/tables/equal_vs_133_best_params.json`",
        "- `results/weighted_compare_equal_vs_133/figures/avg_test_mae_equal_vs_133.png`",
        "- `results/weighted_compare_equal_vs_133/figures/*_test_curve_mae_compare.png`",
        "- `results/weighted_compare_equal_vs_133/<scale>M/curve_compare/*.png`",
        "",
    ]
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    scheme_params: dict[str, dict[str, list[float]]] = {SCHEME_A: {}, SCHEME_B: {}}
    curve_rows: list[dict[str, str | float]] = []
    summary_rows: list[dict[str, str | float]] = []

    for scale in ["25", "100", "400"]:
        data = load_data(str(MPL_ROOT / FOLDER_PATHS[scale].replace("./", "")))
        scale_test_rows = []
        for scheme in [SCHEME_A, SCHEME_B]:
            weights = SCHEMES[scheme]
            init_param = initialize_params_weighted(data, TRAIN_SET, weights)
            best_params, best_loss = mpl_adam_fit_weighted(
                data=data,
                train_set=TRAIN_SET,
                init_params=generate_init_params(init_param),
                weights=weights,
                fig_folder=OUT_ROOT / f"{scale}M" / scheme,
            )
            scheme_params[scheme][scale] = best_params
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
                scale_test_rows.append(row)

        for curve in TEST_SET:
            save_overlay_plot(scale, curve, data, scheme_params[SCHEME_A][scale], scheme_params[SCHEME_B][scale])
        save_scale_bar(scale, curve_rows)

        for scheme in [SCHEME_A, SCHEME_B]:
            subset = [r for r in scale_test_rows if r["scheme"] == scheme]
            summary_rows.append(
                {
                    "scale": scale,
                    "scheme": scheme,
                    "avg_test_mae": float(np.mean([float(r["mae"]) for r in subset])),
                    "avg_test_rmse": float(np.mean([float(r["rmse"]) for r in subset])),
                    "avg_test_mape": float(np.mean([float(r["mape"]) for r in subset])),
                }
            )

    (OUT_ROOT / "tables").mkdir(parents=True, exist_ok=True)
    with (OUT_ROOT / "tables" / "equal_vs_133_curve_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["scale", "scheme", "curve", "mae", "rmse", "mape", "best_loss"])
        writer.writeheader()
        for row in curve_rows:
            writer.writerow(row)

    with (OUT_ROOT / "tables" / "equal_vs_133_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["scale", "scheme", "avg_test_mae", "avg_test_rmse", "avg_test_mape"])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    with (OUT_ROOT / "tables" / "equal_vs_133_best_params.json").open("w", encoding="utf-8") as fh:
        json.dump(scheme_params, fh, indent=2)

    save_overall_bar(summary_rows)
    write_report(summary_rows)
    print(f"Finished equal vs 1-3-3 comparison under {OUT_ROOT}")


if __name__ == "__main__":
    main()
