#!/usr/bin/env python3
"""Official MPL ablation with reweighted train-curve losses."""

from __future__ import annotations

import csv
import json
import logging
import sys
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
MPL_ROOT = ROOT / "external" / "MultiPowerLaw"
sys.path.insert(0, str(MPL_ROOT))

from src.config import (  # noqa: E402
    FIT_EVAL_INTERVAL,
    FIT_GRAD_NORM_THR,
    FIT_LOSS_THR,
    FIT_LR1,
    FIT_LR2,
    FIT_MAX_STEPS,
    FIT_PATIENCE,
    FOLDER_PATHS,
    TRAIN_SET,
    TEST_SET,
)
from src.data_loader import load_data  # noqa: E402
from src.fitting import generate_init_params  # noqa: E402
from src.models import MPL  # noqa: E402
from src.utils import compute_grad_norm, huber_loss, plot_loss_curve, preprocess_data  # noqa: E402


OUT_ROOT = ROOT / "results" / "weighted_mpl_official"
DOC_PATH = ROOT / "docs" / "weighted_mpl_official_report.md"
DELTA = 0.001
SEED = 0


def build_scheme(raw_weights: list[float]) -> dict[str, float]:
    scale = len(TRAIN_SET) / float(sum(raw_weights))
    return {
        name: float(weight * scale)
        for name, weight in zip(TRAIN_SET, raw_weights)
    }


SCHEMES = {
    "equal_1_1_1": build_scheme([1, 1, 1]),
    "constant_wsdcon_1_2_2": build_scheme([1, 2, 2]),
    "constant_wsdcon_1_2_4": build_scheme([1, 2, 4]),
    "constant_wsdcon_1_3_3": build_scheme([1, 3, 3]),
    "constant_wsdcon_1_4_2": build_scheme([1, 4, 2]),
    "constant_wsdcon_1_4_4": build_scheme([1, 4, 4]),
    "cosine_constant_3_3_1": build_scheme([3, 3, 1]),
    "cosine_wsdcon_3_1_3": build_scheme([3, 1, 3]),
}


def initialize_params_weighted(data: dict, train_set: list[str], weights: dict[str, float]) -> list[float]:
    min_loss = min(data[file_name]["loss"].min() for file_name in train_set)
    log_y_list, log_x_list = [], []

    for file_name in train_set:
        log_y = np.log(data[file_name]["loss"] - min_loss + 0.01)
        log_x = np.log(np.cumsum(data[file_name]["lrs"])[data[file_name]["step"]])
        log_y_list.append(log_y)
        log_x_list.append(log_x)

    log_y = np.concatenate(log_y_list)
    log_x = np.concatenate(log_x_list)
    slope, intercept, _, _, _ = linregress(log_x, log_y)

    L0_init_set = np.linspace(min_loss - 0.2, min_loss + 0.2, 5)
    A_init_set = np.linspace(np.exp(intercept) - 0.1, np.exp(intercept) + 0.1, 3)
    alpha_init_set = np.linspace(-slope - 0.1, -slope + 0.1, 3)
    B_init_set = np.linspace(100, 1000, 3)

    def loss_fn0(params: tuple[float, float, float, float]) -> float:
        L0, A, alpha, B = params
        total_loss = 0.0
        for file_name in train_set:
            lr = data[file_name]["lrs"]
            step = data[file_name]["step"]
            pred = L0 + A * np.cumsum(lr)[step] ** (-alpha) - B * (3e-4 - lr[step])
            if np.any(~np.isfinite(pred)) or np.any(pred <= 0):
                return 1e18
            loss = data[file_name]["loss"]
            r = np.log(loss) - np.log(pred)
            total_loss += weights[file_name] * huber_loss(r).sum()
        return float(total_loss)

    init_params = list(product(L0_init_set, A_init_set, alpha_init_set, B_init_set))
    best_loss = float("inf")
    best_params = None
    for init_param in tqdm(init_params, desc="Initializing Parameters"):
        res = minimize(
            loss_fn0,
            init_param,
            method="L-BFGS-B",
            bounds=[(0, np.inf)] * 4,
            options={"maxiter": 100000, "ftol": 1e-9, "gtol": 1e-6, "eps": 1e-8},
        )
        if res.fun < best_loss:
            best_loss = float(res.fun)
            best_params = res.x
    assert best_params is not None
    return list(best_params)


def compute_loss_weighted(model: MPL, torch_data: dict, train_set: list[str], weights: dict[str, float], optimizer) -> torch.Tensor:
    optimizer.zero_grad()
    total_loss = torch.zeros((), dtype=torch.float64)
    for file_name in train_set:
        args = [torch_data[file_name][key] for key in ["S1", "lrs", "lr_sum", "step", "lr_gap", "loss"]]
        total_loss = total_loss + weights[file_name] * model(*args)
    total_loss.backward()
    optimizer.step()
    return total_loss


def mpl_adam_fit_weighted(
    data: dict,
    train_set: list[str],
    init_params: list[tuple[float, ...]],
    weights: dict[str, float],
    fig_folder: Path,
) -> tuple[list[float], float]:
    torch_data = preprocess_data(data, train_set + TEST_SET)
    best_params, best_loss = None, float("inf")
    loss_history: list[float] = []

    for init_param in init_params:
        model = MPL(*init_param)
        optimizer = torch.optim.AdamW(
            [
                {"params": [model.L0, model.A, model.B, model.C], "lr": FIT_LR1},
                {"params": [model.alpha, model.beta, model.gamma], "lr": FIT_LR2},
            ]
        )
        min_loss = float("inf")
        steps_no_improve = 0
        local_history: list[float] = []

        for step in tqdm(range(FIT_MAX_STEPS), desc="Training Progress"):
            total_loss = compute_loss_weighted(model, torch_data, train_set, weights, optimizer)
            value = float(total_loss.item())
            local_history.append(value)

            if value < min_loss - FIT_LOSS_THR:
                min_loss = value
                steps_no_improve = 0
            else:
                steps_no_improve += 1

            grad_norm = float(compute_grad_norm(model).item())
            if step > FIT_PATIENCE and steps_no_improve >= FIT_PATIENCE:
                break
            if grad_norm < FIT_GRAD_NORM_THR:
                break

            if value < best_loss:
                best_loss = value
                best_params = [p.item() for p in model.parameters()]
                loss_history = list(local_history)

    assert best_params is not None
    fig_folder.mkdir(parents=True, exist_ok=True)
    plot_loss_curve(loss_history, str(fig_folder))
    return best_params, best_loss


def predict_curve(data: dict, file_name: str, best_params: list[float]) -> np.ndarray:
    L0, A, alpha, B, C, beta, gamma = best_params
    lrs = data[file_name]["lrs"]
    step = data[file_name]["step"]
    lr_sum = np.cumsum(lrs)
    lr_gap = np.zeros(len(lrs))
    lr_gap[1:] = np.diff(lrs)
    s1 = lr_sum[step]
    ld = np.zeros(len(step))
    for i, s in enumerate(step):
        ld[i] = np.sum(
            lr_gap[1 : s + 1]
            * (1 - (1 + C * lrs[1 : s + 1] ** (-gamma) * (lr_sum[s] - lr_sum[:s])) ** (-beta))
        )
    return L0 + A * s1 ** (-alpha) + B * ld


def evaluate_scheme(
    data: dict,
    curve_set: list[str],
    best_params: list[float],
    split: str,
    fig_folder: Path,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    split_dir = fig_folder / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for file_name in curve_set:
        step = data[file_name]["step"]
        loss = data[file_name]["loss"]
        pred = predict_curve(data, file_name, best_params)
        r = np.log(loss) - np.log(pred)
        row: dict[str, float | str] = {
            "curve": file_name,
            "split": split,
            "huber_loss": float(huber_loss(r).sum()),
            "mse_loss": float(mean_squared_error(loss, pred)),
            "rmse_loss": float(np.sqrt(mean_squared_error(loss, pred))),
            "mae_loss": float(np.mean(np.abs(loss - pred))),
            "prede": float(np.mean(np.abs(loss - pred) / loss)),
            "worste": float(np.max(np.abs(loss - pred) / loss)),
            "r2_score": float(r2_score(loss, pred)),
        }
        rows.append(row)

        plt.figure(figsize=(8.0, 4.8))
        plt.plot(step, pred, label="Pred", linestyle="--")
        plt.plot(step, loss, label="Ground Truth", linestyle="-")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title(f"{file_name.replace('.csv', '')} ({split})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(split_dir / f"{file_name.replace('.csv', '')}_mplfit.png", dpi=180)
        plt.close()
    return rows


def save_summary_chart(summary_rows: list[dict[str, float | str]]) -> None:
    schemes = list(SCHEMES.keys())
    scales = ["25", "100", "400"]
    x = np.arange(len(schemes))
    width = 0.22
    plt.figure(figsize=(10.2, 5.0))
    for idx, scale in enumerate(scales):
        vals = []
        for scheme in schemes:
            matches = [
                row["avg_test_mae"]
                for row in summary_rows
                if row["scale"] == scale and row["scheme"] == scheme
            ]
            vals.append(float(matches[0]))
        plt.bar(x + (idx - 1) * width, vals, width=width, label=f"{scale}M")
    plt.xticks(x, schemes, rotation=18)
    plt.ylabel("Average Test MAE")
    plt.title("Weighted MPL Schemes on Official Test Curves")
    plt.legend()
    plt.tight_layout()
    out_path = OUT_ROOT / "figures" / "avg_test_mae_by_scheme.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def write_report(summary_rows: list[dict[str, float | str]], ranking_rows: list[dict[str, float | str]]) -> None:
    lines = [
        "# 官方 MPL 训练曲线加权实验",
        "",
        "本实验保持官方 `MPL` 的初始化、模型结构、`AdamW` 超参数、训练/测试划分和评估流程不变，只修改训练集三条曲线在目标函数中的聚合权重。",
        "",
        "## 方案定义",
        "",
        "- `equal_1_1_1`: 等权基线",
        "- `cosine_constant_3_3_1`: `cosine_24000` 与 `constant_24000` 偏重，`wsdcon_9` 较轻",
        "- `cosine_wsdcon_3_1_3`: `cosine_24000` 与 `wsdcon_9` 偏重，`constant_24000` 较轻",
        "- `constant_wsdcon_1_3_3`: `constant_24000` 与 `wsdcon_9` 偏重，`cosine_24000` 较轻",
        "",
        "说明：所有 `3+3+1` 方案都做了归一化，使三条训练曲线的总权重仍为 `3`，避免仅仅因为总梯度尺度变化而影响比较。",
        "",
        "## 总排名（跨 25M / 100M / 400M 的平均测试 MAE）",
        "",
        "| Rank | Scheme | Avg Test MAE |",
        "| --- | --- | ---: |",
    ]
    for idx, row in enumerate(ranking_rows, start=1):
        lines.append(f"| {idx} | {row['scheme']} | {float(row['overall_avg_test_mae']):.6f} |")
    lines += [
        "",
        "## 分尺度平均测试 MAE",
        "",
        "| Scale | Scheme | Avg Test MAE | Avg Test Huber |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['scale']}M | {row['scheme']} | {float(row['avg_test_mae']):.6f} | {float(row['avg_test_huber']):.6f} |"
        )
    lines += [
        "",
        "## 结果目录",
        "",
        "- `results/weighted_mpl_official/tables/weighted_scheme_metrics.csv`",
        "- `results/weighted_mpl_official/tables/weighted_scheme_summary.csv`",
        "- `results/weighted_mpl_official/tables/weighted_scheme_overall_ranking.csv`",
        "- `results/weighted_mpl_official/figures/avg_test_mae_by_scheme.png`",
        "- `results/weighted_mpl_official/<scale>M/<scheme>/loss_monitor.png`",
        "- `results/weighted_mpl_official/<scale>M/<scheme>/train/*.png`",
        "- `results/weighted_mpl_official/<scale>M/<scheme>/test/*.png`",
        "",
    ]
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    scheme_json = {
        name: {
            "raw_pattern": list(raw),
            "normalized_weights": weights,
        }
        for name, raw, weights in [
            ("equal_1_1_1", [1, 1, 1], SCHEMES["equal_1_1_1"]),
            ("cosine_constant_3_3_1", [3, 3, 1], SCHEMES["cosine_constant_3_3_1"]),
            ("cosine_wsdcon_3_1_3", [3, 1, 3], SCHEMES["cosine_wsdcon_3_1_3"]),
            ("constant_wsdcon_1_3_3", [1, 3, 3], SCHEMES["constant_wsdcon_1_3_3"]),
        ]
    }
    (OUT_ROOT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "tables" / "scheme_definitions.json").write_text(json.dumps(scheme_json, indent=2), encoding="utf-8")

    metrics_rows: list[dict[str, float | str]] = []
    summary_rows: list[dict[str, float | str]] = []

    for scale in ["25", "100", "400"]:
        folder_path = MPL_ROOT / FOLDER_PATHS[scale].replace("./", "")
        logging.info("Loading %s", folder_path)
        data = load_data(str(folder_path))
        for scheme_name, weights in SCHEMES.items():
            run_dir = OUT_ROOT / f"{scale}M" / scheme_name
            init_param = initialize_params_weighted(data, TRAIN_SET, weights)
            init_params = generate_init_params(init_param)
            best_params, best_loss = mpl_adam_fit_weighted(data, TRAIN_SET, init_params, weights, run_dir)

            train_rows = evaluate_scheme(data, TRAIN_SET, best_params, "train", run_dir)
            test_rows = evaluate_scheme(data, TEST_SET, best_params, "test", run_dir)

            for row in train_rows + test_rows:
                row.update(
                    {
                        "scale": scale,
                        "scheme": scheme_name,
                        "best_loss": best_loss,
                        "weight_cosine_24000": weights["cosine_24000.csv"],
                        "weight_constant_24000": weights["constant_24000.csv"],
                        "weight_wsdcon_9": weights["wsdcon_9.csv"],
                    }
                )
                metrics_rows.append(row)

            summary_rows.append(
                {
                    "scale": scale,
                    "scheme": scheme_name,
                    "avg_test_mae": float(np.mean([row["mae_loss"] for row in test_rows])),
                    "avg_test_huber": float(np.mean([row["huber_loss"] for row in test_rows])),
                    "avg_train_mae": float(np.mean([row["mae_loss"] for row in train_rows])),
                    "best_loss": float(best_loss),
                }
            )

    overall_rows = []
    for scheme_name in SCHEMES:
        matches = [row for row in summary_rows if row["scheme"] == scheme_name]
        overall_rows.append(
            {
                "scheme": scheme_name,
                "overall_avg_test_mae": float(np.mean([row["avg_test_mae"] for row in matches])),
                "overall_avg_test_huber": float(np.mean([row["avg_test_huber"] for row in matches])),
            }
        )
    overall_rows.sort(key=lambda row: row["overall_avg_test_mae"])

    metrics_path = OUT_ROOT / "tables" / "weighted_scheme_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "scale",
            "scheme",
            "split",
            "curve",
            "mae_loss",
            "rmse_loss",
            "mse_loss",
            "huber_loss",
            "prede",
            "worste",
            "r2_score",
            "best_loss",
            "weight_cosine_24000",
            "weight_constant_24000",
            "weight_wsdcon_9",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow(row)

    summary_path = OUT_ROOT / "tables" / "weighted_scheme_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = ["scale", "scheme", "avg_test_mae", "avg_test_huber", "avg_train_mae", "best_loss"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    ranking_path = OUT_ROOT / "tables" / "weighted_scheme_overall_ranking.csv"
    with ranking_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = ["scheme", "overall_avg_test_mae", "overall_avg_test_huber"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in overall_rows:
            writer.writerow(row)

    save_summary_chart(summary_rows)
    write_report(summary_rows, overall_rows)
    print(f"Finished weighted MPL experiment under {OUT_ROOT}")


if __name__ == "__main__":
    main()
