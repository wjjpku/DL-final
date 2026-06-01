#!/usr/bin/env python3
"""Strict optimizer ablation with consistent protocol on 100M official split."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import huber
from scipy.stats import linregress


ROOT = Path(__file__).resolve().parents[1]
MPL_ROOT = ROOT / "external" / "MultiPowerLaw"
sys.path.insert(0, str(MPL_ROOT))

from src.config import FOLDER_PATHS, TRAIN_SET, TEST_SET, FIT_LR1, FIT_LR2, FIT_MAX_STEPS, FIT_PATIENCE, FIT_GRAD_NORM_THR, FIT_LOSS_THR  # noqa: E402
from src.data_loader import load_data  # noqa: E402
from src.fitting import initialize_params, generate_init_params  # noqa: E402


OUT_ROOT = ROOT / "results" / "optimizer_effect_strict"
DOC_PATH = ROOT / "docs" / "optimizer_effect_strict_report.md"
SCALE = "100"
DELTA = 1e-3
SCIPY_MAXITER = 400
SEED = 0
TORCH_DTYPE = torch.float64


@dataclass
class Curve:
    name: str
    step: np.ndarray
    loss: np.ndarray
    lrs: np.ndarray


def to_curves(data: dict[str, dict[str, np.ndarray]], names: list[str]) -> list[Curve]:
    return [
        Curve(name=name, step=data[name]["step"], loss=data[name]["loss"], lrs=data[name]["lrs"])
        for name in names
    ]


def huber_log_residual(y_true: np.ndarray, y_pred: np.ndarray, delta: float = DELTA) -> float:
    safe_pred = np.clip(y_pred, 1e-12, None)
    residual = np.log(np.clip(y_true, 1e-12, None)) - np.log(safe_pred)
    return float(huber(delta, residual).sum())


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_true - y_pred
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": float(np.mean(np.abs(err) / np.clip(y_true, 1e-12, None))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "huber_log": huber_log_residual(y_true, y_pred),
    }


def compute_s1(curve: Curve) -> np.ndarray:
    return np.cumsum(curve.lrs)[curve.step]


def compute_s2(curve: Curve, lam: float) -> np.ndarray:
    eta = curve.lrs
    anneal = np.zeros(len(eta), dtype=np.float64)
    s2_all = np.zeros(len(eta), dtype=np.float64)
    for t in range(1, len(eta)):
        delta = eta[t - 1] - eta[t]
        anneal[t] = lam * anneal[t - 1] + delta
        s2_all[t] = s2_all[t - 1] + anneal[t]
    return s2_all[curve.step]


def compute_ld(curve: Curve, C: float, beta: float, gamma: float) -> np.ndarray:
    lrs = curve.lrs
    lr_sum = np.cumsum(lrs)
    lr_gap = np.zeros(len(lrs), dtype=np.float64)
    lr_gap[1:] = np.diff(lrs)
    ld = np.zeros(len(curve.step), dtype=np.float64)
    for i, s in enumerate(curve.step):
        if s <= 0:
            continue
        hist = lrs[1 : s + 1]
        delta = lr_gap[1 : s + 1]
        remain = lr_sum[s] - lr_sum[:s]
        term = 1.0 - (1.0 + C * np.power(hist, -gamma) * remain) ** (-beta)
        ld[i] = np.sum(delta * term)
    return ld


def mpl_predict(params: np.ndarray, curve: Curve) -> np.ndarray:
    L0, A, alpha, B, C, beta, gamma = params
    s1 = compute_s1(curve)
    ld = compute_ld(curve, C, beta, gamma)
    return L0 + A * np.power(s1, -alpha) + B * ld


def load_official_mpl_params(scale: str) -> np.ndarray:
    log_path = MPL_ROOT / "logs" / f"{scale}.log"
    text = log_path.read_text(encoding="utf-8")
    marker = "Best Parameters: ["
    start = text.rfind(marker)
    if start < 0:
        raise ValueError(f"Cannot parse official MPL params from {log_path}")
    end = text.find("]", start)
    raw = text[start + len("Best Parameters: "): end + 1]
    return np.asarray(json.loads(raw), dtype=np.float64)


def tissue_predict(params: np.ndarray, curve: Curve) -> np.ndarray:
    L0, A, alpha, C, lam = params
    s1 = compute_s1(curve)
    s2 = compute_s2(curve, lam)
    return L0 + A * np.power(s1, -alpha) - C * s2


def tissue_init_from_regression(curves: list[Curve]) -> np.ndarray:
    min_loss = min(float(curve.loss.min()) for curve in curves)
    log_y_all = []
    log_x_all = []
    for curve in curves:
        log_y_all.append(np.log(curve.loss - min_loss + 0.01))
        log_x_all.append(np.log(np.cumsum(curve.lrs)[curve.step]))
    slope, intercept, _, _, _ = linregress(np.concatenate(log_x_all), np.concatenate(log_y_all))
    L0 = max(min_loss - 0.05, 1e-4)
    A = max(float(np.exp(intercept)), 1e-4)
    alpha = max(float(-slope), 1e-3)
    C = 50.0
    lam = 0.995
    return np.array([L0, A, alpha, C, lam], dtype=np.float64)


class TissueTorch(torch.nn.Module):
    def __init__(self, init_param: np.ndarray):
        super().__init__()
        self.L0 = torch.nn.Parameter(torch.tensor(init_param[0], dtype=TORCH_DTYPE))
        self.A = torch.nn.Parameter(torch.tensor(init_param[1], dtype=TORCH_DTYPE))
        self.alpha = torch.nn.Parameter(torch.tensor(init_param[2], dtype=TORCH_DTYPE))
        self.C = torch.nn.Parameter(torch.tensor(init_param[3], dtype=TORCH_DTYPE))
        self.lam = torch.nn.Parameter(torch.tensor(init_param[4], dtype=TORCH_DTYPE))

    def forward(self, curve_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        lam = torch.clamp(self.lam, min=0.9, max=0.9999)
        alpha = torch.clamp(self.alpha, min=1e-4)
        A = torch.clamp(self.A, min=1e-12)
        C = torch.clamp(self.C, min=1e-12)
        S1 = curve_batch["S1"]
        lrs = curve_batch["lrs"]
        step = curve_batch["step"]
        loss = curve_batch["loss"]
        anneal_prev = torch.zeros((), dtype=TORCH_DTYPE)
        s2_prev = torch.zeros((), dtype=TORCH_DTYPE)
        s2_vals = [torch.zeros((), dtype=TORCH_DTYPE)]
        for t in range(1, len(lrs)):
            delta = lrs[t - 1] - lrs[t]
            anneal_prev = lam * anneal_prev + delta
            s2_prev = s2_prev + anneal_prev
            s2_vals.append(s2_prev)
        s2_all = torch.stack(s2_vals)
        pred = self.L0 + A * torch.pow(S1, -alpha) - C * s2_all[step]
        pred = pred.clamp(min=1e-10)
        residual = torch.log(loss) - torch.log(pred)
        return huber_torch(residual).sum()


def huber_torch(residual: torch.Tensor, delta: float = DELTA) -> torch.Tensor:
    abs_res = residual.abs()
    quad = torch.minimum(abs_res, torch.tensor(delta, dtype=TORCH_DTYPE))
    lin = abs_res - quad
    return 0.5 * quad**2 + delta * lin


def preprocess_tissue(curves: list[Curve]) -> dict[str, dict[str, torch.Tensor]]:
    out: dict[str, dict[str, torch.Tensor]] = {}
    for curve in curves:
        step = torch.tensor(curve.step, dtype=torch.int64)
        lrs = torch.tensor(curve.lrs, dtype=TORCH_DTYPE)
        loss = torch.tensor(curve.loss, dtype=TORCH_DTYPE)
        lr_sum = torch.cumsum(lrs, dim=0)
        out[curve.name] = {"step": step, "lrs": lrs, "loss": loss, "S1": lr_sum[step]}
    return out


def grad_norm(model: torch.nn.Module) -> float:
    grads = [p.grad.flatten() for p in model.parameters() if p.grad is not None]
    return float(torch.cat(grads).norm().item()) if grads else 0.0


def fit_tissue_adamw(curves: list[Curve], init_param: np.ndarray) -> tuple[np.ndarray, float]:
    torch.manual_seed(SEED)
    data = preprocess_tissue(curves)
    model = TissueTorch(init_param)
    optimizer = torch.optim.AdamW(
        [
            {"params": [model.L0, model.A, model.C], "lr": FIT_LR1},
            {"params": [model.alpha, model.lam], "lr": FIT_LR2},
        ]
    )
    best_loss = float("inf")
    best_params = None
    min_loss = float("inf")
    steps_no_improve = 0
    for step in range(FIT_MAX_STEPS):
        optimizer.zero_grad()
        total_loss = torch.zeros((), dtype=TORCH_DTYPE)
        for curve in curves:
            total_loss = total_loss + model(data[curve.name])
        total_loss.backward()
        optimizer.step()
        value = float(total_loss.item())
        if value < min_loss - FIT_LOSS_THR:
            min_loss = value
            steps_no_improve = 0
        else:
            steps_no_improve += 1
        if value < best_loss:
            best_loss = value
            best_params = np.asarray([float(p.item()) for p in model.parameters()], dtype=np.float64)
        if step > FIT_PATIENCE and steps_no_improve >= FIT_PATIENCE:
            break
        if grad_norm(model) < FIT_GRAD_NORM_THR:
            break
    assert best_params is not None
    return best_params, best_loss


def fit_tissue_lbfgsb(curves: list[Curve], init_param: np.ndarray) -> tuple[np.ndarray, float]:
    def objective(params: np.ndarray) -> float:
        pred_all = []
        loss_all = []
        for curve in curves:
            pred = tissue_predict(params, curve)
            if np.any(~np.isfinite(pred)) or np.any(pred <= 0):
                return 1e18
            pred_all.append(pred)
            loss_all.append(curve.loss)
        return huber_log_residual(np.concatenate(loss_all), np.concatenate(pred_all))

    bounds = [(0.0, 10.0), (1e-8, 100.0), (1e-4, 3.0), (1e-8, 1e5), (0.9, 0.9999)]
    res = minimize(
        objective,
        x0=init_param,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": SCIPY_MAXITER, "ftol": 1e-10},
    )
    return np.asarray(res.x, dtype=np.float64), float(res.fun)


def fit_mpl_lbfgsb(curves: list[Curve], init_param: np.ndarray) -> tuple[np.ndarray, float]:
    def objective(params: np.ndarray) -> float:
        pred_all = []
        loss_all = []
        for curve in curves:
            pred = mpl_predict(params, curve)
            if np.any(~np.isfinite(pred)) or np.any(pred <= 0):
                return 1e18
            pred_all.append(pred)
            loss_all.append(curve.loss)
        return huber_log_residual(np.concatenate(loss_all), np.concatenate(pred_all))

    bounds = [
        (0.0, 10.0),
        (1e-8, 100.0),
        (1e-4, 3.0),
        (1e-8, 1e5),
        (1e-8, 100.0),
        (1e-4, 5.0),
        (1e-4, 5.0),
    ]
    res = minimize(
        objective,
        x0=init_param,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": SCIPY_MAXITER, "ftol": 1e-10},
    )
    return np.asarray(res.x, dtype=np.float64), float(res.fun)


def evaluate(curves: list[Curve], params: np.ndarray, model_name: str, split: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for curve in curves:
        pred = tissue_predict(params, curve) if model_name == "tissue" else mpl_predict(params, curve)
        row = {"curve": curve.name, "split": split}
        row.update(metrics(curve.loss, pred))
        rows.append(row)
    return rows


def save_example_plot(curve: Curve, model_name: str, lbfgsb_params: np.ndarray, adamw_params: np.ndarray) -> None:
    out_dir = OUT_ROOT / "figures" / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_lb = tissue_predict(lbfgsb_params, curve) if model_name == "tissue" else mpl_predict(lbfgsb_params, curve)
    pred_ad = tissue_predict(adamw_params, curve) if model_name == "tissue" else mpl_predict(adamw_params, curve)
    plt.figure(figsize=(8.0, 4.8))
    plt.plot(curve.step, curve.loss, color="#222222", linewidth=2.2, label="Ground Truth")
    plt.plot(curve.step, pred_lb, color="#4C78A8", linestyle="--", linewidth=2.0, label="L-BFGS-B")
    plt.plot(curve.step, pred_ad, color="#E45756", linestyle="-.", linewidth=2.0, label="AdamW")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"{model_name.upper()} Strict Optimizer Compare: {curve.name.replace('.csv', '')}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{model_name}_{curve.name.replace('.csv', '')}.png", dpi=180)
    plt.close()


def save_summary_bar(metrics_rows: list[dict[str, object]], split: str, out_path: Path) -> None:
    groups = [("tissue", "lbfgsb"), ("tissue", "adamw"), ("mpl", "lbfgsb"), ("mpl", "adamw")]
    labels = ["Tissue-LBFGS", "Tissue-AdamW", "MPL-LBFGS", "MPL-AdamW"]
    values = []
    for model_name, optimizer_name in groups:
        subset = [r for r in metrics_rows if r["split"] == split and r["model"] == model_name and r["optimizer"] == optimizer_name]
        values.append(float(np.mean([r["mae"] for r in subset])))
    plt.figure(figsize=(8.2, 4.6))
    plt.bar(labels, values, color=["#4C78A8", "#72B7B2", "#F58518", "#E45756"])
    plt.ylabel(f"Average {split.capitalize()} MAE")
    plt.title(f"Strict Protocol Optimizer Effect ({split.capitalize()})")
    plt.xticks(rotation=20)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_param_bars(best_params: dict[str, dict[str, np.ndarray]], model_name: str, param_names: list[str]) -> None:
    lb = best_params[model_name]["lbfgsb"]
    ad = best_params[model_name]["adamw"]
    x = np.arange(len(param_names))
    width = 0.36
    plt.figure(figsize=(9.2, 4.8))
    plt.bar(x - width / 2, lb, width=width, label="L-BFGS-B", color="#4C78A8")
    plt.bar(x + width / 2, ad, width=width, label="AdamW", color="#E45756")
    plt.xticks(x, param_names)
    plt.title(f"{model_name.upper()} Best Parameters Under Strict Protocol")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out_path = OUT_ROOT / "figures" / f"{model_name}_best_params.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def write_report(metrics_rows: list[dict[str, object]], best_params: dict[str, dict[str, np.ndarray]], init_params: dict[str, list[float]]) -> None:
    def avg(split: str, model_name: str, optimizer_name: str) -> float:
        subset = [r["mae"] for r in metrics_rows if r["split"] == split and r["model"] == model_name and r["optimizer"] == optimizer_name]
        return float(np.mean(subset))

    lines = [
        "# 严格一致协议下的优化器影响实验",
        "",
        "本实验只在 `100M` 尺度上进行，并且统一以下设置：",
        "",
        "- 相同训练集与测试集划分",
        "- 相同完整训练点与完整评估点",
        "- 相同目标函数：`log(loss)` 残差上的 Huber 损失",
        "- 相同初始化参数，再分别交给 `AdamW` 与 `L-BFGS-B`",
        "- `MPL` 的 `AdamW` 训练流程完整复用官方初始化与官方训练超参数",
        "",
        "## 平均测试 MAE",
        "",
        "| Model | Optimizer | 100M |",
        "| --- | --- | ---: |",
        f"| TISSUE | L-BFGS-B | {avg('test', 'tissue', 'lbfgsb'):.6f} |",
        f"| TISSUE | AdamW | {avg('test', 'tissue', 'adamw'):.6f} |",
        f"| MPL | L-BFGS-B | {avg('test', 'mpl', 'lbfgsb'):.6f} |",
        f"| MPL | AdamW | {avg('test', 'mpl', 'adamw'):.6f} |",
        "",
        "## 平均训练 MAE",
        "",
        "| Model | Optimizer | 100M |",
        "| --- | --- | ---: |",
        f"| TISSUE | L-BFGS-B | {avg('train', 'tissue', 'lbfgsb'):.6f} |",
        f"| TISSUE | AdamW | {avg('train', 'tissue', 'adamw'):.6f} |",
        f"| MPL | L-BFGS-B | {avg('train', 'mpl', 'lbfgsb'):.6f} |",
        f"| MPL | AdamW | {avg('train', 'mpl', 'adamw'):.6f} |",
        "",
        "## 初始化参数",
        "",
        f"- `Tissue init`: {init_params['tissue']}",
        f"- `MPL init`: {init_params['mpl']}",
        "",
        "## 输出文件",
        "",
        "- `results/optimizer_effect_strict/tables/strict_optimizer_metrics.csv`",
        "- `results/optimizer_effect_strict/tables/strict_optimizer_best_params.json`",
        "- `results/optimizer_effect_strict/figures/train_mae_summary.png`",
        "- `results/optimizer_effect_strict/figures/test_mae_summary.png`",
        "- `results/optimizer_effect_strict/figures/tissue_best_params.png`",
        "- `results/optimizer_effect_strict/figures/mpl_best_params.png`",
        "- `results/optimizer_effect_strict/figures/examples/`",
    ]
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    folder_path = MPL_ROOT / FOLDER_PATHS[SCALE].replace("./", "")
    data = load_data(str(folder_path))
    train_curves = to_curves(data, TRAIN_SET)
    test_curves = to_curves(data, TEST_SET)

    mpl_init_4 = initialize_params(data, TRAIN_SET)
    mpl_init = np.asarray(generate_init_params(mpl_init_4)[0], dtype=np.float64)
    tissue_init = tissue_init_from_regression(train_curves)

    mpl_adamw_params = load_official_mpl_params(SCALE)
    mpl_lbfgsb_params, _ = fit_mpl_lbfgsb(train_curves, mpl_init)

    tissue_adamw_params, _ = fit_tissue_adamw(train_curves, tissue_init)
    tissue_lbfgsb_params, _ = fit_tissue_lbfgsb(train_curves, tissue_init)

    best_params = {
        "mpl": {"adamw": mpl_adamw_params, "lbfgsb": mpl_lbfgsb_params},
        "tissue": {"adamw": tissue_adamw_params, "lbfgsb": tissue_lbfgsb_params},
    }

    rows: list[dict[str, object]] = []
    for split, curves in [("train", train_curves), ("test", test_curves)]:
        for model_name, params_map in best_params.items():
            for optimizer_name, params in params_map.items():
                for row in evaluate(curves, params, model_name, split):
                    row.update({"split": split, "model": model_name, "optimizer": optimizer_name, "scale": SCALE})
                    rows.append(row)

    (OUT_ROOT / "tables").mkdir(parents=True, exist_ok=True)
    with (OUT_ROOT / "tables" / "strict_optimizer_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        fieldnames = ["scale", "split", "model", "optimizer", "curve", "mae", "rmse", "mape", "r2", "huber_log"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with (OUT_ROOT / "tables" / "strict_optimizer_best_params.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {model: {opt: params.tolist() for opt, params in opt_map.items()} for model, opt_map in best_params.items()},
            fh,
            indent=2,
        )

    save_summary_bar(rows, "train", OUT_ROOT / "figures" / "train_mae_summary.png")
    save_summary_bar(rows, "test", OUT_ROOT / "figures" / "test_mae_summary.png")
    save_param_bars(best_params, "tissue", ["L0", "A", "alpha", "C", "lam"])
    save_param_bars(best_params, "mpl", ["L0", "A", "alpha", "B", "C", "beta", "gamma"])

    example_map = {curve.name: curve for curve in train_curves + test_curves}
    for model_name in ["tissue", "mpl"]:
        save_example_plot(example_map["wsdcon_9.csv"], model_name, best_params[model_name]["lbfgsb"], best_params[model_name]["adamw"])
        save_example_plot(example_map["wsdcon_3.csv"], model_name, best_params[model_name]["lbfgsb"], best_params[model_name]["adamw"])
        save_example_plot(example_map["wsd_20000_24000.csv"], model_name, best_params[model_name]["lbfgsb"], best_params[model_name]["adamw"])

    write_report(
        rows,
        best_params,
        {"mpl": mpl_init.tolist(), "tissue": tissue_init.tolist()},
    )
    print(f"Finished strict optimizer experiment under {OUT_ROOT}")


if __name__ == "__main__":
    main()
