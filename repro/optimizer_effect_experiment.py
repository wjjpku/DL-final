#!/usr/bin/env python3
"""Compare AdamW and L-BFGS-B on Tissue and MPL under the official public split."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "external" / "MultiPowerLaw" / "loss_curve_repo"
OUT_ROOT = ROOT / "results" / "optimizer_effect"
DOC_PATH = ROOT / "docs" / "optimizer_effect_report.md"

TRAIN_CURVES = ["cosine_24000.csv", "constant_24000.csv", "wsdcon_9.csv"]
TEST_CURVES = [
    "constant_72000.csv",
    "cosine_72000.csv",
    "wsd_20000_24000.csv",
    "wsdld_20000_24000.csv",
    "wsdcon_3.csv",
    "wsdcon_18.csv",
]
SCALES = ["25", "100", "400"]
DELTA = 1e-3
SCIPY_MAXITER = 120
ADAMW_MAX_STEPS = 35
ADAMW_PATIENCE = 10
ADAMW_GRAD_THR = 1e-6
TORCH_DTYPE = torch.float64
SEED = 0
FIT_SUBSAMPLE_STRIDE = 4


@dataclass
class Curve:
    name: str
    scale: str
    step: np.ndarray
    loss: np.ndarray
    lrs: np.ndarray


def subsample_curve(curve: Curve, stride: int = FIT_SUBSAMPLE_STRIDE) -> Curve:
    return Curve(
        name=curve.name,
        scale=curve.scale,
        step=curve.step[::stride],
        loss=curve.loss[::stride],
        lrs=curve.lrs,
    )


def cosine_lrs(warmup: int, total: int, peak_lr: float, end_lr: float) -> np.ndarray:
    step = np.arange(total)[warmup:]
    warmup_lrs = np.linspace(0.0, peak_lr, warmup)
    cosine = end_lr + 0.5 * (peak_lr - end_lr) * (
        1.0 + np.cos(np.pi * (step - warmup) / (total - warmup))
    )
    return np.concatenate((warmup_lrs, cosine))


def const_lrs(warmup: int, total: int, lr: float) -> np.ndarray:
    warmup_lrs = np.linspace(0.0, lr, warmup)
    return np.concatenate((warmup_lrs, np.full(total - warmup, lr)))


def two_stage_lrs(
    warmup: int, total: int, lr_a: float, lr_b: float, stage_a: int
) -> np.ndarray:
    warmup_lrs = np.linspace(0.0, lr_a, warmup)
    stage_a_lrs = np.full(stage_a - warmup, lr_a)
    stage_b_lrs = np.full(total - stage_a, lr_b)
    return np.concatenate((warmup_lrs, stage_a_lrs, stage_b_lrs))


def wsd_lrs(
    warmup: int, total: int, decay: int, peak_lr: float, end_lr: float
) -> np.ndarray:
    step = np.arange(total)[decay:]
    warmup_lrs = np.linspace(0.0, peak_lr, warmup)
    decay_lrs = peak_lr ** ((total - step) / (total - decay)) * end_lr ** (
        (step - decay) / (total - decay)
    )
    return np.concatenate((warmup_lrs, np.full(decay - warmup, peak_lr), decay_lrs))


def wsdld_lrs(
    warmup: int, total: int, decay: int, peak_lr: float, end_lr: float
) -> np.ndarray:
    step = np.arange(total)[decay:]
    warmup_lrs = np.linspace(0.0, peak_lr, warmup)
    decay_lrs = peak_lr * (1.0 - (step - decay) / (total - decay)) + end_lr * (
        step - decay
    ) / (total - decay)
    return np.concatenate((warmup_lrs, np.full(decay - warmup, peak_lr), decay_lrs))


def build_lrs(file_name: str) -> np.ndarray:
    if "cosine" in file_name:
        total = int(file_name.split("_")[1].split(".")[0])
        return cosine_lrs(2160, total, 3e-4, 3e-5)
    if "constant" in file_name:
        total = int(file_name.split("_")[1].split(".")[0])
        return const_lrs(2160, total, 3e-4)
    if "wsdcon" in file_name:
        total = 16000
        lr_b = int(file_name.split("_")[1].split(".")[0]) * 1e-5
        return two_stage_lrs(2160, total, 3e-4, lr_b, 8000)
    if "wsdld" in file_name:
        return wsdld_lrs(2160, 24000, 20000, 3e-4, 3e-5)
    if "wsd" in file_name:
        return wsd_lrs(2160, 24000, 20000, 3e-4, 3e-5)
    raise ValueError(f"Unsupported curve: {file_name}")


def load_curve(scale: str, file_name: str) -> Curve:
    raw = np.genfromtxt(DATA_ROOT / f"csv_{scale}" / file_name, delimiter=",", skip_header=1)
    step = raw[:, 0].astype(int)
    loss = raw[:, 2].astype(float)
    if step.max() == 24000:
        mask = step < 24000
        step = step[mask]
        loss = loss[mask]
    return Curve(name=file_name, scale=scale, step=step, loss=loss, lrs=build_lrs(file_name))


def huber_log_residual(y_true: np.ndarray, y_pred: np.ndarray, delta: float = DELTA) -> float:
    safe_pred = np.clip(y_pred, 1e-12, None)
    residual = np.log(np.clip(y_true, 1e-12, None)) - np.log(safe_pred)
    abs_res = np.abs(residual)
    quad = np.minimum(abs_res, delta)
    lin = abs_res - quad
    return float(np.sum(0.5 * quad**2 + delta * lin))


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_true - y_pred
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": float(np.mean(np.abs(err) / np.clip(y_true, 1e-12, None))),
        "r2": r2,
        "huber_log": huber_log_residual(y_true, y_pred),
    }


def compute_s1(curve: Curve) -> np.ndarray:
    return np.cumsum(curve.lrs)[curve.step]


def compute_s2(curve: Curve, lam: float) -> np.ndarray:
    eta = curve.lrs
    total = len(eta)
    anneal = np.zeros(total, dtype=np.float64)
    s2_all = np.zeros(total, dtype=np.float64)
    for t in range(1, total):
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


def tissue_predict(params: np.ndarray, curve: Curve) -> np.ndarray:
    L0, A, alpha, C, lam = params
    s1 = compute_s1(curve)
    s2 = compute_s2(curve, lam)
    return L0 + A * np.power(s1, -alpha) - C * s2


def mpl_predict(params: np.ndarray, curve: Curve) -> np.ndarray:
    L0, A, alpha, B, C, beta, gamma = params
    s1 = compute_s1(curve)
    ld = compute_ld(curve, C, beta, gamma)
    return L0 + A * np.power(s1, -alpha) + B * ld


def tissue_inits(curves: list[Curve]) -> list[np.ndarray]:
    min_loss = min(float(curve.loss.min()) for curve in curves)
    return [
        np.array([min_loss - 0.05, 0.5, 0.5, 100.0, 0.995], dtype=np.float64),
        np.array([min_loss - 0.1, 1.0, 0.4, 10.0, 0.995], dtype=np.float64),
        np.array([min_loss, 0.2, 0.7, 300.0, 0.999], dtype=np.float64),
        np.array([min_loss - 0.03, 0.8, 0.45, 30.0, 0.999], dtype=np.float64),
    ]


def mpl_inits(curves: list[Curve]) -> list[np.ndarray]:
    min_loss = min(float(curve.loss.min()) for curve in curves)
    return [
        np.array([min_loss - 0.05, 0.5, 0.5, 300.0, 1.0, 0.5, 0.5], dtype=np.float64),
        np.array([min_loss - 0.10, 1.0, 0.4, 100.0, 0.5, 0.3, 0.3], dtype=np.float64),
    ]


def fit_with_lbfgsb(
    init_params: list[np.ndarray],
    bounds: list[tuple[float, float]],
    objective_factory,
) -> tuple[np.ndarray, float, list[dict[str, object]]]:
    best_x = None
    best_fun = float("inf")
    runs: list[dict[str, object]] = []
    for run_id, init in enumerate(init_params):
        res = minimize(
            objective_factory,
            x0=np.asarray(init, dtype=np.float64),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": SCIPY_MAXITER, "ftol": 1e-10},
        )
        runs.append({"run_id": run_id, "objective": float(res.fun), "params": res.x.tolist()})
        if res.fun < best_fun:
            best_fun = float(res.fun)
            best_x = res.x
    assert best_x is not None
    return best_x, best_fun, runs


class TissueTorch(torch.nn.Module):
    def __init__(self, init_param: np.ndarray):
        super().__init__()
        self.L0 = torch.nn.Parameter(torch.tensor(init_param[0], dtype=TORCH_DTYPE))
        self.A = torch.nn.Parameter(torch.tensor(init_param[1], dtype=TORCH_DTYPE))
        self.alpha = torch.nn.Parameter(torch.tensor(init_param[2], dtype=TORCH_DTYPE))
        self.C = torch.nn.Parameter(torch.tensor(init_param[3], dtype=TORCH_DTYPE))
        self.lam = torch.nn.Parameter(torch.tensor(init_param[4], dtype=TORCH_DTYPE))

    def forward(self, S1, lrs, step, loss):
        lam = torch.clamp(self.lam, min=0.9, max=0.9999)
        alpha = torch.clamp(self.alpha, min=1e-4)
        A = torch.clamp(self.A, min=1e-12)
        C = torch.clamp(self.C, min=1e-12)
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
        abs_res = residual.abs()
        quad = torch.minimum(abs_res, torch.tensor(DELTA, dtype=TORCH_DTYPE))
        lin = abs_res - quad
        return torch.sum(0.5 * quad**2 + DELTA * lin)


class MPLTorch(torch.nn.Module):
    def __init__(self, init_param: np.ndarray):
        super().__init__()
        self.L0 = torch.nn.Parameter(torch.tensor(init_param[0], dtype=TORCH_DTYPE))
        self.A = torch.nn.Parameter(torch.tensor(init_param[1], dtype=TORCH_DTYPE))
        self.alpha = torch.nn.Parameter(torch.tensor(init_param[2], dtype=TORCH_DTYPE))
        self.B = torch.nn.Parameter(torch.tensor(init_param[3], dtype=TORCH_DTYPE))
        self.C = torch.nn.Parameter(torch.tensor(init_param[4], dtype=TORCH_DTYPE))
        self.beta = torch.nn.Parameter(torch.tensor(init_param[5], dtype=TORCH_DTYPE))
        self.gamma = torch.nn.Parameter(torch.tensor(init_param[6], dtype=TORCH_DTYPE))

    def forward(self, S1, lrs, lr_sum, step, lr_gap, loss):
        alpha = torch.clamp(self.alpha, min=1e-4)
        A = torch.clamp(self.A, min=1e-12)
        B = torch.clamp(self.B, min=1e-12)
        C = torch.clamp(self.C, min=1e-12)
        beta = torch.clamp(self.beta, min=1e-4)
        gamma = torch.clamp(self.gamma, min=1e-4)
        ld = torch.zeros_like(step, dtype=TORCH_DTYPE)
        for i, s in enumerate(step):
            if s > 0:
                term = 1.0 - (
                    1.0 + C * lrs[1 : s + 1] ** (-gamma) * (lr_sum[s] - lr_sum[:s])
                ) ** (-beta)
                ld[i] = torch.sum(lr_gap[1 : s + 1] * term)
        pred = self.L0 + A * torch.pow(S1, -alpha) + B * ld
        pred = pred.clamp(min=1e-10)
        residual = torch.log(loss) - torch.log(pred)
        abs_res = residual.abs()
        quad = torch.minimum(abs_res, torch.tensor(DELTA, dtype=TORCH_DTYPE))
        lin = abs_res - quad
        return torch.sum(0.5 * quad**2 + DELTA * lin)


def preprocess_torch(curves: list[Curve]) -> dict[str, dict[str, torch.Tensor]]:
    data: dict[str, dict[str, torch.Tensor]] = {}
    for curve in curves:
        step = torch.tensor(curve.step, dtype=torch.int64)
        lrs = torch.tensor(curve.lrs, dtype=TORCH_DTYPE)
        loss = torch.tensor(curve.loss, dtype=TORCH_DTYPE)
        lr_sum = torch.cumsum(lrs, dim=0)
        lr_gap = torch.zeros_like(lrs)
        lr_gap[1:] = torch.diff(lrs)
        data[curve.name] = {
            "step": step,
            "lrs": lrs,
            "loss": loss,
            "S1": lr_sum[step],
            "lr_sum": lr_sum,
            "lr_gap": lr_gap,
        }
    return data


def grad_norm(model: torch.nn.Module) -> float:
    grads = [p.grad.flatten() for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    return float(torch.cat(grads).norm().item())


def fit_adamw_tissue(curves: list[Curve], init_params: list[np.ndarray]) -> tuple[np.ndarray, float, list[dict[str, object]]]:
    torch.manual_seed(SEED)
    data = preprocess_torch(curves)
    best_params = None
    best_loss = float("inf")
    runs: list[dict[str, object]] = []
    for run_id, init in enumerate(init_params):
        model = TissueTorch(init)
        optimizer = torch.optim.AdamW(
            [
                {"params": [model.L0, model.A, model.C], "lr": 5e-2},
                {"params": [model.alpha, model.lam], "lr": 5e-3},
            ]
        )
        local_best_loss = float("inf")
        local_best_params = None
        steps_no_improve = 0
        for _ in range(ADAMW_MAX_STEPS):
            optimizer.zero_grad()
            total_loss = torch.zeros((), dtype=TORCH_DTYPE)
            for curve in curves:
                item = data[curve.name]
                total_loss = total_loss + model(item["S1"], item["lrs"], item["step"], item["loss"])
            total_loss.backward()
            optimizer.step()
            value = float(total_loss.item())
            if value < local_best_loss - 1e-12:
                local_best_loss = value
                local_best_params = [float(p.item()) for p in model.parameters()]
                steps_no_improve = 0
            else:
                steps_no_improve += 1
            if grad_norm(model) < ADAMW_GRAD_THR or steps_no_improve >= ADAMW_PATIENCE:
                break
        assert local_best_params is not None
        runs.append({"run_id": run_id, "objective": local_best_loss, "params": local_best_params})
        if local_best_loss < best_loss:
            best_loss = local_best_loss
            best_params = np.asarray(local_best_params, dtype=np.float64)
    assert best_params is not None
    return best_params, best_loss, runs


def fit_adamw_mpl(curves: list[Curve], init_params: list[np.ndarray]) -> tuple[np.ndarray, float, list[dict[str, object]]]:
    torch.manual_seed(SEED)
    data = preprocess_torch(curves)
    best_params = None
    best_loss = float("inf")
    runs: list[dict[str, object]] = []
    for run_id, init in enumerate(init_params):
        model = MPLTorch(init)
        optimizer = torch.optim.AdamW(
            [
                {"params": [model.L0, model.A, model.B, model.C], "lr": 5e-2},
                {"params": [model.alpha, model.beta, model.gamma], "lr": 5e-3},
            ]
        )
        local_best_loss = float("inf")
        local_best_params = None
        steps_no_improve = 0
        for _ in range(ADAMW_MAX_STEPS):
            optimizer.zero_grad()
            total_loss = torch.zeros((), dtype=TORCH_DTYPE)
            for curve in curves:
                item = data[curve.name]
                total_loss = total_loss + model(
                    item["S1"], item["lrs"], item["lr_sum"], item["step"], item["lr_gap"], item["loss"]
                )
            total_loss.backward()
            optimizer.step()
            value = float(total_loss.item())
            if value < local_best_loss - 1e-12:
                local_best_loss = value
                local_best_params = [float(p.item()) for p in model.parameters()]
                steps_no_improve = 0
            else:
                steps_no_improve += 1
            if grad_norm(model) < ADAMW_GRAD_THR or steps_no_improve >= ADAMW_PATIENCE:
                break
        assert local_best_params is not None
        runs.append({"run_id": run_id, "objective": local_best_loss, "params": local_best_params})
        if local_best_loss < best_loss:
            best_loss = local_best_loss
            best_params = np.asarray(local_best_params, dtype=np.float64)
    assert best_params is not None
    return best_params, best_loss, runs


def fit_tissue_lbfgsb(curves: list[Curve], init_params: list[np.ndarray]):
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
    return fit_with_lbfgsb(init_params, bounds, objective)


def fit_mpl_lbfgsb(curves: list[Curve], init_params: list[np.ndarray]):
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
    return fit_with_lbfgsb(init_params, bounds, objective)


def evaluate_model(model_name: str, params: np.ndarray, curve: Curve) -> dict[str, float]:
    pred = tissue_predict(params, curve) if model_name == "tissue" else mpl_predict(params, curve)
    return metrics(curve.loss, pred)


def save_summary_plot(rows: list[dict[str, object]], split: str, metric: str, out_path: Path, scales: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9.0, 5.0))
    colors = {
        ("tissue", "lbfgsb"): "#4C78A8",
        ("tissue", "adamw"): "#72B7B2",
        ("mpl", "lbfgsb"): "#F58518",
        ("mpl", "adamw"): "#E45756",
    }
    for model_name in ["tissue", "mpl"]:
        for optimizer_name in ["lbfgsb", "adamw"]:
            xs = []
            ys = []
            for scale in scales:
                subset = [
                    row
                    for row in rows
                    if row["split"] == split and row["scale"] == scale and row["model"] == model_name and row["optimizer"] == optimizer_name
                ]
                xs.append(int(scale))
                ys.append(float(np.mean([float(row[metric]) for row in subset])))
            plt.plot(
                xs,
                ys,
                marker="o",
                label=f"{model_name.upper()}-{optimizer_name.upper()}",
                color=colors[(model_name, optimizer_name)],
            )
    plt.xticks([int(scale) for scale in scales])
    plt.xlabel("Model Size (M)")
    plt.ylabel(f"Average {split.capitalize()} {metric.upper()}")
    plt.title(f"{split.capitalize()} {metric.upper()} by Model and Optimizer")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_best_param_bars(best_params: dict[str, dict[str, dict[str, np.ndarray]]], model_name: str, param_names: list[str], scales: list[str]) -> None:
    fig, axes = plt.subplots(len(scales), 1, figsize=(10.0, 3.2 * len(scales)), sharex=True)
    if len(scales) == 1:
        axes = [axes]
    for ax, scale in zip(axes, scales):
        lb = best_params[model_name][scale]["lbfgsb"]
        ad = best_params[model_name][scale]["adamw"]
        x = np.arange(len(param_names))
        width = 0.36
        ax.bar(x - width / 2, lb, width=width, label="L-BFGS-B", color="#4C78A8")
        ax.bar(x + width / 2, ad, width=width, label="AdamW", color="#E45756")
        ax.set_title(f"{model_name.upper()} Best Parameters ({scale}M)")
        ax.set_xticks(x, param_names)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    fig.tight_layout()
    out_path = OUT_ROOT / "figures" / f"{model_name}_best_params.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_example_plot(
    model_name: str,
    scale: str,
    curve: Curve,
    params_lbfgsb: np.ndarray,
    params_adamw: np.ndarray,
) -> None:
    out_dir = OUT_ROOT / "figures" / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_lbfgsb = tissue_predict(params_lbfgsb, curve) if model_name == "tissue" else mpl_predict(params_lbfgsb, curve)
    pred_adamw = tissue_predict(params_adamw, curve) if model_name == "tissue" else mpl_predict(params_adamw, curve)
    plt.figure(figsize=(8.4, 4.8))
    plt.plot(curve.step, curve.loss, label="Ground Truth", linewidth=2.2, color="#222222")
    plt.plot(curve.step, pred_lbfgsb, label="L-BFGS-B", linestyle="--", linewidth=2.0, color="#4C78A8")
    plt.plot(curve.step, pred_adamw, label="AdamW", linestyle="-.", linewidth=2.0, color="#E45756")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"{model_name.upper()} Optimizer Effect on {curve.name.replace('.csv', '')} ({scale}M)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{model_name}_{scale}_{curve.name.replace('.csv', '')}.png", dpi=180)
    plt.close()


def write_report(
    rows: list[dict[str, object]],
    param_rows: list[dict[str, object]],
    best_params: dict[str, dict[str, dict[str, np.ndarray]]],
    scales: list[str],
) -> None:
    lines: list[str] = []
    lines.append("# 优化器影响实验")
    lines.append("")
    lines.append("本实验在同一官方公开训练/测试划分上，对两个公式模型分别比较 `L-BFGS-B` 与 `AdamW`。")
    lines.append("")
    lines.append("## 训练集")
    lines.append("")
    lines.append("- `cosine_24000`")
    lines.append("- `constant_24000`")
    lines.append("- `wsdcon_9`")
    lines.append("")
    lines.append("## 测试集")
    lines.append("")
    lines.append("- `constant_72000`")
    lines.append("- `cosine_72000`")
    lines.append("- `wsd_20000_24000`")
    lines.append("- `wsdld_20000_24000`")
    lines.append("- `wsdcon_3`")
    lines.append("- `wsdcon_18`")
    lines.append("")
    lines.append("## 平均测试 MAE")
    lines.append("")
    lines.append("| Model | Optimizer | 25M | 100M | 400M |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for model_name in ["tissue", "mpl"]:
        for optimizer_name in ["lbfgsb", "adamw"]:
            values = []
            for scale in scales:
                subset = [
                    row
                    for row in rows
                    if row["split"] == "test" and row["model"] == model_name and row["optimizer"] == optimizer_name and row["scale"] == scale
                ]
                values.append(f"{np.mean([float(row['mae']) for row in subset]):.6f}")
            padded = values + ["-"] * max(0, 3 - len(values))
            lines.append(f"| {model_name.upper()} | {optimizer_name.upper()} | {padded[0]} | {padded[1]} | {padded[2]} |")
    lines.append("")
    lines.append("## 平均训练 MAE")
    lines.append("")
    lines.append("| Model | Optimizer | 25M | 100M | 400M |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for model_name in ["tissue", "mpl"]:
        for optimizer_name in ["lbfgsb", "adamw"]:
            values = []
            for scale in scales:
                subset = [
                    row
                    for row in rows
                    if row["split"] == "train" and row["model"] == model_name and row["optimizer"] == optimizer_name and row["scale"] == scale
                ]
                values.append(f"{np.mean([float(row['mae']) for row in subset]):.6f}")
            padded = values + ["-"] * max(0, 3 - len(values))
            lines.append(f"| {model_name.upper()} | {optimizer_name.upper()} | {padded[0]} | {padded[1]} | {padded[2]} |")
    lines.append("")
    lines.append("## 当前观察")
    lines.append("")
    lines.append("- 如果同一公式在两种优化器下的平均测试误差接近，说明优化器影响较小，结构项主导结果。")
    lines.append("- 如果同一公式在训练误差差异较大、但测试误差差异有限，说明优化器主要影响拟合紧度，而不是泛化规律。")
    lines.append("- 参数对比图用于检查不同优化器是否收敛到明显不同的最优参数区域。")
    lines.append("")
    lines.append("## 输出文件")
    lines.append("")
    lines.append("- `results/optimizer_effect/tables/optimizer_effect_metrics.csv`")
    lines.append("- `results/optimizer_effect/tables/optimizer_effect_param_runs.csv`")
    lines.append("- `results/optimizer_effect/figures/test_mae_summary.png`")
    lines.append("- `results/optimizer_effect/figures/train_mae_summary.png`")
    lines.append("- `results/optimizer_effect/figures/tissue_best_params.png`")
    lines.append("- `results/optimizer_effect/figures/mpl_best_params.png`")
    lines.append("- `results/optimizer_effect/figures/examples/`")
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimizer effect experiment for Tissue and MPL.")
    parser.add_argument("--scales", nargs="+", default=["100"], choices=SCALES)
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, object]] = []
    param_rows: list[dict[str, object]] = []
    best_params: dict[str, dict[str, dict[str, np.ndarray]]] = {
        "tissue": {},
        "mpl": {},
    }

    for scale in args.scales:
        print(f"[info] optimizer-effect scale={scale}M")
        train_curves = [load_curve(scale, name) for name in TRAIN_CURVES]
        fit_curves = [subsample_curve(curve) for curve in train_curves]
        test_curves = [load_curve(scale, name) for name in TEST_CURVES]

        t_inits = [tissue_inits(fit_curves)[0]]
        m_inits = [mpl_inits(fit_curves)[0]]

        tissue_lbfgsb, tissue_lbfgsb_obj, tissue_lbfgsb_runs = fit_tissue_lbfgsb(fit_curves, t_inits)
        tissue_adamw, tissue_adamw_obj, tissue_adamw_runs = fit_adamw_tissue(fit_curves, t_inits)
        mpl_lbfgsb, mpl_lbfgsb_obj, mpl_lbfgsb_runs = fit_mpl_lbfgsb(fit_curves, m_inits)
        mpl_adamw, mpl_adamw_obj, mpl_adamw_runs = fit_adamw_mpl(fit_curves, m_inits)

        best_params["tissue"][scale] = {"lbfgsb": tissue_lbfgsb, "adamw": tissue_adamw}
        best_params["mpl"][scale] = {"lbfgsb": mpl_lbfgsb, "adamw": mpl_adamw}

        for optimizer_name, runs, param_names in [
            ("lbfgsb", tissue_lbfgsb_runs, ["L0", "A", "alpha", "C", "lam"]),
            ("adamw", tissue_adamw_runs, ["L0", "A", "alpha", "C", "lam"]),
        ]:
            for run in runs:
                for param_name, value in zip(param_names, run["params"]):
                    param_rows.append(
                        {
                            "model": "tissue",
                            "scale": scale,
                            "optimizer": optimizer_name,
                            "run_id": run["run_id"],
                            "objective": run["objective"],
                            "param_name": param_name,
                            "value": value,
                        }
                    )

        for optimizer_name, runs, param_names in [
            ("lbfgsb", mpl_lbfgsb_runs, ["L0", "A", "alpha", "B", "C", "beta", "gamma"]),
            ("adamw", mpl_adamw_runs, ["L0", "A", "alpha", "B", "C", "beta", "gamma"]),
        ]:
            for run in runs:
                for param_name, value in zip(param_names, run["params"]):
                    param_rows.append(
                        {
                            "model": "mpl",
                            "scale": scale,
                            "optimizer": optimizer_name,
                            "run_id": run["run_id"],
                            "objective": run["objective"],
                            "param_name": param_name,
                            "value": value,
                        }
                    )

        for split, curves in [("train", train_curves), ("test", test_curves)]:
            for curve in curves:
                for model_name, optimizer_name, params in [
                    ("tissue", "lbfgsb", tissue_lbfgsb),
                    ("tissue", "adamw", tissue_adamw),
                    ("mpl", "lbfgsb", mpl_lbfgsb),
                    ("mpl", "adamw", mpl_adamw),
                ]:
                    row = {
                        "scale": scale,
                        "split": split,
                        "model": model_name,
                        "optimizer": optimizer_name,
                        "curve": curve.name,
                    }
                    row.update(evaluate_model(model_name, params, curve))
                    metric_rows.append(row)

        # Representative examples at 100M only to avoid too many figures.
        if scale == "100":
            name_map = {curve.name: curve for curve in train_curves + test_curves}
            for model_name in ["tissue", "mpl"]:
                params_lbfgsb = best_params[model_name][scale]["lbfgsb"]
                params_adamw = best_params[model_name][scale]["adamw"]
                for curve_name in ["wsdcon_9.csv", "wsdcon_3.csv", "wsd_20000_24000.csv"]:
                    save_example_plot(model_name, scale, name_map[curve_name], params_lbfgsb, params_adamw)

    (OUT_ROOT / "tables").mkdir(parents=True, exist_ok=True)
    with (OUT_ROOT / "tables" / "optimizer_effect_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        fieldnames = ["scale", "split", "model", "optimizer", "curve", "mae", "rmse", "mape", "r2", "huber_log"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in metric_rows:
            writer.writerow(row)

    with (OUT_ROOT / "tables" / "optimizer_effect_param_runs.csv").open("w", newline="", encoding="utf-8") as fh:
        fieldnames = ["model", "scale", "optimizer", "run_id", "objective", "param_name", "value"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in param_rows:
            writer.writerow(row)

    json_ready = {
        model_name: {
            scale: {opt: params.tolist() for opt, params in scale_map.items()}
            for scale, scale_map in model_map.items()
        }
        for model_name, model_map in best_params.items()
    }
    with (OUT_ROOT / "tables" / "optimizer_effect_best_params.json").open("w", encoding="utf-8") as fh:
        json.dump(json_ready, fh, indent=2)

    save_summary_plot(metric_rows, "test", "mae", OUT_ROOT / "figures" / "test_mae_summary.png", args.scales)
    save_summary_plot(metric_rows, "train", "mae", OUT_ROOT / "figures" / "train_mae_summary.png", args.scales)
    save_best_param_bars(best_params, "tissue", ["L0", "A", "alpha", "C", "lam"], args.scales)
    save_best_param_bars(best_params, "mpl", ["L0", "A", "alpha", "B", "C", "beta", "gamma"], args.scales)
    write_report(metric_rows, param_rows, best_params, args.scales)

    print(f"Finished optimizer effect experiment under {OUT_ROOT}")


if __name__ == "__main__":
    main()
