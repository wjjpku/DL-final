#!/usr/bin/env python3
"""Proxy schedule search for a 72k -> 144k continual-learning setup."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
BEST_PARAMS_PATH = ROOT / "results" / "weighted_scheme_compare" / "tables" / "best_params.json"
OUT_ROOT = ROOT / "results" / "continual_schedule_144k"
DOC_PATH = ROOT / "docs" / "continual_schedule_144k.md"

SCHEME_NAME = "constant_wsdcon_1_3_3"
SCALES = ["25", "100", "400"]
PREFIX_STEPS = 72_000
SUFFIX_STEPS = 72_000
TOTAL_STEPS = PREFIX_STEPS + SUFFIX_STEPS
PEAK_LR = 3e-4
MIN_LR = 3e-5
PREFIX_KNOTS = 96
SUFFIX_KNOTS = 96
PREFIX_OPT_STEPS = 2200
SUFFIX_OPT_STEPS = 2600
PREFIX_OPT_LR = 0.08
SUFFIX_OPT_LR = 0.06
PREFIX_PATIENCE = 260
SUFFIX_PATIENCE = 320
CURVE_POINTS = 480
TORCH_DTYPE = torch.float64
INITIAL_SUFFIX_LOGIT = -8.0


@dataclass
class ContinualResult:
    scale: str
    prefix_72k_loss: float
    baseline_144k_loss: float
    optimized_144k_loss: float
    improvement_pct: float


def load_best_params() -> dict[str, list[float]]:
    payload = json.loads(BEST_PARAMS_PATH.read_text(encoding="utf-8"))
    return payload[SCHEME_NAME]


def interpolate_knots(knot_lrs: torch.Tensor, total_steps: int) -> torch.Tensor:
    knot_positions = torch.linspace(0, total_steps - 1, steps=knot_lrs.numel(), dtype=TORCH_DTYPE, device=knot_lrs.device)
    full_positions = torch.arange(total_steps, dtype=TORCH_DTYPE, device=knot_lrs.device)
    indices = torch.bucketize(full_positions, knot_positions[1:-1])
    left_pos = knot_positions[indices]
    right_pos = knot_positions[indices + 1]
    left_lr = knot_lrs[indices]
    right_lr = knot_lrs[indices + 1]
    t = (full_positions - left_pos) / (right_pos - left_pos + 1e-12)
    return left_lr * (1 - t) + right_lr * t


def build_monotone_schedule_from_logits(logits: torch.Tensor, total_steps: int) -> torch.Tensor:
    deltas = (PEAK_LR - MIN_LR) * torch.softmax(logits, dim=0)
    knot_lrs = torch.cat(
        [
            torch.tensor([PEAK_LR], dtype=TORCH_DTYPE, device=logits.device),
            PEAK_LR - torch.cumsum(deltas, dim=0),
        ]
    )
    lrs = interpolate_knots(knot_lrs, total_steps)
    lrs[0] = PEAK_LR
    lrs[-1] = MIN_LR
    return lrs


def build_continual_suffix_from_logits(logits: torch.Tensor, total_steps: int = SUFFIX_STEPS) -> torch.Tensor:
    interior = MIN_LR + (PEAK_LR - MIN_LR) * torch.sigmoid(logits)
    knot_lrs = torch.cat(
        [
            torch.tensor([MIN_LR], dtype=TORCH_DTYPE, device=logits.device),
            interior,
            torch.tensor([MIN_LR], dtype=TORCH_DTYPE, device=logits.device),
        ]
    )
    lrs = interpolate_knots(knot_lrs, total_steps)
    lrs[0] = MIN_LR
    lrs[-1] = MIN_LR
    return lrs


def final_predicted_loss_torch(lrs: torch.Tensor, params: list[float]) -> torch.Tensor:
    L0, A, alpha, B, C, beta, gamma = [torch.tensor(x, dtype=TORCH_DTYPE, device=lrs.device) for x in params]
    lr_sum = torch.cumsum(lrs, dim=0)
    total_sum = lr_sum[-1]
    lr_gap = lrs[1:] - lrs[:-1]
    future_budget = total_sum - lr_sum[:-1]
    response = 1 - (1 + C * torch.clamp(lrs[1:], min=1e-12) ** (-gamma) * future_budget) ** (-beta)
    ld = torch.sum(lr_gap * response)
    pred = L0 + A * torch.clamp(total_sum, min=1e-12) ** (-alpha) + B * ld
    return torch.clamp(pred, min=1e-12)


def predicted_curve_numpy(lrs: np.ndarray, params: list[float], steps: np.ndarray) -> np.ndarray:
    L0, A, alpha, B, C, beta, gamma = params
    lr_sum = np.cumsum(lrs)
    lr_gap = np.zeros(len(lrs), dtype=np.float64)
    lr_gap[1:] = np.diff(lrs)
    preds = np.zeros(len(steps), dtype=np.float64)
    for i, s in enumerate(steps):
        s = int(s)
        if s <= 0:
            preds[i] = L0 + A * max(lrs[0], 1e-12) ** (-alpha)
            continue
        ld = np.sum(
            lr_gap[1 : s + 1]
            * (1 - (1 + C * np.clip(lrs[1 : s + 1], 1e-12, None) ** (-gamma) * (lr_sum[s] - lr_sum[:s])) ** (-beta))
        )
        preds[i] = L0 + A * max(lr_sum[s], 1e-12) ** (-alpha) + B * ld
    return np.clip(preds, 1e-12, None)


def optimize_prefix_schedule(params: list[float]) -> tuple[np.ndarray, list[float]]:
    device = torch.device("cpu")
    logits = torch.zeros(PREFIX_KNOTS - 1, dtype=TORCH_DTYPE, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=PREFIX_OPT_LR)
    best_loss = float("inf")
    best_logits = logits.detach().clone()
    history: list[float] = []
    no_improve = 0

    for _ in range(PREFIX_OPT_STEPS):
        optimizer.zero_grad()
        lrs = build_monotone_schedule_from_logits(logits, total_steps=PREFIX_STEPS)
        pred = final_predicted_loss_torch(lrs, params)
        pred.backward()
        optimizer.step()

        value = float(pred.item())
        history.append(value)
        if value + 1e-12 < best_loss:
            best_loss = value
            best_logits = logits.detach().clone()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= PREFIX_PATIENCE:
            break

    best_lrs = build_monotone_schedule_from_logits(best_logits, total_steps=PREFIX_STEPS).detach().cpu().numpy()
    return best_lrs, history


def optimize_continual_suffix(params: list[float], prefix_lrs: np.ndarray) -> tuple[np.ndarray, list[float]]:
    device = torch.device("cpu")
    logits = torch.full((SUFFIX_KNOTS - 2,), INITIAL_SUFFIX_LOGIT, dtype=TORCH_DTYPE, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=SUFFIX_OPT_LR)
    prefix_tensor = torch.tensor(prefix_lrs, dtype=TORCH_DTYPE, device=device)
    best_loss = float("inf")
    best_logits = logits.detach().clone()
    history: list[float] = []
    no_improve = 0

    for _ in range(SUFFIX_OPT_STEPS):
        optimizer.zero_grad()
        suffix = build_continual_suffix_from_logits(logits)
        full_lrs = torch.cat([prefix_tensor, suffix], dim=0)
        pred = final_predicted_loss_torch(full_lrs, params)
        pred.backward()
        optimizer.step()

        value = float(pred.item())
        history.append(value)
        if value + 1e-12 < best_loss:
            best_loss = value
            best_logits = logits.detach().clone()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= SUFFIX_PATIENCE:
            break

    best_suffix = build_continual_suffix_from_logits(best_logits).detach().cpu().numpy()
    return best_suffix, history


def save_outputs(scale: str, prefix_lrs: np.ndarray, suffix_init: np.ndarray, suffix_opt: np.ndarray, params: list[float], prefix_history: list[float], suffix_history: list[float]) -> ContinualResult:
    scale_dir = OUT_ROOT / f"{scale}M"
    scale_dir.mkdir(parents=True, exist_ok=True)

    baseline_full = np.concatenate([prefix_lrs, suffix_init])
    optimized_full = np.concatenate([prefix_lrs, suffix_opt])

    prefix_pred = predicted_curve_numpy(prefix_lrs, params, np.array([PREFIX_STEPS - 1], dtype=int))[0]
    baseline_final = predicted_curve_numpy(baseline_full, params, np.array([TOTAL_STEPS - 1], dtype=int))[0]
    optimized_final = predicted_curve_numpy(optimized_full, params, np.array([TOTAL_STEPS - 1], dtype=int))[0]
    improvement_pct = 100.0 * (baseline_final - optimized_final) / baseline_final

    steps = np.linspace(0, TOTAL_STEPS - 1, CURVE_POINTS, dtype=int)
    baseline_curve = predicted_curve_numpy(baseline_full, params, steps)
    optimized_curve = predicted_curve_numpy(optimized_full, params, steps)

    prefix_schedule_csv = scale_dir / "prefix_schedule_72k.csv"
    with prefix_schedule_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "prefix_optimized_lr"])
        for step, lr in enumerate(prefix_lrs, start=1):
            writer.writerow([step, lr])

    suffix_schedule_csv = scale_dir / "continual_schedule_compare.csv"
    with suffix_schedule_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "baseline_full_lr", "optimized_full_lr"])
        for step, lr_a, lr_b in zip(range(1, TOTAL_STEPS + 1), baseline_full, optimized_full):
            writer.writerow([step, lr_a, lr_b])

    curve_csv = scale_dir / "continual_predicted_curve_compare.csv"
    with curve_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "baseline_pred_loss", "optimized_pred_loss"])
        for step, loss_a, loss_b in zip(steps + 1, baseline_curve, optimized_curve):
            writer.writerow([int(step), loss_a, loss_b])

    plt.figure(figsize=(9.2, 4.8))
    plt.plot(baseline_full, label="72k prefix + all-min suffix", linewidth=2.0)
    plt.plot(optimized_full, label="72k prefix + optimized suffix", linewidth=2.0)
    plt.axvline(PREFIX_STEPS, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title(f"{scale}M Continual Schedule (0-144k)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(scale_dir / "continual_schedule_full.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9.2, 4.8))
    suffix_steps = np.arange(PREFIX_STEPS + 1, TOTAL_STEPS + 1)
    plt.plot(suffix_steps, suffix_init, label="All-min init suffix", linewidth=2.0)
    plt.plot(suffix_steps, suffix_opt, label="Optimized suffix", linewidth=2.0)
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title(f"{scale}M Continual Suffix (72k-144k)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(scale_dir / "continual_schedule_suffix_zoom.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9.2, 4.8))
    plt.plot(steps + 1, baseline_curve, label="72k prefix + all-min suffix", linewidth=2.0)
    plt.plot(steps + 1, optimized_curve, label="72k prefix + optimized suffix", linewidth=2.0)
    plt.axvline(PREFIX_STEPS, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    plt.xlabel("Step")
    plt.ylabel("Predicted Loss")
    plt.title(f"{scale}M Predicted Loss (0-144k)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(scale_dir / "continual_predicted_curve_full.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.8, 4.4))
    plt.plot(prefix_history, linewidth=1.7)
    plt.xlabel("Optimization Step")
    plt.ylabel("Predicted 72k Final Loss")
    plt.title(f"{scale}M Prefix Optimization")
    plt.tight_layout()
    plt.savefig(scale_dir / "prefix_optimization_history.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.8, 4.4))
    plt.plot(suffix_history, linewidth=1.7)
    plt.xlabel("Optimization Step")
    plt.ylabel("Predicted 144k Final Loss")
    plt.title(f"{scale}M Continual Suffix Optimization")
    plt.tight_layout()
    plt.savefig(scale_dir / "continual_suffix_optimization_history.png", dpi=180)
    plt.close()

    return ContinualResult(
        scale=scale,
        prefix_72k_loss=float(prefix_pred),
        baseline_144k_loss=float(baseline_final),
        optimized_144k_loss=float(optimized_final),
        improvement_pct=float(improvement_pct),
    )


def write_report(results: list[ContinualResult]) -> None:
    lines = [
        "# 72k 到 144k 的连续学习后段调度搜索",
        "",
        "本实验继续使用当前最优预测器：`133` 加权方案下的 `MPL` 参数。",
        "",
        "## 设定",
        "",
        "- 先单独学习一个 `0-72k` 的前段最优单调递减 schedule，约束仍为 `lr_max = 3e-4` 到 `lr_min = 3e-5`。",
        "- 将这条 `72k` 前缀固定，并把它对应的预测 loss 作为后续连续学习问题的历史前段。",
        "- 在 `72k-144k` 段，只优化新的后段 schedule。",
        "- 后段初始化为全程 `lr_min = 3e-5`。",
        "- 后段第 `72k` 点与第 `144k` 点都固定为 `lr_min`。",
        "- 后段中间位置允许在 `[lr_min, lr_max]` 内上抬，从而形成可能的再加热形状。",
        "",
        "## 目标",
        "",
        "- 以前 `72k` 固定前缀为条件，最小化第 `144k` 步的代理预测 loss。",
        "- 对照基线是：前 `72k` 使用同一最优前缀，后 `72k` 全部维持 `lr_min` 不变。",
        "",
        "## 结果总表",
        "",
        "| Scale | Pred Loss @72k Prefix End | Pred Loss @144k All-min Suffix | Pred Loss @144k Optimized Suffix | Improvement |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for item in results:
        lines.append(
            f"| {item.scale}M | {item.prefix_72k_loss:.6f} | {item.baseline_144k_loss:.6f} | {item.optimized_144k_loss:.6f} | {item.improvement_pct:.2f}% |"
        )
    lines += [
        "",
        "## 输出目录",
        "",
        "- `results/continual_schedule_144k/<scale>M/prefix_schedule_72k.csv`",
        "- `results/continual_schedule_144k/<scale>M/continual_schedule_compare.csv`",
        "- `results/continual_schedule_144k/<scale>M/continual_predicted_curve_compare.csv`",
        "- `results/continual_schedule_144k/<scale>M/continual_schedule_full.png`",
        "- `results/continual_schedule_144k/<scale>M/continual_schedule_suffix_zoom.png`",
        "- `results/continual_schedule_144k/<scale>M/continual_predicted_curve_full.png`",
        "",
        "## 解释边界",
        "",
        "- 这是在原有 `133-MPL` 代理模型上的 schedule search，不是带真实新数据重新训练后的真实最优控制。",
        "- 这里的“新一批数据进来”被代理成：前 `72k` 训练历史固定，后 `72k` 允许重新设计学习率曲线。",
        "- 本实验没有显式建模数据分布变化、遗忘约束或额外正则，只研究给定代理损失下的最优后段学习率形状。",
        "",
    ]
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    best_params = load_best_params()
    results: list[ContinualResult] = []
    summary_rows: list[dict[str, float | str]] = []

    for scale in SCALES:
        params = best_params[scale]
        prefix_lrs, prefix_history = optimize_prefix_schedule(params)
        suffix_init = np.full(SUFFIX_STEPS, MIN_LR, dtype=np.float64)
        suffix_opt, suffix_history = optimize_continual_suffix(params, prefix_lrs)
        result = save_outputs(scale, prefix_lrs, suffix_init, suffix_opt, params, prefix_history, suffix_history)
        results.append(result)
        summary_rows.append(
            {
                "scale": scale,
                "prefix_72k_loss": result.prefix_72k_loss,
                "baseline_144k_loss": result.baseline_144k_loss,
                "optimized_144k_loss": result.optimized_144k_loss,
                "improvement_pct": result.improvement_pct,
            }
        )

    with (OUT_ROOT / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["scale", "prefix_72k_loss", "baseline_144k_loss", "optimized_144k_loss", "improvement_pct"],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    write_report(results)
    print(f"Finished continual schedule search under {OUT_ROOT}")


if __name__ == "__main__":
    main()
