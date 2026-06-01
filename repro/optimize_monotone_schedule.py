#!/usr/bin/env python3
"""Optimize monotone LR schedules with fixed endpoints using the best weighted MPL model."""

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
OUT_ROOT = ROOT / "results" / "optimized_monotone_schedule"
DOC_PATH = ROOT / "docs" / "optimized_monotone_schedule.md"

SCHEME_NAME = "constant_wsdcon_1_3_3"
SCALES = ["25", "100", "400"]
TOTAL_STEPS = 24_000
PEAK_LR = 3e-4
MIN_LR = 3e-5
NUM_KNOTS = 64
OPT_STEPS = 3000
OPT_LR = 0.08
PATIENCE = 300
CURVE_POINTS = 240
TORCH_DTYPE = torch.float64


@dataclass
class OptimizationResult:
    scale: str
    linear_final_loss: float
    optimized_final_loss: float
    improvement_pct: float
    schedule_path: Path
    curve_path: Path
    loss_path: Path


def load_best_params() -> dict[str, list[float]]:
    payload = json.loads(BEST_PARAMS_PATH.read_text(encoding="utf-8"))
    return payload[SCHEME_NAME]


def build_schedule_from_logits(logits: torch.Tensor, total_steps: int = TOTAL_STEPS) -> torch.Tensor:
    knot_count = logits.numel() + 1
    deltas = (PEAK_LR - MIN_LR) * torch.softmax(logits, dim=0)
    knot_lrs = torch.cat(
        [
            torch.tensor([PEAK_LR], dtype=TORCH_DTYPE, device=logits.device),
            PEAK_LR - torch.cumsum(deltas, dim=0),
        ]
    )
    knot_positions = torch.linspace(0, total_steps - 1, steps=knot_count, dtype=TORCH_DTYPE, device=logits.device)
    full_positions = torch.arange(total_steps, dtype=TORCH_DTYPE, device=logits.device)
    indices = torch.bucketize(full_positions, knot_positions[1:-1])
    left_pos = knot_positions[indices]
    right_pos = knot_positions[indices + 1]
    left_lr = knot_lrs[indices]
    right_lr = knot_lrs[indices + 1]
    t = (full_positions - left_pos) / (right_pos - left_pos + 1e-12)
    lrs = left_lr * (1 - t) + right_lr * t
    lrs[0] = PEAK_LR
    lrs[-1] = MIN_LR
    return lrs


def final_predicted_loss_torch(lrs: torch.Tensor, params: list[float]) -> torch.Tensor:
    L0, A, alpha, B, C, beta, gamma = [torch.tensor(x, dtype=TORCH_DTYPE, device=lrs.device) for x in params]
    total_sum = torch.sum(lrs)
    prefix_sum = torch.cumsum(lrs, dim=0)
    future_budget = total_sum - prefix_sum[:-1]
    delta = lrs[:-1] - lrs[1:]
    response = 1 - (1 + C * lrs[1:] ** (-gamma) * future_budget) ** (-beta)
    ld_positive = torch.sum(delta * response)
    pred = L0 + A * total_sum ** (-alpha) - B * ld_positive
    return torch.clamp(pred, min=1e-12)


def predicted_curve_numpy(lrs: np.ndarray, params: list[float], steps: np.ndarray) -> np.ndarray:
    L0, A, alpha, B, C, beta, gamma = params
    lr_sum = np.cumsum(lrs)
    lr_gap = np.zeros(len(lrs))
    lr_gap[1:] = np.diff(lrs)
    preds = np.zeros(len(steps), dtype=np.float64)
    for i, s in enumerate(steps):
        if s <= 0:
            preds[i] = L0 + A * max(lrs[0], 1e-12) ** (-alpha)
            continue
        ld = np.sum(
            lr_gap[1 : s + 1]
            * (1 - (1 + C * lrs[1 : s + 1] ** (-gamma) * (lr_sum[s] - lr_sum[:s])) ** (-beta))
        )
        preds[i] = L0 + A * lr_sum[s] ** (-alpha) + B * ld
    return np.clip(preds, 1e-12, None)


def optimize_schedule(params: list[float]) -> tuple[np.ndarray, list[float]]:
    device = torch.device("cpu")
    logits = torch.zeros(NUM_KNOTS - 1, dtype=TORCH_DTYPE, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=OPT_LR)
    best_loss = float("inf")
    best_logits = logits.detach().clone()
    history: list[float] = []
    no_improve = 0

    for _ in range(OPT_STEPS):
        optimizer.zero_grad()
        lrs = build_schedule_from_logits(logits)
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

        if no_improve >= PATIENCE:
            break

    best_lrs = build_schedule_from_logits(best_logits).detach().cpu().numpy()
    return best_lrs, history


def save_scale_outputs(scale: str, linear_lrs: np.ndarray, optimized_lrs: np.ndarray, params: list[float], history: list[float]) -> OptimizationResult:
    scale_dir = OUT_ROOT / f"{scale}M"
    scale_dir.mkdir(parents=True, exist_ok=True)
    steps = np.linspace(0, TOTAL_STEPS - 1, CURVE_POINTS, dtype=int)
    linear_curve = predicted_curve_numpy(linear_lrs, params, steps)
    optimized_curve = predicted_curve_numpy(optimized_lrs, params, steps)
    linear_final = float(linear_curve[-1])
    optimized_final = float(optimized_curve[-1])
    improvement_pct = 100.0 * (linear_final - optimized_final) / linear_final

    schedule_csv = scale_dir / "schedule_compare.csv"
    with schedule_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "linear_lr", "optimized_lr"])
        for step, lr_a, lr_b in zip(range(TOTAL_STEPS), linear_lrs, optimized_lrs):
            writer.writerow([step, lr_a, lr_b])

    curve_csv = scale_dir / "predicted_curve_compare.csv"
    with curve_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "linear_pred_loss", "optimized_pred_loss"])
        for step, loss_a, loss_b in zip(steps, linear_curve, optimized_curve):
            writer.writerow([int(step), loss_a, loss_b])

    plt.figure(figsize=(8.2, 4.8))
    plt.plot(linear_lrs, label="Linear Init", linewidth=2.0)
    plt.plot(optimized_lrs, label="Optimized", linewidth=2.0)
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title(f"{scale}M Schedule Compare")
    plt.legend()
    plt.tight_layout()
    schedule_path = scale_dir / "schedule_compare.png"
    plt.savefig(schedule_path, dpi=180)
    plt.close()

    plt.figure(figsize=(8.2, 4.8))
    plt.plot(steps, linear_curve, label="Linear Init", linewidth=2.0)
    plt.plot(steps, optimized_curve, label="Optimized", linewidth=2.0)
    plt.xlabel("Step")
    plt.ylabel("Predicted Loss")
    plt.title(f"{scale}M Predicted Loss Curve")
    plt.legend()
    plt.tight_layout()
    curve_path = scale_dir / "predicted_curve_compare.png"
    plt.savefig(curve_path, dpi=180)
    plt.close()

    plt.figure(figsize=(7.6, 4.4))
    plt.plot(history, linewidth=1.8)
    plt.xlabel("Optimization Step")
    plt.ylabel("Predicted Final Loss")
    plt.title(f"{scale}M Schedule Optimization")
    plt.tight_layout()
    loss_path = scale_dir / "optimization_history.png"
    plt.savefig(loss_path, dpi=180)
    plt.close()

    return OptimizationResult(
        scale=scale,
        linear_final_loss=linear_final,
        optimized_final_loss=optimized_final,
        improvement_pct=improvement_pct,
        schedule_path=schedule_path,
        curve_path=curve_path,
        loss_path=loss_path,
    )


def write_report(results: list[OptimizationResult]) -> None:
    lines = [
        "# 固定端点下的最优单调递减调度搜索",
        "",
        "本实验使用当前最优预测器：`133` 加权方案下的 `MPL` 参数。",
        "",
        "约束条件：",
        "",
        "- 固定初始学习率 `lr_max = 3e-4`",
        "- 固定最终学习率 `lr_min = 3e-5`",
        "- 固定总步数 `24000`",
        "- 调度必须单调不增",
        "- 初始化为线性递减调度",
        "",
        "优化方法：",
        "",
        "- 用 `64` 个控制点参数化整条单调 schedule",
        "- 控制点之间线性插值，保证起点和终点精确满足约束",
        "- 直接最小化 `MPL` 预测的最终一步 loss",
        "",
        "## 结果总表",
        "",
        "| Scale | Linear Final Loss | Optimized Final Loss | Improvement |",
        "| --- | ---: | ---: | ---: |",
    ]
    for item in results:
        lines.append(
            f"| {item.scale}M | {item.linear_final_loss:.6f} | {item.optimized_final_loss:.6f} | {item.improvement_pct:.2f}% |"
        )
    lines += [
        "",
        "## 输出目录",
        "",
        "- `results/optimized_monotone_schedule/<scale>M/schedule_compare.png`",
        "- `results/optimized_monotone_schedule/<scale>M/predicted_curve_compare.png`",
        "- `results/optimized_monotone_schedule/<scale>M/optimization_history.png`",
        "- `results/optimized_monotone_schedule/<scale>M/schedule_compare.csv`",
        "- `results/optimized_monotone_schedule/<scale>M/predicted_curve_compare.csv`",
        "",
    ]
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    best_params = load_best_params()
    linear_lrs = np.linspace(PEAK_LR, MIN_LR, TOTAL_STEPS, dtype=np.float64)
    results: list[OptimizationResult] = []
    summary_rows: list[dict[str, float | str]] = []

    for scale in SCALES:
        params = best_params[scale]
        optimized_lrs, history = optimize_schedule(params)
        result = save_scale_outputs(scale, linear_lrs, optimized_lrs, params, history)
        results.append(result)
        summary_rows.append(
            {
                "scale": scale,
                "linear_final_loss": result.linear_final_loss,
                "optimized_final_loss": result.optimized_final_loss,
                "improvement_pct": result.improvement_pct,
            }
        )

    with (OUT_ROOT / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["scale", "linear_final_loss", "optimized_final_loss", "improvement_pct"])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    write_report(results)
    print(f"Finished schedule optimization under {OUT_ROOT}")


if __name__ == "__main__":
    main()
