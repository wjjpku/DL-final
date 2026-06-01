#!/usr/bin/env python3
"""Plot x(step) summaries and loss-curve comparisons for G(x) variants."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "external" / "MultiPowerLaw" / "loss_curve_repo"
PARAMS_PATH = ROOT / "results" / "g_replacement_official" / "tables" / "official_g_replacement_params.json"
OUT_ROOT = ROOT / "results" / "g_replacement_official" / "diagnostics"

TRAIN_CURVES = [
    "cosine_24000.csv",
    "constant_24000.csv",
    "wsdcon_9.csv",
]
TEST_CURVES = [
    "constant_72000.csv",
    "cosine_72000.csv",
    "wsd_20000_24000.csv",
    "wsdld_20000_24000.csv",
    "wsdcon_3.csv",
    "wsdcon_18.csv",
]

VARIANT_COLORS = {
    "pow": "#F58518",
    "pow_theta": "#E45756",
}
VARIANT_LABELS = {
    "pow": "Power",
    "pow_theta": "Power+Theta",
}


@dataclass(frozen=True)
class Curve:
    name: str
    scale: str
    step: np.ndarray
    loss: np.ndarray
    lrs: np.ndarray


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
    path = DATA_ROOT / f"csv_{scale}" / file_name
    raw = np.genfromtxt(path, delimiter=",", skip_header=1)
    step = raw[:, 0].astype(int)
    loss = raw[:, 2].astype(float)
    if step.max() == 24000:
        mask = step < 24000
        step = step[mask]
        loss = loss[mask]
    return Curve(name=file_name, scale=scale, step=step, loss=loss, lrs=build_lrs(file_name))


def load_variant_params(scale: str, variants: list[str]) -> dict[str, np.ndarray]:
    payload = json.loads(PARAMS_PATH.read_text(encoding="utf-8"))
    scale_info = payload[scale]["variants"]
    return {
        variant: np.asarray(scale_info[variant]["params"], dtype=np.float64)
        for variant in variants
    }


def g_transform(x: np.ndarray, C: float, beta: float, theta: float, variant: str) -> np.ndarray:
    base_x = np.clip(x, 0.0, None)
    with np.errstate(over="ignore", invalid="ignore"):
        if variant == "pow":
            z = np.clip(C * base_x, 0.0, 1e18)
            term = 1.0 - np.power(1.0 + z, -beta)
        elif variant == "pow_theta":
            z = np.clip(C * np.power(base_x, theta), 0.0, 1e18)
            term = 1.0 - np.power(1.0 + z, -beta)
        else:
            raise ValueError(f"Unsupported variant: {variant}")
    return np.clip(np.nan_to_num(term, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def compute_ld(curve: Curve, params: np.ndarray, variant: str) -> np.ndarray:
    _, _, _, _, C, beta, gamma, theta = params
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
        x = np.power(hist, -gamma) * remain
        term = g_transform(x, C=C, beta=beta, theta=theta, variant=variant)
        ld[i] = np.sum(delta * term)
    return ld


def predict(curve: Curve, params: np.ndarray, variant: str) -> np.ndarray:
    L0, A, alpha, _, _, _, _, _ = params
    s1 = np.cumsum(curve.lrs)[curve.step]
    ld = compute_ld(curve, params, variant)
    B = params[3]
    return L0 + A * np.power(s1, -alpha) + B * ld


def compute_x_summary(curve: Curve, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gamma = params[6]
    lrs = curve.lrs
    lr_sum = np.cumsum(lrs)
    lr_gap = np.zeros(len(lrs), dtype=np.float64)
    lr_gap[1:] = np.diff(lrs)
    x_mean = np.zeros(len(curve.step), dtype=np.float64)
    x_max = np.zeros(len(curve.step), dtype=np.float64)

    for i, s in enumerate(curve.step):
        if s <= 0:
            continue
        hist = lrs[1 : s + 1]
        remain = lr_sum[s] - lr_sum[:s]
        x_terms = np.power(hist, -gamma) * remain
        weights = np.abs(lr_gap[1 : s + 1])
        if np.sum(weights) > 0:
            x_mean[i] = float(np.sum(weights * x_terms) / np.sum(weights))
        else:
            x_mean[i] = float(np.mean(x_terms))
        x_max[i] = float(np.max(x_terms))
    return x_mean, x_max


def plot_x_summary(scale: str, curves: list[Curve], params_map: dict[str, np.ndarray], out_file: Path) -> None:
    cols = 2
    rows = math.ceil(len(curves) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12.5, 4.0 * rows), squeeze=False)
    for ax, curve in zip(axes.flat, curves):
        for variant, params in params_map.items():
            x_mean, _ = compute_x_summary(curve, params)
            ax.plot(
                curve.step,
                np.clip(x_mean, 1e-12, None),
                label=f"{VARIANT_LABELS[variant]} weighted-mean x",
                linewidth=2.0,
                color=VARIANT_COLORS[variant],
            )
        ax.set_title(curve.name.replace(".csv", ""))
        ax.set_xlabel("Step")
        ax.set_ylabel("x(step)")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
    for ax in axes.flat[len(curves):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle(f"100M Test Curves: Weighted-Mean x(step) for Power vs Power+Theta" if scale == "100" else f"{scale}M Test Curves: Weighted-Mean x(step) for Power vs Power+Theta")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=160)
    plt.close(fig)


def plot_loss_compare(scale: str, curves: list[Curve], params_map: dict[str, np.ndarray], out_file: Path) -> None:
    cols = 2
    rows = math.ceil(len(curves) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12.5, 4.0 * rows), squeeze=False)
    for ax, curve in zip(axes.flat, curves):
        ax.plot(curve.step, curve.loss, label="Ground Truth", linewidth=2.3, color="#222222")
        for variant, params in params_map.items():
            pred = predict(curve, params, variant)
            ax.plot(
                curve.step,
                pred,
                label=VARIANT_LABELS[variant],
                linewidth=2.0,
                linestyle="--" if variant == "pow" else "-.",
                color=VARIANT_COLORS[variant],
            )
        ax.set_title(curve.name.replace(".csv", ""))
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25)
    for ax in axes.flat[len(curves):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle(f"100M Test Curves: Loss Comparison for Power vs Power+Theta" if scale == "100" else f"{scale}M Test Curves: Loss Comparison for Power vs Power+Theta")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=160)
    plt.close(fig)


def write_curve_csv(
    curve: Curve,
    params_map: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "step",
            "loss",
            "pow_x_mean",
            "pow_x_max",
            "pow_theta_x_mean",
            "pow_theta_x_max",
            "pow_pred",
            "pow_theta_pred",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        pow_x_mean, pow_x_max = compute_x_summary(curve, params_map["pow"])
        theta_x_mean, theta_x_max = compute_x_summary(curve, params_map["pow_theta"])
        pow_pred = predict(curve, params_map["pow"], "pow")
        theta_pred = predict(curve, params_map["pow_theta"], "pow_theta")
        for i, step in enumerate(curve.step):
            writer.writerow(
                {
                    "step": int(step),
                    "loss": float(curve.loss[i]),
                    "pow_x_mean": float(pow_x_mean[i]),
                    "pow_x_max": float(pow_x_max[i]),
                    "pow_theta_x_mean": float(theta_x_mean[i]),
                    "pow_theta_x_max": float(theta_x_max[i]),
                    "pow_pred": float(pow_pred[i]),
                    "pow_theta_pred": float(theta_pred[i]),
                }
            )


def run(scale: str, split: str) -> dict[str, Path]:
    curve_names = TEST_CURVES if split == "test" else TRAIN_CURVES
    curves = [load_curve(scale, name) for name in curve_names]
    params_map = load_variant_params(scale, ["pow", "pow_theta"])

    base_dir = OUT_ROOT / f"{scale}M" / split
    x_fig = base_dir / f"{scale}M_{split}_x_summary.png"
    loss_fig = base_dir / f"{scale}M_{split}_loss_compare.png"
    plot_x_summary(scale, curves, params_map, x_fig)
    plot_loss_compare(scale, curves, params_map, loss_fig)

    for curve in curves:
        write_curve_csv(
            curve,
            params_map,
            base_dir / "csv" / f"{scale}M_{split}_{curve.name}",
        )
    return {"x_fig": x_fig, "loss_fig": loss_fig, "base_dir": base_dir}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot x(step) and loss comparisons for pow vs pow_theta.")
    parser.add_argument("--scale", default="100", choices=["25", "100", "400"])
    parser.add_argument("--split", default="test", choices=["train", "test"])
    args = parser.parse_args()
    result = run(scale=args.scale, split=args.split)
    print(f"Saved x(step) figure to {result['x_fig']}")
    print(f"Saved loss figure to {result['loss_fig']}")


if __name__ == "__main__":
    main()
