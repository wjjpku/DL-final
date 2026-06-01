#!/usr/bin/env python3
"""Reproduce cosine-to-WSD loss-curve prediction experiments on Apple Silicon.

This script compares:
1. Scaling Law with Learning Rate Annealing (Tissue et al., 2024)
2. Multi-Power Law (Luo et al., 2025 repository; user-requested as Luo et al., 2024)

It fits on cosine learning-rate schedules only, then evaluates on WSD-family
schedules. The script uses only CPU-friendly NumPy/SciPy/Matplotlib routines.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "external" / "MultiPowerLaw" / "loss_curve_repo"
RESULT_ROOT = ROOT / "results"

WARMUP = 2160
PEAK_LR = 3e-4
END_LR = 3e-5

TRAIN_CURVES = ["cosine_24000.csv", "cosine_72000.csv"]
TEST_CURVES = [
    "wsd_20000_24000.csv",
    "wsdld_20000_24000.csv",
    "wsdcon_3.csv",
    "wsdcon_9.csv",
    "wsdcon_18.csv",
]

SCALES = ["25", "100", "400"]
FIT_SUBSAMPLE_STRIDE = 4
SCIPY_MAXITER = 400
MPL_PRECOMPUTED_INIT = {
    "25": np.array([3.04045406, 0.52468604, 0.50786857, 363.78751622, 2.06560812, 0.58279013, 0.64142257]),
    "100": np.array([2.6514477, 0.60115152, 0.45295811, 437.9464276, 2.13245612, 0.59785199, 0.65523644]),
    "400": np.array([2.37474466, 0.65421216, 0.42878731, 523.42464371, 2.02462735, 0.59350493, 0.63472457]),
}


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
        return cosine_lrs(WARMUP, total, PEAK_LR, END_LR)
    if "constant" in file_name:
        total = int(file_name.split("_")[1].split(".")[0])
        return const_lrs(WARMUP, total, PEAK_LR)
    if "wsdcon" in file_name:
        total = 16000
        lr_b = int(file_name.split("_")[1].split(".")[0]) * 1e-5
        return two_stage_lrs(WARMUP, total, PEAK_LR, lr_b, 8000)
    if "wsdld" in file_name:
        return wsdld_lrs(WARMUP, 24000, 20000, PEAK_LR, END_LR)
    if "wsd" in file_name:
        return wsd_lrs(WARMUP, 24000, 20000, PEAK_LR, END_LR)
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


def huber_log_residual(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1e-3) -> float:
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


def fit_with_restarts(
    objective_factory: Callable[[np.ndarray], float],
    inits: list[np.ndarray],
    bounds: list[tuple[float, float]],
) -> tuple[np.ndarray, float]:
    best_x = None
    best_fun = float("inf")
    for init in inits:
        res = minimize(
            objective_factory,
            x0=np.asarray(init, dtype=np.float64),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": SCIPY_MAXITER, "ftol": 1e-10},
        )
        if res.fun < best_fun:
            best_fun = float(res.fun)
            best_x = res.x
    assert best_x is not None
    return best_x, best_fun


def fit_tissue(curves: list[Curve]) -> tuple[np.ndarray, float]:
    min_loss = min(float(curve.loss.min()) for curve in curves)
    inits = []
    for lam in [0.99, 0.995, 0.997, 0.999]:
        inits.append(np.array([min_loss - 0.05, 0.5, 0.5, 100.0, lam]))
        inits.append(np.array([min_loss - 0.1, 1.0, 0.4, 10.0, lam]))
        inits.append(np.array([min_loss, 0.2, 0.7, 300.0, lam]))

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
    return fit_with_restarts(objective, inits, bounds)


def fit_mpl(curves: list[Curve], scale: str) -> tuple[np.ndarray, float]:
    min_loss = min(float(curve.loss.min()) for curve in curves)
    inits = [
        MPL_PRECOMPUTED_INIT[scale],
        np.array([min_loss - 0.05, 0.5, 0.5, 300.0, 2.0, 0.5, 0.5]),
    ]

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
    return fit_with_restarts(objective, inits, bounds)


def save_prediction_plot(
    scale: str,
    model_name: str,
    curve: Curve,
    pred: np.ndarray,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.8))
    plt.plot(curve.step, curve.loss, label="Ground Truth", linewidth=2)
    plt.plot(curve.step, pred, label=f"{model_name} Prediction", linestyle="--", linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"{model_name} on {curve.name} ({scale}M)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{scale}_{model_name}_{curve.name.replace('.csv', '')}.png", dpi=160)
    plt.close()


def save_summary_barplot(rows: list[dict[str, object]], metric: str, out_file: Path) -> None:
    scale_order = ["25", "100", "400"]
    model_order = ["tissue", "mpl"]
    curve_order = ["wsd_20000_24000.csv", "wsdld_20000_24000.csv", "wsdcon_3.csv", "wsdcon_9.csv", "wsdcon_18.csv"]
    labels = []
    values = []
    colors = []
    for scale in scale_order:
        for model in model_order:
            subset = [
                row for row in rows
                if row["split"] == "test" and row["scale"] == scale and row["model"] == model and row["curve"] in curve_order
            ]
            if not subset:
                continue
            avg = float(np.mean([float(row[metric]) for row in subset]))
            labels.append(f"{scale}M-{model}")
            values.append(avg)
            colors.append("#4C78A8" if model == "tissue" else "#F58518")
    plt.figure(figsize=(10, 4.8))
    plt.bar(labels, values, color=colors)
    plt.ylabel(metric.upper())
    plt.title(f"Average {metric.upper()} on WSD-family Test Curves")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_file, dpi=160)
    plt.close()


def save_cross_scale_plot(rows: list[dict[str, object]], curve_name: str, metric: str, out_file: Path) -> None:
    plt.figure(figsize=(7.5, 4.8))
    for model, color in [("tissue", "#4C78A8"), ("mpl", "#F58518")]:
        xs = []
        ys = []
        for scale in SCALES:
            matched = [
                item
                for item in rows
                if item["split"] == "test"
                and item["curve"] == curve_name
                and item["scale"] == scale
                and item["model"] == model
            ]
            if not matched:
                continue
            row = matched[0]
            xs.append(int(scale))
            ys.append(float(row[metric]))
        if not xs:
            continue
        plt.plot(xs, ys, marker="o", label=model.upper(), color=color)
    plt.xlabel("Model Size (M)")
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} on {curve_name.replace('.csv', '')}")
    plt.xticks([25, 100, 400])
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=160)
    plt.close()


def save_predictions_csv(
    scale: str,
    model_name: str,
    curve: Curve,
    pred: np.ndarray,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{scale}_{model_name}_{curve.name}"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "loss", "pred"])
        for s, y, p in zip(curve.step, curve.loss, pred):
            writer.writerow([int(s), float(y), float(p)])


def run_experiment(scales: list[str]) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    params_summary: dict[str, dict[str, list[float] | float]] = {}

    for scale in scales:
        print(f"[info] running scale={scale}M")
        train_curves = [load_curve(scale, name) for name in TRAIN_CURVES]
        fit_curves = [subsample_curve(curve) for curve in train_curves]
        test_curves = [load_curve(scale, name) for name in TEST_CURVES]

        tissue_params, tissue_obj = fit_tissue(fit_curves)
        mpl_params, mpl_obj = fit_mpl(fit_curves, scale)
        params_summary[scale] = {
            "tissue_params": tissue_params.tolist(),
            "tissue_objective": tissue_obj,
            "mpl_params": mpl_params.tolist(),
            "mpl_objective": mpl_obj,
        }

        for split, curves in [("train", train_curves), ("test", test_curves)]:
            for curve in curves:
                for model_name, pred_fn, params in [
                    ("tissue", tissue_predict, tissue_params),
                    ("mpl", mpl_predict, mpl_params),
                ]:
                    pred = pred_fn(params, curve)
                    row = {
                        "scale": scale,
                        "split": split,
                        "model": model_name,
                        "curve": curve.name,
                    }
                    row.update(metrics(curve.loss, pred))
                    rows.append(row)
                    save_prediction_plot(scale, model_name, curve, pred, RESULT_ROOT / "figures")
                    save_predictions_csv(scale, model_name, curve, pred, RESULT_ROOT / "predictions")

    table_path = RESULT_ROOT / "tables" / "cosine_to_wsd_metrics.csv"
    with table_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = ["scale", "split", "model", "curve", "mae", "rmse", "mape", "r2", "huber_log"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    (RESULT_ROOT / "tables").mkdir(parents=True, exist_ok=True)
    with (RESULT_ROOT / "tables" / "fitted_params.json").open("w", encoding="utf-8") as fh:
        json.dump(params_summary, fh, indent=2)

    save_summary_barplot(rows, "mae", RESULT_ROOT / "figures" / "avg_test_mae.png")
    save_summary_barplot(rows, "rmse", RESULT_ROOT / "figures" / "avg_test_rmse.png")
    save_cross_scale_plot(rows, "wsd_20000_24000.csv", "mae", RESULT_ROOT / "figures" / "wsd_mae_by_scale.png")
    save_cross_scale_plot(rows, "wsd_20000_24000.csv", "r2", RESULT_ROOT / "figures" / "wsd_r2_by_scale.png")

    return {"rows": rows, "params_summary": params_summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce cosine-to-WSD experiments.")
    parser.add_argument(
        "--scales",
        nargs="+",
        default=SCALES,
        choices=SCALES,
        help="Subset of model sizes to run.",
    )
    args = parser.parse_args()
    result = run_experiment(args.scales)
    test_rows = [row for row in result["rows"] if row["split"] == "test"]
    print(f"Finished. Generated {len(test_rows)} test rows and figures under {RESULT_ROOT}.")


if __name__ == "__main__":
    main()
