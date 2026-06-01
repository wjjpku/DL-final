#!/usr/bin/env python3
"""Compare alternative G(x) formulations on the official MultiPowerLaw split."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "external" / "MultiPowerLaw" / "loss_curve_repo"
MPL_ROOT = ROOT / "external" / "MultiPowerLaw"
RESULT_ROOT = ROOT / "results" / "g_replacement_official"

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
SCALES = ["25", "100", "400"]
VARIANTS = ["pow", "pow_theta", "hill", "weibull"]
DEFAULT_SCIPY_MAXITER = 400
DEFAULT_FIT_SUBSAMPLE_STRIDE = 4

VARIANT_LABELS = {
    "pow": "Power",
    "pow_theta": "Power+Theta",
    "hill": "Hill",
    "weibull": "Weibull",
}
VARIANT_COLORS = {
    "pow": "#F58518",
    "pow_theta": "#E45756",
    "hill": "#54A24B",
    "weibull": "#4C78A8",
}


@dataclass(frozen=True)
class Curve:
    name: str
    scale: str
    step: np.ndarray
    loss: np.ndarray
    lrs: np.ndarray


def subsample_curve(curve: Curve, stride: int = DEFAULT_FIT_SUBSAMPLE_STRIDE) -> Curve:
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


def load_mpl_params_from_logs(scale: str) -> np.ndarray | None:
    log_path = MPL_ROOT / "logs" / f"{scale}.log"
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8")
    matches = re.findall(r"Best Parameters: (\[[^\]]+\])", text)
    if not matches:
        return None
    return np.asarray(json.loads(matches[-1]), dtype=np.float64)


def uses_theta(variant: str) -> bool:
    return variant == "pow_theta"


def g_transform(x: np.ndarray, C: float, beta: float, theta: float, variant: str) -> np.ndarray:
    base_x = np.clip(x, 0.0, None)
    with np.errstate(over="ignore", invalid="ignore"):
        if variant == "pow":
            z = np.clip(C * base_x, 0.0, 1e18)
            term = 1.0 - np.power(1.0 + z, -beta)
        elif variant == "pow_theta":
            z = np.clip(C * np.power(base_x, theta), 0.0, 1e18)
            term = 1.0 - np.power(1.0 + z, -beta)
        elif variant == "hill":
            z = np.clip(C * base_x, 0.0, 1e18)
            z_beta = np.power(z, beta)
            term = z_beta / (1.0 + z_beta)
        elif variant == "weibull":
            z = np.clip(C * base_x, 0.0, 1e18)
            term = 1.0 - np.exp(-np.power(z, beta))
        else:
            raise ValueError(f"Unknown variant: {variant}")
    return np.clip(np.nan_to_num(term, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def compute_ld(
    curve: Curve, C: float, beta: float, gamma: float, theta: float, variant: str
) -> np.ndarray:
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


def predict(params: np.ndarray, curve: Curve, variant: str) -> np.ndarray:
    L0, A, alpha, B, C, beta, gamma, theta = params
    s1 = compute_s1(curve)
    ld = compute_ld(curve, C=C, beta=beta, gamma=gamma, theta=theta, variant=variant)
    return L0 + A * np.power(s1, -alpha) + B * ld


def make_initial_guesses(scale: str, curves: list[Curve], variant: str) -> list[np.ndarray]:
    min_loss = min(float(curve.loss.min()) for curve in curves)
    log_init = load_mpl_params_from_logs(scale)
    guesses = [
        np.array([min_loss - 0.05, 0.5, 0.5, 300.0, 2.0, 0.5, 0.5, 1.0]),
        np.array([min_loss - 0.1, 1.0, 0.4, 100.0, 1.0, 1.0, 0.8, 1.0]),
        np.array([min_loss, 0.2, 0.7, 600.0, 4.0, 0.3, 0.3, 1.0]),
    ]
    if log_init is not None:
        theta_init = 1.0
        if uses_theta(variant):
            guesses.insert(0, np.concatenate([log_init.astype(np.float64), np.array([theta_init])]))
        else:
            guesses.insert(0, np.concatenate([log_init.astype(np.float64), np.array([1.0])]))
    return guesses


def fit_with_restarts(
    objective_factory,
    init_guesses: list[np.ndarray],
    bounds: list[tuple[float, float]],
    maxiter: int,
) -> tuple[np.ndarray, float]:
    best_x = None
    best_fun = float("inf")
    for init in init_guesses:
        res = minimize(
            objective_factory,
            x0=np.asarray(init, dtype=np.float64),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-10},
        )
        if res.fun < best_fun:
            best_fun = float(res.fun)
            best_x = res.x
    assert best_x is not None
    return best_x, best_fun


def fit_variant(
    curves: list[Curve], scale: str, variant: str, maxiter: int
) -> tuple[np.ndarray, float]:
    init_guesses = make_initial_guesses(scale, curves, variant)

    def objective(params: np.ndarray) -> float:
        pred_all = []
        loss_all = []
        for curve in curves:
            pred = predict(params, curve, variant)
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
        (0.2, 3.0) if uses_theta(variant) else (1.0, 1.0),
    ]
    return fit_with_restarts(objective, init_guesses, bounds, maxiter=maxiter)


def select_variant_with_loocv(
    train_curves: list[Curve], scale: str, variants: list[str], maxiter: int
) -> tuple[str, dict[str, dict[str, float]], list[dict[str, object]]]:
    score_rows: list[dict[str, object]] = []
    score_map = {
        variant: {"mae": [], "rmse": [], "huber_log": []}
        for variant in variants
    }
    cache: dict[tuple[str, tuple[str, ...]], tuple[np.ndarray, float]] = {}

    for held_out in train_curves:
        subset = [curve for curve in train_curves if curve.name != held_out.name]
        subset_key = tuple(curve.name for curve in subset)
        for variant in variants:
            cache_key = (variant, subset_key)
            if cache_key not in cache:
                cache[cache_key] = fit_variant(subset, scale, variant, maxiter=maxiter)
            params, objective = cache[cache_key]
            pred = predict(params, held_out, variant)
            fold_metrics = metrics(held_out.loss, pred)
            score_map[variant]["mae"].append(fold_metrics["mae"])
            score_map[variant]["rmse"].append(fold_metrics["rmse"])
            score_map[variant]["huber_log"].append(fold_metrics["huber_log"])
            score_rows.append(
                {
                    "scale": scale,
                    "fold_curve": held_out.name,
                    "variant": variant,
                    "objective": objective,
                    **fold_metrics,
                }
            )

    summary = {
        variant: {metric: float(np.mean(values)) for metric, values in scores.items()}
        for variant, scores in score_map.items()
    }
    selected_variant = min(variants, key=lambda name: (summary[name]["mae"], summary[name]["rmse"]))
    return selected_variant, summary, score_rows


def save_prediction_csv(
    scale: str,
    split: str,
    curve: Curve,
    prediction_map: dict[str, np.ndarray],
) -> None:
    out_dir = RESULT_ROOT / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{scale}_{split}_{curve.name}"
    fieldnames = ["step", "loss"] + [f"{variant}_pred" for variant in prediction_map]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for idx, step in enumerate(curve.step):
            row = {
                "step": int(step),
                "loss": float(curve.loss[idx]),
            }
            for variant, pred in prediction_map.items():
                row[f"{variant}_pred"] = float(pred[idx])
            writer.writerow(row)


def save_compare_plot(
    scale: str,
    split: str,
    curve: Curve,
    prediction_map: dict[str, np.ndarray],
    selected_variant: str,
) -> None:
    out_dir = RESULT_ROOT / "figures" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.4, 4.8))
    plt.plot(curve.step, curve.loss, label="Ground Truth", linewidth=2.3, color="#222222")
    for variant, pred in prediction_map.items():
        linewidth = 2.3 if variant == selected_variant else 1.8
        linestyle = "-" if variant == selected_variant else "--"
        label = f"{VARIANT_LABELS[variant]}"
        if variant == selected_variant:
            label = f"{label} (selected)"
        plt.plot(
            curve.step,
            pred,
            label=label,
            linewidth=linewidth,
            linestyle=linestyle,
            color=VARIANT_COLORS[variant],
        )
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"G(x) replacement on {curve.name.replace('.csv', '')} ({scale}M, {split})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{scale}_{curve.name.replace('.csv', '')}.png", dpi=160)
    plt.close()


def save_summary_plot(rows: list[dict[str, object]]) -> None:
    out_dir = RESULT_ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.6, 4.8))
    for variant in VARIANTS:
        xs = []
        ys = []
        for scale in SCALES:
            subset = [
                row for row in rows
                if row["split"] == "test" and row["scale"] == scale and row["variant"] == variant
            ]
            if not subset:
                continue
            xs.append(int(scale))
            ys.append(float(np.mean([float(row["mae"]) for row in subset])))
        if xs:
            plt.plot(xs, ys, marker="o", label=VARIANT_LABELS[variant], color=VARIANT_COLORS[variant])
    plt.xticks([25, 100, 400])
    plt.xlabel("Model Size (M)")
    plt.ylabel("Average Test MAE")
    plt.title("Official Split: G(x) Replacement Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "official_g_replacement_avg_test_mae.png", dpi=160)
    plt.close()


def save_selection_plot(selection_summary: dict[str, dict[str, dict[str, float]]]) -> None:
    out_dir = RESULT_ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.6, 4.8))
    for variant in VARIANTS:
        xs = []
        ys = []
        for scale in SCALES:
            if scale not in selection_summary or variant not in selection_summary[scale]:
                continue
            xs.append(int(scale))
            ys.append(float(selection_summary[scale][variant]["mae"]))
        if xs:
            plt.plot(xs, ys, marker="o", label=VARIANT_LABELS[variant], color=VARIANT_COLORS[variant])
    plt.xticks([25, 100, 400])
    plt.xlabel("Model Size (M)")
    plt.ylabel("LOO CV MAE on Train Curves")
    plt.title("Train-Curve Leave-One-Out Variant Selection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "official_g_replacement_loocv_mae.png", dpi=160)
    plt.close()


def run(scales: list[str], variants: list[str], fit_stride: int, maxiter: int) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    cv_rows: list[dict[str, object]] = []
    params_summary: dict[str, dict[str, object]] = {}
    selection_summary: dict[str, dict[str, dict[str, float]]] = {}

    for scale in scales:
        print(f"[info] g-replacement scale={scale}M")
        train_curves = [load_curve(scale, name) for name in TRAIN_CURVES]
        fit_curves = [subsample_curve(curve, stride=fit_stride) for curve in train_curves]
        test_curves = [load_curve(scale, name) for name in TEST_CURVES]

        selected_variant, cv_summary, scale_cv_rows = select_variant_with_loocv(
            fit_curves, scale, variants, maxiter=maxiter
        )
        selection_summary[scale] = cv_summary
        cv_rows.extend(scale_cv_rows)
        params_summary[scale] = {
            "selected_variant": selected_variant,
            "cv_summary": cv_summary,
            "variants": {},
        }

        fitted_params: dict[str, np.ndarray] = {}
        for variant in variants:
            params, objective = fit_variant(fit_curves, scale, variant, maxiter=maxiter)
            fitted_params[variant] = params
            params_summary[scale]["variants"][variant] = {
                "params": params.tolist(),
                "objective": objective,
            }

        for split, curves in [("train", train_curves), ("test", test_curves)]:
            for curve in curves:
                prediction_map = {
                    variant: predict(fitted_params[variant], curve, variant)
                    for variant in variants
                }
                for variant, pred in prediction_map.items():
                    row = {
                        "scale": scale,
                        "split": split,
                        "variant": variant,
                        "curve": curve.name,
                        "selected": variant == selected_variant,
                    }
                    row.update(metrics(curve.loss, pred))
                    rows.append(row)
                save_prediction_csv(scale, split, curve, prediction_map)
                save_compare_plot(scale, split, curve, prediction_map, selected_variant)

    (RESULT_ROOT / "tables").mkdir(parents=True, exist_ok=True)
    with (RESULT_ROOT / "tables" / "official_g_replacement_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        fieldnames = [
            "scale",
            "split",
            "variant",
            "curve",
            "selected",
            "mae",
            "rmse",
            "mape",
            "r2",
            "huber_log",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with (RESULT_ROOT / "tables" / "official_g_replacement_cv.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        fieldnames = ["scale", "fold_curve", "variant", "objective", "mae", "rmse", "mape", "r2", "huber_log"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in cv_rows:
            writer.writerow(row)

    with (RESULT_ROOT / "tables" / "official_g_replacement_params.json").open(
        "w", encoding="utf-8"
    ) as fh:
        json.dump(params_summary, fh, indent=2)

    save_summary_plot(rows)
    save_selection_plot(selection_summary)
    return {
        "rows": rows,
        "cv_rows": cv_rows,
        "params_summary": params_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare alternative G(x) forms on the official public split.")
    parser.add_argument("--scales", nargs="+", default=SCALES, choices=SCALES)
    parser.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    parser.add_argument("--fit-stride", type=int, default=DEFAULT_FIT_SUBSAMPLE_STRIDE)
    parser.add_argument("--maxiter", type=int, default=DEFAULT_SCIPY_MAXITER)
    args = parser.parse_args()

    result = run(args.scales, args.variants, fit_stride=args.fit_stride, maxiter=args.maxiter)
    selected = {
        scale: info["selected_variant"]
        for scale, info in result["params_summary"].items()
    }
    print(f"Finished. Selected variants by scale: {selected}")
    print(f"Artifacts written to {RESULT_ROOT}")


if __name__ == "__main__":
    main()
