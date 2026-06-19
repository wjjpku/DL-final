#!/usr/bin/env python3
"""Fast shape-projection audit for strict cosine-only MPL backbones.

The strict finite-response result exposed a backbone problem: independent
cosine-only MPL fits improve after the finite-response correction, but their
absolute WSD error remains far worse than the frozen official MPL reference.

This audit tests a lightweight, interpretable repair for that bottleneck:
project weakly identified LD-kernel shape parameters to a cross-scale shared
value estimated from cosine-only fits, then refit only the remaining per-scale
MPL backbone parameters on cosine curves.  The downstream error correction is
unchanged and still has zero residual-fitted parameters.

This is intentionally not a performance search.  It is a falsifiable diagnosis
of whether cosine-only MPL's WSD failure is mainly a scale-wise LD-shape
identifiability problem.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

TMP_CACHE = Path(os.environ.get("TMPDIR", "/tmp")) / "dl-final-cache"
TMP_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(TMP_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(TMP_CACHE / "xdg"))

from scipy.optimize import minimize

REPO = Path(__file__).resolve().parent
ROOT = REPO.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from reproduce_cosine_to_wsd import (  # noqa: E402
    MPL_PRECOMPUTED_INIT,
    SCALES,
    TRAIN_CURVES,
    WARMUP,
    huber_log_residual,
    load_curve,
    metrics,
    mpl_predict,
    subsample_curve,
)


OUT_DIR = ROOT / "results" / "mpl_ld_lag_response_audit" / "shape_projection_backbone"
STRICT_PARAM_JSON = (
    ROOT
    / "results"
    / "cosine_to_wsd_response_search"
    / "cosine_only_backbone"
    / "cosine_only_mpl_params.json"
)
MAXITER = 160

CORE_TARGETS = [
    ("core_wsd", "wsd_20000_24000.csv", "WSD sharp"),
    ("core_wsd", "wsdld_20000_24000.csv", "WSD linear"),
    ("core_wsd", "wsdcon_3.csv", "WSD-con 3e-5"),
    ("core_wsd", "wsdcon_9.csv", "WSD-con 9e-5"),
    ("core_wsd", "wsdcon_18.csv", "WSD-con 18e-5"),
]
EXTRA_CONTROLS = [
    ("extra_control", "cosine_24000.csv", "Cosine 24k"),
    ("extra_control", "constant_24000.csv", "Constant 24k"),
    ("extra_control", "constant_72000.csv", "Constant 72k"),
]
ALL_TARGETS = CORE_TARGETS + EXTRA_CONTROLS


@dataclass(frozen=True)
class VariantSpec:
    name: str
    backbone_params: int
    residual_params: int
    description: str


SPECS = {
    "frozen_official": VariantSpec(
        "frozen_official",
        21,
        0,
        "official public-split MPL reference",
    ),
    "cosine_independent": VariantSpec(
        "cosine_independent",
        21,
        0,
        "independent cosine-only MPL fits",
    ),
    "median_beta_gamma_refit": VariantSpec(
        "median_beta_gamma_refit",
        17,
        0,
        "beta,gamma shared by cosine-only cross-scale median; L0,A,alpha,B,C refit per scale",
    ),
    "median_c_beta_gamma_refit": VariantSpec(
        "median_c_beta_gamma_refit",
        15,
        0,
        "C,beta,gamma shared by cosine-only cross-scale median; L0,A,alpha,B refit per scale",
    ),
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt_pct(value: float) -> str:
    return f"{value:+.2f}%"


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_s1(curve) -> np.ndarray:
    return np.cumsum(curve.lrs)[curve.step]


def compute_ld(curve, c_value: float, beta: float, gamma: float) -> np.ndarray:
    lrs = curve.lrs.astype(np.float64)
    lr_sum = np.cumsum(lrs)
    lr_gap = np.zeros(len(lrs), dtype=np.float64)
    lr_gap[1:] = np.diff(lrs)
    ld = np.zeros(len(curve.step), dtype=np.float64)
    for idx, step in enumerate(curve.step):
        if step <= 0:
            continue
        hist = lrs[1 : step + 1]
        delta = lr_gap[1 : step + 1]
        remain = lr_sum[step] - lr_sum[:step]
        term = 1.0 - (1.0 + c_value * np.power(hist, -gamma) * remain) ** (-beta)
        ld[idx] = np.sum(delta * term)
    return ld


def predict(params: np.ndarray, curve) -> np.ndarray:
    l0, a_value, alpha, b_value, c_value, beta, gamma = params
    return l0 + a_value * np.power(compute_s1(curve), -alpha) + b_value * compute_ld(curve, c_value, beta, gamma)


def objective_for_curves(params: np.ndarray, curves: list[object]) -> float:
    pred_all: list[np.ndarray] = []
    loss_all: list[np.ndarray] = []
    for curve in curves:
        pred = predict(params, curve)
        if np.any(~np.isfinite(pred)) or np.any(pred <= 0):
            return 1e18
        pred_all.append(pred)
        loss_all.append(curve.loss)
    return huber_log_residual(np.concatenate(loss_all), np.concatenate(pred_all))


def read_independent_params() -> dict[str, np.ndarray]:
    raw = json.loads(STRICT_PARAM_JSON.read_text(encoding="utf-8"))
    return {scale: np.array(raw[scale], dtype=np.float64) for scale in SCALES}


def fit_fixed_beta_gamma(scale: str, init: np.ndarray, beta: float, gamma: float) -> tuple[np.ndarray, float]:
    curves = [subsample_curve(load_curve(scale, name)) for name in TRAIN_CURVES]

    def unpack(x: np.ndarray) -> np.ndarray:
        l0, a_value, alpha, b_value, c_value = x
        return np.array([l0, a_value, alpha, b_value, c_value, beta, gamma], dtype=np.float64)

    def objective(x: np.ndarray) -> float:
        return objective_for_curves(unpack(x), curves)

    starts = [
        np.array([init[0], init[1], init[2], init[3], init[4]], dtype=np.float64),
        np.array([min(float(curve.loss.min()) for curve in curves) - 0.05, 0.5, 0.5, init[3], init[4]], dtype=np.float64),
    ]
    bounds = [(0.0, 10.0), (1e-8, 100.0), (1e-4, 3.0), (1e-8, 1e5), (1e-8, 100.0)]
    best_x: np.ndarray | None = None
    best_fun = float("inf")
    for start in starts:
        res = minimize(
            objective,
            x0=start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": MAXITER, "ftol": 1e-10},
        )
        if float(res.fun) < best_fun:
            best_fun = float(res.fun)
            best_x = np.array(res.x, dtype=np.float64)
    if best_x is None:
        raise RuntimeError(f"failed fixed beta/gamma fit for {scale}")
    return unpack(best_x), best_fun


def fit_fixed_c_beta_gamma(scale: str, init: np.ndarray, c_value: float, beta: float, gamma: float) -> tuple[np.ndarray, float]:
    curves = [subsample_curve(load_curve(scale, name)) for name in TRAIN_CURVES]

    def unpack(x: np.ndarray) -> np.ndarray:
        l0, a_value, alpha, b_value = x
        return np.array([l0, a_value, alpha, b_value, c_value, beta, gamma], dtype=np.float64)

    def objective(x: np.ndarray) -> float:
        return objective_for_curves(unpack(x), curves)

    starts = [
        np.array([init[0], init[1], init[2], init[3]], dtype=np.float64),
        np.array([min(float(curve.loss.min()) for curve in curves) - 0.05, 0.5, 0.5, init[3]], dtype=np.float64),
    ]
    bounds = [(0.0, 10.0), (1e-8, 100.0), (1e-4, 3.0), (1e-8, 1e5)]
    best_x: np.ndarray | None = None
    best_fun = float("inf")
    for start in starts:
        res = minimize(
            objective,
            x0=start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": MAXITER, "ftol": 1e-10},
        )
        if float(res.fun) < best_fun:
            best_fun = float(res.fun)
            best_x = np.array(res.x, dtype=np.float64)
    if best_x is None:
        raise RuntimeError(f"failed fixed C/beta/gamma fit for {scale}")
    return unpack(best_x), best_fun


def fit_projected_backbones(independent: dict[str, np.ndarray]) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, object]]]:
    beta_med = float(np.median([independent[scale][5] for scale in SCALES]))
    gamma_med = float(np.median([independent[scale][6] for scale in SCALES]))
    c_med = float(np.median([independent[scale][4] for scale in SCALES]))

    params_by_variant: dict[str, dict[str, np.ndarray]] = {
        "frozen_official": {scale: np.array(MPL_PRECOMPUTED_INIT[scale], dtype=np.float64) for scale in SCALES},
        "cosine_independent": independent,
        "median_beta_gamma_refit": {},
        "median_c_beta_gamma_refit": {},
    }
    fit_rows: list[dict[str, object]] = []
    for scale in SCALES:
        params, obj = fit_fixed_beta_gamma(scale, independent[scale], beta_med, gamma_med)
        params_by_variant["median_beta_gamma_refit"][scale] = params
        fit_rows.append(
            {
                "variant": "median_beta_gamma_refit",
                "scale": scale,
                "objective": obj,
                "shared_c": "",
                "shared_beta": beta_med,
                "shared_gamma": gamma_med,
            }
        )

        params, obj = fit_fixed_c_beta_gamma(scale, independent[scale], c_med, beta_med, gamma_med)
        params_by_variant["median_c_beta_gamma_refit"][scale] = params
        fit_rows.append(
            {
                "variant": "median_c_beta_gamma_refit",
                "scale": scale,
                "objective": obj,
                "shared_c": c_med,
                "shared_beta": beta_med,
                "shared_gamma": gamma_med,
            }
        )

        for variant in ["frozen_official", "cosine_independent"]:
            curves = [subsample_curve(load_curve(scale, name)) for name in TRAIN_CURVES]
            fit_rows.append(
                {
                    "variant": variant,
                    "scale": scale,
                    "objective": objective_for_curves(params_by_variant[variant][scale], curves),
                    "shared_c": "",
                    "shared_beta": "",
                    "shared_gamma": "",
                }
            )
    return params_by_variant, fit_rows


def modal_observation_interval(step: np.ndarray) -> int:
    diffs = np.diff(step)
    values, counts = np.unique(diffs[diffs > 0], return_counts=True)
    if values.size == 0:
        return 1
    return int(values[int(np.argmax(counts))])


def cooldown_support_span(lrs: np.ndarray) -> int:
    eta = lrs.astype(np.float64)
    drop = np.zeros_like(eta)
    drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0)
    idx = np.flatnonzero(drop > 1e-18)
    return int(idx[-1] - idx[0] + 2) if idx.size else 0


def adiabatic_factor(lrs: np.ndarray) -> float:
    return max(0.0, 1.0 - float(cooldown_support_span(lrs)) / float(max(len(lrs) - WARMUP, 1)))


def support_bracket_tau(step: np.ndarray, lrs: np.ndarray) -> float:
    interval = float(modal_observation_interval(step))
    span = float(cooldown_support_span(lrs))
    return interval * (1.0 + min(1.0, span / max(interval, 1.0)))


def lagged_observed(values: np.ndarray, steps: np.ndarray, tau_steps: float) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    out[0] = float(values[0])
    for idx in range(1, len(values)):
        delta_steps = max(float(steps[idx] - steps[idx - 1]), 1.0)
        rho = math.exp(-delta_steps / max(float(tau_steps), 1e-12))
        out[idx] = rho * out[idx - 1] + (1.0 - rho) * float(values[idx])
    return out


def ld_cooldown_component(curve, params: np.ndarray) -> np.ndarray:
    _, _, _, _, c_value, beta, gamma = params
    lrs = curve.lrs.astype(np.float64)
    lr_sum = np.cumsum(lrs)
    lr_gap = np.zeros(len(lrs), dtype=np.float64)
    lr_gap[1:] = np.diff(lrs)
    selected_gap = np.minimum(lr_gap, 0.0)
    out = np.zeros(len(curve.step), dtype=np.float64)
    for idx, step in enumerate(curve.step):
        if step <= 0:
            continue
        hist = lrs[1 : step + 1]
        delta = selected_gap[1 : step + 1]
        remain = lr_sum[step] - lr_sum[:step]
        term = 1.0 - (1.0 + c_value * np.power(hist, -gamma) * remain) ** (-beta)
        out[idx] = np.sum(delta * term)
    return out


def finite_response_prediction(curve, params: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    baseline = predict(params, curve)
    d_down = ld_cooldown_component(curve, params)
    tau = support_bracket_tau(curve.step, curve.lrs)
    factor = adiabatic_factor(curve.lrs)
    d_lag = lagged_observed(d_down, curve.step, tau)
    return baseline + factor * float(params[3]) * (d_lag - d_down), {
        "effective_tau_steps": tau,
        "adiabatic_factor": factor,
        "cooldown_support_span": float(cooldown_support_span(curve.lrs)),
    }


def detail_rows(params_by_variant: dict[str, dict[str, np.ndarray]]) -> list[dict[str, object]]:
    official_mae: dict[tuple[str, str], float] = {}
    for scale in SCALES:
        for _, curve_name, _ in ALL_TARGETS:
            curve = load_curve(scale, curve_name)
            official_mae[(scale, curve_name)] = metrics(
                curve.loss,
                mpl_predict(MPL_PRECOMPUTED_INIT[scale], curve),
            )["mae"]

    rows: list[dict[str, object]] = []
    for variant, param_map in params_by_variant.items():
        for scale in SCALES:
            params = param_map[scale]
            for group, curve_name, label in ALL_TARGETS:
                curve = load_curve(scale, curve_name)
                baseline = predict(params, curve)
                corrected, features = finite_response_prediction(curve, params)
                base_mae = mae(curve.loss, baseline)
                corr_mae = mae(curve.loss, corrected)
                ref = official_mae[(scale, curve_name)]
                rows.append(
                    {
                        "variant": variant,
                        "description": SPECS[variant].description,
                        "backbone_params": SPECS[variant].backbone_params,
                        "residual_params": SPECS[variant].residual_params,
                        "group": group,
                        "scale": scale,
                        "test_curve": curve_name,
                        "test_label": label,
                        "base_mae": base_mae,
                        "corr_mae": corr_mae,
                        "delta_vs_own_baseline_pct": 100.0 * (corr_mae / base_mae - 1.0),
                        "base_vs_official_baseline_pct": 100.0 * (base_mae / ref - 1.0),
                        "corr_vs_official_baseline_pct": 100.0 * (corr_mae / ref - 1.0),
                        "win_vs_own_baseline": int(corr_mae < base_mae),
                        "nonharm_vs_own_baseline": int(corr_mae <= base_mae + 1e-12),
                        **features,
                    }
                )
    return rows


def parameter_rows(params_by_variant: dict[str, dict[str, np.ndarray]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variant, param_map in params_by_variant.items():
        for scale, params in param_map.items():
            rows.append(
                {
                    "variant": variant,
                    "scale": scale,
                    "backbone_params": SPECS[variant].backbone_params,
                    "residual_params": SPECS[variant].residual_params,
                    **{f"p{i}": float(value) for i, value in enumerate(params)},
                }
            )
    return rows


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for variant in sorted({str(row["variant"]) for row in rows}):
        for group in sorted({str(row["group"]) for row in rows if row["variant"] == variant}):
            sub = [row for row in rows if row["variant"] == variant and row["group"] == group]
            own = np.array([float(row["delta_vs_own_baseline_pct"]) for row in sub], dtype=np.float64)
            base_off = np.array([float(row["base_vs_official_baseline_pct"]) for row in sub], dtype=np.float64)
            corr_off = np.array([float(row["corr_vs_official_baseline_pct"]) for row in sub], dtype=np.float64)
            out.append(
                {
                    "variant": variant,
                    "description": SPECS[variant].description,
                    "backbone_params": SPECS[variant].backbone_params,
                    "residual_params": SPECS[variant].residual_params,
                    "group": group,
                    "rows": len(sub),
                    "mean_delta_vs_own_baseline": float(np.mean(own)),
                    "worst_delta_vs_own_baseline": float(np.max(own)),
                    "wins_vs_own_baseline": int(np.sum(own < 0.0)),
                    "nonharm_vs_own_baseline": int(np.sum(own <= 1e-12)),
                    "mean_base_vs_official_baseline": float(np.mean(base_off)),
                    "worst_base_vs_official_baseline": float(np.max(base_off)),
                    "mean_corr_vs_official_baseline": float(np.mean(corr_off)),
                    "worst_corr_vs_official_baseline": float(np.max(corr_off)),
                }
            )
    return out


def find(summary: list[dict[str, object]], variant: str, group: str) -> dict[str, object]:
    for row in summary:
        if row["variant"] == variant and row["group"] == group:
            return row
    raise KeyError((variant, group))


def write_report(summary: list[dict[str, object]], fit_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Fast MPL Backbone Shape-Projection Audit\n\n",
        "This audit keeps the error model fixed and tests whether strict cosine-only MPL is weak because its LD-kernel shape parameters are poorly identified by smooth cosine curves.\n\n",
        "The downstream correction remains:\n\n",
        "\\[\n",
        "\\hat L_s(t)=L_{\\mathrm{MPL},s}(t)+a_sB_s[D_{\\downarrow,\\tau_s,s}(t)-D_{\\downarrow,s}(t)].\n",
        "\\]\n\n",
        "No residual-fitted parameter is introduced.\n\n",
        "## Summary\n\n",
        "| variant | group | backbone params | correction vs own MPL | wins / non-harm | own MPL vs official | corrected vs official |\n",
        "|---|---|---:|---:|---:|---:|---:|\n",
    ]
    for variant in [
        "frozen_official",
        "cosine_independent",
        "median_beta_gamma_refit",
        "median_c_beta_gamma_refit",
    ]:
        for group_label, group in [("WSD-family", "core_wsd"), ("controls", "extra_control")]:
            row = find(summary, variant, group)
            lines.append(
                f"| {variant} | {group_label} | {int(row['backbone_params'])} | "
                f"{fmt_pct(float(row['mean_delta_vs_own_baseline']))} mean / "
                f"{fmt_pct(float(row['worst_delta_vs_own_baseline']))} worst | "
                f"{int(row['wins_vs_own_baseline'])}/{int(row['rows'])} / "
                f"{int(row['nonharm_vs_own_baseline'])}/{int(row['rows'])} | "
                f"{fmt_pct(float(row['mean_base_vs_official_baseline']))} mean | "
                f"{fmt_pct(float(row['mean_corr_vs_official_baseline']))} mean |\n"
            )

    lines += [
        "\n## Cosine Train Objective\n\n",
        "| variant | scale | objective | shared C | shared beta | shared gamma |\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ]
    for row in sorted(fit_rows, key=lambda item: (str(item["variant"]), str(item["scale"]))):
        lines.append(
            f"| {row['variant']} | {row['scale']} | {float(row['objective']):.8g} | "
            f"{row['shared_c']} | {row['shared_beta']} | {row['shared_gamma']} |\n"
        )

    lines += [
        "\n## Reading\n\n",
        "- This is a backbone identifiability probe, not a new residual estimator.\n",
        "- A good outcome would reduce corrected strict-cosine WSD error versus `cosine_independent` without losing 15/15 non-harm against its own MPL baseline.\n",
        "- If the projection worsens WSD, it means simple cross-scale shape sharing is too crude; the finite-response error formula should remain unchanged while the backbone problem is handled separately.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    independent = read_independent_params()
    params_by_variant, fit_rows = fit_projected_backbones(independent)
    details = detail_rows(params_by_variant)
    summary = aggregate(details)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "params.csv", parameter_rows(params_by_variant))
    write_csv(OUT_DIR / "fit_objectives.csv", fit_rows)
    write_csv(OUT_DIR / "details.csv", details)
    write_csv(OUT_DIR / "summary.csv", summary)
    write_report(summary, fit_rows)
    print(f"wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
