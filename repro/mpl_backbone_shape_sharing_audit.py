#!/usr/bin/env python3
"""Cosine-only MPL backbone audit with shared LD-shape parameters.

The strict cosine-only finite-response audit showed a different bottleneck:
the correction helps every WSD-family target, but the independently fitted
cosine-only MPL backbone is much weaker than the frozen official MPL backbone.

This script tests an interpretable backbone repair, not a new residual model.
MPL's LR-dependent kernel shape parameters are weakly identified by smooth
cosine schedules.  We therefore compare independent per-scale MPL fits against
cosine-only fits that share LD-shape parameters across model scales:

* shared_beta_gamma: each scale has L0, A, alpha, B, C; beta and gamma shared;
* shared_c_beta_gamma: each scale has L0, A, alpha, B; C, beta, gamma shared.

Both variants reduce backbone degrees of freedom and use only cosine training
losses.  The downstream error correction remains the same zero-residual-param
cooldown finite-response formula.
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


OUT_DIR = ROOT / "results" / "mpl_ld_lag_response_audit" / "shape_shared_backbone"
STRICT_PARAM_JSON = (
    ROOT
    / "results"
    / "cosine_to_wsd_response_search"
    / "cosine_only_backbone"
    / "cosine_only_mpl_params.json"
)
MAXITER = 600
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
class FitSpec:
    name: str
    backbone_params: int
    residual_params: int
    description: str


SPECS = {
    "frozen_official": FitSpec(
        "frozen_official",
        backbone_params=21,
        residual_params=0,
        description="official public-split MPL parameters; diagnostic reference",
    ),
    "cosine_independent": FitSpec(
        "cosine_independent",
        backbone_params=21,
        residual_params=0,
        description="three independent MPL fits from cosine_24000 + cosine_72000",
    ),
    "shared_beta_gamma": FitSpec(
        "shared_beta_gamma",
        backbone_params=17,
        residual_params=0,
        description="per-scale L0,A,alpha,B,C with shared beta,gamma",
    ),
    "shared_c_beta_gamma": FitSpec(
        "shared_c_beta_gamma",
        backbone_params=15,
        residual_params=0,
        description="per-scale L0,A,alpha,B with shared C,beta,gamma",
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


def read_independent_params() -> dict[str, np.ndarray]:
    if STRICT_PARAM_JSON.exists():
        raw = json.loads(STRICT_PARAM_JSON.read_text(encoding="utf-8"))
        return {scale: np.array(raw[scale], dtype=np.float64) for scale in SCALES}
    raise FileNotFoundError(f"missing strict cosine-only params: {STRICT_PARAM_JSON}")


def train_curves() -> dict[str, list[object]]:
    return {
        scale: [subsample_curve(load_curve(scale, name)) for name in TRAIN_CURVES]
        for scale in SCALES
    }


def generic_scale_start(scale: str, curves: list[object], beta: float, gamma: float, c_value: float = 2.0) -> np.ndarray:
    min_loss = min(float(curve.loss.min()) for curve in curves)
    b_guess = {"25": 360.0, "100": 440.0, "400": 520.0}[scale]
    return np.array([min_loss - 0.05, 0.55, 0.50, b_guess, c_value, beta, gamma], dtype=np.float64)


def pack_shared_beta_gamma(params: dict[str, np.ndarray]) -> np.ndarray:
    beta = float(np.median([params[scale][5] for scale in SCALES]))
    gamma = float(np.median([params[scale][6] for scale in SCALES]))
    chunks = [params[scale][:5] for scale in SCALES]
    return np.concatenate(chunks + [np.array([beta, gamma], dtype=np.float64)])


def unpack_shared_beta_gamma(x: np.ndarray) -> dict[str, np.ndarray]:
    beta, gamma = float(x[-2]), float(x[-1])
    out: dict[str, np.ndarray] = {}
    pos = 0
    for scale in SCALES:
        l0, a_value, alpha, b_value, c_value = x[pos : pos + 5]
        out[scale] = np.array([l0, a_value, alpha, b_value, c_value, beta, gamma], dtype=np.float64)
        pos += 5
    return out


def pack_shared_c_beta_gamma(params: dict[str, np.ndarray]) -> np.ndarray:
    c_value = float(np.median([params[scale][4] for scale in SCALES]))
    beta = float(np.median([params[scale][5] for scale in SCALES]))
    gamma = float(np.median([params[scale][6] for scale in SCALES]))
    chunks = [params[scale][:4] for scale in SCALES]
    return np.concatenate(chunks + [np.array([c_value, beta, gamma], dtype=np.float64)])


def unpack_shared_c_beta_gamma(x: np.ndarray) -> dict[str, np.ndarray]:
    c_value, beta, gamma = float(x[-3]), float(x[-2]), float(x[-1])
    out: dict[str, np.ndarray] = {}
    pos = 0
    for scale in SCALES:
        l0, a_value, alpha, b_value = x[pos : pos + 4]
        out[scale] = np.array([l0, a_value, alpha, b_value, c_value, beta, gamma], dtype=np.float64)
        pos += 4
    return out


def bounds_shared_beta_gamma() -> list[tuple[float, float]]:
    per_scale = [(0.0, 10.0), (1e-8, 100.0), (1e-4, 3.0), (1e-8, 1e5), (1e-8, 100.0)]
    return per_scale * len(SCALES) + [(1e-4, 5.0), (1e-4, 5.0)]


def bounds_shared_c_beta_gamma() -> list[tuple[float, float]]:
    per_scale = [(0.0, 10.0), (1e-8, 100.0), (1e-4, 3.0), (1e-8, 1e5)]
    return per_scale * len(SCALES) + [(1e-8, 100.0), (1e-4, 5.0), (1e-4, 5.0)]


def train_objective(curves_by_scale: dict[str, list[object]], unpack: Callable[[np.ndarray], dict[str, np.ndarray]]) -> Callable[[np.ndarray], float]:
    def objective(x: np.ndarray) -> float:
        params_by_scale = unpack(x)
        pred_all: list[np.ndarray] = []
        loss_all: list[np.ndarray] = []
        for scale in SCALES:
            params = params_by_scale[scale]
            for curve in curves_by_scale[scale]:
                pred = predict(params, curve)
                if np.any(~np.isfinite(pred)) or np.any(pred <= 0):
                    return 1e18
                pred_all.append(pred)
                loss_all.append(curve.loss)
        return huber_log_residual(np.concatenate(loss_all), np.concatenate(pred_all))

    return objective


def fit_with_restarts(
    name: str,
    inits: list[np.ndarray],
    bounds: list[tuple[float, float]],
    unpack: Callable[[np.ndarray], dict[str, np.ndarray]],
    curves_by_scale: dict[str, list[object]],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    objective = train_objective(curves_by_scale, unpack)
    best_x: np.ndarray | None = None
    best_fun = float("inf")
    rows: list[dict[str, object]] = []
    for restart_id, init in enumerate(inits):
        res = minimize(
            objective,
            x0=init,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": MAXITER, "ftol": 1e-10},
        )
        rows.append(
            {
                "variant": name,
                "restart_id": restart_id,
                "objective": float(res.fun),
                "success": int(bool(res.success)),
                "message": str(res.message),
                "nit": int(getattr(res, "nit", -1)),
            }
        )
        if float(res.fun) < best_fun:
            best_fun = float(res.fun)
            best_x = np.array(res.x, dtype=np.float64)
    if best_x is None:
        raise RuntimeError(f"no fit found for {name}")
    return unpack(best_x), {"variant": name, "objective": best_fun, "restarts": rows}


def shared_inits(independent: dict[str, np.ndarray], curves_by_scale: dict[str, list[object]], variant: str) -> list[np.ndarray]:
    starts: list[np.ndarray] = []
    if variant == "shared_beta_gamma":
        starts.append(pack_shared_beta_gamma(independent))
        for beta, gamma in [(0.5, 0.5), (0.7, 0.7), (1.0, 1.0), (1.5, 1.5)]:
            params = {scale: generic_scale_start(scale, curves_by_scale[scale], beta, gamma) for scale in SCALES}
            starts.append(pack_shared_beta_gamma(params))
        return starts
    if variant == "shared_c_beta_gamma":
        starts.append(pack_shared_c_beta_gamma(independent))
        for c_value, beta, gamma in [(2.0, 0.5, 0.5), (2.0, 0.7, 0.7), (2.0, 1.0, 1.0), (2.5, 1.0, 1.0)]:
            params = {scale: generic_scale_start(scale, curves_by_scale[scale], beta, gamma, c_value) for scale in SCALES}
            starts.append(pack_shared_c_beta_gamma(params))
        return starts
    raise ValueError(variant)


def fit_shape_shared_backbones(independent: dict[str, np.ndarray]) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, object]], list[dict[str, object]]]:
    curves_by_scale = train_curves()
    params_by_variant: dict[str, dict[str, np.ndarray]] = {
        "frozen_official": {scale: np.array(MPL_PRECOMPUTED_INIT[scale], dtype=np.float64) for scale in SCALES},
        "cosine_independent": independent,
    }
    objective_rows: list[dict[str, object]] = []
    restart_rows: list[dict[str, object]] = []

    for name, unpack, bounds in [
        ("shared_beta_gamma", unpack_shared_beta_gamma, bounds_shared_beta_gamma()),
        ("shared_c_beta_gamma", unpack_shared_c_beta_gamma, bounds_shared_c_beta_gamma()),
    ]:
        params, info = fit_with_restarts(
            name=name,
            inits=shared_inits(independent, curves_by_scale, name),
            bounds=bounds,
            unpack=unpack,
            curves_by_scale=curves_by_scale,
        )
        params_by_variant[name] = params
        objective_rows.append({"variant": name, "objective": info["objective"]})
        restart_rows.extend(info["restarts"])

    for name, params in [
        ("frozen_official", params_by_variant["frozen_official"]),
        ("cosine_independent", independent),
    ]:
        obj = train_objective(curves_by_scale, lambda _x, p=params: p)(np.zeros(1, dtype=np.float64))
        objective_rows.append({"variant": name, "objective": obj})
    return params_by_variant, objective_rows, restart_rows


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
    post_warmup = max(len(lrs) - WARMUP, 1)
    return max(0.0, 1.0 - float(cooldown_support_span(lrs)) / float(post_warmup))


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
    pred = baseline + factor * float(params[3]) * (d_lag - d_down)
    return pred, {
        "effective_tau_steps": tau,
        "adiabatic_factor": factor,
        "cooldown_support_span": float(cooldown_support_span(curve.lrs)),
    }


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


def detail_rows(params_by_variant: dict[str, dict[str, np.ndarray]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    official_mae: dict[tuple[str, str], float] = {}
    for scale in SCALES:
        for _, curve_name, _ in ALL_TARGETS:
            curve = load_curve(scale, curve_name)
            official_mae[(scale, curve_name)] = metrics(
                curve.loss,
                mpl_predict(MPL_PRECOMPUTED_INIT[scale], curve),
            )["mae"]

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


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    keys = sorted({(str(row["variant"]), str(row["group"])) for row in rows})
    for variant, group in keys:
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


def write_report(summary: list[dict[str, object]], details: list[dict[str, object]], objectives: list[dict[str, object]]) -> None:
    lines = [
        "# MPL Backbone Shape-Sharing Audit\n\n",
        "This audit targets the current bottleneck: strict cosine-only MPL is weak on WSD before any error correction.  "
        "The tested repair is not a new residual formula; it reduces MPL backbone freedom by sharing LD-kernel shape parameters across scales.\n\n",
        "Downstream correction is unchanged:\n\n",
        "\\[\n",
        "\\hat L_s(t)=L_{\\mathrm{MPL},s}(t)+a_sB_s[D_{\\downarrow,\\tau_s,s}(t)-D_{\\downarrow,s}(t)].\n",
        "\\]\n\n",
        "## Summary\n\n",
        "| variant | group | backbone params | correction vs own MPL | wins / non-harm | own MPL vs official | corrected vs official |\n",
        "|---|---|---:|---:|---:|---:|---:|\n",
    ]
    for variant in ["frozen_official", "cosine_independent", "shared_beta_gamma", "shared_c_beta_gamma"]:
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
        "\n## Train Objective\n\n",
        "| variant | cosine train objective |\n",
        "|---|---:|\n",
    ]
    for row in sorted(objectives, key=lambda item: str(item["variant"])):
        lines.append(f"| {row['variant']} | {float(row['objective']):.8g} |\n")

    lines += [
        "\n## WSD Details After Correction\n\n",
        "| variant | scale | target | correction delta | corrected vs official |\n",
        "|---|---:|---|---:|---:|\n",
    ]
    for row in details:
        if row["group"] != "core_wsd":
            continue
        lines.append(
            f"| {row['variant']} | {row['scale']} | {row['test_label']} | "
            f"{fmt_pct(float(row['delta_vs_own_baseline_pct']))} | "
            f"{fmt_pct(float(row['corr_vs_official_baseline_pct']))} |\n"
        )

    lines += [
        "\n## Reading\n\n",
        "- Sharing LD-shape parameters is interpretable because it treats \\(\\beta,\\gamma\\) or \\(C,\\beta,\\gamma\\) as scale-invariant schedule-response shape, while keeping loss scale parameters per model size.\n",
        "- This reduces backbone parameters from 21 to 17 or 15 and adds no residual-fitted parameters.\n",
        "- If shape sharing improves corrected WSD absolute error versus independent cosine-only MPL, it points to an identifiability issue in the cosine-only backbone rather than a need for a larger residual model.\n",
        "- If it hurts, the result is still useful negative evidence: stronger interpretability constraints on the backbone are not sufficient by themselves.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    independent = read_independent_params()
    params_by_variant, objective_rows, restart_rows = fit_shape_shared_backbones(independent)
    details = detail_rows(params_by_variant)
    summary = aggregate(details)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "params.csv", parameter_rows(params_by_variant))
    write_csv(OUT_DIR / "train_objectives.csv", objective_rows)
    write_csv(OUT_DIR / "restart_log.csv", restart_rows)
    write_csv(OUT_DIR / "details.csv", details)
    write_csv(OUT_DIR / "summary.csv", summary)
    write_report(summary, details, objective_rows)
    print(f"wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
