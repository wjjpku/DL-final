#!/usr/bin/env python3
"""Jointly fit MPL with a decay-lag kernel inside the MPL LR-drop term.

This audit addresses the failure mode of a two-stage residual estimator:
freezing MPL first can leave backbone fitting error in the residual that is
then over-interpreted as non-adiabatic lag.  Here the new term is optimized in
the same objective as the MPL parameters.

The original MPL prediction is

    L(t) = L0 + A S(t)^(-alpha) + B * LD(t; C, beta, gamma)

where LD is the signed LR-drop convolution.  The joint lag variant replaces the
last term by a two-kernel LR-drop response

    B * LD_ad(t; C, beta, gamma) + K * Lag_tau(t)

with

    Lag_tau(t) = sum_{u <= t} max(lr_{u-1} - lr_u, 0) * exp(-(t-u)/tau).

All parameters, including K, are fitted directly against loss curves.  Tau is
schedule-geometric, not loss-fitted, so the only new fitted parameter is K.
"""
from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

TMP_CACHE = Path(os.environ.get("TMPDIR", "/tmp")) / "dl-final-cache"
TMP_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(TMP_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(TMP_CACHE / "xdg"))

import numpy as np
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parent
ROOT = REPO.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from reproduce_cosine_to_wsd import (  # noqa: E402
    MPL_PRECOMPUTED_INIT,
    PEAK_LR,
    SCALES,
    Curve,
    compute_ld,
    compute_s1,
    huber_log_residual,
    load_curve,
    metrics,
    mpl_predict,
)
from step_time_shape_routed_estimator import CORE_CURVES, schedule_stats  # noqa: E402


OUT_DIR = ROOT / "results" / "joint_mpl_lag_fit"
MAXITER = 500
STEP_TAU_BASE = 512.0
STEP_DROP_WEAK = 0.40
STEP_DROP_FULL = 0.90
TAIL_TAU_PER_STEP = 1.25
MAX_TAU = 8192.0


@dataclass(frozen=True)
class CachedCurve:
    curve: Curve
    label: str
    s1: np.ndarray
    lag: np.ndarray


def fmt_pct(value: float) -> str:
    return f"{value:+.1f}%"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def positive_drop(curve: Curve) -> np.ndarray:
    drop = np.zeros_like(curve.lrs, dtype=np.float64)
    drop[1:] = np.maximum(curve.lrs[:-1] - curve.lrs[1:], 0.0)
    return drop


def schedule_geometry_tau(curve: Curve) -> float:
    drop = positive_drop(curve) / PEAK_LR
    idx = np.flatnonzero(drop > 1e-14)
    total_drop = float(np.sum(drop))
    span = float(idx[-1] - idx[0] + 1) if len(idx) else 0.0
    length = float(len(curve.lrs))
    if total_drop <= 0.05:
        return 0.0
    if span > 16000.0 and length <= 30000.0:
        return 0.0
    if span > 100.0:
        return min(MAX_TAU, TAIL_TAU_PER_STEP * span)
    q = np.clip((total_drop - STEP_DROP_WEAK) / (STEP_DROP_FULL - STEP_DROP_WEAK), 0.0, 1.0)
    return STEP_TAU_BASE * (1.0 + 2.0 * float(q) ** 3)


def lag_feature_raw(curve: Curve, tau: float) -> np.ndarray:
    if tau <= 0.0:
        return np.zeros_like(curve.step, dtype=np.float64)
    drop = positive_drop(curve)
    out = np.zeros_like(drop)
    acc = 0.0
    decay = float(np.exp(-1.0 / tau))
    for t in range(len(drop)):
        acc = acc * decay + drop[t]
        out[t] = acc
    return out[curve.step]


def load_cached_curves(curve_defs: tuple[tuple[str, str], ...]) -> dict[tuple[str, str], CachedCurve]:
    cache: dict[tuple[str, str], CachedCurve] = {}
    for scale in SCALES:
        for curve_name, label in curve_defs:
            curve = load_curve(scale, curve_name)
            tau = schedule_geometry_tau(curve)
            cache[(scale, curve_name)] = CachedCurve(
                curve=curve,
                label=label,
                s1=compute_s1(curve),
                lag=lag_feature_raw(curve, tau),
            )
    return cache


def joint_lag_predict(params: np.ndarray, item: CachedCurve) -> np.ndarray:
    L0, A, alpha, B, C, beta, gamma, K = params
    curve = item.curve
    ld = compute_ld(curve, C, beta, gamma)
    return L0 + A * np.power(item.s1, -alpha) + B * ld + K * item.lag


def objective_for(
    params: np.ndarray,
    items: list[CachedCurve],
    *,
    lag: bool,
) -> float:
    pred_all = []
    loss_all = []
    for item in items:
        pred = joint_lag_predict(params, item) if lag else mpl_predict(params[:7], item.curve)
        if np.any(~np.isfinite(pred)) or np.any(pred <= 0.0):
            return 1e18
        pred_all.append(pred)
        loss_all.append(item.curve.loss)
    return huber_log_residual(np.concatenate(loss_all), np.concatenate(pred_all))


def fit_model(
    scale: str,
    items: list[CachedCurve],
    *,
    lag: bool,
) -> tuple[np.ndarray, float]:
    min_loss = min(float(item.curve.loss.min()) for item in items)
    base = MPL_PRECOMPUTED_INIT[scale]
    base_inits = [
        base,
        np.array([min_loss - 0.05, 0.5, 0.5, 300.0, 2.0, 0.5, 0.5], dtype=np.float64),
        np.array([min_loss - 0.10, 1.0, 0.4, 500.0, 1.0, 0.8, 0.8], dtype=np.float64),
    ]
    if lag:
        inits = [np.concatenate([init, np.array([k], dtype=np.float64)]) for init in base_inits for k in [0.0, 20.0, 50.0, 100.0]]
        bounds = [
            (0.0, 10.0),
            (1e-8, 100.0),
            (1e-4, 3.0),
            (1e-8, 1e5),
            (1e-8, 100.0),
            (1e-4, 5.0),
            (1e-4, 5.0),
            (0.0, 1e5),
        ]
    else:
        inits = base_inits
        bounds = [
            (0.0, 10.0),
            (1e-8, 100.0),
            (1e-4, 3.0),
            (1e-8, 1e5),
            (1e-8, 100.0),
            (1e-4, 5.0),
            (1e-4, 5.0),
        ]

    best_x = None
    best_fun = float("inf")
    for init in inits:
        res = minimize(
            lambda x: objective_for(x, items, lag=lag),
            x0=init,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": MAXITER, "ftol": 1e-10},
        )
        if res.fun < best_fun:
            best_fun = float(res.fun)
            best_x = res.x
    assert best_x is not None
    return best_x, best_fun


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    deltas = [float(row["delta_pct"]) for row in rows]
    return {
        "rows": len(rows),
        "mean_delta": float(np.mean(deltas)),
        "worst_delta": float(np.max(deltas)),
        "nonharm": int(sum(int(row["nonharm"]) for row in rows)),
        "wins": int(sum(int(row["win"]) for row in rows)),
    }


def evaluate_leave_one_curve_out() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    cache = load_cached_curves(CORE_CURVES)
    details: list[dict[str, object]] = []
    params_rows: list[dict[str, object]] = []
    curve_names = [name for name, _ in CORE_CURVES]
    labels = dict(CORE_CURVES)

    for scale in SCALES:
        for heldout in curve_names:
            train_names = [name for name in curve_names if name != heldout]
            train_items = [cache[(scale, name)] for name in train_names]
            target = cache[(scale, heldout)]
            base_params, base_obj = fit_model(scale, train_items, lag=False)
            lag_params, lag_obj = fit_model(scale, train_items, lag=True)

            base_pred = mpl_predict(base_params, target.curve)
            lag_pred = joint_lag_predict(lag_params, target)
            base_mae = metrics(target.curve.loss, base_pred)["mae"]
            lag_mae = metrics(target.curve.loss, lag_pred)["mae"]
            delta = 100.0 * (lag_mae / base_mae - 1.0)
            details.append(
                {
                    "scale": scale,
                    "heldout_curve": heldout,
                    "heldout_label": labels[heldout],
                    "train_curves": "+".join(name.replace(".csv", "") for name in train_names),
                    "mpl_train_objective": base_obj,
                    "joint_lag_train_objective": lag_obj,
                    "mpl_mae": base_mae,
                    "joint_lag_mae": lag_mae,
                    "delta_pct": delta,
                    "win": int(lag_mae < base_mae),
                    "nonharm": int(lag_mae <= base_mae * (1.0 + 1e-12)),
                }
            )
            params_rows.append(
                {
                    "scale": scale,
                    "heldout_curve": heldout,
                    "model": "mpl",
                    "objective": base_obj,
                    "L0": base_params[0],
                    "A": base_params[1],
                    "alpha": base_params[2],
                    "B": base_params[3],
                    "C": base_params[4],
                    "beta": base_params[5],
                    "gamma": base_params[6],
                    "K": 0.0,
                }
            )
            params_rows.append(
                {
                    "scale": scale,
                    "heldout_curve": heldout,
                    "model": "joint_lag",
                    "objective": lag_obj,
                    "L0": lag_params[0],
                    "A": lag_params[1],
                    "alpha": lag_params[2],
                    "B": lag_params[3],
                    "C": lag_params[4],
                    "beta": lag_params[5],
                    "gamma": lag_params[6],
                    "K": lag_params[7],
                }
            )

    summary_rows: list[dict[str, object]] = []
    for curve_name, label in CORE_CURVES:
        subset = [row for row in details if row["heldout_curve"] == curve_name]
        summary_rows.append({"heldout_curve": curve_name, "heldout_label": label, **summarize(subset)})
    summary_rows.append({"heldout_curve": "ALL", "heldout_label": "ALL", **summarize(details)})
    return details, summary_rows, params_rows


def write_report(summary_rows: list[dict[str, object]]) -> None:
    all_row = next(row for row in summary_rows if row["heldout_curve"] == "ALL")
    lines = [
        "# Joint MPL-Lag Fit Audit\n\n",
        "This audit modifies MPL's LR-drop term and optimizes the new lag amplitude in the same fitting objective as the original MPL parameters.\n\n",
        "## Formula\n\n",
        "Original MPL:\n\n",
        "```text\n",
        "L(t) = L0 + A S(t)^(-alpha) + B * sum_k delta_eta_k * G(C eta_k^(-gamma) (S(t)-S(k)))\n",
        "G(x) = 1 - (1 + x)^(-beta)\n",
        "```\n\n",
        "Joint lag variant:\n\n",
        "```text\n",
        "L(t) = L0 + A S(t)^(-alpha) + B * LD_ad(t; C,beta,gamma) + K * Lag_tau(t)\n",
        "Lag_tau(t) = sum_{u <= t} max(lr_{u-1} - lr_u, 0) * exp(-(t-u)/tau(schedule))\n",
        "```\n\n",
        "The fitted parameters are `L0,A,alpha,B,C,beta,gamma,K`.  `K` is optimized jointly with MPL, not fitted from frozen-MPL residuals.  `tau` is computed from schedule geometry.\n\n",
        "## Leave-One-Curve-Out Result\n\n",
        f"- Overall: mean `{fmt_pct(float(all_row['mean_delta']))}`, worst `{fmt_pct(float(all_row['worst_delta']))}`, non-harm `{int(all_row['nonharm'])}/{int(all_row['rows'])}`.\n\n",
        "| held-out target | mean delta vs jointly refit MPL | worst | non-harm |\n",
        "|---|---:|---:|---:|\n",
    ]
    for row in summary_rows:
        if row["heldout_curve"] == "ALL":
            continue
        lines.append(
            f"| {row['heldout_label']} | {fmt_pct(float(row['mean_delta']))} | "
            f"{fmt_pct(float(row['worst_delta']))} | {int(row['nonharm'])}/{int(row['rows'])} |\n"
        )
    lines += [
        "\n## Reading\n\n",
        "- This is the right fitting protocol for testing whether the lag term is still useful after MPL can re-optimize around it.\n",
        "- The comparison baseline is not frozen MPL; it is MPL refit on the same train curves for each held-out target.\n",
        "- A weak or unstable result here means the post-hoc residual story is confounded by MPL fitting error and should not be the main claim without this joint objective.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    details, summary_rows, params_rows = evaluate_leave_one_curve_out()
    write_csv(OUT_DIR / "leave_one_curve_out_details.csv", details)
    write_csv(OUT_DIR / "leave_one_curve_out_summary.csv", summary_rows)
    write_csv(OUT_DIR / "fitted_params.csv", params_rows)
    write_report(summary_rows)


if __name__ == "__main__":
    main()
