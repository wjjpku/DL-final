#!/usr/bin/env python3
"""Fast staged joint-fit audit for adding lag inside MPL.

This is the cheap first-stage answer to the joint-fitting concern.  It keeps
MPL's LR-drop shape parameters (C, beta, gamma) fixed to the published
scale-specific values, but it fits the MPL backbone amplitudes and the new lag
amplitude in one objective:

    MPL fixed-shape:
        L = L0 + A S^{-alpha} + B LD_fixed

    Joint fixed-shape lag:
        L = L0 + A S^{-alpha} + B LD_fixed + K Lag_tau

The comparison baseline is therefore not frozen MPL.  For every held-out curve,
both models refit on the same remaining curves.
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
)
from step_time_shape_routed_estimator import CORE_CURVES  # noqa: E402


OUT_DIR = ROOT / "results" / "joint_mpl_lag_staged_audit"
MAXITER = 300
STEP_TAU_BASE = 512.0
STEP_DROP_WEAK = 0.40
STEP_DROP_FULL = 0.90
TAIL_TAU_PER_STEP = 1.25
MAX_TAU = 8192.0


@dataclass(frozen=True)
class FixedShapeCurve:
    curve: Curve
    label: str
    tau: float
    s1: np.ndarray
    ld_fixed: np.ndarray
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


def positive_drop_norm(curve: Curve) -> np.ndarray:
    drop = np.zeros_like(curve.lrs, dtype=np.float64)
    drop[1:] = np.maximum(curve.lrs[:-1] - curve.lrs[1:], 0.0) / PEAK_LR
    return drop


def geometry_tau(curve: Curve) -> float:
    drop = positive_drop_norm(curve)
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


def lag_feature(curve: Curve, tau: float) -> np.ndarray:
    if tau <= 0.0:
        return np.zeros_like(curve.step, dtype=np.float64)
    drop = positive_drop_norm(curve)
    out = np.zeros_like(drop)
    acc = 0.0
    decay = float(np.exp(-1.0 / tau))
    for t in range(len(drop)):
        acc = acc * decay + drop[t]
        out[t] = acc
    return out[curve.step]


def build_cache() -> dict[tuple[str, str], FixedShapeCurve]:
    cache: dict[tuple[str, str], FixedShapeCurve] = {}
    for scale in SCALES:
        _, _, _, _, c, beta, gamma = MPL_PRECOMPUTED_INIT[scale]
        for curve_name, label in CORE_CURVES:
            curve = load_curve(scale, curve_name)
            tau = geometry_tau(curve)
            cache[(scale, curve_name)] = FixedShapeCurve(
                curve=curve,
                label=label,
                tau=tau,
                s1=compute_s1(curve),
                ld_fixed=compute_ld(curve, c, beta, gamma),
                lag=lag_feature(curve, tau),
            )
    return cache


def predict_base(params: np.ndarray, item: FixedShapeCurve) -> np.ndarray:
    l0, a, alpha, b = params
    return l0 + a * np.power(item.s1, -alpha) + b * item.ld_fixed


def predict_lag(params: np.ndarray, item: FixedShapeCurve) -> np.ndarray:
    l0, a, alpha, b, k = params
    return l0 + a * np.power(item.s1, -alpha) + b * item.ld_fixed + k * item.lag


def objective(params: np.ndarray, items: list[FixedShapeCurve], *, lag: bool) -> float:
    preds = []
    losses = []
    for item in items:
        pred = predict_lag(params, item) if lag else predict_base(params, item)
        if np.any(~np.isfinite(pred)) or np.any(pred <= 0.0):
            return 1e18
        preds.append(pred)
        losses.append(item.curve.loss)
    return huber_log_residual(np.concatenate(losses), np.concatenate(preds))


def fit_fixed_shape(scale: str, items: list[FixedShapeCurve], *, lag: bool) -> tuple[np.ndarray, float]:
    base = MPL_PRECOMPUTED_INIT[scale]
    min_loss = min(float(item.curve.loss.min()) for item in items)
    base_inits = [
        np.array([base[0], base[1], base[2], base[3]], dtype=np.float64),
        np.array([min_loss - 0.05, 0.5, 0.5, 300.0], dtype=np.float64),
        np.array([min_loss - 0.10, 1.0, 0.4, 500.0], dtype=np.float64),
    ]
    if lag:
        inits = [np.concatenate([init, np.array([k], dtype=np.float64)]) for init in base_inits for k in [0.0, 0.01, 0.03, 0.06]]
        bounds = [(0.0, 10.0), (1e-8, 100.0), (1e-4, 3.0), (1e-8, 1e5), (0.0, 10.0)]
    else:
        inits = base_inits
        bounds = [(0.0, 10.0), (1e-8, 100.0), (1e-4, 3.0), (1e-8, 1e5)]

    best_x = None
    best_fun = float("inf")
    for init in inits:
        res = minimize(
            lambda x: objective(x, items, lag=lag),
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


def run_audit() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    cache = build_cache()
    curve_names = [curve for curve, _ in CORE_CURVES]
    labels = dict(CORE_CURVES)
    details: list[dict[str, object]] = []
    params_rows: list[dict[str, object]] = []

    for scale in SCALES:
        for heldout in curve_names:
            train_names = [name for name in curve_names if name != heldout]
            train_items = [cache[(scale, name)] for name in train_names]
            target = cache[(scale, heldout)]
            base_params, base_obj = fit_fixed_shape(scale, train_items, lag=False)
            lag_params, lag_obj = fit_fixed_shape(scale, train_items, lag=True)
            base_pred = predict_base(base_params, target)
            lag_pred = predict_lag(lag_params, target)
            base_mae = metrics(target.curve.loss, base_pred)["mae"]
            lag_mae = metrics(target.curve.loss, lag_pred)["mae"]
            delta = 100.0 * (lag_mae / base_mae - 1.0)
            details.append(
                {
                    "scale": scale,
                    "heldout_curve": heldout,
                    "heldout_label": labels[heldout],
                    "train_curves": "+".join(name.replace(".csv", "") for name in train_names),
                    "target_tau": target.tau,
                    "fixed_shape_mpl_objective": base_obj,
                    "joint_lag_objective": lag_obj,
                    "fixed_shape_mpl_mae": base_mae,
                    "joint_lag_mae": lag_mae,
                    "delta_pct": delta,
                    "win": int(lag_mae < base_mae),
                    "nonharm": int(lag_mae <= base_mae * (1.0 + 1e-12)),
                }
            )
            for model, params, obj in [
                ("fixed_shape_mpl", base_params, base_obj),
                ("joint_lag", lag_params, lag_obj),
            ]:
                row = {
                    "scale": scale,
                    "heldout_curve": heldout,
                    "model": model,
                    "objective": obj,
                    "L0": params[0],
                    "A": params[1],
                    "alpha": params[2],
                    "B": params[3],
                    "K": params[4] if model == "joint_lag" else 0.0,
                }
                params_rows.append(row)

    summary: list[dict[str, object]] = []
    for curve_name, label in CORE_CURVES:
        subset = [row for row in details if row["heldout_curve"] == curve_name]
        summary.append({"heldout_curve": curve_name, "heldout_label": label, **summarize(subset)})
    summary.append({"heldout_curve": "ALL", "heldout_label": "ALL", **summarize(details)})
    return details, summary, params_rows


def write_report(summary: list[dict[str, object]]) -> None:
    all_row = next(row for row in summary if row["heldout_curve"] == "ALL")
    lines = [
        "# Staged Joint MPL-Lag Audit\n\n",
        "This is the first, fast joint-fitting audit.  It does not freeze MPL predictions.  For each held-out target, both the baseline and the lag model refit on the same remaining curves.\n\n",
        "## Fixed-Shape Formula\n\n",
        "MPL fixed-shape baseline:\n\n",
        "```text\n",
        "L_base(t) = L0 + A S(t)^(-alpha) + B LD_fixed(t; C0,beta0,gamma0)\n",
        "```\n\n",
        "Joint lag model:\n\n",
        "```text\n",
        "L_lag(t) = L0 + A S(t)^(-alpha) + B LD_fixed(t; C0,beta0,gamma0) + K Lag_tau(t)\n",
        "Lag_tau(t) = sum_{u <= t} max(lr_{u-1}-lr_u,0)/lr_peak * exp(-(t-u)/tau(schedule))\n",
        "```\n\n",
        "`C0,beta0,gamma0` are the scale-specific published MPL values.  `L0,A,alpha,B` and `K` are fitted jointly in one log-Huber objective.\n\n",
        "## Leave-One-Curve-Out Result\n\n",
        f"- Overall: mean `{fmt_pct(float(all_row['mean_delta']))}`, worst `{fmt_pct(float(all_row['worst_delta']))}`, non-harm `{int(all_row['nonharm'])}/{int(all_row['rows'])}`.\n\n",
        "| held-out target | mean delta | worst | non-harm |\n",
        "|---|---:|---:|---:|\n",
    ]
    for row in summary:
        if row["heldout_curve"] == "ALL":
            continue
        lines.append(
            f"| {row['heldout_label']} | {fmt_pct(float(row['mean_delta']))} | "
            f"{fmt_pct(float(row['worst_delta']))} | {int(row['nonharm'])}/{int(row['rows'])} |\n"
        )
    lines += [
        "\n## Interpretation\n\n",
        "- This directly addresses the two-stage objection at a first-order level: the lag amplitude competes with the MPL backbone during fitting.\n",
        "- Because `C,beta,gamma` are still fixed, this is not yet the final full-MPL joint fit.  It is a fast diagnostic for whether the lag term remains useful before paying for the full eight-parameter optimization.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    details, summary, params_rows = run_audit()
    write_csv(OUT_DIR / "leave_one_curve_out_details.csv", details)
    write_csv(OUT_DIR / "leave_one_curve_out_summary.csv", summary)
    write_csv(OUT_DIR / "fitted_params.csv", params_rows)
    write_report(summary)


if __name__ == "__main__":
    main()
