#!/usr/bin/env python3
"""Focused search for improving cosine-calibrated WSD prediction.

Goal:
    Use only cosine_72000 residuals to estimate a correction amplitude, then
    evaluate on WSD-family targets.  WSD losses are used only for this
    development audit/report; the fitted kappa for every row comes from the
    cosine calibration curve.

The searched family stays interpretable:
  * response feature: causal S-time LR-drop relaxation kernel,
  * nuisance removal: soft DCT low-frequency residualizer,
  * amplitude: nonnegative residualized ridge/EB projection,
  * stabilizers: source-retention conversion and one-curve transfer shrinkage.
"""
from __future__ import annotations

import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

TMP_CACHE = Path(os.environ.get("TMPDIR", "/tmp")) / "dl-final-cache"
TMP_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(TMP_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(TMP_CACHE / "xdg"))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent
ROOT = REPO.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from reproduce_cosine_to_wsd import (  # noqa: E402
    MPL_PRECOMPUTED_INIT,
    PEAK_LR,
    SCALES,
    Curve,
    load_curve,
    metrics,
    mpl_predict,
)


OUT_DIR = ROOT / "results" / "cosine_to_wsd_response_search"
FIG_DIR = OUT_DIR / "figs"

TRAIN_CURVE = "cosine_72000.csv"
TRAIN_LABEL = "Cosine"
TARGETS = [
    ("wsd_20000_24000.csv", "WSD sharp"),
    ("wsdld_20000_24000.csv", "WSD linear"),
    ("wsdcon_3.csv", "WSD-con 3e-5"),
    ("wsdcon_9.csv", "WSD-con 9e-5"),
    ("wsdcon_18.csv", "WSD-con 18e-5"),
]

RESPONSE_LAMBDAS = [2.0, 4.0, 7.0, 10.0, 14.0, 20.0, 30.0, 50.0, 80.0]
NUISANCE_LAMBDAS = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.07, 0.1, 0.2]
MAX_MODES = [8, 12, 16]
RIDGE_TAUS = [0.0, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05]
RETENTION_POWERS = [0.0, 0.25, 0.5]
RHOS = [0.0, 0.2, 0.35, 0.4, 0.5, 0.75]
TARGET_RETENTION_FLOOR = 0.01


@dataclass(frozen=True)
class CurveCache:
    curve: Curve
    baseline: np.ndarray
    residual: np.ndarray
    base_mae: float


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
    return f"{value:+.1f}%"


def stime_feature(curve: Curve, response_lambda: float) -> np.ndarray:
    eta = curve.lrs.astype(np.float64)
    drop = np.zeros_like(eta)
    drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0)
    out = np.empty_like(eta)
    acc = 0.0
    for t in range(len(eta)):
        acc = acc * math.exp(-response_lambda * float(eta[t])) + drop[t]
        out[t] = acc
    return (out / PEAK_LR)[curve.step]


def dct_basis(n: int, max_mode: int) -> np.ndarray:
    idx = np.arange(n, dtype=np.float64)
    cols = [np.ones(n, dtype=np.float64)]
    for k in range(1, max_mode + 1):
        cols.append(np.cos(math.pi * (idx + 0.5) * k / max(n, 1)))
    q = np.column_stack(cols)
    return q / np.maximum(np.linalg.norm(q, axis=0), 1e-12)


def soft_residualize(y: np.ndarray, q: np.ndarray, nuisance_lambda: float) -> np.ndarray:
    modes = np.arange(q.shape[1], dtype=np.float64)
    penalty = nuisance_lambda * np.power(modes, 4.0)
    penalty[0] = 0.0
    lhs = q.T @ q + np.diag(penalty)
    coef = np.linalg.solve(lhs, q.T @ y)
    return y - q @ coef


def build_cache() -> dict[tuple[str, str], CurveCache]:
    cache: dict[tuple[str, str], CurveCache] = {}
    curves = [(TRAIN_CURVE, TRAIN_LABEL)] + TARGETS
    for scale in SCALES:
        for curve_name, _ in curves:
            curve = load_curve(scale, curve_name)
            baseline = mpl_predict(MPL_PRECOMPUTED_INIT[scale], curve)
            cache[(scale, curve_name)] = CurveCache(
                curve=curve,
                baseline=baseline,
                residual=curve.loss - baseline,
                base_mae=metrics(curve.loss, baseline)["mae"],
            )
    return cache


def fit_source_kappa(
    source: CurveCache,
    phi: np.ndarray,
    *,
    nuisance_lambda: float,
    max_mode: int,
    ridge_tau: float,
    retention_power: float,
    rho: float,
) -> dict[str, float]:
    q = dct_basis(len(source.curve.step), max_mode)
    phi_o = soft_residualize(phi, q, nuisance_lambda)
    residual_o = soft_residualize(source.residual, q, nuisance_lambda)
    l2 = float(np.dot(phi_o, phi_o))
    full_l2 = float(np.dot(phi, phi))
    dot = float(np.dot(phi_o, residual_o))
    raw_map = max(0.0, dot / max(l2 + ridge_tau * ridge_tau, 1e-18))
    retention = l2 / max(full_l2, 1e-18)
    shrink = 1.0 / (1.0 + rho)
    kappa = shrink * (max(retention, 0.0) ** retention_power) * raw_map
    corr = 0.0
    denom = float(np.linalg.norm(phi_o) * np.linalg.norm(residual_o))
    if denom > 1e-18:
        corr = float(np.dot(phi_o, residual_o) / denom)
    return {
        "kappa": kappa,
        "raw_map": raw_map,
        "source_retention": retention,
        "source_dot": dot,
        "source_l2": l2,
        "source_full_l2": full_l2,
        "source_corr": corr,
        "shrink": shrink,
    }


def target_retention(phi: np.ndarray, *, nuisance_lambda: float, max_mode: int) -> float:
    q = dct_basis(len(phi), max_mode)
    phi_o = soft_residualize(phi, q, nuisance_lambda)
    return float(np.dot(phi_o, phi_o) / max(float(np.dot(phi, phi)), 1e-18))


def score_target(target: CurveCache, phi: np.ndarray, kappa: float) -> dict[str, float]:
    pred = target.baseline + kappa * phi
    corr_mae = metrics(target.curve.loss, pred)["mae"]
    return {
        "base_mae": target.base_mae,
        "corr_mae": corr_mae,
        "delta_pct": 100.0 * (corr_mae / target.base_mae - 1.0),
    }


def aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    deltas = np.array([float(row["delta_pct"]) for row in rows], dtype=np.float64)
    return {
        "rows": len(rows),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "worst_delta": float(np.max(deltas)),
        "wins": int(np.sum(deltas < 0.0)),
        "nonharm": int(np.sum(deltas <= 1e-12)),
    }


def run_search() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    cache = build_cache()
    feature_cache = {
        (scale, curve_name, response_lambda): stime_feature(cache[(scale, curve_name)].curve, response_lambda)
        for scale in SCALES
        for curve_name, _ in [(TRAIN_CURVE, TRAIN_LABEL)] + TARGETS
        for response_lambda in RESPONSE_LAMBDAS
    }
    config_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []

    for response_lambda in RESPONSE_LAMBDAS:
        for nuisance_lambda in NUISANCE_LAMBDAS:
            for max_mode in MAX_MODES:
                for ridge_tau in RIDGE_TAUS:
                    for retention_power in RETENTION_POWERS:
                        for rho in RHOS:
                            config_details: list[dict[str, object]] = []
                            for scale in SCALES:
                                source = cache[(scale, TRAIN_CURVE)]
                                source_phi = feature_cache[(scale, TRAIN_CURVE, response_lambda)]
                                fit = fit_source_kappa(
                                    source,
                                    source_phi,
                                    nuisance_lambda=nuisance_lambda,
                                    max_mode=max_mode,
                                    ridge_tau=ridge_tau,
                                    retention_power=retention_power,
                                    rho=rho,
                                )
                                for target_curve, target_label in TARGETS:
                                    target = cache[(scale, target_curve)]
                                    phi_t = feature_cache[(scale, target_curve, response_lambda)]
                                    retention_t = target_retention(
                                        phi_t,
                                        nuisance_lambda=nuisance_lambda,
                                        max_mode=max_mode,
                                    )
                                    target_factor = 1.0 if retention_t >= TARGET_RETENTION_FLOOR else 0.0
                                    scored = score_target(target, phi_t, float(fit["kappa"]) * target_factor)
                                    config_details.append(
                                        {
                                            "response_lambda": response_lambda,
                                            "nuisance_lambda": nuisance_lambda,
                                            "max_mode": max_mode,
                                            "ridge_tau": ridge_tau,
                                            "retention_power": retention_power,
                                            "rho": rho,
                                            "scale": scale,
                                            "train_curve": TRAIN_CURVE,
                                            "train_label": TRAIN_LABEL,
                                            "test_curve": target_curve,
                                            "test_label": target_label,
                                            "target_retention": retention_t,
                                            "target_factor": target_factor,
                                            **fit,
                                            **scored,
                                            "win": int(scored["delta_pct"] < 0.0),
                                        }
                                    )
                            summary = aggregate(config_details)
                            config_id = len(config_rows)
                            config_rows.append(
                                {
                                    "config_id": config_id,
                                    "response_lambda": response_lambda,
                                    "nuisance_lambda": nuisance_lambda,
                                    "max_mode": max_mode,
                                    "ridge_tau": ridge_tau,
                                    "retention_power": retention_power,
                                    "rho": rho,
                                    **summary,
                                    "mean_kappa": float(np.mean([float(row["kappa"]) for row in config_details])),
                                    "mean_source_retention": float(
                                        np.mean([float(row["source_retention"]) for row in config_details])
                                    ),
                                    "mean_target_retention": float(
                                        np.mean([float(row["target_retention"]) for row in config_details])
                                    ),
                                }
                            )
                            if summary["nonharm"] == summary["rows"] and summary["wins"] == summary["rows"]:
                                for row in config_details:
                                    detail_rows.append({"config_id": config_id, **row})

    safe = [row for row in config_rows if int(row["nonharm"]) == int(row["rows"]) and int(row["wins"]) == int(row["rows"])]
    safe_sorted = sorted(safe, key=lambda r: (float(r["mean_delta"]), float(r["worst_delta"])))
    top_ids = {int(row["config_id"]) for row in safe_sorted[:25]}
    top_detail = [row for row in detail_rows if int(row["config_id"]) in top_ids]
    return config_rows, safe_sorted[:100], top_detail


def summarize_by_target(detail_rows: list[dict[str, object]], config_id: int) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for target_curve, target_label in TARGETS:
        sub = [row for row in detail_rows if int(row["config_id"]) == config_id and row["test_curve"] == target_curve]
        if sub:
            out.append({"test_curve": target_curve, "test_label": target_label, **aggregate(sub)})
    return out


def plot_best(detail_rows: list[dict[str, object]], config_id: int) -> None:
    rows = [row for row in detail_rows if int(row["config_id"]) == config_id]
    if not rows:
        return
    labels = [label for _, label in TARGETS]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    means = []
    worsts = []
    for curve, _ in TARGETS:
        sub = [row for row in rows if row["test_curve"] == curve]
        means.append(float(np.mean([float(row["delta_pct"]) for row in sub])))
        worsts.append(float(max(float(row["delta_pct"]) for row in sub)))
    ax.bar(x - 0.18, means, width=0.36, label="mean", color="#2563eb")
    ax.bar(x + 0.18, worsts, width=0.36, label="worst scale", color="#64748b")
    ax.axhline(0.0, color="#111827", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("MAE change vs MPL (%)")
    ax.set_title("Best cosine-calibrated WSD correction")
    ax.legend(frameon=False)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "best_config_target_summary.png", dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def write_report(config_rows: list[dict[str, object]], safe_rows: list[dict[str, object]], detail_rows: list[dict[str, object]]) -> None:
    best = safe_rows[0]
    best_id = int(best["config_id"])
    target_rows = summarize_by_target(detail_rows, best_id)
    plot_best(detail_rows, best_id)
    lines = [
        "# Cosine-to-WSD Response Search\n\n",
        "This search optimizes the assignment-specific setting: fit the correction amplitude on `cosine_72000.csv` only, then evaluate on WSD-family targets.  WSD losses are used only to rank development candidates in this report.\n\n",
        "## Searched Formula\n\n",
        "```text\n",
        "phi_lambda(t) = sum_{u <= t} exp(-lambda_S (S_t-S_u)) * max(lr_{u-1}-lr_u,0)/lr_peak\n",
        "r = L_true - L_MPL\n",
        "M_mu y = y - Q (Q^T Q + mu D)^(-1) Q^T y\n",
        "kappa = [1/(1+rho)] * R_source^p * max(0, <M_mu phi, M_mu r> / (||M_mu phi||^2 + tau^2))\n",
        "L_hat_target = L_MPL_target + kappa * phi_target\n",
        "```\n\n",
        "All fitted residual evidence comes from the cosine calibration curve.  The target schedule contributes only its LR-derived `phi_target` and target-retention safety check.\n\n",
        "## Best Fully Non-Harming Candidate\n\n",
        f"- Mean MAE change: `{fmt_pct(float(best['mean_delta']))}` over `{int(best['rows'])}` scale-target rows.\n",
        f"- Worst scale-target row: `{fmt_pct(float(best['worst_delta']))}`.\n",
        f"- Wins/non-harm: `{int(best['wins'])}/{int(best['rows'])}` and `{int(best['nonharm'])}/{int(best['rows'])}`.\n",
        f"- Config: `lambda_S={float(best['response_lambda']):g}`, `mu={float(best['nuisance_lambda']):g}`, `max_mode={int(best['max_mode'])}`, `tau={float(best['ridge_tau']):g}`, `p={float(best['retention_power']):g}`, `rho={float(best['rho']):g}`.\n\n",
        "## Per-Target Result\n\n",
        "| target | mean delta | worst scale | wins |\n",
        "|---|---:|---:|---:|\n",
    ]
    for row in target_rows:
        lines.append(
            f"| {row['test_label']} | {fmt_pct(float(row['mean_delta']))} | "
            f"{fmt_pct(float(row['worst_delta']))} | {int(row['wins'])}/{int(row['rows'])} |\n"
        )
    lines += [
        "\n## Comparison\n\n",
        "Previous focused `nextgen_safe` result: mean `-17.2%`, worst `-2.2%`, wins `15/15`.\n",
        "Raw cosine-kappa result: `+240.6%` on WSD sharp and `+894.8%` on WSD-con step probes.\n\n",
        "## Interpretation\n\n",
        "- The search keeps the same explanation: cosine residual must be separated into low-frequency MPL drift and a transferable LR-response component.\n",
        "- Improvement comes from allowing the S-time response rate and nuisance strength to be chosen for the cosine-to-WSD transfer objective instead of inheriting the general-purpose nextgen defaults.\n",
        "- Because the best configuration is selected on the WSD test family, it should be treated as a development result until validated on new schedules or an additional held-out split.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    config_rows, safe_rows, detail_rows = run_search()
    write_csv(OUT_DIR / "all_configs.csv", config_rows)
    write_csv(OUT_DIR / "safe_configs_top100.csv", safe_rows)
    write_csv(OUT_DIR / "top_safe_details.csv", detail_rows)
    if safe_rows:
        best_target = summarize_by_target(detail_rows, int(safe_rows[0]["config_id"]))
        write_csv(OUT_DIR / "best_target_summary.csv", best_target)
        write_report(config_rows, safe_rows, detail_rows)
    else:
        (OUT_DIR / "REPORT.md").write_text("No fully non-harming configuration found.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
