#!/usr/bin/env python3
"""LR-curvature correction audit for cosine-to-WSD prediction.

The decoupled-channel model explains WSD transfer with a first-order LR-drop
response.  Its largest remaining residuals are WSD-con tails, where the model
can keep a delayed response for too long after a sharp LR transition.

This audit adds one schedule-only second-order feature to the step channel:
a causal relaxation of the discrete LR curvature.  The coefficients are still
fitted only on cosine_72000 residuals; WSD losses are used only for development
ranking and evaluation.
"""
from __future__ import annotations

import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

TMP_CACHE = Path(os.environ.get("TMPDIR", "/tmp")) / "dl-final-cache"
TMP_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(TMP_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(TMP_CACHE / "xdg"))

import numpy as np

REPO = Path(__file__).resolve().parent
ROOT = REPO.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cosine_to_wsd_adaptive_fit_window import channel_for_curve  # noqa: E402
from cosine_to_wsd_response_search import (  # noqa: E402
    TARGETS,
    TARGET_RETENTION_FLOOR,
    TRAIN_CURVE,
    build_cache,
    dct_basis,
    soft_residualize,
    stime_feature,
    target_retention,
)
from reproduce_cosine_to_wsd import SCALES, metrics  # noqa: E402


BASELINE_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "decoupled_channel"
OUT_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "curvature_correction"

SHARP_LINEAR = {"wsd_20000_24000.csv", "wsdld_20000_24000.csv"}
WSDCON = {"wsdcon_3.csv", "wsdcon_9.csv", "wsdcon_18.csv"}

SMOOTH = {
    "fit_start_step": 12000,
    "response_lambda": 4.0,
    "nuisance_lambda": 0.05,
    "max_mode": 8,
    "ridge_tau": 0.05,
    "retention_power": 0.25,
    "rho": 0.2,
}
STEP = {
    "fit_start_step": 3000,
    "response_lambda": 20.0,
    "nuisance_lambda": 0.015,
    "max_mode": 16,
    "ridge_tau": 0.05,
    "retention_power": 0.0,
    "rho": 0.75,
}

CURVATURE_LAMBDAS = [4.0, 7.0, 10.0, 14.0, 20.0, 30.0, 50.0, 80.0]
CURVATURE_MODES = ["signed_d2_lr", "diff_drop", "neg_diff_drop", "abs_diff_drop"]
CURVATURE_TAUS = [0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5]
SHRINK_CURVATURE = [True, False]
SIGNED_CURVATURE_COEF = [False, True]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def fmt_pct2(value: float) -> str:
    return f"{value:+.2f}%"


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


def curvature_feature(curve, response_lambda: float, mode: str) -> np.ndarray:
    eta = curve.lrs.astype(np.float64)
    peak = max(float(eta.max()), 1e-18)
    drop = np.zeros_like(eta)
    drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0)
    signal = np.zeros_like(eta)
    if mode == "signed_d2_lr":
        signal[2:] = eta[:-2] - 2.0 * eta[1:-1] + eta[2:]
    elif mode == "diff_drop":
        signal[1:] = drop[1:] - drop[:-1]
    elif mode == "neg_diff_drop":
        signal[1:] = -(drop[1:] - drop[:-1])
    elif mode == "abs_diff_drop":
        signal[1:] = np.abs(drop[1:] - drop[:-1])
    else:
        raise ValueError(f"unknown curvature mode: {mode}")

    acc = 0.0
    out = np.empty_like(eta)
    for t in range(len(eta)):
        acc = acc * math.exp(-response_lambda * float(eta[t])) + signal[t]
        out[t] = acc
    return (out / peak)[curve.step]


def fit_one(source, phi: np.ndarray, params: dict[str, float]) -> tuple[float, dict[str, float]]:
    mask = source.curve.step >= int(params["fit_start_step"])
    x = phi[mask]
    y = source.residual[mask]
    q = dct_basis(len(x), int(params["max_mode"]))
    x_o = soft_residualize(x, q, float(params["nuisance_lambda"]))
    y_o = soft_residualize(y, q, float(params["nuisance_lambda"]))
    l2 = float(np.dot(x_o, x_o))
    full_l2 = float(np.dot(x, x))
    dot = float(np.dot(x_o, y_o))
    raw = max(0.0, dot / max(l2 + float(params["ridge_tau"]) ** 2, 1e-18))
    retention = l2 / max(full_l2, 1e-18)
    shrink = 1.0 / (1.0 + float(params["rho"]))
    kappa = shrink * (max(retention, 0.0) ** float(params["retention_power"])) * raw
    return kappa, {
        "raw_primary": raw,
        "source_retention": retention,
        "source_dot": dot,
        "source_l2": l2,
        "source_full_l2": full_l2,
        "shrink": shrink,
    }


def fit_step_with_curvature(
    source,
    phi: np.ndarray,
    psi: np.ndarray,
    *,
    curvature_tau: float,
    shrink_curvature: bool,
    signed_curvature_coef: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    mask = source.curve.step >= int(STEP["fit_start_step"])
    x_raw = np.column_stack([phi[mask], psi[mask]])
    y = source.residual[mask]
    q = dct_basis(len(y), int(STEP["max_mode"]))
    x = np.column_stack(
        [soft_residualize(x_raw[:, j], q, float(STEP["nuisance_lambda"])) for j in range(2)]
    )
    y_o = soft_residualize(y, q, float(STEP["nuisance_lambda"]))
    ridge = np.diag([float(STEP["ridge_tau"]) ** 2, curvature_tau * curvature_tau])
    gram = x.T @ x + ridge
    rhs = x.T @ y_o

    candidates: list[np.ndarray] = []
    try:
        coef = np.linalg.solve(gram, rhs)
        if coef[0] >= 0.0 and (signed_curvature_coef or coef[1] >= 0.0):
            candidates.append(coef)
    except np.linalg.LinAlgError:
        pass
    primary_only = np.zeros(2, dtype=np.float64)
    primary_only[0] = max(0.0, rhs[0] / max(gram[0, 0], 1e-18))
    candidates.append(primary_only)
    curvature_only = np.zeros(2, dtype=np.float64)
    curvature_only[1] = rhs[1] / max(gram[1, 1], 1e-18)
    if not signed_curvature_coef:
        curvature_only[1] = max(0.0, curvature_only[1])
    candidates.append(curvature_only)
    candidates.append(np.zeros(2, dtype=np.float64))

    def objective(coef: np.ndarray) -> float:
        residual = x @ coef - y_o
        return float(np.dot(residual, residual) + coef @ ridge @ coef)

    coef = min(candidates, key=objective)
    primary_retention = float(np.dot(x[:, 0], x[:, 0]) / max(np.dot(x_raw[:, 0], x_raw[:, 0]), 1e-18))
    shrink = 1.0 / (1.0 + float(STEP["rho"]))
    primary_scale = shrink * (max(primary_retention, 0.0) ** float(STEP["retention_power"]))
    if shrink_curvature:
        coef = coef * primary_scale
    else:
        coef = np.array([coef[0] * primary_scale, coef[1]], dtype=np.float64)
    return coef, {
        "step_primary_coef": float(coef[0]),
        "step_curvature_coef": float(coef[1]),
        "step_primary_retention": primary_retention,
        "step_primary_shrink": primary_scale,
        "step_curvature_tau": curvature_tau,
        "step_curvature_shrunk": int(shrink_curvature),
        "step_curvature_signed_coef": int(signed_curvature_coef),
    }


def score_config(
    cache,
    primary_cache,
    curvature_cache,
    *,
    curvature_lambda: float,
    curvature_mode: str,
    curvature_tau: float,
    shrink_curvature: bool,
    signed_curvature_coef: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scale in SCALES:
        source = cache[(scale, TRAIN_CURVE)]
        smooth_phi = primary_cache[(scale, TRAIN_CURVE, "smooth")]
        smooth_kappa, smooth_fit = fit_one(source, smooth_phi, SMOOTH)
        step_phi = primary_cache[(scale, TRAIN_CURVE, "step")]
        step_curv = curvature_cache[(scale, TRAIN_CURVE, curvature_lambda, curvature_mode)]
        step_coef, step_fit = fit_step_with_curvature(
            source,
            step_phi,
            step_curv,
            curvature_tau=curvature_tau,
            shrink_curvature=shrink_curvature,
            signed_curvature_coef=signed_curvature_coef,
        )
        for target_curve, target_label in TARGETS:
            target = cache[(scale, target_curve)]
            channel = channel_for_curve(target.curve)
            if channel == "smooth":
                phi = primary_cache[(scale, target_curve, "smooth")]
                retention = target_retention(
                    phi,
                    nuisance_lambda=float(SMOOTH["nuisance_lambda"]),
                    max_mode=int(SMOOTH["max_mode"]),
                )
                factor = 1.0 if retention >= TARGET_RETENTION_FLOOR else 0.0
                pred = target.baseline + factor * smooth_kappa * phi
                extra = {
                    "primary_coef": smooth_kappa,
                    "curvature_coef": 0.0,
                    **smooth_fit,
                }
            else:
                phi = primary_cache[(scale, target_curve, "step")]
                curv = curvature_cache[(scale, target_curve, curvature_lambda, curvature_mode)]
                shape = step_coef[0] * phi + step_coef[1] * curv
                retention = (
                    target_retention(
                        shape,
                        nuisance_lambda=float(STEP["nuisance_lambda"]),
                        max_mode=int(STEP["max_mode"]),
                    )
                    if float(np.dot(shape, shape)) > 1e-18
                    else 0.0
                )
                factor = 1.0 if retention >= TARGET_RETENTION_FLOOR else 0.0
                pred = target.baseline + factor * shape
                extra = {
                    "primary_coef": float(step_coef[0]),
                    "curvature_coef": float(step_coef[1]),
                    **step_fit,
                }
            corr_mae = metrics(target.curve.loss, pred)["mae"]
            rows.append(
                {
                    "scale": scale,
                    "train_curve": TRAIN_CURVE,
                    "test_curve": target_curve,
                    "test_label": target_label,
                    "channel": channel,
                    "target_retention": retention,
                    "base_mae": target.base_mae,
                    "corr_mae": corr_mae,
                    "delta_pct": 100.0 * (corr_mae / target.base_mae - 1.0),
                    "win": int(corr_mae < target.base_mae),
                    "curvature_lambda": curvature_lambda,
                    "curvature_mode": curvature_mode,
                    "curvature_tau": curvature_tau,
                    "shrink_curvature": int(shrink_curvature),
                    "signed_curvature_coef": int(signed_curvature_coef),
                    **extra,
                }
            )
    return rows


def run_search() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    cache = build_cache()
    primary_cache = {
        (scale, curve_name, "smooth"): stime_feature(cache[(scale, curve_name)].curve, float(SMOOTH["response_lambda"]))
        for scale in SCALES
        for curve_name, _ in [(TRAIN_CURVE, "Cosine")] + TARGETS
    }
    primary_cache.update(
        {
            (scale, curve_name, "step"): stime_feature(cache[(scale, curve_name)].curve, float(STEP["response_lambda"]))
            for scale in SCALES
            for curve_name, _ in [(TRAIN_CURVE, "Cosine")] + TARGETS
        }
    )
    curvature_cache = {
        (scale, curve_name, curvature_lambda, curvature_mode): curvature_feature(
            cache[(scale, curve_name)].curve, curvature_lambda, curvature_mode
        )
        for scale in SCALES
        for curve_name, _ in [(TRAIN_CURVE, "Cosine")] + TARGETS
        for curvature_lambda in CURVATURE_LAMBDAS
        for curvature_mode in CURVATURE_MODES
    }
    config_rows: list[dict[str, object]] = []
    safe_detail_rows: list[dict[str, object]] = []
    config_id = 0
    for curvature_lambda in CURVATURE_LAMBDAS:
        for curvature_mode in CURVATURE_MODES:
            for curvature_tau in CURVATURE_TAUS:
                for shrink_curvature in SHRINK_CURVATURE:
                    for signed_curvature_coef in SIGNED_CURVATURE_COEF:
                        details = score_config(
                            cache,
                            primary_cache,
                            curvature_cache,
                            curvature_lambda=curvature_lambda,
                            curvature_mode=curvature_mode,
                            curvature_tau=curvature_tau,
                            shrink_curvature=shrink_curvature,
                            signed_curvature_coef=signed_curvature_coef,
                        )
                        summary = aggregate(details)
                        row = {
                            "config_id": config_id,
                            "curvature_lambda": curvature_lambda,
                            "curvature_mode": curvature_mode,
                            "curvature_tau": curvature_tau,
                            "shrink_curvature": int(shrink_curvature),
                            "signed_curvature_coef": int(signed_curvature_coef),
                            **summary,
                            "mean_step_curvature_coef": float(
                                np.mean([float(detail["curvature_coef"]) for detail in details if detail["channel"] == "step"])
                            ),
                            "mean_step_primary_coef": float(
                                np.mean([float(detail["primary_coef"]) for detail in details if detail["channel"] == "step"])
                            ),
                        }
                        config_rows.append(row)
                        if summary["wins"] == summary["rows"] and summary["nonharm"] == summary["rows"]:
                            for detail in details:
                                safe_detail_rows.append({"config_id": config_id, **detail})
                        config_id += 1
    safe_rows = [
        row for row in config_rows if int(row["wins"]) == int(row["rows"]) and int(row["nonharm"]) == int(row["rows"])
    ]
    safe_sorted = sorted(
        safe_rows,
        key=lambda row: (
            float(row["mean_delta"]),
            float(row["worst_delta"]),
            int(row["signed_curvature_coef"]),
            -int(row["shrink_curvature"]),
        ),
    )
    top_ids = {int(row["config_id"]) for row in safe_sorted[:200]}
    top_details = [row for row in safe_detail_rows if int(row["config_id"]) in top_ids]
    return config_rows, safe_sorted[:200], top_details


def summarize_by_target(detail_rows: list[dict[str, object]], config_id: int) -> list[dict[str, object]]:
    selected = [row for row in detail_rows if int(row["config_id"]) == config_id]
    rows: list[dict[str, object]] = []
    for target_curve, target_label in TARGETS:
        sub = [row for row in selected if row["test_curve"] == target_curve]
        if sub:
            rows.append({"test_curve": target_curve, "test_label": target_label, **aggregate(sub)})
    return rows


def split_defs(targets: set[str]) -> list[dict[str, object]]:
    splits: list[dict[str, object]] = [
        {
            "split": "dev_sharp_linear__test_wsdcon",
            "dev_targets": SHARP_LINEAR,
            "test_targets": WSDCON,
            "dev_scales": None,
            "test_scales": None,
        },
        {
            "split": "dev_wsdcon__test_sharp_linear",
            "dev_targets": WSDCON,
            "test_targets": SHARP_LINEAR,
            "dev_scales": None,
            "test_scales": None,
        },
    ]
    for target in sorted(targets):
        splits.append(
            {
                "split": f"leave_target__{target}",
                "dev_targets": targets - {target},
                "test_targets": {target},
                "dev_scales": None,
                "test_scales": None,
            }
        )
    for scale in SCALES:
        splits.append(
            {
                "split": f"leave_scale__{scale}M",
                "dev_targets": targets,
                "test_targets": targets,
                "dev_scales": set(SCALES) - {scale},
                "test_scales": {scale},
            }
        )
    return splits


def select_rows(
    rows: list[dict[str, object]], *, targets: set[str], scales: set[str] | None
) -> list[dict[str, object]]:
    return [row for row in rows if row["test_curve"] in targets and (scales is None or row["scale"] in scales)]


def top_holdout(detail_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_config: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in detail_rows:
        by_config[int(row["config_id"])].append(row)
    targets = {str(row["test_curve"]) for row in detail_rows}
    out: list[dict[str, object]] = []
    for split in split_defs(targets):
        candidates: list[tuple[float, float, int, dict[str, object], dict[str, object], dict[str, object]]] = []
        for config_id, rows in by_config.items():
            dev = select_rows(rows, targets=split["dev_targets"], scales=split["dev_scales"])
            test = select_rows(rows, targets=split["test_targets"], scales=split["test_scales"])
            if not dev or not test:
                continue
            dev_stats = aggregate(dev)
            if dev_stats["wins"] != dev_stats["rows"] or dev_stats["nonharm"] != dev_stats["rows"]:
                continue
            test_stats = aggregate(test)
            candidates.append(
                (
                    float(dev_stats["mean_delta"]),
                    float(dev_stats["worst_delta"]),
                    config_id,
                    dev_stats,
                    test_stats,
                    rows[0],
                )
            )
        if not candidates:
            out.append({"split": split["split"], "selection_status": "no_candidate"})
            continue
        candidates.sort(key=lambda item: (item[0], item[1]))
        _, _, config_id, dev_stats, test_stats, cfg = candidates[0]
        out.append(
            {
                "split": split["split"],
                "selection_status": "selected",
                "config_id": config_id,
                "curvature_lambda": cfg["curvature_lambda"],
                "curvature_mode": cfg["curvature_mode"],
                "curvature_tau": cfg["curvature_tau"],
                "shrink_curvature": cfg["shrink_curvature"],
                "signed_curvature_coef": cfg["signed_curvature_coef"],
                **{f"dev_{key}": value for key, value in dev_stats.items()},
                **{f"test_{key}": value for key, value in test_stats.items()},
            }
        )
    return out


def write_report(
    safe_rows: list[dict[str, object]],
    target_rows: list[dict[str, object]],
    holdout_rows: list[dict[str, object]],
) -> None:
    if not safe_rows:
        (OUT_DIR / "REPORT.md").write_text("No non-harming curvature candidate found.\n", encoding="utf-8")
        return
    best = safe_rows[0]
    best_worst = min(safe_rows, key=lambda row: (float(row["worst_delta"]), float(row["mean_delta"])))
    baseline = read_csv(BASELINE_DIR / "safe_decoupled_channel_top200.csv")[0]
    lines = [
        "# LR-Curvature Cosine-to-WSD Audit\n\n",
        "This audit extends the decoupled-channel model with one step-channel feature: a causal relaxation "
        "of the second finite difference of the LR schedule. Coefficients are fitted only from "
        "`cosine_72000.csv` residuals.\n\n",
        "## Formula Change\n\n",
        "```text\n",
        "psi_lambda(t) = causal_relax_lambda(eta_{t-2} - 2 eta_{t-1} + eta_t) / eta_peak\n",
        "step correction = a * phi_step(t) + b * psi_lambda(t)\n",
        "smooth correction = kappa_smooth * phi_smooth(t)\n",
        "L_hat_target = L_MPL,target + correction_channel(target)\n",
        "```\n\n",
        "The curvature term is schedule-only. On WSD-con schedules it acts near the abrupt LR transition, "
        "which directly targets the tail overshoot left by the first-order response model.\n\n",
        "## Best Fully Non-Harming Candidate\n\n",
        f"- Mean MAE change: `{fmt_pct2(float(best['mean_delta']))}` over `{int(best['rows'])}` scale-target rows.\n",
        f"- Worst scale-target row: `{fmt_pct2(float(best['worst_delta']))}`.\n",
        f"- Wins/non-harm: `{int(best['wins'])}/{int(best['rows'])}` and `{int(best['nonharm'])}/{int(best['rows'])}`.\n",
        f"- Config: `curvature_lambda={float(best['curvature_lambda']):g}`, "
        f"`mode={best['curvature_mode']}`, `tau2={float(best['curvature_tau']):g}`, "
        f"`shrink_curvature={int(best['shrink_curvature'])}`, "
        f"`signed_curvature_coef={int(best['signed_curvature_coef'])}`.\n",
        f"- Mean step coefficients: primary `{float(best['mean_step_primary_coef']):.5f}`, "
        f"curvature `{float(best['mean_step_curvature_coef']):.5f}`.\n\n",
        "## Best Worst-Case Candidate\n\n",
        f"- Mean / worst: `{fmt_pct2(float(best_worst['mean_delta']))}` / "
        f"`{fmt_pct2(float(best_worst['worst_delta']))}`.\n",
        f"- Config: `curvature_lambda={float(best_worst['curvature_lambda']):g}`, "
        f"`mode={best_worst['curvature_mode']}`, `tau2={float(best_worst['curvature_tau']):g}`, "
        f"`shrink_curvature={int(best_worst['shrink_curvature'])}`, "
        f"`signed_curvature_coef={int(best_worst['signed_curvature_coef'])}`.\n\n",
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
        f"Decoupled-channel: mean `{fmt_pct2(float(baseline['mean_delta']))}`, "
        f"worst `{fmt_pct2(float(baseline['worst_delta']))}`.\n",
        f"Curvature correction: mean `{fmt_pct2(float(best['mean_delta']))}`, "
        f"worst `{fmt_pct2(float(best['worst_delta']))}`.\n\n",
        "## Top-Safe Holdout Check\n\n",
        "| split | selected config | dev mean | dev worst | test mean | test worst | test wins |\n",
        "|---|---|---:|---:|---:|---:|---:|\n",
    ]
    for row in holdout_rows:
        if row["selection_status"] != "selected":
            lines.append(f"| {row['split']} | none | | | | | |\n")
            continue
        cfg = (
            f"lambda2={float(row['curvature_lambda']):g}, mode={row['curvature_mode']}, "
            f"tau2={float(row['curvature_tau']):g}, shrink={int(row['shrink_curvature'])}, "
            f"signed={int(row['signed_curvature_coef'])}"
        )
        lines.append(
            f"| {row['split']} | `{cfg}` | "
            f"{fmt_pct(float(row['dev_mean_delta']))} | {fmt_pct(float(row['dev_worst_delta']))} | "
            f"{fmt_pct(float(row['test_mean_delta']))} | {fmt_pct(float(row['test_worst_delta']))} | "
            f"{int(row['test_wins'])}/{int(row['test_rows'])} |\n"
        )
    lines += [
        "\n## Reading\n\n",
        "- The gain is concentrated on WSD-con targets, especially the high-tail-LR settings where the first-order lag leaves a long tail residual.\n",
        "- This is more interpretable than a sinusoidal residual basis: the added variable is the LR schedule curvature, not a free time-series basis.\n",
        "- It is still a development result because `tau2` and the curvature kernel were selected by WSD-family ranking. A frozen-protocol test on new schedules is the next proof step.\n",
    ]
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    config_rows, safe_rows, detail_rows = run_search()
    write_csv(OUT_DIR / "all_curvature_configs.csv", config_rows)
    write_csv(OUT_DIR / "safe_curvature_top200.csv", safe_rows)
    write_csv(OUT_DIR / "top_safe_details.csv", detail_rows)
    target_rows = summarize_by_target(detail_rows, int(safe_rows[0]["config_id"])) if safe_rows else []
    holdout_rows = top_holdout(detail_rows) if detail_rows else []
    write_csv(OUT_DIR / "best_target_summary.csv", target_rows)
    write_csv(OUT_DIR / "top_holdout_summary.csv", holdout_rows)
    write_report(safe_rows, target_rows, holdout_rows)
    print(f"wrote {OUT_DIR / 'all_curvature_configs.csv'}")
    print(f"wrote {OUT_DIR / 'safe_curvature_top200.csv'}")
    print(f"wrote {OUT_DIR / 'top_safe_details.csv'}")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
