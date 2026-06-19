#!/usr/bin/env python3
"""Alternating MPL/error refit audit with one forced step-response channel.

This is the no-routing version of the previous alternating audit:

1. fit / load a strict cosine-only MPL backbone;
2. freeze MPL and fit one step-response correction on cosine_72000 residuals;
3. subtract that correction from the cosine training curves and refit MPL;
4. freeze the refit MPL, fit the same step-response correction again;
5. evaluate every WSD-family target with the same step-response correction.

No target uses a smooth/step routing rule.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TMP_CACHE = Path(os.environ.get("TMPDIR", "/tmp")) / "dl-final-cache"
TMP_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(TMP_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(TMP_CACHE / "xdg"))

REPO = Path(__file__).resolve().parent
ROOT = REPO.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cosine_to_wsd_curvature_correction import curvature_feature, fmt_pct, fmt_pct2  # noqa: E402
from cosine_to_wsd_response_search import (  # noqa: E402
    TARGETS,
    TARGET_RETENTION_FLOOR,
    TRAIN_CURVE,
    dct_basis,
    soft_residualize,
    stime_feature,
    target_retention,
)
from reproduce_cosine_to_wsd import (  # noqa: E402
    MPL_PRECOMPUTED_INIT,
    SCALES,
    TRAIN_CURVES,
    Curve,
    fit_with_restarts,
    huber_log_residual,
    load_curve,
    metrics,
    mpl_predict,
    subsample_curve,
)


STRICT_BACKBONE_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "cosine_only_backbone"
OUT_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "alternating_forced_step_refit"

CONFIGS = [
    {
        "name": "forced_step_best_mean",
        "fit_start_step": 3000,
        "response_lambda": 20.0,
        "nuisance_lambda": 0.03,
        "max_mode": 8,
        "ridge_tau": 0.05,
        "retention_power": 0.25,
        "rho": 0.2,
        "curvature_lambda": 4.0,
        "curvature_mode": "signed_d2_lr",
        "curvature_tau": 0.01,
        "shrink_curvature": False,
        "signed_curvature_coef": False,
    },
    {
        "name": "forced_step_best_worst",
        "fit_start_step": 3000,
        "response_lambda": 20.0,
        "nuisance_lambda": 0.01,
        "max_mode": 16,
        "ridge_tau": 0.05,
        "retention_power": 0.0,
        "rho": 0.35,
        "curvature_lambda": 10.0,
        "curvature_mode": "signed_d2_lr",
        "curvature_tau": 0.003,
        "shrink_curvature": False,
        "signed_curvature_coef": False,
    },
    {
        "name": "forced_step_current_core",
        "fit_start_step": 3000,
        "response_lambda": 20.0,
        "nuisance_lambda": 0.01,
        "max_mode": 8,
        "ridge_tau": 0.05,
        "retention_power": 0.0,
        "rho": 0.35,
        "curvature_lambda": 10.0,
        "curvature_mode": "signed_d2_lr",
        "curvature_tau": 0.003,
        "shrink_curvature": True,
        "signed_curvature_coef": False,
    },
]


@dataclass(frozen=True)
class StepFit:
    primary_coef: float
    curvature_coef: float
    primary_retention: float


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


def aggregate(rows: list[dict[str, object]], key: str) -> dict[str, object]:
    deltas = np.array([float(row[key]) for row in rows], dtype=np.float64)
    return {
        "rows": len(rows),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "worst_delta": float(np.max(deltas)),
        "wins": int(np.sum(deltas < 0.0)),
        "nonharm": int(np.sum(deltas <= 1e-12)),
    }


def load_or_fit_pure_mpl() -> dict[str, np.ndarray]:
    path = STRICT_BACKBONE_DIR / "cosine_only_mpl_params.json"
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {scale: np.array(values, dtype=np.float64) for scale, values in raw.items()}

    from reproduce_cosine_to_wsd import fit_mpl

    params: dict[str, np.ndarray] = {}
    for scale in SCALES:
        curves = [subsample_curve(load_curve(scale, name)) for name in TRAIN_CURVES]
        fitted, _ = fit_mpl(curves, scale)
        params[scale] = fitted
    return params


def fit_mpl_custom(curves: list[Curve], scale: str, init: np.ndarray) -> tuple[np.ndarray, float]:
    min_loss = min(float(curve.loss.min()) for curve in curves)
    inits = [
        np.asarray(init, dtype=np.float64),
        MPL_PRECOMPUTED_INIT[scale],
        np.array([min_loss - 0.05, 0.5, 0.5, 300.0, 2.0, 0.5, 0.5], dtype=np.float64),
    ]

    def objective(params: np.ndarray) -> float:
        pred_all = []
        loss_all = []
        for curve in curves:
            pred = mpl_predict(params, curve)
            if np.any(~np.isfinite(pred)) or np.any(pred <= 0.0):
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


def residual_curve(curve: Curve, params: np.ndarray) -> np.ndarray:
    return curve.loss - mpl_predict(params, curve)


def fit_step(residual: np.ndarray, curve: Curve, cfg: dict[str, object]) -> StepFit:
    phi = stime_feature(curve, float(cfg["response_lambda"]))
    psi = curvature_feature(curve, float(cfg["curvature_lambda"]), str(cfg["curvature_mode"]))
    mask = curve.step >= int(cfg["fit_start_step"])
    x_raw = np.column_stack([phi[mask], psi[mask]])
    y = residual[mask]
    q = dct_basis(len(y), int(cfg["max_mode"]))
    x = np.column_stack(
        [soft_residualize(x_raw[:, j], q, float(cfg["nuisance_lambda"])) for j in range(2)]
    )
    y_o = soft_residualize(y, q, float(cfg["nuisance_lambda"]))
    ridge = np.diag([float(cfg["ridge_tau"]) ** 2, float(cfg["curvature_tau"]) ** 2])
    gram = x.T @ x + ridge
    rhs = x.T @ y_o

    candidates: list[np.ndarray] = []
    try:
        coef = np.linalg.solve(gram, rhs)
        if coef[0] >= 0.0 and (bool(cfg["signed_curvature_coef"]) or coef[1] >= 0.0):
            candidates.append(coef)
    except np.linalg.LinAlgError:
        pass

    primary_only = np.zeros(2, dtype=np.float64)
    primary_only[0] = max(0.0, rhs[0] / max(gram[0, 0], 1e-18))
    candidates.append(primary_only)

    curvature_only = np.zeros(2, dtype=np.float64)
    curvature_only[1] = rhs[1] / max(gram[1, 1], 1e-18)
    if not bool(cfg["signed_curvature_coef"]):
        curvature_only[1] = max(0.0, curvature_only[1])
    candidates.append(curvature_only)
    candidates.append(np.zeros(2, dtype=np.float64))

    def objective(coef: np.ndarray) -> float:
        diff = x @ coef - y_o
        return float(np.dot(diff, diff) + coef @ ridge @ coef)

    coef = min(candidates, key=objective)
    primary_retention = float(np.dot(x[:, 0], x[:, 0]) / max(np.dot(x_raw[:, 0], x_raw[:, 0]), 1e-18))
    primary_scale = (
        (1.0 / (1.0 + float(cfg["rho"])))
        * (max(primary_retention, 0.0) ** float(cfg["retention_power"]))
    )
    if bool(cfg["shrink_curvature"]):
        coef = coef * primary_scale
    else:
        coef = np.array([coef[0] * primary_scale, coef[1]], dtype=np.float64)
    return StepFit(float(coef[0]), float(coef[1]), primary_retention)


def step_correction(curve: Curve, fit: StepFit, cfg: dict[str, object]) -> np.ndarray:
    return fit.primary_coef * stime_feature(curve, float(cfg["response_lambda"])) + fit.curvature_coef * curvature_feature(
        curve,
        float(cfg["curvature_lambda"]),
        str(cfg["curvature_mode"]),
    )


def target_step_correction(curve: Curve, fit: StepFit, cfg: dict[str, object]) -> tuple[np.ndarray, float]:
    corr = step_correction(curve, fit, cfg)
    retention = (
        target_retention(corr, nuisance_lambda=float(cfg["nuisance_lambda"]), max_mode=int(cfg["max_mode"]))
        if float(np.dot(corr, corr)) > 1e-18
        else 0.0
    )
    if retention < TARGET_RETENTION_FLOOR:
        corr = np.zeros_like(curve.loss)
    return corr, retention


def adjusted_train_curves(scale: str, first_fit: StepFit, cfg: dict[str, object]) -> list[Curve]:
    adjusted = []
    for name in TRAIN_CURVES:
        curve = load_curve(scale, name)
        corr = step_correction(curve, first_fit, cfg)
        adjusted.append(Curve(curve.name, curve.scale, curve.step, curve.loss - corr, curve.lrs))
    return [subsample_curve(curve) for curve in adjusted]


def run_config_scale(scale: str, pure_params: np.ndarray, cfg: dict[str, object]) -> tuple[np.ndarray, StepFit, StepFit, float]:
    source = load_curve(scale, TRAIN_CURVE)
    first_fit = fit_step(residual_curve(source, pure_params), source, cfg)
    adjusted = adjusted_train_curves(scale, first_fit, cfg)
    refit_params, obj = fit_mpl_custom(adjusted, scale, pure_params)
    final_fit = fit_step(residual_curve(source, refit_params), source, cfg)
    return refit_params, first_fit, final_fit, obj


def evaluate_config_scale(
    scale: str,
    pure_params: np.ndarray,
    refit_params: np.ndarray,
    first_fit: StepFit,
    final_fit: StepFit,
    cfg: dict[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for target_curve, target_label in TARGETS:
        curve = load_curve(scale, target_curve)
        pure_pred = mpl_predict(pure_params, curve)
        first_corr, first_retention = target_step_correction(curve, first_fit, cfg)
        first_pred = pure_pred + first_corr
        refit_pred = mpl_predict(refit_params, curve)
        final_corr, final_retention = target_step_correction(curve, final_fit, cfg)
        final_pred = refit_pred + final_corr

        pure_mae = metrics(curve.loss, pure_pred)["mae"]
        first_mae = metrics(curve.loss, first_pred)["mae"]
        refit_mae = metrics(curve.loss, refit_pred)["mae"]
        final_mae = metrics(curve.loss, final_pred)["mae"]
        rows.append(
            {
                "config": cfg["name"],
                "scale": scale,
                "test_curve": target_curve,
                "test_label": target_label,
                "forced_channel": "step",
                "pure_mpl_mae": pure_mae,
                "first_error_mae": first_mae,
                "refit_mpl_mae": refit_mae,
                "final_error_mae": final_mae,
                "delta_first_vs_pure_pct": 100.0 * (first_mae / pure_mae - 1.0),
                "delta_refit_mpl_vs_pure_pct": 100.0 * (refit_mae / pure_mae - 1.0),
                "delta_vs_pure_pct": 100.0 * (final_mae / pure_mae - 1.0),
                "delta_vs_first_error_pct": 100.0 * (final_mae / first_mae - 1.0),
                "win_vs_pure": int(final_mae < pure_mae),
                "win_vs_first_error": int(final_mae < first_mae),
                "first_target_retention": first_retention,
                "final_target_retention": final_retention,
                "first_step_primary_coef": first_fit.primary_coef,
                "first_step_curvature_coef": first_fit.curvature_coef,
                "final_step_primary_coef": final_fit.primary_coef,
                "final_step_curvature_coef": final_fit.curvature_coef,
                **{f"cfg_{key}": value for key, value in cfg.items() if key != "name"},
            }
        )
    return rows


def summarize(details: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cfg in CONFIGS:
        sub = [row for row in details if row["config"] == cfg["name"]]
        rows.append({"config": cfg["name"], "baseline": "pure_mpl", **aggregate(sub, "delta_vs_pure_pct")})
        rows.append({"config": cfg["name"], "baseline": "first_error", **aggregate(sub, "delta_vs_first_error_pct")})
        rows.append({"config": cfg["name"], "baseline": "first_vs_pure", **aggregate(sub, "delta_first_vs_pure_pct")})
    return rows


def summarize_targets(details: list[dict[str, object]], config: str, key: str) -> list[dict[str, object]]:
    selected = [row for row in details if row["config"] == config]
    rows: list[dict[str, object]] = []
    for target_curve, target_label in TARGETS:
        sub = [row for row in selected if row["test_curve"] == target_curve]
        rows.append({"test_curve": target_curve, "test_label": target_label, **aggregate(sub, key)})
    return rows


def write_report(details: list[dict[str, object]], summary: list[dict[str, object]]) -> None:
    final_rows = [row for row in summary if row["baseline"] == "pure_mpl"]
    safe = [
        row
        for row in final_rows
        if int(row["wins"]) == int(row["rows"]) and int(row["nonharm"]) == int(row["rows"])
    ]
    best = min(safe if safe else final_rows, key=lambda row: (float(row["mean_delta"]), float(row["worst_delta"])))
    best_first = next(row for row in summary if row["config"] == best["config"] and row["baseline"] == "first_vs_pure")
    best_vs_first = next(row for row in summary if row["config"] == best["config"] and row["baseline"] == "first_error")

    lines = [
        "# Alternating Forced-Step Refit Audit\n\n",
        "This audit removes smooth/step routing. Every target uses the same step-response correction:\n\n",
        "```text\n",
        "C(t) = a * phi_20(t) + b * psi(t)\n",
        "```\n\n",
        "The optimization is the frozen alternating procedure: fit MPL, freeze MPL and fit residual, "
        "freeze residual and refit MPL on corrected cosine losses, then freeze MPL and fit residual again.\n\n",
        "## Best Final Alternating Result\n\n",
        f"- Config: `{best['config']}`.\n",
        f"- First two-stage vs pure MPL: mean `{fmt_pct2(float(best_first['mean_delta']))}`, "
        f"worst `{fmt_pct2(float(best_first['worst_delta']))}`, wins `{int(best_first['wins'])}/{int(best_first['rows'])}`.\n",
        f"- Alternating final vs pure MPL: mean `{fmt_pct2(float(best['mean_delta']))}`, "
        f"worst `{fmt_pct2(float(best['worst_delta']))}`, wins `{int(best['wins'])}/{int(best['rows'])}`.\n",
        f"- Alternating final vs first two-stage: mean `{fmt_pct2(float(best_vs_first['mean_delta']))}`, "
        f"worst `{fmt_pct2(float(best_vs_first['worst_delta']))}`, wins "
        f"`{int(best_vs_first['wins'])}/{int(best_vs_first['rows'])}`.\n\n",
        "## Config Summary\n\n",
        "| config | baseline | mean delta | worst | wins |\n",
        "|---|---|---:|---:|---:|\n",
    ]
    for row in summary:
        lines.append(
            f"| {row['config']} | {row['baseline']} | {fmt_pct2(float(row['mean_delta']))} | "
            f"{fmt_pct2(float(row['worst_delta']))} | {int(row['wins'])}/{int(row['rows'])} |\n"
        )

    lines += [
        "\n## Per-Target Result For Best Final Config\n\n",
        "| target | first two-stage mean | alternating final mean | final worst | final wins |\n",
        "|---|---:|---:|---:|---:|\n",
    ]
    first_targets = summarize_targets(details, str(best["config"]), "delta_first_vs_pure_pct")
    final_targets = summarize_targets(details, str(best["config"]), "delta_vs_pure_pct")
    for first, final in zip(first_targets, final_targets):
        lines.append(
            f"| {final['test_label']} | {fmt_pct(float(first['mean_delta']))} | "
            f"{fmt_pct(float(final['mean_delta']))} | {fmt_pct(float(final['worst_delta']))} | "
            f"{int(final['wins'])}/{int(final['rows'])} |\n"
        )

    lines += [
        "\n## Reading\n\n",
        "- The comparison that matters is `alternating final vs first two-stage`. "
        "If this row is positive, the frozen iteration did not improve the forced-step estimator.\n",
        "- This audit still uses only cosine losses for fitting MPL and residual coefficients. "
        "WSD-family curves are used for evaluation and development ranking only.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pure = load_or_fit_pure_mpl()
    details: list[dict[str, object]] = []
    params_rows: list[dict[str, object]] = []
    for cfg in CONFIGS:
        for scale in SCALES:
            refit_params, first_fit, final_fit, obj = run_config_scale(scale, pure[scale], cfg)
            params_rows.extend(
                [
                    {
                        "config": cfg["name"],
                        "scale": scale,
                        "stage": "pure_mpl",
                        "objective": "",
                        **{f"p{i}": float(value) for i, value in enumerate(pure[scale])},
                        "step_primary_coef": "",
                        "step_curvature_coef": "",
                    },
                    {
                        "config": cfg["name"],
                        "scale": scale,
                        "stage": "refit_mpl",
                        "objective": obj,
                        **{f"p{i}": float(value) for i, value in enumerate(refit_params)},
                        "step_primary_coef": "",
                        "step_curvature_coef": "",
                    },
                    {
                        "config": cfg["name"],
                        "scale": scale,
                        "stage": "first_error",
                        "objective": "",
                        **{f"p{i}": "" for i in range(7)},
                        "step_primary_coef": first_fit.primary_coef,
                        "step_curvature_coef": first_fit.curvature_coef,
                    },
                    {
                        "config": cfg["name"],
                        "scale": scale,
                        "stage": "final_error",
                        "objective": "",
                        **{f"p{i}": "" for i in range(7)},
                        "step_primary_coef": final_fit.primary_coef,
                        "step_curvature_coef": final_fit.curvature_coef,
                    },
                ]
            )
            details.extend(evaluate_config_scale(scale, pure[scale], refit_params, first_fit, final_fit, cfg))

    summary = summarize(details)
    write_csv(OUT_DIR / "details.csv", details)
    write_csv(OUT_DIR / "params.csv", params_rows)
    write_csv(OUT_DIR / "summary.csv", summary)
    write_report(details, summary)
    print(f"wrote {OUT_DIR / 'summary.csv'}")
    print(f"wrote {OUT_DIR / 'details.csv'}")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
