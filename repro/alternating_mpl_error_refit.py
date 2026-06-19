#!/usr/bin/env python3
"""Alternating MPL/error refit audit for cosine-to-WSD prediction.

Algorithm requested by the user:

1. fit a pure MPL backbone on cosine curves;
2. freeze MPL and fit the schedule-error term on cosine_72000 residuals;
3. freeze that error term, subtract it from the original cosine losses, and
   refit MPL;
4. freeze the refit MPL, fit the error term again, and evaluate WSD-family
   targets.

This is a strict cosine-only protocol: no WSD residual is used to fit MPL or
the correction coefficients.
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

from cosine_to_wsd_adaptive_fit_window import channel_for_curve  # noqa: E402
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
    SCIPY_MAXITER,
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
STRICT_CURVATURE_DIR = (
    ROOT / "results" / "cosine_to_wsd_response_search" / "cosine_only_backbone_curvature_calibrated"
)
OUT_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "alternating_mpl_error_refit"

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
    "fit_start_step": 12000,
    "response_lambda": 20.0,
    "nuisance_lambda": 0.02,
    "max_mode": 12,
    "ridge_tau": 0.05,
    "retention_power": 0.0,
    "rho": 0.0,
}
CURVATURE = {
    "curvature_lambda": 30.0,
    "curvature_mode": "diff_drop",
    "curvature_tau": 0.001,
    "shrink_curvature": True,
    "signed_curvature_coef": False,
}
SUBTRACTION_VARIANTS = ["smooth_only", "step_only", "smooth_plus_step"]


@dataclass(frozen=True)
class ErrorFit:
    smooth_coef: float
    step_primary_coef: float
    step_curvature_coef: float
    smooth_retention: float
    step_primary_retention: float


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


def aggregate(rows: list[dict[str, object]], key: str = "delta_vs_pure_pct") -> dict[str, object]:
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

    from reproduce_cosine_to_wsd import fit_mpl  # Local import avoids unused cost when cached.

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
    old_maxiter = SCIPY_MAXITER
    del old_maxiter
    return fit_with_restarts(objective, inits, bounds)


def residual_curve(curve: Curve, params: np.ndarray) -> np.ndarray:
    return curve.loss - mpl_predict(params, curve)


def fit_primary(residual: np.ndarray, curve: Curve, phi: np.ndarray, params: dict[str, float]) -> tuple[float, float]:
    mask = curve.step >= int(params["fit_start_step"])
    x = phi[mask]
    y = residual[mask]
    q = dct_basis(len(x), int(params["max_mode"]))
    x_o = soft_residualize(x, q, float(params["nuisance_lambda"]))
    y_o = soft_residualize(y, q, float(params["nuisance_lambda"]))
    l2 = float(np.dot(x_o, x_o))
    full_l2 = float(np.dot(x, x))
    dot = float(np.dot(x_o, y_o))
    raw = max(0.0, dot / max(l2 + float(params["ridge_tau"]) ** 2, 1e-18))
    retention = l2 / max(full_l2, 1e-18)
    shrink = 1.0 / (1.0 + float(params["rho"]))
    coef = shrink * (max(retention, 0.0) ** float(params["retention_power"])) * raw
    return coef, retention


def fit_step(
    residual: np.ndarray,
    curve: Curve,
    phi: np.ndarray,
    psi: np.ndarray,
) -> tuple[np.ndarray, float]:
    mask = curve.step >= int(STEP["fit_start_step"])
    x_raw = np.column_stack([phi[mask], psi[mask]])
    y = residual[mask]
    q = dct_basis(len(y), int(STEP["max_mode"]))
    x = np.column_stack([soft_residualize(x_raw[:, j], q, float(STEP["nuisance_lambda"])) for j in range(2)])
    y_o = soft_residualize(y, q, float(STEP["nuisance_lambda"]))
    ridge = np.diag([float(STEP["ridge_tau"]) ** 2, float(CURVATURE["curvature_tau"]) ** 2])
    gram = x.T @ x + ridge
    rhs = x.T @ y_o

    candidates: list[np.ndarray] = []
    try:
        coef = np.linalg.solve(gram, rhs)
        if coef[0] >= 0.0 and (bool(CURVATURE["signed_curvature_coef"]) or coef[1] >= 0.0):
            candidates.append(coef)
    except np.linalg.LinAlgError:
        pass
    primary_only = np.zeros(2, dtype=np.float64)
    primary_only[0] = max(0.0, rhs[0] / max(gram[0, 0], 1e-18))
    candidates.append(primary_only)
    curvature_only = np.zeros(2, dtype=np.float64)
    curvature_only[1] = rhs[1] / max(gram[1, 1], 1e-18)
    if not bool(CURVATURE["signed_curvature_coef"]):
        curvature_only[1] = max(0.0, curvature_only[1])
    candidates.append(curvature_only)
    candidates.append(np.zeros(2, dtype=np.float64))

    def objective(coef: np.ndarray) -> float:
        diff = x @ coef - y_o
        return float(np.dot(diff, diff) + coef @ ridge @ coef)

    coef = min(candidates, key=objective)
    primary_retention = float(np.dot(x[:, 0], x[:, 0]) / max(np.dot(x_raw[:, 0], x_raw[:, 0]), 1e-18))
    primary_scale = (
        (1.0 / (1.0 + float(STEP["rho"])))
        * (max(primary_retention, 0.0) ** float(STEP["retention_power"]))
    )
    if bool(CURVATURE["shrink_curvature"]):
        coef = coef * primary_scale
    else:
        coef = np.array([coef[0] * primary_scale, coef[1]], dtype=np.float64)
    return coef, primary_retention


def fit_error(scale: str, mpl_params: np.ndarray) -> ErrorFit:
    curve = load_curve(scale, TRAIN_CURVE)
    residual = residual_curve(curve, mpl_params)
    smooth_phi = stime_feature(curve, float(SMOOTH["response_lambda"]))
    smooth_coef, smooth_retention = fit_primary(residual, curve, smooth_phi, SMOOTH)
    step_phi = stime_feature(curve, float(STEP["response_lambda"]))
    step_psi = curvature_feature(
        curve,
        float(CURVATURE["curvature_lambda"]),
        str(CURVATURE["curvature_mode"]),
    )
    step_coef, step_retention = fit_step(residual, curve, step_phi, step_psi)
    return ErrorFit(
        smooth_coef=float(smooth_coef),
        step_primary_coef=float(step_coef[0]),
        step_curvature_coef=float(step_coef[1]),
        smooth_retention=float(smooth_retention),
        step_primary_retention=float(step_retention),
    )


def smooth_correction(curve: Curve, fit: ErrorFit) -> np.ndarray:
    return fit.smooth_coef * stime_feature(curve, float(SMOOTH["response_lambda"]))


def step_correction(curve: Curve, fit: ErrorFit) -> np.ndarray:
    return (
        fit.step_primary_coef * stime_feature(curve, float(STEP["response_lambda"]))
        + fit.step_curvature_coef
        * curvature_feature(curve, float(CURVATURE["curvature_lambda"]), str(CURVATURE["curvature_mode"]))
    )


def source_correction(curve: Curve, fit: ErrorFit, variant: str) -> np.ndarray:
    if variant == "smooth_only":
        return smooth_correction(curve, fit)
    if variant == "step_only":
        return step_correction(curve, fit)
    if variant == "smooth_plus_step":
        return smooth_correction(curve, fit) + step_correction(curve, fit)
    raise ValueError(f"unknown subtraction variant: {variant}")


def target_correction(curve: Curve, fit: ErrorFit) -> tuple[np.ndarray, float, str]:
    channel = channel_for_curve(curve)
    if channel == "smooth":
        corr = smooth_correction(curve, fit)
        retention = target_retention(
            stime_feature(curve, float(SMOOTH["response_lambda"])),
            nuisance_lambda=float(SMOOTH["nuisance_lambda"]),
            max_mode=int(SMOOTH["max_mode"]),
        )
    else:
        corr = step_correction(curve, fit)
        retention = (
            target_retention(
                corr,
                nuisance_lambda=float(STEP["nuisance_lambda"]),
                max_mode=int(STEP["max_mode"]),
            )
            if float(np.dot(corr, corr)) > 1e-18
            else 0.0
        )
    if retention < TARGET_RETENTION_FLOOR:
        corr = np.zeros_like(curve.loss)
    return corr, retention, channel


def adjusted_train_curves(scale: str, first_fit: ErrorFit, variant: str) -> list[Curve]:
    adjusted = []
    for name in TRAIN_CURVES:
        curve = load_curve(scale, name)
        corr = source_correction(curve, first_fit, variant)
        adjusted.append(Curve(curve.name, curve.scale, curve.step, curve.loss - corr, curve.lrs))
    return [subsample_curve(curve) for curve in adjusted]


def run_variant(scale: str, pure_params: np.ndarray, variant: str) -> tuple[np.ndarray, ErrorFit, ErrorFit, float]:
    first_fit = fit_error(scale, pure_params)
    adjusted = adjusted_train_curves(scale, first_fit, variant)
    refit_params, obj = fit_mpl_custom(adjusted, scale, pure_params)
    final_fit = fit_error(scale, refit_params)
    return refit_params, first_fit, final_fit, obj


def evaluate_variant(
    scale: str,
    pure_params: np.ndarray,
    refit_params: np.ndarray,
    first_fit: ErrorFit,
    final_fit: ErrorFit,
    variant: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for target_curve, target_label in TARGETS:
        curve = load_curve(scale, target_curve)
        pure_pred = mpl_predict(pure_params, curve)
        first_corr, first_retention, channel = target_correction(curve, first_fit)
        first_pred = pure_pred + first_corr
        refit_pred = mpl_predict(refit_params, curve)
        final_corr, final_retention, _ = target_correction(curve, final_fit)
        final_pred = refit_pred + final_corr

        pure_mae = metrics(curve.loss, pure_pred)["mae"]
        first_mae = metrics(curve.loss, first_pred)["mae"]
        refit_mae = metrics(curve.loss, refit_pred)["mae"]
        final_mae = metrics(curve.loss, final_pred)["mae"]
        rows.append(
            {
                "scale": scale,
                "variant": variant,
                "test_curve": target_curve,
                "test_label": target_label,
                "channel": channel,
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
                "first_smooth_coef": first_fit.smooth_coef,
                "first_step_primary_coef": first_fit.step_primary_coef,
                "first_step_curvature_coef": first_fit.step_curvature_coef,
                "final_smooth_coef": final_fit.smooth_coef,
                "final_step_primary_coef": final_fit.step_primary_coef,
                "final_step_curvature_coef": final_fit.step_curvature_coef,
            }
        )
    return rows


def summarize_by_variant(details: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variant in SUBTRACTION_VARIANTS:
        sub = [row for row in details if row["variant"] == variant]
        rows.append({"variant": variant, "baseline": "pure_mpl", **aggregate(sub, "delta_vs_pure_pct")})
        rows.append({"variant": variant, "baseline": "first_error", **aggregate(sub, "delta_vs_first_error_pct")})
    return rows


def summarize_targets(details: list[dict[str, object]], variant: str) -> list[dict[str, object]]:
    selected = [row for row in details if row["variant"] == variant]
    rows: list[dict[str, object]] = []
    for target_curve, target_label in TARGETS:
        sub = [row for row in selected if row["test_curve"] == target_curve]
        rows.append({"test_curve": target_curve, "test_label": target_label, **aggregate(sub, "delta_vs_pure_pct")})
    return rows


def write_report(details: list[dict[str, object]], summary: list[dict[str, object]], params_rows: list[dict[str, object]]) -> None:
    pure_rows = [row for row in summary if row["baseline"] == "pure_mpl"]
    safe_pure_rows = [
        row
        for row in pure_rows
        if int(row["wins"]) == int(row["rows"]) and int(row["nonharm"]) == int(row["rows"])
    ]
    best_pool = safe_pure_rows if safe_pure_rows else pure_rows
    best = min(best_pool, key=lambda row: (float(row["mean_delta"]), float(row["worst_delta"])))
    best_mean = min(pure_rows, key=lambda row: (float(row["mean_delta"]), float(row["worst_delta"])))
    first_rows = [row for row in summary if row["baseline"] == "first_error" and row["variant"] == best["variant"]]
    best_vs_first = first_rows[0]
    target_rows = summarize_targets(details, str(best["variant"]))
    strict_ref = read_csv(STRICT_CURVATURE_DIR / "safe_curvature_top200.csv")[0]
    lines = [
        "# Alternating MPL/Error Refit Audit\n\n",
        "This audit implements the requested alternating procedure under the strict cosine-only protocol. "
        "Pure MPL is fit on `cosine_24000.csv` and `cosine_72000.csv`; the error coefficients are fit "
        "only from `cosine_72000.csv` residuals.\n\n",
        "## Fixed Error Model Used\n\n",
        "The error model is the strict-calibrated LR-curvature correction:\n\n",
        "```text\n",
        "smooth correction = k_smooth * phi_lambda=4\n",
        "step correction   = a_step * phi_lambda=20 + b_curv * psi_diff_drop,lambda=30\n",
        "```\n\n",
        "Channel calibration:\n\n",
        "```text\n",
        "smooth: start=12000, mu=0.05, modes=8, tau=0.05, p=0.25, rho=0.2\n",
        "step:   start=12000, mu=0.02, modes=12, tau=0.05, p=0,    rho=0\n",
        "curv:   tau=0.001, shrink_curvature=true, nonnegative coefficient\n",
        "```\n\n",
        "## Best Fully Non-Harming Alternating Variant\n\n",
        f"- Subtraction variant: `{best['variant']}`.\n",
        f"- Final vs pure strict MPL: mean `{fmt_pct2(float(best['mean_delta']))}`, "
        f"worst `{fmt_pct2(float(best['worst_delta']))}`, wins `{int(best['wins'])}/{int(best['rows'])}`.\n",
        f"- Final vs first two-stage correction: mean `{fmt_pct2(float(best_vs_first['mean_delta']))}`, "
        f"worst `{fmt_pct2(float(best_vs_first['worst_delta']))}`, wins "
        f"`{int(best_vs_first['wins'])}/{int(best_vs_first['rows'])}`.\n",
        f"- Reference strict-calibrated two-stage correction: mean `{fmt_pct2(float(strict_ref['mean_delta']))}`, "
        f"worst `{fmt_pct2(float(strict_ref['worst_delta']))}`.\n\n",
        "Best mean-only variant:\n\n",
        f"- `{best_mean['variant']}` reaches mean `{fmt_pct2(float(best_mean['mean_delta']))}`, "
        f"but worst `{fmt_pct2(float(best_mean['worst_delta']))}` and wins "
        f"`{int(best_mean['wins'])}/{int(best_mean['rows'])}`.\n\n",
        "## Variant Summary\n\n",
        "| variant | baseline | mean delta | worst | wins |\n",
        "|---|---|---:|---:|---:|\n",
    ]
    for row in summary:
        lines.append(
            f"| {row['variant']} | {row['baseline']} | {fmt_pct2(float(row['mean_delta']))} | "
            f"{fmt_pct2(float(row['worst_delta']))} | {int(row['wins'])}/{int(row['rows'])} |\n"
        )
    lines += [
        "\n## Per-Target Result For Best Variant\n\n",
        "| target | mean delta vs pure MPL | worst scale | wins |\n",
        "|---|---:|---:|---:|\n",
    ]
    for row in target_rows:
        lines.append(
            f"| {row['test_label']} | {fmt_pct(float(row['mean_delta']))} | "
            f"{fmt_pct(float(row['worst_delta']))} | {int(row['wins'])}/{int(row['rows'])} |\n"
        )
    first_coef = [row for row in params_rows if row["variant"] == best["variant"] and row["stage"] == "first_error"]
    final_coef = [row for row in params_rows if row["variant"] == best["variant"] and row["stage"] == "final_error"]
    lines += [
        "\n## Coefficient Drift\n\n",
        "| scale | first smooth | final smooth | first step | final step | first curv | final curv |\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for scale in SCALES:
        first = next(row for row in first_coef if row["scale"] == scale)
        final = next(row for row in final_coef if row["scale"] == scale)
        lines.append(
            f"| {scale}M | {float(first['smooth_coef']):.5f} | {float(final['smooth_coef']):.5f} | "
            f"{float(first['step_primary_coef']):.5f} | {float(final['step_primary_coef']):.5f} | "
            f"{float(first['step_curvature_coef']):.5f} | {float(final['step_curvature_coef']):.5f} |\n"
        )
    lines += [
        "\n## Reading\n\n",
        "- The useful test is the `final vs first two-stage correction` row. If it is positive, the alternating refit made WSD transfer worse even if it still beats pure MPL.\n",
        "- `smooth_only` subtracts the correction that would actually be applied to cosine schedules. `step_only` and `smooth_plus_step` are diagnostics for whether the WSD-con channel should also be treated as part of the backbone-refit residual.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pure = load_or_fit_pure_mpl()
    details: list[dict[str, object]] = []
    params_rows: list[dict[str, object]] = []
    for variant in SUBTRACTION_VARIANTS:
        for scale in SCALES:
            refit_params, first_fit, final_fit, refit_obj = run_variant(scale, pure[scale], variant)
            params_rows.extend(
                [
                    {
                        "variant": variant,
                        "scale": scale,
                        "stage": "pure_mpl",
                        "objective": "",
                        **{f"p{i}": float(value) for i, value in enumerate(pure[scale])},
                        "smooth_coef": "",
                        "step_primary_coef": "",
                        "step_curvature_coef": "",
                    },
                    {
                        "variant": variant,
                        "scale": scale,
                        "stage": "refit_mpl",
                        "objective": refit_obj,
                        **{f"p{i}": float(value) for i, value in enumerate(refit_params)},
                        "smooth_coef": "",
                        "step_primary_coef": "",
                        "step_curvature_coef": "",
                    },
                    {
                        "variant": variant,
                        "scale": scale,
                        "stage": "first_error",
                        "objective": "",
                        **{f"p{i}": "" for i in range(7)},
                        "smooth_coef": first_fit.smooth_coef,
                        "step_primary_coef": first_fit.step_primary_coef,
                        "step_curvature_coef": first_fit.step_curvature_coef,
                    },
                    {
                        "variant": variant,
                        "scale": scale,
                        "stage": "final_error",
                        "objective": "",
                        **{f"p{i}": "" for i in range(7)},
                        "smooth_coef": final_fit.smooth_coef,
                        "step_primary_coef": final_fit.step_primary_coef,
                        "step_curvature_coef": final_fit.step_curvature_coef,
                    },
                ]
            )
            details.extend(evaluate_variant(scale, pure[scale], refit_params, first_fit, final_fit, variant))

    summary = summarize_by_variant(details)
    write_csv(OUT_DIR / "details.csv", details)
    write_csv(OUT_DIR / "summary.csv", summary)
    write_csv(OUT_DIR / "params.csv", params_rows)
    write_report(details, summary, params_rows)
    print(f"wrote {OUT_DIR / 'details.csv'}")
    print(f"wrote {OUT_DIR / 'summary.csv'}")
    print(f"wrote {OUT_DIR / 'params.csv'}")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
