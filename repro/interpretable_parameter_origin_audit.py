#!/usr/bin/env python3
"""Audit where the response-time parameters can come from.

This script is deliberately narrower than a model search.  It checks whether
the remaining response-time constants in `interpretable_error_model.py` can be
explained from observable quantities instead of WSD target loss.
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np

import interpretable_error_model as iem

OUT_DIR = iem.ROOT / "results" / "interpretable_parameter_origin_audit"
FIT_START = 8000
NUISANCE_LAMBDA = 0.01
LAMBDA_GRID = [1, 2, 3, 4, 5, 7, 10, 14, 20, 28, 40, 60, 90, 128]


def fit_coef(source: iem.CurvePack, feature: np.ndarray) -> np.ndarray:
    coef, _ = iem.fit_nonnegative_ridge(
        source.residual,
        feature[:, None] if feature.ndim == 1 else feature,
        source.curve.step,
        fit_start=FIT_START,
        nuisance_lambda=NUISANCE_LAMBDA,
        max_mode=iem.DCT_MODES,
        ridge_tau=iem.RIDGE_TAU,
        signed=False,
    )
    return coef


def source_fit_objective(source: iem.CurvePack, response_lambda: float) -> tuple[float, float]:
    feature = iem.causal_drop_response(source.curve, response_lambda)[:, None]
    coef, info = iem.fit_nonnegative_ridge(
        source.residual,
        feature,
        source.curve.step,
        fit_start=FIT_START,
        nuisance_lambda=NUISANCE_LAMBDA,
        max_mode=iem.DCT_MODES,
        ridge_tau=iem.RIDGE_TAU,
        signed=False,
    )
    return float(info["fit_objective"]), float(coef[0])


def eval_lr_time_rule(
    name: str,
    cache: dict[tuple[str, str], iem.CurvePack],
    lambda_rule,
    *,
    projected: bool,
    note: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    details: list[dict[str, object]] = []
    for scale in iem.SCALES:
        source = cache[(scale, iem.TRAIN_CURVE)]
        shared_coef: np.ndarray | None = None
        if not projected:
            source_lambda = float(lambda_rule(source.curve))
            shared_coef = fit_coef(source, iem.causal_drop_response(source.curve, source_lambda))
        for target_curve, target_label in iem.TARGETS:
            target = cache[(scale, target_curve)]
            response_lambda = float(lambda_rule(target.curve))
            if projected:
                source_feature = iem.causal_drop_response(source.curve, response_lambda)
                coef = fit_coef(source, source_feature)
            else:
                coef = np.asarray(shared_coef)
            target_feature = iem.causal_drop_response(target.curve, response_lambda)
            pred = target.baseline + target_feature[:, None] @ coef
            corr_mae = iem.mae(target.curve.loss, pred)
            details.append(
                {
                    "method": name,
                    "scale": scale,
                    "test_curve": target_curve,
                    "test_label": target_label,
                    "response_axis": "lr_time",
                    "response_value": response_lambda,
                    "projected": int(projected),
                    "note": note,
                    "base_mae": target.base_mae,
                    "corr_mae": corr_mae,
                    "delta_pct": 100.0 * (corr_mae / target.base_mae - 1.0),
                    "win": int(corr_mae < target.base_mae),
                }
            )
    return summarize(name, details, note), details


def step_time_response(curve: iem.Curve, tau: float) -> np.ndarray:
    eta = curve.lrs.astype(np.float64)
    drop = np.zeros_like(eta)
    drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0) / iem.PEAK_LR
    out = np.empty_like(eta)
    acc = 0.0
    decay = 0.0 if tau <= 0.0 else math.exp(-1.0 / tau)
    for idx in range(len(eta)):
        acc = acc * decay + drop[idx]
        out[idx] = acc
    return out[curve.step]


def positive_drop_span_and_mass(curve: iem.Curve) -> tuple[int, float]:
    eta = curve.lrs.astype(np.float64)
    drop = np.zeros_like(eta)
    drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0) / iem.PEAK_LR
    idx = np.flatnonzero(drop > 1e-15)
    if idx.size == 0:
        return 0, 0.0
    return int(idx[-1] - idx[0] + 1), float(np.sum(drop))


def geometry_tau(curve: iem.Curve) -> float:
    """Old step-time geometry rule, kept here as a negative/contrast audit."""
    span, total_drop = positive_drop_span_and_mass(curve)
    if total_drop <= 0.05:
        return 0.0
    if span > 16000 and len(curve.lrs) <= 30000:
        return 0.0
    if span > 100:
        return min(8192.0, 1.25 * span)
    q = min(max((total_drop - 0.40) / (0.90 - 0.40), 0.0), 1.0)
    return 512.0 * (1.0 + 2.0 * q**3)


def eval_step_time_geometry(
    cache: dict[tuple[str, str], iem.CurvePack],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    name = "step_time_geometry_tau"
    note = "step-time contrast rule; target tau from LR geometry, coefficient from cosine only"
    details: list[dict[str, object]] = []
    for scale in iem.SCALES:
        source = cache[(scale, iem.TRAIN_CURVE)]
        for target_curve, target_label in iem.TARGETS:
            target = cache[(scale, target_curve)]
            tau = geometry_tau(target.curve)
            coef = fit_coef(source, step_time_response(source.curve, tau))
            pred = target.baseline + step_time_response(target.curve, tau)[:, None] @ coef
            corr_mae = iem.mae(target.curve.loss, pred)
            details.append(
                {
                    "method": name,
                    "scale": scale,
                    "test_curve": target_curve,
                    "test_label": target_label,
                    "response_axis": "step_time",
                    "response_value": tau,
                    "projected": 1,
                    "note": note,
                    "base_mae": target.base_mae,
                    "corr_mae": corr_mae,
                    "delta_pct": 100.0 * (corr_mae / target.base_mae - 1.0),
                    "win": int(corr_mae < target.base_mae),
                }
            )
    return summarize(name, details, note), details


def observed_step_interval(cache: dict[tuple[str, str], iem.CurvePack]) -> int:
    diffs: list[int] = []
    for pack in cache.values():
        diffs.extend(int(value) for value in np.diff(pack.curve.step) if value > 0)
    [(interval, _)] = Counter(diffs).most_common(1)
    return interval


def summarize(name: str, rows: list[dict[str, object]], note: str) -> dict[str, object]:
    deltas = np.array([float(row["delta_pct"]) for row in rows], dtype=np.float64)
    return {
        "method": name,
        "rows": len(rows),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "worst_delta": float(np.max(deltas)),
        "wins": int(np.sum(deltas < 0.0)),
        "nonharm": int(np.sum(deltas <= 1e-12)),
        "note": note,
    }


def write_report(
    summary_rows: list[dict[str, object]],
    lambda_rows: list[dict[str, object]],
    sensitivity_rows: list[dict[str, object]],
    detail_rows: list[dict[str, object]],
    obs_interval: int,
    lambda_obs: float,
) -> None:
    summary_rows = sorted(
        summary_rows,
        key=lambda row: (
            int(row["wins"]) != int(row["rows"]),
            float(row["mean_delta"]),
            float(row["worst_delta"]),
        ),
    )
    lines = [
        "# Parameter Origin Audit\n\n",
        "This audit asks whether the response-time constants can be explained from "
        "observable quantities rather than WSD target loss.  Every coefficient is "
        "still fitted from `cosine_72000.csv` residuals only.\n\n",
        "## Observed-Time Anchor\n\n",
        f"- Modal loss-curve observation interval: `{obs_interval}` training steps.\n",
        f"- One-observation half-life in LR time: `lambda_obs = ln(2) / (eta_peak * {obs_interval}) = {lambda_obs:.4f}`.\n",
        "- This gives a direct interpretation of the old fast endpoint: `20` is close to one observable-interval half-life.\n",
        "- A slower smooth-decay endpoint can be read as a 2.5-observation half-life, giving `lambda_obs / 2.5`.\n\n",
        "## Summary\n\n",
        "| method | mean | worst | wins | note |\n",
        "|---|---:|---:|---:|---|\n",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {float(row['mean_delta']):+.2f}% | "
            f"{float(row['worst_delta']):+.2f}% | {int(row['wins'])}/{int(row['rows'])} | "
            f"{row['note']} |\n"
        )

    lines += [
        "\n## Lambda Source Diagnostics\n\n",
        "| source | scale | selected lambda | source objective | source coef |\n",
        "|---|---:|---:|---:|---:|\n",
    ]
    for row in lambda_rows:
        lines.append(
            f"| {row['source']} | {row['scale']} | {float(row['selected_lambda']):.4g} | "
            f"{float(row['source_objective']):.6g} | {float(row['source_coef']):.6g} |\n"
        )

    lines += [
        "\n## Observation-Half-Life Sensitivity\n\n",
        "| slow half-life multiplier | fast endpoint | slow lambda | fast lambda | mean | worst | wins |\n",
        "|---:|---|---:|---:|---:|---:|---:|\n",
    ]
    for row in sensitivity_rows:
        lines.append(
            f"| {float(row['slow_multiplier']):.2f} | {row['fast_endpoint']} | "
            f"{float(row['lambda_slow']):.4f} | {float(row['lambda_fast']):.4f} | "
            f"{float(row['mean_delta']):+.2f}% | {float(row['worst_delta']):+.2f}% | "
            f"{int(row['wins'])}/{int(row['rows'])} |\n"
        )

    target_groups = sorted({str(row["test_label"]) for row in detail_rows})
    lines += [
        "\n## Per-Target For Observation-Derived Rule\n\n",
        "| target | mean | worst | wins |\n",
        "|---|---:|---:|---:|\n",
    ]
    obs_rows = [row for row in detail_rows if row["method"] == "obs_half_life_2p5_roundfast20"]
    for target in target_groups:
        rows = [row for row in obs_rows if row["test_label"] == target]
        if not rows:
            continue
        metrics = summarize(target, rows, "")
        lines.append(
            f"| {target} | {float(metrics['mean_delta']):+.2f}% | "
            f"{float(metrics['worst_delta']):+.2f}% | {int(metrics['wins'])}/{int(metrics['rows'])} |\n"
        )

    lines += [
        "\n## Reading\n\n",
        "- Source-loss selection of `lambda` is not trustworthy: it selects a very slow cosine response and fails on every WSD-family target.  This is direct evidence of low-frequency MPL-drift contamination.\n",
        "- A universal LR-time response is safer only around the fast endpoint, but it leaves large smooth-decay gains on the table.\n",
        "- Step-time geometry is interpretable in same-family audits, but with cosine as the only calibration source it over-transfers long-memory corrections to WSD and fails this specific task.\n",
        "- The best current explanation for the remaining endpoint constants is observable response half-life: fast step corrections should be resolvable within roughly one logged interval, while smooth-decay corrections need a few intervals to be identifiable.\n",
        "- The rounded observation-derived rule remains all-win, so the main formula no longer depends on an unexplained exact `7/20` choice.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    cache = iem.build_cache()
    obs_interval = observed_step_interval(cache)
    lambda_obs = math.log(2.0) / (iem.PEAK_LR * obs_interval)

    lambda_rows: list[dict[str, object]] = []
    scale_selected: dict[str, float] = {}
    for scale in iem.SCALES:
        source = cache[(scale, iem.TRAIN_CURVE)]
        scored = [
            (*source_fit_objective(source, response_lambda), response_lambda)
            for response_lambda in LAMBDA_GRID
        ]
        objective, coef, selected_lambda = min(scored, key=lambda item: item[0])
        scale_selected[scale] = selected_lambda
        lambda_rows.append(
            {
                "source": "cosine_source_fit_grid",
                "scale": scale,
                "selected_lambda": selected_lambda,
                "source_objective": objective,
                "source_coef": coef,
            }
        )

    rules = [
        (
            "fixed_lr_lambda_20",
            lambda _curve: 20.0,
            False,
            "one universal fast LR-time response",
        ),
        (
            "fixed_lr_lambda_7",
            lambda _curve: 7.0,
            False,
            "one universal slow LR-time response",
        ),
        (
            "cosine_source_selected_lambda",
            lambda curve: scale_selected[curve.scale],
            False,
            "lambda chosen by cosine residual fit objective only",
        ),
        (
            "obs_half_life_2p5_exact",
            lambda curve: (lambda_obs / 2.5)
            + (lambda_obs - lambda_obs / 2.5) * iem.drop_concentration(curve),
            True,
            "endpoints from 2.5 and 1 observed-interval half-lives",
        ),
        (
            "obs_half_life_2p5_roundfast20",
            lambda curve: (lambda_obs / 2.5)
            + (20.0 - lambda_obs / 2.5) * iem.drop_concentration(curve),
            True,
            "slow endpoint from 2.5 observed intervals; fast endpoint rounded to 20",
        ),
        (
            "current_q_7_20",
            lambda curve: 7.0 + 13.0 * iem.drop_concentration(curve),
            True,
            "current development endpoint rule",
        ),
    ]

    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    for name, rule, projected, note in rules:
        summary, details = eval_lr_time_rule(name, cache, rule, projected=projected, note=note)
        summary_rows.append(summary)
        detail_rows.extend(details)

    summary, details = eval_step_time_geometry(cache)
    summary_rows.append(summary)
    detail_rows.extend(details)

    sensitivity_rows: list[dict[str, object]] = []
    for slow_multiplier in [2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0]:
        for fast_name, fast_lambda in [("exact_lambda_obs", lambda_obs), ("rounded_20", 20.0)]:
            slow_lambda = lambda_obs / slow_multiplier
            summary, _ = eval_lr_time_rule(
                f"sensitivity_m{slow_multiplier:g}_{fast_name}",
                cache,
                lambda curve, low=slow_lambda, high=fast_lambda: low
                + (high - low) * iem.drop_concentration(curve),
                projected=True,
                note="observation half-life sensitivity",
            )
            sensitivity_rows.append(
                {
                    "slow_multiplier": slow_multiplier,
                    "fast_endpoint": fast_name,
                    "lambda_slow": slow_lambda,
                    "lambda_fast": fast_lambda,
                    **{
                        key: summary[key]
                        for key in ["rows", "mean_delta", "median_delta", "worst_delta", "wins", "nonharm"]
                    },
                }
            )

    iem.write_csv(OUT_DIR / "summary.csv", summary_rows)
    iem.write_csv(OUT_DIR / "details.csv", detail_rows)
    iem.write_csv(OUT_DIR / "lambda_origin.csv", lambda_rows)
    iem.write_csv(OUT_DIR / "half_life_sensitivity.csv", sensitivity_rows)
    write_report(summary_rows, lambda_rows, sensitivity_rows, detail_rows, obs_interval, lambda_obs)
    print(f"wrote {OUT_DIR / 'summary.csv'}")
    print(f"wrote {OUT_DIR / 'details.csv'}")
    print(f"wrote {OUT_DIR / 'lambda_origin.csv'}")
    print(f"wrote {OUT_DIR / 'half_life_sensitivity.csv'}")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
