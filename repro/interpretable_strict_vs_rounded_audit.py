#!/usr/bin/env python3
"""Compare exact observation-half-life endpoints with rounded fast endpoint.

The main WSD result currently uses a rounded fast endpoint (`lambda_fast=20`).
This audit checks the interpretability/performance tradeoff against the fully
derived exact endpoint (`lambda_fast=lambda_obs`) and records extra controls.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import interpretable_error_model as iem

OUT_DIR = iem.ROOT / "results" / "interpretable_strict_vs_rounded"
FIT_START = 8000
NUISANCE_LAMBDA = 0.01
DCT_MODES = iem.DCT_MODES
RIDGE_TAU = iem.RIDGE_TAU

CORE_TARGETS = iem.TARGETS
EXTRA_CONTROLS = [
    ("cosine_24000.csv", "Cosine 24k"),
    ("constant_24000.csv", "Constant 24k"),
    ("constant_72000.csv", "Constant 72k"),
]


def modal_interval(curve: iem.Curve) -> int:
    diffs = np.diff(curve.step)
    values, counts = np.unique(diffs[diffs > 0], return_counts=True)
    return int(values[int(np.argmax(counts))])


def lambda_obs(curve: iem.Curve) -> float:
    return math.log(2.0) / (iem.PEAK_LR * modal_interval(curve))


def response_lambda(curve: iem.Curve, variant: str) -> float:
    base = lambda_obs(curve)
    slow = base / iem.OBS_HALF_LIFE_MULTIPLIER
    base_variant = variant.replace("_sqrtlocalized", "").replace("_localized", "")
    if base_variant == "strict_exact":
        fast = base
    elif base_variant == "rounded_fast20":
        fast = iem.OBS_FAST_LAMBDA
    elif base_variant == "legacy_7_20":
        slow = 7.0
        fast = 20.0
    else:
        raise ValueError(f"unknown variant: {variant}")
    return slow + (fast - slow) * iem.drop_concentration(curve)


def localization_factor(curve: iem.Curve, variant: str) -> float:
    if variant.endswith("_sqrtlocalized"):
        return math.sqrt(iem.drop_localization_factor(curve))
    if variant.endswith("_localized"):
        return iem.drop_localization_factor(curve)
    return 1.0


def fit_projected(source: iem.CurvePack, lam: float) -> tuple[np.ndarray, dict[str, float]]:
    source_feature = iem.causal_drop_response(source.curve, lam)[:, None]
    return iem.fit_nonnegative_ridge(
        source.residual,
        source_feature,
        source.curve.step,
        fit_start=FIT_START,
        nuisance_lambda=NUISANCE_LAMBDA,
        max_mode=DCT_MODES,
        ridge_tau=RIDGE_TAU,
        signed=False,
    )


def load_pack(scale: str, curve_name: str) -> iem.CurvePack:
    curve = iem.load_curve(scale, curve_name)
    params = iem.MPL_PRECOMPUTED_INIT[scale]
    baseline = iem.mpl_predict(params, curve)
    residual = curve.loss - baseline
    slope_raw, slope_norm = iem.mpl_slope_features(curve, baseline)
    ld_basis, dlogc_basis = iem.mpl_sensitivity_features(params, curve)
    return iem.CurvePack(
        curve=curve,
        baseline=baseline,
        residual=residual,
        base_mae=iem.mae(curve.loss, baseline),
        slope_raw=slope_raw,
        slope_norm=slope_norm,
        ld_basis=ld_basis,
        dlogc_basis=dlogc_basis,
    )


def run_variant(variant: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    targets = [("core_wsd", *item) for item in CORE_TARGETS] + [
        ("extra_control", *item) for item in EXTRA_CONTROLS
    ]
    for scale in iem.SCALES:
        source = load_pack(scale, iem.TRAIN_CURVE)
        for group, curve_name, label in targets:
            target = load_pack(scale, curve_name)
            lam = response_lambda(target.curve, variant)
            coef, fit_info = fit_projected(source, lam)
            feature = iem.causal_drop_response(target.curve, lam)[:, None]
            localization = localization_factor(target.curve, variant)
            pred = target.baseline + localization * (feature @ coef)
            corr_mae = iem.mae(target.curve.loss, pred)
            rows.append(
                {
                    "variant": variant,
                    "group": group,
                    "scale": scale,
                    "test_curve": curve_name,
                    "test_label": label,
                    "lambda": lam,
                    "coef": float(coef[0]),
                    "localization": localization,
                    "fit_objective": float(fit_info["fit_objective"]),
                    "residualized_corr": float(fit_info["residualized_corr"]),
                    "base_mae": target.base_mae,
                    "corr_mae": corr_mae,
                    "delta_pct": 100.0 * (corr_mae / target.base_mae - 1.0),
                    "win": int(corr_mae < target.base_mae),
                }
            )
    return rows


def summarize(rows: list[dict[str, object]], variant: str, group: str) -> dict[str, object]:
    sub = [row for row in rows if row["variant"] == variant and row["group"] == group]
    deltas = np.array([float(row["delta_pct"]) for row in sub], dtype=np.float64)
    return {
        "variant": variant,
        "group": group,
        "rows": len(sub),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "worst_delta": float(np.max(deltas)),
        "wins": int(np.sum(deltas < 0.0)),
        "nonharm": int(np.sum(deltas <= 1e-12)),
    }


def write_report(
    summary_rows: list[dict[str, object]],
    detail_rows: list[dict[str, object]],
    variants: list[str],
) -> None:
    lines = [
        "# Strict vs Rounded Endpoint Audit\n\n",
        "This audit compares the fully derived observation-half-life endpoint with the rounded fast endpoint.  It keeps the same one-coefficient cosine-only projected estimator and adds constant / short-cosine controls.\n\n",
        "## Summary\n\n",
        "| variant | group | mean | worst | wins | non-harm |\n",
        "|---|---|---:|---:|---:|---:|\n",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['variant']} | {row['group']} | {float(row['mean_delta']):+.2f}% | "
            f"{float(row['worst_delta']):+.2f}% | {int(row['wins'])}/{int(row['rows'])} | "
            f"{int(row['nonharm'])}/{int(row['rows'])} |\n"
        )

    lines += [
        "\n## Extra Controls By Curve\n\n",
        "| variant | control | mean | worst | non-harm |\n",
        "|---|---|---:|---:|---:|\n",
    ]
    for variant in variants:
        for _, label in EXTRA_CONTROLS:
            sub = [
                row
                for row in detail_rows
                if row["variant"] == variant and row["test_label"] == label
            ]
            deltas = np.array([float(row["delta_pct"]) for row in sub], dtype=np.float64)
            lines.append(
                f"| {variant} | {label} | {float(np.mean(deltas)):+.2f}% | "
                f"{float(np.max(deltas)):+.2f}% | {int(np.sum(deltas <= 1e-12))}/{len(sub)} |\n"
            )

    lines += [
        "\n## Reading\n\n",
        "- The strict exact endpoint is the cleanest endpoint formula: `lambda_fast = lambda_obs`, with no rounded constant.  It keeps all WSD-family rows improving, but is more conservative than the rounded fast endpoint.\n",
        "- The rounded fast endpoint gives stronger WSD-family gains, especially on step-to-constant targets, but it should be described as a rounded observable-resolution prior rather than as a fitted parameter.\n",
        "- The localized variants multiply the correction by a continuous schedule-locality factor, not a fitted gate: full-run diffuse cosine decay receives no local transient correction, while finite WSD cooldown and single-step WSD-con remain active.  The sqrt-localized variant is the current deployable amplitude rule.\n",
        "- Constant controls are unaffected because the positive LR-drop feature is exactly zero after warmup.\n",
        "- Without localization, the short-cosine control is the main limitation.  With localization, this failure is removed at the cost of a smaller but still all-win WSD-family gain.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    variants = [
        "strict_exact",
        "strict_exact_sqrtlocalized",
        "strict_exact_localized",
        "rounded_fast20",
        "rounded_fast20_sqrtlocalized",
        "rounded_fast20_localized",
        "legacy_7_20",
    ]
    detail_rows: list[dict[str, object]] = []
    for variant in variants:
        detail_rows.extend(run_variant(variant))
    summary_rows = [
        summarize(detail_rows, variant, group)
        for variant in variants
        for group in ["core_wsd", "extra_control"]
    ]
    iem.write_csv(OUT_DIR / "details.csv", detail_rows)
    iem.write_csv(OUT_DIR / "summary.csv", summary_rows)
    write_report(summary_rows, detail_rows, variants)
    print(f"wrote {OUT_DIR / 'details.csv'}")
    print(f"wrote {OUT_DIR / 'summary.csv'}")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
