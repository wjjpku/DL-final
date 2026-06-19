#!/usr/bin/env python3
"""Sensitivity audit for the schedule-locality factor.

The deployable formula uses a parameter-free linear factor

    a_s = max(0, 1 - drop_support_span / post_warmup_span).

This script checks whether the conclusion depends on this exact linear shape.
It evaluates a small fixed set of powers for the same locality ratio, without
fitting any extra parameters.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

import interpretable_error_model as iem

OUT_DIR = iem.ROOT / "results" / "interpretable_localization_sensitivity"
FIT_START = 8000
NUISANCE_LAMBDA = 0.01
RIDGE_TAU = 0.05
POWERS = [0.0, 0.5, 1.0, 2.0, 3.0]

CORE_TARGETS = iem.TARGETS
EXTRA_CONTROLS = [
    ("cosine_24000.csv", "Cosine 24k"),
    ("constant_24000.csv", "Constant 24k"),
    ("constant_72000.csv", "Constant 72k"),
]


@lru_cache(maxsize=None)
def pack(scale: str, curve_name: str) -> iem.CurvePack:
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


def base_locality(curve: iem.Curve) -> float:
    return iem.drop_localization_factor(curve)


def locality(curve: iem.Curve, power: float) -> float:
    if power == 0.0:
        return 1.0
    return base_locality(curve) ** power


def response_lambda(curve: iem.Curve) -> float:
    candidate = iem.Candidate(
        "obs_half_life_projected_2p5_roundfast20",
        "adaptive_observed_projected_raw",
        (iem.OBS_HALF_LIFE_MULTIPLIER, iem.OBS_FAST_LAMBDA),
    )
    return iem.candidate_response_lambda(curve, candidate)


def fit_coef(source: iem.CurvePack, lam: float) -> np.ndarray:
    feature = iem.causal_drop_response(source.curve, lam)[:, None]
    coef, _ = iem.fit_nonnegative_ridge(
        source.residual,
        feature,
        source.curve.step,
        fit_start=FIT_START,
        nuisance_lambda=NUISANCE_LAMBDA,
        max_mode=iem.DCT_MODES,
        ridge_tau=RIDGE_TAU,
        signed=False,
    )
    return coef


def run_power(power: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    targets = [("core_wsd", *item) for item in CORE_TARGETS] + [
        ("extra_control", *item) for item in EXTRA_CONTROLS
    ]
    for scale in iem.SCALES:
        source = pack(scale, iem.TRAIN_CURVE)
        for group, curve_name, label in targets:
            target = pack(scale, curve_name)
            lam = response_lambda(target.curve)
            coef = fit_coef(source, lam)
            factor = locality(target.curve, power)
            feature = iem.causal_drop_response(target.curve, lam)[:, None]
            pred = target.baseline + factor * (feature @ coef)
            corr_mae = iem.mae(target.curve.loss, pred)
            rows.append(
                {
                    "power": power,
                    "group": group,
                    "scale": scale,
                    "test_curve": curve_name,
                    "test_label": label,
                    "base_locality": base_locality(target.curve),
                    "locality": factor,
                    "lambda": lam,
                    "coef": float(coef[0]),
                    "base_mae": target.base_mae,
                    "corr_mae": corr_mae,
                    "delta_pct": 100.0 * (corr_mae / target.base_mae - 1.0),
                    "win": int(corr_mae < target.base_mae),
                }
            )
    return rows


def summarize(rows: list[dict[str, object]], power: float, group: str) -> dict[str, object]:
    sub = [row for row in rows if float(row["power"]) == power and row["group"] == group]
    deltas = np.array([float(row["delta_pct"]) for row in sub], dtype=np.float64)
    return {
        "power": power,
        "group": group,
        "rows": len(sub),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "worst_delta": float(np.max(deltas)),
        "wins": int(np.sum(deltas < 0.0)),
        "nonharm": int(np.sum(deltas <= 1e-12)),
    }


def write_report(summary_rows: list[dict[str, object]], detail_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Localization Sensitivity Audit\n\n",
        "This audit keeps the observation-half-life response and cosine-only coefficient fixed, and varies only the shape of the schedule-locality factor.  `power=0` is the unlocalized upper variant; `power=1` is the current deployable linear rule.\n\n",
        "## Summary\n\n",
        "| power | interpretation | group | mean | worst | wins | non-harm |\n",
        "|---:|---|---|---:|---:|---:|---:|\n",
    ]
    labels = {
        0.0: "no localization",
        0.5: "milder localization",
        1.0: "linear default",
        2.0: "strong localization",
        3.0: "very strong localization",
    }
    for row in summary_rows:
        lines.append(
            f"| {float(row['power']):.1f} | {labels[float(row['power'])]} | {row['group']} | "
            f"{float(row['mean_delta']):+.2f}% | {float(row['worst_delta']):+.2f}% | "
            f"{int(row['wins'])}/{int(row['rows'])} | {int(row['nonharm'])}/{int(row['rows'])} |\n"
        )

    lines += [
        "\n## Locality Values\n\n",
        "| curve | base locality |\n",
        "|---|---:|\n",
    ]
    seen = set()
    for row in detail_rows:
        key = str(row["test_label"])
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"| {key} | {float(row['base_locality']):.4f} |\n")

    lines += [
        "\n## Reading\n\n",
        "- Any positive localization power removes the short-cosine failure while keeping constant schedules unchanged.\n",
        "- Stronger powers are safer but increasingly conservative on WSD sharp/linear, because their finite cooldown occupies about 18% of the post-warmup horizon.\n",
        "- The linear default is the least additional structure that removes the control failure while preserving all WSD-family wins and most of the WSD-only gain.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    detail_rows: list[dict[str, object]] = []
    for power in POWERS:
        detail_rows.extend(run_power(power))
    summary_rows = [
        summarize(detail_rows, power, group)
        for power in POWERS
        for group in ["core_wsd", "extra_control"]
    ]
    iem.write_csv(OUT_DIR / "details.csv", detail_rows)
    iem.write_csv(OUT_DIR / "summary.csv", summary_rows)
    write_report(summary_rows, detail_rows)
    print(f"wrote {OUT_DIR / 'details.csv'}")
    print(f"wrote {OUT_DIR / 'summary.csv'}")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
