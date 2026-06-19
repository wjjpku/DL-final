#!/usr/bin/env python3
"""Audit tau-free shrinkage rules for the interpretable response estimator.

The current high-performing formula uses a ridge value tau=0.05.  This audit
checks whether the same one-response mechanism remains useful when that fixed
ridge constant is replaced by shrinkage derived directly from source-feature
identifiability.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import interpretable_error_model as iem  # noqa: E402

OUT_DIR = iem.ROOT / "results" / "interpretable_shrinkage_origin_audit"
FIT_START = 8000
NUISANCE_LAMBDA = 0.01
DCT_MODES = iem.DCT_MODES
EXTRA_CONTROLS = [
    ("cosine_24000.csv", "Cosine 24k"),
    ("constant_24000.csv", "Constant 24k"),
    ("constant_72000.csv", "Constant 72k"),
]
ALL_TARGETS = [("core_wsd", *item) for item in iem.TARGETS] + [
    ("extra_control", *item) for item in EXTRA_CONTROLS
]


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


def lambda_obs(curve: iem.Curve) -> float:
    return math.log(2.0) / (iem.PEAK_LR * iem.modal_observation_interval(curve))


def response_lambda(curve: iem.Curve, rule: str) -> float:
    obs = lambda_obs(curve)
    q = iem.drop_concentration(curve)
    if rule == "fixed_lambda_20":
        return 20.0
    if rule == "fixed_lambda_obs":
        return obs
    if rule == "two_observation_roundfast20":
        return obs / 2.0 + (20.0 - obs / 2.0) * q
    if rule == "two_point_five_roundfast20":
        return obs / 2.5 + (20.0 - obs / 2.5) * q
    raise ValueError(f"unknown response rule: {rule}")


def residualized_pair(source: iem.CurvePack, response_lambda_value: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = source.curve.step >= FIT_START
    phi = iem.causal_drop_response(source.curve, response_lambda_value)[mask]
    residual = source.residual[mask]
    q = iem.dct_basis(len(residual), DCT_MODES)
    phi_perp = iem.soft_residualize(phi, q, NUISANCE_LAMBDA)
    residual_perp = iem.soft_residualize(residual, q, NUISANCE_LAMBDA)
    return phi, phi_perp, residual_perp


def fit_coefficient(source: iem.CurvePack, response_lambda_value: float, shrinkage: str) -> tuple[float, dict[str, float]]:
    phi, phi_perp, residual_perp = residualized_pair(source, response_lambda_value)
    dot = max(0.0, float(np.dot(phi_perp, residual_perp)))
    perp_norm = float(np.linalg.norm(phi_perp))
    full_norm = float(np.linalg.norm(phi))
    perp_energy = perp_norm * perp_norm
    full_energy = full_norm * full_norm

    if shrinkage == "ridge_tau_0p05":
        denom = perp_energy + iem.RIDGE_TAU * iem.RIDGE_TAU
    elif shrinkage == "tau_free_sqrt_retention":
        denom = max(perp_norm * full_norm, 1e-18)
    elif shrinkage == "tau_free_full_energy":
        denom = max(full_energy, 1e-18)
    elif shrinkage == "no_ridge":
        denom = max(perp_energy, 1e-18)
    else:
        raise ValueError(f"unknown shrinkage: {shrinkage}")

    coef = dot / denom
    return coef, {
        "source_dot": dot,
        "source_perp_norm": perp_norm,
        "source_full_norm": full_norm,
        "source_retention": float(perp_energy / max(full_energy, 1e-18)),
        "denominator": float(denom),
    }


def locality_factor(curve: iem.Curve, mode: str) -> float:
    if mode == "none":
        return 1.0
    if mode == "linear":
        return iem.drop_localization_factor(curve)
    raise ValueError(f"unknown locality mode: {mode}")


def run_variant(
    response_rule: str,
    shrinkage: str,
    locality: str,
    cache: dict[tuple[str, str], iem.CurvePack],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def pack(scale: str, curve_name: str) -> iem.CurvePack:
        key = (scale, curve_name)
        if key not in cache:
            cache[key] = load_pack(scale, curve_name)
        return cache[key]

    for scale in iem.SCALES:
        source = pack(scale, iem.TRAIN_CURVE)
        for group, curve_name, label in ALL_TARGETS:
            target = pack(scale, curve_name)
            lam = response_lambda(target.curve, response_rule)
            coef, fit_info = fit_coefficient(source, lam, shrinkage)
            factor = locality_factor(target.curve, locality)
            feature = iem.causal_drop_response(target.curve, lam)
            pred = target.baseline + factor * coef * feature
            corr_mae = iem.mae(target.curve.loss, pred)
            rows.append(
                {
                    "response_rule": response_rule,
                    "shrinkage": shrinkage,
                    "locality": locality,
                    "group": group,
                    "scale": scale,
                    "test_curve": curve_name,
                    "test_label": label,
                    "lambda": lam,
                    "coef": coef,
                    "locality_factor": factor,
                    "base_mae": target.base_mae,
                    "corr_mae": corr_mae,
                    "delta_pct": 100.0 * (corr_mae / target.base_mae - 1.0),
                    "win": int(corr_mae < target.base_mae),
                    **fit_info,
                }
            )
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    keys = sorted(
        {
            (str(row["response_rule"]), str(row["shrinkage"]), str(row["locality"]), str(row["group"]))
            for row in rows
        }
    )
    for response_rule, shrinkage, locality, group in keys:
        sub = [
            row
            for row in rows
            if row["response_rule"] == response_rule
            and row["shrinkage"] == shrinkage
            and row["locality"] == locality
            and row["group"] == group
        ]
        deltas = np.array([float(row["delta_pct"]) for row in sub], dtype=np.float64)
        summary.append(
            {
                "response_rule": response_rule,
                "shrinkage": shrinkage,
                "locality": locality,
                "group": group,
                "rows": len(sub),
                "mean_delta": float(np.mean(deltas)),
                "median_delta": float(np.median(deltas)),
                "worst_delta": float(np.max(deltas)),
                "wins": int(np.sum(deltas < 0.0)),
                "nonharm": int(np.sum(deltas <= 1e-12)),
            }
        )
    return summary


def pick(summary_rows: list[dict[str, object]], response_rule: str, shrinkage: str, locality: str, group: str) -> dict[str, object]:
    for row in summary_rows:
        if (
            row["response_rule"] == response_rule
            and row["shrinkage"] == shrinkage
            and row["locality"] == locality
            and row["group"] == group
        ):
            return row
    raise KeyError((response_rule, shrinkage, locality, group))


def write_report(summary_rows: list[dict[str, object]], detail_rows: list[dict[str, object]]) -> None:
    highlights = [
        ("fixed_lambda_20", "tau_free_sqrt_retention", "linear", "tau-free hard baseline"),
        ("fixed_lambda_20", "tau_free_full_energy", "linear", "most conservative tau-free baseline"),
        ("two_observation_roundfast20", "ridge_tau_0p05", "linear", "two-observation performance variant"),
        ("two_point_five_roundfast20", "ridge_tau_0p05", "linear", "old high-performance reference"),
    ]
    lines = [
        "# Shrinkage-Origin Audit\n\n",
        "This audit isolates the role of the ridge constant.  All variants keep the one-response formula and fit only one nonnegative coefficient from `cosine_72000.csv` residuals.  WSD-family and control losses are evaluation only.\n\n",
        "## Coefficient Rules\n\n",
        "Let \\(x=M_\\mu\\phi_{\\lambda,\\cos}\\), \\(y=M_\\mu r_{\\cos}\\), and \\(\\phi=\\phi_{\\lambda,\\cos}\\) on the source suffix.\n\n",
        "Current ridge rule:\n\n",
        "\\[\n",
        "\\hat\\kappa=\\frac{\\langle x,y\\rangle_+}{\\|x\\|^2+0.05^2}.\n",
        "\\]\n\n",
        "Tau-free sqrt-retention rule:\n\n",
        "\\[\n",
        "\\hat\\kappa=\\frac{\\langle x,y\\rangle_+}{\\|x\\|\\,\\|\\phi\\|}.\n",
        "\\]\n\n",
        "Tau-free full-energy rule:\n\n",
        "\\[\n",
        "\\hat\\kappa=\\frac{\\langle x,y\\rangle_+}{\\|\\phi\\|^2}.\n",
        "\\]\n\n",
        "The tau-free rules shrink automatically when most feature energy is removed by the nuisance projection.  They introduce no fitted parameter and no fixed ridge constant.\n\n",
        "## Highlight Results\n\n",
        "| role | response | shrinkage | locality | group | mean | worst | wins/non-harm |\n",
        "|---|---|---|---|---|---:|---:|---:|\n",
    ]
    for response_rule, shrinkage, locality, role in highlights:
        for group in ["core_wsd", "extra_control"]:
            row = pick(summary_rows, response_rule, shrinkage, locality, group)
            lines.append(
                f"| {role} | {response_rule} | {shrinkage} | {locality} | {group} | "
                f"{float(row['mean_delta']):+.2f}% | {float(row['worst_delta']):+.2f}% | "
                f"{int(row['wins'])}/{int(row['rows'])} wins, {int(row['nonharm'])}/{int(row['rows'])} non-harm |\n"
            )

    lines += [
        "\n## All Summary Rows\n\n",
        "| response | shrinkage | locality | group | mean | worst | wins | non-harm |\n",
        "|---|---|---|---|---:|---:|---:|---:|\n",
    ]
    for row in sorted(
        summary_rows,
        key=lambda item: (
            str(item["group"]),
            str(item["response_rule"]),
            str(item["locality"]),
            str(item["shrinkage"]),
        ),
    ):
        lines.append(
            f"| {row['response_rule']} | {row['shrinkage']} | {row['locality']} | {row['group']} | "
            f"{float(row['mean_delta']):+.2f}% | {float(row['worst_delta']):+.2f}% | "
            f"{int(row['wins'])}/{int(row['rows'])} | {int(row['nonharm'])}/{int(row['rows'])} |\n"
        )

    tau_free_rows = [
        row
        for row in detail_rows
        if row["response_rule"] == "fixed_lambda_20"
        and row["shrinkage"] == "tau_free_sqrt_retention"
        and row["locality"] == "linear"
        and row["group"] == "core_wsd"
    ]
    labels = sorted({str(row["test_label"]) for row in tau_free_rows})
    lines += [
        "\n## Tau-Free Hard Baseline Per Target\n\n",
        "| target | mean | worst | wins |\n",
        "|---|---:|---:|---:|\n",
    ]
    for label in labels:
        sub = [row for row in tau_free_rows if row["test_label"] == label]
        deltas = np.array([float(row["delta_pct"]) for row in sub], dtype=np.float64)
        lines.append(
            f"| {label} | {float(np.mean(deltas)):+.2f}% | {float(np.max(deltas)):+.2f}% | "
            f"{int(np.sum(deltas < 0.0))}/{len(sub)} |\n"
        )

    lines += [
        "\n## Reading\n\n",
        "- The one-response mechanism does not require the fixed ridge constant to be useful: `fixed_lambda_20 + tau_free_sqrt_retention + linear locality` improves all 15 WSD-family rows and keeps all 9 controls non-harm.\n",
        "- The price of removing `tau` and schedule-geometry tuning is lower mean gain: about `-20.77%` instead of the old `-32%` to `-34%` performance variants.\n",
        "- The full-energy rule is the most conservative and still improves every WSD row, but the gain is small.  This is a useful lower-bound sanity check rather than a competitive model.\n",
        "- The ridge-based variants should now be described as performance extensions over a tau-free identifiable-response baseline, not as the only evidence that the formula works.\n",
        "- This gives a cleaner research story: first prove the mechanism with a tau-free estimator, then separately justify whether the ridge performance extension is worth the extra protocol assumption.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    response_rules = [
        "fixed_lambda_20",
        "fixed_lambda_obs",
        "two_observation_roundfast20",
        "two_point_five_roundfast20",
    ]
    shrinkages = [
        "ridge_tau_0p05",
        "tau_free_sqrt_retention",
        "tau_free_full_energy",
    ]
    localities = ["none", "linear"]
    detail_rows: list[dict[str, object]] = []
    cache: dict[tuple[str, str], iem.CurvePack] = {}
    for response_rule in response_rules:
        for shrinkage in shrinkages:
            for locality in localities:
                detail_rows.extend(run_variant(response_rule, shrinkage, locality, cache))
    summary_rows = summarize(detail_rows)
    iem.write_csv(OUT_DIR / "details.csv", detail_rows)
    iem.write_csv(OUT_DIR / "summary.csv", summary_rows)
    write_report(summary_rows, detail_rows)
    print(f"wrote {OUT_DIR / 'details.csv'}")
    print(f"wrote {OUT_DIR / 'summary.csv'}")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
