#!/usr/bin/env python3
"""Audit nuisance spaces for the interpretable response estimator.

The current estimator removes low-frequency cosine residual drift with a soft
DCT nuisance projection.  This script tests a more mechanism-native nuisance
space: the local tangent space of the MPL predictor.  If residual drift is due
to small MPL parameter error, the response amplitude should be estimated after
projecting both the source response feature and the source residual away from
MPL tangent directions.
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

OUT_DIR = iem.ROOT / "results" / "interpretable_nuisance_origin_audit"
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

PARAM_GROUPS = {
    "constant_only": ["L0"],
    "mpl_core3": ["L0", "logA", "alpha"],
    "mpl_ld4": ["logB", "logC", "logBeta", "logGamma"],
    "mpl_all7": ["L0", "logA", "alpha", "logB", "logC", "logBeta", "logGamma"],
}


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
    if rule == "two_observation_roundfast20":
        return obs / 2.0 + (20.0 - obs / 2.0) * q
    if rule == "two_point_five_roundfast20":
        return obs / 2.5 + (20.0 - obs / 2.5) * q
    raise ValueError(f"unknown response rule: {rule}")


def orthonormal_columns(z: np.ndarray) -> np.ndarray:
    cols: list[np.ndarray] = []
    for j in range(z.shape[1]):
        col = z[:, j].astype(np.float64)
        norm = float(np.linalg.norm(col))
        if norm > 1e-12:
            cols.append(col / norm)
    if not cols:
        return np.zeros((z.shape[0], 0), dtype=np.float64)
    q, r = np.linalg.qr(np.column_stack(cols))
    keep = np.abs(np.diag(r)) > 1e-8
    return q[:, keep]


def tangent_basis(pack: iem.CurvePack, group: str) -> np.ndarray:
    params = np.array(iem.MPL_PRECOMPUTED_INIT[pack.curve.scale], dtype=np.float64)
    curve = pack.curve
    cols: list[np.ndarray] = []
    eps = 1e-4

    def add_column(name: str) -> None:
        if name == "L0":
            cols.append(np.ones_like(pack.baseline))
            return
        idx_by_name = {
            "logA": 1,
            "alpha": 2,
            "logB": 3,
            "logC": 4,
            "logBeta": 5,
            "logGamma": 6,
        }
        idx = idx_by_name[name]
        pp = params.copy()
        pm = params.copy()
        if name.startswith("log"):
            pp[idx] = params[idx] * math.exp(eps)
            pm[idx] = params[idx] * math.exp(-eps)
            denom = 2.0 * eps
        else:
            step = eps * max(abs(float(params[idx])), 1.0)
            pp[idx] = params[idx] + step
            pm[idx] = max(params[idx] - step, 1e-8)
            denom = pp[idx] - pm[idx]
        cols.append((iem.mpl_predict(pp, curve) - iem.mpl_predict(pm, curve)) / denom)

    for name in PARAM_GROUPS[group]:
        add_column(name)

    mask = curve.step >= FIT_START
    return orthonormal_columns(np.column_stack(cols)[mask])


def residualize_with_nuisance(
    source: iem.CurvePack,
    lam: float,
    nuisance: str,
    basis_cache: dict[tuple[str, str], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    mask = source.curve.step >= FIT_START
    phi = iem.causal_drop_response(source.curve, lam)[mask]
    residual = source.residual[mask]

    if nuisance == "none":
        return phi, phi, residual, 0
    if nuisance == "dct_soft":
        q = iem.dct_basis(len(residual), DCT_MODES)
        phi_o = iem.soft_residualize(phi, q, NUISANCE_LAMBDA)
        residual_o = iem.soft_residualize(residual, q, NUISANCE_LAMBDA)
        return phi, phi_o, residual_o, q.shape[1]

    key = (source.curve.scale, nuisance)
    if key not in basis_cache:
        basis_cache[key] = tangent_basis(source, nuisance)
    q = basis_cache[key]
    if q.size == 0:
        return phi, phi, residual, 0
    phi_o = phi - q @ (q.T @ phi)
    residual_o = residual - q @ (q.T @ residual)
    return phi, phi_o, residual_o, q.shape[1]


def fit_coefficient(
    source: iem.CurvePack,
    lam: float,
    nuisance: str,
    shrinkage: str,
    basis_cache: dict[tuple[str, str], np.ndarray],
) -> tuple[float, dict[str, float]]:
    phi, phi_o, residual_o, basis_dim = residualize_with_nuisance(source, lam, nuisance, basis_cache)
    dot = max(0.0, float(np.dot(phi_o, residual_o)))
    full_norm = float(np.linalg.norm(phi))
    perp_norm = float(np.linalg.norm(phi_o))
    full_energy = full_norm * full_norm
    perp_energy = perp_norm * perp_norm

    if shrinkage == "tau_free_sqrt_retention":
        denom = max(full_norm * perp_norm, 1e-18)
    elif shrinkage == "tau_free_full_energy":
        denom = max(full_energy, 1e-18)
    elif shrinkage == "ridge_tau_0p05":
        denom = perp_energy + iem.RIDGE_TAU * iem.RIDGE_TAU
    else:
        raise ValueError(f"unknown shrinkage: {shrinkage}")

    return dot / denom, {
        "basis_dim": basis_dim,
        "source_dot": dot,
        "source_full_norm": full_norm,
        "source_perp_norm": perp_norm,
        "source_retention": float(perp_energy / max(full_energy, 1e-18)),
        "denominator": float(denom),
    }


def run_variant(
    response_rule: str,
    nuisance: str,
    shrinkage: str,
    locality: str,
    cache: dict[tuple[str, str], iem.CurvePack],
    basis_cache: dict[tuple[str, str], np.ndarray],
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
            coef, fit_info = fit_coefficient(source, lam, nuisance, shrinkage, basis_cache)
            factor = iem.drop_localization_factor(target.curve) if locality == "linear" else 1.0
            pred = target.baseline + factor * coef * iem.causal_drop_response(target.curve, lam)
            corr_mae = iem.mae(target.curve.loss, pred)
            rows.append(
                {
                    "response_rule": response_rule,
                    "nuisance": nuisance,
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
            (
                str(row["response_rule"]),
                str(row["nuisance"]),
                str(row["shrinkage"]),
                str(row["locality"]),
                str(row["group"]),
            )
            for row in rows
        }
    )
    for response_rule, nuisance, shrinkage, locality, group in keys:
        sub = [
            row
            for row in rows
            if row["response_rule"] == response_rule
            and row["nuisance"] == nuisance
            and row["shrinkage"] == shrinkage
            and row["locality"] == locality
            and row["group"] == group
        ]
        deltas = np.array([float(row["delta_pct"]) for row in sub], dtype=np.float64)
        summary.append(
            {
                "response_rule": response_rule,
                "nuisance": nuisance,
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


def find(summary_rows: list[dict[str, object]], response_rule: str, nuisance: str, shrinkage: str, locality: str, group: str) -> dict[str, object]:
    for row in summary_rows:
        if (
            row["response_rule"] == response_rule
            and row["nuisance"] == nuisance
            and row["shrinkage"] == shrinkage
            and row["locality"] == locality
            and row["group"] == group
        ):
            return row
    raise KeyError((response_rule, nuisance, shrinkage, locality, group))


def write_report(summary_rows: list[dict[str, object]], detail_rows: list[dict[str, object]]) -> None:
    highlights = [
        ("fixed_lambda_20", "none", "tau_free_sqrt_retention", "linear", "no-nuisance raw projection"),
        ("fixed_lambda_20", "dct_soft", "tau_free_sqrt_retention", "linear", "DCT tau-free baseline"),
        ("fixed_lambda_20", "mpl_all7", "tau_free_sqrt_retention", "linear", "MPL-all tangent lower bound"),
        ("two_point_five_roundfast20", "mpl_ld4", "ridge_tau_0p05", "linear", "MPL-LD tangent main candidate"),
        ("two_point_five_roundfast20", "dct_soft", "ridge_tau_0p05", "linear", "DCT performance reference"),
        ("two_point_five_roundfast20", "mpl_core3", "ridge_tau_0p05", "linear", "MPL-core negative evidence"),
    ]
    lines = [
        "# Nuisance-Origin Audit\n\n",
        "This audit compares the current soft DCT nuisance projection with exact projections onto MPL tangent spaces.  All variants keep the one-response formula, fit only one nonnegative coefficient from `cosine_72000.csv`, and use WSD/control losses only for evaluation.\n\n",
        "## Nuisance Spaces\n\n",
        "- `none`: no nuisance removal; this is the raw one-dimensional projection and should fail if cosine residual is contaminated by MPL drift.\n",
        "- `dct_soft`: current low-frequency residualizer, with `8` DCT modes and `mu=0.01`.\n",
        "- `mpl_core3`: tangent space of MPL backbone parameters \\(L_0,A,\\alpha\\).\n",
        "- `mpl_ld4`: tangent space of MPL LR-dependent parameters \\(B,C,\\beta,\\gamma\\).\n",
        "- `mpl_all7`: all local MPL parameter directions.\n\n",
        "The tangent variants remove residual directions that could be explained by local MPL parameter error, which is more mechanism-native than generic low-frequency filtering.\n\n",
        "## Highlight Results\n\n",
        "| role | response | nuisance | shrinkage | group | mean | worst | wins/non-harm |\n",
        "|---|---|---|---|---|---:|---:|---:|\n",
    ]
    for response_rule, nuisance, shrinkage, locality, role in highlights:
        for group in ["core_wsd", "extra_control"]:
            row = find(summary_rows, response_rule, nuisance, shrinkage, locality, group)
            lines.append(
                f"| {role} | {response_rule} | {nuisance} | {shrinkage} | {group} | "
                f"{float(row['mean_delta']):+.2f}% | {float(row['worst_delta']):+.2f}% | "
                f"{int(row['wins'])}/{int(row['rows'])} wins, {int(row['nonharm'])}/{int(row['rows'])} non-harm |\n"
            )

    lines += [
        "\n## Core WSD Summary\n\n",
        "| response | nuisance | shrinkage | locality | mean | worst | wins |\n",
        "|---|---|---|---|---:|---:|---:|\n",
    ]
    for row in sorted(
        [row for row in summary_rows if row["group"] == "core_wsd"],
        key=lambda item: (
            str(item["response_rule"]),
            str(item["shrinkage"]),
            float(item["mean_delta"]),
        ),
    ):
        lines.append(
            f"| {row['response_rule']} | {row['nuisance']} | {row['shrinkage']} | {row['locality']} | "
            f"{float(row['mean_delta']):+.2f}% | {float(row['worst_delta']):+.2f}% | "
            f"{int(row['wins'])}/{int(row['rows'])} |\n"
        )

    best_tangent = min(
        [
            row
            for row in summary_rows
            if row["group"] == "core_wsd"
            and str(row["nuisance"]).startswith("mpl_")
            and row["locality"] == "linear"
        ],
        key=lambda row: (int(row["wins"]) != int(row["rows"]), float(row["mean_delta"]), float(row["worst_delta"])),
    )
    lines += [
        "\n## Reading\n\n",
        f"- Best tangent nuisance row: `{best_tangent['response_rule']} / {best_tangent['nuisance']} / {best_tangent['shrinkage']}`, mean `{float(best_tangent['mean_delta']):+.2f}%`, worst `{float(best_tangent['worst_delta']):+.2f}%`, wins `{int(best_tangent['wins'])}/{int(best_tangent['rows'])}`.\n",
        "- The no-nuisance row is intentionally included as a failure mode: raw projection lets smooth MPL residual drift masquerade as the LR-drop response.\n",
        "- The MPL-LD tangent nuisance (`mpl_ld4`) is now the mechanism-native main candidate.  It removes only local MPL LR-term error directions before estimating the response amplitude.\n",
        "- DCT remains numerically stronger, but its generic low-frequency basis is less defensible as a core theory term.  It should be treated as a performance reference or diagnostic upper bound.\n",
        "- `mpl_core3` fails, which is useful negative evidence: the removable drift is not just an error in the backbone trend \\((L_0,A,\\alpha)\\).\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    response_rules = ["fixed_lambda_20", "two_point_five_roundfast20"]
    nuisances = ["none", "dct_soft", "constant_only", "mpl_core3", "mpl_ld4", "mpl_all7"]
    shrinkages = ["tau_free_sqrt_retention", "ridge_tau_0p05"]
    localities = ["none", "linear"]
    detail_rows: list[dict[str, object]] = []
    cache: dict[tuple[str, str], iem.CurvePack] = {}
    basis_cache: dict[tuple[str, str], np.ndarray] = {}
    for response_rule in response_rules:
        for nuisance in nuisances:
            for shrinkage in shrinkages:
                for locality in localities:
                    detail_rows.extend(
                        run_variant(
                            response_rule,
                            nuisance,
                            shrinkage,
                            locality,
                            cache,
                            basis_cache,
                        )
                    )
    summary_rows = summarize(detail_rows)
    iem.write_csv(OUT_DIR / "details.csv", detail_rows)
    iem.write_csv(OUT_DIR / "summary.csv", summary_rows)
    write_report(summary_rows, detail_rows)
    print(f"wrote {OUT_DIR / 'details.csv'}")
    print(f"wrote {OUT_DIR / 'summary.csv'}")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
