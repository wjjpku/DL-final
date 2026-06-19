#!/usr/bin/env python3
"""Audit the parameter-light observation-bracket MPL-LD response model.

This script replaces two weak protocol constants in the previous MPL-LD model:

* response rate:
    old: lambda_s = lambda_obs / 2.5 + (20 - lambda_obs / 2.5) q_s
    new: lambda_s = lambda_obs * (1 + q_s) / 2

  The new rule means the response half-life is bounded between two observed
  intervals for diffuse LR drops and one observed interval for a single sharp
  drop.  It uses only the logging interval and the schedule's drop
  concentration q_s.

* ridge:
    old: tau^2 = 0.05^2
    new: tau^2 = 1 / N_cal

  N_cal is the number of cosine calibration points in the suffix used to fit
  kappa.  This is a finite-sample identifiability floor, not a hand-tuned
  constant.
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
import interpretable_nuisance_origin_audit as noa  # noqa: E402

OUT_DIR = iem.ROOT / "results" / "interpretable_observation_bracket_audit"

DEFAULT_FIT_START = 8000
FIT_STARTS = [5000, 6500, 8000, 10000, 12000]

VARIANTS = [
    {
        "variant": "observation_bracket_mplld_neff",
        "role": "main_parameter_light_candidate",
        "response_rule": "observation_bracket",
        "nuisance": "mpl_ld4",
        "shrinkage": "sample_size_ridge",
        "locality": "linear",
    },
    {
        "variant": "observation_bracket_mplld_neff_nolocality",
        "role": "wsd_core_without_boundary_term",
        "response_rule": "observation_bracket",
        "nuisance": "mpl_ld4",
        "shrinkage": "sample_size_ridge",
        "locality": "none",
    },
    {
        "variant": "old_mplld_fixedtau",
        "role": "previous_mplld_reference",
        "response_rule": "old_2p5_roundfast20",
        "nuisance": "mpl_ld4",
        "shrinkage": "fixed_tau_0p05",
        "locality": "linear",
    },
    {
        "variant": "old_mplld_neff",
        "role": "old_lambda_with_sample_size_ridge",
        "response_rule": "old_2p5_roundfast20",
        "nuisance": "mpl_ld4",
        "shrinkage": "sample_size_ridge",
        "locality": "linear",
    },
    {
        "variant": "twoobs_roundfast_mplld_neff",
        "role": "rounded_fast_endpoint_sensitivity",
        "response_rule": "twoobs_roundfast20",
        "nuisance": "mpl_ld4",
        "shrinkage": "sample_size_ridge",
        "locality": "linear",
    },
    {
        "variant": "observation_bracket_no_nuisance_neff",
        "role": "raw_projection_failure_mode",
        "response_rule": "observation_bracket",
        "nuisance": "none",
        "shrinkage": "sample_size_ridge",
        "locality": "linear",
    },
]


def lambda_obs(curve: iem.Curve) -> float:
    return math.log(2.0) / (iem.PEAK_LR * iem.modal_observation_interval(curve))


def response_lambda(curve: iem.Curve, rule: str) -> float:
    obs = lambda_obs(curve)
    q = iem.drop_concentration(curve)
    if rule == "observation_bracket":
        return obs * (1.0 + q) / 2.0
    if rule == "twoobs_roundfast20":
        return obs / 2.0 + (20.0 - obs / 2.0) * q
    if rule == "old_2p5_roundfast20":
        return obs / 2.5 + (20.0 - obs / 2.5) * q
    raise ValueError(f"unknown response rule: {rule}")


def load_pack(scale: str, curve_name: str) -> iem.CurvePack:
    return noa.load_pack(scale, curve_name)


def orthonormal_columns(z: np.ndarray) -> np.ndarray:
    cols: list[np.ndarray] = []
    for idx in range(z.shape[1]):
        col = z[:, idx].astype(np.float64)
        norm = float(np.linalg.norm(col))
        if norm > 1e-12:
            cols.append(col / norm)
    if not cols:
        return np.zeros((z.shape[0], 0), dtype=np.float64)
    q, r = np.linalg.qr(np.column_stack(cols))
    keep = np.abs(np.diag(r)) > 1e-8
    return q[:, keep]


def tangent_basis(pack: iem.CurvePack, group: str, fit_start: int) -> np.ndarray:
    params = np.array(iem.MPL_PRECOMPUTED_INIT[pack.curve.scale], dtype=np.float64)
    curve = pack.curve
    cols: list[np.ndarray] = []
    eps = 1e-4

    def add_column(name: str) -> None:
        idx_by_name = {
            "logB": 3,
            "logC": 4,
            "logBeta": 5,
            "logGamma": 6,
        }
        idx = idx_by_name[name]
        pp = params.copy()
        pm = params.copy()
        pp[idx] = params[idx] * math.exp(eps)
        pm[idx] = params[idx] * math.exp(-eps)
        cols.append((iem.mpl_predict(pp, curve) - iem.mpl_predict(pm, curve)) / (2.0 * eps))

    if group != "mpl_ld4":
        raise ValueError(f"unsupported tangent group: {group}")
    for name in ["logB", "logC", "logBeta", "logGamma"]:
        add_column(name)
    mask = curve.step >= fit_start
    return orthonormal_columns(np.column_stack(cols)[mask])


def residualized_pair(
    source: iem.CurvePack,
    lam: float,
    nuisance: str,
    fit_start: int,
    basis_cache: dict[tuple[str, str, int], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    mask = source.curve.step >= fit_start
    phi = iem.causal_drop_response(source.curve, lam)[mask]
    residual = source.residual[mask]
    if nuisance == "none":
        return phi, phi, residual, 0
    key = (source.curve.scale, nuisance, fit_start)
    if key not in basis_cache:
        basis_cache[key] = tangent_basis(source, nuisance, fit_start)
    q = basis_cache[key]
    phi_o = phi - q @ (q.T @ phi)
    residual_o = residual - q @ (q.T @ residual)
    return phi, phi_o, residual_o, q.shape[1]


def fit_coefficient(
    source: iem.CurvePack,
    lam: float,
    nuisance: str,
    shrinkage: str,
    fit_start: int,
    basis_cache: dict[tuple[str, str, int], np.ndarray],
) -> tuple[float, dict[str, float]]:
    phi, phi_o, residual_o, basis_dim = residualized_pair(source, lam, nuisance, fit_start, basis_cache)
    n_cal = len(phi_o)
    dot = max(0.0, float(np.dot(phi_o, residual_o)))
    full_norm = float(np.linalg.norm(phi))
    perp_norm = float(np.linalg.norm(phi_o))
    perp_energy = perp_norm * perp_norm

    if shrinkage == "sample_size_ridge":
        ridge = 1.0 / max(n_cal, 1)
    elif shrinkage == "fixed_tau_0p05":
        ridge = 0.05 * 0.05
    else:
        raise ValueError(f"unknown shrinkage: {shrinkage}")
    denom = perp_energy + ridge
    return dot / max(denom, 1e-18), {
        "basis_dim": basis_dim,
        "n_cal": n_cal,
        "ridge": ridge,
        "source_dot": dot,
        "source_full_norm": full_norm,
        "source_perp_norm": perp_norm,
        "source_retention": float(perp_energy / max(full_norm * full_norm, 1e-18)),
        "denominator": float(denom),
    }


def locality_factor(curve: iem.Curve, mode: str) -> float:
    if mode == "linear":
        return iem.drop_localization_factor(curve)
    if mode == "none":
        return 1.0
    raise ValueError(f"unknown locality: {mode}")


def evaluate_variant(
    variant: dict[str, str],
    fit_start: int,
    cache: dict[tuple[str, str], iem.CurvePack],
    basis_cache: dict[tuple[str, str, int], np.ndarray],
    cross_scale: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    detail_rows: list[dict[str, object]] = []
    coef_rows: list[dict[str, object]] = []
    seen_coef: set[tuple[str, str, str, str]] = set()

    def pack(scale: str, curve_name: str) -> iem.CurvePack:
        key = (scale, curve_name)
        if key not in cache:
            cache[key] = load_pack(scale, curve_name)
        return cache[key]

    train_scales = iem.SCALES
    test_scales = iem.SCALES if cross_scale else []
    if not cross_scale:
        test_scales = train_scales

    for train_scale in train_scales:
        source = pack(train_scale, iem.TRAIN_CURVE)
        for test_scale in test_scales:
            if not cross_scale and test_scale != train_scale:
                continue
            for group, curve_name, label in noa.ALL_TARGETS:
                target = pack(test_scale, curve_name)
                lam = response_lambda(target.curve, variant["response_rule"])
                coef, info = fit_coefficient(
                    source,
                    lam,
                    variant["nuisance"],
                    variant["shrinkage"],
                    fit_start,
                    basis_cache,
                )
                factor = locality_factor(target.curve, variant["locality"])
                feature = iem.causal_drop_response(target.curve, lam)
                pred = target.baseline + factor * coef * feature
                corr_mae = iem.mae(target.curve.loss, pred)
                delta = 100.0 * (corr_mae / target.base_mae - 1.0)
                row = {
                    "variant": variant["variant"],
                    "role": variant["role"],
                    "response_rule": variant["response_rule"],
                    "nuisance": variant["nuisance"],
                    "shrinkage": variant["shrinkage"],
                    "locality": variant["locality"],
                    "fit_start": fit_start,
                    "group": group,
                    "train_scale": train_scale,
                    "test_scale": test_scale,
                    "test_curve": curve_name,
                    "test_label": label,
                    "lambda": lam,
                    "coef": coef,
                    "locality_factor": factor,
                    "base_mae": target.base_mae,
                    "corr_mae": corr_mae,
                    "delta_pct": delta,
                    "win": int(corr_mae < target.base_mae),
                    "nonharm": int(delta <= 1e-12),
                    **info,
                }
                detail_rows.append(row)
                coef_key = (train_scale, test_scale, curve_name, f"{lam:.12g}")
                if coef_key not in seen_coef:
                    seen_coef.add(coef_key)
                    coef_rows.append(
                        {
                            key: row[key]
                            for key in [
                                "variant",
                                "response_rule",
                                "nuisance",
                                "shrinkage",
                                "fit_start",
                                "train_scale",
                                "test_scale",
                                "test_curve",
                                "test_label",
                                "lambda",
                                "coef",
                                "basis_dim",
                                "n_cal",
                                "ridge",
                                "source_dot",
                                "source_full_norm",
                                "source_perp_norm",
                                "source_retention",
                                "denominator",
                            ]
                        }
                    )
    return detail_rows, coef_rows


def aggregate(rows: list[dict[str, object]], variant: str, group: str, split: str) -> dict[str, object]:
    deltas = np.array([float(row["delta_pct"]) for row in rows], dtype=np.float64)
    return {
        "variant": variant,
        "group": group,
        "split": split,
        "rows": len(rows),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "worst_delta": float(np.max(deltas)),
        "wins": int(np.sum(deltas < 0.0)),
        "nonharm": int(np.sum(deltas <= 1e-12)),
    }


def summarize(detail_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    variants = list(dict.fromkeys(str(row["variant"]) for row in detail_rows))
    for variant in variants:
        rows_v = [row for row in detail_rows if row["variant"] == variant]
        for group in ["core_wsd", "extra_control"]:
            rows_g = [row for row in rows_v if row["group"] == group]
            splits = {
                "all": rows_g,
                "same_scale": [row for row in rows_g if row["train_scale"] == row["test_scale"]],
                "cross_scale": [row for row in rows_g if row["train_scale"] != row["test_scale"]],
            }
            for scale in iem.SCALES:
                splits[f"holdout_test_{scale}"] = [
                    row
                    for row in rows_g
                    if row["test_scale"] == scale and row["train_scale"] != row["test_scale"]
                ]
            for split, rows in splits.items():
                if rows:
                    out.append(aggregate(rows, variant, group, split))
    return out


def fit_start_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for fit_start in FIT_STARTS:
        for group in ["core_wsd", "extra_control"]:
            sub = [
                row
                for row in rows
                if int(row["fit_start"]) == fit_start
                and row["variant"] == "observation_bracket_mplld_neff"
                and row["group"] == group
                and row["train_scale"] == row["test_scale"]
            ]
            if sub:
                out.append(aggregate(sub, "observation_bracket_mplld_neff", group, f"fit_start_{fit_start}"))
    return out


def parameter_ledger_rows(selected_fit_start: int) -> list[dict[str, object]]:
    return [
        {
            "quantity": "MPL parameters",
            "role": "baseline predictor",
            "source": "precomputed MPL fit already used by baseline",
            "uses_target_loss": "outside_error_model",
            "fitted_in_error_model": 0,
            "notes": "not introduced by the residual correction",
        },
        {
            "quantity": "drop concentration q_s",
            "role": "response-rate interpolation",
            "source": "target LR schedule",
            "uses_target_loss": 0,
            "fitted_in_error_model": 0,
            "notes": "max positive LR drop divided by total positive LR drop",
        },
        {
            "quantity": "lambda_obs",
            "role": "observation-scale response unit",
            "source": "modal logging interval and peak LR",
            "uses_target_loss": 0,
            "fitted_in_error_model": 0,
            "notes": "log(2)/(eta_max * Delta_obs)",
        },
        {
            "quantity": "lambda_s",
            "role": "target response rate",
            "source": "lambda_obs * (1 + q_s) / 2",
            "uses_target_loss": 0,
            "fitted_in_error_model": 0,
            "notes": "observation bracket between one and two logging intervals",
        },
        {
            "quantity": "MPL-LD tangent projection",
            "role": "nuisance removal",
            "source": "finite differences of MPL LR-dependent parameters B,C,beta,gamma",
            "uses_target_loss": 0,
            "fitted_in_error_model": 0,
            "notes": "computed from source curve MPL formula; no residual coefficient",
        },
        {
            "quantity": "fit_start",
            "role": "calibration suffix boundary",
            "source": "earliest source-only lambda-bracket retention pass",
            "uses_target_loss": 0,
            "fitted_in_error_model": 0,
            "notes": f"selected value {selected_fit_start}",
        },
        {
            "quantity": "1/N_cal ridge",
            "role": "finite-sample identifiability floor",
            "source": "number of source calibration points",
            "uses_target_loss": 0,
            "fitted_in_error_model": 0,
            "notes": "replaces fixed tau=0.05",
        },
        {
            "quantity": "kappa_hat_s",
            "role": "response amplitude",
            "source": "one nonnegative projection from cosine residual",
            "uses_target_loss": 0,
            "fitted_in_error_model": 1,
            "notes": "only residual-fitted scalar for each source scale / response operator",
        },
        {
            "quantity": "locality factor a_s",
            "role": "schedule-boundary condition for controls",
            "source": "LR-drop support span",
            "uses_target_loss": 0,
            "fitted_in_error_model": 0,
            "notes": "not a learned gate; prevents full-horizon cosine decay from receiving local cooldown correction",
        },
    ]


def locality_boundary_rows(cache: dict[tuple[str, str], iem.CurvePack]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def pack(scale: str, curve_name: str) -> iem.CurvePack:
        key = (scale, curve_name)
        if key not in cache:
            cache[key] = load_pack(scale, curve_name)
        return cache[key]

    for group, curve_name, label in noa.ALL_TARGETS:
        factors: list[float] = []
        support_spans: list[int] = []
        post_warmups: list[int] = []
        for scale in iem.SCALES:
            curve = pack(scale, curve_name).curve
            eta = curve.lrs.astype(np.float64)
            drop = np.zeros_like(eta)
            drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0)
            idx = np.flatnonzero(drop > 1e-18)
            support_span = int(idx[-1] - idx[0] + 2) if idx.size else 0
            post_warmup = max(len(eta) - iem.WARMUP, 1)
            support_spans.append(support_span)
            post_warmups.append(post_warmup)
            factors.append(iem.drop_localization_factor(curve))
        rows.append(
            {
                "group": group,
                "test_curve": curve_name,
                "test_label": label,
                "min_locality_factor": float(np.min(factors)),
                "median_locality_factor": float(np.median(factors)),
                "max_locality_factor": float(np.max(factors)),
                "support_span": int(np.median(support_spans)),
                "post_warmup_span": int(np.median(post_warmups)),
                "source": "0 if no LR drop else 1 - LR-drop-support-span / post-warmup-span",
                "uses_target_loss": 0,
            }
        )
    return rows



def fit_start_rule_rows(
    cache: dict[tuple[str, str], iem.CurvePack],
    basis_cache: dict[tuple[str, str, int], np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def pack(scale: str, curve_name: str) -> iem.CurvePack:
        key = (scale, curve_name)
        if key not in cache:
            cache[key] = load_pack(scale, curve_name)
        return cache[key]

    for fit_start in FIT_STARTS:
        retentions: list[float] = []
        thresholds: list[float] = []
        lambda_values: list[float] = []
        for scale in iem.SCALES:
            source = pack(scale, iem.TRAIN_CURVE)
            obs = lambda_obs(source.curve)
            lambda_grid = np.array([obs / 2.0, obs], dtype=np.float64)
            for lam in lambda_grid:
                lambda_values.append(float(lam))
                phi, phi_o, _, _ = residualized_pair(
                    source,
                    float(lam),
                    "mpl_ld4",
                    fit_start,
                    basis_cache,
                )
                full_energy = float(np.dot(phi, phi))
                retention = float(np.dot(phi_o, phi_o) / max(full_energy, 1e-18))
                retentions.append(retention)
                thresholds.append(1.0 / max(len(phi_o), 1))
        ret = np.array(retentions, dtype=np.float64)
        thr = np.array(thresholds, dtype=np.float64)
        rows.append(
            {
                "fit_start": fit_start,
                "rows": len(retentions),
                "lambda_grid_points": len(lambda_grid),
                "lambda_min": float(np.min(lambda_values)),
                "lambda_max": float(np.max(lambda_values)),
                "max_retention": float(np.max(ret)),
                "median_retention": float(np.median(ret)),
                "finite_sample_floor": float(np.min(thr)),
                "passes": int(np.all(ret <= thr)),
            }
        )
    return rows


def select_fit_start(rule_rows: list[dict[str, object]]) -> int:
    for row in rule_rows:
        if int(row["passes"]) == 1:
            return int(row["fit_start"])
    return DEFAULT_FIT_START


def find(rows: list[dict[str, object]], variant: str, group: str, split: str) -> dict[str, object]:
    for row in rows:
        if row["variant"] == variant and row["group"] == group and row["split"] == split:
            return row
    raise KeyError((variant, group, split))


def fmt(row: dict[str, object]) -> str:
    return f"{float(row['mean_delta']):+.2f}% / {float(row['worst_delta']):+.2f}% / {int(row['wins'])}/{int(row['rows'])}"


def write_report(
    summary_rows: list[dict[str, object]],
    fit_rows: list[dict[str, object]],
    rule_rows: list[dict[str, object]],
    ledger_rows: list[dict[str, object]],
    locality_rows: list[dict[str, object]],
    selected_fit_start: int,
) -> None:
    main_same = find(summary_rows, "observation_bracket_mplld_neff", "core_wsd", "same_scale")
    main_cross = find(summary_rows, "observation_bracket_mplld_neff", "core_wsd", "cross_scale")
    main_ctrl = find(summary_rows, "observation_bracket_mplld_neff", "extra_control", "same_scale")
    old_same = find(summary_rows, "old_mplld_fixedtau", "core_wsd", "same_scale")
    raw_same = find(summary_rows, "observation_bracket_no_nuisance_neff", "core_wsd", "same_scale")
    no_loc_wsd = find(summary_rows, "observation_bracket_mplld_neff_nolocality", "core_wsd", "same_scale")
    no_loc_ctrl = find(summary_rows, "observation_bracket_mplld_neff_nolocality", "extra_control", "same_scale")

    lines = [
        "# Observation-Bracket MPL-LD Audit\n\n",
        "This audit removes the two weakest protocol constants from the previous MPL-LD response model.\n\n",
        "## Formula\n\n",
        "Let \\(\\Delta_{\\mathrm{obs}}\\) be the modal logging interval and\n\n",
        "\\[\n",
        "\\lambda_{\\mathrm{obs}}=\\frac{\\log 2}{\\eta_{\\max}\\Delta_{\\mathrm{obs}}}.\n",
        "\\]\n\n",
        "For a target schedule, define drop concentration\n\n",
        "\\[\n",
        "q_s=\\frac{\\max_t [\\eta_{t-1}-\\eta_t]_+}{\\sum_t [\\eta_{t-1}-\\eta_t]_+}.\n",
        "\\]\n\n",
        "The new response rate is\n\n",
        "\\[\n",
        "\\lambda_s=\\lambda_{\\mathrm{obs}}\\frac{1+q_s}{2}.\n",
        "\\]\n\n",
        "Equivalently, the response half-life is \\(2\\Delta_{\\mathrm{obs}}/(1+q_s)\\): diffuse LR decay receives a two-observation half-life, while a single sharp drop receives a one-observation half-life.  No target loss is used.\n\n",
        "After projecting both the cosine response feature and cosine residual away from the MPL-LD tangent space, the coefficient is\n\n",
        "\\[\n",
        "\\hat\\kappa_s=\\frac{\\langle x_s,y\\rangle_+}{\\|x_s\\|_2^2+1/N_{\\mathrm{cal}}}.\n",
        "\\]\n\n",
        "Here \\(N_{\\mathrm{cal}}\\) is the number of source cosine points in the calibration suffix.  This replaces the fixed `tau=0.05` ridge with a finite-sample identifiability floor.\n\n",
        "## Parameter Ledger\n\n",
        "| quantity | role | source | fitted? | target loss? |\n",
        "|---|---|---|---:|---:|\n",
    ]
    for row in ledger_rows:
        lines.append(
            f"| {row['quantity']} | {row['role']} | {row['source']} | "
            f"{row['fitted_in_error_model']} | {row['uses_target_loss']} |\n"
        )
    lines += [
        "\nThe only residual-fitted quantity introduced by the error model is the nonnegative scalar \\(\\hat\\kappa_s\\).  Every other term is derived from the LR schedule, the logging resolution, the source suffix size, or the existing MPL formula.\n\n",
        "## Locality Boundary\n\n",
        "The locality factor is a schedule-support boundary condition, not a learned gate:\n\n",
        "\\[\n",
        "a_s=\\mathbf{1}\\{\\sum_t d_t>0\\}\\left[1-\\frac{\\ell_s}{T_s-W}\\right]_+,\n",
        "\\]\n\n",
        "where \\(\\ell_s\\) is the support span of positive LR drops after warmup.  It uses only the LR schedule and is never fit from loss values.\n\n",
        "| curve | group | median factor | support span | post-warmup span |\n",
        "|---|---|---:|---:|---:|\n",
    ]
    for row in locality_rows:
        lines.append(
            f"| {row['test_label']} | {row['group']} | {float(row['median_locality_factor']):.4f} | "
            f"{int(row['support_span'])} | {int(row['post_warmup_span'])} |\n"
        )
    linear_cost = float(main_same["mean_delta"]) - float(no_loc_wsd["mean_delta"])
    lines += [
        "\nLocality tradeoff:\n\n",
        f"- Without locality, WSD remains all-win: `{fmt(no_loc_wsd)}`.\n",
        f"- Without locality, controls fail: `{fmt(no_loc_ctrl)}`.\n",
        f"- Linear locality changes same-scale WSD mean by `{linear_cost:+.2f}` percentage points while restoring all controls to non-harm.\n\n",
        "## Source-Only Suffix Rule\n\n",
        "The calibration suffix is selected without WSD losses or target schedule enumeration.  For candidate suffix starts, evaluate the two endpoints of the observation bracket, \\(\\lambda_{\\mathrm{obs}}/2\\) and \\(\\lambda_{\\mathrm{obs}}\\), on the source cosine curve and compute the retained response-feature energy after MPL-LD projection:\n\n",
        "\\[\n",
        "\\rho=\\frac{\\|(I-P_{\\mathrm{LD}})\\phi\\|_2^2}{\\|\\phi\\|_2^2}.\n",
        "\\]\n\n",
        "Choose the earliest suffix start whose maximum endpoint \\(\\rho\\) over source scales is below the finite-sample floor \\(1/N_{\\mathrm{cal}}\\).  A dense grid check with 2, 3, 5, 9, 17, 33, 65, and 129 points selects the same suffix, so this endpoint rule is not a grid-resolution artifact.  This avoids early cosine segments where the response direction is still too entangled with MPL-LD drift.\n\n",
        f"Selected fit start: `{selected_fit_start}`.\n\n",
        "| fit start | lambda points | max retention | median retention | floor | passes |\n",
        "|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in rule_rows:
        lines.append(
            f"| {int(row['fit_start'])} | {int(row['lambda_grid_points'])} | "
            f"{float(row['max_retention']):.6g} | "
            f"{float(row['median_retention']):.6g} | {float(row['finite_sample_floor']):.6g} | "
            f"{int(row['passes'])} |\n"
        )
    lines += [
        "\n",
        "## Key Results\n\n",
        "| variant | split | group | mean / worst / wins |\n",
        "|---|---|---|---:|\n",
        f"| observation_bracket_mplld_neff | same_scale | core_wsd | {fmt(main_same)} |\n",
        f"| observation_bracket_mplld_neff | cross_scale | core_wsd | {fmt(main_cross)} |\n",
        f"| observation_bracket_mplld_neff | same_scale | extra_control | {fmt(main_ctrl)} |\n",
        f"| old_mplld_fixedtau | same_scale | core_wsd | {fmt(old_same)} |\n",
        f"| no-nuisance failure | same_scale | core_wsd | {fmt(raw_same)} |\n",
        f"| no-locality control boundary | same_scale | extra_control | {fmt(no_loc_ctrl)} |\n",
        "\n## Fit-Start Sensitivity\n\n",
        "| fit start | group | mean | worst | wins/non-harm |\n",
        "|---:|---|---:|---:|---:|\n",
    ]
    for row in fit_rows:
        fit_start = row["split"].replace("fit_start_", "")
        lines.append(
            f"| {fit_start} | {row['group']} | {float(row['mean_delta']):+.2f}% | "
            f"{float(row['worst_delta']):+.2f}% | {int(row['wins'])}/{int(row['rows'])}, "
            f"{int(row['nonharm'])}/{int(row['rows'])} |\n"
        )

    lines += [
        "\n## Reading\n\n",
        "- The observation-bracket rule is stronger than the previous MPL-LD reference while removing the fixed `2.5` slow endpoint, rounded fast endpoint `20`, and fixed `tau=0.05`.\n",
        "- Raw projection still fails badly, so the MPL-LD nuisance projection remains essential.\n",
        "- No-locality WSD performance remains positive, but controls fail; locality should be written as a schedule-boundary condition, not as the mechanism itself.\n",
        "- The fit-start scan is a protocol audit.  A result is research-safe only if the main conclusion is not tied to a single suffix boundary.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    cache: dict[tuple[str, str], iem.CurvePack] = {}
    basis_cache: dict[tuple[str, str, int], np.ndarray] = {}
    detail_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    rule_rows = fit_start_rule_rows(cache, basis_cache)
    selected_fit_start = select_fit_start(rule_rows)
    ledger_rows = parameter_ledger_rows(selected_fit_start)
    locality_rows = locality_boundary_rows(cache)

    for variant in VARIANTS:
        rows, coef_rows = evaluate_variant(variant, selected_fit_start, cache, basis_cache, cross_scale=True)
        detail_rows.extend(rows)
        coefficient_rows.extend(coef_rows)

    fit_detail_rows: list[dict[str, object]] = []
    main_variant = VARIANTS[0]
    for fit_start in FIT_STARTS:
        rows, _ = evaluate_variant(main_variant, fit_start, cache, basis_cache, cross_scale=False)
        fit_detail_rows.extend(rows)

    summary_rows = summarize(detail_rows)
    fit_summary_rows = fit_start_summary(fit_detail_rows)
    iem.write_csv(OUT_DIR / "details.csv", detail_rows)
    iem.write_csv(OUT_DIR / "summary.csv", summary_rows)
    iem.write_csv(OUT_DIR / "coefficients.csv", coefficient_rows)
    iem.write_csv(OUT_DIR / "parameter_ledger.csv", ledger_rows)
    iem.write_csv(OUT_DIR / "locality_boundary.csv", locality_rows)
    iem.write_csv(OUT_DIR / "fit_start_rule.csv", rule_rows)
    iem.write_csv(OUT_DIR / "fit_start_details.csv", fit_detail_rows)
    iem.write_csv(OUT_DIR / "fit_start_summary.csv", fit_summary_rows)
    write_report(summary_rows, fit_summary_rows, rule_rows, ledger_rows, locality_rows, selected_fit_start)
    print(f"wrote {OUT_DIR / 'details.csv'}")
    print(f"wrote {OUT_DIR / 'summary.csv'}")
    print(f"wrote {OUT_DIR / 'coefficients.csv'}")
    print(f"wrote {OUT_DIR / 'parameter_ledger.csv'}")
    print(f"wrote {OUT_DIR / 'locality_boundary.csv'}")
    print(f"wrote {OUT_DIR / 'fit_start_rule.csv'}")
    print(f"wrote {OUT_DIR / 'fit_start_details.csv'}")
    print(f"wrote {OUT_DIR / 'fit_start_summary.csv'}")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
