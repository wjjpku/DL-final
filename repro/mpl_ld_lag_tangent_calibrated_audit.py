#!/usr/bin/env python3
"""Tangent-calibrated MPL-LD finite-response audit.

Raw cosine-fitted amplitudes failed because cosine residuals contain two
effects at once:

1. local finite-response error in MPL's LR-dependent cooldown term;
2. ordinary MPL backbone parameter error.

This audit keeps the finite-response feature fixed, but estimates one scalar
amplitude after removing the local MPL tangent space from the cosine residual.
It is a constrained calibration, not a new residual basis:

    phi_s(t) = B_s [D_down,tau(t) - D_down(t)]

    kappa_s = <P_perp phi_s, P_perp residual_s>_+
              / ||P_perp phi_s||^2

    L_hat(t) = L_MPL(t) + a_target kappa_s phi_target(t)

where P_perp projects away infinitesimal changes of MPL parameters on the
cosine training curves.  Target losses are used only for evaluation.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
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

from reproduce_cosine_to_wsd import (  # noqa: E402
    MPL_PRECOMPUTED_INIT,
    SCALES,
    TRAIN_CURVES,
    WARMUP,
    load_curve,
    metrics,
    mpl_predict,
)


OUT_DIR = ROOT / "results" / "mpl_ld_lag_response_audit" / "tangent_calibrated"
STRICT_PARAM_JSON = (
    ROOT
    / "results"
    / "cosine_to_wsd_response_search"
    / "cosine_only_backbone"
    / "cosine_only_mpl_params.json"
)
FIT_STARTS = [WARMUP, 5000, 8000]
MAIN_FIT_START = 5000
SHRINKAGES = ["orth_ols", "sqrt_retention", "full_energy"]
CORE_TARGETS = [
    ("core_wsd", "wsd_20000_24000.csv", "WSD sharp"),
    ("core_wsd", "wsdld_20000_24000.csv", "WSD linear"),
    ("core_wsd", "wsdcon_3.csv", "WSD-con 3e-5"),
    ("core_wsd", "wsdcon_9.csv", "WSD-con 9e-5"),
    ("core_wsd", "wsdcon_18.csv", "WSD-con 18e-5"),
]
EXTRA_CONTROLS = [
    ("extra_control", "cosine_24000.csv", "Cosine 24k"),
    ("extra_control", "constant_24000.csv", "Constant 24k"),
    ("extra_control", "constant_72000.csv", "Constant 72k"),
]
ALL_TARGETS = CORE_TARGETS + EXTRA_CONTROLS
TANGENT_GROUPS = {
    "none": [],
    "ld4": ["logB", "logC", "logBeta", "logGamma"],
    "all7": ["L0", "logA", "alpha", "logB", "logC", "logBeta", "logGamma"],
}


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
    return f"{value:+.2f}%"


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def read_strict_params() -> dict[str, np.ndarray]:
    raw = json.loads(STRICT_PARAM_JSON.read_text(encoding="utf-8"))
    return {scale: np.array(raw[scale], dtype=np.float64) for scale in SCALES}


def backbone_params() -> dict[str, dict[str, np.ndarray]]:
    strict = read_strict_params()
    return {
        "frozen_official": {scale: np.array(MPL_PRECOMPUTED_INIT[scale], dtype=np.float64) for scale in SCALES},
        "cosine_only": strict,
    }


def modal_observation_interval(step: np.ndarray) -> int:
    diffs = np.diff(step)
    values, counts = np.unique(diffs[diffs > 0], return_counts=True)
    if values.size == 0:
        return 1
    return int(values[int(np.argmax(counts))])


def cooldown_support_span(lrs: np.ndarray) -> int:
    eta = lrs.astype(np.float64)
    drop = np.zeros_like(eta)
    drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0)
    idx = np.flatnonzero(drop > 1e-18)
    return int(idx[-1] - idx[0] + 2) if idx.size else 0


def adiabatic_factor(lrs: np.ndarray) -> float:
    return max(0.0, 1.0 - float(cooldown_support_span(lrs)) / float(max(len(lrs) - WARMUP, 1)))


def support_bracket_tau(step: np.ndarray, lrs: np.ndarray) -> float:
    interval = float(modal_observation_interval(step))
    span = float(cooldown_support_span(lrs))
    return interval * (1.0 + min(1.0, span / max(interval, 1.0)))


def lagged_observed(values: np.ndarray, steps: np.ndarray, tau_steps: float) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    out[0] = float(values[0])
    for idx in range(1, len(values)):
        delta_steps = max(float(steps[idx] - steps[idx - 1]), 1.0)
        rho = math.exp(-delta_steps / max(float(tau_steps), 1e-12))
        out[idx] = rho * out[idx - 1] + (1.0 - rho) * float(values[idx])
    return out


def ld_cooldown(curve, params: np.ndarray) -> np.ndarray:
    _, _, _, _, c_value, beta, gamma = params
    lrs = curve.lrs.astype(np.float64)
    lr_sum = np.cumsum(lrs)
    lr_gap = np.zeros(len(lrs), dtype=np.float64)
    lr_gap[1:] = np.diff(lrs)
    selected_gap = np.minimum(lr_gap, 0.0)
    out = np.zeros(len(curve.step), dtype=np.float64)
    for idx, step in enumerate(curve.step):
        if step <= 0:
            continue
        hist = lrs[1 : step + 1]
        delta = selected_gap[1 : step + 1]
        remain = lr_sum[step] - lr_sum[:step]
        term = 1.0 - (1.0 + c_value * np.power(hist, -gamma) * remain) ** (-beta)
        out[idx] = np.sum(delta * term)
    return out


def response_feature(curve, params: np.ndarray, attenuate: bool) -> np.ndarray:
    d_down = ld_cooldown(curve, params)
    tau = support_bracket_tau(curve.step, curve.lrs)
    d_lag = lagged_observed(d_down, curve.step, tau)
    factor = adiabatic_factor(curve.lrs) if attenuate else 1.0
    return factor * float(params[3]) * (d_lag - d_down)


def tangent_columns(curves: list[object], params: np.ndarray, names: list[str], masks: list[np.ndarray]) -> np.ndarray:
    cols: list[np.ndarray] = []
    eps = 1e-4

    def concat_prediction(p: np.ndarray) -> np.ndarray:
        parts = [mpl_predict(p, curve)[mask] for curve, mask in zip(curves, masks)]
        return np.concatenate(parts)

    for name in names:
        if name == "L0":
            cols.append(np.ones(sum(int(np.sum(mask)) for mask in masks), dtype=np.float64))
            continue
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
        cols.append((concat_prediction(pp) - concat_prediction(pm)) / denom)
    if not cols:
        return np.zeros((sum(int(np.sum(mask)) for mask in masks), 0), dtype=np.float64)
    mat = np.column_stack(cols)
    keep = np.linalg.norm(mat, axis=0) > 1e-10
    if not np.any(keep):
        return np.zeros((mat.shape[0], 0), dtype=np.float64)
    q, r = np.linalg.qr(mat[:, keep])
    keep_q = np.abs(np.diag(r)) > 1e-8
    return q[:, keep_q]


def project_out(x: np.ndarray, q: np.ndarray) -> np.ndarray:
    if q.size == 0:
        return x
    return x - q @ (q.T @ x)


def fit_kappa_base(scale: str, params: np.ndarray, fit_start: int, tangent_group: str) -> dict[str, object]:
    curves = [load_curve(scale, name) for name in TRAIN_CURVES]
    masks = [curve.step >= fit_start for curve in curves]
    residual = np.concatenate([(curve.loss - mpl_predict(params, curve))[mask] for curve, mask in zip(curves, masks)])
    feature = np.concatenate([response_feature(curve, params, attenuate=False)[mask] for curve, mask in zip(curves, masks)])
    q = tangent_columns(curves, params, TANGENT_GROUPS[tangent_group], masks)
    feature_o = project_out(feature, q)
    residual_o = project_out(residual, q)
    denom = max(float(np.dot(feature_o, feature_o)), 1e-18)
    raw_num = float(np.dot(feature, residual))
    orth_num = float(np.dot(feature_o, residual_o))
    orth_ols = max(0.0, orth_num) / denom
    retention = float(np.dot(feature_o, feature_o)) / max(float(np.dot(feature, feature)), 1e-18)
    return {
        "orth_ols_kappa": orth_ols,
        "raw_cosine_alignment": raw_num / max(float(np.dot(feature, feature)), 1e-18),
        "orthogonal_alignment": orth_num / denom,
        "feature_retention": retention,
        "tangent_dim": int(q.shape[1]),
    }


def shrink_kappa(info: dict[str, object], shrinkage: str) -> float:
    orth_ols = float(info["orth_ols_kappa"])
    retention = max(float(info["feature_retention"]), 0.0)
    if shrinkage == "orth_ols":
        return orth_ols
    if shrinkage == "sqrt_retention":
        return orth_ols * math.sqrt(retention)
    if shrinkage == "full_energy":
        return orth_ols * retention
    raise ValueError(f"unknown shrinkage: {shrinkage}")


def detail_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    params_by_protocol = backbone_params()
    official_mae: dict[tuple[str, str], float] = {}
    for scale in SCALES:
        for _, curve_name, _ in ALL_TARGETS:
            curve = load_curve(scale, curve_name)
            official_mae[(scale, curve_name)] = metrics(
                curve.loss,
                mpl_predict(MPL_PRECOMPUTED_INIT[scale], curve),
            )["mae"]

    detail: list[dict[str, object]] = []
    kappa_rows: list[dict[str, object]] = []
    for protocol, param_map in params_by_protocol.items():
        for fit_start in FIT_STARTS:
            for tangent_group in TANGENT_GROUPS:
                kappa_by_scale: dict[str, dict[str, object]] = {}
                for scale in SCALES:
                    info = fit_kappa_base(scale, param_map[scale], fit_start, tangent_group)
                    kappa_by_scale[scale] = info
                    for shrinkage in SHRINKAGES:
                        kappa_rows.append(
                            {
                                "protocol": protocol,
                                "scale": scale,
                                "fit_start": fit_start,
                                "tangent_group": tangent_group,
                                "shrinkage": shrinkage,
                                "kappa": shrink_kappa(info, shrinkage),
                                **info,
                            }
                        )

                for scale in SCALES:
                    params = param_map[scale]
                    for shrinkage in SHRINKAGES:
                        kappa = shrink_kappa(kappa_by_scale[scale], shrinkage)
                        for group, curve_name, label in ALL_TARGETS:
                            curve = load_curve(scale, curve_name)
                            baseline = mpl_predict(params, curve)
                            pred = baseline + kappa * response_feature(curve, params, attenuate=True)
                            base_mae = mae(curve.loss, baseline)
                            corr_mae = mae(curve.loss, pred)
                            ref = official_mae[(scale, curve_name)]
                            detail.append(
                                {
                                    "protocol": protocol,
                                    "fit_start": fit_start,
                                    "tangent_group": tangent_group,
                                    "shrinkage": shrinkage,
                                    "group": group,
                                    "scale": scale,
                                    "test_curve": curve_name,
                                    "test_label": label,
                                    "kappa": kappa,
                                    "orth_ols_kappa": float(kappa_by_scale[scale]["orth_ols_kappa"]),
                                    "tangent_dim": int(kappa_by_scale[scale]["tangent_dim"]),
                                    "feature_retention": float(kappa_by_scale[scale]["feature_retention"]),
                                    "base_mae": base_mae,
                                    "corr_mae": corr_mae,
                                    "delta_vs_own_baseline_pct": 100.0 * (corr_mae / base_mae - 1.0),
                                    "base_vs_official_baseline_pct": 100.0 * (base_mae / ref - 1.0),
                                    "corr_vs_official_baseline_pct": 100.0 * (corr_mae / ref - 1.0),
                                    "win_vs_own_baseline": int(corr_mae < base_mae),
                                    "nonharm_vs_own_baseline": int(corr_mae <= base_mae + 1e-12),
                                }
                            )
    return detail, kappa_rows


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    keys = sorted(
        {
            (
                str(row["protocol"]),
                str(row["fit_start"]),
                str(row["tangent_group"]),
                str(row["shrinkage"]),
                str(row["group"]),
            )
            for row in rows
        }
    )
    for protocol, fit_start, tangent_group, shrinkage, group in keys:
        sub = [
            row
            for row in rows
            if row["protocol"] == protocol
            and str(row["fit_start"]) == fit_start
            and row["tangent_group"] == tangent_group
            and row["shrinkage"] == shrinkage
            and row["group"] == group
        ]
        own = np.array([float(row["delta_vs_own_baseline_pct"]) for row in sub], dtype=np.float64)
        base_off = np.array([float(row["base_vs_official_baseline_pct"]) for row in sub], dtype=np.float64)
        corr_off = np.array([float(row["corr_vs_official_baseline_pct"]) for row in sub], dtype=np.float64)
        out.append(
            {
                "protocol": protocol,
                "fit_start": int(fit_start),
                "tangent_group": tangent_group,
                "shrinkage": shrinkage,
                "group": group,
                "rows": len(sub),
                "mean_delta_vs_own_baseline": float(np.mean(own)),
                "worst_delta_vs_own_baseline": float(np.max(own)),
                "wins_vs_own_baseline": int(np.sum(own < 0.0)),
                "nonharm_vs_own_baseline": int(np.sum(own <= 1e-12)),
                "mean_base_vs_official_baseline": float(np.mean(base_off)),
                "mean_corr_vs_official_baseline": float(np.mean(corr_off)),
                "worst_corr_vs_official_baseline": float(np.max(corr_off)),
            }
        )
    return out


def find(
    rows: list[dict[str, object]],
    protocol: str,
    fit_start: int,
    tangent_group: str,
    shrinkage: str,
    group: str,
) -> dict[str, object]:
    for row in rows:
        if (
            row["protocol"] == protocol
            and int(row["fit_start"]) == fit_start
            and row["tangent_group"] == tangent_group
            and row["shrinkage"] == shrinkage
            and row["group"] == group
        ):
            return row
    raise KeyError((protocol, fit_start, tangent_group, shrinkage, group))


def write_report(summary: list[dict[str, object]]) -> None:
    lines = [
        "# Tangent-Calibrated MPL-LD Finite-Response Audit\n\n",
        "This audit estimates one scalar finite-response amplitude from cosine residuals after projecting out the local MPL tangent space.  It is designed to test whether cosine residual contamination can be removed without gate/channel/DCT/sinusoidal terms.\n\n",
        "Calibration formula:\n\n",
        "\\[\n",
        "\\hat\\kappa_s=\\frac{\\langle P_\\perp\\phi_s, P_\\perp r_s\\rangle_+}{\\|P_\\perp\\phi_s\\|^2},\\qquad "
        "\\phi_s(t)=B_s[D_{\\downarrow,\\tau_s,s}(t)-D_{\\downarrow,s}(t)].\n",
        "\\]\n\n",
        "Prediction formula:\n\n",
        "\\[\n",
        "\\hat L_s(t)=L_{\\mathrm{MPL},s}(t)+a_s\\hat\\kappa_s\\phi_s(t).\n",
        "\\]\n\n",
        "## Main Rows\n\n",
        "| protocol | fit start | tangent | shrinkage | WSD correction | wins / non-harm | corrected vs official | controls non-harm |\n",
        "|---|---:|---|---|---:|---:|---:|---:|\n",
    ]
    for protocol in ["frozen_official", "cosine_only"]:
        for tangent_group in ["none", "ld4", "all7"]:
            for shrinkage in SHRINKAGES:
                core = find(summary, protocol, MAIN_FIT_START, tangent_group, shrinkage, "core_wsd")
                ctrl = find(summary, protocol, MAIN_FIT_START, tangent_group, shrinkage, "extra_control")
                lines.append(
                    f"| {protocol} | {MAIN_FIT_START} | {tangent_group} | {shrinkage} | "
                    f"{fmt_pct(float(core['mean_delta_vs_own_baseline']))} mean / "
                    f"{fmt_pct(float(core['worst_delta_vs_own_baseline']))} worst | "
                    f"{int(core['wins_vs_own_baseline'])}/{int(core['rows'])} / "
                    f"{int(core['nonharm_vs_own_baseline'])}/{int(core['rows'])} | "
                    f"{fmt_pct(float(core['mean_corr_vs_official_baseline']))} mean | "
                    f"{int(ctrl['nonharm_vs_own_baseline'])}/{int(ctrl['rows'])} |\n"
                )
    lines += [
        "\n## Fit-Start Sensitivity\n\n",
        "| protocol | tangent | shrinkage | fit start | WSD mean | WSD worst | wins |\n",
        "|---|---|---|---:|---:|---:|---:|\n",
    ]
    for protocol in ["frozen_official", "cosine_only"]:
        for tangent_group in ["all7", "ld4", "none"]:
            for shrinkage in ["sqrt_retention", "full_energy", "orth_ols"]:
                for fit_start in FIT_STARTS:
                    core = find(summary, protocol, fit_start, tangent_group, shrinkage, "core_wsd")
                    lines.append(
                        f"| {protocol} | {tangent_group} | {shrinkage} | {fit_start} | "
                        f"{fmt_pct(float(core['mean_delta_vs_own_baseline']))} | "
                        f"{fmt_pct(float(core['worst_delta_vs_own_baseline']))} | "
                        f"{int(core['wins_vs_own_baseline'])}/{int(core['rows'])} |\n"
                    )
    lines += [
        "\n## Reading\n\n",
        "- `none` is the known bad raw cosine-amplitude path if it over-transfers.\n",
        "- `ld4` removes only MPL's LR-dependent tangent directions; `all7` removes the full local MPL parameter tangent.\n",
        "- `orth_ols` is the direct projected least-squares estimator; `sqrt_retention` and `full_energy` are no-hyperparameter energy normalizations that prevent tiny projected feature energy from creating a huge amplitude.\n",
        "- This still has one fitted scalar per scale.  It is more flexible than the zero-parameter finite-response row, so it must be judged by strict cosine-only performance and sensitivity, not just the best frozen-backbone number.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    details, kappas = detail_rows()
    summary = aggregate(details)
    write_csv(OUT_DIR / "details.csv", details)
    write_csv(OUT_DIR / "kappas.csv", kappas)
    write_csv(OUT_DIR / "summary.csv", summary)
    write_report(summary)
    print(f"wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
