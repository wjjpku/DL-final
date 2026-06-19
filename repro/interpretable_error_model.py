#!/usr/bin/env python3
"""Small fixed-candidate audit for interpretable cosine-to-WSD error models.

This script intentionally avoids broad architecture search.  It evaluates a
few pre-declared mechanism candidates:

* raw causal LR-drop response,
* slope-modulated lag response,
* two-timescale slope-modulated lag response.

All coefficients are fitted only from cosine_72000 residuals.  WSD-family
curves are used only for evaluation.
"""
from __future__ import annotations

import csv
import math
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

OUT_DIR = ROOT / "results" / "interpretable_error_model"
DATA_ROOT = ROOT / "external" / "MultiPowerLaw" / "loss_curve_repo"
TRAIN_CURVE = "cosine_72000.csv"
WARMUP = 2160
PEAK_LR = 3e-4
END_LR = 3e-5
SCALES = ["25", "100", "400"]
MPL_PRECOMPUTED_INIT = {
    "25": np.array([3.04045406, 0.52468604, 0.50786857, 363.78751622, 2.06560812, 0.58279013, 0.64142257]),
    "100": np.array([2.6514477, 0.60115152, 0.45295811, 437.9464276, 2.13245612, 0.59785199, 0.65523644]),
    "400": np.array([2.37474466, 0.65421216, 0.42878731, 523.42464371, 2.02462735, 0.59350493, 0.63472457]),
}
TARGETS = [
    ("wsd_20000_24000.csv", "WSD sharp"),
    ("wsdld_20000_24000.csv", "WSD linear"),
    ("wsdcon_3.csv", "WSD-con 3e-5"),
    ("wsdcon_9.csv", "WSD-con 9e-5"),
    ("wsdcon_18.csv", "WSD-con 18e-5"),
]


@dataclass(frozen=True)
class Curve:
    name: str
    scale: str
    step: np.ndarray
    loss: np.ndarray
    lrs: np.ndarray

FIT_STARTS = [5000, 8000]
MU_VALUES = [0.01, 0.02, 0.05]
DCT_MODES = 8
RIDGE_TAU = 0.05
OBS_HALF_LIFE_MULTIPLIER = 2.5
OBS_FAST_LAMBDA = 20.0


@dataclass(frozen=True)
class CurvePack:
    curve: Curve
    baseline: np.ndarray
    residual: np.ndarray
    base_mae: float
    slope_raw: np.ndarray
    slope_norm: np.ndarray
    ld_basis: np.ndarray
    dlogc_basis: np.ndarray


@dataclass(frozen=True)
class Candidate:
    name: str
    kind: str
    lambdas: tuple[float, ...]
    signed: bool = False


CANDIDATES = [
    Candidate("raw_drop_l20", "raw", (20.0,)),
    Candidate("raw_drop_l7", "raw", (7.0,)),
    Candidate("continuous_lambda_raw_7_20", "adaptive_raw", (7.0, 20.0)),
    Candidate("continuous_lambda_raw_4_20", "adaptive_raw", (4.0, 20.0)),
    Candidate("continuous_lambda_projected_7_20", "adaptive_projected_raw", (7.0, 20.0)),
    Candidate("continuous_lambda_projected_4_20", "adaptive_projected_raw", (4.0, 20.0)),
    Candidate(
        "obs_half_life_projected_2p5_roundfast20",
        "adaptive_observed_projected_raw",
        (OBS_HALF_LIFE_MULTIPLIER, OBS_FAST_LAMBDA),
    ),
    Candidate(
        "obs_half_life_localized_projected_2p5_roundfast20",
        "adaptive_observed_localized_projected_raw",
        (OBS_HALF_LIFE_MULTIPLIER, OBS_FAST_LAMBDA),
    ),
    Candidate(
        "obs_half_life_sqrtlocalized_projected_2p5_roundfast20",
        "adaptive_observed_sqrtlocalized_projected_raw",
        (OBS_HALF_LIFE_MULTIPLIER, OBS_FAST_LAMBDA),
    ),
    Candidate("lag_rawslope_l20", "lag_raw", (20.0,)),
    Candidate("lag_rawslope_l7", "lag_raw", (7.0,)),
    Candidate("lag_rawslope_l4", "lag_raw", (4.0,)),
    Candidate("lag_normslope_l20", "lag_norm", (20.0,)),
    Candidate("lag_spectrum_raw_l7_l20", "lag_raw", (7.0, 20.0)),
    Candidate("lag_spectrum_raw_l4_l20", "lag_raw", (4.0, 20.0)),
    Candidate("mpl_sensitivity_B", "mpl_ld_b", (), True),
    Candidate("mpl_sensitivity_logC", "mpl_logc", (), True),
]


def cosine_lrs(warmup: int, total: int, peak_lr: float, end_lr: float) -> np.ndarray:
    step = np.arange(total)[warmup:]
    warmup_lrs = np.linspace(0.0, peak_lr, warmup)
    cosine = end_lr + 0.5 * (peak_lr - end_lr) * (
        1.0 + np.cos(np.pi * (step - warmup) / (total - warmup))
    )
    return np.concatenate((warmup_lrs, cosine))


def const_lrs(warmup: int, total: int, lr: float) -> np.ndarray:
    warmup_lrs = np.linspace(0.0, lr, warmup)
    return np.concatenate((warmup_lrs, np.full(total - warmup, lr)))


def two_stage_lrs(warmup: int, total: int, lr_a: float, lr_b: float, stage_a: int) -> np.ndarray:
    warmup_lrs = np.linspace(0.0, lr_a, warmup)
    stage_a_lrs = np.full(stage_a - warmup, lr_a)
    stage_b_lrs = np.full(total - stage_a, lr_b)
    return np.concatenate((warmup_lrs, stage_a_lrs, stage_b_lrs))


def wsd_lrs(warmup: int, total: int, decay: int, peak_lr: float, end_lr: float) -> np.ndarray:
    step = np.arange(total)[decay:]
    warmup_lrs = np.linspace(0.0, peak_lr, warmup)
    decay_lrs = peak_lr ** ((total - step) / (total - decay)) * end_lr ** (
        (step - decay) / (total - decay)
    )
    return np.concatenate((warmup_lrs, np.full(decay - warmup, peak_lr), decay_lrs))


def wsdld_lrs(warmup: int, total: int, decay: int, peak_lr: float, end_lr: float) -> np.ndarray:
    step = np.arange(total)[decay:]
    warmup_lrs = np.linspace(0.0, peak_lr, warmup)
    decay_lrs = peak_lr * (1.0 - (step - decay) / (total - decay)) + end_lr * (
        step - decay
    ) / (total - decay)
    return np.concatenate((warmup_lrs, np.full(decay - warmup, peak_lr), decay_lrs))


def build_lrs(file_name: str) -> np.ndarray:
    if "cosine" in file_name:
        total = int(file_name.split("_")[1].split(".")[0])
        return cosine_lrs(WARMUP, total, PEAK_LR, END_LR)
    if "constant" in file_name:
        total = int(file_name.split("_")[1].split(".")[0])
        return const_lrs(WARMUP, total, PEAK_LR)
    if "wsdcon" in file_name:
        lr_b = int(file_name.split("_")[1].split(".")[0]) * 1e-5
        return two_stage_lrs(WARMUP, 16000, PEAK_LR, lr_b, 8000)
    if "wsdld" in file_name:
        return wsdld_lrs(WARMUP, 24000, 20000, PEAK_LR, END_LR)
    if "wsd" in file_name:
        return wsd_lrs(WARMUP, 24000, 20000, PEAK_LR, END_LR)
    raise ValueError(f"Unsupported curve: {file_name}")


def load_curve(scale: str, file_name: str) -> Curve:
    path = DATA_ROOT / f"csv_{scale}" / file_name
    raw = np.genfromtxt(path, delimiter=",", skip_header=1)
    step = raw[:, 0].astype(int)
    loss = raw[:, 2].astype(float)
    if step.max() == 24000:
        mask = step < 24000
        step = step[mask]
        loss = loss[mask]
    return Curve(name=file_name, scale=scale, step=step, loss=loss, lrs=build_lrs(file_name))


def compute_s1(curve: Curve) -> np.ndarray:
    return np.cumsum(curve.lrs)[curve.step]


def compute_ld(curve: Curve, c_value: float, beta: float, gamma: float) -> np.ndarray:
    lrs = curve.lrs
    lr_sum = np.cumsum(lrs)
    lr_gap = np.zeros(len(lrs), dtype=np.float64)
    lr_gap[1:] = np.diff(lrs)
    ld = np.zeros(len(curve.step), dtype=np.float64)
    for idx, step in enumerate(curve.step):
        if step <= 0:
            continue
        hist = lrs[1 : step + 1]
        delta = lr_gap[1 : step + 1]
        remain = lr_sum[step] - lr_sum[:step]
        term = 1.0 - (1.0 + c_value * np.power(hist, -gamma) * remain) ** (-beta)
        ld[idx] = np.sum(delta * term)
    return ld


def mpl_predict(params: np.ndarray, curve: Curve) -> np.ndarray:
    l0, a_value, alpha, b_value, c_value, beta, gamma = params
    s1 = compute_s1(curve)
    ld = compute_ld(curve, c_value, beta, gamma)
    return l0 + a_value * np.power(s1, -alpha) + b_value * ld


def mpl_sensitivity_features(params: np.ndarray, curve: Curve) -> tuple[np.ndarray, np.ndarray]:
    """First-order directions inside MPL's LR-dependent term."""
    _, _, _, b_value, c_value, beta, gamma = params
    ld = compute_ld(curve, c_value, beta, gamma)
    eps = 1e-3
    ld_plus = compute_ld(curve, c_value * math.exp(eps), beta, gamma)
    ld_minus = compute_ld(curve, c_value * math.exp(-eps), beta, gamma)
    dlogc = b_value * (ld_plus - ld_minus) / (2.0 * eps)
    return ld, dlogc


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dct_basis(n: int, max_mode: int) -> np.ndarray:
    idx = np.arange(n, dtype=np.float64)
    cols = [np.ones(n, dtype=np.float64)]
    for k in range(1, max_mode + 1):
        cols.append(np.cos(math.pi * (idx + 0.5) * k / max(n, 1)))
    q = np.column_stack(cols)
    return q / np.maximum(np.linalg.norm(q, axis=0), 1e-12)


def soft_residualize(y: np.ndarray, q: np.ndarray, nuisance_lambda: float) -> np.ndarray:
    modes = np.arange(q.shape[1], dtype=np.float64)
    penalty = nuisance_lambda * np.power(modes, 4.0)
    penalty[0] = 0.0
    lhs = q.T @ q + np.diag(penalty)
    coef = np.linalg.solve(lhs, q.T @ y)
    return y - q @ coef


def causal_drop_response(curve: Curve, response_lambda: float) -> np.ndarray:
    """Use the same LR-time convention as prior cosine-to-WSD audits."""
    eta = curve.lrs.astype(np.float64)
    drop = np.zeros_like(eta)
    drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0)
    out = np.empty_like(eta)
    acc = 0.0
    for t in range(len(eta)):
        acc = acc * math.exp(-response_lambda * float(eta[t])) + drop[t]
        out[t] = acc
    return (out / PEAK_LR)[curve.step]


def drop_concentration(curve: Curve) -> float:
    eta = curve.lrs.astype(np.float64)
    drop = np.zeros_like(eta)
    drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0)
    total = float(np.sum(drop))
    return float(np.max(drop) / total) if total > 1e-18 else 0.0


def adaptive_lambda(curve: Curve, lambdas: tuple[float, ...]) -> float:
    low, high = lambdas
    return low + (high - low) * drop_concentration(curve)


def modal_observation_interval(curve: Curve) -> int:
    diffs = np.diff(curve.step)
    values, counts = np.unique(diffs[diffs > 0], return_counts=True)
    if values.size == 0:
        return 1
    return int(values[int(np.argmax(counts))])


def observed_half_life_lambda(curve: Curve, lambdas: tuple[float, ...]) -> float:
    slow_multiplier, fast_lambda = lambdas
    interval = modal_observation_interval(curve)
    one_observation_lambda = math.log(2.0) / (PEAK_LR * interval)
    slow_lambda = one_observation_lambda / slow_multiplier
    return slow_lambda + (fast_lambda - slow_lambda) * drop_concentration(curve)


def candidate_response_lambda(curve: Curve, candidate: Candidate) -> float:
    if candidate.kind in {"adaptive_raw", "adaptive_projected_raw"}:
        return adaptive_lambda(curve, candidate.lambdas)
    if candidate.kind in {
        "adaptive_observed_projected_raw",
        "adaptive_observed_localized_projected_raw",
        "adaptive_observed_sqrtlocalized_projected_raw",
    }:
        return observed_half_life_lambda(curve, candidate.lambdas)
    raise ValueError(f"candidate has no adaptive lambda: {candidate.name}")


def drop_localization_factor(curve: Curve) -> float:
    eta = curve.lrs.astype(np.float64)
    drop = np.zeros_like(eta)
    drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0)
    idx = np.flatnonzero(drop > 1e-18)
    if idx.size == 0:
        return 0.0
    support_span = int(idx[-1] - idx[0] + 2)
    post_warmup = max(len(eta) - WARMUP, 1)
    return max(0.0, 1.0 - support_span / post_warmup)


def candidate_localization_factor(curve: Curve, candidate: Candidate) -> float:
    base = drop_localization_factor(curve)
    if candidate.kind == "adaptive_observed_sqrtlocalized_projected_raw":
        return math.sqrt(base)
    if candidate.kind == "adaptive_observed_localized_projected_raw":
        return base
    return 1.0


def mpl_slope_features(curve: Curve, baseline: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Positive local MPL decrease per cumulative-LR unit."""
    s_time = np.cumsum(curve.lrs)[curve.step]
    grad = np.gradient(baseline, s_time, edge_order=1)
    slope = np.maximum(-grad, 0.0)
    positive = slope[slope > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    return slope, slope / max(scale, 1e-18)


def build_cache() -> dict[tuple[str, str], CurvePack]:
    cache: dict[tuple[str, str], CurvePack] = {}
    curves = [(TRAIN_CURVE, "Cosine")] + TARGETS
    for scale in SCALES:
        params = MPL_PRECOMPUTED_INIT[scale]
        for curve_name, _ in curves:
            curve = load_curve(scale, curve_name)
            baseline = mpl_predict(params, curve)
            residual = curve.loss - baseline
            slope_raw, slope_norm = mpl_slope_features(curve, baseline)
            ld_basis, dlogc_basis = mpl_sensitivity_features(params, curve)
            cache[(scale, curve_name)] = CurvePack(
                curve=curve,
                baseline=baseline,
                residual=residual,
                base_mae=mae(curve.loss, baseline),
                slope_raw=slope_raw,
                slope_norm=slope_norm,
                ld_basis=ld_basis,
                dlogc_basis=dlogc_basis,
            )
    return cache


def feature_matrix(pack: CurvePack, candidate: Candidate) -> np.ndarray:
    if candidate.kind in {
        "adaptive_raw",
        "adaptive_projected_raw",
        "adaptive_observed_projected_raw",
        "adaptive_observed_localized_projected_raw",
        "adaptive_observed_sqrtlocalized_projected_raw",
    }:
        response_lambda = candidate_response_lambda(pack.curve, candidate)
        feature = causal_drop_response(pack.curve, response_lambda)
        feature = candidate_localization_factor(pack.curve, candidate) * feature
        return feature[:, None]
    if candidate.kind == "mpl_ld_b":
        return pack.ld_basis[:, None]
    if candidate.kind == "mpl_logc":
        return pack.dlogc_basis[:, None]
    cols = []
    for response_lambda in candidate.lambdas:
        phi = causal_drop_response(pack.curve, response_lambda)
        if candidate.kind == "raw":
            cols.append(phi)
        elif candidate.kind == "lag_raw":
            cols.append(phi * pack.slope_raw)
        elif candidate.kind == "lag_norm":
            cols.append(phi * pack.slope_norm)
        else:
            raise ValueError(f"unknown candidate kind: {candidate.kind}")
    return np.column_stack(cols)


def fit_nonnegative_ridge(
    residual: np.ndarray,
    features: np.ndarray,
    steps: np.ndarray,
    *,
    fit_start: int,
    nuisance_lambda: float,
    max_mode: int,
    ridge_tau: float,
    signed: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    mask = steps >= fit_start
    x = features[mask]
    y = residual[mask]
    q = dct_basis(len(y), max_mode)
    x_o = np.column_stack([soft_residualize(x[:, j], q, nuisance_lambda) for j in range(x.shape[1])])
    y_o = soft_residualize(y, q, nuisance_lambda)
    ridge = (ridge_tau * ridge_tau) * np.eye(x.shape[1])
    gram = x_o.T @ x_o + ridge
    rhs = x_o.T @ y_o

    if signed:
        try:
            best_coef = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            best_coef = np.linalg.lstsq(gram, rhs, rcond=None)[0]
        diff = x_o @ best_coef - y_o
        best_obj = float(np.dot(diff, diff) + best_coef @ ridge @ best_coef)
        denom = float(np.linalg.norm(x_o @ best_coef) * np.linalg.norm(y_o))
        corr = float(np.dot(x_o @ best_coef, y_o) / denom) if denom > 1e-18 else 0.0
        return best_coef, {"fit_objective": best_obj, "residualized_corr": corr}

    best_obj = float("inf")
    best_coef = np.zeros(x.shape[1], dtype=np.float64)
    for active_bits in range(1 << x.shape[1]):
        active = [idx for idx in range(x.shape[1]) if active_bits & (1 << idx)]
        coef = np.zeros(x.shape[1], dtype=np.float64)
        if active:
            try:
                sol = np.linalg.solve(gram[np.ix_(active, active)], rhs[active])
            except np.linalg.LinAlgError:
                continue
            if np.any(sol < -1e-12):
                continue
            coef[active] = np.maximum(sol, 0.0)
        diff = x_o @ coef - y_o
        obj = float(np.dot(diff, diff) + coef @ ridge @ coef)
        if obj < best_obj:
            best_obj = obj
            best_coef = coef

    denom = float(np.linalg.norm(x_o) * np.linalg.norm(y_o))
    corr = float(np.dot(x_o @ best_coef, y_o) / denom) if denom > 1e-18 else 0.0
    return best_coef, {"fit_objective": best_obj, "residualized_corr": corr}


def aggregate(rows: list[dict[str, object]], key: str = "delta_pct") -> dict[str, object]:
    deltas = np.array([float(row[key]) for row in rows], dtype=np.float64)
    return {
        "rows": len(rows),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "worst_delta": float(np.max(deltas)),
        "wins": int(np.sum(deltas < 0.0)),
        "nonharm": int(np.sum(deltas <= 1e-12)),
    }


def config_key(row: dict[str, object]) -> tuple[str, int, float]:
    return (str(row["candidate"]), int(row["fit_start"]), float(row["nuisance_lambda"]))


def holdout_summary(detail_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    def select(dev_rows: list[dict[str, object]]) -> tuple[tuple[str, int, float], dict[str, object]]:
        grouped: dict[tuple[str, int, float], list[dict[str, object]]] = {}
        for row in dev_rows:
            grouped.setdefault(config_key(row), []).append(row)
        candidates = []
        for cfg, rows in grouped.items():
            metrics_row = aggregate(rows)
            safe = int(metrics_row["wins"]) == int(metrics_row["rows"]) and int(metrics_row["nonharm"]) == int(metrics_row["rows"])
            candidates.append((0 if safe else 1, float(metrics_row["mean_delta"]), float(metrics_row["worst_delta"]), cfg, metrics_row))
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        return candidates[0][3], candidates[0][4]

    splits: list[tuple[str, object, object]] = [
        (
            "dev_sharp_linear__test_wsdcon",
            lambda row: "WSD-con" not in str(row["test_label"]),
            lambda row: "WSD-con" in str(row["test_label"]),
        ),
        (
            "dev_wsdcon__test_sharp_linear",
            lambda row: "WSD-con" in str(row["test_label"]),
            lambda row: "WSD-con" not in str(row["test_label"]),
        ),
    ]
    for _, target_label in TARGETS:
        splits.append(
            (
                f"leave_target__{target_label}",
                lambda row, label=target_label: row["test_label"] != label,
                lambda row, label=target_label: row["test_label"] == label,
            )
        )
    for scale in SCALES:
        splits.append(
            (
                f"leave_scale__{scale}M",
                lambda row, heldout=scale: row["scale"] != heldout,
                lambda row, heldout=scale: row["scale"] == heldout,
            )
        )

    rows_out: list[dict[str, object]] = []
    for split, dev_filter, test_filter in splits:
        dev_rows = [row for row in detail_rows if dev_filter(row)]
        test_rows = [row for row in detail_rows if test_filter(row)]
        cfg, dev_metrics = select(dev_rows)
        selected_test = [row for row in test_rows if config_key(row) == cfg]
        test_metrics = aggregate(selected_test)
        rows_out.append(
            {
                "split": split,
                "candidate": cfg[0],
                "fit_start": cfg[1],
                "nuisance_lambda": cfg[2],
                "dev_mean_delta": dev_metrics["mean_delta"],
                "dev_worst_delta": dev_metrics["worst_delta"],
                "dev_wins": dev_metrics["wins"],
                "dev_rows": dev_metrics["rows"],
                "test_mean_delta": test_metrics["mean_delta"],
                "test_worst_delta": test_metrics["worst_delta"],
                "test_wins": test_metrics["wins"],
                "test_rows": test_metrics["rows"],
            }
        )
    return rows_out


def run_audit() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    cache = build_cache()
    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    coef_rows: list[dict[str, object]] = []

    feature_cache = {
        (scale, curve_name, candidate.name): feature_matrix(pack, candidate)
        for (scale, curve_name), pack in cache.items()
        for candidate in CANDIDATES
    }

    for candidate in CANDIDATES:
        for fit_start in FIT_STARTS:
            for nuisance_lambda in MU_VALUES:
                rows: list[dict[str, object]] = []
                for scale in SCALES:
                    source = cache[(scale, TRAIN_CURVE)]
                    if candidate.kind in {
                        "adaptive_projected_raw",
                        "adaptive_observed_projected_raw",
                        "adaptive_observed_localized_projected_raw",
                        "adaptive_observed_sqrtlocalized_projected_raw",
                    }:
                        for target_curve, target_label in TARGETS:
                            target = cache[(scale, target_curve)]
                            response_lambda = candidate_response_lambda(target.curve, candidate)
                            source_features = causal_drop_response(source.curve, response_lambda)[:, None]
                            target_features_raw = causal_drop_response(target.curve, response_lambda)
                            localization = candidate_localization_factor(target.curve, candidate)
                            target_features = (localization * target_features_raw)[:, None]
                            coef, fit_info = fit_nonnegative_ridge(
                                source.residual,
                                source_features,
                                source.curve.step,
                                fit_start=fit_start,
                                nuisance_lambda=nuisance_lambda,
                                max_mode=DCT_MODES,
                                ridge_tau=RIDGE_TAU,
                                signed=False,
                            )
                            coef_rows.append(
                                {
                                    "candidate": candidate.name,
                                    "kind": candidate.kind,
                                    "lambdas": ",".join(f"{value:g}" for value in candidate.lambdas),
                                    "signed": int(candidate.signed),
                                    "fit_start": fit_start,
                                    "nuisance_lambda": nuisance_lambda,
                                    "scale": scale,
                                    "test_curve": target_curve,
                                    "test_label": target_label,
                                    "target_lambda": response_lambda,
                                    "localization": localization,
                                    **{f"coef_{idx}": float(value) for idx, value in enumerate(coef)},
                                    **fit_info,
                                }
                            )
                            correction = target_features @ coef
                            pred = target.baseline + correction
                            corr_mae = mae(target.curve.loss, pred)
                            row = {
                                "candidate": candidate.name,
                                "kind": candidate.kind,
                                "lambdas": ",".join(f"{value:g}" for value in candidate.lambdas),
                                "signed": int(candidate.signed),
                                "fit_start": fit_start,
                                "nuisance_lambda": nuisance_lambda,
                                "scale": scale,
                                "test_curve": target_curve,
                                "test_label": target_label,
                                "target_lambda": response_lambda,
                                "localization": localization,
                                "base_mae": target.base_mae,
                                "corr_mae": corr_mae,
                                "delta_pct": 100.0 * (corr_mae / target.base_mae - 1.0),
                                "win": int(corr_mae < target.base_mae),
                            }
                            rows.append(row)
                            detail_rows.append(row)
                        continue
                    coef, fit_info = fit_nonnegative_ridge(
                        source.residual,
                        feature_cache[(scale, TRAIN_CURVE, candidate.name)],
                        source.curve.step,
                        fit_start=fit_start,
                        nuisance_lambda=nuisance_lambda,
                        max_mode=DCT_MODES,
                        ridge_tau=RIDGE_TAU,
                        signed=candidate.signed,
                    )
                    coef_rows.append(
                        {
                            "candidate": candidate.name,
                            "kind": candidate.kind,
                            "lambdas": ",".join(f"{value:g}" for value in candidate.lambdas),
                            "signed": int(candidate.signed),
                            "fit_start": fit_start,
                            "nuisance_lambda": nuisance_lambda,
                            "scale": scale,
                            **{f"coef_{idx}": float(value) for idx, value in enumerate(coef)},
                            **fit_info,
                        }
                    )
                    for target_curve, target_label in TARGETS:
                        target = cache[(scale, target_curve)]
                        correction = feature_cache[(scale, target_curve, candidate.name)] @ coef
                        pred = target.baseline + correction
                        corr_mae = mae(target.curve.loss, pred)
                        row = {
                            "candidate": candidate.name,
                            "kind": candidate.kind,
                            "lambdas": ",".join(f"{value:g}" for value in candidate.lambdas),
                            "signed": int(candidate.signed),
                            "fit_start": fit_start,
                            "nuisance_lambda": nuisance_lambda,
                            "scale": scale,
                            "test_curve": target_curve,
                            "test_label": target_label,
                            "base_mae": target.base_mae,
                            "corr_mae": corr_mae,
                            "delta_pct": 100.0 * (corr_mae / target.base_mae - 1.0),
                            "win": int(corr_mae < target.base_mae),
                        }
                        rows.append(row)
                        detail_rows.append(row)
                summary_rows.append(
                    {
                        "candidate": candidate.name,
                        "kind": candidate.kind,
                        "lambdas": ",".join(f"{value:g}" for value in candidate.lambdas),
                        "signed": int(candidate.signed),
                        "fit_start": fit_start,
                        "nuisance_lambda": nuisance_lambda,
                        "dct_modes": DCT_MODES,
                        "ridge_tau": RIDGE_TAU,
                        **aggregate(rows),
                    }
                )
    summary_rows.sort(key=lambda row: (int(row["wins"]) != int(row["rows"]), float(row["mean_delta"]), float(row["worst_delta"])))
    return summary_rows, detail_rows, coef_rows


def write_report(
    summary_rows: list[dict[str, object]],
    detail_rows: list[dict[str, object]],
    holdout_rows: list[dict[str, object]],
) -> None:
    best = summary_rows[0]
    selected = [
        row
        for row in detail_rows
        if row["candidate"] == best["candidate"]
        and int(row["fit_start"]) == int(best["fit_start"])
        and abs(float(row["nuisance_lambda"]) - float(best["nuisance_lambda"])) < 1e-12
    ]
    target_rows = []
    for _, target_label in TARGETS:
        sub = [row for row in selected if row["test_label"] == target_label]
        target_rows.append({"test_label": target_label, **aggregate(sub)})
    localized_name = "obs_half_life_sqrtlocalized_projected_2p5_roundfast20"
    localized_rows = [
        row
        for row in detail_rows
        if row["candidate"] == localized_name
        and int(row["fit_start"]) == 8000
        and abs(float(row["nuisance_lambda"]) - 0.01) < 1e-12
    ]
    localized_metrics = aggregate(localized_rows) if localized_rows else None

    lines = [
        "# Interpretable Error Model Audit\n\n",
        "This audit intentionally evaluates a small fixed set of mechanism candidates. "
        "All coefficients are fit from `cosine_72000.csv` residuals only; WSD-family curves are evaluation only.\n\n",
        "## Interpretability Reset\n\n",
        "The numerically best WSD-only row below is not automatically the recommended research-facing formula. "
        "After the later interpretability repair, the current mechanism-facing main candidate is "
        "`MPL-LD tangent + one causal LR-drop response`, documented in `MODEL_DECISION.md`. "
        "The DCT-projected rows in this report are useful numerical references and ablations, not the core theory. "
        "`sqrt-localized` is retained as an ablation rather than the main formula because its square-root amplitude explanation is weaker.\n\n",
        "## Best WSD-Only Candidate\n\n",
        f"- Candidate: `{best['candidate']}`.\n",
        f"- Mean / worst: `{float(best['mean_delta']):+.2f}%` / `{float(best['worst_delta']):+.2f}%`.\n",
        f"- Wins/non-harm: `{int(best['wins'])}/{int(best['rows'])}` / `{int(best['nonharm'])}/{int(best['rows'])}`.\n",
        f"- Fit start: `{int(best['fit_start'])}`, nuisance lambda: `{float(best['nuisance_lambda']):g}`.\n\n",
        "## Candidate Summary\n\n",
        "| candidate | kind | lambdas | fit start | mu | mean | worst | wins |\n",
        "|---|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in summary_rows[:20]:
        lines.append(
            f"| {row['candidate']} | {row['kind']} | {row['lambdas']} | {int(row['fit_start'])} | "
            f"{float(row['nuisance_lambda']):g} | {float(row['mean_delta']):+.2f}% | "
            f"{float(row['worst_delta']):+.2f}% | {int(row['wins'])}/{int(row['rows'])} |\n"
        )
    lines += [
        "\n## Per-Target Result For Best Candidate\n\n",
        "| target | mean delta | worst scale | wins |\n",
        "|---|---:|---:|---:|\n",
    ]
    for row in target_rows:
        lines.append(
            f"| {row['test_label']} | {float(row['mean_delta']):+.2f}% | "
            f"{float(row['worst_delta']):+.2f}% | {int(row['wins'])}/{int(row['rows'])} |\n"
        )
    if localized_metrics is not None:
        lines += [
            "\n## Localized Control-Safety Ablation\n\n",
            "`obs_half_life_projected_2p5_roundfast20` is the WSD-only upper variant.  "
            "`obs_half_life_sqrtlocalized_projected_2p5_roundfast20` multiplies the transferred correction by the square root of a continuous LR-drop localization factor, "
            "which removes the short-cosine control failure without adding fitted parameters.  It is useful evidence, but no longer recommended as the main formula.\n\n",
            "| variant | mean | worst | wins |\n",
            "|---|---:|---:|---:|\n",
            f"| sqrt-localized ablation | {float(localized_metrics['mean_delta']):+.2f}% | "
            f"{float(localized_metrics['worst_delta']):+.2f}% | "
            f"{int(localized_metrics['wins'])}/{int(localized_metrics['rows'])} |\n",
            "\nSee `results/interpretable_strict_vs_rounded/REPORT.md` for the constant and short-cosine control audit.\n",
        ]
    lines += [
        "\n## Holdout Selection Check\n\n",
        "| split | selected candidate | dev mean | dev worst | test mean | test worst | test wins |\n",
        "|---|---|---:|---:|---:|---:|---:|\n",
    ]
    for row in holdout_rows:
        candidate = f"{row['candidate']} / start={int(row['fit_start'])}, mu={float(row['nuisance_lambda']):g}"
        lines.append(
            f"| {row['split']} | `{candidate}` | "
            f"{float(row['dev_mean_delta']):+.2f}% | {float(row['dev_worst_delta']):+.2f}% | "
            f"{float(row['test_mean_delta']):+.2f}% | {float(row['test_worst_delta']):+.2f}% | "
            f"{int(row['test_wins'])}/{int(row['test_rows'])} |\n"
        )
    lines += [
        "\n## Reading\n\n",
        "- This file is now a historical fixed-candidate audit.  Use `MODEL_DECISION.md` for the current interpretable formula.\n",
        "- `raw_drop` is the minimal causal LR-drop response baseline inside the old DCT-projected audit.\n",
        "- `obs_half_life_projected` replaces the unexplained exact `7/20` endpoints with observable response half-life anchors.  The modal loss-curve interval is 128 steps; the slow endpoint is a 2.5-interval half-life, and the fast endpoint is rounded to the one-interval response rate.\n",
        "- The target LR schedule determines a continuous response rate through drop concentration, and the matching response operator is calibrated only on cosine residuals.\n",
        "- The projected continuous model recovers most of the previous adaptive-fit-window gain while avoiding discrete smooth/step routing.\n",
        "- The sqrt-localized variant adds a parameter-free schedule-locality amplitude factor `sqrt(1 - drop_support_span / post_warmup_span)`; it is now treated as a control-safety ablation rather than the main formula.\n",
        "- DCT nuisance is not mechanism-native enough for the main story.  The stronger current interpretation projects out MPL LR-dependent tangent directions instead.\n",
        "- `lag_rawslope` and unrelated MPL-internal sensitivity directions do not beat the response baseline; they should stay as negative evidence, not main-method components.\n",
        "- The remaining weak points are ridge identifiability, the `2.5` slow response prior, locality as a boundary condition, and external validation.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    summary_rows, detail_rows, coef_rows = run_audit()
    holdout_rows = holdout_summary(detail_rows)
    write_csv(OUT_DIR / "summary.csv", summary_rows)
    write_csv(OUT_DIR / "details.csv", detail_rows)
    write_csv(OUT_DIR / "coefficients.csv", coef_rows)
    write_csv(OUT_DIR / "holdout_summary.csv", holdout_rows)
    write_report(summary_rows, detail_rows, holdout_rows)
    print(f"wrote {OUT_DIR / 'summary.csv'}")
    print(f"wrote {OUT_DIR / 'details.csv'}")
    print(f"wrote {OUT_DIR / 'coefficients.csv'}")
    print(f"wrote {OUT_DIR / 'holdout_summary.csv'}")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
