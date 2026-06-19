#!/usr/bin/env python3
"""Robustness audits for the projected cosine-kappa response model.

The script keeps the deployable prediction rule fixed:

    L_hat_s(t) = L_MPL,s(t) + kappa_hat_s * phi_s(t)

All deployable coefficients are fitted from source cosine residuals only.
Target losses are used only for evaluation, or explicitly marked as oracle
diagnostics.
"""
from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path

TMP_CACHE = Path(os.environ.get("TMPDIR", "/tmp")) / "dl-final-cache"
TMP_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(TMP_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(TMP_CACHE / "xdg"))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import interpretable_error_model as iem  # noqa: E402
import interpretable_nuisance_origin_audit as noa  # noqa: E402
import interpretable_observation_bracket_audit as oba  # noqa: E402

OUT_DIR = iem.ROOT / "results" / "schedule_response_robustness"
FIG_DIR = OUT_DIR / "figs"
SLIDE_FIG_DIR = iem.ROOT / "slides" / "figs"
FIT_STARTS = [5000, 6500, 8000, 10000, 12000]
FIT_START = 8000
CORE_TARGETS = [(curve, label) for group, curve, label in noa.ALL_TARGETS if group == "core_wsd"]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def positive_drops(curve: iem.Curve, signed: bool = False) -> np.ndarray:
    eta = curve.lrs.astype(np.float64)
    out = np.zeros_like(eta)
    if signed:
        out[1:] = eta[:-1] - eta[1:]
    else:
        out[1:] = np.maximum(eta[:-1] - eta[1:], 0.0)
    return out


def q2_drop(curve: iem.Curve) -> float:
    drop = positive_drops(curve)
    total = float(np.sum(drop))
    if total <= 1e-18:
        return 0.0
    p = drop[drop > 1e-18] / total
    return float(np.sum(p * p))


def lambda_obs(curve: iem.Curve) -> float:
    return math.log(2.0) / (iem.PEAK_LR * iem.modal_observation_interval(curve))


def lambda_q2_halflife(curve: iem.Curve) -> float:
    q = q2_drop(curve)
    return lambda_obs(curve) / max(2.0 - q, 1e-12)


def half_life_steps_q2(curve: iem.Curve) -> float:
    return (2.0 - q2_drop(curve)) * iem.modal_observation_interval(curve)


def load_pack(scale: str, curve_name: str, cache: dict[tuple[str, str], iem.CurvePack]) -> iem.CurvePack:
    key = (scale, curve_name)
    if key not in cache:
        cache[key] = noa.load_pack(scale, curve_name)
    return cache[key]


def lr_time_exp_feature(curve: iem.Curve, lam: float, signed: bool = False) -> np.ndarray:
    eta = curve.lrs.astype(np.float64)
    force = positive_drops(curve, signed=signed)
    out = np.empty_like(eta)
    acc = 0.0
    for t in range(len(eta)):
        acc = acc * math.exp(-lam * float(eta[t])) + force[t]
        out[t] = acc
    return (out / iem.PEAK_LR)[curve.step]


def step_time_exp_feature(curve: iem.Curve, half_life_steps: float) -> np.ndarray:
    drop = positive_drops(curve)
    decay = math.exp(-math.log(2.0) / max(half_life_steps, 1e-12))
    out = np.empty_like(drop)
    acc = 0.0
    for t in range(len(drop)):
        acc = acc * decay + drop[t]
        out[t] = acc
    return (out / iem.PEAK_LR)[curve.step]


def power_law_lr_time_feature(curve: iem.Curve, half_life_steps: float) -> np.ndarray:
    eta = curve.lrs.astype(np.float64)
    drop = positive_drops(curve)
    idx = np.flatnonzero(drop > 1e-18)
    if idx.size == 0:
        return np.zeros_like(curve.step, dtype=np.float64)
    cumlr = np.cumsum(eta)
    scale = iem.PEAK_LR * max(half_life_steps, 1e-12)
    vals: list[float] = []
    for step in curve.step:
        active = idx[idx <= step]
        if active.size == 0:
            vals.append(0.0)
            continue
        dt = np.maximum(cumlr[step] - cumlr[active], 0.0)
        vals.append(float(np.sum(drop[active] / (1.0 + dt / scale))))
    return np.array(vals, dtype=np.float64) / iem.PEAK_LR


def feature(curve: iem.Curve, kind: str, param: float) -> np.ndarray:
    if kind == "lr_time_exp":
        return lr_time_exp_feature(curve, param)
    if kind == "signed_lr_time_exp":
        return lr_time_exp_feature(curve, param, signed=True)
    if kind == "step_time_exp":
        return step_time_exp_feature(curve, param)
    if kind == "power_law_lr_time":
        return power_law_lr_time_feature(curve, param)
    if kind == "eta_level":
        return curve.lrs[curve.step] / iem.PEAK_LR
    if kind == "drop_cumsum":
        return np.cumsum(positive_drops(curve))[curve.step] / iem.PEAK_LR
    if kind == "drop_impulse":
        return positive_drops(curve)[curve.step] / iem.PEAK_LR
    raise ValueError(f"unknown feature kind: {kind}")


def tangent_basis(
    source: iem.CurvePack,
    fit_start: int,
    basis_cache: dict[tuple[str, int], np.ndarray],
) -> np.ndarray:
    key = (source.curve.scale, fit_start)
    if key not in basis_cache:
        basis_cache[key] = oba.tangent_basis(source, "mpl_ld4", fit_start)
    return basis_cache[key]


def residualized_source_pair(
    source: iem.CurvePack,
    phi_source: np.ndarray,
    fit_start: int,
    basis_cache: dict[tuple[str, int], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, int]:
    mask = source.curve.step >= fit_start
    x = phi_source[mask].astype(np.float64)
    y = source.residual[mask].astype(np.float64)
    q = tangent_basis(source, fit_start, basis_cache)
    if q.size:
        x = x - q @ (q.T @ x)
        y = y - q @ (q.T @ y)
    return x, y, len(x)


def fit_kappa_from_sources(
    train_scales: list[str],
    source_feature_by_scale: dict[str, np.ndarray],
    cache: dict[tuple[str, str], iem.CurvePack],
    basis_cache: dict[tuple[str, int], np.ndarray],
    fit_start: int = FIT_START,
) -> tuple[float, dict[str, float]]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    n_total = 0
    for scale in train_scales:
        source = load_pack(scale, iem.TRAIN_CURVE, cache)
        x, y, n = residualized_source_pair(source, source_feature_by_scale[scale], fit_start, basis_cache)
        xs.append(x)
        ys.append(y)
        n_total += n
    x_all = np.concatenate(xs)
    y_all = np.concatenate(ys)
    dot = max(0.0, float(np.dot(x_all, y_all)))
    energy = float(np.dot(x_all, x_all))
    ridge = 1.0 / max(n_total, 1)
    return dot / max(energy + ridge, 1e-18), {
        "source_dot": dot,
        "source_energy": energy,
        "ridge": ridge,
        "n_cal": n_total,
    }


def kappa_star(phi: np.ndarray, residual: np.ndarray) -> float:
    denom = float(np.dot(phi, phi))
    if denom <= 1e-18:
        return 0.0
    return max(0.0, float(np.dot(phi, residual)) / denom)


def corr(x: np.ndarray, y: np.ndarray) -> float:
    xx = x.astype(np.float64) - float(np.mean(x))
    yy = y.astype(np.float64) - float(np.mean(y))
    denom = float(np.linalg.norm(xx) * np.linalg.norm(yy))
    if denom <= 1e-18:
        return float("nan")
    return float(np.dot(xx, yy) / denom)


def summarize(rows: list[dict[str, object]], group_key: str, value_key: str = "delta_pct") -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    keys = list(dict.fromkeys(str(row[group_key]) for row in rows))
    for key in keys:
        sub = [row for row in rows if str(row[group_key]) == key]
        deltas = np.array([float(row[value_key]) for row in sub], dtype=np.float64)
        hat = np.array([float(row["kappa_hat"]) for row in sub if "kappa_hat" in row], dtype=np.float64)
        star = np.array([float(row["kappa_star"]) for row in sub if "kappa_star" in row], dtype=np.float64)
        out.append(
            {
                group_key: key,
                "rows": len(sub),
                "mean_delta_pct": float(np.mean(deltas)),
                "median_delta_pct": float(np.median(deltas)),
                "worst_delta_pct": float(np.max(deltas)),
                "wins": int(np.sum(deltas < 0.0)),
                "pearson_kappa_star": corr(hat, star) if len(hat) == len(star) and len(hat) > 1 else float("nan"),
                "mean_kappa_hat": float(np.mean(hat)) if len(hat) else float("nan"),
                "mean_kappa_star": float(np.mean(star)) if len(star) else float("nan"),
            }
        )
    return out


def score_one(
    test_scale: str,
    curve_name: str,
    phi_target: np.ndarray,
    kappa: float,
    cache: dict[tuple[str, str], iem.CurvePack],
) -> tuple[float, float, float, float]:
    target = load_pack(test_scale, curve_name, cache)
    pred = target.baseline + kappa * phi_target
    corr_mae = iem.mae(target.curve.loss, pred)
    delta = 100.0 * (corr_mae / target.base_mae - 1.0)
    terminal_mpl_error = float(target.baseline[-1] - target.curve.loss[-1])
    terminal_corr_error = float(pred[-1] - target.curve.loss[-1])
    return corr_mae, delta, terminal_mpl_error, terminal_corr_error


def lambda_sensitivity(
    cache: dict[tuple[str, str], iem.CurvePack],
    basis_cache: dict[tuple[str, int], np.ndarray],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    grid_multipliers = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]

    for scale in iem.SCALES:
        for curve_name, label in CORE_TARGETS:
            target = load_pack(scale, curve_name, cache)
            obs = lambda_obs(target.curve)
            variants = [
                ("q2_half_life", lambda_q2_halflife(target.curve), 0),
                ("fixed_1obs", obs, 0),
                ("fixed_2obs", obs / 2.0, 0),
                ("fixed_4obs", obs / 4.0, 0),
                ("wrong_fast_lambda20", 20.0, 0),
            ]
            oracle_candidates: list[dict[str, object]] = []
            for mult in grid_multipliers:
                lam = obs / mult
                source_features = {
                    s: feature(load_pack(s, iem.TRAIN_CURVE, cache).curve, "lr_time_exp", lam)
                    for s in iem.SCALES
                }
                khat, info = fit_kappa_from_sources([scale], source_features, cache, basis_cache)
                phi_t = feature(target.curve, "lr_time_exp", lam)
                corr_mae, delta, terminal_mpl, terminal_corr = score_one(scale, curve_name, phi_t, khat, cache)
                oracle_candidates.append(
                    {
                        "lambda_rule": f"oracle_grid_{mult:g}obs",
                        "lambda": lam,
                        "uses_target_loss_for_lambda": 1,
                        "kappa_hat": khat,
                        "phi_target": phi_t,
                        "corr_mae": corr_mae,
                        "delta_pct": delta,
                        "terminal_mpl_error": terminal_mpl,
                        "terminal_corr_error": terminal_corr,
                        **info,
                    }
                )
            best = min(oracle_candidates, key=lambda row: float(row["corr_mae"]))
            variants.append(("oracle_grid_best", float(best["lambda"]), 1))

            for name, lam, uses_target_loss in variants:
                source_features = {
                    s: feature(load_pack(s, iem.TRAIN_CURVE, cache).curve, "lr_time_exp", lam)
                    for s in iem.SCALES
                }
                khat, info = fit_kappa_from_sources([scale], source_features, cache, basis_cache)
                phi_t = feature(target.curve, "lr_time_exp", lam)
                corr_mae, delta, terminal_mpl, terminal_corr = score_one(scale, curve_name, phi_t, khat, cache)
                rows.append(
                    {
                        "lambda_rule": name,
                        "uses_target_loss_for_lambda": uses_target_loss,
                        "scale": scale,
                        "target_curve": curve_name,
                        "target_label": label,
                        "lambda": lam,
                        "kappa_hat": khat,
                        "kappa_star": kappa_star(phi_t, target.residual),
                        "base_mae": target.base_mae,
                        "corr_mae": corr_mae,
                        "delta_pct": delta,
                        "terminal_mpl_error": terminal_mpl,
                        "terminal_corr_error": terminal_corr,
                        **info,
                    }
                )
    return rows, summarize(rows, "lambda_rule")


def kernel_ablation(
    cache: dict[tuple[str, str], iem.CurvePack],
    basis_cache: dict[tuple[str, int], np.ndarray],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for scale in iem.SCALES:
        for curve_name, label in CORE_TARGETS:
            target = load_pack(scale, curve_name, cache)
            lam = lambda_q2_halflife(target.curve)
            hsteps = half_life_steps_q2(target.curve)
            variants = [
                ("lr_time_exp", "lr_time_exp", lam),
                ("eta_level", "eta_level", 0.0),
                ("drop_cumsum", "drop_cumsum", 0.0),
                ("drop_impulse", "drop_impulse", 0.0),
                ("step_time_exp", "step_time_exp", hsteps),
                ("power_law_lr_time", "power_law_lr_time", hsteps),
                ("signed_lr_time_exp", "signed_lr_time_exp", lam),
            ]
            for variant, kind, param in variants:
                source_features = {
                    s: feature(load_pack(s, iem.TRAIN_CURVE, cache).curve, kind, param)
                    for s in iem.SCALES
                }
                khat, info = fit_kappa_from_sources([scale], source_features, cache, basis_cache)
                phi_t = feature(target.curve, kind, param)
                corr_mae, delta, terminal_mpl, terminal_corr = score_one(scale, curve_name, phi_t, khat, cache)
                rows.append(
                    {
                        "kernel": variant,
                        "scale": scale,
                        "target_curve": curve_name,
                        "target_label": label,
                        "param": param,
                        "kappa_hat": khat,
                        "kappa_star": kappa_star(phi_t, target.residual),
                        "base_mae": target.base_mae,
                        "corr_mae": corr_mae,
                        "delta_pct": delta,
                        "terminal_mpl_error": terminal_mpl,
                        "terminal_corr_error": terminal_corr,
                        **info,
                    }
                )
    return rows, summarize(rows, "kernel")


def cross_scale_transfer(
    cache: dict[tuple[str, str], iem.CurvePack],
    basis_cache: dict[tuple[str, int], np.ndarray],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for test_scale in iem.SCALES:
        for curve_name, label in CORE_TARGETS:
            target = load_pack(test_scale, curve_name, cache)
            lam = lambda_q2_halflife(target.curve)
            source_features = {
                s: feature(load_pack(s, iem.TRAIN_CURVE, cache).curve, "lr_time_exp", lam)
                for s in iem.SCALES
            }
            phi_t = feature(target.curve, "lr_time_exp", lam)
            protocols = [
                ("same_scale", [test_scale]),
                ("leave_one_scale_out_pooled", [s for s in iem.SCALES if s != test_scale]),
                ("all_scale_pooled", list(iem.SCALES)),
            ]
            for train_scale in iem.SCALES:
                if train_scale != test_scale:
                    protocols.append((f"single_source_{train_scale}M", [train_scale]))
            for protocol, train_scales in protocols:
                khat, info = fit_kappa_from_sources(train_scales, source_features, cache, basis_cache)
                corr_mae, delta, terminal_mpl, terminal_corr = score_one(test_scale, curve_name, phi_t, khat, cache)
                rows.append(
                    {
                        "protocol": protocol,
                        "train_scales": ",".join(train_scales),
                        "test_scale": test_scale,
                        "target_curve": curve_name,
                        "target_label": label,
                        "lambda": lam,
                        "kappa_hat": khat,
                        "kappa_star": kappa_star(phi_t, target.residual),
                        "base_mae": target.base_mae,
                        "corr_mae": corr_mae,
                        "delta_pct": delta,
                        "terminal_mpl_error": terminal_mpl,
                        "terminal_corr_error": terminal_corr,
                        **info,
                    }
                )
            aggregate_protocols = [
                ("leave_one_scale_out_mean_kappa", [s for s in iem.SCALES if s != test_scale], "mean"),
                ("leave_one_scale_out_median_kappa", [s for s in iem.SCALES if s != test_scale], "median"),
                ("all_scale_mean_kappa", list(iem.SCALES), "mean"),
                ("all_scale_median_kappa", list(iem.SCALES), "median"),
            ]
            for protocol, train_scales, reducer in aggregate_protocols:
                kappas = []
                infos = []
                for train_scale in train_scales:
                    khat_i, info_i = fit_kappa_from_sources([train_scale], source_features, cache, basis_cache)
                    kappas.append(khat_i)
                    infos.append(info_i)
                khat = float(np.mean(kappas)) if reducer == "mean" else float(np.median(kappas))
                corr_mae, delta, terminal_mpl, terminal_corr = score_one(test_scale, curve_name, phi_t, khat, cache)
                rows.append(
                    {
                        "protocol": protocol,
                        "train_scales": ",".join(train_scales),
                        "test_scale": test_scale,
                        "target_curve": curve_name,
                        "target_label": label,
                        "lambda": lam,
                        "kappa_hat": khat,
                        "kappa_star": kappa_star(phi_t, target.residual),
                        "base_mae": target.base_mae,
                        "corr_mae": corr_mae,
                        "delta_pct": delta,
                        "terminal_mpl_error": terminal_mpl,
                        "terminal_corr_error": terminal_corr,
                        "source_dot": float(np.mean([info["source_dot"] for info in infos])),
                        "source_energy": float(np.mean([info["source_energy"] for info in infos])),
                        "ridge": float(np.mean([info["ridge"] for info in infos])),
                        "n_cal": int(np.sum([info["n_cal"] for info in infos])),
                    }
                )
    return rows, summarize(rows, "protocol")


def no_projection_negative_control(
    cache: dict[tuple[str, str], iem.CurvePack],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for scale in iem.SCALES:
        source = load_pack(scale, iem.TRAIN_CURVE, cache)
        source_mask = source.curve.step >= FIT_START
        for curve_name, label in CORE_TARGETS:
            target = load_pack(scale, curve_name, cache)
            lam = lambda_q2_halflife(target.curve)
            phi_source = feature(source.curve, "lr_time_exp", lam)
            x = phi_source[source_mask].astype(np.float64)
            y = source.residual[source_mask].astype(np.float64)
            n_cal = int(np.sum(source_mask))
            dot = max(0.0, float(np.dot(x, y)))
            energy = float(np.dot(x, x))
            ridge = 1.0 / max(n_cal, 1)
            khat = dot / max(energy + ridge, 1e-18)
            phi_t = feature(target.curve, "lr_time_exp", lam)
            corr_mae, delta, terminal_mpl, terminal_corr = score_one(scale, curve_name, phi_t, khat, cache)
            rows.append(
                {
                    "estimator": "direct_no_projection",
                    "scale": scale,
                    "target_curve": curve_name,
                    "target_label": label,
                    "lambda": lam,
                    "kappa_hat": khat,
                    "kappa_star": kappa_star(phi_t, target.residual),
                    "base_mae": target.base_mae,
                    "corr_mae": corr_mae,
                    "delta_pct": delta,
                    "terminal_mpl_error": terminal_mpl,
                    "terminal_corr_error": terminal_corr,
                    "source_dot": dot,
                    "source_energy": energy,
                    "ridge": ridge,
                    "n_cal": n_cal,
                }
            )
    return rows, summarize(rows, "estimator")


def window_audit(
    cache: dict[tuple[str, str], iem.CurvePack],
    basis_cache: dict[tuple[str, int], np.ndarray],
) -> tuple[list[dict[str, object]], list[dict[str, object]], int]:
    rule_rows: list[dict[str, object]] = []
    eval_rows: list[dict[str, object]] = []
    for fit_start in FIT_STARTS:
        retentions: list[float] = []
        floors: list[float] = []
        for scale in iem.SCALES:
            source = load_pack(scale, iem.TRAIN_CURVE, cache)
            obs = lambda_obs(source.curve)
            for lam in [obs / 2.0, obs]:
                phi = feature(source.curve, "lr_time_exp", lam)
                x, _, n = residualized_source_pair(source, phi, fit_start, basis_cache)
                full = float(np.dot(phi[source.curve.step >= fit_start], phi[source.curve.step >= fit_start]))
                retentions.append(float(np.dot(x, x) / max(full, 1e-18)))
                floors.append(1.0 / max(n, 1))
        rule_rows.append(
            {
                "fit_start": fit_start,
                "max_retention": float(np.max(retentions)),
                "median_retention": float(np.median(retentions)),
                "finite_sample_floor": float(np.min(floors)),
                "passes": int(np.max(retentions) <= np.min(floors)),
            }
        )
    selected = next((int(row["fit_start"]) for row in rule_rows if int(row["passes"]) == 1), FIT_START)

    for fit_start in FIT_STARTS:
        for scale in iem.SCALES:
            for curve_name, label in CORE_TARGETS:
                target = load_pack(scale, curve_name, cache)
                lam = lambda_q2_halflife(target.curve)
                source_features = {
                    s: feature(load_pack(s, iem.TRAIN_CURVE, cache).curve, "lr_time_exp", lam)
                    for s in iem.SCALES
                }
                khat, info = fit_kappa_from_sources([scale], source_features, cache, basis_cache, fit_start=fit_start)
                phi_t = feature(target.curve, "lr_time_exp", lam)
                corr_mae, delta, terminal_mpl, terminal_corr = score_one(scale, curve_name, phi_t, khat, cache)
                eval_rows.append(
                    {
                        "fit_start": fit_start,
                        "selected_by_source_rule": int(fit_start == selected),
                        "scale": scale,
                        "target_curve": curve_name,
                        "target_label": label,
                        "kappa_hat": khat,
                        "kappa_star": kappa_star(phi_t, target.residual),
                        "base_mae": target.base_mae,
                        "corr_mae": corr_mae,
                        "delta_pct": delta,
                        "terminal_mpl_error": terminal_mpl,
                        "terminal_corr_error": terminal_corr,
                        **info,
                    }
                )
    return rule_rows, eval_rows, selected


def wsdcon_failure_rows(
    main_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows = [
        row
        for row in main_rows
        if str(row["target_curve"]).startswith("wsdcon") and row.get("protocol", "") == "same_scale"
    ]
    out: list[dict[str, object]] = []
    for row in rows:
        kstar = float(row["kappa_star"])
        khat = float(row["kappa_hat"])
        out.append(
            {
                "scale": row["test_scale"],
                "target_curve": row["target_curve"],
                "target_label": row["target_label"],
                "kappa_hat": khat,
                "kappa_star": kstar,
                "kappa_ratio": khat / kstar if kstar > 1e-18 else float("nan"),
                "delta_pct": row["delta_pct"],
                "terminal_mpl_error": row["terminal_mpl_error"],
                "terminal_corr_error": row["terminal_corr_error"],
                "abs_terminal_mpl_error": abs(float(row["terminal_mpl_error"])),
                "abs_terminal_corr_error": abs(float(row["terminal_corr_error"])),
            }
        )
    return out


def plot_heatmap(main_rows: list[dict[str, object]]) -> None:
    rows = [row for row in main_rows if row.get("protocol") == "same_scale"]
    targets = [curve for curve, _ in CORE_TARGETS]
    labels = [label for _, label in CORE_TARGETS]
    data = np.zeros((len(iem.SCALES), len(targets)), dtype=np.float64)
    for i, scale in enumerate(iem.SCALES):
        for j, curve in enumerate(targets):
            match = [
                row
                for row in rows
                if row["test_scale"] == scale and row["target_curve"] == curve
            ][0]
            data[i, j] = float(match["delta_pct"])
    fig, ax = plt.subplots(figsize=(8.8, 3.2), constrained_layout=True)
    im = ax.imshow(data, cmap="RdBu_r", vmin=-60, vmax=60)
    ax.set_xticks(np.arange(len(targets)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(iem.SCALES)))
    ax.set_yticklabels([f"{s}M" for s in iem.SCALES])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i,j]:.1f}%", ha="center", va="center", fontsize=8)
    ax.set_title("MAE change vs MPL, same-scale WSD-family")
    fig.colorbar(im, ax=ax, label="MAE change (%)")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"dpi": 180, "bbox_inches": "tight", "pad_inches": 0.04}
    fig.savefig(FIG_DIR / "mae_change_heatmap.png", **save_kwargs)
    SLIDE_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(SLIDE_FIG_DIR / "fig_schedule_response_mae_heatmap.png", **save_kwargs)
    plt.close(fig)


def plot_representative_errors(
    cache: dict[tuple[str, str], iem.CurvePack],
    main_rows: list[dict[str, object]],
) -> None:
    same = [row for row in main_rows if row.get("protocol") == "same_scale" and row["test_scale"] == "100"]
    wsdcon = [row for row in same if str(row["target_curve"]).startswith("wsdcon")]
    worst_con = max(wsdcon, key=lambda row: float(row["delta_pct"]))
    chosen = [
        ("wsd_20000_24000.csv", "WSD sharp"),
        ("wsdld_20000_24000.csv", "WSD linear"),
        (str(worst_con["target_curve"]), f"WSD-con worst: {worst_con['target_label']}"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 6.2), constrained_layout=True)
    for col, (curve_name, title) in enumerate(chosen):
        row = [r for r in same if r["target_curve"] == curve_name][0]
        target = load_pack("100", curve_name, cache)
        lam = lambda_q2_halflife(target.curve)
        phi_t = feature(target.curve, "lr_time_exp", lam)
        pred = target.baseline + float(row["kappa_hat"]) * phi_t
        step_k = target.curve.step / 1000.0
        axes[0, col].plot(step_k, target.curve.loss, color="#111827", lw=1.7, label="true")
        axes[0, col].plot(step_k, target.baseline, color="#dc2626", lw=1.2, label="MPL")
        axes[0, col].plot(step_k, pred, color="#2563eb", lw=1.2, label="MPL+kappa phi")
        axes[0, col].set_title(f"{title}\nMAE change {float(row['delta_pct']):+.1f}%")
        axes[0, col].set_xlabel("step (k)")
        axes[0, col].set_ylabel("loss")
        axes[0, col].grid(alpha=0.18)
        axes[1, col].axhline(0, color="#111827", lw=0.8)
        axes[1, col].plot(step_k, target.baseline - target.curve.loss, color="#dc2626", lw=1.2, label="MPL error")
        axes[1, col].plot(step_k, pred - target.curve.loss, color="#2563eb", lw=1.2, label="corrected error")
        axes[1, col].set_xlabel("step (k)")
        axes[1, col].set_ylabel("prediction error")
        axes[1, col].grid(alpha=0.18)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "representative_time_errors_100M.png", dpi=180)
    SLIDE_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(SLIDE_FIG_DIR / "fig_schedule_response_time_errors_100M.png", dpi=180)
    plt.close(fig)


def plot_mpl_residual_anomaly(cache: dict[tuple[str, str], iem.CurvePack]) -> None:
    chosen = [
        ("wsd_20000_24000.csv", "WSD sharp"),
        ("wsdld_20000_24000.csv", "WSD linear"),
        ("wsdcon_18.csv", "WSD-con final LR 18e-5"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.2), constrained_layout=True)
    for ax, (curve_name, title) in zip(axes, chosen):
        target = load_pack("100", curve_name, cache)
        step_k = target.curve.step / 1000.0
        residual = target.baseline - target.curve.loss
        lr = target.curve.lrs[target.curve.step] / max(float(np.max(target.curve.lrs)), 1e-18)
        ax.axhline(0, color="#111827", lw=0.8)
        ax.plot(step_k, residual, color="#dc2626", lw=1.3, label="MPL prediction error")
        ax2 = ax.twinx()
        ax2.plot(step_k, lr, color="#64748b", lw=0.9, alpha=0.55, label="normalized LR")
        ax2.set_ylim(-0.03, 1.05)
        ax2.set_yticks([])
        ax.set_title(title)
        ax.set_xlabel("step (k)")
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("MPL error (prediction - true)")
    axes[0].legend(fontsize=8, loc="lower left")
    axes[0].text(
        0.02,
        0.96,
        "100M targets",
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="#374151",
    )
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "mpl_residual_anomaly_100M.png", dpi=180)
    SLIDE_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(SLIDE_FIG_DIR / "fig_mpl_residual_anomaly_100M.png", dpi=180)
    plt.close(fig)


def _standardize(y: np.ndarray) -> np.ndarray:
    yy = y.astype(np.float64)
    scale = float(np.std(yy))
    if scale <= 1e-18:
        return yy - float(np.mean(yy))
    return (yy - float(np.mean(yy))) / scale


def plot_projection_decomposition(
    cache: dict[tuple[str, str], iem.CurvePack],
    basis_cache: dict[tuple[str, int], np.ndarray],
) -> None:
    scale = "100"
    source = load_pack(scale, iem.TRAIN_CURVE, cache)
    target = load_pack(scale, "wsd_20000_24000.csv", cache)
    lam = lambda_q2_halflife(target.curve)
    phi_source = feature(source.curve, "lr_time_exp", lam)
    mask = source.curve.step >= FIT_START
    step_k = source.curve.step[mask] / 1000.0
    residual = source.residual[mask].astype(np.float64)
    phi = phi_source[mask].astype(np.float64)
    q = tangent_basis(source, FIT_START, basis_cache)
    nuisance = q @ (q.T @ residual) if q.size else np.zeros_like(residual)
    projected_residual = residual - nuisance
    projected_phi = phi - q @ (q.T @ phi) if q.size else phi.copy()

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.5), constrained_layout=True)
    axes[0].plot(step_k, _standardize(residual), color="#dc2626", lw=1.2, label=r"raw $r_{\cos}$")
    axes[0].plot(step_k, _standardize(phi), color="#2563eb", lw=1.2, label=r"raw $\phi_{\lambda,\cos}$")
    axes[0].axhline(0, color="#111827", lw=0.7)
    axes[0].set_title("raw source signals")
    axes[0].set_ylabel("z-score on calibration suffix")
    axes[0].legend(fontsize=8)

    axes[1].plot(step_k, residual, color="#9ca3af", lw=1.0, label=r"$r_{\cos}$")
    axes[1].plot(step_k, nuisance, color="#7c3aed", lw=1.4, label=r"$P_{\rm LD} r_{\cos}$")
    axes[1].axhline(0, color="#111827", lw=0.7)
    axes[1].set_title("MPL-LD nuisance component")
    axes[1].set_ylabel("loss residual")
    axes[1].legend(fontsize=8)

    axes[2].plot(
        step_k,
        _standardize(projected_residual),
        color="#dc2626",
        lw=1.2,
        label=r"$(I-P_{\rm LD})r_{\cos}$",
    )
    axes[2].plot(
        step_k,
        _standardize(projected_phi),
        color="#2563eb",
        lw=1.2,
        label=r"$(I-P_{\rm LD})\phi_{\lambda,\cos}$",
    )
    axes[2].axhline(0, color="#111827", lw=0.7)
    axes[2].set_title("projected alignment")
    axes[2].set_ylabel("z-score on calibration suffix")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("step (k)")
        ax.grid(alpha=0.18)
    fig.suptitle("100M source cosine, calibration suffix t >= 8000; lambda from WSD sharp")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "projection_decomposition_cosine_100M.png", dpi=180)
    SLIDE_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(SLIDE_FIG_DIR / "fig_projection_decomposition_cosine_100M.png", dpi=180)
    plt.close(fig)


def plot_projection_ablation_errors(
    cache: dict[tuple[str, str], iem.CurvePack],
    basis_cache: dict[tuple[str, int], np.ndarray],
) -> None:
    scale = "100"
    chosen = [
        ("wsd_20000_24000.csv", "WSD sharp"),
        ("wsdcon_18.csv", "WSD-con 18e-5"),
    ]
    source = load_pack(scale, iem.TRAIN_CURVE, cache)
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 5.4), constrained_layout=True, sharex=False)
    for col, (curve_name, title) in enumerate(chosen):
        target = load_pack(scale, curve_name, cache)
        lam = lambda_q2_halflife(target.curve)
        phi_source = feature(source.curve, "lr_time_exp", lam)
        mask = source.curve.step >= FIT_START
        x_raw = phi_source[mask].astype(np.float64)
        y_raw = source.residual[mask].astype(np.float64)
        ridge_raw = 1.0 / max(int(np.sum(mask)), 1)
        k_raw = max(0.0, float(np.dot(x_raw, y_raw))) / max(float(np.dot(x_raw, x_raw)) + ridge_raw, 1e-18)
        x_proj, y_proj, n_cal = residualized_source_pair(source, phi_source, FIT_START, basis_cache)
        k_proj = max(0.0, float(np.dot(x_proj, y_proj))) / max(
            float(np.dot(x_proj, x_proj)) + 1.0 / max(n_cal, 1),
            1e-18,
        )
        phi_target = feature(target.curve, "lr_time_exp", lam)
        step_k = target.curve.step / 1000.0
        mpl_error = target.baseline - target.curve.loss
        proj_error = target.baseline + k_proj * phi_target - target.curve.loss
        raw_error = target.baseline + k_raw * phi_target - target.curve.loss

        top = axes[0, col]
        top.axhline(0, color="#111827", lw=0.8)
        top.plot(step_k, mpl_error, color="#dc2626", lw=1.1, label="MPL error")
        top.plot(step_k, proj_error, color="#2563eb", lw=1.2, label="projected kappa")
        top.set_title(f"{title}: projected estimator")
        top.set_ylabel("prediction error")
        top.grid(alpha=0.18)
        top.legend(fontsize=8)

        bottom = axes[1, col]
        bottom.axhline(0, color="#111827", lw=0.8)
        bottom.plot(step_k, raw_error, color="#7c3aed", lw=1.2, label="no-projection kappa")
        bottom.plot(step_k, proj_error, color="#2563eb", lw=0.9, alpha=0.7, label="projected kappa")
        bottom.set_title(f"same target: no-projection failure")
        bottom.set_xlabel("step (k)")
        bottom.set_ylabel("prediction error")
        bottom.grid(alpha=0.18)
        bottom.legend(fontsize=8)
        bottom.text(
            0.02,
            0.95,
            f"k raw={k_raw:.3f}, projected={k_proj:.3f}",
            transform=bottom.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            color="#374151",
        )
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "projection_ablation_time_errors_100M.png", dpi=180)
    SLIDE_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(SLIDE_FIG_DIR / "fig_projection_ablation_time_errors_100M.png", dpi=180)
    plt.close(fig)


def write_leakage_audit() -> None:
    lines = [
        "# Target-Leakage Audit\n\n",
        "| stage | files / quantities used | target WSD loss used? | notes |\n",
        "|---|---|---:|---|\n",
        "| Calibration | `cosine_72000.csv`, cosine LR schedule, frozen MPL prediction | no | Computes source residual and projected kappa. |\n",
        "| Target feature construction | target LR schedule only | no | Computes `q2`, `lambda_s`, and `phi_s(t)`. |\n",
        "| Prediction | frozen MPL target prediction, `phi_s(t)`, source-only `kappa_hat` | no | Outputs `L_MPL + kappa_hat phi`. |\n",
        "| Evaluation | target loss curve | yes | Computes MAE, terminal error, and oracle diagnostics only. |\n",
        "| Oracle kappa star | target residual | yes | Diagnostic; never used in deployable prediction. |\n",
        "| Oracle lambda grid | target MAE | yes | Upper-bound diagnostic; separated from deployable q2 rule. |\n",
        "\nDeployable rows in the main tables use no target WSD loss for calibration or prediction.\n",
    ]
    (OUT_DIR / "LEAKAGE_AUDIT.md").write_text("".join(lines), encoding="utf-8")


def fmt_summary(row: dict[str, object]) -> str:
    return (
        f"{float(row['mean_delta_pct']):+.2f}% / "
        f"{float(row['worst_delta_pct']):+.2f}% / "
        f"{int(row['wins'])}/{int(row['rows'])}, "
        f"Pearson {float(row['pearson_kappa_star']):+.3f}"
    )


def find_summary(rows: list[dict[str, object]], key: str, value: str) -> dict[str, object]:
    for row in rows:
        if str(row[key]) == value:
            return row
    raise KeyError((key, value))


def write_report(
    lambda_summary: list[dict[str, object]],
    kernel_summary: list[dict[str, object]],
    cross_summary: list[dict[str, object]],
    no_projection_summary: list[dict[str, object]],
    window_rule_rows: list[dict[str, object]],
    window_eval_rows: list[dict[str, object]],
    selected_fit_start: int,
    wsdcon_rows: list[dict[str, object]],
) -> None:
    q2 = find_summary(lambda_summary, "lambda_rule", "q2_half_life")
    oracle = find_summary(lambda_summary, "lambda_rule", "oracle_grid_best")
    fixed1 = find_summary(lambda_summary, "lambda_rule", "fixed_1obs")
    fixed2 = find_summary(lambda_summary, "lambda_rule", "fixed_2obs")
    wrong = find_summary(lambda_summary, "lambda_rule", "wrong_fast_lambda20")
    main_kernel = find_summary(kernel_summary, "kernel", "lr_time_exp")
    step_kernel = find_summary(kernel_summary, "kernel", "step_time_exp")
    pooled = find_summary(cross_summary, "protocol", "all_scale_pooled")
    loso = find_summary(cross_summary, "protocol", "leave_one_scale_out_pooled")
    loso_mean = find_summary(cross_summary, "protocol", "leave_one_scale_out_mean_kappa")
    all_mean = find_summary(cross_summary, "protocol", "all_scale_mean_kappa")
    same = find_summary(cross_summary, "protocol", "same_scale")
    no_projection = find_summary(no_projection_summary, "estimator", "direct_no_projection")
    window_summary = summarize(window_eval_rows, "fit_start")
    selected_row = find_summary(window_summary, "fit_start", str(selected_fit_start))

    lines = [
        "# Schedule-Response Robustness Audit\n\n",
        "This audit adds evidence around the projected cosine-kappa formula without changing the deployable model.\n\n",
        "## Data Boundary\n\n",
        "The repository contains five WSD-family target schedules for each of three scales.  It does not contain additional unseen WSD-family training runs.  Therefore a truly new held-out schedule experiment must be run externally; this audit focuses on robustness checks possible with the current data.\n\n",
        "## Lambda Sensitivity\n\n",
        "| lambda rule | mean / worst / wins / Pearson |\n",
        "|---|---:|\n",
        f"| q2 half-life | {fmt_summary(q2)} |\n",
        f"| fixed 1 observation | {fmt_summary(fixed1)} |\n",
        f"| fixed 2 observations | {fmt_summary(fixed2)} |\n",
        f"| wrong fast lambda=20 | {fmt_summary(wrong)} |\n",
        f"| oracle grid best (target-loss diagnostic) | {fmt_summary(oracle)} |\n",
        "\nThe q2 rule is not presented as a target-tuned optimum.  The oracle grid is marked as diagnostic because it uses target loss.  The deployable q2 rule remains stable and close to the best fixed observation-scale rules without target loss.\n\n",
        "## Projection Negative Control\n\n",
        "| estimator | mean / worst / wins / Pearson |\n",
        "|---|---:|\n",
        f"| projected q2 half-life kappa | {fmt_summary(q2)} |\n",
        f"| direct cosine kappa without MPL-LD projection | {fmt_summary(no_projection)} |\n",
        "\nThe no-projection row uses the same LR-time response feature and the same source cosine residual, but estimates kappa before removing the MPL-LD tangent nuisance.  It fails catastrophically, which is the main evidence that this is an identification problem rather than arbitrary residual fitting.\n\n",
        "## Same-Capacity Kernel Alternatives\n\n",
        "| kernel | mean / worst / wins / Pearson |\n",
        "|---|---:|\n",
    ]
    for row in kernel_summary:
        lines.append(f"| {row['kernel']} | {fmt_summary(row)} |\n")
    lines += [
        "\nThe LR-time exponential drop response is the main one-scalar feature.  Same-capacity alternatives are useful controls; the important comparison is that arbitrary level/drop features do not explain the response as cleanly as a causal drop relaxation kernel.\n\n",
        "## Cross-Scale Transfer\n\n",
        "| protocol | mean / worst / wins / Pearson |\n",
        "|---|---:|\n",
        f"| same scale | {fmt_summary(same)} |\n",
        f"| leave-one-scale-out pooled | {fmt_summary(loso)} |\n",
        f"| leave-one-scale-out mean kappa | {fmt_summary(loso_mean)} |\n",
        f"| all-scale pooled | {fmt_summary(pooled)} |\n",
        f"| all-scale mean kappa | {fmt_summary(all_mean)} |\n",
    ]
    for row in cross_summary:
        if str(row["protocol"]).startswith("single_source"):
            lines.append(f"| {row['protocol']} | {fmt_summary(row)} |\n")
    lines += [
        "\nCross-scale transfer is a stricter test because kappa amplitudes are not guaranteed to be scale invariant.  These rows define the boundary of the current method more clearly than same-scale evaluation alone.\n\n",
        "## Source-Only Calibration Window Rule\n\n",
        "| fit start | max retention | finite-sample floor | passes source rule |\n",
        "|---:|---:|---:|---:|\n",
    ]
    for row in window_rule_rows:
        lines.append(
            f"| {int(row['fit_start'])} | {float(row['max_retention']):.6f} | "
            f"{float(row['finite_sample_floor']):.6f} | {int(row['passes'])} |\n"
        )
    lines += [
        f"\nThe selected source-only fit start is `{selected_fit_start}`.  Its target evaluation is {fmt_summary(selected_row)}.  Target losses are not used in the selection rule.\n\n",
        "## WSD-con Failure-Mode Slice\n\n",
        "| scale | target | kappa_hat | kappa_star | ratio | MAE change | terminal MPL err | terminal corrected err |\n",
        "|---:|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in wsdcon_rows:
        lines.append(
            f"| {row['scale']} | {row['target_label']} | "
            f"{float(row['kappa_hat']):.4f} | {float(row['kappa_star']):.4f} | "
            f"{float(row['kappa_ratio']):.3f} | {float(row['delta_pct']):+.2f}% | "
            f"{float(row['terminal_mpl_error']):+.4e} | {float(row['terminal_corr_error']):+.4e} |\n"
        )
    lines += [
        "\nWSD-con remains the main fine-grained limitation: aggregate MAE improves, but the ordering across final LR values is weaker than the WSD sharp/linear split.\n\n",
        "## Figures\n\n",
        "- `figs/mpl_residual_anomaly_100M.png`\n",
        "- `figs/projection_decomposition_cosine_100M.png`\n",
        "- `figs/projection_ablation_time_errors_100M.png`\n",
        "- `figs/representative_time_errors_100M.png`\n",
        "- `figs/mae_change_heatmap.png`\n",
        "- `LEAKAGE_AUDIT.md`\n",
    ]
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cache: dict[tuple[str, str], iem.CurvePack] = {}
    basis_cache: dict[tuple[str, int], np.ndarray] = {}

    lambda_rows, lambda_summary = lambda_sensitivity(cache, basis_cache)
    kernel_rows, kernel_summary = kernel_ablation(cache, basis_cache)
    cross_rows, cross_summary = cross_scale_transfer(cache, basis_cache)
    no_projection_rows, no_projection_summary = no_projection_negative_control(cache)
    window_rule_rows, window_eval_rows, selected_fit_start = window_audit(cache, basis_cache)
    wsdcon_rows = wsdcon_failure_rows(cross_rows)

    write_csv(OUT_DIR / "lambda_sensitivity_details.csv", lambda_rows)
    write_csv(OUT_DIR / "lambda_sensitivity_summary.csv", lambda_summary)
    write_csv(OUT_DIR / "kernel_ablation_details.csv", kernel_rows)
    write_csv(OUT_DIR / "kernel_ablation_summary.csv", kernel_summary)
    write_csv(OUT_DIR / "cross_scale_details.csv", cross_rows)
    write_csv(OUT_DIR / "cross_scale_summary.csv", cross_summary)
    write_csv(OUT_DIR / "projection_ablation_details.csv", no_projection_rows)
    write_csv(OUT_DIR / "projection_ablation_summary.csv", no_projection_summary)
    write_csv(OUT_DIR / "window_rule.csv", window_rule_rows)
    write_csv(OUT_DIR / "window_sweep_details.csv", window_eval_rows)
    write_csv(OUT_DIR / "window_sweep_summary.csv", summarize(window_eval_rows, "fit_start"))
    write_csv(OUT_DIR / "wsdcon_failure_slice.csv", wsdcon_rows)

    plot_heatmap(cross_rows)
    plot_mpl_residual_anomaly(cache)
    plot_projection_decomposition(cache, basis_cache)
    plot_projection_ablation_errors(cache, basis_cache)
    plot_representative_errors(cache, cross_rows)
    write_leakage_audit()
    write_report(
        lambda_summary,
        kernel_summary,
        cross_summary,
        no_projection_summary,
        window_rule_rows,
        window_eval_rows,
        selected_fit_start,
        wsdcon_rows,
    )
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
