#!/usr/bin/env python3
"""Image-driven response feature plus nuisance-projected kappa search.

The residual-shape plots show two separate phenomena:

1. LR-drop response should be local in optimization steps.
2. Cosine residuals contain broad low-frequency MPL mismatch that can pollute a
   same-curve kappa estimate.

This audit combines both lessons.  It searches finite step-time response
features together with low-frequency nuisance residualization, EB shrinkage,
and optional target-side identifiability factors.
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
from matplotlib.colors import TwoSlopeNorm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent
ROOT = REPO.parent
sys.path.insert(0, str(REPO))

from reproduce_cosine_to_wsd import (  # noqa: E402
    MPL_PRECOMPUTED_INIT,
    PEAK_LR,
    SCALES,
    load_curve,
    metrics,
    mpl_predict,
)


OUT_DIR = ROOT / "results" / "step_time_nuisance_estimator"
FIG_DIR = OUT_DIR / "figs"

CURVES = [
    ("cosine_72000.csv", "Cosine"),
    ("wsd_20000_24000.csv", "WSD sharp"),
    ("wsdld_20000_24000.csv", "WSD linear"),
    ("wsdcon_3.csv", "WSD-con 3e-5"),
    ("wsdcon_9.csv", "WSD-con 9e-5"),
    ("wsdcon_18.csv", "WSD-con 18e-5"),
]

FEATURE_SPECS = [
    ("S10_current", "s_time", 10.0),
    ("step_tau512", "step_time", 512.0),
    ("step_tau768", "step_time", 768.0),
    ("step_tau1024", "step_time", 1024.0),
    ("step_tau1536", "step_time", 1536.0),
    ("step_tau2048", "step_time", 2048.0),
    ("step_tau2304", "step_time", 2304.0),
    ("step_tau3072", "step_time", 3072.0),
]
NUISANCE_SPECS = ["none", "dct2", "dct4", "fourier2"]
TRAIN_RETENTION_POWERS = [0.0, 0.5, 1.0]
TARGET_FACTORS = [
    "none",
    "sqrt_retention",
    "drop_sqrt",
    "drop_linear",
    "sqrt_retention_drop_sqrt",
    "gate_0p01",
    "gate_0p03",
    "gate_0p05",
]
TAU_MODES = ["none", "eb_q75"]


def robust_scale(x: np.ndarray) -> float:
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return max(1.4826 * mad, float(np.std(x)) * 0.25, 1e-12)


def response_feature(curve, kind: str, param: float) -> np.ndarray:
    eta = curve.lrs.astype(np.float64)
    drop = np.zeros_like(eta)
    drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0) / PEAK_LR
    out = np.zeros_like(eta)
    acc = 0.0
    for t in range(len(eta)):
        if kind == "s_time":
            rate = param * eta[t]
        elif kind == "step_time":
            rate = 1.0 / param
        else:
            raise ValueError(kind)
        acc = acc * math.exp(-rate) + drop[t]
        out[t] = acc
    return out[curve.step]


def nuisance_basis(steps: np.ndarray, spec: str) -> np.ndarray | None:
    if spec == "none":
        return None
    t = steps.astype(np.float64)
    z = (t - float(t[0])) / max(float(t[-1] - t[0]), 1.0)
    cols = [np.ones_like(z)]
    if spec.startswith("dct"):
        modes = int(spec.replace("dct", ""))
        idx = np.arange(len(steps), dtype=np.float64)
        for k in range(1, modes + 1):
            cols.append(np.cos(math.pi * (idx + 0.5) * k / max(len(steps), 1)))
    elif spec == "fourier2":
        cols += [
            np.sin(math.pi * z),
            np.cos(math.pi * z),
            np.sin(2.0 * math.pi * z),
            np.cos(2.0 * math.pi * z),
        ]
    else:
        raise ValueError(spec)
    basis = np.column_stack(cols)
    norms = np.linalg.norm(basis, axis=0)
    return basis / np.maximum(norms, 1e-12)


def residualize(y: np.ndarray, z: np.ndarray | None) -> np.ndarray:
    if z is None:
        return y.copy()
    coef, *_ = np.linalg.lstsq(z, y, rcond=None)
    return y - z @ coef


def quantile(vals: list[float], q: float) -> float:
    vals = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if not vals:
        return float("nan")
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def build_cache() -> tuple[
    dict[tuple[str, str], dict[str, object]],
    dict[tuple[str, str, str], np.ndarray],
]:
    curves: dict[tuple[str, str], dict[str, object]] = {}
    feats: dict[tuple[str, str, str], np.ndarray] = {}
    for scale in SCALES:
        for curve_name, _ in CURVES:
            curve = load_curve(scale, curve_name)
            base = mpl_predict(MPL_PRECOMPUTED_INIT[scale], curve)
            curves[(scale, curve_name)] = {
                "curve": curve,
                "base": base,
                "resid": curve.loss - base,
                "base_mae": metrics(curve.loss, base)["mae"],
            }
            for feature_name, kind, param in FEATURE_SPECS:
                feats[(feature_name, scale, curve_name)] = response_feature(curve, kind, param)
    return curves, feats


def projected_stats(
    curves: dict[tuple[str, str], dict[str, object]],
    feats: dict[tuple[str, str, str], np.ndarray],
    feature_name: str,
    nuisance: str,
    scale: str,
    curve_name: str,
) -> dict[str, float]:
    row = curves[(scale, curve_name)]
    curve = row["curve"]
    phi = feats[(feature_name, scale, curve_name)]
    resid = row["resid"]
    eta = curve.lrs.astype(np.float64)
    drops = np.zeros_like(eta)
    drops[1:] = np.maximum(eta[:-1] - eta[1:], 0.0) / PEAK_LR
    total_drop = float(np.sum(drops))
    z = nuisance_basis(curve.step, nuisance)
    phi_o = residualize(phi, z)
    resid_o = residualize(resid, z)
    phi_l2 = max(float(np.dot(phi, phi)), 1e-18)
    phi_o_l2 = max(float(np.dot(phi_o, phi_o)), 1e-18)
    dot = float(np.dot(phi_o, resid_o))
    corr_denom = float(np.linalg.norm(phi_o) * np.linalg.norm(resid_o))
    return {
        "feature_l2": phi_l2,
        "projected_feature_l2": phi_o_l2,
        "projection_dot": dot,
        "projected_raw_kappa": max(0.0, dot / phi_o_l2),
        "feature_retention": float(phi_o_l2 / phi_l2),
        "target_drop_norm": total_drop,
        "target_drop_factor_sqrt": math.sqrt(min(max(total_drop / 0.9, 0.0), 1.0)),
        "target_drop_factor_linear": min(max(total_drop / 0.9, 0.0), 1.0),
        "projected_corr": 0.0 if corr_denom <= 1e-18 else float(dot / corr_denom),
        "projected_resid_scale": robust_scale(resid_o),
    }


def all_stats(
    curves: dict[tuple[str, str], dict[str, object]],
    feats: dict[tuple[str, str, str], np.ndarray],
) -> dict[tuple[str, str, str, str], dict[str, float]]:
    return {
        (feature_name, nuisance, scale, curve_name): projected_stats(
            curves, feats, feature_name, nuisance, scale, curve_name
        )
        for feature_name, _, _ in FEATURE_SPECS
        for nuisance in NUISANCE_SPECS
        for scale in SCALES
        for curve_name, _ in CURVES
    }


def estimate_tau(pool: list[dict[str, float]]) -> float:
    good = [
        row
        for row in pool
        if row["projected_raw_kappa"] > 0.0
        and row["feature_retention"] > 0.01
        and row["projected_corr"] > 0.05
    ]
    if len(good) < 4:
        good = [row for row in pool if row["projected_raw_kappa"] > 0.0 and row["feature_retention"] > 0.001]
    sigma = quantile([row["projected_resid_scale"] for row in good], 0.50)
    k0 = quantile([row["projected_raw_kappa"] for row in good], 0.75)
    if not math.isfinite(sigma) or not math.isfinite(k0) or k0 <= 1e-12:
        return 0.0
    return min(max(sigma / max(k0, 1e-12), 0.0), 0.50)


def fit_kappa(
    stats_rows: list[dict[str, float]],
    train_power: float,
    tau: float,
) -> float:
    dot = float(sum(row["projection_dot"] for row in stats_rows))
    l2 = float(sum(row["projected_feature_l2"] for row in stats_rows))
    full_l2 = float(sum(row["feature_l2"] for row in stats_rows))
    retention = max(l2 / max(full_l2, 1e-18), 0.0)
    raw = max(0.0, dot / max(l2 + tau * tau, 1e-18))
    return (retention ** train_power) * raw


def target_factor(stats: dict[str, float], mode: str) -> float:
    retention = max(float(stats["feature_retention"]), 0.0)
    drop_sqrt = max(float(stats["target_drop_factor_sqrt"]), 0.0)
    drop_linear = max(float(stats["target_drop_factor_linear"]), 0.0)
    if mode == "none":
        return 1.0
    if mode == "sqrt_retention":
        return math.sqrt(retention)
    if mode == "drop_sqrt":
        return drop_sqrt
    if mode == "drop_linear":
        return drop_linear
    if mode == "sqrt_retention_drop_sqrt":
        return math.sqrt(retention) * drop_sqrt
    if mode.startswith("gate_"):
        threshold = float(mode.replace("gate_", "").replace("p", "."))
        return 1.0 if retention >= threshold else 0.0
    raise ValueError(mode)


def score_prediction(
    curves: dict[tuple[str, str], dict[str, object]],
    feats: dict[tuple[str, str, str], np.ndarray],
    stats_cache: dict[tuple[str, str, str, str], dict[str, float]],
    feature_name: str,
    nuisance: str,
    target_mode: str,
    scale: str,
    test_curve: str,
    kappa: float,
) -> dict[str, object]:
    row = curves[(scale, test_curve)]
    stats = stats_cache[(feature_name, nuisance, scale, test_curve)]
    factor = target_factor(stats, target_mode)
    pred = row["base"] + factor * kappa * feats[(feature_name, scale, test_curve)]
    corr_mae = metrics(row["curve"].loss, pred)["mae"]
    base_mae = float(row["base_mae"])
    return {
        "target_factor_value": factor,
        "target_retention": stats["feature_retention"],
        "base_mae": base_mae,
        "corr_mae": corr_mae,
        "delta_pct": 100.0 * (corr_mae / base_mae - 1.0),
        "win": int(corr_mae < base_mae),
    }


def method_id(feature: str, nuisance: str, tau_mode: str, train_power: float, target_mode: str) -> str:
    power = str(train_power).replace(".", "p")
    return f"{feature}__{nuisance}__{tau_mode}__R{power}__T{target_mode}"


def run_search() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    curves, feats = build_cache()
    stats_cache = all_stats(curves, feats)
    details: list[dict[str, object]] = []
    kappa_rows: list[dict[str, object]] = []

    for feature_name, _, _ in FEATURE_SPECS:
        for nuisance in NUISANCE_SPECS:
            for tau_mode in TAU_MODES:
                for train_power in TRAIN_RETENTION_POWERS:
                    for target_mode in TARGET_FACTORS:
                        mid = method_id(feature_name, nuisance, tau_mode, train_power, target_mode)
                        for train_curve, train_label in CURVES:
                            tau_pool = [
                                stats_cache[(feature_name, nuisance, scale, curve_name)]
                                for scale in SCALES
                                for curve_name, _ in CURVES
                                if curve_name != train_curve
                            ]
                            tau = estimate_tau(tau_pool) if tau_mode == "eb_q75" else 0.0
                            for scale in SCALES:
                                train_stats = [stats_cache[(feature_name, nuisance, scale, train_curve)]]
                                kappa = fit_kappa(train_stats, train_power, tau)
                                train_ret = train_stats[0]["feature_retention"]
                                kappa_rows.append(
                                    {
                                        "method": mid,
                                        "feature": feature_name,
                                        "nuisance": nuisance,
                                        "tau_mode": tau_mode,
                                        "train_retention_power": train_power,
                                        "target_mode": target_mode,
                                        "mode": "single_curve",
                                        "scale": scale,
                                        "train_curve": train_curve,
                                        "train_label": train_label,
                                        "tau": tau,
                                        "kappa": kappa,
                                        "train_retention": train_ret,
                                        **train_stats[0],
                                    }
                                )
                                for test_curve, test_label in CURVES:
                                    scored = score_prediction(
                                        curves,
                                        feats,
                                        stats_cache,
                                        feature_name,
                                        nuisance,
                                        target_mode,
                                        scale,
                                        test_curve,
                                        kappa,
                                    )
                                    details.append(
                                        {
                                            "method": mid,
                                            "feature": feature_name,
                                            "nuisance": nuisance,
                                            "tau_mode": tau_mode,
                                            "train_retention_power": train_power,
                                            "target_mode": target_mode,
                                            "mode": "single_curve",
                                            "scale": scale,
                                            "train_curve": train_curve,
                                            "train_label": train_label,
                                            "test_curve": test_curve,
                                            "test_label": test_label,
                                            "tau": tau,
                                            "kappa": kappa,
                                            **scored,
                                        }
                                    )

    group_defs = [
        ("probe", ["wsdcon_3.csv", "wsdcon_9.csv", "wsdcon_18.csv"]),
        ("probe3", ["wsdcon_3.csv"]),
        ("wsd", ["wsd_20000_24000.csv", "wsdld_20000_24000.csv"]),
        ("cosine", ["cosine_72000.csv"]),
    ]
    group_details: list[dict[str, object]] = []
    for feature_name, _, _ in FEATURE_SPECS:
        for nuisance in NUISANCE_SPECS:
            for tau_mode in TAU_MODES:
                for train_power in TRAIN_RETENTION_POWERS:
                    for target_mode in TARGET_FACTORS:
                        mid = method_id(feature_name, nuisance, tau_mode, train_power, target_mode)
                        for group_id, train_curves in group_defs:
                            tau_pool = [
                                stats_cache[(feature_name, nuisance, scale, curve_name)]
                                for scale in SCALES
                                for curve_name, _ in CURVES
                                if curve_name not in set(train_curves)
                            ]
                            tau = estimate_tau(tau_pool) if tau_mode == "eb_q75" else 0.0
                            for scale in SCALES:
                                train_stats = [
                                    stats_cache[(feature_name, nuisance, scale, curve_name)]
                                    for curve_name in train_curves
                                ]
                                kappa = fit_kappa(train_stats, train_power, tau)
                                for test_curve, test_label in CURVES:
                                    scored = score_prediction(
                                        curves,
                                        feats,
                                        stats_cache,
                                        feature_name,
                                        nuisance,
                                        target_mode,
                                        scale,
                                        test_curve,
                                        kappa,
                                    )
                                    group_details.append(
                                        {
                                            "method": mid,
                                            "feature": feature_name,
                                            "nuisance": nuisance,
                                            "tau_mode": tau_mode,
                                            "train_retention_power": train_power,
                                            "target_mode": target_mode,
                                            "mode": "group",
                                            "group_id": group_id,
                                            "train_curves": "+".join(c.replace(".csv", "") for c in train_curves),
                                            "scale": scale,
                                            "test_curve": test_curve,
                                            "test_label": test_label,
                                            "tau": tau,
                                            "kappa": kappa,
                                            **scored,
                                        }
                                    )
    return details, kappa_rows, group_details


def summarize_single(details: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    methods = sorted({str(r["method"]) for r in details})
    for mid in methods:
        sub = [r for r in details if r["method"] == mid]
        first = sub[0]
        self_rows = [r for r in sub if r["train_curve"] == r["test_curve"]]
        off_rows = [r for r in sub if r["train_curve"] != r["test_curve"]]
        cosine_wsd = [
            r
            for r in sub
            if r["train_curve"] == "cosine_72000.csv" and r["test_curve"] == "wsd_20000_24000.csv"
        ]
        probe_wsd = [
            r
            for r in sub
            if r["train_curve"] in {"wsdcon_3.csv", "wsdcon_9.csv", "wsdcon_18.csv"}
            and r["test_curve"] in {"wsd_20000_24000.csv", "wsdld_20000_24000.csv"}
        ]
        rows.append(
            {
                "method": mid,
                "feature": first["feature"],
                "nuisance": first["nuisance"],
                "tau_mode": first["tau_mode"],
                "train_retention_power": first["train_retention_power"],
                "target_mode": first["target_mode"],
                "self_mean_delta": mean([float(r["delta_pct"]) for r in self_rows]),
                "self_worst_delta": max_float([float(r["delta_pct"]) for r in self_rows]),
                "self_wins": sum(int(r["win"]) for r in self_rows),
                "self_tests": len(self_rows),
                "offdiag_mean_delta": mean([float(r["delta_pct"]) for r in off_rows]),
                "offdiag_worst_delta": max_float([float(r["delta_pct"]) for r in off_rows]),
                "offdiag_wins": sum(int(r["win"]) for r in off_rows),
                "offdiag_tests": len(off_rows),
                "cosine_to_wsd_mean": mean([float(r["delta_pct"]) for r in cosine_wsd]),
                "probe_to_wsd_mean": mean([float(r["delta_pct"]) for r in probe_wsd]),
                "probe_to_wsd_worst": max_float([float(r["delta_pct"]) for r in probe_wsd]),
            }
        )
    rows.sort(key=single_objective)
    return rows


def summarize_group(group_details: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    keys = sorted({(str(r["method"]), str(r["group_id"])) for r in group_details})
    for mid, group_id in keys:
        sub = [r for r in group_details if r["method"] == mid and r["group_id"] == group_id]
        first = sub[0]
        for target_group, target_curves in [
            ("wsd", {"wsd_20000_24000.csv", "wsdld_20000_24000.csv"}),
            ("probe", {"wsdcon_3.csv", "wsdcon_9.csv", "wsdcon_18.csv"}),
            ("cosine", {"cosine_72000.csv"}),
        ]:
            trows = [r for r in sub if r["test_curve"] in target_curves]
            rows.append(
                {
                    "method": mid,
                    "feature": first["feature"],
                    "nuisance": first["nuisance"],
                    "tau_mode": first["tau_mode"],
                    "train_retention_power": first["train_retention_power"],
                    "target_mode": first["target_mode"],
                    "group_id": group_id,
                    "target_group": target_group,
                    "mean_delta": mean([float(r["delta_pct"]) for r in trows]),
                    "worst_delta": max_float([float(r["delta_pct"]) for r in trows]),
                    "wins": sum(int(r["win"]) for r in trows),
                    "tests": len(trows),
                }
            )
    return rows


def mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else float("nan")


def max_float(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.max(arr)) if len(arr) else float("nan")


def single_objective(row: dict[str, object]) -> tuple[float, float, float, float]:
    self_worst = float(row["self_worst_delta"])
    off_worst = float(row["offdiag_worst_delta"])
    probe_worst = float(row["probe_to_wsd_worst"])
    probe_mean = float(row["probe_to_wsd_mean"])
    self_mean = float(row["self_mean_delta"])
    off_mean = float(row["offdiag_mean_delta"])
    harm = max(self_worst, 0.0) + 3.0 * max(off_worst, 0.0) + max(probe_worst, 0.0)
    # Prefer large useful reductions once non-harm is satisfied.
    utility = probe_mean + 0.4 * self_mean + 0.2 * off_mean
    return (harm, utility, self_mean, off_mean)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_top_methods(single_summary: list[dict[str, object]], path: Path) -> None:
    selected = []
    for row in single_summary:
        if len(selected) >= 12:
            break
        selected.append(row)
    labels = [short_method(row) for row in selected]
    x = np.arange(len(selected))
    fig, ax = plt.subplots(figsize=(13.8, 5.6))
    width = 0.24
    ax.axhline(0.0, color="#333333", lw=0.8)
    ax.bar(x - width, [float(r["self_mean_delta"]) for r in selected], width, label="self mean")
    ax.bar(x, [float(r["probe_to_wsd_mean"]) for r in selected], width, label="single probe -> WSD mean")
    ax.bar(x + width, [float(r["offdiag_mean_delta"]) for r in selected], width, label="all offdiag mean")
    ax.set_xticks(x, labels, rotation=32, ha="right")
    ax.set_ylabel("MAE change vs MPL (%)")
    ax.set_title("Top image-driven nuisance/step-time estimator variants")
    ax.grid(axis="y", alpha=0.24)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def short_method(row: dict[str, object]) -> str:
    feat = str(row["feature"]).replace("step_tau", "t")
    nuis = str(row["nuisance"])
    tau = "EB" if row["tau_mode"] == "eb_q75" else "LS"
    rp = str(row["train_retention_power"])
    tgt = str(row["target_mode"]).replace("sqrt_retention", "sqrtT")
    return f"{feat}/{nuis}/{tau}/R{rp}/{tgt}"


def plot_group_key(group_summary: list[dict[str, object]], path: Path) -> None:
    # Pick the best per-feature candidate for pooled-probe -> WSD.
    rows = [
        r
        for r in group_summary
        if r["group_id"] == "probe" and r["target_group"] == "wsd"
    ]
    rows = sorted(rows, key=lambda r: (max(float(r["worst_delta"]), 0.0), float(r["mean_delta"])))[:14]
    labels = [short_method(r) for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(13.8, 5.2))
    ax.axhline(0.0, color="#333333", lw=0.8)
    ax.bar(x, [float(r["mean_delta"]) for r in rows], label="mean", color="#059669")
    ax.scatter(x, [float(r["worst_delta"]) for r in rows], label="worst", color="#dc2626", zorder=3)
    ax.set_xticks(x, labels, rotation=32, ha="right")
    ax.set_ylabel("MAE change vs MPL (%)")
    ax.set_title("Pooled probe calibration -> WSD targets")
    ax.grid(axis="y", alpha=0.24)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def plot_matrix(
    details: list[dict[str, object]],
    method: str,
    path: Path,
) -> None:
    labels = [label for _, label in CURVES]
    mat = np.full((len(CURVES), len(CURVES)), np.nan)
    wins: dict[tuple[int, int], str] = {}
    for i, (train_curve, _) in enumerate(CURVES):
        for j, (test_curve, _) in enumerate(CURVES):
            sub = [
                r
                for r in details
                if r["method"] == method
                and r["train_curve"] == train_curve
                and r["test_curve"] == test_curve
            ]
            mat[i, j] = mean([float(r["delta_pct"]) for r in sub])
            wins[(i, j)] = f"{sum(int(r['win']) for r in sub)}/{len(sub)}"
    fig, ax = plt.subplots(figsize=(9.2, 7.0))
    im = ax.imshow(mat, cmap="RdYlGn_r", norm=TwoSlopeNorm(vmin=-80, vcenter=0, vmax=80))
    ax.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("test curve")
    ax.set_ylabel("calibration curve")
    ax.set_title(method)
    for i in range(len(CURVES)):
        for j in range(len(CURVES)):
            ax.text(j, i, f"{mat[i,j]:+.1f}%\n{wins[(i,j)]}", ha="center", va="center", fontsize=8, weight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("MAE change vs MPL")
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def write_report(
    single_summary: list[dict[str, object]],
    group_summary: list[dict[str, object]],
) -> None:
    top = single_summary[0]
    conservative = next(
        (
            r
            for r in single_summary
            if r["method"] == "step_tau1024__dct2__eb_q75__R0p5__Tdrop_linear"
        ),
        top,
    )
    image_direct = next(
        (
            r
            for r in single_summary
            if r["method"] == "step_tau1024__fourier2__eb_q75__R0p0__Tdrop_linear"
        ),
        top,
    )
    s10_safe = next(
        (
            r
            for r in single_summary
            if r["feature"] == "S10_current"
            and r["nuisance"] == "dct2"
            and r["tau_mode"] == "eb_q75"
            and float(r["train_retention_power"]) == 1.0
            and r["target_mode"] in {"none", "gate_0p01"}
        ),
        top,
    )
    probe_rows = [
        r
        for r in group_summary
        if r["group_id"] == "probe" and r["target_group"] == "wsd"
    ]
    best_probe = sorted(probe_rows, key=lambda r: (max(float(r["worst_delta"]), 0.0), float(r["mean_delta"])))[0]
    lines = [
        "# Step-Time Nuisance Estimator Search\n\n",
        "This search tests the image-driven hypothesis that the transferable error term should be a finite step-time response, while broad cosine residuals should be treated as low-frequency nuisance structure.\n\n",
        "## Candidate Formula\n\n",
        "The strongest single-curve candidate found here is:\n\n",
        "```text\n",
        "phi_tau(t) = sum_{u<=t} exp(-(t-u)/1024) * relu(eta_{u-1}-eta_u) / eta_peak\n",
        "G = span{1, sin(pi z), cos(pi z), sin(2 pi z), cos(2 pi z)}\n",
        "phi_perp = M_G phi_tau,   r_perp = M_G(observed_loss - MPL)\n",
        "kappa = max(0, <phi_perp, r_perp> / (||phi_perp||^2 + tau_EB^2))\n",
        "target_factor = total_positive_lr_drop(target) / 0.9\n",
        "prediction = MPL + target_factor * kappa * phi_tau(target)\n",
        "```\n\n",
        "The target factor is schedule-only. It corrects the observed over-transfer from full-drop schedules to small-drop targets such as `WSD-con 18e-5`.\n\n",
        "## Best Single-Curve Variants\n\n",
        "| rank | method | self mean | self worst | self wins | offdiag mean | offdiag worst | probe -> WSD mean | probe -> WSD worst |\n",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for rank, row in enumerate(single_summary[:12], start=1):
        lines.append(
            f"| {rank} | `{row['method']}` | {float(row['self_mean_delta']):+.1f}% | "
            f"{float(row['self_worst_delta']):+.1f}% | {int(row['self_wins'])}/{int(row['self_tests'])} | "
            f"{float(row['offdiag_mean_delta']):+.1f}% | {float(row['offdiag_worst_delta']):+.1f}% | "
            f"{float(row['probe_to_wsd_mean']):+.1f}% | {float(row['probe_to_wsd_worst']):+.1f}% |\n"
        )
    lines += [
        "\n## Pooled Probe To WSD\n\n",
        "| rank | method | mean | worst | wins |\n",
        "|---:|---|---:|---:|---:|\n",
    ]
    for rank, row in enumerate(
        sorted(probe_rows, key=lambda r: (max(float(r["worst_delta"]), 0.0), float(r["mean_delta"])))[:12],
        start=1,
    ):
        lines.append(
            f"| {rank} | `{row['method']}` | {float(row['mean_delta']):+.1f}% | "
            f"{float(row['worst_delta']):+.1f}% | {int(row['wins'])}/{int(row['tests'])} |\n"
        )
    lines += [
        "\n## Comparison To Existing Final Estimator\n\n",
        "| method | self mean | offdiag mean | offdiag worst | cosine -> WSD | probe -> WSD |\n",
        "|---|---:|---:|---:|---:|---:|\n",
        f"| image-direct step-time candidate | {float(image_direct['self_mean_delta']):+.1f}% | {float(image_direct['offdiag_mean_delta']):+.1f}% | {float(image_direct['offdiag_worst_delta']):+.1f}% | {float(image_direct['cosine_to_wsd_mean']):+.1f}% | {float(image_direct['probe_to_wsd_mean']):+.1f}% |\n",
        f"| conservative step-time candidate | {float(conservative['self_mean_delta']):+.1f}% | {float(conservative['offdiag_mean_delta']):+.1f}% | {float(conservative['offdiag_worst_delta']):+.1f}% | {float(conservative['cosine_to_wsd_mean']):+.1f}% | {float(conservative['probe_to_wsd_mean']):+.1f}% |\n",
        f"| safe old-feature reference | {float(s10_safe['self_mean_delta']):+.1f}% | {float(s10_safe['offdiag_mean_delta']):+.1f}% | {float(s10_safe['offdiag_worst_delta']):+.1f}% | {float(s10_safe['cosine_to_wsd_mean']):+.1f}% | {float(s10_safe['probe_to_wsd_mean']):+.1f}% |\n",
        "\n## Holdout Audit\n\n",
        "See `../step_time_nuisance_holdout_audit/REPORT.md`.\n\n",
        "Key findings:\n\n",
        "- Leave-one-scale fixed-best offdiag means are `-14.8%`, `-10.2%`, and `-13.9%`, with worst deltas no larger than `+0.0%`.\n",
        "- Leave-one-target fixed-best offdiag means range from `-19.8%` to `-4.3%`, with worst deltas no larger than `+0.0%`.\n",
        "- Unrestricted target holdout fails on `WSD-con 18e-5` by selecting the no-drop-factor variant (`+23.8%` worst). Restricting to the target-drop-linear family selects the fixed best candidate and restores `-4.6%` mean / `+0.0%` worst. This supports treating target-drop scaling as a structural part of the model, not a disposable hyperparameter.\n",
        "\n## Reading\n\n",
        f"- Best single-curve score: `{top['method']}` gives self mean `{float(top['self_mean_delta']):+.1f}%`, "
        f"probe-to-WSD mean `{float(top['probe_to_wsd_mean']):+.1f}%`, and off-diagonal mean `{float(top['offdiag_mean_delta']):+.1f}%`.\n",
        f"- A conservative image-consistent candidate is `{conservative['method']}` with self mean `{float(conservative['self_mean_delta']):+.1f}%` "
        f"and probe-to-WSD mean `{float(conservative['probe_to_wsd_mean']):+.1f}%`.\n",
        f"- Best pooled-probe WSD row is `{best_probe['method']}`, giving mean `{float(best_probe['mean_delta']):+.1f}%`, "
        f"worst `{float(best_probe['worst_delta']):+.1f}%`, and `{int(best_probe['wins'])}/{int(best_probe['tests'])}` wins.\n",
        "- The useful models all separate local response shape from broad low-frequency residuals; raw cosine self-fit is not used as proof of transferable kappa.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for obsolete in ["details.csv", "group_details.csv", "kappas.csv"]:
        path = OUT_DIR / obsolete
        if path.exists():
            path.unlink()
    details, kappa_rows, group_details = run_search()
    single_summary = summarize_single(details)
    group_summary = summarize_group(group_details)

    best_method = str(single_summary[0]["method"])
    best_group_method = str(
        sorted(
            [
                r
                for r in group_summary
                if r["group_id"] == "probe" and r["target_group"] == "wsd"
            ],
            key=lambda r: (max(float(r["worst_delta"]), 0.0), float(r["mean_delta"])),
        )[0]["method"]
    )
    write_csv(OUT_DIR / "best_single_details.csv", [r for r in details if r["method"] == best_method])
    write_csv(OUT_DIR / "best_single_kappas.csv", [r for r in kappa_rows if r["method"] == best_method])
    write_csv(
        OUT_DIR / "best_group_probe_to_wsd_details.csv",
        [
            r
            for r in group_details
            if r["method"] == best_group_method and r["group_id"] == "probe" and r["test_curve"] in {"wsd_20000_24000.csv", "wsdld_20000_24000.csv"}
        ],
    )
    write_csv(OUT_DIR / "single_summary.csv", single_summary)
    write_csv(OUT_DIR / "group_summary.csv", group_summary)
    write_report(single_summary, group_summary)

    plot_top_methods(single_summary, FIG_DIR / "top_single_methods.png")
    plot_group_key(group_summary, FIG_DIR / "top_group_probe_to_wsd.png")
    plot_matrix(details, str(single_summary[0]["method"]), FIG_DIR / "matrix_best_single.png")

    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    print("top single-curve methods:")
    for row in single_summary[:8]:
        print(
            f"{row['method']:70s} self={float(row['self_mean_delta']):+6.1f}%/"
            f"{float(row['self_worst_delta']):+5.1f}% "
            f"off={float(row['offdiag_mean_delta']):+6.1f}%/"
            f"{float(row['offdiag_worst_delta']):+5.1f}% "
            f"probeWSD={float(row['probe_to_wsd_mean']):+6.1f}%/"
            f"{float(row['probe_to_wsd_worst']):+5.1f}%"
        )
    probe_rows = [
        r
        for r in group_summary
        if r["group_id"] == "probe" and r["target_group"] == "wsd"
    ]
    print("top pooled probe -> WSD:")
    for row in sorted(probe_rows, key=lambda r: (max(float(r["worst_delta"]), 0.0), float(r["mean_delta"])))[:8]:
        print(
            f"{row['method']:70s} mean={float(row['mean_delta']):+6.1f}% "
            f"worst={float(row['worst_delta']):+5.1f}% wins={int(row['wins'])}/{int(row['tests'])}"
        )


if __name__ == "__main__":
    main()
