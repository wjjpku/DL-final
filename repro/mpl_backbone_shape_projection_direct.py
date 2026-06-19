#!/usr/bin/env python3
"""No-optimization shape projection for cosine-only MPL backbones.

This is the lightweight version of the backbone identifiability audit.  It does
not refit MPL.  It only projects weakly identified LD-kernel shape parameters
from independent cosine-only fits to their cross-scale median and evaluates the
same finite-response correction.

The goal is diagnostic: if this simple projection already improves WSD, the
cosine-only failure is likely driven by LD-shape instability.  If it hurts, a
more sophisticated backbone repair is needed and we should not hide the issue
behind extra residual terms.
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
    WARMUP,
    load_curve,
    metrics,
    mpl_predict,
)


OUT_DIR = ROOT / "results" / "mpl_ld_lag_response_audit" / "shape_projection_direct"
STRICT_PARAM_JSON = (
    ROOT
    / "results"
    / "cosine_to_wsd_response_search"
    / "cosine_only_backbone"
    / "cosine_only_mpl_params.json"
)
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
BACKBONE_PARAMS = {
    "frozen_official": 21,
    "cosine_independent": 21,
    "median_beta_gamma_projected": 17,
    "median_c_beta_gamma_projected": 15,
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


def load_independent_params() -> dict[str, np.ndarray]:
    raw = json.loads(STRICT_PARAM_JSON.read_text(encoding="utf-8"))
    return {scale: np.array(raw[scale], dtype=np.float64) for scale in SCALES}


def projected_params(independent: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    beta_med = float(np.median([independent[scale][5] for scale in SCALES]))
    gamma_med = float(np.median([independent[scale][6] for scale in SCALES]))
    c_med = float(np.median([independent[scale][4] for scale in SCALES]))
    out: dict[str, dict[str, np.ndarray]] = {
        "frozen_official": {scale: np.array(MPL_PRECOMPUTED_INIT[scale], dtype=np.float64) for scale in SCALES},
        "cosine_independent": independent,
        "median_beta_gamma_projected": {},
        "median_c_beta_gamma_projected": {},
    }
    for scale in SCALES:
        p = independent[scale].copy()
        p[5] = beta_med
        p[6] = gamma_med
        out["median_beta_gamma_projected"][scale] = p

        p = independent[scale].copy()
        p[4] = c_med
        p[5] = beta_med
        p[6] = gamma_med
        out["median_c_beta_gamma_projected"][scale] = p
    return out


def compute_ld_component(curve, params: np.ndarray, cooldown_only: bool) -> np.ndarray:
    _, _, _, _, c_value, beta, gamma = params
    lrs = curve.lrs.astype(np.float64)
    lr_sum = np.cumsum(lrs)
    lr_gap = np.zeros(len(lrs), dtype=np.float64)
    lr_gap[1:] = np.diff(lrs)
    selected_gap = np.minimum(lr_gap, 0.0) if cooldown_only else lr_gap
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


def finite_response_prediction(curve, params: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    baseline = mpl_predict(params, curve)
    d_down = compute_ld_component(curve, params, cooldown_only=True)
    tau = support_bracket_tau(curve.step, curve.lrs)
    factor = adiabatic_factor(curve.lrs)
    d_lag = lagged_observed(d_down, curve.step, tau)
    pred = baseline + factor * float(params[3]) * (d_lag - d_down)
    return pred, {
        "effective_tau_steps": tau,
        "adiabatic_factor": factor,
        "cooldown_support_span": float(cooldown_support_span(curve.lrs)),
    }


def detail_rows(param_sets: dict[str, dict[str, np.ndarray]]) -> list[dict[str, object]]:
    official_mae: dict[tuple[str, str], float] = {}
    for scale in SCALES:
        for _, curve_name, _ in ALL_TARGETS:
            curve = load_curve(scale, curve_name)
            official_mae[(scale, curve_name)] = metrics(
                curve.loss,
                mpl_predict(MPL_PRECOMPUTED_INIT[scale], curve),
            )["mae"]

    rows: list[dict[str, object]] = []
    for variant, by_scale in param_sets.items():
        for scale in SCALES:
            params = by_scale[scale]
            for group, curve_name, label in ALL_TARGETS:
                curve = load_curve(scale, curve_name)
                baseline = mpl_predict(params, curve)
                corrected, features = finite_response_prediction(curve, params)
                base_mae = mae(curve.loss, baseline)
                corr_mae = mae(curve.loss, corrected)
                ref = official_mae[(scale, curve_name)]
                rows.append(
                    {
                        "variant": variant,
                        "backbone_params": BACKBONE_PARAMS[variant],
                        "residual_params": 0,
                        "group": group,
                        "scale": scale,
                        "test_curve": curve_name,
                        "test_label": label,
                        "base_mae": base_mae,
                        "corr_mae": corr_mae,
                        "delta_vs_own_baseline_pct": 100.0 * (corr_mae / base_mae - 1.0),
                        "base_vs_official_baseline_pct": 100.0 * (base_mae / ref - 1.0),
                        "corr_vs_official_baseline_pct": 100.0 * (corr_mae / ref - 1.0),
                        "win_vs_own_baseline": int(corr_mae < base_mae),
                        "nonharm_vs_own_baseline": int(corr_mae <= base_mae + 1e-12),
                        **features,
                    }
                )
    return rows


def param_rows(param_sets: dict[str, dict[str, np.ndarray]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variant, by_scale in param_sets.items():
        for scale, params in by_scale.items():
            rows.append(
                {
                    "variant": variant,
                    "scale": scale,
                    "backbone_params": BACKBONE_PARAMS[variant],
                    "residual_params": 0,
                    **{f"p{i}": float(value) for i, value in enumerate(params)},
                }
            )
    return rows


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for variant in sorted({str(row["variant"]) for row in rows}):
        for group in sorted({str(row["group"]) for row in rows if row["variant"] == variant}):
            sub = [row for row in rows if row["variant"] == variant and row["group"] == group]
            own = np.array([float(row["delta_vs_own_baseline_pct"]) for row in sub], dtype=np.float64)
            base_off = np.array([float(row["base_vs_official_baseline_pct"]) for row in sub], dtype=np.float64)
            corr_off = np.array([float(row["corr_vs_official_baseline_pct"]) for row in sub], dtype=np.float64)
            out.append(
                {
                    "variant": variant,
                    "backbone_params": BACKBONE_PARAMS[variant],
                    "residual_params": 0,
                    "group": group,
                    "rows": len(sub),
                    "mean_delta_vs_own_baseline": float(np.mean(own)),
                    "worst_delta_vs_own_baseline": float(np.max(own)),
                    "wins_vs_own_baseline": int(np.sum(own < 0.0)),
                    "nonharm_vs_own_baseline": int(np.sum(own <= 1e-12)),
                    "mean_base_vs_official_baseline": float(np.mean(base_off)),
                    "worst_base_vs_official_baseline": float(np.max(base_off)),
                    "mean_corr_vs_official_baseline": float(np.mean(corr_off)),
                    "worst_corr_vs_official_baseline": float(np.max(corr_off)),
                }
            )
    return out


def find(summary: list[dict[str, object]], variant: str, group: str) -> dict[str, object]:
    for row in summary:
        if row["variant"] == variant and row["group"] == group:
            return row
    raise KeyError((variant, group))


def write_report(summary: list[dict[str, object]]) -> None:
    lines = [
        "# Direct MPL Backbone Shape-Projection Audit\n\n",
        "This no-optimization audit projects LD-kernel shape parameters from independent cosine-only MPL fits to cross-scale medians.  It tests whether simple shape stabilization is enough before introducing any more residual modeling.\n\n",
        "| variant | group | backbone params | correction vs own MPL | wins / non-harm | own MPL vs official | corrected vs official |\n",
        "|---|---|---:|---:|---:|---:|---:|\n",
    ]
    for variant in [
        "frozen_official",
        "cosine_independent",
        "median_beta_gamma_projected",
        "median_c_beta_gamma_projected",
    ]:
        for group_label, group in [("WSD-family", "core_wsd"), ("controls", "extra_control")]:
            row = find(summary, variant, group)
            lines.append(
                f"| {variant} | {group_label} | {int(row['backbone_params'])} | "
                f"{fmt_pct(float(row['mean_delta_vs_own_baseline']))} mean / "
                f"{fmt_pct(float(row['worst_delta_vs_own_baseline']))} worst | "
                f"{int(row['wins_vs_own_baseline'])}/{int(row['rows'])} / "
                f"{int(row['nonharm_vs_own_baseline'])}/{int(row['rows'])} | "
                f"{fmt_pct(float(row['mean_base_vs_official_baseline']))} mean | "
                f"{fmt_pct(float(row['mean_corr_vs_official_baseline']))} mean |\n"
            )
    lines += [
        "\n## Reading\n\n",
        "- This audit has zero optimization after the original cosine-only MPL fits.\n",
        "- If median projection hurts, simple cross-scale LD-shape stabilization is not enough and should not be used as a main result.\n",
        "- If it helps, the next step is a controlled, efficient backbone refit around the projected shape.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    params = projected_params(load_independent_params())
    details = detail_rows(params)
    summary = aggregate(details)
    write_csv(OUT_DIR / "params.csv", param_rows(params))
    write_csv(OUT_DIR / "details.csv", details)
    write_csv(OUT_DIR / "summary.csv", summary)
    write_report(summary)
    print(f"wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
