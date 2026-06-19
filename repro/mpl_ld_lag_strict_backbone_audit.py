#!/usr/bin/env python3
"""Strict-backbone audit for the MPL-LD finite-response correction.

The main finite-response audit freezes MPL at ``MPL_PRECOMPUTED_INIT``.  Those
parameters come from the official public split, not from a strict cosine-only
protocol.  This script keeps the correction formula unchanged and changes only
the MPL backbone source:

    1. frozen_official: existing public MPL parameters;
    2. cosine_only: MPL refit on cosine_24000 + cosine_72000.

If the correction only looks good with the frozen official backbone, the result
is a mechanism diagnostic rather than a clean cosine-to-WSD deployment claim.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
ROOT = REPO.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from reproduce_cosine_to_wsd import (  # noqa: E402
    MPL_PRECOMPUTED_INIT,
    SCALES,
    TRAIN_CURVES,
    WARMUP,
    fit_mpl,
    load_curve,
    metrics,
    mpl_predict,
    subsample_curve,
)


OUT_DIR = ROOT / "results" / "mpl_ld_lag_response_audit" / "strict_cosine_backbone"
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


def load_or_fit_cosine_only_params() -> dict[str, np.ndarray]:
    if STRICT_PARAM_JSON.exists():
        data = json.loads(STRICT_PARAM_JSON.read_text(encoding="utf-8"))
        return {scale: np.array(values, dtype=np.float64) for scale, values in data.items()}

    params: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for scale in SCALES:
        train = [subsample_curve(load_curve(scale, name)) for name in TRAIN_CURVES]
        fitted, obj = fit_mpl(train, scale)
        params[scale] = fitted
        rows.append(
            {
                "scale": scale,
                "objective": obj,
                **{f"p{i}": float(value) for i, value in enumerate(fitted)},
            }
        )
    write_csv(OUT_DIR / "cosine_only_mpl_params.csv", rows)
    (OUT_DIR / "cosine_only_mpl_params.json").write_text(
        json.dumps({scale: params[scale].tolist() for scale in SCALES}, indent=2),
        encoding="utf-8",
    )
    return params


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
    span = cooldown_support_span(lrs)
    post_warmup = max(len(lrs) - WARMUP, 1)
    return max(0.0, 1.0 - float(span) / float(post_warmup))


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


def ld_component(curve, params: np.ndarray, component: str) -> np.ndarray:
    _, _, _, _, c_value, beta, gamma = params
    lrs = curve.lrs.astype(np.float64)
    lr_sum = np.cumsum(lrs)
    lr_gap = np.zeros(len(lrs), dtype=np.float64)
    lr_gap[1:] = np.diff(lrs)
    if component == "cooldown":
        selected_gap = np.minimum(lr_gap, 0.0)
    elif component == "full":
        selected_gap = lr_gap
    else:
        raise ValueError(f"unsupported component: {component}")

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


def finite_response_prediction(curve, params: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    baseline = mpl_predict(params, curve)
    d_down = ld_component(curve, params, "cooldown")
    tau = support_bracket_tau(curve.step, curve.lrs)
    factor = adiabatic_factor(curve.lrs)
    d_lag = lagged_observed(d_down, curve.step, tau)
    correction = factor * float(params[3]) * (d_lag - d_down)
    return baseline + correction, {
        "effective_tau_steps": tau,
        "adiabatic_factor": factor,
        "cooldown_support_span": float(cooldown_support_span(curve.lrs)),
        "delta_obs": float(modal_observation_interval(curve.step)),
    }


def detail_rows(strict_params: dict[str, np.ndarray]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    protocols = [
        ("frozen_official", "official_public_split", MPL_PRECOMPUTED_INIT),
        ("cosine_only", "cosine_24000+cosine_72000", strict_params),
    ]
    for scale in SCALES:
        official_by_curve: dict[str, float] = {}
        for _, curve_name, _ in ALL_TARGETS:
            curve = load_curve(scale, curve_name)
            official_by_curve[curve_name] = metrics(
                curve.loss,
                mpl_predict(MPL_PRECOMPUTED_INIT[scale], curve),
            )["mae"]

        for protocol, backbone_train, param_map in protocols:
            params = np.array(param_map[scale], dtype=np.float64)
            for group, curve_name, label in ALL_TARGETS:
                curve = load_curve(scale, curve_name)
                baseline = mpl_predict(params, curve)
                pred, features = finite_response_prediction(curve, params)
                base_mae = mae(curve.loss, baseline)
                corr_mae = mae(curve.loss, pred)
                official_base_mae = official_by_curve[curve_name]
                rows.append(
                    {
                        "protocol": protocol,
                        "backbone_train": backbone_train,
                        "group": group,
                        "scale": scale,
                        "test_curve": curve_name,
                        "test_label": label,
                        "base_mae": base_mae,
                        "corr_mae": corr_mae,
                        "delta_vs_own_baseline_pct": 100.0 * (corr_mae / base_mae - 1.0),
                        "base_vs_official_baseline_pct": 100.0 * (base_mae / official_base_mae - 1.0),
                        "corr_vs_official_baseline_pct": 100.0 * (corr_mae / official_base_mae - 1.0),
                        "win_vs_own_baseline": int(corr_mae < base_mae),
                        "nonharm_vs_own_baseline": int(corr_mae <= base_mae + 1e-12),
                        **features,
                    }
                )
    return rows


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    keys = sorted({(str(row["protocol"]), str(row["group"])) for row in rows})
    for protocol, group in keys:
        sub = [row for row in rows if row["protocol"] == protocol and row["group"] == group]
        delta = np.array([float(row["delta_vs_own_baseline_pct"]) for row in sub], dtype=np.float64)
        base_vs_official = np.array([float(row["base_vs_official_baseline_pct"]) for row in sub], dtype=np.float64)
        corr_vs_official = np.array([float(row["corr_vs_official_baseline_pct"]) for row in sub], dtype=np.float64)
        out.append(
            {
                "protocol": protocol,
                "group": group,
                "rows": len(sub),
                "mean_delta_vs_own_baseline": float(np.mean(delta)),
                "worst_delta_vs_own_baseline": float(np.max(delta)),
                "wins_vs_own_baseline": int(np.sum(delta < 0.0)),
                "nonharm_vs_own_baseline": int(np.sum(delta <= 1e-12)),
                "mean_base_vs_official_baseline": float(np.mean(base_vs_official)),
                "worst_base_vs_official_baseline": float(np.max(base_vs_official)),
                "mean_corr_vs_official_baseline": float(np.mean(corr_vs_official)),
                "worst_corr_vs_official_baseline": float(np.max(corr_vs_official)),
            }
        )
    return out


def find(summary: list[dict[str, object]], protocol: str, group: str) -> dict[str, object]:
    for row in summary:
        if row["protocol"] == protocol and row["group"] == group:
            return row
    raise KeyError((protocol, group))


def write_report(summary: list[dict[str, object]], details: list[dict[str, object]]) -> None:
    official_core = find(summary, "frozen_official", "core_wsd")
    official_ctrl = find(summary, "frozen_official", "extra_control")
    strict_core = find(summary, "cosine_only", "core_wsd")
    strict_ctrl = find(summary, "cosine_only", "extra_control")
    strict_wsd = [row for row in details if row["protocol"] == "cosine_only" and row["group"] == "core_wsd"]

    lines = [
        "# Strict Cosine-Only Backbone Audit for MPL-LD Finite Response\n\n",
        "This audit keeps the finite-response correction fixed and changes only the MPL backbone source.  "
        "It separates mechanism evidence from protocol evidence.\n\n",
        "Recommended correction under audit:\n\n",
        "\\[\n",
        "\\hat L_s(t)=L_{\\mathrm{MPL},s}(t)+a_sB_s[D_{\\downarrow,\\tau_s,s}(t)-D_{\\downarrow,s}(t)].\n",
        "\\]\n\n",
        "No residual amplitude, gate, channel selector, sinusoid, DCT basis, or target-loss-fitted parameter is used here.\n\n",
        "## Summary\n\n",
        "| backbone | group | correction vs own MPL | wins / non-harm | own MPL vs official MPL | corrected vs official MPL |\n",
        "|---|---|---:|---:|---:|---:|\n",
    ]
    for label, core, ctrl in [
        ("official frozen MPL", official_core, official_ctrl),
        ("cosine-only MPL", strict_core, strict_ctrl),
    ]:
        for group_label, row in [("WSD-family", core), ("controls", ctrl)]:
            lines.append(
                f"| {label} | {group_label} | "
                f"{fmt_pct(float(row['mean_delta_vs_own_baseline']))} mean / "
                f"{fmt_pct(float(row['worst_delta_vs_own_baseline']))} worst | "
                f"{int(row['wins_vs_own_baseline'])}/{int(row['rows'])} / "
                f"{int(row['nonharm_vs_own_baseline'])}/{int(row['rows'])} | "
                f"{fmt_pct(float(row['mean_base_vs_official_baseline']))} mean | "
                f"{fmt_pct(float(row['mean_corr_vs_official_baseline']))} mean |\n"
            )

    lines += [
        "\n## Strict Backbone Per-Target WSD Rows\n\n",
        "| scale | target | own MPL MAE | corrected MAE | correction delta | corrected vs official MPL |\n",
        "|---:|---|---:|---:|---:|---:|\n",
    ]
    for row in strict_wsd:
        lines.append(
            f"| {row['scale']} | {row['test_label']} | "
            f"{float(row['base_mae']):.6f} | {float(row['corr_mae']):.6f} | "
            f"{fmt_pct(float(row['delta_vs_own_baseline_pct']))} | "
            f"{fmt_pct(float(row['corr_vs_official_baseline_pct']))} |\n"
        )

    lines += [
        "\n## Reading\n\n",
        "- The correction formula itself is still parameter-free on top of MPL: all new quantities come from the LR schedule, the logging interval, and MPL's own \\(D_\\downarrow\\) term.\n",
        "- The frozen-official result should be treated as a mechanism diagnostic, because the MPL backbone was not trained under a strict cosine-only split.\n",
        "- The strict cosine-only rows are the fairer protocol for the assignment question.  If they are weaker than the frozen-official rows, the honest conclusion is that the current formula helps but is not yet a complete cosine-to-WSD solution.\n",
        "- This audit intentionally does not add extra fitted residual parameters to recover performance; doing so would reintroduce the interpretability problem this audit is meant to expose.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    strict_params = load_or_fit_cosine_only_params()
    details = detail_rows(strict_params)
    summary = aggregate(details)
    write_csv(OUT_DIR / "details.csv", details)
    write_csv(OUT_DIR / "summary.csv", summary)
    write_report(summary, details)
    print(f"wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
