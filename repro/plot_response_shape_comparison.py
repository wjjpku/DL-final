#!/usr/bin/env python3
"""Compare residual response shapes across LR schedules.

This diagnostic is motivated by the observation that a real LR-drop relaxation
should catch up within a finite number of optimization steps.  If a correction
keeps lagging through most of a cosine decay, it is likely fitting low-frequency
MPL mismatch rather than a transferable decay transient.
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
ROOT = REPO.parent
sys.path.insert(0, str(REPO))

from current_law_decay_matrix import LAMBDA  # noqa: E402
from reproduce_cosine_to_wsd import (  # noqa: E402
    MPL_PRECOMPUTED_INIT,
    PEAK_LR,
    SCALES,
    load_curve,
    metrics,
    mpl_predict,
)


OUT_DIR = ROOT / "results" / "step_time_robust_matrix" / "response_shapes"
CURVES = [
    ("constant_72000.csv", "constant 72k"),
    ("cosine_72000.csv", "cosine 72k"),
    ("cosine_24000.csv", "cosine 24k"),
    ("wsd_20000_24000.csv", "WSD exp cooldown"),
    ("wsdld_20000_24000.csv", "WSD linear cooldown"),
    ("wsdcon_3.csv", "step to 3e-5"),
    ("wsdcon_9.csv", "step to 9e-5"),
    ("wsdcon_18.csv", "step to 18e-5"),
]
FEATURES = [
    ("S10_current", "s_time", LAMBDA, "#2563eb", "--"),
    ("step_tau1024", "step_time", 1024.0, "#059669", "-"),
    ("step_tau2304", "step_time", 2304.0, "#d97706", "-."),
]
PEAK_MIN_STEP = 5000


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


def fit_origin_nonnegative(x: np.ndarray, y: np.ndarray) -> float:
    denom = float(np.dot(x, x))
    if denom <= 1e-18:
        return 0.0
    return max(0.0, float(np.dot(x, y) / denom))


def smooth(y: np.ndarray) -> np.ndarray:
    if len(y) < 7:
        return y.copy()
    window = max(5, int(round(0.03 * len(y))))
    if window % 2 == 0:
        window += 1
    window = min(window, len(y) if len(y) % 2 == 1 else len(y) - 1)
    if window < 3:
        return y.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(y, kernel, mode="same")


def mass_width_steps(steps: np.ndarray, values: np.ndarray) -> float:
    weights = np.maximum(values.astype(np.float64), 0.0)
    total = float(np.sum(weights))
    if total <= 1e-18:
        return float("nan")
    cdf = np.cumsum(weights) / total
    lo = min(int(np.searchsorted(cdf, 0.05, side="left")), len(steps) - 1)
    hi = min(int(np.searchsorted(cdf, 0.95, side="left")), len(steps) - 1)
    return float(steps[hi] - steps[lo])


def lowfreq_r2(steps: np.ndarray, y: np.ndarray) -> float:
    if len(y) < 8:
        return float("nan")
    y = smooth(y.astype(np.float64))
    span = float(steps[-1] - steps[0])
    if span <= 0:
        return float("nan")
    z = (steps.astype(np.float64) - float(steps[0])) / span
    basis = [
        np.ones_like(z),
        np.sin(np.pi * z),
        np.cos(np.pi * z),
        np.sin(2.0 * np.pi * z),
        np.cos(2.0 * np.pi * z),
    ]
    x = np.vstack(basis).T
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    fit = x @ beta
    denom = float(np.sum((y - float(np.mean(y))) ** 2))
    if denom <= 1e-18:
        return float("nan")
    return 1.0 - float(np.sum((y - fit) ** 2) / denom)


def peak_step_after_warmup(steps: np.ndarray, y: np.ndarray) -> float:
    mask = steps >= PEAK_MIN_STEP
    if not np.any(mask):
        mask = np.ones_like(steps, dtype=bool)
    idxs = np.flatnonzero(mask)
    return float(steps[int(idxs[np.argmax(y[mask])])])


def analyze_curve(scale: str, curve_name: str, label: str) -> dict[str, object]:
    curve = load_curve(scale, curve_name)
    base = mpl_predict(MPL_PRECOMPUTED_INIT[scale], curve)
    residual = curve.loss - base
    residual_s = smooth(residual)
    resid_peak = peak_step_after_warmup(curve.step, residual_s)
    base_mae = metrics(curve.loss, base)["mae"]
    feature_rows: dict[str, dict[str, float | np.ndarray]] = {}
    for name, kind, param, _, _ in FEATURES:
        phi = response_feature(curve, kind, param)
        kappa = fit_origin_nonnegative(phi, residual)
        estimate = kappa * phi
        corrected = base + estimate
        remaining = curve.loss - corrected
        if float(np.max(phi)) > 1e-18:
            phi_peak = float(curve.step[int(np.argmax(phi))])
            width = mass_width_steps(curve.step, estimate)
            width_frac = width / float(curve.step[-1] - curve.step[0])
        else:
            phi_peak = float("nan")
            width = float("nan")
            width_frac = float("nan")
        feature_rows[name] = {
            "phi": phi,
            "kappa": kappa,
            "estimate": estimate,
            "remaining": remaining,
            "corr_mae": metrics(curve.loss, corrected)["mae"],
            "delta_pct": 100.0 * (metrics(curve.loss, corrected)["mae"] / base_mae - 1.0),
            "phi_peak_step": phi_peak,
            "peak_lag_steps": phi_peak - resid_peak if math.isfinite(phi_peak) else float("nan"),
            "feature_width_steps": width,
            "feature_width_frac": width_frac,
            "remaining_lowfreq_r2": lowfreq_r2(curve.step, remaining),
        }
    return {
        "scale": scale,
        "curve_name": curve_name,
        "label": label,
        "curve": curve,
        "base": base,
        "residual": residual,
        "residual_smooth": residual_s,
        "resid_peak_step": resid_peak,
        "base_mae": base_mae,
        "raw_lowfreq_r2": lowfreq_r2(curve.step, residual),
        "features": feature_rows,
    }


def plot_shape_grid(scale: str, rows: list[dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(14.2, 13.2), constrained_layout=True)
    for i, row in enumerate(rows):
        ax = axes.ravel()[i]
        curve = row["curve"]
        steps = curve.step
        residual = row["residual"]
        residual_s = row["residual_smooth"]
        ax.axhline(0.0, color="#333333", lw=0.75, alpha=0.75)
        ax.plot(steps, residual, color="#9ca3af", lw=0.7, alpha=0.55, label="MPL residual")
        ax.plot(steps, residual_s, color="#111111", lw=1.35, label="smoothed residual")
        for name, _, _, color, ls in FEATURES:
            frow = row["features"][name]
            if float(frow["kappa"]) > 0.0:
                ax.plot(steps, frow["estimate"], color=color, lw=1.2, ls=ls, label=name)
        ax.axvline(float(row["resid_peak_step"]), color="#111111", lw=0.7, ls=":", alpha=0.65)

        ax2 = ax.twinx()
        ax2.plot(steps, curve.lrs[curve.step] / PEAK_LR, color="#b45309", lw=0.9, alpha=0.34)
        ax2.set_ylim(-0.04, 1.06)
        ax2.tick_params(labelsize=7, colors="#92400e")

        s10 = row["features"]["S10_current"]
        st = row["features"]["step_tau1024"]
        title = (
            f"{row['label']} | "
            f"S10 lag {fmt(s10['peak_lag_steps'], '+.0f')}, "
            f"tau1024 lag {fmt(st['peak_lag_steps'], '+.0f')}"
        )
        ax.set_title(title, fontsize=9.4)
        ax.set_xlabel("step", fontsize=8)
        ax.set_ylabel("loss residual", fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.grid(alpha=0.22, lw=0.5)
        if i == 0:
            ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle(
        f"{scale}M: MPL residual vs old cumulative-LR and finite step-time responses",
        fontsize=13.5,
    )
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def plot_remaining_grid(scale: str, rows: list[dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(14.2, 13.2), constrained_layout=True)
    for i, row in enumerate(rows):
        ax = axes.ravel()[i]
        curve = row["curve"]
        steps = curve.step
        ax.axhline(0.0, color="#333333", lw=0.75, alpha=0.75)
        ax.plot(steps, smooth(row["residual"]), color="#111111", lw=1.25, label="MPL residual")
        for name, _, _, color, ls in FEATURES:
            frow = row["features"][name]
            if float(frow["kappa"]) > 0.0:
                ax.plot(steps, smooth(frow["remaining"]), color=color, lw=1.05, ls=ls, label=f"remaining after {name}")
        title = (
            f"{row['label']} | raw wave R2 {fmt(row['raw_lowfreq_r2'], '.2f')}, "
            f"S10 rem {fmt(row['features']['S10_current']['remaining_lowfreq_r2'], '.2f')}, "
            f"tau1024 rem {fmt(row['features']['step_tau1024']['remaining_lowfreq_r2'], '.2f')}"
        )
        ax.set_title(title, fontsize=9.2)
        ax.set_xlabel("step", fontsize=8)
        ax.set_ylabel("remaining residual", fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.grid(alpha=0.22, lw=0.5)
        if i == 0:
            ax.legend(frameon=False, fontsize=7.2, loc="upper left")
    fig.suptitle(f"{scale}M: remaining error after self-fit response corrections", fontsize=13.5)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def plot_summary(all_rows: list[dict[str, object]], path: Path) -> None:
    labels = [label for _, label in CURVES]
    x = np.arange(len(labels))

    def mean_for(curve_name: str, feature: str, key: str) -> float:
        vals = []
        for row in all_rows:
            if row["curve_name"] == curve_name:
                vals.append(float(row["features"][feature][key]))
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        return float(np.mean(arr)) if len(arr) else float("nan")

    def mean_raw(curve_name: str, key: str) -> float:
        vals = [float(row[key]) for row in all_rows if row["curve_name"] == curve_name]
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        return float(np.mean(arr)) if len(arr) else float("nan")

    fig, axes = plt.subplots(3, 1, figsize=(13.8, 10.5), constrained_layout=True)
    width = 0.27
    ax = axes[0]
    ax.axhline(0.0, color="#333333", lw=0.8)
    for offset, (name, _, _, color, _) in zip([-width, 0.0, width], FEATURES):
        ax.bar(x + offset, [mean_for(c, name, "peak_lag_steps") for c, _ in CURVES], width, label=name, color=color)
    ax.set_title("Response peak step minus residual peak step")
    ax.set_ylabel("steps")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.24)
    ax.legend(frameon=False, ncol=3)

    ax = axes[1]
    for offset, (name, _, _, color, _) in zip([-width, 0.0, width], FEATURES):
        ax.bar(
            x + offset,
            [100.0 * mean_for(c, name, "feature_width_frac") for c, _ in CURVES],
            width,
            label=name,
            color=color,
        )
    ax.set_title("Response 5-95% mass width")
    ax.set_ylabel("% of curve span")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.24)

    ax = axes[2]
    ax.bar(x - width, [mean_raw(c, "raw_lowfreq_r2") for c, _ in CURVES], width, label="raw MPL residual", color="#111111")
    ax.bar(
        x,
        [mean_for(c, "S10_current", "remaining_lowfreq_r2") for c, _ in CURVES],
        width,
        label="remaining after S10",
        color="#2563eb",
    )
    ax.bar(
        x + width,
        [mean_for(c, "step_tau1024", "remaining_lowfreq_r2") for c, _ in CURVES],
        width,
        label="remaining after tau1024",
        color="#059669",
    )
    ax.set_title("Low-frequency sinusoidal share of residual")
    ax.set_ylabel("R2 by first two Fourier modes")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.24)
    ax.legend(frameon=False, ncol=3)

    fig.suptitle("Response-shape diagnostics averaged across scales", fontsize=13.5)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def fmt(value: object, spec: str) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(v):
        return "n/a"
    return format(v, spec)


def flatten_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for row in rows:
        for name, _, _, _, _ in FEATURES:
            frow = row["features"][name]
            out.append(
                {
                    "scale": row["scale"],
                    "curve_name": row["curve_name"],
                    "label": row["label"],
                    "feature": name,
                    "base_mae": row["base_mae"],
                    "corr_mae": frow["corr_mae"],
                    "delta_pct": frow["delta_pct"],
                    "kappa": frow["kappa"],
                    "resid_peak_step": row["resid_peak_step"],
                    "phi_peak_step": frow["phi_peak_step"],
                    "peak_lag_steps": frow["peak_lag_steps"],
                    "feature_width_steps": frow["feature_width_steps"],
                    "feature_width_frac": frow["feature_width_frac"],
                    "raw_lowfreq_r2": row["raw_lowfreq_r2"],
                    "remaining_lowfreq_r2": frow["remaining_lowfreq_r2"],
                }
            )
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "scale",
        "curve_name",
        "label",
        "feature",
        "base_mae",
        "corr_mae",
        "delta_pct",
        "kappa",
        "resid_peak_step",
        "phi_peak_step",
        "peak_lag_steps",
        "feature_width_steps",
        "feature_width_frac",
        "raw_lowfreq_r2",
        "remaining_lowfreq_r2",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_metric(rows: list[dict[str, object]], curve_name: str, feature: str, key: str) -> float:
    vals = [
        float(row[key])
        for row in rows
        if row["curve_name"] == curve_name and row["feature"] == feature
    ]
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else float("nan")


def write_report(path: Path, flat_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Response Shape Comparison\n\n",
        "This diagnostic compares the original cumulative-LR response with finite step-time responses. "
        "It focuses on whether the residual behaves like a local LR-drop catch-up transient or like a broad low-frequency MPL mismatch.\n\n",
        "## Figures\n\n",
    ]
    for scale in SCALES:
        lines.append(f"- `{scale}M` response shape: `response_shape_comparison_{scale}M.png`\n")
        lines.append(f"- `{scale}M` remaining error: `response_remaining_error_{scale}M.png`\n")
    lines.append("- Cross-scale metric summary: `response_shape_metric_summary.png`\n\n")

    lines += [
        "## Mean Metrics Across Scales\n\n",
        "| schedule | S10 lag | tau1024 lag | S10 width | tau1024 width | raw low-freq R2 | S10 delta | tau1024 delta |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for curve_name, label in CURVES:
        s10_lag = mean_metric(flat_rows, curve_name, "S10_current", "peak_lag_steps")
        st_lag = mean_metric(flat_rows, curve_name, "step_tau1024", "peak_lag_steps")
        s10_width = mean_metric(flat_rows, curve_name, "S10_current", "feature_width_frac")
        st_width = mean_metric(flat_rows, curve_name, "step_tau1024", "feature_width_frac")
        raw_r2 = mean_metric(flat_rows, curve_name, "S10_current", "raw_lowfreq_r2")
        s10_delta = mean_metric(flat_rows, curve_name, "S10_current", "delta_pct")
        st_delta = mean_metric(flat_rows, curve_name, "step_tau1024", "delta_pct")
        lines.append(
            f"| {label} | {fmt(s10_lag, '+.0f')} | {fmt(st_lag, '+.0f')} | "
            f"{fmt(100.0 * s10_width, '.1f')}% | {fmt(100.0 * st_width, '.1f')}% | "
            f"{fmt(raw_r2, '.2f')} | {fmt(s10_delta, '+.1f')}% | {fmt(st_delta, '+.1f')}% |\n"
        )

    lines += [
        "\n## Reading\n\n",
        "- Cosine is the outlier: its old cumulative-LR response is both late and wide, so a same-curve kappa can absorb a global sinusoidal MPL residual rather than a local decay transient.\n",
        "- WSD schedules have localized cooldowns. The finite step-time response follows those changes with much shorter memory, which matches the idea that the loss should catch up after a finite delay.\n",
        "- Step probes are useful calibration curves because their LR perturbation is identifiable. They do not let a smooth low-frequency residual masquerade as a schedule response as easily as cosine does.\n",
        "- Therefore, cosine should remain a diagnostic for nuisance structure, while kappa calibration should prefer sharp or endpoint-matched decay probes for WSD-style targets.\n",
    ]
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for scale in SCALES:
        rows = [analyze_curve(scale, curve_name, label) for curve_name, label in CURVES]
        all_rows.extend(rows)
        plot_shape_grid(scale, rows, OUT_DIR / f"response_shape_comparison_{scale}M.png")
        plot_remaining_grid(scale, rows, OUT_DIR / f"response_remaining_error_{scale}M.png")
    flat_rows = flatten_rows(all_rows)
    plot_summary(all_rows, OUT_DIR / "response_shape_metric_summary.png")
    write_csv(OUT_DIR / "response_shape_metrics.csv", flat_rows)
    write_report(OUT_DIR / "REPORT.md", flat_rows)

    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    print(f"wrote {OUT_DIR / 'response_shape_metrics.csv'}")
    print(f"wrote {OUT_DIR / 'response_shape_metric_summary.png'}")
    for curve_name, label in CURVES:
        s10_lag = mean_metric(flat_rows, curve_name, "S10_current", "peak_lag_steps")
        st_lag = mean_metric(flat_rows, curve_name, "step_tau1024", "peak_lag_steps")
        s10_width = mean_metric(flat_rows, curve_name, "S10_current", "feature_width_frac")
        st_width = mean_metric(flat_rows, curve_name, "step_tau1024", "feature_width_frac")
        print(
            f"{label:22s} lag S10/tau1024={fmt(s10_lag, '+.0f')}/{fmt(st_lag, '+.0f')} "
            f"width={fmt(100*s10_width, '.1f')}%/{fmt(100*st_width, '.1f')}%"
        )


if __name__ == "__main__":
    main()
