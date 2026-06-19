#!/usr/bin/env python3
"""Residual-shape gallery across schedule types.

This diagnostic compares MPL residuals with a same-curve DropRelaxS self-fit
across several LR schedules.  The goal is not to claim transfer performance; it
is to inspect whether the residual shape looks like a local LR-drop relaxation
or a broader low-frequency MPL mismatch.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent
ROOT = REPO.parent
sys.path.insert(0, str(REPO))

from current_law_decay_matrix import LAMBDA  # noqa: E402
from deep_stime import stime_feature  # noqa: E402
from reproduce_cosine_to_wsd import (  # noqa: E402
    MPL_PRECOMPUTED_INIT,
    PEAK_LR,
    SCALES,
    load_curve,
    metrics,
    mpl_predict,
)


OUT_DIR = ROOT / "results" / "current_law_decay_matrix" / "error_visualization"
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
PEAK_MIN_STEP = 5000


def fit_origin_nonnegative(x: np.ndarray, y: np.ndarray) -> float:
    denom = float(np.dot(x, x))
    if denom <= 1e-18:
        return 0.0
    return max(0.0, float(np.dot(x, y) / denom))


def r2_origin(y: np.ndarray, yhat: np.ndarray) -> float:
    denom = float(np.dot(y, y))
    if denom <= 1e-18:
        return float("nan")
    return 1.0 - float(np.dot(y - yhat, y - yhat) / denom)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - float(np.mean(x))
    yc = y - float(np.mean(y))
    denom = float(np.linalg.norm(xc) * np.linalg.norm(yc))
    if denom <= 1e-18:
        return float("nan")
    return float(np.dot(xc, yc) / denom)


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


def mass_width_steps(steps: np.ndarray, weights: np.ndarray) -> float:
    w = np.maximum(weights.astype(np.float64), 0.0)
    total = float(np.sum(w))
    if total <= 1e-18:
        return float("nan")
    cdf = np.cumsum(w) / total
    lo = int(np.searchsorted(cdf, 0.05, side="left"))
    hi = int(np.searchsorted(cdf, 0.95, side="left"))
    lo = min(max(lo, 0), len(steps) - 1)
    hi = min(max(hi, 0), len(steps) - 1)
    return float(steps[hi] - steps[lo])


def decrement_stats(lrs: np.ndarray) -> dict[str, float]:
    drops = np.maximum(lrs[:-1] - lrs[1:], 0.0)
    total = float(np.sum(drops))
    if total <= 1e-18:
        return {"drop_total": 0.0, "drop_concentration": 0.0, "drop_neff": 0.0}
    return {
        "drop_total": total,
        "drop_concentration": float(np.max(drops) / total),
        "drop_neff": float(total * total / max(float(np.sum(drops * drops)), 1e-18)),
    }


def finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    return float(np.mean(arr))


def fmt_float(value: float, spec: str, nan_text: str = "n/a") -> str:
    if not math.isfinite(float(value)):
        return nan_text
    return format(float(value), spec)


def analyze_curve(scale: str, curve_name: str, label: str) -> dict[str, object]:
    curve = load_curve(scale, curve_name)
    mpl = mpl_predict(MPL_PRECOMPUTED_INIT[scale], curve)
    residual = curve.loss - mpl
    phi = stime_feature(curve, LAMBDA)
    kappa = fit_origin_nonnegative(phi, residual)
    estimate = kappa * phi
    corrected = mpl + estimate
    base_mae = metrics(curve.loss, mpl)["mae"]
    corrected_mae = metrics(curve.loss, corrected)["mae"]

    residual_s = smooth(residual)
    mask = curve.step >= PEAK_MIN_STEP
    if int(np.sum(mask)) == 0:
        mask = np.ones_like(curve.step, dtype=bool)
    masked_idx = np.flatnonzero(mask)
    resid_peak_idx = int(masked_idx[np.argmax(residual_s[mask])])
    phi_peak_idx = int(np.argmax(phi)) if float(np.max(phi)) > 1e-18 else -1
    resid_peak_step = float(curve.step[resid_peak_idx])
    phi_peak_step = float(curve.step[phi_peak_idx]) if phi_peak_idx >= 0 else float("nan")
    peak_lag = phi_peak_step - resid_peak_step if phi_peak_idx >= 0 else float("nan")
    total_span = float(curve.step[-1] - curve.step[0])
    feature_width = mass_width_steps(curve.step, phi)
    residual_width = mass_width_steps(curve.step, np.maximum(residual_s, 0.0))
    dstat = decrement_stats(curve.lrs)

    return {
        "scale": scale,
        "curve_name": curve_name,
        "label": label,
        "curve": curve,
        "mpl": mpl,
        "residual": residual,
        "residual_smooth": residual_s,
        "phi": phi,
        "estimate": estimate,
        "remaining": curve.loss - corrected,
        "kappa": kappa,
        "base_mae": base_mae,
        "corrected_mae": corrected_mae,
        "delta_pct": 100.0 * (corrected_mae / base_mae - 1.0) if base_mae > 0 else float("nan"),
        "origin_r2": r2_origin(residual, estimate),
        "pearson": pearson(residual, estimate),
        "resid_peak_step": resid_peak_step,
        "phi_peak_step": phi_peak_step,
        "peak_lag_steps": peak_lag,
        "feature_width_steps": feature_width,
        "feature_width_frac": feature_width / total_span if total_span > 0 and math.isfinite(feature_width) else float("nan"),
        "positive_residual_width_steps": residual_width,
        "positive_residual_width_frac": residual_width / total_span if total_span > 0 and math.isfinite(residual_width) else float("nan"),
        **dstat,
    }


def plot_scale_gallery(scale: str, rows: list[dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(13.8, 13.0), sharex=False, constrained_layout=True)
    axes_flat = axes.ravel()
    for i, row in enumerate(rows):
        ax = axes_flat[i]
        curve = row["curve"]
        steps = curve.step
        residual = row["residual"]
        estimate = row["estimate"]
        lr = curve.lrs[curve.step] / PEAK_LR

        ax.axhline(0.0, color="#333333", lw=0.7, alpha=0.7)
        ax.plot(steps, residual, color="#111111", lw=1.15, label="MPL residual")
        if float(np.max(row["phi"])) > 1e-18:
            ax.plot(steps, estimate, color="#2563eb", lw=1.15, ls="--", label="kappa * DropRelaxS")
            ax.axvline(float(row["resid_peak_step"]), color="#111111", lw=0.75, ls=":", alpha=0.65)
            ax.axvline(float(row["phi_peak_step"]), color="#2563eb", lw=0.75, ls=":", alpha=0.65)
        else:
            ax.text(
                0.03,
                0.12,
                "no positive LR-drop feature",
                transform=ax.transAxes,
                fontsize=8,
                color="#555555",
                ha="left",
                va="bottom",
            )

        ax2 = ax.twinx()
        ax2.plot(steps, lr, color="#d97706", lw=0.9, alpha=0.38, label="LR / peak")
        ax2.set_ylim(-0.04, 1.06)
        ax2.tick_params(labelsize=7, colors="#9a5b0a")
        ax2.set_ylabel("LR/peak", fontsize=7, color="#9a5b0a")

        lag = row["peak_lag_steps"]
        lag_text = "nan" if not math.isfinite(float(lag)) else f"{float(lag):+.0f}"
        width = row["feature_width_frac"]
        width_text = "nan" if not math.isfinite(float(width)) else f"{100*float(width):.0f}%"
        ax.set_title(
            f"{row['label']} | k={float(row['kappa']):.3f}, "
            f"R2={float(row['origin_r2']):.2f}, lag={lag_text}, width={width_text}",
            fontsize=9.5,
        )
        ax.set_xlabel("step", fontsize=8)
        ax.set_ylabel("residual", fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.grid(alpha=0.22, lw=0.5)
        if i == 0:
            ax.legend(frameon=False, fontsize=7.5, loc="upper left")

    fig.suptitle(
        f"{scale}M residual-shape gallery: MPL residual vs same-curve DropRelaxS self-fit",
        fontsize=13.5,
    )
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def plot_summary(rows: list[dict[str, object]], path: Path) -> None:
    labels = [label for _, label in CURVES]
    x = np.arange(len(labels))
    means: dict[str, list[float]] = {
        "delta": [],
        "kappa": [],
        "lag": [],
        "width": [],
    }
    for curve_name, _ in CURVES:
        sub = [row for row in rows if row["curve_name"] == curve_name]
        means["delta"].append(finite_mean([float(row["delta_pct"]) for row in sub]))
        means["kappa"].append(finite_mean([float(row["kappa"]) for row in sub]))
        means["lag"].append(finite_mean([float(row["peak_lag_steps"]) for row in sub]))
        means["width"].append(finite_mean([float(row["feature_width_frac"]) for row in sub]))

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.4), constrained_layout=True)
    ax = axes[0, 0]
    ax.axhline(0.0, color="#333333", lw=0.8)
    ax.bar(x, means["delta"], color="#2563eb")
    ax.set_title("Self-fit MAE change")
    ax.set_ylabel("Delta MAE vs MPL (%)")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.24)

    ax = axes[0, 1]
    ax.bar(x, means["kappa"], color="#dc2626")
    ax.set_title("Same-curve fitted kappa")
    ax.set_ylabel("kappa")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.24)

    ax = axes[1, 0]
    ax.axhline(0.0, color="#333333", lw=0.8)
    ax.bar(x, means["lag"], color="#7c3aed")
    ax.set_title("Feature peak minus residual peak")
    ax.set_ylabel("steps")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.24)

    ax = axes[1, 1]
    ax.bar(x, [100.0 * v if math.isfinite(v) else np.nan for v in means["width"]], color="#059669")
    ax.set_title("DropRelaxS feature width")
    ax.set_ylabel("5--95% mass width (% of curve span)")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.24)

    fig.suptitle("Schedule residual-shape summary across 25M/100M/400M", fontsize=13.5)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "scale",
        "curve_name",
        "label",
        "kappa",
        "base_mae",
        "corrected_mae",
        "delta_pct",
        "origin_r2",
        "pearson",
        "resid_peak_step",
        "phi_peak_step",
        "peak_lag_steps",
        "feature_width_steps",
        "feature_width_frac",
        "positive_residual_width_steps",
        "positive_residual_width_frac",
        "drop_total",
        "drop_concentration",
        "drop_neff",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def write_report(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Schedule Residual Gallery\n\n",
        "This diagnostic compares MPL residuals with same-curve DropRelaxS self-fits across schedule types. "
        "Because `kappa` is fitted on the same curve, the plots are shape diagnostics rather than transfer evidence.\n\n",
        "## Mean Metrics Across Scales\n\n",
        "| schedule | mean delta | mean kappa | mean peak lag | mean feature width | mean drop N_eff |\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ]
    for curve_name, label in CURVES:
        sub = [row for row in rows if row["curve_name"] == curve_name]
        mean_delta = finite_mean([float(row["delta_pct"]) for row in sub])
        mean_kappa = finite_mean([float(row["kappa"]) for row in sub])
        mean_lag = finite_mean([float(row["peak_lag_steps"]) for row in sub])
        mean_width = finite_mean([float(row["feature_width_frac"]) for row in sub])
        mean_neff = finite_mean([float(row["drop_neff"]) for row in sub])
        lines.append(
            f"| {label} | {fmt_float(mean_delta, '+.1f')}% | {fmt_float(mean_kappa, '.4f')} | "
            f"{fmt_float(mean_lag, '+.0f')} | {fmt_float(100*mean_width, '.1f')}% | "
            f"{fmt_float(mean_neff, '.1f')} |\n"
        )

    lines += [
        "\n## Reading\n\n",
        "- Constant schedules are a useful control: they can have MPL residual structure even though the positive-drop feature is zero, so not every residual is a non-adiabatic LR-drop lag.\n",
        "- Cosine schedules produce a very diffuse DropRelaxS feature. The feature peak is late relative to the broad MPL residual, matching the visual impression that the correction is lagging and trying to fit a low-frequency wave.\n",
        "- WSD and WSD-con schedules have more localized LR changes; their residuals are better suited for estimating a transient response amplitude.\n",
        "- This supports treating diffuse cosine correction as non-transferable unless a nuisance projection or target-localization gate removes the low-frequency component.\n",
    ]
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for scale in SCALES:
        rows = [analyze_curve(scale, name, label) for name, label in CURVES]
        all_rows.extend(rows)
        plot_scale_gallery(scale, rows, OUT_DIR / f"schedule_residual_gallery_{scale}M.png")
    plot_summary(all_rows, OUT_DIR / "schedule_residual_shape_summary.png")
    write_csv(OUT_DIR / "schedule_residual_shape_metrics.csv", all_rows)
    write_report(OUT_DIR / "SCHEDULE_RESIDUAL_GALLERY.md", all_rows)

    print(f"wrote {OUT_DIR / 'schedule_residual_shape_summary.png'}")
    print(f"wrote {OUT_DIR / 'schedule_residual_shape_metrics.csv'}")
    print(f"wrote {OUT_DIR / 'SCHEDULE_RESIDUAL_GALLERY.md'}")
    for scale in SCALES:
        print(f"wrote {OUT_DIR / f'schedule_residual_gallery_{scale}M.png'}")
    for curve_name, label in CURVES:
        sub = [row for row in all_rows if row["curve_name"] == curve_name]
        mean_delta = finite_mean([float(r["delta_pct"]) for r in sub])
        mean_kappa = finite_mean([float(r["kappa"]) for r in sub])
        mean_lag = finite_mean([float(r["peak_lag_steps"]) for r in sub])
        print(
            f"{label:20s} mean_delta={fmt_float(mean_delta, '+6.1f')}% "
            f"mean_kappa={fmt_float(mean_kappa, '.4f')} "
            f"mean_lag={fmt_float(mean_lag, '+.0f')}"
        )


if __name__ == "__main__":
    main()
