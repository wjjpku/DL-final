#!/usr/bin/env python3
"""Residual plots for adaptive vs suffix-fitted adaptive cosine calibration."""
from __future__ import annotations

import csv
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
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cosine_to_wsd_adaptive_fit_window import fit_source_kappa_window  # noqa: E402
from cosine_to_wsd_adaptive_search import channel_for_curve  # noqa: E402
from cosine_to_wsd_response_search import (  # noqa: E402
    TARGETS,
    TARGET_RETENTION_FLOOR,
    TRAIN_CURVE,
    build_cache,
    fit_source_kappa,
    stime_feature,
    target_retention,
)
from reproduce_cosine_to_wsd import MPL_PRECOMPUTED_INIT, PEAK_LR, SCALES, load_curve, metrics, mpl_predict  # noqa: E402


OUT_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "adaptive_fit_window" / "error_comparison"
FIG_DIR = OUT_DIR / "figs"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def smooth(y: np.ndarray) -> np.ndarray:
    if len(y) < 9:
        return y.copy()
    window = max(5, int(round(0.025 * len(y))))
    if window % 2 == 0:
        window += 1
    window = min(window, len(y) if len(y) % 2 == 1 else len(y) - 1)
    if window < 3:
        return y.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(y, kernel, mode="same")


def fmt_pct(value: float) -> str:
    return f"{value:+.1f}%"


def fit_for_cfg(cache, scale: str, cfg: dict[str, str], *, window: bool) -> dict[str, tuple[float, float]]:
    smooth_lambda = float(cfg["smooth_lambda"])
    step_lambda = float(cfg["step_lambda"])
    common = {
        "nuisance_lambda": float(cfg["nuisance_lambda"]),
        "max_mode": int(cfg["max_mode"]),
        "ridge_tau": float(cfg["ridge_tau"]),
        "retention_power": float(cfg["retention_power"]),
        "rho": float(cfg["rho"]),
    }
    source = cache[(scale, TRAIN_CURVE)]
    out: dict[str, tuple[float, float]] = {}
    for channel, response_lambda in {"smooth": smooth_lambda, "step": step_lambda}.items():
        phi = stime_feature(source.curve, response_lambda)
        if window:
            fit = fit_source_kappa_window(
                source,
                phi,
                fit_start_step=int(cfg["fit_start_step"]),
                **common,
            )
        else:
            fit = fit_source_kappa(source, phi, **common)
        out[channel] = (float(fit["kappa"]), response_lambda)
    return out


def build_rows() -> tuple[list[dict[str, object]], dict[tuple[str, str], dict[str, object]]]:
    adaptive_cfg = read_csv(ROOT / "results" / "cosine_to_wsd_response_search" / "adaptive_search" / "safe_configs_top200.csv")[0]
    window_cfg = read_csv(ROOT / "results" / "cosine_to_wsd_response_search" / "adaptive_fit_window" / "safe_window_top200.csv")[0]
    cache = build_cache()
    rows: list[dict[str, object]] = []
    panels: dict[tuple[str, str], dict[str, object]] = {}
    for scale in SCALES:
        adaptive_fit = fit_for_cfg(cache, scale, adaptive_cfg, window=False)
        window_fit = fit_for_cfg(cache, scale, window_cfg, window=True)
        for curve_name, label in TARGETS:
            curve = load_curve(scale, curve_name)
            baseline = mpl_predict(MPL_PRECOMPUTED_INIT[scale], curve)
            channel = channel_for_curve(curve)

            adaptive_kappa, adaptive_lam = adaptive_fit[channel]
            adaptive_phi = stime_feature(curve, adaptive_lam)
            adaptive_ret = target_retention(
                adaptive_phi,
                nuisance_lambda=float(adaptive_cfg["nuisance_lambda"]),
                max_mode=int(adaptive_cfg["max_mode"]),
            )
            adaptive_pred = baseline + adaptive_kappa * (1.0 if adaptive_ret >= TARGET_RETENTION_FLOOR else 0.0) * adaptive_phi

            window_kappa, window_lam = window_fit[channel]
            window_phi = stime_feature(curve, window_lam)
            window_ret = target_retention(
                window_phi,
                nuisance_lambda=float(window_cfg["nuisance_lambda"]),
                max_mode=int(window_cfg["max_mode"]),
            )
            window_pred = baseline + window_kappa * (1.0 if window_ret >= TARGET_RETENTION_FLOOR else 0.0) * window_phi

            base_mae = metrics(curve.loss, baseline)["mae"]
            adaptive_mae = metrics(curve.loss, adaptive_pred)["mae"]
            window_mae = metrics(curve.loss, window_pred)["mae"]
            row = {
                "scale": scale,
                "test_curve": curve_name,
                "test_label": label,
                "channel": channel,
                "mpl_mae": base_mae,
                "adaptive_mae": adaptive_mae,
                "window_mae": window_mae,
                "adaptive_delta_pct": 100.0 * (adaptive_mae / base_mae - 1.0),
                "window_delta_pct": 100.0 * (window_mae / base_mae - 1.0),
                "adaptive_lambda": adaptive_lam,
                "window_lambda": window_lam,
                "window_fit_start": int(window_cfg["fit_start_step"]),
            }
            rows.append(row)
            panels[(scale, curve_name)] = {
                **row,
                "curve": curve,
                "residual_mpl": curve.loss - baseline,
                "residual_adaptive": curve.loss - adaptive_pred,
                "residual_window": curve.loss - window_pred,
            }
    return rows, panels


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for target_curve, target_label in TARGETS + [("ALL", "All WSD targets")]:
        sub = rows if target_curve == "ALL" else [row for row in rows if row["test_curve"] == target_curve]
        item: dict[str, object] = {"test_curve": target_curve, "test_label": target_label, "rows": len(sub)}
        for method in ["adaptive", "window"]:
            vals = np.array([float(row[f"{method}_delta_pct"]) for row in sub], dtype=np.float64)
            item[f"{method}_mean_delta_pct"] = float(np.mean(vals))
            item[f"{method}_worst_delta_pct"] = float(np.max(vals))
            item[f"{method}_wins"] = int(np.sum(vals < 0.0))
        out.append(item)
    return out


def plot_scale(scale: str, panels: dict[tuple[str, str], dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2), constrained_layout=True)
    axes_flat = axes.ravel()
    for i, (curve_name, label) in enumerate(TARGETS):
        panel = panels[(scale, curve_name)]
        curve = panel["curve"]
        steps = curve.step
        ax = axes_flat[i]
        ax.axhline(0.0, color="#111111", lw=0.8, alpha=0.75)
        ax.plot(steps, smooth(panel["residual_mpl"]), color="#111827", lw=1.2, label="MPL")
        ax.plot(steps, smooth(panel["residual_adaptive"]), color="#16a34a", lw=1.15, label="adaptive")
        ax.plot(steps, smooth(panel["residual_window"]), color="#2563eb", lw=1.25, label="fit-window")
        ax2 = ax.twinx()
        ax2.plot(steps, curve.lrs[curve.step] / PEAK_LR, color="#b45309", lw=0.8, alpha=0.25)
        ax2.set_ylim(-0.04, 1.05)
        ax2.tick_params(labelsize=7, colors="#92400e")
        ax.grid(axis="y", alpha=0.18)
        ax.tick_params(labelsize=8)
        ax.set_title(
            f"{label} | window {fmt_pct(float(panel['window_delta_pct']))}, "
            f"start={int(panel['window_fit_start'])}",
            fontsize=9.0,
        )
        if i == 0:
            ax.legend(frameon=False, fontsize=8, loc="upper left")
    axes_flat[-1].axis("off")
    fig.suptitle(f"Suffix-fitted adaptive cosine transfer ({scale}M)", fontsize=12.5)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def write_report(summary: list[dict[str, object]]) -> None:
    all_row = next(row for row in summary if row["test_curve"] == "ALL")
    lines = [
        "# Adaptive Fit-Window Error Comparison\n\n",
        "Residual plots compare the previous adaptive model and the suffix-fitted adaptive model. "
        "Both fit kappa only from cosine residuals.\n\n",
        f"- Adaptive: mean `{fmt_pct(float(all_row['adaptive_mean_delta_pct']))}`, worst `{fmt_pct(float(all_row['adaptive_worst_delta_pct']))}`.\n",
        f"- Fit-window adaptive: mean `{fmt_pct(float(all_row['window_mean_delta_pct']))}`, worst `{fmt_pct(float(all_row['window_worst_delta_pct']))}`.\n\n",
        "| target | adaptive mean | fit-window mean | fit-window worst |\n",
        "|---|---:|---:|---:|\n",
    ]
    for row in summary:
        if row["test_curve"] == "ALL":
            continue
        lines.append(
            f"| {row['test_label']} | {fmt_pct(float(row['adaptive_mean_delta_pct']))} | "
            f"{fmt_pct(float(row['window_mean_delta_pct']))} | "
            f"{fmt_pct(float(row['window_worst_delta_pct']))} |\n"
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows, panels = build_rows()
    summary = aggregate(rows)
    write_csv(OUT_DIR / "error_metrics.csv", rows)
    write_csv(OUT_DIR / "target_summary.csv", summary)
    for scale in SCALES:
        plot_scale(scale, panels, FIG_DIR / f"fit_window_residuals_{scale}M.png")
    write_report(summary)
    print(f"wrote {OUT_DIR / 'error_metrics.csv'}")
    print(f"wrote {OUT_DIR / 'target_summary.csv'}")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    print(f"wrote {FIG_DIR}")


if __name__ == "__main__":
    main()
