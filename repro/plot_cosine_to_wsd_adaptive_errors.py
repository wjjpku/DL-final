#!/usr/bin/env python3
"""Residual plots for the best schedule-adaptive cosine-to-WSD correction."""
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

from deep_stime import stime_feature as old_stime_feature  # noqa: E402
from cosine_to_wsd_response_search import (  # noqa: E402
    TARGETS,
    TARGET_RETENTION_FLOOR,
    TRAIN_CURVE,
    build_cache,
    fit_source_kappa,
    score_target,
    stime_feature,
    target_retention,
)
from cosine_to_wsd_schedule_adaptive_audit import drop_concentration  # noqa: E402
from reproduce_cosine_to_wsd import MPL_PRECOMPUTED_INIT, PEAK_LR, SCALES, load_curve, metrics, mpl_predict  # noqa: E402


OUT_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "adaptive_search" / "error_comparison"
FIG_DIR = OUT_DIR / "figs"
OLD_RESPONSE_LAMBDA = 10.0
DROP_CONCENTRATION_THRESHOLD = 0.2


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


def best_rows() -> tuple[dict[str, str], dict[str, str]]:
    global_best = read_csv(ROOT / "results" / "cosine_to_wsd_response_search" / "safe_configs_top100.csv")[0]
    adaptive_best = read_csv(ROOT / "results" / "cosine_to_wsd_response_search" / "adaptive_search" / "safe_configs_top200.csv")[0]
    return global_best, adaptive_best


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


def channel_for_curve(curve) -> str:
    return "step" if drop_concentration(curve) >= DROP_CONCENTRATION_THRESHOLD else "smooth"


def fit_global(cache, scale: str, cfg: dict[str, str]) -> tuple[float, float, int]:
    lam = float(cfg["response_lambda"])
    source = cache[(scale, TRAIN_CURVE)]
    phi = stime_feature(source.curve, lam)
    fit = fit_source_kappa(
        source,
        phi,
        nuisance_lambda=float(cfg["nuisance_lambda"]),
        max_mode=int(cfg["max_mode"]),
        ridge_tau=float(cfg["ridge_tau"]),
        retention_power=float(cfg["retention_power"]),
        rho=float(cfg["rho"]),
    )
    return float(fit["kappa"]), lam, int(cfg["max_mode"])


def fit_adaptive(cache, scale: str, cfg: dict[str, str], channel: str) -> tuple[float, float, int]:
    lam = float(cfg["step_lambda"] if channel == "step" else cfg["smooth_lambda"])
    source = cache[(scale, TRAIN_CURVE)]
    phi = stime_feature(source.curve, lam)
    fit = fit_source_kappa(
        source,
        phi,
        nuisance_lambda=float(cfg["nuisance_lambda"]),
        max_mode=int(cfg["max_mode"]),
        ridge_tau=float(cfg["ridge_tau"]),
        retention_power=float(cfg["retention_power"]),
        rho=float(cfg["rho"]),
    )
    return float(fit["kappa"]), lam, int(cfg["max_mode"])


def build_rows() -> tuple[list[dict[str, object]], dict[tuple[str, str], dict[str, object]]]:
    cache = build_cache()
    global_cfg, adaptive_cfg = best_rows()
    old_lookup = {
        (row["scale"], row["test_curve"]): row
        for row in read_csv(ROOT / "results" / "cosine_to_wsd_focus" / "nextgen_safe_details.csv")
    }
    rows: list[dict[str, object]] = []
    panels: dict[tuple[str, str], dict[str, object]] = {}
    for scale in SCALES:
        global_fit = fit_global(cache, scale, global_cfg)
        adaptive_fit_cache: dict[str, tuple[float, float, int]] = {}
        for curve_name, label in TARGETS:
            curve = load_curve(scale, curve_name)
            baseline = mpl_predict(MPL_PRECOMPUTED_INIT[scale], curve)

            old_info = old_lookup[(scale, curve_name)]
            old_pred = baseline + float(old_info["kappa"]) * float(old_info["target_factor"]) * old_stime_feature(
                curve, OLD_RESPONSE_LAMBDA
            )

            global_kappa, global_lam, global_mode = global_fit
            global_phi = stime_feature(curve, global_lam)
            global_ret = target_retention(
                global_phi,
                nuisance_lambda=float(global_cfg["nuisance_lambda"]),
                max_mode=global_mode,
            )
            global_factor = 1.0 if global_ret >= TARGET_RETENTION_FLOOR else 0.0
            global_pred = baseline + global_kappa * global_factor * global_phi

            channel = channel_for_curve(curve)
            if channel not in adaptive_fit_cache:
                adaptive_fit_cache[channel] = fit_adaptive(cache, scale, adaptive_cfg, channel)
            adaptive_kappa, adaptive_lam, adaptive_mode = adaptive_fit_cache[channel]
            adaptive_phi = stime_feature(curve, adaptive_lam)
            adaptive_ret = target_retention(
                adaptive_phi,
                nuisance_lambda=float(adaptive_cfg["nuisance_lambda"]),
                max_mode=adaptive_mode,
            )
            adaptive_factor = 1.0 if adaptive_ret >= TARGET_RETENTION_FLOOR else 0.0
            adaptive_pred = baseline + adaptive_kappa * adaptive_factor * adaptive_phi

            base_mae = metrics(curve.loss, baseline)["mae"]
            old_mae = metrics(curve.loss, old_pred)["mae"]
            global_mae = metrics(curve.loss, global_pred)["mae"]
            adaptive_mae = metrics(curve.loss, adaptive_pred)["mae"]
            row = {
                "scale": scale,
                "test_curve": curve_name,
                "test_label": label,
                "channel": channel,
                "drop_concentration": drop_concentration(curve),
                "mpl_mae": base_mae,
                "old_mae": old_mae,
                "global_mae": global_mae,
                "adaptive_mae": adaptive_mae,
                "old_delta_pct": 100.0 * (old_mae / base_mae - 1.0),
                "global_delta_pct": 100.0 * (global_mae / base_mae - 1.0),
                "adaptive_delta_pct": 100.0 * (adaptive_mae / base_mae - 1.0),
                "global_lambda": global_lam,
                "adaptive_lambda": adaptive_lam,
                "global_kappa": global_kappa,
                "adaptive_kappa": adaptive_kappa,
            }
            rows.append(row)
            panels[(scale, curve_name)] = {
                **row,
                "curve": curve,
                "residual_mpl": curve.loss - baseline,
                "residual_old": curve.loss - old_pred,
                "residual_global": curve.loss - global_pred,
                "residual_adaptive": curve.loss - adaptive_pred,
            }
    return rows, panels


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for target_curve, target_label in TARGETS + [("ALL", "All WSD targets")]:
        sub = rows if target_curve == "ALL" else [row for row in rows if row["test_curve"] == target_curve]
        item: dict[str, object] = {"test_curve": target_curve, "test_label": target_label, "rows": len(sub)}
        for method in ["old", "global", "adaptive"]:
            vals = np.array([float(row[f"{method}_delta_pct"]) for row in sub], dtype=np.float64)
            item[f"{method}_mean_delta_pct"] = float(np.mean(vals))
            item[f"{method}_worst_delta_pct"] = float(np.max(vals))
            item[f"{method}_wins"] = int(np.sum(vals < 0.0))
        out.append(item)
    return out


def fmt_pct(value: float) -> str:
    return f"{value:+.1f}%"


def plot_scale(scale: str, panels: dict[tuple[str, str], dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2), constrained_layout=True)
    axes_flat = axes.ravel()
    for i, (curve_name, label) in enumerate(TARGETS):
        panel = panels[(scale, curve_name)]
        curve = panel["curve"]
        steps = curve.step
        ax = axes_flat[i]
        ax.axhline(0.0, color="#111111", lw=0.8, alpha=0.75)
        ax.plot(steps, smooth(panel["residual_mpl"]), color="#111827", lw=1.25, label="MPL")
        ax.plot(steps, smooth(panel["residual_old"]), color="#dc2626", lw=1.1, label="old")
        ax.plot(steps, smooth(panel["residual_global"]), color="#16a34a", lw=1.1, label="global")
        ax.plot(steps, smooth(panel["residual_adaptive"]), color="#2563eb", lw=1.25, label="adaptive")
        ax2 = ax.twinx()
        ax2.plot(steps, curve.lrs[curve.step] / PEAK_LR, color="#b45309", lw=0.8, alpha=0.25)
        ax2.set_ylim(-0.04, 1.05)
        ax2.tick_params(labelsize=7, colors="#92400e")
        ax.grid(axis="y", alpha=0.18)
        ax.tick_params(labelsize=8)
        ax.set_title(
            f"{label} | adaptive {fmt_pct(float(panel['adaptive_delta_pct']))}, "
            f"lambda={float(panel['adaptive_lambda']):g}",
            fontsize=9.0,
        )
        if i == 0:
            ax.legend(frameon=False, fontsize=8, loc="upper left")
    axes_flat[-1].axis("off")
    fig.suptitle(f"Adaptive cosine-calibrated residual transfer ({scale}M)", fontsize=12.5)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def write_report(summary: list[dict[str, object]]) -> None:
    all_row = next(row for row in summary if row["test_curve"] == "ALL")
    lines = [
        "# Adaptive Error Comparison\n\n",
        "Residual plots compare MPL, old nextgen, the global response candidate, and the best schedule-adaptive candidate. "
        "All correction amplitudes are fitted from cosine residuals only.\n\n",
        "## Aggregate\n\n",
        f"- Old: mean `{fmt_pct(float(all_row['old_mean_delta_pct']))}`, worst `{fmt_pct(float(all_row['old_worst_delta_pct']))}`.\n",
        f"- Global response: mean `{fmt_pct(float(all_row['global_mean_delta_pct']))}`, worst `{fmt_pct(float(all_row['global_worst_delta_pct']))}`.\n",
        f"- Adaptive response: mean `{fmt_pct(float(all_row['adaptive_mean_delta_pct']))}`, worst `{fmt_pct(float(all_row['adaptive_worst_delta_pct']))}`.\n\n",
        "| target | old mean | global mean | adaptive mean | adaptive worst |\n",
        "|---|---:|---:|---:|---:|\n",
    ]
    for row in summary:
        if row["test_curve"] == "ALL":
            continue
        lines.append(
            f"| {row['test_label']} | {fmt_pct(float(row['old_mean_delta_pct']))} | "
            f"{fmt_pct(float(row['global_mean_delta_pct']))} | "
            f"{fmt_pct(float(row['adaptive_mean_delta_pct']))} | "
            f"{fmt_pct(float(row['adaptive_worst_delta_pct']))} |\n"
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
        plot_scale(scale, panels, FIG_DIR / f"adaptive_residuals_{scale}M.png")
    write_report(summary)
    print(f"wrote {OUT_DIR / 'error_metrics.csv'}")
    print(f"wrote {OUT_DIR / 'target_summary.csv'}")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    print(f"wrote {FIG_DIR}")


if __name__ == "__main__":
    main()
