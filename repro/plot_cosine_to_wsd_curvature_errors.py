#!/usr/bin/env python3
"""Residual plots for LR-curvature cosine-to-WSD correction."""
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

from cosine_to_wsd_adaptive_fit_window import channel_for_curve, fit_source_kappa_window  # noqa: E402
from cosine_to_wsd_curvature_correction import (  # noqa: E402
    SMOOTH,
    STEP,
    curvature_feature,
    fit_one,
    fit_step_with_curvature,
)
from cosine_to_wsd_response_search import (  # noqa: E402
    TARGETS,
    TARGET_RETENTION_FLOOR,
    TRAIN_CURVE,
    build_cache,
    stime_feature,
    target_retention,
)
from reproduce_cosine_to_wsd import MPL_PRECOMPUTED_INIT, PEAK_LR, SCALES, load_curve, metrics, mpl_predict  # noqa: E402


OUT_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "curvature_correction" / "error_comparison"
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


def smooth_line(y: np.ndarray) -> np.ndarray:
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


def fit_decoupled_channel(cache, scale: str, prefix: str) -> tuple[float, float, float, int]:
    params = SMOOTH if prefix == "smooth" else STEP
    source = cache[(scale, TRAIN_CURVE)]
    response_lambda = float(params["response_lambda"])
    phi = stime_feature(source.curve, response_lambda)
    fit = fit_source_kappa_window(
        source,
        phi,
        fit_start_step=int(params["fit_start_step"]),
        nuisance_lambda=float(params["nuisance_lambda"]),
        max_mode=int(params["max_mode"]),
        ridge_tau=float(params["ridge_tau"]),
        retention_power=float(params["retention_power"]),
        rho=float(params["rho"]),
    )
    return float(fit["kappa"]), response_lambda, float(params["nuisance_lambda"]), int(params["max_mode"])


def predict_decoupled(curve, baseline: np.ndarray, fit: tuple[float, float, float, int]) -> tuple[np.ndarray, float]:
    kappa, response_lambda, nuisance_lambda, max_mode = fit
    phi = stime_feature(curve, response_lambda)
    retention = target_retention(phi, nuisance_lambda=nuisance_lambda, max_mode=max_mode)
    factor = 1.0 if retention >= TARGET_RETENTION_FLOOR else 0.0
    return baseline + factor * kappa * phi, retention


def fit_curvature(cache, scale: str, cfg: dict[str, str]) -> dict[str, object]:
    source = cache[(scale, TRAIN_CURVE)]
    smooth_phi = stime_feature(source.curve, float(SMOOTH["response_lambda"]))
    smooth_kappa, _ = fit_one(source, smooth_phi, SMOOTH)

    step_phi = stime_feature(source.curve, float(STEP["response_lambda"]))
    step_curv = curvature_feature(source.curve, float(cfg["curvature_lambda"]), cfg["curvature_mode"])
    step_coef, _ = fit_step_with_curvature(
        source,
        step_phi,
        step_curv,
        curvature_tau=float(cfg["curvature_tau"]),
        shrink_curvature=bool(int(cfg["shrink_curvature"])),
        signed_curvature_coef=bool(int(cfg["signed_curvature_coef"])),
    )
    return {"smooth_kappa": smooth_kappa, "step_coef": step_coef}


def predict_curvature(curve, baseline: np.ndarray, fit: dict[str, object], cfg: dict[str, str]) -> tuple[np.ndarray, float]:
    channel = channel_for_curve(curve)
    if channel == "smooth":
        phi = stime_feature(curve, float(SMOOTH["response_lambda"]))
        retention = target_retention(
            phi,
            nuisance_lambda=float(SMOOTH["nuisance_lambda"]),
            max_mode=int(SMOOTH["max_mode"]),
        )
        shape = float(fit["smooth_kappa"]) * phi
    else:
        phi = stime_feature(curve, float(STEP["response_lambda"]))
        curv = curvature_feature(curve, float(cfg["curvature_lambda"]), cfg["curvature_mode"])
        coef = np.asarray(fit["step_coef"], dtype=np.float64)
        shape = coef[0] * phi + coef[1] * curv
        retention = target_retention(
            shape,
            nuisance_lambda=float(STEP["nuisance_lambda"]),
            max_mode=int(STEP["max_mode"]),
        )
    factor = 1.0 if retention >= TARGET_RETENTION_FLOOR else 0.0
    return baseline + factor * shape, retention


def build_rows() -> tuple[list[dict[str, object]], dict[tuple[str, str], dict[str, object]]]:
    cfg = read_csv(
        ROOT
        / "results"
        / "cosine_to_wsd_response_search"
        / "curvature_correction"
        / "safe_curvature_top200.csv"
    )[0]
    cache = build_cache()
    rows: list[dict[str, object]] = []
    panels: dict[tuple[str, str], dict[str, object]] = {}
    for scale in SCALES:
        decoupled_fit = {
            "smooth": fit_decoupled_channel(cache, scale, "smooth"),
            "step": fit_decoupled_channel(cache, scale, "step"),
        }
        curvature_fit = fit_curvature(cache, scale, cfg)
        for curve_name, label in TARGETS:
            curve = load_curve(scale, curve_name)
            baseline = mpl_predict(MPL_PRECOMPUTED_INIT[scale], curve)
            channel = channel_for_curve(curve)
            decoupled_pred, decoupled_ret = predict_decoupled(curve, baseline, decoupled_fit[channel])
            curvature_pred, curvature_ret = predict_curvature(curve, baseline, curvature_fit, cfg)

            base_mae = metrics(curve.loss, baseline)["mae"]
            decoupled_mae = metrics(curve.loss, decoupled_pred)["mae"]
            curvature_mae = metrics(curve.loss, curvature_pred)["mae"]
            row = {
                "scale": scale,
                "test_curve": curve_name,
                "test_label": label,
                "channel": channel,
                "mpl_mae": base_mae,
                "decoupled_mae": decoupled_mae,
                "curvature_mae": curvature_mae,
                "decoupled_delta_pct": 100.0 * (decoupled_mae / base_mae - 1.0),
                "curvature_delta_pct": 100.0 * (curvature_mae / base_mae - 1.0),
                "decoupled_target_retention": decoupled_ret,
                "curvature_target_retention": curvature_ret,
            }
            rows.append(row)
            panels[(scale, curve_name)] = {
                **row,
                "curve": curve,
                "residual_mpl": curve.loss - baseline,
                "residual_decoupled": curve.loss - decoupled_pred,
                "residual_curvature": curve.loss - curvature_pred,
            }
    return rows, panels


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for target_curve, target_label in TARGETS + [("ALL", "All WSD targets")]:
        sub = rows if target_curve == "ALL" else [row for row in rows if row["test_curve"] == target_curve]
        item: dict[str, object] = {"test_curve": target_curve, "test_label": target_label, "rows": len(sub)}
        for method in ["decoupled", "curvature"]:
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
        ax.plot(steps, smooth_line(panel["residual_mpl"]), color="#111827", lw=1.2, label="MPL")
        ax.plot(steps, smooth_line(panel["residual_decoupled"]), color="#dc2626", lw=1.1, label="decoupled")
        ax.plot(steps, smooth_line(panel["residual_curvature"]), color="#2563eb", lw=1.25, label="curvature")
        ax2 = ax.twinx()
        ax2.plot(steps, curve.lrs[curve.step] / PEAK_LR, color="#b45309", lw=0.8, alpha=0.25)
        ax2.set_ylim(-0.04, 1.05)
        ax2.tick_params(labelsize=7, colors="#92400e")
        ax.grid(axis="y", alpha=0.18)
        ax.tick_params(labelsize=8)
        ax.set_title(
            f"{label} | curvature {fmt_pct(float(panel['curvature_delta_pct']))}",
            fontsize=9.0,
        )
        if i == 0:
            ax.legend(frameon=False, fontsize=8, loc="upper left")
    axes_flat[-1].axis("off")
    fig.suptitle(f"LR-curvature cosine transfer residuals ({scale}M)", fontsize=12.5)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def write_report(summary: list[dict[str, object]]) -> None:
    all_row = next(row for row in summary if row["test_curve"] == "ALL")
    lines = [
        "# LR-Curvature Error Comparison\n\n",
        "Residual plots compare MPL, the decoupled-channel model, and the LR-curvature model. "
        "Both corrected models fit coefficients only from cosine residuals.\n\n",
        f"- Decoupled-channel: mean `{fmt_pct(float(all_row['decoupled_mean_delta_pct']))}`, "
        f"worst `{fmt_pct(float(all_row['decoupled_worst_delta_pct']))}`.\n",
        f"- LR-curvature: mean `{fmt_pct(float(all_row['curvature_mean_delta_pct']))}`, "
        f"worst `{fmt_pct(float(all_row['curvature_worst_delta_pct']))}`.\n\n",
        "| target | decoupled mean | curvature mean | curvature worst |\n",
        "|---|---:|---:|---:|\n",
    ]
    for row in summary:
        if row["test_curve"] == "ALL":
            continue
        lines.append(
            f"| {row['test_label']} | {fmt_pct(float(row['decoupled_mean_delta_pct']))} | "
            f"{fmt_pct(float(row['curvature_mean_delta_pct']))} | "
            f"{fmt_pct(float(row['curvature_worst_delta_pct']))} |\n"
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
        plot_scale(scale, panels, FIG_DIR / f"curvature_residuals_{scale}M.png")
    write_report(summary)
    print(f"wrote {OUT_DIR / 'error_metrics.csv'}")
    print(f"wrote {OUT_DIR / 'target_summary.csv'}")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    print(f"wrote {FIG_DIR}")


if __name__ == "__main__":
    main()
