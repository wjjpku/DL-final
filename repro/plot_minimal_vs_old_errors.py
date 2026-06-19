#!/usr/bin/env python3
"""Plot residual errors for MPL, old S-time correction, and minimal step-time.

The old curve is the pre-step-time cumulative-LR response (`S10_current`) used
in the response-shape diagnostics.  It is fitted on the target residual in each
panel to show the best residual shape that old feature can explain.  The
minimal curve is the target-holdout one-kappa estimator from
`step_time_minimal_estimator.py`; it does not use target residuals.
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
if str(REPO) not in sys.path:
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
from step_time_minimal_estimator import score_routes  # noqa: E402
from step_time_shape_routed_estimator import (  # noqa: E402
    CORE_CURVES,
    EXTRA_SAFETY_CURVES,
)


OUT_DIR = ROOT / "results" / "step_time_minimal_estimator" / "error_comparison"
CURVES = list(CORE_CURVES) + list(EXTRA_SAFETY_CURVES)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def old_s_time_feature(curve) -> np.ndarray:
    eta = curve.lrs.astype(np.float64)
    drop = np.zeros_like(eta)
    drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0) / PEAK_LR
    out = np.zeros_like(eta)
    acc = 0.0
    for t in range(len(eta)):
        acc = acc * math.exp(-LAMBDA * eta[t]) + drop[t]
        out[t] = acc
    return out[curve.step]


def fit_origin_nonnegative(x: np.ndarray, y: np.ndarray) -> float:
    denom = float(np.dot(x, x))
    if denom <= 1e-18:
        return 0.0
    return max(0.0, float(np.dot(x, y) / denom))


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


def minimal_prediction_lookup() -> dict[tuple[str, str], dict[str, object]]:
    routes, details, _ = score_routes(CURVES, mode="minimal_error_comparison")
    route_by_target = {str(row["target_curve"]): row for row in routes}
    detail_by_key = {(str(row["scale"]), str(row["target_curve"])): row for row in details}
    out: dict[tuple[str, str], dict[str, object]] = {}
    for scale in SCALES:
        for curve_name, _ in CURVES:
            row = detail_by_key[(scale, curve_name)]
            out[(scale, curve_name)] = {
                "kappa": float(row["kappa"]),
                "tau": float(row["tau"]),
                "route": row["route"],
                "train_curves": row["train_curves"],
                "route_row": route_by_target[curve_name],
            }
    return out


def minimal_step_feature(curve, tau: float) -> np.ndarray:
    if tau <= 0.0:
        return np.zeros_like(curve.step, dtype=np.float64)
    eta = curve.lrs.astype(np.float64)
    drop = np.zeros_like(eta)
    drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0) / PEAK_LR
    out = np.zeros_like(eta)
    acc = 0.0
    decay = math.exp(-1.0 / tau)
    for t in range(len(eta)):
        acc = acc * decay + drop[t]
        out[t] = acc
    return out[curve.step]


def analyze() -> tuple[list[dict[str, object]], dict[tuple[str, str], dict[str, object]]]:
    minimal_lookup = minimal_prediction_lookup()
    rows: list[dict[str, object]] = []
    panels: dict[tuple[str, str], dict[str, object]] = {}
    for scale in SCALES:
        for curve_name, label in CURVES:
            curve = load_curve(scale, curve_name)
            base = mpl_predict(MPL_PRECOMPUTED_INIT[scale], curve)
            residual_mpl = curve.loss - base

            old_phi = old_s_time_feature(curve)
            old_kappa = fit_origin_nonnegative(old_phi, residual_mpl)
            old_pred = base + old_kappa * old_phi
            residual_old = curve.loss - old_pred

            minimal_info = minimal_lookup[(scale, curve_name)]
            minimal_phi = minimal_step_feature(curve, float(minimal_info["tau"]))
            minimal_pred = base + float(minimal_info["kappa"]) * minimal_phi
            residual_minimal = curve.loss - minimal_pred

            base_mae = metrics(curve.loss, base)["mae"]
            old_mae = metrics(curve.loss, old_pred)["mae"]
            minimal_mae = metrics(curve.loss, minimal_pred)["mae"]
            rows.append(
                {
                    "scale": scale,
                    "curve": curve_name,
                    "label": label,
                    "mpl_mae": base_mae,
                    "old_samefit_mae": old_mae,
                    "minimal_holdout_mae": minimal_mae,
                    "old_delta_pct": 100.0 * (old_mae / base_mae - 1.0),
                    "minimal_delta_pct": 100.0 * (minimal_mae / base_mae - 1.0),
                    "old_samefit_kappa": old_kappa,
                    "minimal_kappa": minimal_info["kappa"],
                    "minimal_tau": minimal_info["tau"],
                    "minimal_route": minimal_info["route"],
                    "minimal_train_curves": minimal_info["train_curves"],
                    "old_uses_target_residual": 1,
                    "minimal_uses_target_residual": 0,
                }
            )
            panels[(scale, curve_name)] = {
                "scale": scale,
                "curve_name": curve_name,
                "label": label,
                "curve": curve,
                "residual_mpl": residual_mpl,
                "residual_old": residual_old,
                "residual_minimal": residual_minimal,
                "old_kappa": old_kappa,
                "minimal_info": minimal_info,
                "base_mae": base_mae,
                "old_mae": old_mae,
                "minimal_mae": minimal_mae,
            }
    return rows, panels


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for group_name, group_rows in [
        ("core", [r for r in rows if r["curve"] in {c for c, _ in CORE_CURVES}]),
        ("extended", rows),
        ("safety_controls", [r for r in rows if r["curve"] in {c for c, _ in EXTRA_SAFETY_CURVES}]),
    ]:
        if not group_rows:
            continue
        old = [float(r["old_delta_pct"]) for r in group_rows]
        new = [float(r["minimal_delta_pct"]) for r in group_rows]
        out.append(
            {
                "group": group_name,
                "rows": len(group_rows),
                "old_mean_delta": float(np.mean(old)),
                "old_worst_delta": float(np.max(old)),
                "old_nonharm": int(sum(v <= 1e-10 for v in old)),
                "minimal_mean_delta": float(np.mean(new)),
                "minimal_worst_delta": float(np.max(new)),
                "minimal_nonharm": int(sum(v <= 1e-10 for v in new)),
            }
        )
    return out


def plot_scale(scale: str, panels: dict[tuple[str, str], dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(16.0, 12.0), constrained_layout=True)
    axes_flat = axes.ravel()
    for i, (curve_name, label) in enumerate(CURVES):
        ax = axes_flat[i]
        panel = panels[(scale, curve_name)]
        curve = panel["curve"]
        steps = curve.step
        ax.axhline(0.0, color="#111111", lw=0.8, alpha=0.75)
        ax.plot(steps, smooth(panel["residual_mpl"]), color="#111827", lw=1.25, label="MPL error")
        ax.plot(steps, smooth(panel["residual_old"]), color="#dc2626", lw=1.15, label="MPL+old error")
        ax.plot(steps, smooth(panel["residual_minimal"]), color="#2563eb", lw=1.15, label="MPL+minimal error")
        ax2 = ax.twinx()
        ax2.plot(steps, curve.lrs[curve.step] / PEAK_LR, color="#b45309", lw=0.85, alpha=0.28)
        ax2.set_ylim(-0.04, 1.05)
        ax2.tick_params(labelsize=7, colors="#92400e")
        old_delta = 100.0 * (float(panel["old_mae"]) / float(panel["base_mae"]) - 1.0)
        min_delta = 100.0 * (float(panel["minimal_mae"]) / float(panel["base_mae"]) - 1.0)
        ax.set_title(
            f"{label} | old {old_delta:+.1f}% vs minimal {min_delta:+.1f}%",
            fontsize=9.0,
        )
        ax.grid(axis="y", alpha=0.18)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"{scale}M residual errors", fontsize=12.5)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def method_deltas(panel: dict[str, object]) -> tuple[float, float]:
    old_delta = 100.0 * (float(panel["old_mae"]) / float(panel["base_mae"]) - 1.0)
    min_delta = 100.0 * (float(panel["minimal_mae"]) / float(panel["base_mae"]) - 1.0)
    return old_delta, min_delta


def plot_residual_panel(ax, panel: dict[str, object], show_ylabel: bool = False) -> None:
    curve = panel["curve"]
    steps = curve.step
    y_mpl = smooth(panel["residual_mpl"])
    y_old = smooth(panel["residual_old"])
    y_min = smooth(panel["residual_minimal"])
    scale = max(float(np.max(np.abs(y_mpl))), float(np.max(np.abs(y_old))), float(np.max(np.abs(y_min))), 1e-6)
    ax.axhline(0.0, color="#111111", lw=0.8, alpha=0.8)
    ax.plot(steps, y_mpl, color="#111827", lw=1.45, label="MPL")
    ax.plot(steps, y_old, color="#dc2626", lw=1.35, label="MPL+old same-fit")
    ax.plot(steps, y_min, color="#2563eb", lw=1.35, ls="--", label="MPL+minimal holdout")
    ax.set_ylim(-1.12 * scale, 1.12 * scale)
    old_delta, min_delta = method_deltas(panel)
    ax.set_title(f"{panel['label']}\nold {old_delta:+.1f}% | minimal {min_delta:+.1f}%", fontsize=10.2)
    ax.grid(axis="y", alpha=0.2)
    ax.tick_params(labelsize=8)
    ax.set_xlabel("step", fontsize=8.5)
    if show_ylabel:
        ax.set_ylabel("loss residual", fontsize=8.5)


def plot_core_scale(scale: str, panels: dict[tuple[str, str], dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 7.6), constrained_layout=True)
    for i, (curve_name, _) in enumerate(CORE_CURVES):
        ax = axes.ravel()[i]
        plot_residual_panel(ax, panels[(scale, curve_name)], show_ylabel=(i % 3 == 0))
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9.0, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.045))
    fig.suptitle(f"{scale}M core schedules: residual errors after correction", fontsize=13)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def plot_safety_scale(scale: str, panels: dict[tuple[str, str], dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 3.2), constrained_layout=True)
    for i, (curve_name, _) in enumerate(EXTRA_SAFETY_CURVES):
        ax = axes.ravel()[i]
        plot_residual_panel(ax, panels[(scale, curve_name)], show_ylabel=(i == 0))
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9.0, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(f"{scale}M safety controls: correction should abstain", fontsize=13)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def target_summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for curve_name, label in CURVES:
        sub = [row for row in rows if row["curve"] == curve_name]
        old = [float(row["old_delta_pct"]) for row in sub]
        minimal = [float(row["minimal_delta_pct"]) for row in sub]
        out.append(
            {
                "curve": curve_name,
                "label": label,
                "old_mean_delta": float(np.mean(old)),
                "old_worst_delta": float(np.max(old)),
                "minimal_mean_delta": float(np.mean(minimal)),
                "minimal_worst_delta": float(np.max(minimal)),
                "rows": len(sub),
            }
        )
    return out


def plot_bar_summary(path: Path, summary: list[dict[str, object]]) -> None:
    core_names = {name for name, _ in CORE_CURVES}
    summary = [row for row in summary if row["curve"] in core_names]
    labels = [
        str(row["label"])
        .replace("WSD sharp", "WSD\nsharp")
        .replace("WSD linear", "WSD\nlinear")
        .replace("WSD-con ", "con\n")
        .replace("Constant ", "Const\n")
        .replace("Cosine ", "Cos\n")
        for row in summary
    ]
    old = np.array([float(row["old_mean_delta"]) for row in summary])
    minimal = np.array([float(row["minimal_mean_delta"]) for row in summary])
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12.6, 4.6), constrained_layout=True)
    ax.axhline(0.0, color="#111111", lw=0.9)
    b1 = ax.bar(x - width / 2, old, width, color="#dc2626", label="MPL+old same-fit")
    b2 = ax.bar(x + width / 2, minimal, width, color="#2563eb", label="MPL+minimal holdout")
    ax.set_xticks(x, labels)
    ax.set_ylabel("mean MAE change vs MPL (%)")
    ax.set_title("Core schedules: old same-fit vs minimal target-holdout")
    ax.set_ylim(min(float(old.min()), float(minimal.min())) - 7.0, 2.0)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    for bars in [b1, b2]:
        for bar in bars:
            val = float(bar.get_height())
            if abs(val) < 0.05:
                continue
            text = f"{val:+.0f}"
            if val < 0:
                y = val + 1.8
                va = "bottom"
                color = "white"
            else:
                y = val + 0.8
                va = "bottom"
                color = "#111111"
            ax.text(bar.get_x() + bar.get_width() / 2, y, text, ha="center", va=va, fontsize=8.5, color=color)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def write_report(rows: list[dict[str, object]], aggregate: list[dict[str, object]]) -> None:
    core = next(row for row in aggregate if row["group"] == "core")
    extended = next(row for row in aggregate if row["group"] == "extended")
    safety = next(row for row in aggregate if row["group"] == "safety_controls")
    lines = [
        "# MPL vs Old vs Minimal Error Comparison\n\n",
        "This reruns the residual-style plots with three error curves: `MPL`, `MPL+old`, and `MPL+minimal`.\n\n",
        "- `MPL+old` uses the previous cumulative-LR / S-time response feature and fits its amplitude on the target residual in each panel.  It is a same-curve shape diagnostic, not a deployment protocol.\n",
        "- `MPL+minimal` is the current one-kappa target-holdout rule: source and tau come from the LR schedule, and target residuals are not used.\n\n",
        "## Aggregate MAE Change vs MPL\n\n",
        "| group | old same-fit mean | old worst | old non-harm | minimal mean | minimal worst | minimal non-harm |\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['group']} | {float(row['old_mean_delta']):+.1f}% | "
            f"{float(row['old_worst_delta']):+.1f}% | {int(row['old_nonharm'])}/{int(row['rows'])} | "
            f"{float(row['minimal_mean_delta']):+.1f}% | {float(row['minimal_worst_delta']):+.1f}% | "
            f"{int(row['minimal_nonharm'])}/{int(row['rows'])} |\n"
        )
    lines += [
        "\n## Figures\n\n",
    ]
    for scale in SCALES:
        lines.append(f"- `{scale}M` core: `core_residuals_{scale}M.png`\n")
        lines.append(f"- `{scale}M` safety controls: `safety_residuals_{scale}M.png`\n")
        lines.append(f"- `{scale}M` compact overview: `error_comparison_{scale}M.png`\n")
    lines.append("- Core MAE bar summary: `mae_bar_summary.png`\n")
    lines += [
        "\n## Reading\n\n",
        f"- On core targets, the old same-curve feature gives mean `{float(core['old_mean_delta']):+.1f}%` and worst `{float(core['old_worst_delta']):+.1f}%`; the minimal holdout rule gives mean `{float(core['minimal_mean_delta']):+.1f}%` and worst `{float(core['minimal_worst_delta']):+.1f}%`.\n",
        f"- On extended controls, minimal remains non-harming (`{int(extended['minimal_nonharm'])}/{int(extended['rows'])}`), while the old same-fit curve is only a target-residual diagnostic.\n",
        f"- Safety controls are intentionally unchanged by minimal (`{int(safety['minimal_nonharm'])}/{int(safety['rows'])}` non-harm), which is the desired behavior for short-smooth and zero-drop schedules.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, panels = analyze()
    aggregate = aggregate_rows(rows)
    write_csv(OUT_DIR / "error_metrics.csv", rows)
    write_csv(OUT_DIR / "aggregate_metrics.csv", aggregate)
    target_summary = target_summary_rows(rows)
    write_csv(OUT_DIR / "target_summary.csv", target_summary)
    for scale in SCALES:
        plot_scale(scale, panels, OUT_DIR / f"error_comparison_{scale}M.png")
        plot_core_scale(scale, panels, OUT_DIR / f"core_residuals_{scale}M.png")
        plot_safety_scale(scale, panels, OUT_DIR / f"safety_residuals_{scale}M.png")
    plot_bar_summary(OUT_DIR / "mae_bar_summary.png", target_summary)
    write_report(rows, aggregate)
    core = next(row for row in aggregate if row["group"] == "core")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    print(
        f"core old={float(core['old_mean_delta']):+.1f}%/"
        f"{float(core['old_worst_delta']):+.1f}% "
        f"minimal={float(core['minimal_mean_delta']):+.1f}%/"
        f"{float(core['minimal_worst_delta']):+.1f}%"
    )


if __name__ == "__main__":
    main()
