#!/usr/bin/env python3
"""Plot residual errors for table-tau vs geometry-tau one-kappa rules."""
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

from audit_step_time_geometry_tau import eval_shape_geometry  # noqa: E402
from reproduce_cosine_to_wsd import (  # noqa: E402
    MPL_PRECOMPUTED_INIT,
    PEAK_LR,
    SCALES,
    load_curve,
    metrics,
    mpl_predict,
)
from step_time_minimal_estimator import score_routes  # noqa: E402
from step_time_shape_routed_estimator import CORE_CURVES, EXTENDED_CURVES, EXTRA_SAFETY_CURVES  # noqa: E402


OUT_DIR = ROOT / "results" / "step_time_geometry_tau" / "error_comparison"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({k for row in rows for k in row})
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


def step_feature(curve, tau: float) -> np.ndarray:
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


def detail_lookup(rows: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, object]]:
    return {(str(row["scale"]), str(row["target_curve"])): row for row in rows}


def analyze() -> tuple[list[dict[str, object]], dict[tuple[str, str], dict[str, object]]]:
    _, table_details, _ = score_routes(EXTENDED_CURVES, mode="table_tau_no_nuisance")
    _, geometry_details, _ = eval_shape_geometry(
        EXTENDED_CURVES,
        mode="geometry_tau_no_nuisance",
        force_nuisance="none",
        self_fit=False,
    )
    table_by_key = detail_lookup(table_details)
    geometry_by_key = detail_lookup(geometry_details)
    rows: list[dict[str, object]] = []
    panels: dict[tuple[str, str], dict[str, object]] = {}

    for scale in SCALES:
        for curve_name, label in EXTENDED_CURVES:
            curve = load_curve(scale, curve_name)
            base = mpl_predict(MPL_PRECOMPUTED_INIT[scale], curve)
            residual_mpl = curve.loss - base

            table = table_by_key[(scale, curve_name)]
            table_tau = float(table["tau"])
            table_kappa = float(table["kappa"])
            table_pred = base + table_kappa * step_feature(curve, table_tau)
            residual_table = curve.loss - table_pred

            geometry = geometry_by_key[(scale, curve_name)]
            geometry_tau = float(geometry["geometry_tau"])
            geometry_kappa = float(geometry["kappa"])
            geometry_pred = base + geometry_kappa * step_feature(curve, geometry_tau)
            residual_geometry = curve.loss - geometry_pred

            base_mae = metrics(curve.loss, base)["mae"]
            table_mae = metrics(curve.loss, table_pred)["mae"]
            geometry_mae = metrics(curve.loss, geometry_pred)["mae"]
            row = {
                "scale": scale,
                "curve": curve_name,
                "label": label,
                "mpl_mae": base_mae,
                "table_tau_mae": table_mae,
                "geometry_tau_mae": geometry_mae,
                "table_delta_pct": 100.0 * (table_mae / base_mae - 1.0),
                "geometry_delta_pct": 100.0 * (geometry_mae / base_mae - 1.0),
                "geometry_minus_table_delta": 100.0 * (geometry_mae / table_mae - 1.0),
                "table_tau": table_tau,
                "geometry_tau": geometry_tau,
                "table_kappa": table_kappa,
                "geometry_kappa": geometry_kappa,
                "route": table["route"],
                "train_curves": table["train_curves"],
                "uses_target_residual": 0,
            }
            rows.append(row)
            panels[(scale, curve_name)] = {
                **row,
                "curve_obj": curve,
                "residual_mpl": residual_mpl,
                "residual_table": residual_table,
                "residual_geometry": residual_geometry,
            }
    return rows, panels


def summarize_group(rows: list[dict[str, object]], group: str, group_rows: list[dict[str, object]]) -> dict[str, object]:
    table = [float(row["table_delta_pct"]) for row in group_rows]
    geometry = [float(row["geometry_delta_pct"]) for row in group_rows]
    rel = [float(row["geometry_minus_table_delta"]) for row in group_rows]
    return {
        "group": group,
        "rows": len(group_rows),
        "table_mean_delta": float(np.mean(table)),
        "table_worst_delta": float(np.max(table)),
        "table_nonharm": int(sum(v <= 1e-10 for v in table)),
        "geometry_mean_delta": float(np.mean(geometry)),
        "geometry_worst_delta": float(np.max(geometry)),
        "geometry_nonharm": int(sum(v <= 1e-10 for v in geometry)),
        "geometry_vs_table_mean_pct": float(np.mean(rel)),
        "geometry_vs_table_worst_pct": float(np.max(rel)),
        "geometry_beats_table": int(sum(g < t for g, t in zip(geometry, table))),
    }


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    core_names = {name for name, _ in CORE_CURVES}
    safety_names = {name for name, _ in EXTRA_SAFETY_CURVES}
    return [
        summarize_group(rows, "core", [row for row in rows if row["curve"] in core_names]),
        summarize_group(rows, "extended", rows),
        summarize_group(rows, "safety_controls", [row for row in rows if row["curve"] in safety_names]),
    ]


def target_summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for curve_name, label in EXTENDED_CURVES:
        sub = [row for row in rows if row["curve"] == curve_name]
        table = [float(row["table_delta_pct"]) for row in sub]
        geometry = [float(row["geometry_delta_pct"]) for row in sub]
        out.append(
            {
                "curve": curve_name,
                "label": label,
                "rows": len(sub),
                "table_mean_delta": float(np.mean(table)),
                "table_worst_delta": float(np.max(table)),
                "geometry_mean_delta": float(np.mean(geometry)),
                "geometry_worst_delta": float(np.max(geometry)),
                "geometry_beats_table": int(sum(g < t for g, t in zip(geometry, table))),
            }
        )
    return out


def plot_panel(ax, panel: dict[str, object], show_ylabel: bool = False) -> None:
    steps = panel["curve_obj"].step
    y_mpl = smooth(panel["residual_mpl"])
    y_table = smooth(panel["residual_table"])
    y_geometry = smooth(panel["residual_geometry"])
    scale = max(float(np.max(np.abs(y_mpl))), float(np.max(np.abs(y_table))), float(np.max(np.abs(y_geometry))), 1e-6)
    ax.axhline(0.0, color="#111111", lw=0.8, alpha=0.82)
    ax.plot(steps, y_mpl, color="#111827", lw=1.45, label="MPL")
    ax.plot(steps, y_table, color="#6b7280", lw=1.35, label="table tau")
    ax.plot(steps, y_geometry, color="#2563eb", lw=1.35, ls="--", label="geometry tau")
    ax.set_ylim(-1.12 * scale, 1.12 * scale)
    ax.set_title(
        f"{panel['label']}\ntable {float(panel['table_delta_pct']):+.1f}% | geom {float(panel['geometry_delta_pct']):+.1f}%",
        fontsize=10.0,
    )
    ax.grid(axis="y", alpha=0.2)
    ax.tick_params(labelsize=8)
    ax.set_xlabel("step", fontsize=8.5)
    if show_ylabel:
        ax.set_ylabel("loss residual", fontsize=8.5)


def plot_core_scale(scale: str, panels: dict[tuple[str, str], dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 7.6), constrained_layout=True)
    for i, (curve_name, _) in enumerate(CORE_CURVES):
        plot_panel(axes.ravel()[i], panels[(scale, curve_name)], show_ylabel=(i % 3 == 0))
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9.0, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.045))
    fig.suptitle(f"{scale}M core residuals: table tau vs geometry tau", fontsize=13)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def plot_safety_scale(scale: str, panels: dict[tuple[str, str], dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 3.2), constrained_layout=True)
    for i, (curve_name, _) in enumerate(EXTRA_SAFETY_CURVES):
        plot_panel(axes.ravel()[i], panels[(scale, curve_name)], show_ylabel=(i == 0))
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9.0, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(f"{scale}M safety controls: table tau vs geometry tau", fontsize=13)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def plot_bar_summary(summary: list[dict[str, object]], path: Path) -> None:
    core_names = {name for name, _ in CORE_CURVES}
    summary = [row for row in summary if row["curve"] in core_names]
    labels = [
        str(row["label"]).replace("WSD sharp", "WSD\nsharp").replace("WSD linear", "WSD\nlinear").replace("WSD-con ", "con\n")
        for row in summary
    ]
    table = np.array([float(row["table_mean_delta"]) for row in summary])
    geometry = np.array([float(row["geometry_mean_delta"]) for row in summary])
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12.4, 4.6), constrained_layout=True)
    ax.axhline(0.0, color="#111111", lw=0.9)
    b1 = ax.bar(x - width / 2, table, width, color="#6b7280", label="table tau")
    b2 = ax.bar(x + width / 2, geometry, width, color="#2563eb", label="geometry tau")
    ax.set_xticks(x, labels)
    ax.set_ylabel("mean MAE change vs MPL (%)")
    ax.set_title("Core schedules: table tau vs geometry tau one-kappa")
    ax.set_ylim(min(float(table.min()), float(geometry.min())) - 7.0, 2.0)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    for bars in [b1, b2]:
        for bar in bars:
            val = float(bar.get_height())
            if abs(val) < 0.05:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 1.5 if val < 0 else val + 0.8,
                f"{val:+.0f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="white" if val < -8.0 else "#111111",
            )
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def write_report(rows: list[dict[str, object]], aggregate: list[dict[str, object]]) -> None:
    core = next(row for row in aggregate if row["group"] == "core")
    extended = next(row for row in aggregate if row["group"] == "extended")
    safety = next(row for row in aggregate if row["group"] == "safety_controls")
    lines = [
        "# Geometry Tau Residual Error Comparison\n\n",
        "This compares two target-holdout one-kappa corrections: the previous discrete route-tau table and the schedule-geometry tau formula.  Neither curve uses target residuals.\n\n",
        "## Aggregate MAE Change vs MPL\n\n",
        "| group | table mean | table worst | table non-harm | geometry mean | geometry worst | geometry non-harm | geometry beats table |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['group']} | {float(row['table_mean_delta']):+.1f}% | "
            f"{float(row['table_worst_delta']):+.1f}% | {int(row['table_nonharm'])}/{int(row['rows'])} | "
            f"{float(row['geometry_mean_delta']):+.1f}% | {float(row['geometry_worst_delta']):+.1f}% | "
            f"{int(row['geometry_nonharm'])}/{int(row['rows'])} | {int(row['geometry_beats_table'])}/{int(row['rows'])} |\n"
        )
    lines += [
        "\n## Figures\n\n",
    ]
    for scale in SCALES:
        lines.append(f"- `{scale}M` core: `core_residuals_{scale}M.png`\n")
        lines.append(f"- `{scale}M` safety controls: `safety_residuals_{scale}M.png`\n")
    lines.append("- Core MAE bar summary: `mae_bar_summary.png`\n")
    lines += [
        "\n## Reading\n\n",
        f"- Core target-holdout changes from table mean `{float(core['table_mean_delta']):+.1f}%` / worst `{float(core['table_worst_delta']):+.1f}%` to geometry mean `{float(core['geometry_mean_delta']):+.1f}%` / worst `{float(core['geometry_worst_delta']):+.1f}%`.\n",
        f"- Extended safety remains non-harming under geometry tau: `{int(extended['geometry_nonharm'])}/{int(extended['rows'])}` rows, with safety controls `{int(safety['geometry_nonharm'])}/{int(safety['rows'])}`.\n",
        "- The plots show that geometry tau leaves most residual shapes unchanged, but tightens the weak and medium single-step corrections enough to improve the no-nuisance worst case.\n",
    ]
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, panels = analyze()
    aggregate = aggregate_rows(rows)
    target_summary = target_summary_rows(rows)
    write_csv(OUT_DIR / "error_metrics.csv", rows)
    write_csv(OUT_DIR / "aggregate_metrics.csv", aggregate)
    write_csv(OUT_DIR / "target_summary.csv", target_summary)
    for scale in SCALES:
        plot_core_scale(scale, panels, OUT_DIR / f"core_residuals_{scale}M.png")
        plot_safety_scale(scale, panels, OUT_DIR / f"safety_residuals_{scale}M.png")
    plot_bar_summary(target_summary, OUT_DIR / "mae_bar_summary.png")
    write_report(rows, aggregate)
    core = next(row for row in aggregate if row["group"] == "core")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    print(
        f"core table={float(core['table_mean_delta']):+.1f}%/{float(core['table_worst_delta']):+.1f}% "
        f"geometry={float(core['geometry_mean_delta']):+.1f}%/{float(core['geometry_worst_delta']):+.1f}%"
    )


if __name__ == "__main__":
    main()
