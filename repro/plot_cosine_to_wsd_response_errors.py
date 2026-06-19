#!/usr/bin/env python3
"""Plot cosine-calibrated WSD prediction errors.

This script is deliberately assignment-focused:
  * fitted amplitudes come from cosine_72000 only;
  * WSD losses are used here only for evaluation and visualization;
  * paper/slides artifacts are not touched.

It compares three curves on each WSD target:
  1. MPL baseline,
  2. the previous cosine-calibrated nextgen correction,
  3. the current S-time response-kernel candidate selected by
     cosine_to_wsd_response_search.py.
"""
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
from reproduce_cosine_to_wsd import (  # noqa: E402
    MPL_PRECOMPUTED_INIT,
    PEAK_LR,
    SCALES,
    load_curve,
    metrics,
    mpl_predict,
)
from cosine_to_wsd_response_search import stime_feature as response_feature  # noqa: E402


TARGETS = [
    ("wsd_20000_24000.csv", "WSD sharp"),
    ("wsdld_20000_24000.csv", "WSD linear"),
    ("wsdcon_3.csv", "WSD-con 3e-5"),
    ("wsdcon_9.csv", "WSD-con 9e-5"),
    ("wsdcon_18.csv", "WSD-con 18e-5"),
]

SOURCE_CURVE = "cosine_72000.csv"
OLD_RESPONSE_LAMBDA = 10.0
OUT_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "error_comparison"
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


def best_config() -> dict[str, str]:
    rows = read_csv(ROOT / "results" / "cosine_to_wsd_response_search" / "safe_configs_top100.csv")
    if not rows:
        raise RuntimeError("safe_configs_top100.csv is empty")
    return rows[0]


def detail_lookup(path: Path, *, config_id: str | None = None) -> dict[tuple[str, str], dict[str, str]]:
    rows = read_csv(path)
    if config_id is not None:
        rows = [row for row in rows if row["config_id"] == config_id]
    out: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        out[(row["scale"], row["test_curve"])] = row
    return out


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


def fmt_delta(value: float) -> str:
    return f"{value:+.1f}%"


def build_panels() -> tuple[list[dict[str, object]], dict[tuple[str, str], dict[str, object]], dict[str, str]]:
    best = best_config()
    best_id = best["config_id"]
    response_lam = float(best["response_lambda"])
    old_rows = detail_lookup(ROOT / "results" / "cosine_to_wsd_focus" / "nextgen_safe_details.csv")
    new_rows = detail_lookup(
        ROOT / "results" / "cosine_to_wsd_response_search" / "top_safe_details.csv",
        config_id=best_id,
    )
    expected = {(scale, curve) for scale in SCALES for curve, _ in TARGETS}
    missing_old = expected - set(old_rows)
    missing_new = expected - set(new_rows)
    if missing_old:
        raise RuntimeError(f"missing old nextgen rows: {sorted(missing_old)}")
    if missing_new:
        raise RuntimeError(f"missing response rows for config {best_id}: {sorted(missing_new)}")

    rows: list[dict[str, object]] = []
    panels: dict[tuple[str, str], dict[str, object]] = {}
    for scale in SCALES:
        for curve_name, label in TARGETS:
            curve = load_curve(scale, curve_name)
            baseline = mpl_predict(MPL_PRECOMPUTED_INIT[scale], curve)

            old_info = old_rows[(scale, curve_name)]
            old_phi = old_stime_feature(curve, OLD_RESPONSE_LAMBDA)
            old_kappa = float(old_info["kappa"]) * float(old_info["target_factor"])
            old_pred = baseline + old_kappa * old_phi

            new_info = new_rows[(scale, curve_name)]
            new_phi = response_feature(curve, response_lam)
            new_kappa = float(new_info["kappa"]) * float(new_info["target_factor"])
            new_pred = baseline + new_kappa * new_phi

            base_mae = metrics(curve.loss, baseline)["mae"]
            old_mae = metrics(curve.loss, old_pred)["mae"]
            new_mae = metrics(curve.loss, new_pred)["mae"]
            row = {
                "scale": scale,
                "test_curve": curve_name,
                "test_label": label,
                "source_curve": SOURCE_CURVE,
                "mpl_mae": base_mae,
                "old_mae": old_mae,
                "new_mae": new_mae,
                "old_delta_pct": 100.0 * (old_mae / base_mae - 1.0),
                "new_delta_pct": 100.0 * (new_mae / base_mae - 1.0),
                "new_minus_old_delta_pct": 100.0 * (new_mae / old_mae - 1.0),
                "old_kappa": old_kappa,
                "new_kappa": new_kappa,
                "old_response_lambda": OLD_RESPONSE_LAMBDA,
                "new_response_lambda": response_lam,
                "new_config_id": best_id,
                "new_nuisance_lambda": float(best["nuisance_lambda"]),
                "new_max_mode": int(best["max_mode"]),
                "new_ridge_tau": float(best["ridge_tau"]),
                "new_retention_power": float(best["retention_power"]),
                "new_rho": float(best["rho"]),
            }
            rows.append(row)
            panels[(scale, curve_name)] = {
                **row,
                "curve": curve,
                "residual_mpl": curve.loss - baseline,
                "residual_old": curve.loss - old_pred,
                "residual_new": curve.loss - new_pred,
            }
    return rows, panels, best


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for curve_name, label in TARGETS:
        sub = [row for row in rows if row["test_curve"] == curve_name]
        old = np.array([float(row["old_delta_pct"]) for row in sub], dtype=np.float64)
        new = np.array([float(row["new_delta_pct"]) for row in sub], dtype=np.float64)
        out.append(
            {
                "test_curve": curve_name,
                "test_label": label,
                "rows": len(sub),
                "old_mean_delta_pct": float(np.mean(old)),
                "old_worst_delta_pct": float(np.max(old)),
                "old_wins": int(np.sum(old < 0.0)),
                "new_mean_delta_pct": float(np.mean(new)),
                "new_worst_delta_pct": float(np.max(new)),
                "new_wins": int(np.sum(new < 0.0)),
                "new_mean_gain_vs_old_pct": float(np.mean([float(row["new_minus_old_delta_pct"]) for row in sub])),
            }
        )
    old_all = np.array([float(row["old_delta_pct"]) for row in rows], dtype=np.float64)
    new_all = np.array([float(row["new_delta_pct"]) for row in rows], dtype=np.float64)
    out.append(
        {
            "test_curve": "ALL",
            "test_label": "All WSD targets",
            "rows": len(rows),
            "old_mean_delta_pct": float(np.mean(old_all)),
            "old_worst_delta_pct": float(np.max(old_all)),
            "old_wins": int(np.sum(old_all < 0.0)),
            "new_mean_delta_pct": float(np.mean(new_all)),
            "new_worst_delta_pct": float(np.max(new_all)),
            "new_wins": int(np.sum(new_all < 0.0)),
            "new_mean_gain_vs_old_pct": float(np.mean([float(row["new_minus_old_delta_pct"]) for row in rows])),
        }
    )
    return out


def plot_scale(scale: str, panels: dict[tuple[str, str], dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.2), constrained_layout=True)
    axes_flat = axes.ravel()
    for i, (curve_name, label) in enumerate(TARGETS):
        panel = panels[(scale, curve_name)]
        curve = panel["curve"]
        ax = axes_flat[i]
        steps = curve.step
        ax.axhline(0.0, color="#111111", lw=0.8, alpha=0.75)
        ax.plot(steps, smooth(panel["residual_mpl"]), color="#111827", lw=1.35, label="MPL")
        ax.plot(steps, smooth(panel["residual_old"]), color="#dc2626", lw=1.25, label="MPL+old")
        ax.plot(steps, smooth(panel["residual_new"]), color="#2563eb", lw=1.35, label="MPL+response")
        ax2 = ax.twinx()
        ax2.plot(steps, curve.lrs[curve.step] / PEAK_LR, color="#b45309", lw=0.8, alpha=0.28)
        ax2.set_ylim(-0.04, 1.05)
        ax2.tick_params(labelsize=7, colors="#92400e")
        ax.grid(axis="y", alpha=0.18)
        ax.tick_params(labelsize=8)
        ax.set_title(
            f"{label} | old {fmt_delta(float(panel['old_delta_pct']))}, "
            f"new {fmt_delta(float(panel['new_delta_pct']))}",
            fontsize=9.0,
        )
        if i == 0:
            ax.legend(frameon=False, fontsize=8, loc="upper left")
    axes_flat[-1].axis("off")
    fig.suptitle(f"Cosine-calibrated residual transfer to WSD targets ({scale}M)", fontsize=12.5)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def plot_summary(summary: list[dict[str, object]], path: Path) -> None:
    rows = [row for row in summary if row["test_curve"] != "ALL"]
    labels = [str(row["test_label"]).replace("WSD-con ", "con ") for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    old = [float(row["old_mean_delta_pct"]) for row in rows]
    new = [float(row["new_mean_delta_pct"]) for row in rows]

    fig, ax = plt.subplots(figsize=(9.0, 4.2), constrained_layout=True)
    ax.axhline(0.0, color="#111111", lw=0.9, alpha=0.75)
    ax.bar(x - width / 2, old, width, color="#dc2626", label="MPL+old")
    ax.bar(x + width / 2, new, width, color="#2563eb", label="MPL+response")
    ax.set_xticks(x, labels, rotation=12, ha="right")
    ax.set_ylabel("Mean MAE change vs MPL (%)")
    ax.set_title("Cosine-fitted correction on WSD targets")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def write_report(summary: list[dict[str, object]], best: dict[str, str]) -> None:
    all_row = next(row for row in summary if row["test_curve"] == "ALL")
    lines = [
        "# Cosine-to-WSD Error Comparison\n\n",
        "This visualization keeps the assignment protocol explicit: correction amplitudes are fitted from `cosine_72000.csv`; WSD-family losses are used only for evaluation and plotting.\n\n",
        "## Compared Methods\n\n",
        "- `MPL`: original MPL baseline.\n",
        "- `MPL+old`: previous cosine-calibrated nextgen correction with S-time response `lambda=10` and target-retention gate.\n",
        "- `MPL+response`: current cosine-calibrated response-kernel candidate.\n\n",
        "Current response-kernel candidate:\n\n",
        "```text\n",
        f"response_lambda = {float(best['response_lambda']):g}\n",
        f"nuisance_lambda = {float(best['nuisance_lambda']):g}\n",
        f"max_mode = {int(best['max_mode'])}\n",
        f"ridge_tau = {float(best['ridge_tau']):g}\n",
        f"retention_power = {float(best['retention_power']):g}\n",
        f"rho = {float(best['rho']):g}\n",
        "```\n\n",
        "## Aggregate Result\n\n",
        f"- Old mean / worst: `{fmt_delta(float(all_row['old_mean_delta_pct']))}` / `{fmt_delta(float(all_row['old_worst_delta_pct']))}`.\n",
        f"- New mean / worst: `{fmt_delta(float(all_row['new_mean_delta_pct']))}` / `{fmt_delta(float(all_row['new_worst_delta_pct']))}`.\n",
        f"- Wins: old `{int(all_row['old_wins'])}/{int(all_row['rows'])}`, new `{int(all_row['new_wins'])}/{int(all_row['rows'])}`.\n\n",
        "## Target Breakdown\n\n",
        "| target | old mean | old worst | new mean | new worst | new wins |\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ]
    for row in summary:
        if row["test_curve"] == "ALL":
            continue
        lines.append(
            f"| {row['test_label']} | {fmt_delta(float(row['old_mean_delta_pct']))} | "
            f"{fmt_delta(float(row['old_worst_delta_pct']))} | "
            f"{fmt_delta(float(row['new_mean_delta_pct']))} | "
            f"{fmt_delta(float(row['new_worst_delta_pct']))} | "
            f"{int(row['new_wins'])}/{int(row['rows'])} |\n"
        )
    lines += [
        "\n## Reading\n\n",
        "- The new candidate remains a cosine-fitted model: WSD schedules only contribute their LR-derived response feature at prediction time.\n",
        "- The visible improvement is mainly from changing the response time scale from the old general-purpose `lambda=10` to a cosine-to-WSD transfer value near `lambda=20`, plus stronger nuisance removal before estimating kappa from cosine.\n",
        "- This is still a development result because the final candidate was selected by WSD-family ranking; a stronger final protocol should use a held-out split of WSD types or new schedules.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows, panels, best = build_panels()
    summary = aggregate(rows)
    write_csv(OUT_DIR / "error_metrics.csv", rows)
    write_csv(OUT_DIR / "target_summary.csv", summary)
    for scale in SCALES:
        plot_scale(scale, panels, FIG_DIR / f"residuals_{scale}M.png")
    plot_summary(summary, FIG_DIR / "mae_summary_old_vs_response.png")
    write_report(summary, best)
    print(f"wrote {OUT_DIR / 'error_metrics.csv'}")
    print(f"wrote {OUT_DIR / 'target_summary.csv'}")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    print(f"wrote {FIG_DIR}")


if __name__ == "__main__":
    main()
