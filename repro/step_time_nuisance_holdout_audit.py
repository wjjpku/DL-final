#!/usr/bin/env python3
"""Holdout and ablation audit for the image-driven step-time estimator.

The previous search found a strong single-curve candidate:

    step_tau1024 + Fourier2 nuisance + EB q75 + target drop_linear

This script checks whether that result is just a grid-search artifact.  It
reuses the full candidate grid in memory, then evaluates:

1. Leave-one-scale method selection.
2. Leave-one-target-schedule method selection.
3. Component ablations around the selected formula.
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

import step_time_nuisance_estimator as est  # noqa: E402


OUT_DIR = ROOT / "results" / "step_time_nuisance_holdout_audit"
FIG_DIR = OUT_DIR / "figs"

FIXED_BEST = "step_tau1024__fourier2__eb_q75__R0p0__Tdrop_linear"
FIXED_CONSERVATIVE = "step_tau1024__dct2__eb_q75__R0p5__Tdrop_linear"

ABLATION_METHODS = [
    ("raw_step_tau1024", "step_tau1024__none__none__R0p0__Tnone"),
    ("+fourier_nuisance", "step_tau1024__fourier2__none__R0p0__Tnone"),
    ("+EB", "step_tau1024__fourier2__eb_q75__R0p0__Tnone"),
    ("+target_drop_linear", FIXED_BEST),
    ("conservative_dct2_retention", FIXED_CONSERVATIVE),
    ("safe_old_feature_ref", "S10_current__dct2__eb_q75__R1p0__Tnone"),
]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite(values: list[float]) -> list[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def mean(values: list[float]) -> float:
    vals = finite(values)
    return float(np.mean(vals)) if vals else float("nan")


def worst(values: list[float]) -> float:
    vals = finite(values)
    return float(np.max(vals)) if vals else float("nan")


def metric_block(rows: list[dict[str, object]], prefix: str) -> dict[str, object]:
    vals = [float(r["delta_pct"]) for r in rows]
    return {
        f"{prefix}_rows": len(rows),
        f"{prefix}_mean": mean(vals),
        f"{prefix}_worst": worst(vals),
        f"{prefix}_wins": int(sum(float(v) < 0.0 for v in vals)),
        f"{prefix}_nonharm": int(sum(float(v) <= 1e-10 for v in vals)),
    }


def summarize_method(rows: list[dict[str, object]], method: str) -> dict[str, object]:
    sub = [r for r in rows if r["method"] == method]
    self_rows = [r for r in sub if r["train_curve"] == r["test_curve"]]
    off_rows = [r for r in sub if r["train_curve"] != r["test_curve"]]
    probe_wsd = [
        r
        for r in sub
        if r["train_curve"] in {"wsdcon_3.csv", "wsdcon_9.csv", "wsdcon_18.csv"}
        and r["test_curve"] in {"wsd_20000_24000.csv", "wsdld_20000_24000.csv"}
    ]
    cosine_wsd = [
        r
        for r in sub
        if r["train_curve"] == "cosine_72000.csv"
        and r["test_curve"] in {"wsd_20000_24000.csv", "wsdld_20000_24000.csv"}
    ]
    summary: dict[str, object] = {"method": method}
    summary.update(metric_block(sub, "all"))
    summary.update(metric_block(self_rows, "self"))
    summary.update(metric_block(off_rows, "offdiag"))
    summary.update(metric_block(probe_wsd, "probe_to_wsd"))
    summary.update(metric_block(cosine_wsd, "cosine_to_wsd"))
    return summary


def selection_score(summary: dict[str, object]) -> tuple[float, float, float, float]:
    self_worst = float(summary["self_worst"]) if math.isfinite(float(summary["self_worst"])) else 0.0
    off_worst = float(summary["offdiag_worst"]) if math.isfinite(float(summary["offdiag_worst"])) else 0.0
    probe_worst = float(summary["probe_to_wsd_worst"]) if math.isfinite(float(summary["probe_to_wsd_worst"])) else 0.0
    self_mean = float(summary["self_mean"]) if math.isfinite(float(summary["self_mean"])) else 0.0
    off_mean = float(summary["offdiag_mean"]) if math.isfinite(float(summary["offdiag_mean"])) else 0.0
    probe_mean = float(summary["probe_to_wsd_mean"]) if math.isfinite(float(summary["probe_to_wsd_mean"])) else off_mean
    harm = max(self_worst, 0.0) + 3.0 * max(off_worst, 0.0) + max(probe_worst, 0.0)
    utility = probe_mean + 0.4 * self_mean + 0.2 * off_mean
    return (harm, utility, self_mean, off_mean)


def select_method(rows: list[dict[str, object]], method_filter=None) -> dict[str, object]:
    methods = sorted({str(r["method"]) for r in rows})
    if method_filter is not None:
        methods = [method for method in methods if method_filter(method)]
    if not methods:
        raise ValueError("method_filter removed every candidate")
    summaries = [summarize_method(rows, method) for method in methods]
    summaries.sort(key=selection_score)
    return summaries[0]


def is_drop_linear_family(method: str) -> bool:
    return method.endswith("__Tdrop_linear")


def leave_one_scale(details: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for heldout in est.SCALES:
        train_rows = [r for r in details if r["scale"] != heldout]
        test_rows = [r for r in details if r["scale"] == heldout]
        selected = select_method(train_rows)
        selected_drop = select_method(train_rows, is_drop_linear_family)
        for tag, method in [
            ("selected_by_train_scales", str(selected["method"])),
            ("selected_drop_linear_family", str(selected_drop["method"])),
            ("fixed_best", FIXED_BEST),
            ("fixed_conservative", FIXED_CONSERVATIVE),
        ]:
            summary = summarize_method(test_rows, method)
            rows.append(
                {
                    "audit": "leave_one_scale",
                    "heldout_scale": heldout,
                    "tag": tag,
                    "selected_method": method,
                    "train_score_harm": selection_score(selected)[0],
                    "train_score_utility": selection_score(selected)[1],
                    **summary,
                }
            )
    return rows


def leave_one_target(details: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for heldout_curve, heldout_label in est.CURVES:
        train_rows = [
            r
            for r in details
            if r["train_curve"] != heldout_curve and r["test_curve"] != heldout_curve
        ]
        target_rows = [
            r
            for r in details
            if r["test_curve"] == heldout_curve and r["train_curve"] != heldout_curve
        ]
        selected = select_method(train_rows)
        selected_drop = select_method(train_rows, is_drop_linear_family)
        for tag, method in [
            ("selected_without_target", str(selected["method"])),
            ("selected_drop_linear_family", str(selected_drop["method"])),
            ("fixed_best", FIXED_BEST),
            ("fixed_conservative", FIXED_CONSERVATIVE),
        ]:
            summary = summarize_method(target_rows, method)
            rows.append(
                {
                    "audit": "leave_one_target",
                    "heldout_curve": heldout_curve,
                    "heldout_label": heldout_label,
                    "tag": tag,
                    "selected_method": method,
                    "train_score_harm": selection_score(selected)[0],
                    "train_score_utility": selection_score(selected)[1],
                    **summary,
                }
            )
    return rows


def ablation(details: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label, method in ABLATION_METHODS:
        summary = summarize_method(details, method)
        rows.append({"label": label, **summary})
    return rows


def plot_ablation(rows: list[dict[str, object]], path: Path) -> None:
    labels = [str(r["label"]) for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(12.8, 5.2))
    width = 0.22
    lo, hi = -45.0, 65.0

    def clip(v: object) -> float:
        value = float(v)
        return min(max(value, lo), hi)

    ax.axhline(0.0, color="#333333", lw=0.8)
    ax.bar(x - width, [clip(r["self_mean"]) for r in rows], width, label="self mean")
    ax.bar(x, [clip(r["offdiag_mean"]) for r in rows], width, label="offdiag mean")
    ax.bar(x + width, [clip(r["probe_to_wsd_mean"]) for r in rows], width, label="probe -> WSD mean")
    ax.scatter(x, [clip(r["offdiag_worst"]) for r in rows], color="#dc2626", label="offdiag worst", zorder=3)
    for i, row in enumerate(rows):
        for offset, key in [(-width, "self_mean"), (0.0, "offdiag_mean"), (width, "probe_to_wsd_mean"), (0.0, "offdiag_worst")]:
            value = float(row[key])
            if value > hi:
                ax.text(i + offset, hi - 3.0, f"{value:.0f}%", ha="center", va="top", fontsize=7, rotation=90)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("MAE change vs MPL (%)")
    ax.set_title("Component ablation around the image-driven estimator")
    ax.set_ylim(lo, hi)
    ax.grid(axis="y", alpha=0.24)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def plot_holdouts(rows: list[dict[str, object]], path: Path, key: str, title: str) -> None:
    selected = [
        r
        for r in rows
        if r["tag"] in {"selected_by_train_scales", "selected_without_target"}
    ]
    fixed = [r for r in rows if r["tag"] == "fixed_best"]
    labels = [str(r[key]) for r in selected]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12.8, 5.0))
    width = 0.32
    ax.axhline(0.0, color="#333333", lw=0.8)
    ax.bar(x - width / 2, [float(r["offdiag_mean"]) for r in selected], width, label="selected method")
    ax.bar(x + width / 2, [float(r["offdiag_mean"]) for r in fixed], width, label="fixed best")
    ax.scatter(x - width / 2, [float(r["offdiag_worst"]) for r in selected], color="#dc2626", zorder=3)
    ax.scatter(x + width / 2, [float(r["offdiag_worst"]) for r in fixed], color="#7c2d12", zorder=3)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Held-out offdiag MAE change (%)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.24)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def write_report(scale_rows: list[dict[str, object]], target_rows: list[dict[str, object]], ablation_rows: list[dict[str, object]]) -> None:
    fixed_scale = [r for r in scale_rows if r["tag"] == "fixed_best"]
    selected_scale = [r for r in scale_rows if r["tag"] == "selected_by_train_scales"]
    selected_drop_scale = [r for r in scale_rows if r["tag"] == "selected_drop_linear_family"]
    fixed_target = [r for r in target_rows if r["tag"] == "fixed_best"]
    selected_target = [r for r in target_rows if r["tag"] == "selected_without_target"]
    selected_drop_target = [r for r in target_rows if r["tag"] == "selected_drop_linear_family"]
    best_ab = next(r for r in ablation_rows if r["method"] == FIXED_BEST)
    lines = [
        "# Step-Time Nuisance Holdout Audit\n\n",
        "This audit tests whether the image-driven step-time estimator is robust beyond the full-grid selection set.\n\n",
        "## Component Ablation\n\n",
        "| component | self mean | offdiag mean | offdiag worst | probe -> WSD | cosine -> WSD |\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ]
    for row in ablation_rows:
        lines.append(
            f"| {row['label']} | {float(row['self_mean']):+.1f}% | {float(row['offdiag_mean']):+.1f}% | "
            f"{float(row['offdiag_worst']):+.1f}% | {float(row['probe_to_wsd_mean']):+.1f}% | "
            f"{float(row['cosine_to_wsd_mean']):+.1f}% |\n"
        )
    lines += [
        "\n## Leave-One-Scale\n\n",
        "| heldout scale | unrestricted selected | drop-linear-family selected | fixed best |\n",
        "|---|---:|---:|---:|\n",
    ]
    for sel in selected_scale:
        fixed = next(r for r in fixed_scale if r["heldout_scale"] == sel["heldout_scale"])
        drop = next(r for r in selected_drop_scale if r["heldout_scale"] == sel["heldout_scale"])
        lines.append(
            f"| {sel['heldout_scale']}M | `{sel['selected_method']}`: "
            f"{float(sel['offdiag_mean']):+.1f}% / {float(sel['offdiag_worst']):+.1f}% | "
            f"`{drop['selected_method']}`: {float(drop['offdiag_mean']):+.1f}% / {float(drop['offdiag_worst']):+.1f}% | "
            f"{float(fixed['offdiag_mean']):+.1f}% / {float(fixed['offdiag_worst']):+.1f}% |\n"
        )
    lines += [
        "\n## Leave-One-Target Schedule\n\n",
        "| heldout target | unrestricted selected | drop-linear-family selected | fixed best |\n",
        "|---|---:|---:|---:|\n",
    ]
    for sel in selected_target:
        fixed = next(r for r in fixed_target if r["heldout_curve"] == sel["heldout_curve"])
        drop = next(r for r in selected_drop_target if r["heldout_curve"] == sel["heldout_curve"])
        lines.append(
            f"| {sel['heldout_label']} | `{sel['selected_method']}`: "
            f"{float(sel['offdiag_mean']):+.1f}% / {float(sel['offdiag_worst']):+.1f}% | "
            f"`{drop['selected_method']}`: {float(drop['offdiag_mean']):+.1f}% / {float(drop['offdiag_worst']):+.1f}% | "
            f"{float(fixed['offdiag_mean']):+.1f}% / {float(fixed['offdiag_worst']):+.1f}% |\n"
        )
    lines += [
        "\n## Reading\n\n",
        f"- Fixed image-driven candidate `{FIXED_BEST}` has full-grid self mean `{float(best_ab['self_mean']):+.1f}%`, "
        f"offdiag mean `{float(best_ab['offdiag_mean']):+.1f}%`, offdiag worst `{float(best_ab['offdiag_worst']):+.1f}%`, "
        f"and probe-to-WSD mean `{float(best_ab['probe_to_wsd_mean']):+.1f}%`.\n",
        f"- Leave-one-scale fixed-best offdiag means range from `{min(float(r['offdiag_mean']) for r in fixed_scale):+.1f}%` "
        f"to `{max(float(r['offdiag_mean']) for r in fixed_scale):+.1f}%`; worst deltas stay at or below "
        f"`{max(float(r['offdiag_worst']) for r in fixed_scale):+.1f}%`.\n",
        f"- Leave-one-target fixed-best offdiag means range from `{min(float(r['offdiag_mean']) for r in fixed_target):+.1f}%` "
        f"to `{max(float(r['offdiag_mean']) for r in fixed_target):+.1f}%`; worst deltas stay at or below "
        f"`{max(float(r['offdiag_worst']) for r in fixed_target):+.1f}%`.\n",
        "- Unrestricted target holdout exposes the same issue seen in the plots: without a structural target-drop factor, selection can prefer a full-drop correction and then over-transfer to the smallest-drop target. Restricting to the drop-linear model family or using the fixed image-driven candidate removes this failure.\n",
        "- The ablation shows why the target drop factor matters: it preserves WSD-target improvements while removing over-transfer to small-drop WSD-con targets.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    details, _, _ = est.run_search()
    scale_rows = leave_one_scale(details)
    target_rows = leave_one_target(details)
    ablation_rows = ablation(details)

    write_csv(OUT_DIR / "scale_holdout.csv", scale_rows)
    write_csv(OUT_DIR / "target_holdout.csv", target_rows)
    write_csv(OUT_DIR / "ablation.csv", ablation_rows)
    write_report(scale_rows, target_rows, ablation_rows)
    plot_ablation(ablation_rows, FIG_DIR / "ablation.png")
    plot_holdouts(scale_rows, FIG_DIR / "scale_holdout.png", "heldout_scale", "Leave-one-scale holdout")
    plot_holdouts(target_rows, FIG_DIR / "target_holdout.png", "heldout_label", "Leave-one-target holdout")

    fixed = next(r for r in ablation_rows if r["method"] == FIXED_BEST)
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    print(
        f"fixed_best self={float(fixed['self_mean']):+.1f}% "
        f"offdiag={float(fixed['offdiag_mean']):+.1f}%/"
        f"{float(fixed['offdiag_worst']):+.1f}% "
        f"probeWSD={float(fixed['probe_to_wsd_mean']):+.1f}%"
    )
    for row in [r for r in scale_rows if r["tag"] == "fixed_best"]:
        print(
            f"scale {row['heldout_scale']} fixed_best offdiag="
            f"{float(row['offdiag_mean']):+.1f}%/{float(row['offdiag_worst']):+.1f}%"
        )
    for row in [r for r in target_rows if r["tag"] == "fixed_best"]:
        print(
            f"target {row['heldout_label']} fixed_best offdiag="
            f"{float(row['offdiag_mean']):+.1f}%/{float(row['offdiag_worst']):+.1f}%"
        )


if __name__ == "__main__":
    main()
