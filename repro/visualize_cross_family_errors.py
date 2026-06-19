#!/usr/bin/env python3
"""Visualize why cross-family DropRelaxS transfer can fail.

The matrix experiment reports aggregate MAE changes.  This diagnostic script
opens the largest failure cells and plots the actual time-course error:

    prediction - true loss

It is intentionally read-only with respect to the experiment definition.  The
same lambda, features, MPL backbone, and per-scale kappa fits are imported from
current_law_decay_matrix.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent
ROOT = REPO.parent
sys.path.insert(0, str(REPO))

from current_law_decay_matrix import (  # noqa: E402
    FAMILIES,
    LAMBDA,
    feature_cache,
    fit_kappa,
    run_matrix,
)
from reproduce_cosine_to_wsd import (  # noqa: E402
    MPL_PRECOMPUTED_INIT,
    load_curve,
    metrics,
    mpl_predict,
)


OUT_DIR = ROOT / "results" / "current_law_decay_matrix" / "error_visualization"

CASE_SELECTORS = [
    {
        "key": "cosine_to_wsdcon_worst",
        "label": "Worst: cosine -> WSD-con",
        "train_family": "Cosine decay",
        "test_family": "WSD-con step",
        "mode": "max_delta",
    },
    {
        "key": "cosine_to_wsd_worst",
        "label": "Worst: cosine -> sharp WSD",
        "train_family": "Cosine decay",
        "test_family": "WSD sharp",
        "mode": "max_delta",
    },
    {
        "key": "sharp_to_wsdcon_worst",
        "label": "Worst: sharp WSD -> WSD-con",
        "train_family": "WSD sharp",
        "test_family": "WSD-con step",
        "mode": "max_delta",
    },
    {
        "key": "probe_to_wsd_best",
        "label": "Contrast: WSD-con -> sharp WSD",
        "train_family": "WSD-con step",
        "test_family": "WSD sharp",
        "mode": "min_delta",
    },
]


def choose_cases(details: list[dict[str, object]]) -> list[dict[str, object]]:
    cases = []
    for selector in CASE_SELECTORS:
        rows = [
            row
            for row in details
            if row["train_family"] == selector["train_family"]
            and row["test_family"] == selector["test_family"]
        ]
        if not rows:
            raise RuntimeError(f"No rows match selector {selector}")
        reverse = selector["mode"] == "max_delta"
        row = sorted(rows, key=lambda r: float(r["delta_pct"]), reverse=reverse)[0]
        out = dict(row)
        out["case_key"] = selector["key"]
        out["case_label"] = selector["label"]
        cases.append(out)
    return cases


def build_case(row: dict[str, object], feats: dict[tuple[str, str], np.ndarray]) -> dict[str, object]:
    scale = str(row["scale"])
    test_curve = str(row["test_curve"])
    train_family = str(row["train_family"])
    train_names = dict(FAMILIES)[train_family]

    kappa = fit_kappa(scale, train_names, feats)
    curve = load_curve(scale, test_curve)
    base = mpl_predict(MPL_PRECOMPUTED_INIT[scale], curve)
    feature = feats[(scale, test_curve)]
    correction = kappa * feature
    pred = base + correction

    base_metrics = metrics(curve.loss, base)
    corr_metrics = metrics(curve.loss, pred)
    return {
        **row,
        "train_names": train_names,
        "kappa": kappa,
        "curve": curve,
        "true": curve.loss,
        "base": base,
        "pred": pred,
        "feature": feature,
        "correction": correction,
        "base_error": base - curve.loss,
        "corr_error": pred - curve.loss,
        "base_metrics": base_metrics,
        "corr_metrics": corr_metrics,
        "delta_pct": 100.0 * (corr_metrics["mae"] / base_metrics["mae"] - 1.0),
    }


def focus_xlim(curve, correction: np.ndarray) -> tuple[float, float]:
    steps = curve.step.astype(float)
    signal = np.abs(correction)
    max_signal = float(signal.max()) if len(signal) else 0.0
    if max_signal > 0:
        idx = np.flatnonzero(signal >= 0.02 * max_signal)
    else:
        idx = np.array([], dtype=int)

    if len(idx) >= 2:
        lo = float(steps[idx[0]])
        hi = float(steps[idx[-1]])
    else:
        lr_drop = np.maximum(curve.lrs[:-1] - curve.lrs[1:], 0.0)
        if len(lr_drop) and float(lr_drop.max()) > 0:
            drop_idx = np.flatnonzero(lr_drop >= 0.02 * float(lr_drop.max()))
            lo = float(drop_idx[0])
            hi = float(steps[-1])
        else:
            lo = float(steps[0])
            hi = float(steps[-1])

    span = max(hi - lo, 1.0)
    pad = 0.12 * span
    return max(float(steps[0]), lo - pad), min(float(steps[-1]), hi + pad)


def case_title(case: dict[str, object]) -> str:
    curve = str(case["test_curve"]).replace(".csv", "")
    scale = str(case["scale"])
    delta = float(case["delta_pct"])
    return f"{case['case_label']}\n{scale}M / {curve} / Delta MAE {delta:+.0f}%"


def plot_case_on_axes(case: dict[str, object], ax_top, ax_err, show_legend: bool) -> None:
    curve = case["curve"]
    steps = curve.step
    true = case["true"]
    base = case["base"]
    pred = case["pred"]
    correction = case["correction"]
    base_error = case["base_error"]
    corr_error = case["corr_error"]

    ax_top.plot(steps, true, color="#111111", lw=1.7, label="true")
    ax_top.plot(steps, base, color="#7a7f87", lw=1.4, ls="--", label="MPL")
    ax_top.plot(steps, pred, color="#2563eb", lw=1.5, label="MPL + DropRelaxS")
    ax_top.set_title(case_title(case), fontsize=9.5, linespacing=1.25)
    ax_top.set_ylabel("loss", fontsize=8.5)
    ax_top.tick_params(labelsize=7.5)
    ax_top.grid(alpha=0.22, lw=0.5)

    ax_err.axhline(0.0, color="#111111", lw=0.8, alpha=0.7)
    ax_err.plot(steps, base_error, color="#7a7f87", lw=1.3, ls="--", label="MPL error")
    ax_err.plot(steps, corr_error, color="#2563eb", lw=1.4, label="corrected error")
    ax_err.plot(steps, correction, color="#d97706", lw=1.2, ls=":", label="added correction")
    ax_err.set_xlabel("step", fontsize=8.5)
    ax_err.set_ylabel("pred - true", fontsize=8.5)
    ax_err.tick_params(labelsize=7.5)
    ax_err.grid(alpha=0.22, lw=0.5)

    xlim = focus_xlim(curve, correction)
    ax_top.set_xlim(xlim)
    ax_err.set_xlim(xlim)

    if show_legend:
        ax_top.legend(loc="best", fontsize=7.2, frameon=False)
        ax_err.legend(loc="best", fontsize=7.2, frameon=False)

    text = (
        f"kappa={float(case['kappa']):.3f}\n"
        f"MPL MAE={case['base_metrics']['mae']:.4f}\n"
        f"corr MAE={case['corr_metrics']['mae']:.4f}"
    )
    ax_top.text(
        0.02,
        0.04,
        text,
        transform=ax_top.transAxes,
        fontsize=7.4,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#cccccc", "alpha": 0.88},
    )


def plot_grid(cases: list[dict[str, object]], path: Path) -> None:
    fig = plt.figure(figsize=(13.4, 10.8))
    grid = fig.add_gridspec(4, 2, height_ratios=[1.15, 1.0, 1.15, 1.0], hspace=0.48, wspace=0.24)
    for i, case in enumerate(cases):
        row = 0 if i < 2 else 2
        col = i % 2
        ax_top = fig.add_subplot(grid[row, col])
        ax_err = fig.add_subplot(grid[row + 1, col], sharex=ax_top)
        plot_case_on_axes(case, ax_top, ax_err, show_legend=(i == 0))
        plt.setp(ax_top.get_xticklabels(), visible=False)
        if i < 2:
            ax_err.set_xlabel("")
    fig.suptitle(
        "Cross-family DropRelaxS transfer errors: failures are dominated by correction mismatch",
        fontsize=13.5,
        y=0.992,
    )
    fig.text(
        0.5,
        0.008,
        "All panels use the same fixed law MPL + kappa * DropRelaxS(lambda=10). "
        "Lower panels plot prediction error; positive values mean over-predicted loss.",
        ha="center",
        fontsize=9.0,
        color="#333333",
    )
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def plot_individual(case: dict[str, object], path: Path) -> None:
    fig, (ax_top, ax_err) = plt.subplots(
        2,
        1,
        figsize=(8.2, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.10},
    )
    plot_case_on_axes(case, ax_top, ax_err, show_legend=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def write_report(path: Path, cases: list[dict[str, object]]) -> None:
    lines = [
        "# Cross-Family Error Visualization\n\n",
        "The plots diagnose the largest cross-schedule gaps in the current-law matrix. "
        "Errors are plotted as `prediction - true loss`; positive error means the model predicts a loss that is too high.\n\n",
        "| case | train -> test | scale | curve | kappa | MPL MAE | corrected MAE | delta |\n",
        "|---|---|---:|---|---:|---:|---:|---:|\n",
    ]
    for case in cases:
        lines.append(
            f"| {case['case_label']} | {case['train_family']} -> {case['test_family']} | "
            f"{case['scale']}M | {case['test_curve']} | {float(case['kappa']):.4f} | "
            f"{case['base_metrics']['mae']:.5f} | {case['corr_metrics']['mae']:.5f} | "
            f"{float(case['delta_pct']):+.1f}% |\n"
        )

    lines += [
        "\n## Reading\n\n",
        "1. The largest failures are correction-amplitude failures.  The MPL baseline is often close enough, but the transferred correction term has the wrong magnitude for the target schedule family.\n",
        "2. Cosine-calibrated kappa is large because cosine fitting treats a smooth long-horizon residual as positive lag.  When the same kappa multiplies a step-like WSD-con feature, it creates a large positive post-drop error.\n",
        "3. Sharp-WSD calibration also over-transfers to WSD-con tails, but less severely.  This indicates that the residual shape learned from a terminal cooldown is not identical to the long constant tail after a step drop.\n",
        "4. WSD-con probes transfer back to sharp WSD because their fitted kappa is much smaller and the correction aligns with the late-cooldown residual instead of dominating it.\n",
        "\n## Files\n\n",
        f"- Combined grid: `{path.parent / 'cross_family_error_cases.png'}`\n",
    ]
    for case in cases:
        lines.append(f"- Individual case: `{path.parent / (str(case['case_key']) + '.png')}`\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _, details = run_matrix()
    feats = feature_cache()
    cases = [build_case(row, feats) for row in choose_cases(details)]

    plot_grid(cases, OUT_DIR / "cross_family_error_cases.png")
    for case in cases:
        plot_individual(case, OUT_DIR / f"{case['case_key']}.png")
    write_report(OUT_DIR / "REPORT.md", cases)

    print(f"wrote {OUT_DIR / 'cross_family_error_cases.png'}")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    for case in cases:
        print(
            f"{case['case_key']:26s} {case['train_family']} -> {case['test_family']} "
            f"scale={case['scale']} curve={case['test_curve']} "
            f"kappa={float(case['kappa']):.4f} "
            f"delta={float(case['delta_pct']):+.1f}%"
        )


if __name__ == "__main__":
    main()
