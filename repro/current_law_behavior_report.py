#!/usr/bin/env python3
"""Comprehensive behavior report for the current DropRelaxS correction.

This is an exploratory report, not a slide artifact.  It keeps the formula fixed:

    prediction = cosine-fit MPL + kappa * DropRelaxS_lambda

and varies only the calibration plan:

  * fixed lambda=10, nonnegative per-scale kappa;
  * lambda selected on the train family, nonnegative per-scale kappa;
  * fixed lambda=10, one shared kappa across scales;
  * fixed lambda=10, signed per-scale kappa.

The output is a standalone Markdown report plus CSV tables and curve figures.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import TwoSlopeNorm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent
ROOT = REPO.parent
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import (  # noqa: E402
    MPL_PRECOMPUTED_INIT,
    PEAK_LR,
    SCALES,
    load_curve,
    metrics,
    mpl_predict,
)
from deep_stime import stime_feature  # noqa: E402
from nonadiabatic_theory import fit_origin  # noqa: E402


OUT_DIR = ROOT / "results" / "current_law_behavior_report"
FIG_DIR = OUT_DIR / "figs"
LAM_GRID = [2, 3, 5, 7, 10, 14, 20, 30, 50, 80]

FAMILIES: list[tuple[str, list[str]]] = [
    ("Cosine decay", ["cosine_72000.csv"]),
    ("WSD sharp", ["wsd_20000_24000.csv"]),
    ("WSD linear", ["wsdld_20000_24000.csv"]),
    ("WSD-con step", ["wsdcon_3.csv", "wsdcon_9.csv", "wsdcon_18.csv"]),
]
SANITY_CURVES = ["constant_72000.csv"]


def all_curve_names() -> list[str]:
    names = {name for _, curves in FAMILIES for name in curves}
    names.update(SANITY_CURVES)
    return sorted(names)


def family_names() -> list[str]:
    return [label for label, _ in FAMILIES]


def curve_family_map() -> dict[str, str]:
    out = {}
    for label, curves in FAMILIES:
        for curve in curves:
            out[curve] = label
    return out


def feature_cache() -> dict[tuple[str, str, float], np.ndarray]:
    cache = {}
    for scale in SCALES:
        for curve_name in all_curve_names():
            curve = load_curve(scale, curve_name)
            for lam in LAM_GRID:
                cache[(scale, curve_name, float(lam))] = stime_feature(curve, float(lam))
    return cache


def fit_kappa_origin(x: np.ndarray, y: np.ndarray, signed: bool) -> float:
    if float(np.dot(x, x)) <= 1e-18:
        return 0.0
    kappa = fit_origin(x, y)[0]
    if not signed:
        kappa = max(0.0, kappa)
    return float(kappa)


def fit_kappa_per_scale(scale: str, train_curves: list[str], lam: float, feats, signed: bool) -> float:
    p = MPL_PRECOMPUTED_INIT[scale]
    xs, ys = [], []
    for name in train_curves:
        curve = load_curve(scale, name)
        xs.append(feats[(scale, name, float(lam))])
        ys.append(curve.loss - mpl_predict(p, curve))
    return fit_kappa_origin(np.concatenate(xs), np.concatenate(ys), signed=signed)


def fit_kappa_shared(train_curves: list[str], lam: float, feats, signed: bool) -> float:
    xs, ys = [], []
    for scale in SCALES:
        p = MPL_PRECOMPUTED_INIT[scale]
        for name in train_curves:
            curve = load_curve(scale, name)
            xs.append(feats[(scale, name, float(lam))])
            ys.append(curve.loss - mpl_predict(p, curve))
    return fit_kappa_origin(np.concatenate(xs), np.concatenate(ys), signed=signed)


def predict_curve(scale: str, name: str, lam: float, kappa: float, feats) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    curve = load_curve(scale, name)
    base = mpl_predict(MPL_PRECOMPUTED_INIT[scale], curve)
    feature = feats[(scale, name, float(lam))]
    return base, base + kappa * feature, feature


def score_curves(scale: str, curves: list[str], lam: float, kappa: float, feats) -> list[dict[str, object]]:
    rows = []
    for name in curves:
        curve = load_curve(scale, name)
        base, pred, _ = predict_curve(scale, name, lam, kappa, feats)
        base_mae = metrics(curve.loss, base)["mae"]
        corr_mae = metrics(curve.loss, pred)["mae"]
        rows.append(
            {
                "scale": scale,
                "curve": name,
                "base_mae": base_mae,
                "corr_mae": corr_mae,
                "delta_pct": 100.0 * (corr_mae / base_mae - 1.0),
                "win": int(corr_mae < base_mae),
            }
        )
    return rows


def train_mae(train_curves: list[str], lam: float, feats, signed: bool, shared: bool) -> float:
    maes = []
    shared_kappa = fit_kappa_shared(train_curves, lam, feats, signed=signed) if shared else None
    for scale in SCALES:
        kappa = shared_kappa
        if kappa is None:
            kappa = fit_kappa_per_scale(scale, train_curves, lam, feats, signed=signed)
        for row in score_curves(scale, train_curves, lam, float(kappa), feats):
            maes.append(float(row["corr_mae"]))
    return float(np.mean(maes))


def select_lambda(train_curves: list[str], feats, signed: bool, shared: bool) -> float:
    return float(min(LAM_GRID, key=lambda lam: train_mae(train_curves, float(lam), feats, signed=signed, shared=shared)))


def run_matrix(mode: str, feats) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if mode == "fixed10_per_scale_nonnegative":
        shared, signed, select = False, False, False
    elif mode == "selected_lambda_per_scale_nonnegative":
        shared, signed, select = False, False, True
    elif mode == "fixed10_shared_nonnegative":
        shared, signed, select = True, False, False
    elif mode == "fixed10_per_scale_signed":
        shared, signed, select = False, True, False
    else:
        raise ValueError(mode)

    summary, details = [], []
    for train_label, train_curves in FAMILIES:
        lam = select_lambda(train_curves, feats, signed=signed, shared=shared) if select else 10.0
        shared_kappa = fit_kappa_shared(train_curves, lam, feats, signed=signed) if shared else None
        for test_label, test_curves in FAMILIES:
            cell_rows = []
            kappas = []
            for scale in SCALES:
                kappa = shared_kappa
                if kappa is None:
                    kappa = fit_kappa_per_scale(scale, train_curves, lam, feats, signed=signed)
                kappas.append(float(kappa))
                for row in score_curves(scale, test_curves, lam, float(kappa), feats):
                    row.update(
                        {
                            "mode": mode,
                            "train_family": train_label,
                            "test_family": test_label,
                            "train_curves": "+".join(x.replace(".csv", "") for x in train_curves),
                            "lambda": lam,
                            "kappa": float(kappa),
                        }
                    )
                    details.append(row)
                    cell_rows.append(row)

            base = np.array([float(r["base_mae"]) for r in cell_rows])
            corr = np.array([float(r["corr_mae"]) for r in cell_rows])
            summary.append(
                {
                    "mode": mode,
                    "train_family": train_label,
                    "test_family": test_label,
                    "lambda": lam,
                    "base_mae": float(base.mean()),
                    "corr_mae": float(corr.mean()),
                    "delta_pct": 100.0 * (float(corr.mean()) / float(base.mean()) - 1.0),
                    "wins": sum(int(r["win"]) for r in cell_rows),
                    "tests": len(cell_rows),
                    "mean_kappa": float(np.mean(kappas)),
                    "min_kappa": float(np.min(kappas)),
                    "max_kappa": float(np.max(kappas)),
                }
            )
    return summary, details


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_matrix(path: Path, rows: list[dict[str, object]], title: str) -> None:
    labels = family_names()
    mat = np.full((len(labels), len(labels)), np.nan)
    wins = {}
    for row in rows:
        i = labels.index(str(row["train_family"]))
        j = labels.index(str(row["test_family"]))
        mat[i, j] = float(row["delta_pct"])
        wins[(i, j)] = f"{int(row['wins'])}/{int(row['tests'])}"

    fig, ax = plt.subplots(figsize=(7.5, 6.1))
    norm = TwoSlopeNorm(vmin=-60, vcenter=0, vmax=150)
    im = ax.imshow(mat, cmap="RdYlGn_r", norm=norm)
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel("Test schedule family")
    ax.set_ylabel("Calibration schedule family")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=22, ha="right", rotation_mode="anchor")
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = mat[i, j]
            text_color = "white" if value > 55 else "black"
            ax.text(j, i, f"{value:+.1f}%\n{wins[(i, j)]}", ha="center", va="center",
                    fontsize=10.5, fontweight="bold", color=text_color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("MAE change vs MPL (negative is better)")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def plot_schedule_and_feature(path: Path, feats) -> None:
    selected = [
        ("Cosine decay", "cosine_72000.csv"),
        ("WSD sharp", "wsd_20000_24000.csv"),
        ("WSD linear", "wsdld_20000_24000.csv"),
        ("WSD-con step", "wsdcon_9.csv"),
        ("Constant sanity", "constant_72000.csv"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.0), sharex=False)
    for label, curve_name in selected:
        curve = load_curve("100", curve_name)
        axes[0].plot(curve.step, curve.lrs[curve.step] / PEAK_LR, label=label, lw=1.8)
        axes[1].plot(curve.step, feats[("100", curve_name, 10.0)], label=label, lw=1.8)
    axes[0].set_title("LR schedule shapes at 100M")
    axes[0].set_ylabel("LR / peak LR")
    axes[1].set_title("DropRelaxS feature shape (lambda=10)")
    axes[1].set_ylabel("feature")
    axes[1].set_xlabel("training step")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def plot_kappas(path: Path, feats) -> list[dict[str, object]]:
    rows = []
    width = 0.2
    x = np.arange(len(FAMILIES))
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for si, scale in enumerate(SCALES):
        vals = []
        for train_label, train_curves in FAMILIES:
            kappa = fit_kappa_per_scale(scale, train_curves, 10.0, feats, signed=False)
            vals.append(kappa)
            rows.append({"scale": scale, "train_family": train_label, "lambda": 10.0, "kappa": kappa})
        ax.bar(x + (si - 1) * width, vals, width=width, label=f"{scale}M")
    ax.set_xticks(x, [label for label, _ in FAMILIES], rotation=18, ha="right")
    ax.set_ylabel("fitted kappa")
    ax.set_title("Amplitude learned from each schedule family (fixed lambda=10)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return rows


def representative_target(test_family: str) -> str:
    if test_family == "WSD-con step":
        return "wsdcon_9.csv"
    return dict(FAMILIES)[test_family][0]


def plot_representative_curves(path: Path, feats) -> list[dict[str, object]]:
    scenarios = [
        ("WSD-con step", "WSD sharp", "probe -> target: conservative success"),
        ("WSD sharp", "WSD-con step", "target -> probe: over-correction failure"),
        ("Cosine decay", "WSD sharp", "smooth -> sharp: fails"),
        ("WSD sharp", "WSD linear", "sharp -> linear WSD: strong transfer"),
        ("WSD-con step", "Cosine decay", "probe -> cosine: almost no effect"),
    ]
    fig, axes = plt.subplots(len(scenarios), 2, figsize=(10.5, 13.0))
    details = []
    family_to_curves = dict(FAMILIES)
    for row_idx, (train_family, test_family, label) in enumerate(scenarios):
        train_curves = family_to_curves[train_family]
        test_curve = representative_target(test_family)
        scale = "100"
        lam = 10.0
        kappa = fit_kappa_per_scale(scale, train_curves, lam, feats, signed=False)
        curve = load_curve(scale, test_curve)
        base, pred, feature = predict_curve(scale, test_curve, lam, kappa, feats)
        base_mae = metrics(curve.loss, base)["mae"]
        corr_mae = metrics(curve.loss, pred)["mae"]
        delta = 100.0 * (corr_mae / base_mae - 1.0)
        details.append(
            {
                "scale": scale,
                "scenario": label,
                "train_family": train_family,
                "test_curve": test_curve,
                "lambda": lam,
                "kappa": kappa,
                "base_mae": base_mae,
                "corr_mae": corr_mae,
                "delta_pct": delta,
            }
        )

        ax0, ax1 = axes[row_idx]
        ax0.plot(curve.step, curve.loss, color="black", lw=1.4, label="actual")
        ax0.plot(curve.step, base, color="#777777", lw=1.2, label="MPL")
        ax0.plot(curve.step, pred, color="#d62728", lw=1.2, label="MPL + correction")
        ax0.set_title(f"{label}\n{train_family} -> {test_curve.replace('.csv', '')}, delta={delta:+.1f}%")
        ax0.set_ylabel("loss")
        ax0.grid(alpha=0.22)

        residual = curve.loss - base
        ax1.plot(curve.step, residual, color="black", lw=1.2, label="actual - MPL")
        ax1.plot(curve.step, kappa * feature, color="#d62728", lw=1.2, label="kappa * feature")
        ax1.axhline(0.0, color="#999999", lw=0.8)
        ax1.set_title(f"residual fit: kappa={kappa:.4g}")
        ax1.grid(alpha=0.22)
        if row_idx == 0:
            ax0.legend(fontsize=8)
            ax1.legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("training step")
    fig.tight_layout(h_pad=2.0)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return details


def plot_lambda_sweep(path: Path, feats) -> list[dict[str, object]]:
    rows = []
    pairs = [
        ("WSD-con step", "WSD sharp"),
        ("WSD sharp", "WSD-con step"),
        ("Cosine decay", "WSD sharp"),
        ("WSD sharp", "WSD linear"),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for train_family, test_family in pairs:
        train_curves = dict(FAMILIES)[train_family]
        test_curves = dict(FAMILIES)[test_family]
        deltas = []
        for lam in LAM_GRID:
            cell = []
            for scale in SCALES:
                kappa = fit_kappa_per_scale(scale, train_curves, float(lam), feats, signed=False)
                for score in score_curves(scale, test_curves, float(lam), kappa, feats):
                    cell.append(score)
            base = np.mean([float(r["base_mae"]) for r in cell])
            corr = np.mean([float(r["corr_mae"]) for r in cell])
            delta = 100.0 * (corr / base - 1.0)
            deltas.append(delta)
            rows.append(
                {
                    "train_family": train_family,
                    "test_family": test_family,
                    "lambda": lam,
                    "delta_pct": delta,
                    "wins": sum(int(r["win"]) for r in cell),
                    "tests": len(cell),
                }
            )
        ax.plot(LAM_GRID, deltas, marker="o", lw=1.8, label=f"{train_family} -> {test_family}")
    ax.axhline(0.0, color="#777777", lw=0.9)
    ax.axvline(10.0, color="#777777", lw=0.9, ls="--")
    ax.set_xscale("log")
    ax.set_xticks(LAM_GRID, [str(x) for x in LAM_GRID])
    ax.set_xlabel("lambda")
    ax.set_ylabel("MAE change vs MPL (%)")
    ax.set_title("Transfer sensitivity to lambda")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return rows


def write_report(
    matrix_rows_by_mode: dict[str, list[dict[str, object]]],
    kappa_rows: list[dict[str, object]],
    scenario_rows: list[dict[str, object]],
) -> None:
    primary = matrix_rows_by_mode["fixed10_per_scale_nonnegative"]

    def find(train: str, test: str) -> dict[str, object]:
        for row in primary:
            if row["train_family"] == train and row["test_family"] == test:
                return row
        raise KeyError((train, test))

    lines = [
        "# Current-Law Behavior Report\n\n",
        "Formula under audit: `prediction = cosine-fit MPL + kappa * DropRelaxS_lambda`.\n\n",
        "This report is intentionally diagnostic. It does not try to make the method look universal; "
        "it shows where the correction transfers, where it over-corrects, and which fitting choices change the conclusion.\n\n",
        "## Schedule and Feature Geometry\n\n",
        "![LR schedules and DropRelaxS features](figs/schedules_and_features.png)\n\n",
        "- Constant LR has essentially zero positive-drop feature after warmup, so the correction cannot do anything there.\n",
        "- Cosine produces many small drops spread over a long horizon; WSD sharp creates a late concentrated cooldown; WSD-con creates a discrete drop followed by a long constant-tail relaxation probe.\n\n",
        "## Main Matrix: fixed lambda=10, nonnegative per-scale kappa\n\n",
        "![primary matrix](figs/matrix_fixed10_per_scale_nonnegative.png)\n\n",
        "| key transfer | MAE change | wins | reading |\n",
        "|---|---:|---:|---|\n",
    ]
    key_rows = [
        ("WSD-con step", "WSD sharp", "clean no-target-WSD probe; stable but conservative"),
        ("WSD sharp", "WSD-con step", "over-corrects probe tails"),
        ("Cosine decay", "WSD sharp", "smooth-decay calibration does not transfer to sharp cooldown"),
        ("WSD sharp", "WSD linear", "strong transfer inside WSD-like cooldowns"),
        ("WSD-con step", "Cosine decay", "almost no useful correction needed"),
    ]
    for train, test, reading in key_rows:
        row = find(train, test)
        lines.append(
            f"| `{train} -> {test}` | {float(row['delta_pct']):+.1f}% | "
            f"{int(row['wins'])}/{int(row['tests'])} | {reading} |\n"
        )

    lines += [
        "\n## Alternative Fitting Plans\n\n",
        "These are not new formulas. They only change how `lambda` and `kappa` are fitted.\n\n",
        "### Lambda selected on the train family\n\n",
        "![selected lambda matrix](figs/matrix_selected_lambda_per_scale_nonnegative.png)\n\n",
        "Selecting `lambda` on the training family can improve the diagonal cells, but it does not fix bad cross-family transfers. "
        "The main failure is amplitude/shape mismatch, not merely a poor fixed value of `lambda`.\n\n",
        "### Shared kappa across scales\n\n",
        "![shared kappa matrix](figs/matrix_fixed10_shared_nonnegative.png)\n\n",
        "A single shared `kappa` is a stricter amplitude assumption. It usually weakens the result because the residual amplitude is scale-dependent.\n\n",
        "### Signed kappa\n\n",
        "![signed kappa matrix](figs/matrix_fixed10_per_scale_signed.png)\n\n",
        "Allowing negative `kappa` is useful as a diagnostic, but it violates the positive-lag interpretation. The defensible theory version keeps `kappa >= 0`.\n\n",
        "## Learned Amplitudes\n\n",
        "![kappa by train family](figs/kappa_by_train_family.png)\n\n",
        "| scale | train family | kappa |\n",
        "|---:|---|---:|\n",
    ]
    for row in kappa_rows:
        lines.append(f"| {row['scale']}M | {row['train_family']} | {float(row['kappa']):.5f} |\n")

    lines += [
        "\nThe important pattern is that cosine calibration learns a very large amplitude for a smooth distributed feature, "
        "which then badly over-corrects sharp/step schedules. WSD-con learns a smaller amplitude; it under-fits sharp WSD but transfers in the right direction.\n\n",
        "## Representative Curve Fits at 100M\n\n",
        "![representative curve fits](figs/representative_curve_fits_100M.png)\n\n",
        "| scenario | train family | test curve | kappa | MAE change |\n",
        "|---|---|---|---:|---:|\n",
    ]
    for row in scenario_rows:
        lines.append(
            f"| {row['scenario']} | {row['train_family']} | `{str(row['test_curve']).replace('.csv', '')}` | "
            f"{float(row['kappa']):.5f} | {float(row['delta_pct']):+.1f}% |\n"
        )

    lines += [
        "\n## Lambda Sensitivity\n\n",
        "![lambda sweep](figs/lambda_sweep.png)\n\n",
        "`lambda=10` is a reasonable theory-first operating point, but the plots show the more important fact: "
        "some train/test families remain bad across the lambda range. That is evidence of schedule-family mismatch, not a tuning accident.\n\n",
        "## Bottom Line\n\n",
        "1. The formula has a real, interpretable success case: `WSD-con step -> WSD sharp` gives stable conservative improvement without using a full WSD target curve.\n",
        "2. The formula is not universal: `cosine -> WSD` and `WSD sharp -> WSD-con` are clear failures.\n",
        "3. The deciding factor is not simply more calibration data. It is whether the calibration schedule excites the same relaxation shape and amplitude as the target schedule.\n",
        "4. For the paper, the honest claim should stay narrow: a low-calibration sharp-cooldown correction, not a general residual smoother.\n",
    ]
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    feats = feature_cache()

    modes = [
        "fixed10_per_scale_nonnegative",
        "selected_lambda_per_scale_nonnegative",
        "fixed10_shared_nonnegative",
        "fixed10_per_scale_signed",
    ]
    all_summary, all_details = [], []
    matrix_rows_by_mode = {}
    for mode in modes:
        summary, details = run_matrix(mode, feats)
        matrix_rows_by_mode[mode] = summary
        all_summary.extend(summary)
        all_details.extend(details)
        plot_matrix(FIG_DIR / f"matrix_{mode}.png", summary, mode.replace("_", " "))

    write_csv(OUT_DIR / "matrix_summary.csv", all_summary)
    write_csv(OUT_DIR / "matrix_details.csv", all_details)

    plot_schedule_and_feature(FIG_DIR / "schedules_and_features.png", feats)
    kappa_rows = plot_kappas(FIG_DIR / "kappa_by_train_family.png", feats)
    write_csv(OUT_DIR / "kappa_by_train_family.csv", kappa_rows)

    scenario_rows = plot_representative_curves(FIG_DIR / "representative_curve_fits_100M.png", feats)
    write_csv(OUT_DIR / "representative_curve_fits_100M.csv", scenario_rows)

    lambda_rows = plot_lambda_sweep(FIG_DIR / "lambda_sweep.png", feats)
    write_csv(OUT_DIR / "lambda_sweep.csv", lambda_rows)

    write_report(matrix_rows_by_mode, kappa_rows, scenario_rows)
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    print(f"wrote {OUT_DIR / 'matrix_summary.csv'}")
    print(f"wrote {FIG_DIR}")

    primary = matrix_rows_by_mode["fixed10_per_scale_nonnegative"]
    for row in primary:
        print(
            f"{row['train_family']:14s} -> {row['test_family']:14s} "
            f"delta={float(row['delta_pct']):+7.1f}% "
            f"wins={int(row['wins'])}/{int(row['tests'])}"
        )


if __name__ == "__main__":
    main()
