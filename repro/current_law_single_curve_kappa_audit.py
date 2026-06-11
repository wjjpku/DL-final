#!/usr/bin/env python3
"""Single-curve audit for robust DropRelaxS kappa estimation.

The family-level audit can hide failures because several calibration curves are
pooled.  This script tests the stricter question:

    If I am given exactly one calibration curve, can I estimate a non-exploding
    kappa and transfer it to all other schedule families?

It compares raw least-squares kappa to the recommended theory-gated projected
estimator from current_law_kappa_estimators.py:

    if the DropRelaxS response is not identifiable -> kappa = 0
    else kappa = min(raw nonnegative LS kappa, 0.03)
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


OUT_DIR = ROOT / "results" / "current_law_single_curve_kappa_audit"
FIG_DIR = OUT_DIR / "figs"
LAMBDA = 10.0
KAPPA_CAP = 0.03

CURVES = [
    ("cosine_72000.csv", "Cosine"),
    ("wsd_20000_24000.csv", "WSD sharp"),
    ("wsdld_20000_24000.csv", "WSD linear"),
    ("wsdcon_3.csv", "WSD-con 3e-5"),
    ("wsdcon_9.csv", "WSD-con 9e-5"),
    ("wsdcon_18.csv", "WSD-con 18e-5"),
]


def feature_cache() -> dict[tuple[str, str], np.ndarray]:
    return {
        (scale, curve): stime_feature(load_curve(scale, curve), LAMBDA)
        for scale in SCALES
        for curve, _ in CURVES
    }


def drop_stats(scale: str, curve_name: str, feats) -> dict[str, float]:
    curve = load_curve(scale, curve_name)
    eta = curve.lrs.astype(np.float64)
    drop = np.zeros_like(eta)
    drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0) / PEAK_LR
    f = feats[(scale, curve_name)]
    total_drop = float(np.sum(drop))
    drop_l2 = float(np.dot(drop, drop))
    eff_steps = float(total_drop * total_drop / drop_l2) if drop_l2 > 1e-18 else float("inf")
    return {
        "total_drop": total_drop,
        "max_drop": float(np.max(drop)),
        "drop_eff_steps": eff_steps,
        "feature_max": float(np.max(f)),
        "feature_mean": float(np.mean(f)),
    }


def identifiable(stats: dict[str, float]) -> bool:
    return (
        stats["total_drop"] >= 0.05
        and stats["feature_max"] >= 0.05
        and stats["drop_eff_steps"] <= 6000
    )


def raw_kappa(scale: str, curve_name: str, feats) -> float:
    curve = load_curve(scale, curve_name)
    x = feats[(scale, curve_name)]
    y = curve.loss - mpl_predict(MPL_PRECOMPUTED_INIT[scale], curve)
    if float(np.dot(x, x)) <= 1e-18:
        return 0.0
    return max(0.0, float(fit_origin(x, y)[0]))


def robust_kappa(scale: str, curve_name: str, feats) -> tuple[float, dict[str, float]]:
    stats = drop_stats(scale, curve_name, feats)
    raw = raw_kappa(scale, curve_name, feats)
    if not identifiable(stats):
        return 0.0, {**stats, "raw_kappa": raw, "identifiable": 0.0}
    return min(raw, KAPPA_CAP), {**stats, "raw_kappa": raw, "identifiable": 1.0}


def score(scale: str, test_curve: str, kappa: float, feats) -> dict[str, object]:
    curve = load_curve(scale, test_curve)
    base = mpl_predict(MPL_PRECOMPUTED_INIT[scale], curve)
    pred = base + kappa * feats[(scale, test_curve)]
    base_mae = metrics(curve.loss, base)["mae"]
    corr_mae = metrics(curve.loss, pred)["mae"]
    return {
        "scale": scale,
        "test_curve": test_curve,
        "base_mae": base_mae,
        "corr_mae": corr_mae,
        "delta_pct": 100.0 * (corr_mae / base_mae - 1.0),
        "win": int(corr_mae < base_mae),
    }


def run() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    feats = feature_cache()
    rows, kappa_rows = [], []
    for estimator in ["raw_ls", "robust_gate_cap"]:
        for train_curve, train_label in CURVES:
            for scale in SCALES:
                if estimator == "raw_ls":
                    kappa = raw_kappa(scale, train_curve, feats)
                    stats = drop_stats(scale, train_curve, feats)
                    stats = {**stats, "raw_kappa": kappa, "identifiable": float(identifiable(stats))}
                else:
                    kappa, stats = robust_kappa(scale, train_curve, feats)
                kappa_rows.append(
                    {
                        "estimator": estimator,
                        "scale": scale,
                        "train_curve": train_curve,
                        "train_label": train_label,
                        "kappa": kappa,
                        **stats,
                    }
                )
                for test_curve, test_label in CURVES:
                    scored = score(scale, test_curve, kappa, feats)
                    rows.append(
                        {
                            "estimator": estimator,
                            "train_curve": train_curve,
                            "train_label": train_label,
                            "test_label": test_label,
                            "kappa": kappa,
                            **scored,
                        }
                    )
    return rows, kappa_rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary = []
    for estimator in ["raw_ls", "robust_gate_cap"]:
        for train_curve, train_label in CURVES:
            for test_curve, test_label in CURVES:
                subset = [
                    r for r in rows
                    if r["estimator"] == estimator
                    and r["train_curve"] == train_curve
                    and r["test_curve"] == test_curve
                ]
                base = np.array([float(r["base_mae"]) for r in subset])
                corr = np.array([float(r["corr_mae"]) for r in subset])
                summary.append(
                    {
                        "estimator": estimator,
                        "train_curve": train_curve,
                        "train_label": train_label,
                        "test_curve": test_curve,
                        "test_label": test_label,
                        "base_mae": float(base.mean()),
                        "corr_mae": float(corr.mean()),
                        "delta_pct": 100.0 * (float(corr.mean()) / float(base.mean()) - 1.0),
                        "wins": sum(int(r["win"]) for r in subset),
                        "tests": len(subset),
                        "mean_kappa": float(np.mean([float(r["kappa"]) for r in subset])),
                        "max_kappa": float(np.max([float(r["kappa"]) for r in subset])),
                    }
                )
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_matrix(path: Path, rows: list[dict[str, object]], estimator: str) -> None:
    labels = [label for _, label in CURVES]
    mat = np.zeros((len(labels), len(labels)))
    wins = {}
    for r in [x for x in rows if x["estimator"] == estimator]:
        i = labels.index(str(r["train_label"]))
        j = labels.index(str(r["test_label"]))
        mat[i, j] = float(r["delta_pct"])
        wins[(i, j)] = f"{int(r['wins'])}/{int(r['tests'])}"
    fig, ax = plt.subplots(figsize=(9.0, 7.0))
    norm = TwoSlopeNorm(vmin=-60, vcenter=0, vmax=150)
    im = ax.imshow(mat, cmap="RdYlGn_r", norm=norm)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=24, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("Test curve")
    ax.set_ylabel("Single calibration curve")
    ax.set_title(f"Single-curve kappa audit: {estimator}")
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = mat[i, j]
            color = "white" if value > 55 else "black"
            ax.text(j, i, f"{value:+.1f}%\n{wins[(i, j)]}", ha="center", va="center",
                    fontsize=8.8, fontweight="bold", color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("MAE change vs MPL (negative is better)")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def plot_kappas(path: Path, kappa_rows: list[dict[str, object]]) -> None:
    labels = [label for _, label in CURVES]
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    x = np.arange(len(labels))
    width = 0.23
    for si, scale in enumerate(SCALES):
        vals = []
        colors = []
        for curve, _ in CURVES:
            row = next(
                r for r in kappa_rows
                if r["estimator"] == "robust_gate_cap" and r["scale"] == scale and r["train_curve"] == curve
            )
            vals.append(float(row["kappa"]))
            colors.append("#2ca02c" if float(row["identifiable"]) else "#bbbbbb")
        ax.bar(x + (si - 1) * width, vals, width=width, label=f"{scale}M")
    ax.axhline(KAPPA_CAP, color="#d62728", ls="--", lw=1.0, label=f"cap={KAPPA_CAP}")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("robust kappa")
    ax.set_title("Single-curve robust kappas; gated curves return zero")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def write_report(summary: list[dict[str, object]], kappa_rows: list[dict[str, object]]) -> None:
    def get(estimator: str, train: str, test: str) -> dict[str, object]:
        return next(
            r for r in summary
            if r["estimator"] == estimator and r["train_curve"] == train and r["test_curve"] == test
        )

    robust = [r for r in summary if r["estimator"] == "robust_gate_cap"]
    raw = [r for r in summary if r["estimator"] == "raw_ls"]
    robust_offdiag = [float(r["delta_pct"]) for r in robust if r["train_curve"] != r["test_curve"]]
    raw_offdiag = [float(r["delta_pct"]) for r in raw if r["train_curve"] != r["test_curve"]]

    lines = [
        "# Single-Curve Kappa Audit\n\n",
        "This audit tests the stricter setting where exactly one curve is available for calibration. "
        "The recommended estimator is the theory-gated projected estimator:\n\n",
        "```text\n",
        "if total_positive_drop < 0.05 or feature_max < 0.05 or drop_effective_steps > 6000:\n",
        "    kappa = 0\n",
        "else:\n",
        f"    kappa = min(raw_nonnegative_ls_kappa, {KAPPA_CAP})\n",
        "```\n\n",
        "## Raw LS vs robust single-curve matrices\n\n",
        "Raw LS:\n\n![raw single curve matrix](figs/single_curve_matrix_raw_ls.png)\n\n",
        "Robust gate+cap:\n\n![robust single curve matrix](figs/single_curve_matrix_robust_gate_cap.png)\n\n",
        "## Learned single-curve kappas\n\n",
        "![robust kappas](figs/single_curve_robust_kappas.png)\n\n",
        "## Summary\n\n",
        f"- Raw LS worst off-diagonal: `{max(raw_offdiag):+.1f}%`.\n",
        f"- Robust worst off-diagonal: `{max(robust_offdiag):+.1f}%`.\n",
        f"- Robust median off-diagonal: `{np.median(robust_offdiag):+.1f}%`.\n",
        "- Cosine is correctly declared non-identifiable and returns `kappa=0` at all scales.\n",
        "\n## Key cells\n\n",
        "| train curve | test curve | raw LS | robust | robust wins |\n",
        "|---|---|---:|---:|---:|\n",
    ]
    pairs = [
        ("cosine_72000.csv", "wsd_20000_24000.csv"),
        ("wsdcon_3.csv", "wsd_20000_24000.csv"),
        ("wsdcon_9.csv", "wsd_20000_24000.csv"),
        ("wsdcon_18.csv", "wsd_20000_24000.csv"),
        ("wsd_20000_24000.csv", "wsdcon_9.csv"),
        ("wsd_20000_24000.csv", "wsdld_20000_24000.csv"),
    ]
    for train, test in pairs:
        raw_r = get("raw_ls", train, test)
        robust_r = get("robust_gate_cap", train, test)
        lines.append(
            f"| `{train.replace('.csv', '')}` | `{test.replace('.csv', '')}` | "
            f"{float(raw_r['delta_pct']):+.1f}% | {float(robust_r['delta_pct']):+.1f}% | "
            f"{int(robust_r['wins'])}/{int(robust_r['tests'])} |\n"
        )

    lines += [
        "\n## Interpretation\n\n",
        "The robust estimator satisfies the single-curve safety requirement much better than raw LS. "
        "It does not force a correction from cosine, because cosine does not identify the non-adiabatic response. "
        "Single WSD-con probes still transfer to WSD sharp with a modest gain, and single WSD sharp/linear curves remain useful inside the WSD-like family. "
        "The price is intentional conservatism: capped WSD-family kappas give less diagonal improvement than raw LS, but avoid large cross-family over-correction.\n",
    ]
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    details, kappa_rows = run()
    summary = summarize(details)
    write_csv(OUT_DIR / "details.csv", details)
    write_csv(OUT_DIR / "summary.csv", summary)
    write_csv(OUT_DIR / "kappa_diagnostics.csv", kappa_rows)
    plot_matrix(FIG_DIR / "single_curve_matrix_raw_ls.png", summary, "raw_ls")
    plot_matrix(FIG_DIR / "single_curve_matrix_robust_gate_cap.png", summary, "robust_gate_cap")
    plot_kappas(FIG_DIR / "single_curve_robust_kappas.png", kappa_rows)
    write_report(summary, kappa_rows)
    robust = [r for r in summary if r["estimator"] == "robust_gate_cap" and r["train_curve"] != r["test_curve"]]
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    print(f"robust worst offdiag = {max(float(r['delta_pct']) for r in robust):+.1f}%")
    for train, test in [
        ("cosine_72000.csv", "wsd_20000_24000.csv"),
        ("wsdcon_3.csv", "wsd_20000_24000.csv"),
        ("wsdcon_9.csv", "wsd_20000_24000.csv"),
        ("wsdcon_18.csv", "wsd_20000_24000.csv"),
    ]:
        r = next(x for x in robust if x["train_curve"] == train and x["test_curve"] == test)
        print(f"{train.replace('.csv',''):22s} -> {test.replace('.csv',''):22s} {float(r['delta_pct']):+6.1f}%")


if __name__ == "__main__":
    main()
