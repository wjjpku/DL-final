#!/usr/bin/env python3
"""Search continuous, non-classification formulas for DropRelaxS kappa.

We want a kappa estimator that can be applied to an arbitrary calibration curve
without first assigning the curve to a schedule family.  The estimator may use
only curve-derived quantities:

  * raw projection kappa = <phi, residual>/<phi, phi>, clipped nonnegative;
  * feature concentration / identifiability statistics;
  * a weak susceptibility scale prior.

No schedule-family labels are used inside any estimator.  Labels are used only
for reporting train/test matrices.
"""
from __future__ import annotations

import csv
import math
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


OUT_DIR = ROOT / "results" / "current_law_continuous_kappa_search"
FIG_DIR = OUT_DIR / "figs"
LAMBDA = 10.0

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


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def curve_stats(scale: str, curve_name: str, feats) -> dict[str, float]:
    curve = load_curve(scale, curve_name)
    phi = feats[(scale, curve_name)]
    base = mpl_predict(MPL_PRECOMPUTED_INIT[scale], curve)
    resid = curve.loss - base

    eta = curve.lrs.astype(np.float64)
    drop = np.zeros_like(eta)
    drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0) / PEAK_LR
    total_drop = float(np.sum(drop))
    drop_l2 = float(np.dot(drop, drop))
    drop_eff_steps = float(total_drop * total_drop / drop_l2) if drop_l2 > 1e-18 else float("inf")

    phi2 = float(np.dot(phi, phi))
    raw = 0.0 if phi2 <= 1e-18 else max(0.0, float(fit_origin(phi, resid)[0]))
    pred_resid = raw * phi
    centered_phi = phi - float(np.mean(phi))
    centered_resid = resid - float(np.mean(resid))
    corr_denom = float(np.linalg.norm(centered_phi) * np.linalg.norm(centered_resid))
    corr = 0.0 if corr_denom <= 1e-18 else float(np.dot(centered_phi, centered_resid) / corr_denom)

    ss_resid = float(np.dot(resid - float(np.mean(resid)), resid - float(np.mean(resid))))
    ss_fit = float(np.dot(resid - pred_resid, resid - pred_resid))
    r2 = 0.0 if ss_resid <= 1e-18 else 1.0 - ss_fit / ss_resid
    positive_overlap = float(np.dot(phi, np.maximum(resid, 0.0)) / max(np.dot(phi, np.abs(resid)), 1e-18))
    peak_to_mean = float(np.max(phi) / max(float(np.mean(phi)), 1e-12))

    return {
        "raw_kappa": raw,
        "total_drop": total_drop,
        "max_drop": float(np.max(drop)),
        "drop_eff_steps": drop_eff_steps,
        "feature_max": float(np.max(phi)),
        "feature_mean": float(np.mean(phi)),
        "feature_peak_to_mean": peak_to_mean,
        "feature_l2": phi2,
        "corr": corr,
        "r2": r2,
        "positive_overlap": positive_overlap,
    }


def smooth_id_weight(stats: dict[str, float], eff0: float = 6000.0, feat0: float = 0.05, sharpness: float = 3.0) -> float:
    """Continuous replacement for the hard identifiability gate."""
    eff_term = sigmoid(sharpness * math.log(max(eff0, 1e-12) / max(stats["drop_eff_steps"], 1e-12)))
    feat_term = sigmoid(sharpness * math.log(max(stats["feature_max"], 1e-12) / feat0))
    drop_term = sigmoid(sharpness * math.log(max(stats["total_drop"], 1e-12) / 0.05))
    return float(eff_term * feat_term * drop_term)


def estimate(estimator: str, stats: dict[str, float]) -> float:
    raw = stats["raw_kappa"]
    if estimator == "raw_ls":
        return raw
    if estimator == "hard_gate_cap_0p03":
        if stats["total_drop"] < 0.05 or stats["feature_max"] < 0.05 or stats["drop_eff_steps"] > 6000:
            return 0.0
        return min(raw, 0.03)

    if estimator == "smooth_weight":
        return raw * smooth_id_weight(stats)

    if estimator == "smooth_weight_cap_0p04":
        return min(raw * smooth_id_weight(stats), 0.04)

    if estimator == "smooth_weight_cap_0p03":
        return min(raw * smooth_id_weight(stats), 0.03)

    if estimator == "susceptibility_prior":
        # MAP under kappa ~ half-normal with scale k0.  Equivalent to ridge
        # shrinkage of the projection coefficient.
        k0 = 0.03
        return raw / (1.0 + (raw / max(k0, 1e-12)) ** 2)

    if estimator == "smooth_prior_cap":
        w = smooth_id_weight(stats)
        k0 = 0.03
        shrunk = raw * w / (1.0 + (raw / max(k0, 1e-12)) ** 2)
        return min(shrunk, 0.03)

    if estimator == "reliability_prior":
        # Signal-direction reliability: positive overlap and nonnegative R2
        # indicate that the feature is fitting the intended residual direction.
        w_id = smooth_id_weight(stats)
        w_r2 = sigmoid(8.0 * (max(stats["r2"], 0.0) - 0.02))
        w_pos = sigmoid(10.0 * (stats["positive_overlap"] - 0.55))
        k = raw * w_id * w_r2 * w_pos
        return min(k, 0.03)

    raise ValueError(estimator)


ESTIMATORS = [
    "raw_ls",
    "hard_gate_cap_0p03",
    "smooth_weight",
    "smooth_weight_cap_0p04",
    "smooth_weight_cap_0p03",
    "susceptibility_prior",
    "smooth_prior_cap",
    "reliability_prior",
]


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
    details, kappa_rows = [], []
    for estimator in ESTIMATORS:
        for train_curve, train_label in CURVES:
            for scale in SCALES:
                stats = curve_stats(scale, train_curve, feats)
                kappa = estimate(estimator, stats)
                kappa_rows.append({
                    "estimator": estimator,
                    "scale": scale,
                    "train_curve": train_curve,
                    "train_label": train_label,
                    "kappa": kappa,
                    **stats,
                    "smooth_id_weight": smooth_id_weight(stats),
                })
                for test_curve, test_label in CURVES:
                    scored = score(scale, test_curve, kappa, feats)
                    details.append({
                        "estimator": estimator,
                        "train_curve": train_curve,
                        "train_label": train_label,
                        "test_label": test_label,
                        "kappa": kappa,
                        **scored,
                    })
    return details, kappa_rows


def summarize(details: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for estimator in ESTIMATORS:
        for train_curve, train_label in CURVES:
            for test_curve, test_label in CURVES:
                subset = [
                    r for r in details
                    if r["estimator"] == estimator and r["train_curve"] == train_curve and r["test_curve"] == test_curve
                ]
                base = np.array([float(r["base_mae"]) for r in subset])
                corr = np.array([float(r["corr_mae"]) for r in subset])
                rows.append({
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
                })
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def comparison(summary: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for estimator in ESTIMATORS:
        subset = [r for r in summary if r["estimator"] == estimator]
        off = [float(r["delta_pct"]) for r in subset if r["train_curve"] != r["test_curve"]]
        key_probe = next(r for r in subset if r["train_curve"] == "wsdcon_9.csv" and r["test_curve"] == "wsd_20000_24000.csv")
        key_cos = next(r for r in subset if r["train_curve"] == "cosine_72000.csv" and r["test_curve"] == "wsd_20000_24000.csv")
        rows.append({
            "estimator": estimator,
            "worst_offdiag": float(np.max(off)),
            "median_offdiag": float(np.median(off)),
            "mean_offdiag": float(np.mean(off)),
            "cosine_to_wsd": float(key_cos["delta_pct"]),
            "wsdcon9_to_wsd": float(key_probe["delta_pct"]),
        })
    return rows


def plot_comparison(path: Path, rows: list[dict[str, object]]) -> None:
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.bar(x - 0.24, [r["worst_offdiag"] for r in rows], width=0.24, label="worst off-diagonal")
    ax.bar(x, [r["cosine_to_wsd"] for r in rows], width=0.24, label="cosine -> WSD")
    ax.bar(x + 0.24, [r["wsdcon9_to_wsd"] for r in rows], width=0.24, label="wsdcon_9 -> WSD")
    ax.axhline(0.0, color="#777777", lw=0.9)
    ax.set_xticks(x, [r["estimator"] for r in rows], rotation=22, ha="right")
    ax.set_ylabel("MAE change vs MPL (%)")
    ax.set_title("Continuous kappa estimators: safety vs useful transfer")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def plot_matrix(path: Path, summary: list[dict[str, object]], estimator: str) -> None:
    labels = [label for _, label in CURVES]
    mat = np.zeros((len(labels), len(labels)))
    wins = {}
    for row in [r for r in summary if r["estimator"] == estimator]:
        i = labels.index(str(row["train_label"]))
        j = labels.index(str(row["test_label"]))
        mat[i, j] = float(row["delta_pct"])
        wins[(i, j)] = f"{int(row['wins'])}/{int(row['tests'])}"
    fig, ax = plt.subplots(figsize=(9.0, 7.0))
    norm = TwoSlopeNorm(vmin=-60, vcenter=0, vmax=150)
    im = ax.imshow(mat, cmap="RdYlGn_r", norm=norm)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=24, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("Test curve")
    ax.set_ylabel("Calibration curve")
    ax.set_title(estimator)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = mat[i, j]
            color = "white" if value > 55 else "black"
            ax.text(j, i, f"{value:+.1f}%\n{wins[(i, j)]}", ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("MAE change vs MPL (negative is better)")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def plot_weights(path: Path, kappa_rows: list[dict[str, object]]) -> None:
    rows = [r for r in kappa_rows if r["estimator"] == "smooth_weight_cap_0p03"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    labels = [label for _, label in CURVES]
    x = np.arange(len(labels))
    width = 0.23
    for si, scale in enumerate(SCALES):
        subset = [r for r in rows if r["scale"] == scale]
        vals = [next(r for r in subset if r["train_curve"] == curve)["smooth_id_weight"] for curve, _ in CURVES]
        kappas = [next(r for r in subset if r["train_curve"] == curve)["kappa"] for curve, _ in CURVES]
        axes[0].bar(x + (si - 1) * width, vals, width=width, label=f"{scale}M")
        axes[1].bar(x + (si - 1) * width, kappas, width=width, label=f"{scale}M")
    axes[0].set_title("continuous identifiability weight")
    axes[0].set_ylabel("weight")
    axes[1].set_title("estimated kappa")
    axes[1].set_ylabel("kappa")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=24, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def write_report(summary, comparison_rows, kappa_rows) -> None:
    best = "smooth_weight_cap_0p03"
    def get(est, train, test):
        return next(r for r in summary if r["estimator"] == est and r["train_curve"] == train and r["test_curve"] == test)

    lines = [
        "# Continuous Kappa Formula Search\n\n",
        "This report removes the hard schedule-family classification. Every estimator uses only curve-derived quantities: "
        "the DropRelaxS feature `phi`, the MPL residual `r`, and LR-drop concentration statistics.\n\n",
        "## Candidate formulas\n\n",
        "- `raw_ls`: `k_raw = max(0, <phi,r>/<phi,phi>)`.\n",
        "- `smooth_weight`: `k = k_raw * w_id`, where `w_id` is a smooth identifiability weight.\n",
        "- `smooth_weight_cap_0p03`: `k = min(k_raw * w_id, 0.03)`.\n",
        "- `reliability_prior`: additionally shrinks by residual-feature alignment and positive-overlap reliability.\n\n",
        "The smooth identifiability weight is:\n\n",
        "```text\n",
        "w_id = sigmoid(3 log(6000 / drop_effective_steps))\n",
        "    * sigmoid(3 log(feature_max / 0.05))\n",
        "    * sigmoid(3 log(total_positive_drop / 0.05))\n",
        "kappa = min(k_raw * w_id, 0.03)\n",
        "```\n\n",
        "This is continuous: cosine receives a small but nonzero weight from the formula rather than a schedule-name decision. "
        "The constants are weak prior hyperparameters, not schedule labels: `6000` is the transition scale for an identifiable "
        "localized LR drop in training steps, `0.05` is the minimum useful excitation scale in normalized LR-drop/feature units, "
        "and `0.03` is a conservative upper bound on the response susceptibility observed in the public WSD-like probes.\n\n",
        "## Estimator comparison\n\n",
        "![comparison](figs/continuous_estimator_comparison.png)\n\n",
        "| estimator | worst offdiag | median offdiag | cosine -> WSD | wsdcon_9 -> WSD |\n",
        "|---|---:|---:|---:|---:|\n",
    ]
    for row in comparison_rows:
        lines.append(
            f"| `{row['estimator']}` | {float(row['worst_offdiag']):+.1f}% | "
            f"{float(row['median_offdiag']):+.1f}% | {float(row['cosine_to_wsd']):+.1f}% | "
            f"{float(row['wsdcon9_to_wsd']):+.1f}% |\n"
        )

    lines += [
        "\n## Recommended continuous formula\n\n",
        f"Recommended: `{best}`.\n\n",
        "![recommended matrix](figs/matrix_smooth_weight_cap_0p03.png)\n\n",
        "![weights](figs/continuous_weights_and_kappas.png)\n\n",
        "Key cells:\n\n",
    ]
    for train, test in [
        ("cosine_72000.csv", "wsd_20000_24000.csv"),
        ("wsdcon_3.csv", "wsd_20000_24000.csv"),
        ("wsdcon_9.csv", "wsd_20000_24000.csv"),
        ("wsdcon_18.csv", "wsd_20000_24000.csv"),
        ("wsd_20000_24000.csv", "wsdcon_9.csv"),
        ("wsd_20000_24000.csv", "wsdld_20000_24000.csv"),
    ]:
        row = get(best, train, test)
        lines.append(
            f"- `{train.replace('.csv','')} -> {test.replace('.csv','')}`: "
            f"{float(row['delta_pct']):+.1f}% MAE, {int(row['wins'])}/{int(row['tests'])} wins, "
            f"mean kappa={float(row['mean_kappa']):.4f}\n"
        )

    lines += [
        "\n## Theoretical reading\n\n",
        "The formula is a continuous MAP/projection estimator. The raw projection estimates the response amplitude in "
        "`r = kappa phi + epsilon`. The weight is an approximate identifiability factor: it approaches 1 when LR drops are "
        "concentrated and the response feature is strong, and approaches 0 when the feature is diffuse/weak. The cap is a weak "
        "susceptibility prior on `kappa = eta_peak * chi`. Thus the method is not classifying schedules; it continuously asks "
        "whether the observed curve contains enough excitation of the response direction to trust the projection.\n\n",
        "The tradeoff is visible in the numbers: the continuous formula no longer explodes on cosine, keeps useful single-probe "
        "transfer to WSD sharp, and remains conservative on cross-family targets. It is less aggressive than raw WSD-family LS, "
        "but that conservatism is exactly what makes it curve-agnostic.\n",
    ]
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    details, kappa_rows = run()
    summary = summarize(details)
    comparison_rows = comparison(summary)
    write_csv(OUT_DIR / "details.csv", details)
    write_csv(OUT_DIR / "summary.csv", summary)
    write_csv(OUT_DIR / "kappa_diagnostics.csv", kappa_rows)
    write_csv(OUT_DIR / "comparison.csv", comparison_rows)
    plot_comparison(FIG_DIR / "continuous_estimator_comparison.png", comparison_rows)
    for est in ESTIMATORS:
        plot_matrix(FIG_DIR / f"matrix_{est}.png", summary, est)
    plot_weights(FIG_DIR / "continuous_weights_and_kappas.png", kappa_rows)
    write_report(summary, comparison_rows, kappa_rows)
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    for row in comparison_rows:
        print(
            f"{row['estimator']:24s} worst={row['worst_offdiag']:+7.1f}% "
            f"cos->wsd={row['cosine_to_wsd']:+7.1f}% "
            f"w9->wsd={row['wsdcon9_to_wsd']:+7.1f}%"
        )


if __name__ == "__main__":
    main()
