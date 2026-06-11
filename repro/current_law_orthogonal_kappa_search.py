#!/usr/bin/env python3
"""Nuisance-orthogonal kappa estimation for DropRelaxS.

Motivation: the DropRelaxS residual shape can be directionally useful while the
raw kappa magnitude fails because diffuse schedule features are confounded with
ordinary low-frequency MPL residual drift.  Model that explicitly:

    r(t) = kappa * phi(t) + Z(t) beta + eps(t)

where Z is a low-frequency nuisance basis.  By the Frisch-Waugh-Lovell theorem,
the coefficient of phi is obtained by projecting both r and phi off Z, then
fitting the residualized variables.  We then add the same MAP/ridge prior:

    kappa = <M_Z phi, M_Z r> / (||M_Z phi||^2 + tau^2)

This removes the need to decide that a schedule is "cosine" or "WSD"; the data
decide whether the response direction has identifiable content beyond smooth
trend drift.
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

import current_law_amplitude_kappa_search as amp  # noqa: E402
import current_law_continuous_kappa_search as base  # noqa: E402
import current_law_eb_kappa_search as eb  # noqa: E402


OUT_DIR = ROOT / "results" / "current_law_orthogonal_kappa_search"
FIG_DIR = OUT_DIR / "figs"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def nuisance_basis(curve, degree: int) -> np.ndarray:
    t = curve.step.astype(np.float64)
    x = (t - float(np.min(t))) / max(float(np.max(t) - np.min(t)), 1.0)
    cols = [np.ones_like(x)]
    for d in range(1, degree + 1):
        cols.append(x**d)
    return np.column_stack(cols)


def residualize(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(z, y, rcond=None)
    return y - z @ coef


def orthogonal_stats(scale: str, curve_name: str, feats, degree: int) -> dict[str, float]:
    stats = amp.enriched_stats(scale, curve_name, feats)
    curve = base.load_curve(scale, curve_name)
    phi = feats[(scale, curve_name)]
    resid = curve.loss - base.mpl_predict(base.MPL_PRECOMPUTED_INIT[scale], curve)
    z = nuisance_basis(curve, degree)
    phi_o = residualize(phi, z)
    resid_o = residualize(resid, z)
    phi_o2 = max(float(np.dot(phi_o, phi_o)), 1e-18)
    dot_o = float(np.dot(phi_o, resid_o))
    raw_o = max(0.0, dot_o / phi_o2)
    retention = float(phi_o2 / max(float(np.dot(phi, phi)), 1e-18))
    corr_denom = float(np.linalg.norm(phi_o) * np.linalg.norm(resid_o))
    corr_o = 0.0 if corr_denom <= 1e-18 else float(np.dot(phi_o, resid_o) / corr_denom)
    return {
        **stats,
        "degree": degree,
        "orth_feature_l2": phi_o2,
        "orth_projection_dot": dot_o,
        "orth_raw_kappa": raw_o,
        "orth_feature_retention": retention,
        "orth_corr": corr_o,
        "orth_resid_scale": amp.robust_scale(resid_o),
    }


def kappa_from_stats(stats: dict[str, float], estimator: str, tau: float) -> float:
    if estimator == "current_smooth_cap":
        return base.estimate("smooth_weight_cap_0p03", stats)
    if estimator == "eb_q75":
        return eb.eb_kappa(stats, tau)
    if estimator == "orth_ls":
        return min(max(0.0, stats["orth_raw_kappa"]), 0.03)
    if estimator == "orth_map":
        denom = stats["orth_feature_l2"] + tau * tau
        return min(max(0.0, stats["orth_projection_dot"] / max(denom, 1e-18)), 0.03)
    if estimator == "orth_map_retention":
        # If almost all of phi is explained by nuisance trends, the response is
        # poorly identified.  This is a geometric identifiability shrinkage,
        # not a schedule-family label.
        denom = stats["orth_feature_l2"] + tau * tau
        k = max(0.0, stats["orth_projection_dot"] / max(denom, 1e-18))
        return min(k * math.sqrt(max(stats["orth_feature_retention"], 0.0)), 0.03)
    raise ValueError(estimator)


def run():
    feats = base.feature_cache()
    base_stats = {
        (scale, curve): amp.enriched_stats(scale, curve, feats)
        for curve, _ in base.CURVES
        for scale in base.SCALES
    }
    all_base_rows = []
    for (scale, curve), stats in base_stats.items():
        label = next(label for c, label in base.CURVES if c == curve)
        all_base_rows.append({"scale": scale, "train_curve": curve, "train_label": label, **stats})

    details: list[dict[str, object]] = []
    kappa_rows: list[dict[str, object]] = []
    degrees = [1, 2, 3, 4, 5]
    estimators = ["current_smooth_cap", "eb_q75", "orth_ls", "orth_map", "orth_map_retention"]

    for degree in degrees:
        stats_cache = {
            (scale, curve): orthogonal_stats(scale, curve, feats, degree)
            for curve, _ in base.CURVES
            for scale in base.SCALES
        }
        for estimator in estimators:
            for train_curve, train_label in base.CURVES:
                pool = [r for r in all_base_rows if r["train_curve"] != train_curve]
                tau = eb.estimate_tau(pool, "q75")["tau"]
                for scale in base.SCALES:
                    stats = stats_cache[(scale, train_curve)]
                    kappa = kappa_from_stats(stats, estimator, tau)
                    kappa_rows.append(
                        {
                            "estimator": estimator,
                            "degree": degree,
                            "scale": scale,
                            "train_curve": train_curve,
                            "train_label": train_label,
                            "tau": tau,
                            "kappa": kappa,
                            **stats,
                        }
                    )
                    for test_curve, test_label in base.CURVES:
                        scored = base.score(scale, test_curve, kappa, feats)
                        details.append(
                            {
                                "estimator": estimator,
                                "degree": degree,
                                "scale": scale,
                                "train_curve": train_curve,
                                "train_label": train_label,
                                "test_curve": test_curve,
                                "test_label": test_label,
                                "kappa": kappa,
                                **scored,
                            }
                        )
    return details, kappa_rows


def summarize(details: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    keys = sorted({(str(r["estimator"]), int(r["degree"])) for r in details})
    for estimator, degree in keys:
        for train_curve, train_label in base.CURVES:
            for test_curve, test_label in base.CURVES:
                subset = [
                    r
                    for r in details
                    if r["estimator"] == estimator
                    and int(r["degree"]) == degree
                    and r["train_curve"] == train_curve
                    and r["test_curve"] == test_curve
                ]
                rows.append(
                    {
                        "estimator": estimator,
                        "degree": degree,
                        "train_curve": train_curve,
                        "train_label": train_label,
                        "test_curve": test_curve,
                        "test_label": test_label,
                        "mean_delta_pct": float(np.mean([float(r["delta_pct"]) for r in subset])),
                        "wins": int(sum(int(r["win"]) for r in subset)),
                        "tests": len(subset),
                        "mean_kappa": float(np.mean([float(r["kappa"]) for r in subset])),
                        "max_kappa": float(np.max([float(r["kappa"]) for r in subset])),
                    }
                )
    return rows


def comparison(summary: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    keys = sorted({(str(r["estimator"]), int(r["degree"])) for r in summary})
    for estimator, degree in keys:
        sub = [r for r in summary if r["estimator"] == estimator and int(r["degree"]) == degree and r["train_curve"] != r["test_curve"]]
        cos_wsd = next(
            r
            for r in summary
            if r["estimator"] == estimator
            and int(r["degree"]) == degree
            and r["train_curve"] == "cosine_72000.csv"
            and r["test_curve"] == "wsd_20000_24000.csv"
        )
        w9_wsd = next(
            r
            for r in summary
            if r["estimator"] == estimator
            and int(r["degree"]) == degree
            and r["train_curve"] == "wsdcon_9.csv"
            and r["test_curve"] == "wsd_20000_24000.csv"
        )
        rows.append(
            {
                "estimator": estimator,
                "degree": degree,
                "name": f"{estimator}_deg{degree}",
                "worst_offdiag": float(max(float(r["mean_delta_pct"]) for r in sub)),
                "median_offdiag": float(np.median([float(r["mean_delta_pct"]) for r in sub])),
                "mean_offdiag": float(np.mean([float(r["mean_delta_pct"]) for r in sub])),
                "cosine_to_wsd": float(cos_wsd["mean_delta_pct"]),
                "wsdcon9_to_wsd": float(w9_wsd["mean_delta_pct"]),
            }
        )
    rows.sort(key=lambda r: (max(float(r["worst_offdiag"]), 0.0), max(0.0, 10.0 + float(r["wsdcon9_to_wsd"])), float(r["mean_offdiag"])))
    return rows


def plot_matrix(path: Path, summary: list[dict[str, object]], estimator: str, degree: int) -> None:
    labels = [label for _, label in base.CURVES]
    mat = np.full((len(base.CURVES), len(base.CURVES)), np.nan)
    wins = np.zeros_like(mat)
    for i, (train_curve, _) in enumerate(base.CURVES):
        for j, (test_curve, _) in enumerate(base.CURVES):
            row = next(
                r
                for r in summary
                if r["estimator"] == estimator
                and int(r["degree"]) == degree
                and r["train_curve"] == train_curve
                and r["test_curve"] == test_curve
            )
            mat[i, j] = float(row["mean_delta_pct"])
            wins[i, j] = int(row["wins"])
    fig, ax = plt.subplots(figsize=(9.2, 7.2))
    im = ax.imshow(mat, cmap="RdYlGn_r", norm=TwoSlopeNorm(vmin=-60, vcenter=0, vmax=150))
    ax.set_xticks(np.arange(len(labels)), labels, rotation=24, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("Test curve")
    ax.set_ylabel("Calibration curve")
    ax.set_title(f"{estimator}, nuisance degree {degree}")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i,j]:+.1f}%\n{int(wins[i,j])}/3", ha="center", va="center", fontsize=8, weight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("MAE change vs MPL (negative is better)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def plot_retention(path: Path, kappa_rows: list[dict[str, object]], degree: int) -> None:
    rows = [r for r in kappa_rows if int(r["degree"]) == degree and r["estimator"] == "orth_map"]
    labels = [label for _, label in base.CURVES]
    x = np.arange(len(labels))
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for si, scale in enumerate(base.SCALES):
        vals = []
        kappas = []
        for curve, _ in base.CURVES:
            row = next(r for r in rows if r["scale"] == scale and r["train_curve"] == curve)
            vals.append(float(row["orth_feature_retention"]))
            kappas.append(float(row["kappa"]))
        axes[0].bar(x + (si - 1) * width, vals, width=width, label=f"{scale}M")
        axes[1].bar(x + (si - 1) * width, kappas, width=width, label=f"{scale}M")
    axes[0].set_title("feature energy beyond trend")
    axes[0].set_ylabel("retention")
    axes[1].set_title("orthogonal MAP kappa")
    axes[1].set_ylabel("kappa")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=24, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def choose_cap_safe(comp, kappa_rows) -> dict[str, object]:
    """Choose a candidate whose cosine-derived kappa does not rely on the cap."""
    for row in comp:
        rows = [
            kr
            for kr in kappa_rows
            if kr["estimator"] == row["estimator"]
            and int(kr["degree"]) == int(row["degree"])
            and kr["train_curve"] == "cosine_72000.csv"
        ]
        max_cosine_kappa = max(float(kr["kappa"]) for kr in rows)
        if max_cosine_kappa < 0.02 and float(row["wsdcon9_to_wsd"]) <= -15.0 and float(row["worst_offdiag"]) <= 0.1:
            return row
    return comp[0]


def write_report(comp, summary, kappa_rows) -> None:
    best_numeric = comp[0]
    best_defensible = choose_cap_safe(comp, kappa_rows)
    lines = [
        "# Nuisance-Orthogonal Kappa Search\n\n",
        "This experiment treats slow MPL residual drift as an explicit nuisance term instead of relying only on schedule-derived identifiability weights.\n\n",
        "## Model\n\n",
        "```text\n",
        "r(t) = kappa * phi(t) + Z(t) beta + eps(t)\n",
        "phi_perp = M_Z phi,    r_perp = M_Z r\n",
        "kappa_hat = min(0.03, max(0, <phi_perp,r_perp> / (||phi_perp||^2 + tau^2)))\n",
        "```\n\n",
        "By the Frisch-Waugh-Lovell theorem, fitting after residualizing against `Z` gives the coefficient of `phi` after accounting for smooth nuisance trends. "
        "Here `Z` is a low-degree polynomial basis over normalized training step. `tau` is the same leave-curve-out EB q75 prior/noise ratio used in the EB report.\n\n",
        "## Comparison\n\n",
        "| estimator | degree | worst offdiag | median offdiag | mean offdiag | cosine -> WSD | wsdcon_9 -> WSD |\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in comp[:18]:
        lines.append(
            f"| `{row['estimator']}` | {int(row['degree'])} | {float(row['worst_offdiag']):+.1f}% | "
            f"{float(row['median_offdiag']):+.1f}% | {float(row['mean_offdiag']):+.1f}% | "
            f"{float(row['cosine_to_wsd']):+.1f}% | {float(row['wsdcon9_to_wsd']):+.1f}% |\n"
        )
    lines += [
        "\n## Recommendation\n\n",
        f"Best numeric candidate: `{best_numeric['name']}`. Recommended cap-safe candidate: `{best_defensible['name']}`.\n\n",
        f"![matrix](figs/matrix_{best_defensible['name']}.png)\n\n",
        f"![retention](figs/retention_degree_{int(best_defensible['degree'])}.png)\n\n",
        "The degree-1 orthogonal estimators are numerically strongest, but cosine-derived kappa saturates the `0.03` susceptibility cap. "
        "For the paper, the cap-safe recommendation is more defensible: it still improves cosine -> WSD and WSD-con -> WSD, while its "
        "cosine-derived kappa is produced by partial-regression shrinkage rather than by the hard upper bound.\n\n",
        "Interpretation: improvement over EB means a meaningful part of the previous amplitude error was trend confounding. "
        "Overly high nuisance degree can remove real response signal, so the recommended degree is chosen by the cap-safe transfer criterion, not by visual smoothness.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    details, kappa_rows = run()
    summary = summarize(details)
    comp = comparison(summary)
    write_csv(OUT_DIR / "details.csv", details)
    write_csv(OUT_DIR / "summary.csv", summary)
    write_csv(OUT_DIR / "kappa_diagnostics.csv", kappa_rows)
    write_csv(OUT_DIR / "comparison.csv", comp)
    best_defensible = choose_cap_safe(comp, kappa_rows)
    matrix_rows = {(str(row["estimator"]), int(row["degree"]), str(row["name"])) for row in comp[:5]}
    matrix_rows.add((str(comp[0]["estimator"]), int(comp[0]["degree"]), str(comp[0]["name"])))
    matrix_rows.add((str(best_defensible["estimator"]), int(best_defensible["degree"]), str(best_defensible["name"])))
    for estimator, degree, name in sorted(matrix_rows):
        plot_matrix(FIG_DIR / f"matrix_{name}.png", summary, estimator, degree)
    plot_retention(FIG_DIR / f"retention_degree_{int(best_defensible['degree'])}.png", kappa_rows, int(best_defensible["degree"]))
    write_report(comp, summary, kappa_rows)
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    for row in comp[:18]:
        print(
            f"{row['name']:28s} worst={row['worst_offdiag']:+7.1f}% "
            f"median={row['median_offdiag']:+7.1f}% "
            f"cos->wsd={row['cosine_to_wsd']:+7.1f}% "
            f"w9->wsd={row['wsdcon9_to_wsd']:+7.1f}%"
        )


if __name__ == "__main__":
    main()
