#!/usr/bin/env python3
"""Cap and retention sensitivity for the recommended kappa estimator.

The current recommended estimator is nuisance-orthogonal MAP with a degree-2
low-frequency nuisance basis and geometric feature-retention shrinkage:

    k = (<phi_perp,r_perp> / (||phi_perp||^2 + tau^2))_+
    kappa = min(cap, k * retention^rho)

This script checks whether the empirical gains depend critically on the hard
cap=0.03 or whether the MAP + retention structure already controls magnitude.
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
import current_law_orthogonal_kappa_search as orth  # noqa: E402


OUT_DIR = ROOT / "results" / "current_law_cap_sensitivity"
FIG_DIR = OUT_DIR / "figs"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def apply_cap(k: float, cap: float | None) -> float:
    if cap is None:
        return k
    return min(k, cap)


def estimate(stats: dict[str, float], tau: float, rho: float, cap: float | None) -> float:
    denom = stats["orth_feature_l2"] + tau * tau
    raw = max(0.0, stats["orth_projection_dot"] / max(denom, 1e-18))
    retention = max(float(stats["orth_feature_retention"]), 0.0)
    return apply_cap(raw * (retention ** rho), cap)


def run():
    feats = base.feature_cache()
    base_rows = []
    for curve, label in base.CURVES:
        for scale in base.SCALES:
            stats = amp.enriched_stats(scale, curve, feats)
            base_rows.append({"scale": scale, "train_curve": curve, "train_label": label, **stats})

    stats_cache = {
        (scale, curve): orth.orthogonal_stats(scale, curve, feats, 2)
        for curve, _ in base.CURVES
        for scale in base.SCALES
    }

    caps: list[tuple[str, float | None]] = [
        ("none", None),
        ("cap_0p05", 0.05),
        ("cap_0p04", 0.04),
        ("cap_0p03", 0.03),
        ("cap_0p02", 0.02),
    ]
    rhos = [0.0, 0.25, 0.5, 0.75, 1.0]

    details: list[dict[str, object]] = []
    kappa_rows: list[dict[str, object]] = []
    for rho in rhos:
        for cap_name, cap in caps:
            estimator = f"orth_deg2_rho_{str(rho).replace('.', 'p')}_{cap_name}"
            for train_curve, train_label in base.CURVES:
                pool = [r for r in base_rows if r["train_curve"] != train_curve]
                tau = eb.estimate_tau(pool, "q75")["tau"]
                for scale in base.SCALES:
                    stats = stats_cache[(scale, train_curve)]
                    kappa = estimate(stats, tau, rho, cap)
                    kappa_rows.append(
                        {
                            "estimator": estimator,
                            "rho": rho,
                            "cap_name": cap_name,
                            "cap": "" if cap is None else cap,
                            "scale": scale,
                            "train_curve": train_curve,
                            "train_label": train_label,
                            "tau": tau,
                            "kappa": kappa,
                            "cap_saturated": int(cap is not None and kappa >= cap - 1e-12),
                            **stats,
                        }
                    )
                    for test_curve, test_label in base.CURVES:
                        scored = base.score(scale, test_curve, kappa, feats)
                        details.append(
                            {
                                "estimator": estimator,
                                "rho": rho,
                                "cap_name": cap_name,
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
    for estimator in sorted({str(r["estimator"]) for r in details}):
        for train_curve, train_label in base.CURVES:
            for test_curve, test_label in base.CURVES:
                subset = [
                    r
                    for r in details
                    if r["estimator"] == estimator
                    and r["train_curve"] == train_curve
                    and r["test_curve"] == test_curve
                ]
                rows.append(
                    {
                        "estimator": estimator,
                        "rho": float(subset[0]["rho"]),
                        "cap_name": subset[0]["cap_name"],
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


def comparison(summary: list[dict[str, object]], kappa_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for estimator in sorted({str(r["estimator"]) for r in summary}):
        sub = [r for r in summary if r["estimator"] == estimator and r["train_curve"] != r["test_curve"]]
        cos_wsd = next(r for r in summary if r["estimator"] == estimator and r["train_curve"] == "cosine_72000.csv" and r["test_curve"] == "wsd_20000_24000.csv")
        w9_wsd = next(r for r in summary if r["estimator"] == estimator and r["train_curve"] == "wsdcon_9.csv" and r["test_curve"] == "wsd_20000_24000.csv")
        krows = [r for r in kappa_rows if r["estimator"] == estimator]
        cosine_krows = [r for r in krows if r["train_curve"] == "cosine_72000.csv"]
        rows.append(
            {
                "estimator": estimator,
                "rho": float(sub[0]["rho"]),
                "cap_name": sub[0]["cap_name"],
                "worst_offdiag": float(max(float(r["mean_delta_pct"]) for r in sub)),
                "median_offdiag": float(np.median([float(r["mean_delta_pct"]) for r in sub])),
                "mean_offdiag": float(np.mean([float(r["mean_delta_pct"]) for r in sub])),
                "cosine_to_wsd": float(cos_wsd["mean_delta_pct"]),
                "wsdcon9_to_wsd": float(w9_wsd["mean_delta_pct"]),
                "max_kappa": float(max(float(r["kappa"]) for r in krows)),
                "max_cosine_kappa": float(max(float(r["kappa"]) for r in cosine_krows)),
                "cap_saturation_rate": float(np.mean([int(r["cap_saturated"]) for r in krows])),
            }
        )
    rows.sort(key=lambda r: (max(float(r["worst_offdiag"]), 0.0), max(0.0, 10.0 + float(r["wsdcon9_to_wsd"])), float(r["mean_offdiag"])))
    return rows


def plot_heatmap(path: Path, comp: list[dict[str, object]], metric: str) -> None:
    rhos = sorted({float(r["rho"]) for r in comp})
    caps = ["none", "cap_0p05", "cap_0p04", "cap_0p03", "cap_0p02"]
    mat = np.full((len(rhos), len(caps)), np.nan)
    for i, rho in enumerate(rhos):
        for j, cap in enumerate(caps):
            row = next(r for r in comp if float(r["rho"]) == rho and r["cap_name"] == cap)
            mat[i, j] = float(row[metric])
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    im = ax.imshow(mat, cmap="RdYlGn_r", norm=TwoSlopeNorm(vmin=-60, vcenter=0, vmax=60))
    ax.set_xticks(np.arange(len(caps)), caps, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(rhos)), [str(r) for r in rhos])
    ax.set_xlabel("cap")
    ax.set_ylabel("retention exponent rho")
    ax.set_title(metric)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i,j]:+.1f}", ha="center", va="center", fontsize=8, weight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("MAE change vs MPL (%)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def write_report(comp: list[dict[str, object]]) -> None:
    recommended = next(r for r in comp if r["rho"] == 0.5 and r["cap_name"] == "cap_0p03")
    cap_free = [r for r in comp if r["cap_name"] == "none"]
    best_cap_free = sorted(cap_free, key=lambda r: (max(float(r["worst_offdiag"]), 0.0), float(r["mean_offdiag"])))[0]
    lines = [
        "# Cap Sensitivity for Nuisance-Orthogonal Kappa\n\n",
        "This audit checks whether the recommended degree-2 nuisance-orthogonal MAP estimator depends critically on the hard susceptibility cap.\n\n",
        "## Top Variants\n\n",
        "| estimator | worst offdiag | median offdiag | mean offdiag | cosine -> WSD | wsdcon_9 -> WSD | max cosine kappa | cap saturation |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in comp[:12]:
        lines.append(
            f"| `{row['estimator']}` | {float(row['worst_offdiag']):+.1f}% | {float(row['median_offdiag']):+.1f}% | "
            f"{float(row['mean_offdiag']):+.1f}% | {float(row['cosine_to_wsd']):+.1f}% | {float(row['wsdcon9_to_wsd']):+.1f}% | "
            f"{float(row['max_cosine_kappa']):.4f} | {100*float(row['cap_saturation_rate']):.1f}% |\n"
        )
    lines += [
        "\n## Sensitivity Maps\n\n",
        "![worst](figs/heatmap_worst_offdiag.png)\n\n",
        "![mean](figs/heatmap_mean_offdiag.png)\n\n",
        "![cosine](figs/heatmap_cosine_to_wsd.png)\n\n",
        "## Reading\n\n",
        f"Recommended paper setting: `{recommended['estimator']}`. "
        f"Best cap-free setting: `{best_cap_free['estimator']}`.\n\n",
        "If cap-free variants remain non-catastrophic, the cap is not carrying the entire method. "
        "If capped variants are clearly better, the cap should be described as a susceptibility prior rather than hidden as a tuning trick.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    details, kappa_rows = run()
    summary = summarize(details)
    comp = comparison(summary, kappa_rows)
    write_csv(OUT_DIR / "details.csv", details)
    write_csv(OUT_DIR / "summary.csv", summary)
    write_csv(OUT_DIR / "kappa_diagnostics.csv", kappa_rows)
    write_csv(OUT_DIR / "comparison.csv", comp)
    for metric in ["worst_offdiag", "mean_offdiag", "cosine_to_wsd", "wsdcon9_to_wsd"]:
        plot_heatmap(FIG_DIR / f"heatmap_{metric}.png", comp, metric)
    write_report(comp)
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    for row in comp[:16]:
        print(
            f"{row['estimator']:34s} worst={row['worst_offdiag']:+7.1f}% "
            f"mean={row['mean_offdiag']:+7.1f}% cos->wsd={row['cosine_to_wsd']:+7.1f}% "
            f"w9->wsd={row['wsdcon9_to_wsd']:+7.1f}% maxcosk={row['max_cosine_kappa']:.4f}"
        )


if __name__ == "__main__":
    main()
