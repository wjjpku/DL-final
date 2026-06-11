#!/usr/bin/env python3
"""Multi-curve calibration audit for the final kappa estimator.

The single-curve estimator extends naturally to multiple calibration curves by
concatenating the nuisance-projected regression problems.  For a train set S,

    dot_S = sum_c <M_G phi_c, M_G r_c>
    l2_S = sum_c ||M_G phi_c||^2
    full_l2_S = sum_c ||phi_c||^2

and the cap-free pooled estimator is

    kappa = sqrt(l2_S / full_l2_S) * (dot_S / (l2_S + tau^2))_+.

This script evaluates every non-empty proper train subset and scores only the
held-out curves.
"""
from __future__ import annotations

import csv
import itertools
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent
ROOT = REPO.parent
sys.path.insert(0, str(REPO))

import current_law_amplitude_kappa_search as amp  # noqa: E402
import current_law_continuous_kappa_search as base  # noqa: E402
import current_law_eb_kappa_search as eb  # noqa: E402
import current_law_orthogonal_kappa_search as orth  # noqa: E402


OUT_DIR = ROOT / "results" / "current_law_multicurve_kappa_audit"
FIG_DIR = OUT_DIR / "figs"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def train_name(curves: tuple[str, ...]) -> str:
    labels = {curve: label for curve, label in base.CURVES}
    return " + ".join(labels[c] for c in curves)


def pooled_kappa(stats_rows: list[dict[str, float]], tau: float, cap: float | None = None) -> dict[str, float]:
    dot = float(sum(float(r["orth_projection_dot"]) for r in stats_rows))
    l2 = float(sum(float(r["orth_feature_l2"]) for r in stats_rows))
    full_l2 = float(sum(float(r["feature_l2"]) for r in stats_rows))
    raw = max(0.0, dot / max(l2 + tau * tau, 1e-18))
    retention = l2 / max(full_l2, 1e-18)
    kappa = (max(retention, 0.0) ** 0.5) * raw
    if cap is not None:
        kappa = min(kappa, cap)
    return {
        "kappa": kappa,
        "raw_map": raw,
        "pooled_dot": dot,
        "pooled_orth_l2": l2,
        "pooled_full_l2": full_l2,
        "pooled_retention": retention,
    }


def run() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    feats = base.feature_cache()
    curves = tuple(curve for curve, _ in base.CURVES)
    labels = {curve: label for curve, label in base.CURVES}
    base_rows = []
    orth_stats = {}
    for curve, label in base.CURVES:
        for scale in base.SCALES:
            enriched = amp.enriched_stats(scale, curve, feats)
            base_rows.append({"scale": scale, "train_curve": curve, "train_label": label, **enriched})
            orth_stats[(scale, curve)] = orth.orthogonal_stats(scale, curve, feats, 2)

    details: list[dict[str, object]] = []
    kappa_rows: list[dict[str, object]] = []
    for train_size in range(1, len(curves)):
        for train_curves in itertools.combinations(curves, train_size):
            heldout = [curve for curve in curves if curve not in train_curves]
            pool = [r for r in base_rows if r["train_curve"] in train_curves]
            tau = eb.estimate_tau(pool, "q75")["tau"]
            train_id = "|".join(train_curves)
            train_label = train_name(train_curves)
            for scale in base.SCALES:
                rows = [orth_stats[(scale, curve)] for curve in train_curves]
                estimate = pooled_kappa(rows, tau, cap=None)
                kappa_rows.append(
                    {
                        "train_id": train_id,
                        "train_label": train_label,
                        "train_size": train_size,
                        "scale": scale,
                        "tau": tau,
                        **estimate,
                    }
                )
                for test_curve in heldout:
                    scored = base.score(scale, test_curve, float(estimate["kappa"]), feats)
                    details.append(
                        {
                            "train_id": train_id,
                            "train_label": train_label,
                            "train_size": train_size,
                            "scale": scale,
                            "test_curve": test_curve,
                            "test_label": labels[test_curve],
                            "kappa": estimate["kappa"],
                            **scored,
                        }
                    )
    return details, kappa_rows


def summarize_subsets(details: list[dict[str, object]], kappa_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for train_id in sorted({str(r["train_id"]) for r in details}):
        sub = [r for r in details if r["train_id"] == train_id]
        ksub = [r for r in kappa_rows if r["train_id"] == train_id]
        rows.append(
            {
                "train_id": train_id,
                "train_label": sub[0]["train_label"],
                "train_size": int(sub[0]["train_size"]),
                "heldout_tests": len(sub),
                "worst_heldout": float(max(float(r["delta_pct"]) for r in sub)),
                "mean_heldout": float(np.mean([float(r["delta_pct"]) for r in sub])),
                "median_heldout": float(np.median([float(r["delta_pct"]) for r in sub])),
                "wins": int(sum(int(r["win"]) for r in sub)),
                "mean_kappa": float(np.mean([float(r["kappa"]) for r in ksub])),
                "max_kappa": float(np.max([float(r["kappa"]) for r in ksub])),
                "mean_retention": float(np.mean([float(r["pooled_retention"]) for r in ksub])),
            }
        )
    return sorted(rows, key=lambda r: (int(r["train_size"]), float(r["worst_heldout"]), float(r["mean_heldout"])))


def summarize_sizes(subsets: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for train_size in sorted({int(r["train_size"]) for r in subsets}):
        sub = [r for r in subsets if int(r["train_size"]) == train_size]
        rows.append(
            {
                "train_size": train_size,
                "subset_count": len(sub),
                "mean_worst_heldout": float(np.mean([float(r["worst_heldout"]) for r in sub])),
                "median_worst_heldout": float(np.median([float(r["worst_heldout"]) for r in sub])),
                "best_worst_heldout": float(min(float(r["worst_heldout"]) for r in sub)),
                "worst_worst_heldout": float(max(float(r["worst_heldout"]) for r in sub)),
                "mean_heldout": float(np.mean([float(r["mean_heldout"]) for r in sub])),
                "median_heldout": float(np.median([float(r["mean_heldout"]) for r in sub])),
            }
        )
    return rows


def plot_size_summary(path: Path, rows: list[dict[str, object]]) -> None:
    x = [int(r["train_size"]) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].plot(x, [float(r["median_worst_heldout"]) for r in rows], marker="o", label="median worst")
    axes[0].plot(x, [float(r["best_worst_heldout"]) for r in rows], marker="s", label="best worst")
    axes[0].plot(x, [float(r["worst_worst_heldout"]) for r in rows], marker="^", label="worst worst")
    axes[0].axhline(0, color="#333333", linewidth=0.8)
    axes[0].set_xlabel("number of calibration curves")
    axes[0].set_ylabel("held-out delta pct")
    axes[0].set_title("worst held-out transfer")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].plot(x, [float(r["mean_heldout"]) for r in rows], marker="o")
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_xlabel("number of calibration curves")
    axes[1].set_ylabel("held-out delta pct")
    axes[1].set_title("mean held-out transfer")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def write_report(size_rows: list[dict[str, object]], subset_rows: list[dict[str, object]]) -> None:
    best_by_size = {
        int(size): min(
            [r for r in subset_rows if int(r["train_size"]) == int(size)],
            key=lambda r: (float(r["worst_heldout"]), float(r["mean_heldout"])),
        )
        for size in sorted({int(r["train_size"]) for r in subset_rows})
    }
    lines = [
        "# Multi-Curve Kappa Calibration Audit\n\n",
        "This audit extends the final cap-free estimator to multiple calibration curves by concatenating the nuisance-projected regression problems. "
        "For each train subset, `tau` is estimated from the training curves only, and evaluation is performed only on held-out curves.\n\n",
        "```text\n",
        "dot_S = sum_c <M_G phi_c, M_G r_c>\n",
        "l2_S = sum_c ||M_G phi_c||^2\n",
        "full_l2_S = sum_c ||phi_c||^2\n",
        "kappa_S = sqrt(l2_S / full_l2_S) * max(0, dot_S / (l2_S + tau^2))\n",
        "```\n\n",
        "![size summary](figs/train_size_summary.png)\n\n",
        "## Train-Size Summary\n\n",
        "| train curves | subsets | median worst heldout | best worst heldout | worst worst heldout | mean heldout |\n",
        "|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in size_rows:
        lines.append(
            f"| {int(row['train_size'])} | {int(row['subset_count'])} | "
            f"{float(row['median_worst_heldout']):+.1f}% | {float(row['best_worst_heldout']):+.1f}% | "
            f"{float(row['worst_worst_heldout']):+.1f}% | {float(row['mean_heldout']):+.1f}% |\n"
        )
    lines += [
        "\n## Best Subset By Train Size\n\n",
        "| train curves | best subset | worst heldout | mean heldout | max kappa |\n",
        "|---:|---|---:|---:|---:|\n",
    ]
    for size, row in best_by_size.items():
        lines.append(
            f"| {size} | {row['train_label']} | {float(row['worst_heldout']):+.1f}% | "
            f"{float(row['mean_heldout']):+.1f}% | {float(row['max_kappa']):.4f} |\n"
        )
    lines += [
        "\n## Reading\n\n",
        "The pooled estimator is the same formula as the single-curve estimator, with inner products and norms summed over calibration curves. "
        "This gives a principled way to use more calibration data without introducing schedule-family labels.\n\n",
        "The audit should be read as a calibration-coverage test. Larger train sets reduce dependence on any single response shape, but they can also average together "
        "curves whose residual response amplitudes differ. The useful question is therefore not whether every pooled subset is better, but whether the formula remains "
        "stable and whether well-covered train sets improve held-out transfer without a hard cap.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    details, kappas = run()
    subset_rows = summarize_subsets(details, kappas)
    size_rows = summarize_sizes(subset_rows)
    write_csv(OUT_DIR / "details.csv", details)
    write_csv(OUT_DIR / "kappa_diagnostics.csv", kappas)
    write_csv(OUT_DIR / "subset_summary.csv", subset_rows)
    write_csv(OUT_DIR / "train_size_summary.csv", size_rows)
    plot_size_summary(FIG_DIR / "train_size_summary.png", size_rows)
    write_report(size_rows, subset_rows)
    for row in size_rows:
        print(
            f"n={int(row['train_size'])} subsets={int(row['subset_count'])} "
            f"median_worst={float(row['median_worst_heldout']):+6.1f}% "
            f"best_worst={float(row['best_worst_heldout']):+6.1f}% "
            f"mean={float(row['mean_heldout']):+6.1f}%"
        )


if __name__ == "__main__":
    main()
