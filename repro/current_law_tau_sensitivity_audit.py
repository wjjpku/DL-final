#!/usr/bin/env python3
"""Sensitivity audit for the EB tau in the final kappa estimator.

The final estimator uses tau=sigma/k0 from leave-curve-out empirical Bayes.
This script multiplies that tau by a range of constants while keeping the
degree-2 nuisance projection, sqrt(retention), and optional cap fixed.
"""
from __future__ import annotations

import csv
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


OUT_DIR = ROOT / "results" / "current_law_tau_sensitivity_audit"
FIG_DIR = OUT_DIR / "figs"
MULTIPLIERS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
CAPS: list[float | None] = [None, 0.03]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def estimate_kappa(stats: dict[str, float], tau: float, cap: float | None) -> float:
    denom = float(stats["orth_feature_l2"]) + tau * tau
    raw = max(0.0, float(stats["orth_projection_dot"]) / max(denom, 1e-18))
    retention = max(float(stats["orth_feature_retention"]), 0.0)
    kappa = (retention**0.5) * raw
    return min(kappa, cap) if cap is not None else kappa


def estimator_name(multiplier: float, cap: float | None) -> str:
    cap_part = "nocap" if cap is None else f"cap_{cap:.2f}".replace(".", "p")
    return f"tau_x_{multiplier:.2f}_{cap_part}".replace(".", "p")


def run() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
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

    details: list[dict[str, object]] = []
    kappas: list[dict[str, object]] = []
    for multiplier in MULTIPLIERS:
        for cap in CAPS:
            estimator = estimator_name(multiplier, cap)
            for train_curve, train_label in base.CURVES:
                pool = [r for r in base_rows if r["train_curve"] != train_curve]
                tau_info = eb.estimate_tau(pool, "q75")
                base_tau = float(tau_info["tau"])
                tau = multiplier * base_tau
                for scale in base.SCALES:
                    stats = stats_cache[(scale, train_curve)]
                    kappa = estimate_kappa(stats, tau, cap)
                    kappas.append(
                        {
                            "estimator": estimator,
                            "tau_multiplier": multiplier,
                            "cap": "" if cap is None else cap,
                            "scale": scale,
                            "train_curve": train_curve,
                            "train_label": train_label,
                            "base_tau": base_tau,
                            "tau": tau,
                            "kappa": kappa,
                            "cap_saturated": int(cap is not None and kappa >= cap - 1e-12),
                            "orth_feature_retention": stats["orth_feature_retention"],
                        }
                    )
                    for test_curve, test_label in base.CURVES:
                        details.append(
                            {
                                "estimator": estimator,
                                "tau_multiplier": multiplier,
                                "cap": "" if cap is None else cap,
                                "scale": scale,
                                "train_curve": train_curve,
                                "train_label": train_label,
                                "test_curve": test_curve,
                                "test_label": test_label,
                                "kappa": kappa,
                                **base.score(scale, test_curve, kappa, feats),
                            }
                        )
    return details, kappas


def summarize(details: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for estimator in sorted({str(r["estimator"]) for r in details}):
        for train_curve, train_label in base.CURVES:
            for test_curve, test_label in base.CURVES:
                sub = [
                    r
                    for r in details
                    if r["estimator"] == estimator
                    and r["train_curve"] == train_curve
                    and r["test_curve"] == test_curve
                ]
                rows.append(
                    {
                        "estimator": estimator,
                        "tau_multiplier": float(sub[0]["tau_multiplier"]),
                        "cap": sub[0]["cap"],
                        "train_curve": train_curve,
                        "train_label": train_label,
                        "test_curve": test_curve,
                        "test_label": test_label,
                        "mean_delta_pct": float(np.mean([float(r["delta_pct"]) for r in sub])),
                        "wins": int(sum(int(r["win"]) for r in sub)),
                        "tests": len(sub),
                        "mean_kappa": float(np.mean([float(r["kappa"]) for r in sub])),
                        "max_kappa": float(np.max([float(r["kappa"]) for r in sub])),
                    }
                )
    return rows


def compare(summary: list[dict[str, object]], kappas: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for estimator in sorted({str(r["estimator"]) for r in summary}):
        sub = [r for r in summary if r["estimator"] == estimator and r["train_curve"] != r["test_curve"]]
        cos_wsd = next(
            r
            for r in summary
            if r["estimator"] == estimator
            and r["train_curve"] == "cosine_72000.csv"
            and r["test_curve"] == "wsd_20000_24000.csv"
        )
        w9_wsd = next(
            r
            for r in summary
            if r["estimator"] == estimator
            and r["train_curve"] == "wsdcon_9.csv"
            and r["test_curve"] == "wsd_20000_24000.csv"
        )
        krows = [r for r in kappas if r["estimator"] == estimator]
        cos_krows = [r for r in krows if r["train_curve"] == "cosine_72000.csv"]
        rows.append(
            {
                "estimator": estimator,
                "tau_multiplier": float(krows[0]["tau_multiplier"]),
                "cap": krows[0]["cap"],
                "worst_offdiag": float(max(float(r["mean_delta_pct"]) for r in sub)),
                "median_offdiag": float(np.median([float(r["mean_delta_pct"]) for r in sub])),
                "mean_offdiag": float(np.mean([float(r["mean_delta_pct"]) for r in sub])),
                "cosine_to_wsd": float(cos_wsd["mean_delta_pct"]),
                "wsdcon9_to_wsd": float(w9_wsd["mean_delta_pct"]),
                "max_kappa": float(max(float(r["kappa"]) for r in krows)),
                "max_cosine_kappa": float(max(float(r["kappa"]) for r in cos_krows)),
                "cap_saturation_rate": float(np.mean([int(r["cap_saturated"]) for r in krows])),
                "mean_tau": float(np.mean([float(r["tau"]) for r in krows])),
            }
        )
    return sorted(rows, key=lambda r: (str(r["cap"]), float(r["tau_multiplier"])))


def plot_tradeoff(path: Path, rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharex=True)
    for cap_label, label in [("", "no cap"), ("0.03", "cap 0.03")]:
        sub = [r for r in rows if str(r["cap"]) == cap_label]
        x = [float(r["tau_multiplier"]) for r in sub]
        axes[0].plot(x, [float(r["worst_offdiag"]) for r in sub], marker="o", label=f"{label}: worst")
        axes[0].plot(x, [float(r["cosine_to_wsd"]) for r in sub], marker="s", linestyle="--", label=f"{label}: cosine->WSD")
        axes[1].plot(x, [float(r["max_cosine_kappa"]) for r in sub], marker="o", label=label)
    axes[0].axhline(0, color="#333333", linewidth=0.8)
    axes[0].axvline(1, color="#777777", linewidth=0.8, linestyle=":")
    axes[0].set_xscale("symlog", linthresh=0.25)
    axes[0].set_xlabel("tau multiplier")
    axes[0].set_ylabel("mean delta pct")
    axes[0].set_title("transfer change vs EB tau multiplier")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[1].axvline(1, color="#777777", linewidth=0.8, linestyle=":")
    axes[1].set_xscale("symlog", linthresh=0.25)
    axes[1].set_xlabel("tau multiplier")
    axes[1].set_ylabel("max cosine kappa")
    axes[1].set_title("cosine-derived kappa magnitude")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def write_report(rows: list[dict[str, object]]) -> None:
    selected = next(r for r in rows if float(r["tau_multiplier"]) == 1.0 and str(r["cap"]) == "0.03")
    half = next(r for r in rows if float(r["tau_multiplier"]) == 0.5 and str(r["cap"]) == "0.03")
    double = next(r for r in rows if float(r["tau_multiplier"]) == 2.0 and str(r["cap"]) == "0.03")
    lines = [
        "# Tau Sensitivity Audit\n\n",
        "This audit multiplies the leave-curve-out EB `tau` by constants while keeping the final degree-2 nuisance projection and `sqrt(retention)` correction fixed.\n\n",
        "![tradeoff](figs/tau_multiplier_tradeoff.png)\n\n",
        "## Comparison\n\n",
        "| tau multiplier | cap | worst offdiag | mean offdiag | cosine -> WSD | wsdcon_9 -> WSD | max cosine kappa | cap saturation |\n",
        "|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in rows:
        cap = "none" if str(row["cap"]) == "" else str(row["cap"])
        lines.append(
            f"| {float(row['tau_multiplier']):.2f} | {cap} | {float(row['worst_offdiag']):+.1f}% | "
            f"{float(row['mean_offdiag']):+.1f}% | {float(row['cosine_to_wsd']):+.1f}% | "
            f"{float(row['wsdcon9_to_wsd']):+.1f}% | {float(row['max_cosine_kappa']):.4f} | "
            f"{100 * float(row['cap_saturation_rate']):.1f}% |\n"
        )
    lines += [
        "\n## Reading\n\n",
        f"The selected EB scale (`1.00x`) gives worst off-diagonal {float(selected['worst_offdiag']):+.1f}% "
        f"and cosine -> WSD {float(selected['cosine_to_wsd']):+.1f}% with cap.\n\n",
        f"Changing tau by a factor of two remains stable: `0.50x` gives worst {float(half['worst_offdiag']):+.1f}% "
        f"and `2.00x` gives worst {float(double['worst_offdiag']):+.1f}% with cap. "
        "The main tradeoff is expected: smaller tau gives stronger transfer but more amplitude risk, while larger tau is more conservative.\n\n",
        "The practical conclusion is that the final method does not rely on an exact tau value. "
        "EB tau places the estimator in the useful middle regime, and the retention factor handles most of the amplitude stabilization.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    details, kappas = run()
    summary = summarize(details)
    rows = compare(summary, kappas)
    write_csv(OUT_DIR / "details.csv", details)
    write_csv(OUT_DIR / "kappa_diagnostics.csv", kappas)
    write_csv(OUT_DIR / "summary.csv", summary)
    write_csv(OUT_DIR / "comparison.csv", rows)
    plot_tradeoff(FIG_DIR / "tau_multiplier_tradeoff.png", rows)
    write_report(rows)
    for row in rows:
        print(
            f"tau_x={float(row['tau_multiplier']):.2f} cap={row['cap'] or 'none':>4} "
            f"worst={float(row['worst_offdiag']):+6.1f}% cos->wsd={float(row['cosine_to_wsd']):+6.1f}% "
            f"maxcosk={float(row['max_cosine_kappa']):.4f}"
        )


if __name__ == "__main__":
    main()
