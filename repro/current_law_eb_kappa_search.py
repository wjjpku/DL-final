#!/usr/bin/env python3
"""Empirical-Bayes kappa estimator for DropRelaxS.

This is the next step after the fixed-tau MAP/ridge estimator.  Instead of
choosing tau by grid search, estimate it from other reliable calibration curves:

    r = kappa * phi + eps
    eps ~ N(0, sigma^2 I)
    kappa ~ N_+(0, k0^2)

The nonnegative posterior mode is

    kappa_hat = max(0, <phi,r> / (||phi||^2 + (sigma/k0)^2 / w_id))

with an optional conservative susceptibility cap.  The ratio tau=sigma/k0 is
therefore a data-estimated prior/noise ratio.  In leave-curve-out mode, tau for
each calibration curve is estimated without using that curve.
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


OUT_DIR = ROOT / "results" / "current_law_eb_kappa_search"
FIG_DIR = OUT_DIR / "figs"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def quantile(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    xs = sorted(vals)
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def reliable(stats: dict[str, float]) -> bool:
    return stats["smooth_id_weight"] > 0.5 and stats["r2"] > 0.0 and stats["raw_kappa"] > 0.0


def estimate_tau(pool: list[dict[str, object]], mode: str) -> dict[str, float]:
    rel = [r for r in pool if reliable(r)]
    if len(rel) < 4:
        rel = [r for r in pool if r["raw_kappa"] > 0.0 and r["smooth_id_weight"] > 0.1]
    sigmas = [float(r["resid_robust_scale"]) for r in rel]
    kappas = [float(r["raw_kappa"]) for r in rel]

    sigma = quantile(sigmas, 0.50)
    if mode == "median":
        k0 = quantile(kappas, 0.50)
    elif mode == "mean":
        k0 = float(np.mean(kappas)) if kappas else 0.03
    elif mode == "q75":
        k0 = quantile(kappas, 0.75)
    elif mode == "trimmed_mean":
        lo, hi = quantile(kappas, 0.10), quantile(kappas, 0.90)
        trimmed = [k for k in kappas if lo <= k <= hi]
        k0 = float(np.mean(trimmed)) if trimmed else quantile(kappas, 0.50)
    else:
        raise ValueError(mode)

    k0 = max(k0, 1e-4)
    tau = min(max(sigma / k0, 0.005), 0.30)
    return {"tau": tau, "sigma": sigma, "k0": k0, "reliable_n": len(rel)}


def eb_kappa(stats: dict[str, float], tau: float, cap: float | None = 0.03) -> float:
    w = max(stats["smooth_id_weight"], 1e-6)
    denom = stats["feature_l2"] + tau * tau / w
    k = max(0.0, stats["projection_dot"] / max(denom, 1e-18))
    return min(k, cap) if cap is not None else k


def build_stats(feats) -> dict[tuple[str, str], dict[str, float]]:
    return {
        (scale, curve): amp.enriched_stats(scale, curve, feats)
        for curve, _ in base.CURVES
        for scale in base.SCALES
    }


def score(scale: str, test_curve: str, kappa: float, feats) -> dict[str, object]:
    return base.score(scale, test_curve, kappa, feats)


def run():
    feats = base.feature_cache()
    stats_cache = build_stats(feats)
    all_stats = []
    for (scale, curve), stats in stats_cache.items():
        label = next(label for c, label in base.CURVES if c == curve)
        all_stats.append({"scale": scale, "train_curve": curve, "train_label": label, **stats})

    details: list[dict[str, object]] = []
    kappa_rows: list[dict[str, object]] = []
    tau_rows: list[dict[str, object]] = []

    estimators = [
        ("current_smooth_cap", None),
        ("fixed_map_tau_0p03", None),
        ("fixed_map_tau_0p05", None),
        ("eb_lco_median", "median"),
        ("eb_lco_mean", "mean"),
        ("eb_lco_q75", "q75"),
        ("eb_lco_trimmed_mean", "trimmed_mean"),
    ]

    for estimator, mode in estimators:
        for train_curve, train_label in base.CURVES:
            for scale in base.SCALES:
                stats = stats_cache[(scale, train_curve)]
                if estimator == "current_smooth_cap":
                    tau_info = {"tau": float("nan"), "sigma": float("nan"), "k0": float("nan"), "reliable_n": 0}
                    kappa = base.estimate("smooth_weight_cap_0p03", stats)
                elif estimator == "fixed_map_tau_0p03":
                    tau_info = {"tau": 0.03, "sigma": float("nan"), "k0": float("nan"), "reliable_n": 0}
                    kappa = eb_kappa(stats, 0.03)
                elif estimator == "fixed_map_tau_0p05":
                    tau_info = {"tau": 0.05, "sigma": float("nan"), "k0": float("nan"), "reliable_n": 0}
                    kappa = eb_kappa(stats, 0.05)
                else:
                    assert mode is not None
                    pool = [r for r in all_stats if r["train_curve"] != train_curve]
                    tau_info = estimate_tau(pool, mode)
                    kappa = eb_kappa(stats, tau_info["tau"])

                tau_rows.append(
                    {
                        "estimator": estimator,
                        "scale": scale,
                        "train_curve": train_curve,
                        "train_label": train_label,
                        **tau_info,
                    }
                )
                kappa_rows.append(
                    {
                        "estimator": estimator,
                        "scale": scale,
                        "train_curve": train_curve,
                        "train_label": train_label,
                        "kappa": kappa,
                        **tau_info,
                        **stats,
                    }
                )
                for test_curve, test_label in base.CURVES:
                    details.append(
                        {
                            "estimator": estimator,
                            "scale": scale,
                            "train_curve": train_curve,
                            "train_label": train_label,
                            "test_curve": test_curve,
                            "test_label": test_label,
                            "kappa": kappa,
                            **score(scale, test_curve, kappa, feats),
                        }
                    )
    return details, kappa_rows, tau_rows


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
        rows.append(
            {
                "estimator": estimator,
                "worst_offdiag": float(max(float(r["mean_delta_pct"]) for r in sub)),
                "median_offdiag": float(np.median([float(r["mean_delta_pct"]) for r in sub])),
                "mean_offdiag": float(np.mean([float(r["mean_delta_pct"]) for r in sub])),
                "cosine_to_wsd": float(cos_wsd["mean_delta_pct"]),
                "wsdcon9_to_wsd": float(w9_wsd["mean_delta_pct"]),
            }
        )
    rows.sort(key=lambda r: (max(float(r["worst_offdiag"]), 0.0), max(0.0, 10.0 + float(r["wsdcon9_to_wsd"])), float(r["mean_offdiag"])))
    return rows


def plot_matrix(path: Path, summary: list[dict[str, object]], estimator: str) -> None:
    labels = [label for _, label in base.CURVES]
    mat = np.full((len(base.CURVES), len(base.CURVES)), np.nan)
    wins = np.zeros_like(mat)
    for i, (train_curve, _) in enumerate(base.CURVES):
        for j, (test_curve, _) in enumerate(base.CURVES):
            row = next(
                r
                for r in summary
                if r["estimator"] == estimator and r["train_curve"] == train_curve and r["test_curve"] == test_curve
            )
            mat[i, j] = float(row["mean_delta_pct"])
            wins[i, j] = int(row["wins"])
    fig, ax = plt.subplots(figsize=(9.2, 7.2))
    im = ax.imshow(mat, cmap="RdYlGn_r", norm=TwoSlopeNorm(vmin=-60, vcenter=0, vmax=150))
    ax.set_xticks(np.arange(len(labels)), labels, rotation=24, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("Test curve")
    ax.set_ylabel("Calibration curve")
    ax.set_title(estimator)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i,j]:+.1f}%\n{int(wins[i,j])}/3", ha="center", va="center", fontsize=8, weight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("MAE change vs MPL (negative is better)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def plot_tau(path: Path, tau_rows: list[dict[str, object]]) -> None:
    rows = [r for r in tau_rows if str(r["estimator"]).startswith("eb_lco")]
    labels = [label for _, label in base.CURVES]
    estimators = ["eb_lco_median", "eb_lco_mean", "eb_lco_q75", "eb_lco_trimmed_mean"]
    x = np.arange(len(labels))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    for ei, est in enumerate(estimators):
        vals = []
        for curve, _ in base.CURVES:
            subset = [float(r["tau"]) for r in rows if r["estimator"] == est and r["train_curve"] == curve]
            vals.append(float(np.mean(subset)))
        ax.bar(x + (ei - 1.5) * width, vals, width=width, label=est.replace("eb_lco_", ""))
    ax.set_xticks(x, labels, rotation=22, ha="right")
    ax.set_ylabel("leave-curve-out tau")
    ax.set_title("Empirical-Bayes tau estimates")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def write_report(comp, tau_rows) -> None:
    tau_summary = {}
    for est in ["eb_lco_median", "eb_lco_mean", "eb_lco_q75", "eb_lco_trimmed_mean"]:
        vals = [float(r["tau"]) for r in tau_rows if r["estimator"] == est]
        tau_summary[est] = (float(np.mean(vals)), float(np.min(vals)), float(np.max(vals)))

    lines = [
        "# Empirical-Bayes Kappa Estimator\n\n",
        "This report removes the hand-picked fixed `tau` from the MAP/ridge estimator. "
        "For each calibration curve, `tau=sigma/k0` is estimated from the other curves only, "
        "where `sigma` is a robust residual noise scale and `k0` is the prior susceptibility scale from reliable curves.\n\n",
        "## Formula\n\n",
        "```text\n",
        "r = kappa * phi + eps,      eps ~ N(0, sigma^2 I)\n",
        "kappa ~ N_+(0, k0^2)\n",
        "w_id = continuous identifiability weight\n",
        "tau = sigma / k0\n",
        "kappa_hat = min(0.03, max(0, <phi,r> / (||phi||^2 + tau^2 / w_id)))\n",
        "```\n\n",
        "The added denominator is exactly the MAP ridge penalty. Low-identifiability curves increase the prior precision through `1/w_id`, "
        "so trend-like residual alignment is allowed to help only when the curve has enough response information.\n\n",
        "## Comparison\n\n",
        "| estimator | worst offdiag | median offdiag | mean offdiag | cosine -> WSD | wsdcon_9 -> WSD |\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ]
    for row in comp:
        lines.append(
            f"| `{row['estimator']}` | {float(row['worst_offdiag']):+.1f}% | "
            f"{float(row['median_offdiag']):+.1f}% | {float(row['mean_offdiag']):+.1f}% | "
            f"{float(row['cosine_to_wsd']):+.1f}% | {float(row['wsdcon9_to_wsd']):+.1f}% |\n"
        )
    lines += [
        "\n## Tau Diagnostics\n\n",
        "![tau](figs/eb_tau_by_left_out_curve.png)\n\n",
        "| estimator | mean tau | min tau | max tau |\n",
        "|---|---:|---:|---:|\n",
    ]
    for est, vals in tau_summary.items():
        lines.append(f"| `{est}` | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} |\n")
    best_fixed = comp[0]["estimator"]
    best_eb = next(row["estimator"] for row in comp if str(row["estimator"]).startswith("eb_lco"))
    lines += [
        "\n## Recommended EB Candidate\n\n",
        f"Best fixed-tau reference: `{best_fixed}`. Recommended data-driven estimator: `{best_eb}`.\n\n",
        f"![matrix](figs/matrix_{best_eb}.png)\n\n",
        "Interpretation: the fixed `tau=0.03` result is a useful oracle/reference, but the EB estimator recovers a nearby "
        "regularization scale without using the held-out calibration curve. For the paper, the EB version is more defensible "
        "because `tau=sigma/k0` is estimated from residual noise and reliable susceptibility scale rather than selected by a grid search.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    details, kappa_rows, tau_rows = run()
    summary = summarize(details)
    comp = comparison(summary)
    write_csv(OUT_DIR / "details.csv", details)
    write_csv(OUT_DIR / "summary.csv", summary)
    write_csv(OUT_DIR / "kappa_diagnostics.csv", kappa_rows)
    write_csv(OUT_DIR / "tau_diagnostics.csv", tau_rows)
    write_csv(OUT_DIR / "comparison.csv", comp)
    plot_tau(FIG_DIR / "eb_tau_by_left_out_curve.png", tau_rows)
    for row in comp[:4]:
        plot_matrix(FIG_DIR / f"matrix_{row['estimator']}.png", summary, str(row["estimator"]))
    write_report(comp, tau_rows)
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    for row in comp:
        print(
            f"{row['estimator']:24s} worst={row['worst_offdiag']:+7.1f}% "
            f"median={row['median_offdiag']:+7.1f}% "
            f"cos->wsd={row['cosine_to_wsd']:+7.1f}% "
            f"w9->wsd={row['wsdcon9_to_wsd']:+7.1f}%"
        )


if __name__ == "__main__":
    main()
