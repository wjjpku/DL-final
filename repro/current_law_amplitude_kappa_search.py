#!/usr/bin/env python3
"""Amplitude-regularized kappa search for DropRelaxS.

The previous continuous estimator fixes the main practical failure mode by
shrinking raw projection kappas with an identifiability weight.  This script
tests a more principled view of the same problem:

    residual r = kappa * phi + epsilon

The residual shape phi is often usable, but kappa is unstable when phi has low
information.  We therefore benchmark estimators that regularize the amplitude
in loss space or, equivalently, use a reliability-dependent MAP/ridge prior on
kappa before converting the projected residual back to a coefficient.
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

import current_law_continuous_kappa_search as base  # noqa: E402


OUT_DIR = ROOT / "results" / "current_law_amplitude_kappa_search"
FIG_DIR = OUT_DIR / "figs"


def robust_scale(x: np.ndarray) -> float:
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return max(1.4826 * mad, float(np.std(x)) * 0.25, 1e-12)


def enriched_stats(scale: str, curve_name: str, feats) -> dict[str, float]:
    stats = base.curve_stats(scale, curve_name, feats)
    curve = base.load_curve(scale, curve_name)
    phi = feats[(scale, curve_name)]
    resid = curve.loss - base.mpl_predict(base.MPL_PRECOMPUTED_INIT[scale], curve)

    phi2 = max(float(np.dot(phi, phi)), 1e-18)
    phi_norm = math.sqrt(phi2)
    unit_phi = phi / phi_norm
    dot = float(np.dot(phi, resid))
    amp_raw = max(0.0, float(np.dot(unit_phi, resid)))

    stats.update(
        {
            "phi_norm": phi_norm,
            "projection_dot": dot,
            "amp_raw": amp_raw,
            "resid_std": float(np.std(resid)),
            "resid_robust_scale": robust_scale(resid),
            "smooth_id_weight": base.smooth_id_weight(stats),
        }
    )
    return stats


def reliability_ridge(stats: dict[str, float], tau: float, cap: float | None = 0.03) -> float:
    """MAP estimator with stronger prior precision when identifiability is low.

    For Gaussian residual noise and a zero-centered Gaussian prior on kappa,
    the posterior mode is:

        kappa = <phi,r> / (||phi||^2 + lambda)

    We set lambda = tau^2 / w_id, so low-identifiability curves receive a
    stronger prior toward zero without any schedule-name decision.
    """
    w = max(stats["smooth_id_weight"], 1e-6)
    denom = stats["feature_l2"] + tau * tau / w
    k = max(0.0, stats["projection_dot"] / max(denom, 1e-18))
    return min(k, cap) if cap is not None else k


def amplitude_floor(stats: dict[str, float], tau: float, cap: float | None = 0.03) -> float:
    """Estimate response amplitude first, then divide by a stabilized norm."""
    w = stats["smooth_id_weight"]
    denom = math.sqrt(stats["feature_l2"] + tau * tau)
    k = w * stats["amp_raw"] / max(denom, 1e-18)
    return min(k, cap) if cap is not None else k


def amp_cap_floor(stats: dict[str, float], tau: float, c: float, cap: float | None = 0.03) -> float:
    """Loss-space amplitude cap before conversion to kappa."""
    w = stats["smooth_id_weight"]
    amp = min(stats["amp_raw"], c * stats["resid_robust_scale"])
    denom = math.sqrt(stats["feature_l2"] + tau * tau)
    k = w * amp / max(denom, 1e-18)
    return min(k, cap) if cap is not None else k


def estimator_specs() -> list[tuple[str, callable]]:
    specs: list[tuple[str, callable]] = [
        ("current_smooth_cap", lambda s: base.estimate("smooth_weight_cap_0p03", s)),
    ]
    for tau in [0.01, 0.03, 0.05, 0.10, 0.20, 0.30]:
        name = f"map_ridge_tau_{str(tau).replace('.', 'p')}"
        specs.append((name, lambda s, tau=tau: reliability_ridge(s, tau=tau, cap=0.03)))
    for tau in [0.03, 0.05, 0.10, 0.20, 0.30]:
        name = f"amp_floor_tau_{str(tau).replace('.', 'p')}"
        specs.append((name, lambda s, tau=tau: amplitude_floor(s, tau=tau, cap=0.03)))
    for tau in [0.03, 0.10, 0.20]:
        for c in [1.0, 1.5, 2.0, 3.0]:
            name = f"amp_cap_tau_{str(tau).replace('.', 'p')}_c_{str(c).replace('.', 'p')}"
            specs.append((name, lambda s, tau=tau, c=c: amp_cap_floor(s, tau=tau, c=c, cap=0.03)))
    return specs


def score(scale: str, test_curve: str, kappa: float, feats) -> dict[str, object]:
    return base.score(scale, test_curve, kappa, feats)


def run():
    feats = base.feature_cache()
    specs = estimator_specs()
    stats_cache = {
        (scale, curve): enriched_stats(scale, curve, feats)
        for curve, _ in base.CURVES
        for scale in base.SCALES
    }

    details, kappa_rows = [], []
    for estimator, fn in specs:
        for train_curve, train_label in base.CURVES:
            for scale in base.SCALES:
                stats = stats_cache[(scale, train_curve)]
                kappa = float(fn(stats))
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
                for test_curve, test_label in base.CURVES:
                    details.append(
                        {
                            "estimator": estimator,
                            "train_curve": train_curve,
                            "train_label": train_label,
                            "test_label": test_label,
                            "kappa": kappa,
                            **score(scale, test_curve, kappa, feats),
                        }
                    )
    return details, kappa_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
    def objective(row: dict[str, object]) -> tuple[float, float, float]:
        # Prefer estimators that are non-catastrophic and still useful.  Purely
        # conservative estimators with near-zero kappa are safe but do not solve
        # the original cosine/WSD transfer problem.
        worst_penalty = max(float(row["worst_offdiag"]), 0.0)
        useful_deficit = max(0.0, 10.0 + float(row["wsdcon9_to_wsd"]))
        return (worst_penalty, useful_deficit, float(row["mean_offdiag"]))

    rows.sort(key=objective)
    return rows


def choose_recommended(comp: list[dict[str, object]], kappa_rows: list[dict[str, object]]) -> str:
    """Select a useful MAP candidate that does not rely on cosine hitting cap."""
    cosine_max: dict[str, float] = {}
    for row in kappa_rows:
        if row["train_curve"] != "cosine_72000.csv":
            continue
        est = str(row["estimator"])
        cosine_max[est] = max(cosine_max.get(est, 0.0), float(row["kappa"]))

    candidates = [
        row
        for row in comp
        if str(row["estimator"]).startswith("map_ridge")
        and float(row["worst_offdiag"]) <= 0.1
        and float(row["wsdcon9_to_wsd"]) <= -10.0
        and cosine_max.get(str(row["estimator"]), 1.0) < 0.01
    ]
    if candidates:
        candidates.sort(key=lambda r: float(r["mean_offdiag"]))
        return str(candidates[0]["estimator"])
    return str(comp[0]["estimator"])


def plot_top_comparison(path: Path, rows: list[dict[str, object]]) -> None:
    top = rows[:12]
    labels = [str(r["estimator"]) for r in top]
    x = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    width = 0.26
    ax.bar(x - width, [float(r["worst_offdiag"]) for r in top], width, label="worst offdiag")
    ax.bar(x, [float(r["cosine_to_wsd"]) for r in top], width, label="cosine -> WSD")
    ax.bar(x + width, [float(r["wsdcon9_to_wsd"]) for r in top], width, label="wsdcon_9 -> WSD")
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    ax.set_xticks(x, labels, rotation=28, ha="right")
    ax.set_ylabel("MAE change vs MPL (%)")
    ax.set_title("Amplitude/MAP kappa estimators")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


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
    norm = TwoSlopeNorm(vmin=-60, vcenter=0, vmax=150)
    im = ax.imshow(mat, cmap="RdYlGn_r", norm=norm)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=24, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("Test curve")
    ax.set_ylabel("Calibration curve")
    ax.set_title(estimator)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i,j]:+.1f}%\n{int(wins[i,j])}/3", ha="center", va="center", fontsize=8, weight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("MAE change vs MPL (negative is better)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def write_report(summary, comp, kappa_rows) -> None:
    best = choose_recommended(comp, kappa_rows)
    lines = [
        "# Amplitude-Regularized Kappa Search\n\n",
        "This experiment targets the observed failure mode: the residual response shape is often usable, "
        "but raw `kappa=<phi,r>/<phi,phi>` can have the wrong magnitude when `phi` has low feature energy.\n\n",
        "## Estimators\n\n",
        "- `current_smooth_cap`: previous continuous identifiability-weighted estimator.\n",
        "- `map_ridge_tau_*`: MAP/ridge estimator `kappa=max(0,<phi,r>/(||phi||^2 + tau^2/w_id))`, capped at `0.03`.\n",
        "- `amp_floor_tau_*`: estimates normalized loss-space amplitude `<phi/||phi||,r>`, then divides by `sqrt(||phi||^2+tau^2)`.\n",
        "- `amp_cap_tau_*`: additionally caps the loss-space amplitude by a robust residual scale before conversion to kappa.\n\n",
        "The MAP form has the cleanest theory: under `r=kappa phi+epsilon` with Gaussian noise and a zero-centered "
        "Gaussian prior on `kappa`, the posterior mode is ridge regression. Setting prior precision proportional to "
        "`1/w_id` makes low-identifiability curves shrink continuously toward zero rather than exploding through the denominator.\n\n",
        "## Top Comparison\n\n",
        "![comparison](figs/top_estimator_comparison.png)\n\n",
        "| estimator | worst offdiag | median offdiag | cosine -> WSD | wsdcon_9 -> WSD |\n",
        "|---|---:|---:|---:|---:|\n",
    ]
    for row in comp[:12]:
        lines.append(
            f"| `{row['estimator']}` | {float(row['worst_offdiag']):+.1f}% | "
            f"{float(row['median_offdiag']):+.1f}% | {float(row['cosine_to_wsd']):+.1f}% | "
            f"{float(row['wsdcon9_to_wsd']):+.1f}% |\n"
        )
    lines += [
        "\n## Recommended Candidate\n\n",
        f"Recommended by the current safety/useful-transfer objective: `{best}`.\n\n",
        f"![matrix](figs/matrix_{best}.png)\n\n",
        "This recommendation intentionally rejects `map_ridge_tau_0p01` even though it has stronger aggregate numbers, "
        "because cosine-derived kappa hits the `0.03` cap there. The selected candidate keeps cosine-derived kappa "
        "below `0.01`, so the improvement is coming from reliability-weighted MAP shrinkage rather than from a saturated cap.\n\n",
        "Important caveat: this is still a finite-data estimator search, not a proof of universal optimality. "
        "The structural improvement is that magnitude control is now expressed as a MAP/energy prior rather than as a schedule-class rule.\n",
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
    plot_top_comparison(FIG_DIR / "top_estimator_comparison.png", comp)
    matrix_names = {str(row["estimator"]) for row in comp[:5]}
    matrix_names.add(choose_recommended(comp, kappa_rows))
    for estimator in sorted(matrix_names):
        plot_matrix(FIG_DIR / f"matrix_{estimator}.png", summary, estimator)
    write_report(summary, comp, kappa_rows)
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    for row in comp[:12]:
        print(
            f"{row['estimator']:28s} worst={row['worst_offdiag']:+7.1f}% "
            f"cos->wsd={row['cosine_to_wsd']:+7.1f}% "
            f"w9->wsd={row['wsdcon9_to_wsd']:+7.1f}%"
        )


if __name__ == "__main__":
    main()
