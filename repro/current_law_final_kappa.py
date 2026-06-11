#!/usr/bin/env python3
"""Final consolidated kappa estimator audit.

This script keeps only the estimators that matter after the exploratory search:

1. MPL only / existing smooth-cap baseline.
2. EB ridge without nuisance orthogonalization.
3. Final recommended estimator:

       r = loss - MPL
       phi_perp = M_G phi
       r_perp   = M_G r
       tau = sigma / k0      (leave-curve-out empirical Bayes)
       retention = ||phi_perp||^2 / ||phi||^2
       kappa = retention^0.5 * (<phi_perp,r_perp> / (||phi_perp||^2 + tau^2))_+

   Here G is a lightweight low-frequency MPL-residual nuisance subspace.  The
   cap-free estimator is the paper-facing version; cap=0.03 is an optional
   truncated susceptibility prior.
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


OUT_DIR = ROOT / "results" / "current_law_final_kappa"
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


def final_kappa(stats: dict[str, float], tau: float, cap: float | None = 0.03) -> float:
    denom = stats["orth_feature_l2"] + tau * tau
    raw = max(0.0, stats["orth_projection_dot"] / max(denom, 1e-18))
    retention = max(float(stats["orth_feature_retention"]), 0.0)
    kappa = (retention ** 0.5) * raw
    return min(kappa, cap) if cap is not None else kappa


def final_kappa_no_retention(stats: dict[str, float], tau: float, cap: float | None = 0.03) -> float:
    denom = stats["orth_feature_l2"] + tau * tau
    kappa = max(0.0, stats["orth_projection_dot"] / max(denom, 1e-18))
    return min(kappa, cap) if cap is not None else kappa


def dct_low_frequency_basis(n: int, modes: int) -> np.ndarray:
    idx = np.arange(n, dtype=np.float64)
    cols = [np.ones(n, dtype=np.float64)]
    for k in range(1, modes + 1):
        cols.append(np.cos(math.pi * (idx + 0.5) * k / max(n, 1)))
    z = np.column_stack(cols)
    norms = np.linalg.norm(z, axis=0)
    return z / np.maximum(norms, 1e-12)


def residualize(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(z, y, rcond=None)
    return y - z @ coef


def spectral_orthogonal_stats(scale: str, curve_name: str, feats, modes: int) -> dict[str, float]:
    stats = amp.enriched_stats(scale, curve_name, feats)
    curve = base.load_curve(scale, curve_name)
    phi = feats[(scale, curve_name)]
    resid = curve.loss - base.mpl_predict(base.MPL_PRECOMPUTED_INIT[scale], curve)
    z = dct_low_frequency_basis(len(curve.step), modes)
    phi_o = residualize(phi, z)
    resid_o = residualize(resid, z)
    phi_o2 = max(float(np.dot(phi_o, phi_o)), 1e-18)
    dot_o = float(np.dot(phi_o, resid_o))
    retention = float(phi_o2 / max(float(np.dot(phi, phi)), 1e-18))
    corr_denom = float(np.linalg.norm(phi_o) * np.linalg.norm(resid_o))
    corr_o = 0.0 if corr_denom <= 1e-18 else float(np.dot(phi_o, resid_o) / corr_denom)
    return {
        **stats,
        "nuisance_family": "dct_low_frequency",
        "nuisance_modes": modes,
        "orth_feature_l2": phi_o2,
        "orth_projection_dot": dot_o,
        "orth_raw_kappa": max(0.0, dot_o / phi_o2),
        "orth_feature_retention": retention,
        "orth_corr": corr_o,
        "orth_resid_scale": amp.robust_scale(resid_o),
    }


def build_base_rows(feats) -> list[dict[str, object]]:
    rows = []
    for curve, label in base.CURVES:
        for scale in base.SCALES:
            stats = amp.enriched_stats(scale, curve, feats)
            rows.append({"scale": scale, "train_curve": curve, "train_label": label, **stats})
    return rows


def run():
    feats = base.feature_cache()
    base_rows = build_base_rows(feats)
    orth_stats = {
        (scale, curve): orth.orthogonal_stats(scale, curve, feats, 2)
        for curve, _ in base.CURVES
        for scale in base.SCALES
    }
    spectral4_stats = {
        (scale, curve): spectral_orthogonal_stats(scale, curve, feats, 4)
        for curve, _ in base.CURVES
        for scale in base.SCALES
    }

    estimators = [
        "smooth_cap",
        "eb_q75",
        "final_no_cap",
        "final_spectral_G4_no_cap",
        "final_cap_0p03",
        "final_spectral_G4_cap_0p03",
        "no_retention_cap_0p03",
        "numeric_oracle_deg1",
    ]
    details: list[dict[str, object]] = []
    kappa_rows: list[dict[str, object]] = []

    for estimator in estimators:
        for train_curve, train_label in base.CURVES:
            pool = [r for r in base_rows if r["train_curve"] != train_curve]
            tau = eb.estimate_tau(pool, "q75")["tau"]
            for scale in base.SCALES:
                if estimator == "numeric_oracle_deg1":
                    stats = orth.orthogonal_stats(scale, train_curve, feats, 1)
                    kappa = orth.kappa_from_stats(stats, "orth_map_retention", tau)
                elif estimator.startswith("final_spectral_G4"):
                    stats = spectral4_stats[(scale, train_curve)]
                    cap = 0.03 if estimator.endswith("cap_0p03") else None
                    kappa = final_kappa(stats, tau, cap=cap)
                else:
                    stats = orth_stats[(scale, train_curve)]
                    if estimator == "smooth_cap":
                        kappa = base.estimate("smooth_weight_cap_0p03", stats)
                    elif estimator == "eb_q75":
                        kappa = eb.eb_kappa(stats, tau)
                    elif estimator == "final_no_cap":
                        kappa = final_kappa(stats, tau, cap=None)
                    elif estimator == "final_cap_0p03":
                        kappa = final_kappa(stats, tau, cap=0.03)
                    elif estimator == "no_retention_cap_0p03":
                        kappa = final_kappa_no_retention(stats, tau, cap=0.03)
                    else:
                        raise ValueError(estimator)

                kappa_rows.append(
                    {
                        "estimator": estimator,
                        "scale": scale,
                        "train_curve": train_curve,
                        "train_label": train_label,
                        "tau": tau,
                        "kappa": kappa,
                        "cap_saturated": int(estimator.endswith("cap_0p03") and kappa >= 0.03 - 1e-12),
                        **stats,
                    }
                )
                for test_curve, test_label in base.CURVES:
                    scored = base.score(scale, test_curve, kappa, feats)
                    details.append(
                        {
                            "estimator": estimator,
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
    order = [
        "smooth_cap",
        "eb_q75",
        "final_no_cap",
        "final_spectral_G4_no_cap",
        "final_cap_0p03",
        "final_spectral_G4_cap_0p03",
        "no_retention_cap_0p03",
        "numeric_oracle_deg1",
    ]
    for estimator in order:
        sub = [r for r in summary if r["estimator"] == estimator and r["train_curve"] != r["test_curve"]]
        cos_wsd = next(r for r in summary if r["estimator"] == estimator and r["train_curve"] == "cosine_72000.csv" and r["test_curve"] == "wsd_20000_24000.csv")
        w9_wsd = next(r for r in summary if r["estimator"] == estimator and r["train_curve"] == "wsdcon_9.csv" and r["test_curve"] == "wsd_20000_24000.csv")
        krows = [r for r in kappa_rows if r["estimator"] == estimator]
        cosine_krows = [r for r in krows if r["train_curve"] == "cosine_72000.csv"]
        rows.append(
            {
                "estimator": estimator,
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


def plot_kappa_diagnostics(path: Path, kappa_rows: list[dict[str, object]]) -> None:
    rows = [r for r in kappa_rows if r["estimator"] == "final_cap_0p03"]
    labels = [label for _, label in base.CURVES]
    x = np.arange(len(labels))
    width = 0.24
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for si, scale in enumerate(base.SCALES):
        subset = [r for r in rows if r["scale"] == scale]
        kappas, rets, taus = [], [], []
        for curve, _ in base.CURVES:
            row = next(r for r in subset if r["train_curve"] == curve)
            kappas.append(float(row["kappa"]))
            rets.append(float(row["orth_feature_retention"]))
            taus.append(float(row["tau"]))
        axes[0].bar(x + (si - 1) * width, kappas, width=width, label=f"{scale}M")
        axes[1].bar(x + (si - 1) * width, rets, width=width, label=f"{scale}M")
        axes[2].bar(x + (si - 1) * width, taus, width=width, label=f"{scale}M")
    axes[0].set_title("final kappa")
    axes[1].set_title("retention")
    axes[2].set_title("EB tau")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=24, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def write_report(comp: list[dict[str, object]]) -> None:
    final_row = next(r for r in comp if r["estimator"] == "final_no_cap")
    spectral_row = next(r for r in comp if r["estimator"] == "final_spectral_G4_no_cap")
    capped = next(r for r in comp if r["estimator"] == "final_cap_0p03")
    no_cap = next(r for r in comp if r["estimator"] == "final_no_cap")
    lines = [
        "# Final Kappa Estimator\n\n",
        "This report consolidates the selected schedule-agnostic kappa estimator and the main baselines.\n\n",
        "See [`MANIFEST.md`](MANIFEST.md) for the artifact index and reproduction commands. See [`THEORY.md`](THEORY.md) for the assumptions, proposition-style derivation, empirical-Bayes interpretation, "
        "identifiable-amplitude conversion, and limitations of the final estimator. See [`PAPER_METHOD.md`](PAPER_METHOD.md) "
        "for a concise paper-ready method paragraph.\n\n",
        "## Formula\n\n",
        "```text\n",
        "r = observed_loss - MPL\n",
        "G = low-frequency MPL-residual nuisance subspace\n",
        "phi_perp = M_G phi,      r_perp = M_G r\n",
        "tau = sigma / k0         # leave-curve-out empirical Bayes prior/noise ratio\n",
        "retention = ||phi_perp||^2 / ||phi||^2\n",
        "kappa = sqrt(retention) * max(0, <phi_perp,r_perp> / (||phi_perp||^2 + tau^2))\n",
        "optional: kappa = min(kappa, 0.03)\n",
        "```\n\n",
        "The estimator is a Frisch-Waugh-Lovell partial-regression coefficient with an empirical-Bayes MAP denominator and an identifiable-amplitude conversion. "
        "It uses no schedule-family labels.\n\n",
        "## Main Comparison\n\n",
        "| estimator | worst offdiag | median offdiag | mean offdiag | cosine -> WSD | wsdcon_9 -> WSD | max cosine kappa | cap saturation |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in comp:
        if row["estimator"] == "numeric_oracle_deg1":
            continue
        lines.append(
            f"| `{row['estimator']}` | {float(row['worst_offdiag']):+.1f}% | {float(row['median_offdiag']):+.1f}% | "
            f"{float(row['mean_offdiag']):+.1f}% | {float(row['cosine_to_wsd']):+.1f}% | {float(row['wsdcon9_to_wsd']):+.1f}% | "
            f"{float(row['max_cosine_kappa']):.4f} | {100*float(row['cap_saturation_rate']):.1f}% |\n"
        )
    lines += [
        "\n## Final Matrix\n\n",
        "![matrix](figs/matrix_final_no_cap.png)\n\n",
        "![spectral matrix](figs/matrix_final_spectral_G4_no_cap.png)\n\n",
        "![diagnostics](figs/final_kappa_diagnostics.png)\n\n",
        "## Interpretation\n\n",
        f"The balanced spectral implementation gives worst off-diagonal change {float(spectral_row['worst_offdiag']):+.1f}% and cosine -> WSD {float(spectral_row['cosine_to_wsd']):+.1f}%. "
        f"It is slightly more conservative than the legacy smooth implementation ({float(no_cap['worst_offdiag']):+.1f}% and {float(no_cap['cosine_to_wsd']):+.1f}%), but it is useful because the nuisance basis is schedule-agnostic and non-polynomial. "
        f"The capped final estimator gives worst off-diagonal change {float(capped['worst_offdiag']):+.1f}% and cosine -> WSD {float(capped['cosine_to_wsd']):+.1f}%. "
        f"The legacy cap-free version gives worst off-diagonal change {float(no_cap['worst_offdiag']):+.1f}%, so the hard cap is not the mechanism preventing failure. "
        "The important control is the partial-regression residualization plus `sqrt(retention)`, which converts the response norm identified outside "
        "the nuisance subspace into a full-feature effective amplitude.\n\n",
        "The paper-facing formula should present the cap-free nuisance-projected EB estimator as the main estimator. "
        "`final_no_cap` is the strongest empirical implementation on the current matrix; `final_spectral_G4_no_cap` is the basis-neutral spectral robustness audit. "
        "The capped version is best described as an optional truncated-prior variant.\n\n",
        "## Additional Audits\n\n",
        "- Subset robustness: `../current_law_final_kappa_robustness/REPORT.md`.\n",
        "- Bootstrap uncertainty: `../current_law_final_kappa_bootstrap/REPORT.md`.\n",
        "- Retention exponent sweep: `../current_law_retention_power_audit/REPORT.md`.\n",
        "- Tau multiplier sweep: `../current_law_tau_sensitivity_audit/REPORT.md`.\n",
        "- Train-only tau audit: `../current_law_trainonly_tau_audit/REPORT.md`.\n",
        "- Multi-curve calibration: `../current_law_multicurve_kappa_audit/REPORT.md`.\n",
        "- Spectral nuisance-subspace audit: `../current_law_spectral_nuisance_audit/REPORT.md`; four-mode spectral `G` gives worst off-diagonal `-1.8%` and cosine-to-WSD `-3.6%`, while the automatic constrained retention-target rule gives worst `-1.7%` and cosine-to-WSD `-10.1%`.\n",
        "- Soft spectral nuisance-prior audit: `../current_law_soft_spectral_kappa_audit/REPORT.md`; a continuous DCT/Sobolev nuisance residualizer around `lambda=0.02--0.03` improves mean and cosine-to-WSD substantially but does not yet dominate the legacy worst-case result.\n",
        "- Soft spectral lambda-selection audit: `../current_law_soft_spectral_selection_audit/REPORT.md`; calibration-only GCV/BIC/retention rules do not recover the useful soft-prior strength, so fixed soft `lambda` remains exploratory rather than the main estimator.\n",
        "- Soft spectral multi-curve selection audit: `../current_law_soft_spectral_multicurve_selection_audit/REPORT.md`; with four or five calibration curves, fixed soft `lambda=0.025` becomes non-failing. Band-limited inner-CV improves small-train selection, but does not fully solve the worst small-train failures.\n",
        "- Predictive shrinkage audit: `../current_law_predictive_shrinkage_audit/REPORT.md`; applying a train-size posterior-predictive shrinkage `c_n = n/(n+0.5)` to the band-limited soft spectral kappa removes the observed WSD-con over-correction failures while preserving substantial cosine-to-WSD transfer.\n",
        "- Next-gen lambda stability audit: `../current_law_nextgen_lambda_stability_audit/REPORT.md`; all `rho=0.5` next-generation kappa rows stay inside the identifiable band `lambda in [0.01, 0.03]` (`186/186`), with median selected lambda `0.030`.\n",
        "- Next-gen rho margin audit: `../current_law_nextgen_rho_margin_audit/REPORT.md`; with the target gate fixed, `rho=0.40` is the first fully non-harming grid value and `rho=0.40` through `rho=2.00` preserve all `558/558` main-matrix wins. The selected `rho=0.50` lies inside this plateau, with mean `-5.9%`, worst `+0.0%`, and `1116/1116` non-harming cells.\n",
        "- Target-identifiability attenuation audit: `../current_law_target_identifiability_audit/REPORT.md`; applying the target-side gate `R_target(lambda) >= 0.01` to the next-generation estimator gives `1116/1116` non-harming cells across all calibration train sizes, with worst `+0.0%` and mean `-5.9%`.\n",
        "- Target-retention margin audit: `../current_law_target_retention_margin_audit/REPORT.md`; the chosen `0.01` threshold lies between the maximum raw-harmful retention `0.005721` and the minimum main-matrix retention `0.014797`, with `1.75x` lower-side and `1.48x` upper-side margins.\n",
        "- Next-gen component ablation audit: `../current_law_nextgen_component_ablation_audit/REPORT.md`; without predictive shrinkage the combined audit has worst `+32.6%`, with `rho=0.5` shrinkage the worst improves to `+22.5%`, and adding the `R_target(lambda) >= 0.01` gate gives `1116/1116` non-harming cells. This isolates shrinkage as finite-calibration amplitude control and target retention as the non-identifiable-target control.\n",
        "- Next-gen stress-slice audit: `../current_law_nextgen_stress_slice_audit/REPORT.md`; the safe formula has `0` slice failures across scale, train-size, target-curve, train-group, and scale-by-train-size checks, with every audited slice non-harming.\n",
        "- Next-gen deployment estimator audit: `../current_law_nextgen_deployment_audit/REPORT.md`; the reusable `NextGenKappaEstimator` reproduces the rho-margin reference exactly across `1116` rows, with max absolute delta, kappa, target-retention, lambda, and target-factor differences all `0.000e+00`.\n",
        "- Next-gen target-loss blindness audit: `../current_law_nextgen_target_loss_blindness_audit/REPORT.md`; replacing every target loss curve with deterministic fake losses changes max target retention and max `kappa_safe` by `0.000e+00` across `1116` rows, confirming target loss is used only for evaluation.\n",
        "- Next-gen scale-holdout constant audit: `../current_law_nextgen_scale_holdout_audit/REPORT.md`; holding out each model scale in turn, the `0.01` target-retention floor remains inside the two-scale margin (`3/3` splits) and selected `rho=0.50` stays on the safe side of the two-scale rho boundary (`3/3` splits), with every held-out scale `372/372` non-harming and `186/186` main wins.\n",
        "- Next-gen vs final audit: `../current_law_nextgen_vs_final_audit/REPORT.md`; on the common single-curve matrix, next-gen safe is comparable to `final_no_cap` in cell mean (`-12.0%` vs `-12.1%`) and has stronger scale-level non-harm (`90/90` vs `87/90`), but it does not strictly dominate the paper-facing estimator.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def write_paper_method(comp: list[dict[str, object]]) -> None:
    final_row = next(r for r in comp if r["estimator"] == "final_no_cap")
    spectral_row = next(r for r in comp if r["estimator"] == "final_spectral_G4_no_cap")
    no_ret = next(r for r in comp if r["estimator"] == "no_retention_cap_0p03")
    lines = [
        "# Paper-Ready Method Text\n\n",
        "This file contains a concise English version of the final `kappa` estimator for direct use in the paper. "
        "The longer derivation and audit references are in [`THEORY.md`](THEORY.md) and [`REPORT.md`](REPORT.md).\n\n",
        "## Method Paragraph\n\n",
        "We estimate the schedule-response amplitude with a nuisance-projected empirical-Bayes estimator. "
        "For a calibration curve, let `r = observed_loss - MPL` be the residual of the main power-law prediction, "
        "and let `phi` be the response feature induced by the proposed schedule-response law. We assume that the residual decomposes as\n\n",
        "```text\n",
        "r = kappa_* phi + g_* + eps,    g_* in G,\n",
        "```\n\n",
        "where `G` is a small low-frequency nuisance subspace that captures smooth MPL residual drift, and `kappa_* >= 0` "
        "is the schedule-response amplitude. Let `M_G` denote the orthogonal residualizer against `G`, and define\n\n",
        "```text\n",
        "phi_perp = M_G phi,\n",
        "r_perp = M_G r.\n",
        "```\n\n",
        "By the Frisch-Waugh-Lovell theorem, the coefficient of `phi` after controlling for `G` is estimated by regressing "
        "`r_perp` on `phi_perp`. We use a nonnegative empirical-Bayes MAP estimate with prior/noise ratio `tau = sigma / k0`, "
        "where `tau` is estimated in a leave-curve-out manner from the other calibration curves:\n\n",
        "```text\n",
        "kappa_MAP = max(0, <phi_perp, r_perp> / (||phi_perp||^2 + tau^2)).\n",
        "```\n\n",
        "The projected regression only identifies the response norm outside the nuisance subspace. To avoid extrapolating amplitude "
        "from the nuisance-confounded component of `phi`, we convert the identifiable response norm into a full-feature effective amplitude:\n\n",
        "```text\n",
        "R = ||phi_perp||^2 / ||phi||^2,\n",
        "kappa_hat = sqrt(R) * kappa_MAP.\n",
        "```\n\n",
        "Thus the final cap-free estimator is\n\n",
        "```text\n",
        "kappa_hat =\n",
        "sqrt(||M_G phi||^2 / ||phi||^2)\n",
        "* max(0, <M_G phi, M_G r> / (||M_G phi||^2 + tau^2)).\n",
        "```\n\n",
        "An optional capped variant imposes `kappa_hat <= 0.03`, which can be interpreted as a truncated susceptibility prior. "
        "Our main paper-facing estimator is cap-free because it does not require a hard upper bound. "
        "The strongest empirical implementation on the current matrix is the legacy smooth low-frequency basis. "
        "A balanced spectral nuisance subspace `G=G_4`, the span of the constant vector and the first four DCT modes, is retained as a basis-neutral robustness audit.\n\n",
        "## Symbol Table\n\n",
        "| symbol | meaning |\n",
        "|---|---|\n",
        "| `r` | residual after the main MPL prediction, `observed_loss - MPL` |\n",
        "| `phi` | schedule-response feature generated by the proposed law |\n",
        "| `kappa` | scalar response amplitude transferred across schedules |\n",
        "| `G` | low-frequency nuisance subspace for smooth MPL residual drift |\n",
        "| `M_G` | orthogonal residualizer against `G` |\n",
        "| `phi_perp` | identifiable response feature, `M_G phi` |\n",
        "| `r_perp` | nuisance-projected residual, `M_G r` |\n",
        "| `sigma` | residual noise scale estimated from calibration curves |\n",
        "| `k0` | empirical prior scale of response amplitudes |\n",
        "| `tau` | prior/noise ratio, `sigma / k0` |\n",
        "| `R` | identifiable feature-energy fraction, `||phi_perp||^2 / ||phi||^2` |\n\n",
        "## Short Result Statement\n\n",
        "On the current transfer matrix, the strongest cap-free estimator gives worst off-diagonal change "
        f"`{float(final_row['worst_offdiag']):+.1f}%`, mean off-diagonal change `{float(final_row['mean_offdiag']):+.1f}%`, "
        f"cosine-to-WSD change `{float(final_row['cosine_to_wsd']):+.1f}%`, and WSD-con 9e-5 to WSD change "
        f"`{float(final_row['wsdcon9_to_wsd']):+.1f}%`. The basis-neutral spectral cap-free audit gives worst off-diagonal change "
        f"`{float(spectral_row['worst_offdiag']):+.1f}%`, mean off-diagonal change `{float(spectral_row['mean_offdiag']):+.1f}%`, "
        f"cosine-to-WSD change `{float(spectral_row['cosine_to_wsd']):+.1f}%`, and WSD-con 9e-5 to WSD change "
        f"`{float(spectral_row['wsdcon9_to_wsd']):+.1f}%`. The capped legacy variant has the same worst off-diagonal result, showing that "
        "the hard cap is not the mechanism preventing amplitude failure. Removing the identifiable-amplitude conversion causes a positive worst "
        f"off-diagonal case (`{float(no_ret['worst_offdiag']):+.1f}%`) and makes cosine-derived `kappa` saturate at the cap, "
        "supporting the need for the `sqrt(R)` conversion.\n\n",
        "The same estimator extends to multiple calibration curves by summing the projected inner products and norms across the train set. "
        "In the multi-curve held-out audit, `tau` is estimated from training curves only, and median worst held-out change improves from "
        "`-1.0%` with one calibration curve to `-8.4%` with five calibration curves, showing that the formula can use additional "
        "calibration coverage without introducing schedule-family labels.\n\n",
        "As a stricter single-curve check, replacing the previous other-curves EB `tau` with a train-only `tau` preserves the main result: "
        "worst off-diagonal remains `-2.7%`, mean off-diagonal remains `-12.1%`, and cosine-to-WSD changes from `-4.3%` to `-5.6%`. "
        "Thus the single-curve conclusion does not rely on using held-out test curves to set the regularization scale.\n\n",
        "In the Spectral nuisance-subspace audit, replacing the legacy smooth basis with a discrete-cosine low-frequency "
        "`G` preserves a non-failing transfer matrix: the four-mode spectral `G` gives worst off-diagonal `-1.8%` and cosine-to-WSD "
        "`-3.6%`. This supports the interpretation that the method is exploiting low-frequency residual control rather than a specific "
        "polynomial-shaped implementation of `G`. The spectral sweep also shows the bandwidth tradeoff: one or two modes under-cover MPL "
        "drift and can fail, while eight or more modes over-cover the response and become nearly conservative. A more automatic variant chooses "
        "the DCT bandwidth by targeting identifiable feature energy after enforcing `K_min=3`; with target `R=0.35`, this gives worst "
        "off-diagonal `-1.7%`, mean off-diagonal `-11.2%`, and cosine-to-WSD `-10.1%`. Without `K_min=3`, the same retention-target idea "
        "selects under-covered bandwidths and fails, so the defensible rule is two-stage: minimum low-frequency drift control first, then "
        "identifiable-energy targeting.\n\n",
        "As a next-generation extension, the soft spectral Predictive shrinkage audit adds a finite-calibration transfer correction after estimating the pooled amplitude. "
        "For `n` calibration curves, it uses `c_n = n/(n+0.5)` and transfers `c_n * kappa_hat`. This can be derived as a posterior-predictive shrinkage for applying a scalar amplitude learned from finitely many schedules to an unseen schedule. "
        "In the audit, this removes the WSD-con over-correction failures for one-, two-, and three-curve calibration sets (`-1.0%`, `-1.1%`, and `-1.2%` worst worst-heldout) while preserving substantial cosine-to-WSD transfer. "
        "A rho sensitivity sweep shows that `rho=0.25` is still unsafe, while `rho=0.5`, `0.75`, and `1.0` are all non-failing in the predictive-shrinkage audit. "
        "With the target-identifiability gate fixed, the rho-margin audit finds that `rho=0.40` is the first fully non-harming grid value and that `rho=0.40` through `rho=2.00` preserve all `558/558` main-matrix wins. "
        "`rho=0.5` is therefore not a knife-edge; it is a simple half-degree prior inside the stable safe range. "
        "We keep it as an extension rather than the main paper-facing estimator because its half-degree-of-freedom prior is promising but still needs a broader external validation set.\n\n",
        "## Suggested Paper Claim\n\n",
        "The estimator should be described as a theoretically motivated transfer amplitude estimator, not as a universal optimal estimator. "
        "The defensible claim is that partial regression removes low-frequency MPL residual drift, empirical-Bayes regularization prevents "
        "weak-feature instability, and identifiable-amplitude conversion avoids transferring response amplitude from parts of the feature "
        "that are confounded with nuisance drift.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "PAPER_METHOD.md").write_text("".join(lines), encoding="utf-8")


def write_nextgen_formula_card() -> None:
    lines = [
        "# Next-Gen Kappa Formula Card\n\n",
        "This is the compact, implementation-facing statement of the strongest current\n",
        "research `kappa` candidate. It is not the paper-facing main estimator; use\n",
        "`PAPER_METHOD.md` for the conservative paper claim.\n\n",
        "## Estimator\n\n",
        "For calibration curves `c in S`, define\n\n",
        "```text\n",
        "r_c = observed_loss_c - MPL_c\n",
        "phi_c = response_feature(schedule_c)\n",
        "```\n\n",
        "Use the soft DCT/Sobolev nuisance residualizer\n\n",
        "```text\n",
        "M_lambda y = y - Q (Q^T Q + lambda D)^(-1) Q^T y,\n",
        "D_jj = j^4, D_00 = 0,\n",
        "lambda in [0.01, 0.03].\n",
        "```\n\n",
        "Pool calibration evidence:\n\n",
        "```text\n",
        "dot_S = sum_c <M_lambda phi_c, M_lambda r_c>\n",
        "l2_S = sum_c ||M_lambda phi_c||^2\n",
        "full_l2_S = sum_c ||phi_c||^2\n",
        "n = |S|\n",
        "```\n\n",
        "The transferable amplitude is\n\n",
        "```text\n",
        "kappa_transfer\n",
        "  = [n / (n + 0.5)]\n",
        "    * sqrt(l2_S / full_l2_S)\n",
        "    * max(0, dot_S / (l2_S + tau^2)).\n",
        "```\n\n",
        "Apply the target-identifiability gate:\n\n",
        "```text\n",
        "R_target(lambda) = ||M_lambda phi_target||^2 / ||phi_target||^2\n",
        "a_target = 1{R_target(lambda) >= 0.01}\n",
        "kappa_safe = a_target * kappa_transfer\n",
        "```\n\n",
        "## Interpretation\n\n",
        "- `M_lambda` removes low-frequency MPL residual drift.\n",
        "- `sqrt(l2_S / full_l2_S)` converts the identifiable projected response norm\n",
        "  into a full-feature effective amplitude.\n",
        "- `n/(n+0.5)` is a weak posterior-predictive shrinkage for finite calibration\n",
        "  coverage.\n",
        "- `R_target(lambda) >= 0.01` requires the target response direction to remain\n",
        "  identifiable after the same nuisance residualization.\n",
        "- No schedule-family labels are used.\n\n",
        "## Current Evidence\n\n",
        "- Lambda stability audit: all `rho=0.5` next-generation kappa rows stay inside\n",
        "  the identifiable band `lambda in [0.01, 0.03]` (`186/186`), with median\n",
        "  selected lambda `0.030`.\n",
        "- Predictive shrinkage with `rho=0.5` is non-failing across train sizes in the\n",
        "  current matrix and preserves useful cosine-to-WSD transfer.\n",
        "- Rho margin audit with the target gate fixed finds a stable safe plateau:\n",
        "  `rho=0.40` is the first fully non-harming grid value, and `rho=0.40`\n",
        "  through `rho=2.00` preserve all `558/558` main-matrix wins.\n",
        "- Target-identifiability gating gives `1116/1116` non-harming cells across all\n",
        "  calibration train sizes, with worst `+0.0%` and mean `-5.9%`.\n",
        "- Target-retention margin audit places the chosen threshold inside the interval\n",
        "  `0.005721 < 0.01 < 0.014797`, with `1.75x` lower-side and `1.48x`\n",
        "  upper-side margins; `0.005` restores the `+22.5%` diffuse-cosine failure.\n",
        "- Component ablation isolates the two stabilizers: no predictive shrinkage has\n",
        "  worst `+32.6%`, `rho=0.5` shrinkage improves worst to `+22.5%`, and adding\n",
        "  the `R_target(lambda) >= 0.01` gate gives `1116/1116` non-harming cells.\n",
        "- Stress-slice audit finds `0` safe-formula slice failures across scale,\n",
        "  train-size, target-curve, train-group, and scale-by-train-size checks;\n",
        "  every audited slice remains non-harming.\n",
        "- Deployment estimator audit verifies the reusable `NextGenKappaEstimator`:\n",
        "  it reproduces the rho-margin reference exactly across `1116` rows, with\n",
        "  max absolute delta and kappa differences `0.000e+00`.\n",
        "- Target-loss blindness audit replaces every target loss curve with fake\n",
        "  losses and leaves `R_target`, the target gate, and `kappa_safe` unchanged\n",
        "  across `1116` rows; target loss is used only for evaluation.\n",
        "- Scale-holdout constant audit holds out each model scale in turn; `0.01`\n",
        "  remains inside the two-scale target-retention margin in `3/3` splits,\n",
        "  selected `rho=0.50` stays on the safe side in `3/3` splits, and every\n",
        "  held-out scale is `372/372` non-harming with `186/186` main wins.\n",
        "- Raw next-gen transfer fails the same all-train-size audit with worst\n",
        "  `+22.5%`, driven by diffuse `cosine_24000` targets.\n\n",
        "## Limitation\n\n",
        "This is the best current general-purpose `kappa` candidate in the worktree, but\n",
        "it remains a next-generation extension until validated on additional independent\n",
        "schedule families or runs.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "NEXTGEN_FORMULA_CARD.md").write_text("".join(lines), encoding="utf-8")


def write_nextgen_method() -> None:
    lines = [
        "# Next-Generation Kappa Formula Candidate\n\n",
        "This note consolidates the strongest current research candidate into one auditable formula. "
        "It is not yet the paper-facing main estimator because the fixed transfer prior needs external validation, but it is the most promising general `kappa` method found in the current worktree.\n\n",
        "## Formula\n\n",
        "For each calibration curve `c`, define the MPL residual and response feature\n\n",
        "```text\n",
        "r_c = observed_loss_c - MPL_c\n",
        "phi_c = response_feature(schedule_c)\n",
        "```\n\n",
        "Use a soft spectral nuisance residualizer. Let `Q` be the normalized DCT basis up to mode 12 and define\n\n",
        "```text\n",
        "A_lambda = (Q^T Q + lambda D)^(-1) Q^T,\n",
        "M_lambda y = y - Q A_lambda y,\n",
        "D_jj = j^4, D_00 = 0.\n",
        "```\n\n",
        "Select `lambda` by train-only inner-CV restricted to the identifiable band\n\n",
        "```text\n",
        "lambda in [0.01, 0.03].\n",
        "```\n\n",
        "For a train set `S` of `n` calibration curves, pool projected evidence:\n\n",
        "```text\n",
        "dot_S = sum_{c in S} <M_lambda phi_c, M_lambda r_c>\n",
        "l2_S = sum_{c in S} ||M_lambda phi_c||^2\n",
        "full_l2_S = sum_{c in S} ||phi_c||^2\n",
        "R_S = l2_S / full_l2_S\n",
        "```\n\n",
        "With empirical-Bayes `tau = sigma/k0`, estimate the identifiable amplitude\n\n",
        "```text\n",
        "kappa_pool = sqrt(R_S) * max(0, dot_S / (l2_S + tau^2)).\n",
        "```\n\n",
        "Finally apply posterior-predictive transfer shrinkage for finite calibration coverage:\n\n",
        "```text\n",
        "c_n = n / (n + 0.5)\n",
        "kappa_transfer = c_n * kappa_pool.\n",
        "```\n\n",
        "## Interpretation\n\n",
        "- `M_lambda` removes low-frequency MPL residual drift without hard schedule-family labels.\n",
        "- The band `lambda in [0.01, 0.03]` is an identifiable soft-prior region: weaker smoothing leaves WSD-con over-transfer, while stronger smoothing starts to remove response signal.\n",
        "- `sqrt(R_S)` converts the amplitude identified outside the nuisance subspace into a full-feature effective amplitude.\n",
        "- `c_n = n/(n+0.5)` is a posterior-predictive shrinkage factor from a scalar random-effects view of schedule transfer: finite calibration coverage should not be trusted as a fully population-level amplitude.\n\n",
        "## Proposition-Style Derivation\n\n",
        "Assume the following weak model for calibration curves `c in S`:\n\n",
        "```text\n",
        "r_c = kappa_c phi_c + g_c + eps_c,\n",
        "g_c in span(Q) approximately low-frequency,\n",
        "eps_c is zero-mean with approximately isotropic scale sigma,\n",
        "kappa_c = theta + u_c,  u_c is schedule-specific transfer variation.\n",
        "```\n\n",
        "The soft residualizer `M_lambda` is the MAP residualizer for a nuisance coefficient vector with a Sobolev-type Gaussian prior penalizing high DCT modes by `j^4`. "
        "Thus `M_lambda r_c` removes the low-frequency MPL drift component while preserving response directions not explained by the nuisance prior. "
        "Conditional on `lambda`, the pooled projected likelihood for a common amplitude is quadratic in `kappa`, with sufficient statistics `dot_S` and `l2_S`. "
        "Combining this likelihood with the nonnegative empirical-Bayes prior gives\n\n",
        "```text\n",
        "kappa_MAP,S = max(0, dot_S / (l2_S + tau^2)).\n",
        "```\n\n",
        "Only `l2_S` of the full response energy `full_l2_S` is identifiable after nuisance residualization, so the transferable full-feature amplitude is\n\n",
        "```text\n",
        "kappa_pool = sqrt(l2_S / full_l2_S) * kappa_MAP,S.\n",
        "```\n\n",
        "Finally, under the scalar random-effects layer `kappa_c = theta + u_c`, applying an amplitude learned from `n` calibration schedules to a new schedule introduces transfer variance. "
        "A conjugate Gaussian posterior-predictive mean has the shrinkage form `n/(n+rho)`; the current audit uses the conservative weak prior `rho=0.5`. "
        "Therefore the next-generation transfer amplitude is\n\n",
        "```text\n",
        "kappa_transfer = [n / (n + 0.5)] * sqrt(l2_S / full_l2_S) * max(0, dot_S / (l2_S + tau^2)).\n",
        "```\n\n",
        "The derivation uses no schedule-family label. Schedule information enters only through `phi_c`, the observed residual `r_c`, and the train-set size `n`.\n\n",
        "## Evidence\n\n",
        "Primary audit: `../current_law_predictive_shrinkage_audit/REPORT.md`.\n\n",
        "- The selected soft residualizer strengths stay inside the identifiable band: lambda stability audit reports `186/186` `rho=0.5` kappa rows with `lambda in [0.01, 0.03]`, median `0.030`.\n",
        "- Without predictive shrinkage, the band-limited soft spectral estimator has positive worst held-out failures for small train sets: `+13.2%`, `+5.6%`, and `+3.3%` for one-, two-, and three-curve calibration.\n",
        "- With `rho=0.5`, the same settings become non-failing: `-1.0%`, `-1.1%`, and `-1.2%`.\n",
        "- With the target-identifiability gate fixed, rho-margin audit reports `rho=0.40` as the first fully non-harming grid value; `rho=0.40` through `rho=2.00` preserve all `558/558` main-matrix wins, and selected `rho=0.50` has mean `-5.9%`, worst `+0.0%`, and `1116/1116` non-harming cells.\n",
        "- Core transfer remains useful: `rho=0.5` gives Cosine -> WSD sharp `-20.5%` with `3/3` wins, and WSD-con 9e-5 -> WSD sharp `-8.7%` with `3/3` wins.\n",
        "- The complete single-curve off-diagonal matrix has `30/30` improving cells, worst mean cell `-1.5%`, and mean off-diagonal `-12.0%`.\n",
        "- The scale-specific single-curve matrix has `90/90` improving cells across 25M, 100M, and 400M, with worst cell `-1.0%`.\n",
        "- Rho sensitivity supports the fixed prior: `rho=0.25` is still unsafe, `rho=0.35` is near the boundary, and `rho in {0.5, 0.75, 1.0}` is non-failing but larger values are more conservative.\n",
        "- A fully automatic train-only rho selector is currently unreliable on this small calibration matrix; it often chooses weak or zero shrinkage and reintroduces held-out failures.\n\n",
        "## Target Identifiability And External Holdout Limitation\n\n",
        "Additional repo curves not included in the main six-schedule matrix expose an important boundary condition. "
        "Raw next-gen transfer is unsafe on `cosine_24000` (mean `+7.2%`, worst `+21.8%`), while `constant_24000` and `constant_72000` are unaffected because their response feature is zero. "
        "`cosine_24000` is also one of the MPL baseline fitting curves, so this is not a clean independent benchmark, but it is a useful warning about diffuse target schedules.\n\n",
        "The more model-native target safety rule uses the same soft residualizer as the estimator. Define\n\n",
        "```text\n",
        "R_target(lambda) = ||M_lambda phi_target||^2 / ||phi_target||^2.\n",
        "```\n\n",
        "Then use an identifiability gate\n\n",
        "```text\n",
        "a_target = 0 if R_target(lambda) < 0.01,\n",
        "a_target = 1 otherwise,\n",
        "kappa_transfer_safe = a_target * kappa_transfer.\n",
        "```\n\n",
        "This gate abstains on target response directions whose energy is almost entirely removed by the nuisance residualizer. "
        "The theoretical interpretation is direct: if `M_lambda phi_target` has negligible energy, transferring a positive amplitude into `phi_target` is not identifiable apart from low-frequency MPL drift unless target residual evidence is available.\n\n",
        "The target-identifiability audit compares this rule to the older peak/mean schedule-localization gate across all calibration train sizes. "
        "Raw next-gen has `930/1116` non-harming cells and worst `+22.5%` because of `cosine_24000`. "
        "A retention gate with `R_target(lambda) >= 0.01` has `1116/1116` non-harming cells, worst `+0.0%`, and mean `-5.9%`; this is slightly stronger than the peak/mean gate (`-5.7%`) and is tied to the estimator's nuisance model. "
        "The train-size breakdown is non-harming for every calibration size: `144/144`, `315/315`, `360/360`, `225/225`, and `72/72` for train sizes one through five. "
        "Thresholds from `0.0075` through `0.015` give non-harming results on the current audit, while `0.005` is unsafe because it lets `cosine_24000` through. "
        "The threshold also has a margin interpretation: the lowest positive main-matrix target retention is `0.014797`, the highest positive diffuse extra-holdout retention is `0.005721`, and their geometric midpoint is `0.009201`, so `0.01` separates the two regimes on a log scale without using held-out loss values.\n\n",
        "The target-retention margin audit makes this separation explicit: the chosen floor is `1.75x` above the maximum raw-harmful retention and the nearest main-matrix target is `1.48x` above the floor. "
        "A threshold of `0.005` restores the `+22.5%` diffuse-cosine failure, while thresholds above the main cosine retention remain non-harming but begin dropping useful main-matrix transfers. "
        "This supports treating `0.01` as a margin-based identifiability floor rather than a loss-tuned optimum.\n\n",
        "The stress-slice audit checks that this conclusion is not hiding an aggregate-only failure. "
        "The safe formula has `1116/1116` non-harming rows overall and `0` slice failures across scale, train-size, target-curve, train-group, and scale-by-train-size summaries. "
        "Each scale has `372/372` non-harming rows, each train-size slice is non-harming (`144/144`, `315/315`, `360/360`, `225/225`, `72/72`), and all main-matrix targets improve while non-identifiable extra targets abstain.\n\n",
        "A purely train-relative target threshold is not as good on the current evidence. "
        "Weak relative gates such as `train_relative_gate_0p05` preserve transfer but still let the diffuse external cosine target through (`+22.5%` worst), while the first safe pure relative gate, `train_relative_gate_0p5`, is more conservative (mean `-5.4%`, `438/1116` wins). "
        "Adding the absolute floor back, for example `max(0.01, beta * R_train)`, restores safety, which indicates that the essential rule is the absolute target-identifiability floor rather than a threshold defined only relative to calibration curves.\n\n",
        "For comparison, the older peak/mean gate\n\n",
        "```text\n",
        "a_target = 0 if peak(phi_target) / mean(phi_target) < 2\n",
        "```\n\n",
        "also gives `1116/1116` non-harming cells and worst `+0.0%`, but it is less tightly connected to the residualized likelihood and has slightly weaker mean improvement. "
        "The retention gate should therefore be treated as the stronger current next-generation deployment rule, while both gates remain limitation-aware safety rules rather than evidence that raw next-gen transfer is universally safe.\n\n",
        "## Current Status\n\n",
        "This is the best current general-purpose `kappa` candidate, but it should be described as a next-generation extension rather than a final paper claim until validated on additional schedule families or independent runs. "
        "The main unresolved issue is not the response direction; it is estimating the population-transfer amplitude from limited calibration coverage.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "NEXTGEN_METHOD.md").write_text("".join(lines), encoding="utf-8")


def write_manifest(comp: list[dict[str, object]]) -> None:
    final_row = next(r for r in comp if r["estimator"] == "final_no_cap")
    spectral_row = next(r for r in comp if r["estimator"] == "final_spectral_G4_no_cap")
    lines = [
        "# Final Kappa Artifact Manifest\n\n",
        "This directory is the paper-facing entry point for the final `kappa` estimator. "
        "Use this manifest to avoid mixing the final method with earlier exploratory variants.\n\n",
        "## Final Estimator\n\n",
        "The paper-facing estimator is the cap-free nuisance-projected empirical-Bayes amplitude estimator. "
        "The strongest current implementation uses the legacy smooth low-frequency nuisance basis; the balanced spectral `G_4` version is the basis-neutral robustness audit.\n\n",
        "```text\n",
        "r = observed_loss - MPL\n",
        "G = low-frequency MPL-residual nuisance subspace\n",
        "phi_perp = M_G phi\n",
        "r_perp = M_G r\n",
        "tau = sigma / k0\n",
        "R = ||phi_perp||^2 / ||phi||^2\n\n",
        "kappa_hat =\n",
        "sqrt(R) * max(0, <phi_perp, r_perp> / (||phi_perp||^2 + tau^2))\n",
        "```\n\n",
        "The optional capped variant `kappa <= 0.03` should be described only as a truncated-prior variant. "
        "It is not the main mechanism.\n\n",
        "## Main Result\n\n",
        "From `comparison.csv`, the paper-facing `final_no_cap` row is:\n\n",
        "| metric | value |\n",
        "|---|---:|\n",
        f"| worst off-diagonal | `{float(final_row['worst_offdiag']):+.1f}%` |\n",
        f"| mean off-diagonal | `{float(final_row['mean_offdiag']):+.1f}%` |\n",
        f"| cosine -> WSD | `{float(final_row['cosine_to_wsd']):+.1f}%` |\n",
        f"| WSD-con 9e-5 -> WSD | `{float(final_row['wsdcon9_to_wsd']):+.1f}%` |\n",
        f"| max cosine kappa | `{float(final_row['max_cosine_kappa']):.4f}` |\n",
        f"| cap saturation | `{100 * float(final_row['cap_saturation_rate']):.1f}%` |\n\n",
        "The spectral `final_spectral_G4_no_cap` audit row is:\n\n",
        "| metric | value |\n",
        "|---|---:|\n",
        f"| worst off-diagonal | `{float(spectral_row['worst_offdiag']):+.1f}%` |\n",
        f"| mean off-diagonal | `{float(spectral_row['mean_offdiag']):+.1f}%` |\n",
        f"| cosine -> WSD | `{float(spectral_row['cosine_to_wsd']):+.1f}%` |\n",
        f"| WSD-con 9e-5 -> WSD | `{float(spectral_row['wsdcon9_to_wsd']):+.1f}%` |\n",
        f"| max cosine kappa | `{float(spectral_row['max_cosine_kappa']):.4f}` |\n",
        f"| cap saturation | `{100 * float(spectral_row['cap_saturation_rate']):.1f}%` |\n\n",
        "## Required Reading Order\n\n",
        "1. [`PAPER_METHOD.md`](PAPER_METHOD.md): concise paper-ready method text.\n",
        "2. [`REPORT.md`](REPORT.md): main matrix, comparison table, and audit links.\n",
        "3. [`THEORY.md`](THEORY.md): assumptions, proposition-style derivation, and limitations.\n",
        "4. [`NEXTGEN_FORMULA_CARD.md`](NEXTGEN_FORMULA_CARD.md): compact next-generation formula card for writing slides or notes.\n",
        "5. [`NEXTGEN_METHOD.md`](NEXTGEN_METHOD.md): full next-generation research candidate with predictive transfer shrinkage and target-identifiability gating.\n",
        "6. [`APPENDIX_LATEX.md`](APPENDIX_LATEX.md): LaTeX-ready appendix derivation.\n\n",
        "## Supporting Audits\n\n",
        "- Subset robustness: `../current_law_final_kappa_robustness/REPORT.md`\n",
        "- Bootstrap uncertainty: `../current_law_final_kappa_bootstrap/REPORT.md`\n",
        "- Retention exponent sweep: `../current_law_retention_power_audit/REPORT.md`\n",
        "- Tau multiplier sweep: `../current_law_tau_sensitivity_audit/REPORT.md`\n",
        "- Train-only tau audit: `../current_law_trainonly_tau_audit/REPORT.md`\n",
        "- Multi-curve calibration: `../current_law_multicurve_kappa_audit/REPORT.md`\n",
        "- Spectral nuisance-subspace audit: `../current_law_spectral_nuisance_audit/REPORT.md`\n",
        "- Soft spectral nuisance-prior audit: `../current_law_soft_spectral_kappa_audit/REPORT.md`\n",
        "- Soft spectral lambda-selection audit: `../current_law_soft_spectral_selection_audit/REPORT.md`\n",
        "- Soft spectral multi-curve selection audit: `../current_law_soft_spectral_multicurve_selection_audit/REPORT.md`\n",
        "- Predictive shrinkage audit: `../current_law_predictive_shrinkage_audit/REPORT.md`\n\n",
        "- Next-gen lambda stability audit: `../current_law_nextgen_lambda_stability_audit/REPORT.md`\n",
        "- Next-gen rho margin audit: `../current_law_nextgen_rho_margin_audit/REPORT.md`\n",
        "- Next-gen external holdout sanity audit: `../current_law_nextgen_external_holdout_audit/REPORT.md`\n\n",
        "- Next-gen target safety gate audit: `../current_law_nextgen_safety_gate_audit/REPORT.md`\n\n",
        "- Next-gen target-identifiability attenuation audit: `../current_law_target_identifiability_audit/REPORT.md`\n",
        "- Next-gen target-retention margin audit: `../current_law_target_retention_margin_audit/REPORT.md`\n",
        "- Next-gen component ablation audit: `../current_law_nextgen_component_ablation_audit/REPORT.md`\n",
        "- Next-gen stress-slice audit: `../current_law_nextgen_stress_slice_audit/REPORT.md`\n",
        "- Next-gen deployment estimator audit: `../current_law_nextgen_deployment_audit/REPORT.md`\n",
        "- Next-gen target-loss blindness audit: `../current_law_nextgen_target_loss_blindness_audit/REPORT.md`\n",
        "- Next-gen scale-holdout constant audit: `../current_law_nextgen_scale_holdout_audit/REPORT.md`\n",
        "- Next-gen vs final common-matrix audit: `../current_law_nextgen_vs_final_audit/REPORT.md`\n\n",
        "## Reproduction Commands\n\n",
        "Generation note: `current_law_final_kappa.py` regenerates the main CSVs, figures, `REPORT.md`, `PAPER_METHOD.md`, `NEXTGEN_FORMULA_CARD.md`, `NEXTGEN_METHOD.md`, and `MANIFEST.md`. "
        "`THEORY.md` and `APPENDIX_LATEX.md` are maintained derivation artifacts and are checked by `validate_final_kappa_artifacts.py`.\n\n",
        "Regenerate the main final artifacts:\n\n",
        "```bash\n",
        "python3 repro/current_law_final_kappa.py\n",
        "```\n\n",
        "Regenerate the train-only tau audit:\n\n",
        "```bash\n",
        "python3 repro/current_law_trainonly_tau_audit.py\n",
        "```\n\n",
        "Regenerate the multi-curve calibration audit:\n\n",
        "```bash\n",
        "python3 repro/current_law_multicurve_kappa_audit.py\n",
        "```\n\n",
        "Regenerate the spectral nuisance-subspace audit:\n\n",
        "```bash\n",
        "python3 repro/current_law_spectral_nuisance_audit.py\n",
        "```\n\n",
        "Regenerate the soft spectral nuisance-prior audit:\n\n",
        "```bash\n",
        "python3 repro/current_law_soft_spectral_kappa_audit.py\n",
        "```\n\n",
        "Regenerate the soft spectral lambda-selection audit:\n\n",
        "```bash\n",
        "python3 repro/current_law_soft_spectral_selection_audit.py\n",
        "```\n\n",
        "Regenerate the soft spectral multi-curve selection audit:\n\n",
        "```bash\n",
        "python3 repro/current_law_soft_spectral_multicurve_selection_audit.py\n",
        "```\n\n",
        "Regenerate the predictive shrinkage audit:\n\n",
        "```bash\n",
        "python3 repro/current_law_predictive_shrinkage_audit.py\n",
        "```\n\n",
        "Regenerate the next-gen lambda stability audit:\n\n",
        "```bash\n",
        "python3 repro/current_law_nextgen_lambda_stability_audit.py\n",
        "```\n\n",
        "Regenerate the next-gen rho margin audit:\n\n",
        "```bash\n",
        "python3 repro/current_law_nextgen_rho_margin_audit.py\n",
        "```\n\n",
        "Regenerate the next-gen external holdout sanity audit:\n\n",
        "```bash\n",
        "python3 repro/current_law_nextgen_external_holdout_audit.py\n",
        "```\n\n",
        "Regenerate the next-gen target safety gate audit:\n\n",
        "```bash\n",
        "python3 repro/current_law_nextgen_safety_gate_audit.py\n",
        "```\n\n",
        "Regenerate the next-gen target-identifiability attenuation audit:\n\n",
        "```bash\n",
        "python3 repro/current_law_target_identifiability_audit.py\n",
        "```\n\n",
        "Regenerate the next-gen target-retention margin audit:\n\n",
        "```bash\n",
        "python3 repro/current_law_target_retention_margin_audit.py\n",
        "```\n\n",
        "Regenerate the next-gen component ablation audit:\n\n",
        "```bash\n",
        "python3 repro/current_law_nextgen_component_ablation_audit.py\n",
        "```\n\n",
        "Regenerate the next-gen stress-slice audit:\n\n",
        "```bash\n",
        "python3 repro/current_law_nextgen_stress_slice_audit.py\n",
        "```\n\n",
        "Regenerate the next-gen deployment estimator audit:\n\n",
        "```bash\n",
        "python3 repro/current_law_nextgen_deployment_audit.py\n",
        "```\n\n",
        "Regenerate the next-gen target-loss blindness audit:\n\n",
        "```bash\n",
        "python3 repro/current_law_nextgen_target_loss_blindness_audit.py\n",
        "```\n\n",
        "Regenerate the next-gen scale-holdout constant audit:\n\n",
        "```bash\n",
        "python3 repro/current_law_nextgen_scale_holdout_audit.py\n",
        "```\n\n",
        "Regenerate the next-gen vs final common-matrix audit:\n\n",
        "```bash\n",
        "python3 repro/current_law_nextgen_vs_final_audit.py\n",
        "```\n\n",
        "Validate the final paper-facing artifacts:\n\n",
        "```bash\n",
        "python3 repro/validate_final_kappa_artifacts.py\n",
        "```\n\n",
        "Expected validator output:\n\n",
        "```text\n",
        "final kappa artifacts validated\n",
        "```\n\n",
        "## Do Not Use As Main Claim\n\n",
        "The following are useful diagnostics but should not be presented as the main method:\n\n",
        "- `numeric_oracle_deg1`: internal warning/diagnostic only.\n",
        "- `final_cap_0p03`: optional truncated-prior variant, not the main estimator.\n",
        "- Any degree-selection narrative for `G`: the theoretical object is the low-frequency nuisance subspace, not a polynomial fitting law.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "MANIFEST.md").write_text("".join(lines), encoding="utf-8")


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
    for estimator in [
        "smooth_cap",
        "eb_q75",
        "final_no_cap",
        "final_spectral_G4_no_cap",
        "final_cap_0p03",
        "final_spectral_G4_cap_0p03",
        "numeric_oracle_deg1",
    ]:
        plot_matrix(FIG_DIR / f"matrix_{estimator}.png", summary, estimator)
    plot_kappa_diagnostics(FIG_DIR / "final_kappa_diagnostics.png", kappa_rows)
    write_report(comp)
    write_paper_method(comp)
    write_nextgen_formula_card()
    write_nextgen_method()
    write_manifest(comp)
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    for row in comp:
        print(
            f"{row['estimator']:24s} worst={row['worst_offdiag']:+7.1f}% "
            f"mean={row['mean_offdiag']:+7.1f}% cos->wsd={row['cosine_to_wsd']:+7.1f}% "
            f"w9->wsd={row['wsdcon9_to_wsd']:+7.1f}% maxcosk={row['max_cosine_kappa']:.4f}"
        )


if __name__ == "__main__":
    main()
