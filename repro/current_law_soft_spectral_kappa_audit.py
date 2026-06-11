#!/usr/bin/env python3
"""Soft spectral nuisance audit for the final kappa estimator.

This audit replaces the hard low-frequency projection M_G with a soft
Tikhonov residualizer in a DCT basis.  It keeps the final kappa formula fixed:

    kappa = sqrt(R) * (<phi_perp, r_perp> / (||phi_perp||^2 + tau^2))_+

The only change is how the nuisance component is removed.  Instead of choosing
a discrete DCT bandwidth K, we fit a low-frequency nuisance drift with a
Sobolev-like penalty on higher DCT coefficients:

    min_a ||y - Qa||^2 + lambda * sum_j j^(2s) a_j^2.

The residual y_perp is y - Q a_hat.  Small lambda approaches hard projection;
large lambda approaches no nuisance removal.  This gives a continuous,
theory-facing alternative to hand-selecting a polynomial or DCT bandwidth.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
ROOT = REPO.parent
sys.path.insert(0, str(REPO))

import current_law_amplitude_kappa_search as amp  # noqa: E402
import current_law_continuous_kappa_search as base  # noqa: E402
import current_law_eb_kappa_search as eb  # noqa: E402
import current_law_final_kappa as final  # noqa: E402


OUT_DIR = ROOT / "results" / "current_law_soft_spectral_kappa_audit"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dct_basis(n: int, max_mode: int) -> np.ndarray:
    idx = np.arange(n, dtype=np.float64)
    cols = [np.ones(n, dtype=np.float64)]
    for k in range(1, max_mode + 1):
        cols.append(np.cos(math.pi * (idx + 0.5) * k / max(n, 1)))
    q = np.column_stack(cols)
    norms = np.linalg.norm(q, axis=0)
    return q / np.maximum(norms, 1e-12)


def soft_residualize(y: np.ndarray, q: np.ndarray, lam: float, smooth_order: int) -> np.ndarray:
    modes = np.arange(q.shape[1], dtype=np.float64)
    penalty = lam * np.power(modes, 2 * smooth_order)
    penalty[0] = 0.0
    lhs = q.T @ q + np.diag(penalty)
    rhs = q.T @ y
    coef = np.linalg.solve(lhs, rhs)
    return y - q @ coef


def stats_for(scale: str, curve_name: str, feats, max_mode: int, lam: float, smooth_order: int) -> dict[str, float]:
    stats = amp.enriched_stats(scale, curve_name, feats)
    curve = base.load_curve(scale, curve_name)
    phi = feats[(scale, curve_name)]
    resid = curve.loss - base.mpl_predict(base.MPL_PRECOMPUTED_INIT[scale], curve)
    q = dct_basis(len(curve.step), max_mode)
    phi_o = soft_residualize(phi, q, lam, smooth_order)
    resid_o = soft_residualize(resid, q, lam, smooth_order)
    phi_o2 = max(float(np.dot(phi_o, phi_o)), 1e-18)
    dot_o = float(np.dot(phi_o, resid_o))
    retention = float(phi_o2 / max(float(np.dot(phi, phi)), 1e-18))
    corr_denom = float(np.linalg.norm(phi_o) * np.linalg.norm(resid_o))
    corr_o = 0.0 if corr_denom <= 1e-18 else float(np.dot(phi_o, resid_o) / corr_denom)
    return {
        **stats,
        "nuisance_family": "soft_dct_sobolev",
        "max_mode": max_mode,
        "lambda": lam,
        "smooth_order": smooth_order,
        "orth_feature_l2": phi_o2,
        "orth_projection_dot": dot_o,
        "orth_raw_kappa": max(0.0, dot_o / phi_o2),
        "orth_feature_retention": retention,
        "orth_corr": corr_o,
        "orth_resid_scale": amp.robust_scale(resid_o),
    }


def base_rows(feats) -> list[dict[str, object]]:
    rows = []
    for curve, label in base.CURVES:
        for scale in base.SCALES:
            rows.append({"scale": scale, "train_curve": curve, "train_label": label, **amp.enriched_stats(scale, curve, feats)})
    return rows


def run() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    feats = base.feature_cache()
    rows_for_tau = base_rows(feats)
    max_mode = 12
    smooth_order = 2
    lambdas = [
        0.0,
        1e-8,
        3e-8,
        1e-7,
        3e-7,
        1e-6,
        3e-6,
        1e-5,
        3e-5,
        1e-4,
        3e-4,
        1e-3,
        3e-3,
        1e-2,
        1.5e-2,
        2e-2,
        2.5e-2,
        3e-2,
        4e-2,
        5e-2,
        7e-2,
        1e-1,
        3e-1,
        1.0,
    ]
    details: list[dict[str, object]] = []
    kappa_rows: list[dict[str, object]] = []

    stat_cache = {
        (scale, curve, lam): stats_for(scale, curve, feats, max_mode, lam, smooth_order)
        for curve, _ in base.CURVES
        for scale in base.SCALES
        for lam in lambdas
    }
    for lam in lambdas:
        estimator = f"soft_dct_m{max_mode}_s{smooth_order}_lam{lam:g}"
        for train_curve, train_label in base.CURVES:
            pool = [r for r in rows_for_tau if r["train_curve"] != train_curve]
            tau = eb.estimate_tau(pool, "q75")["tau"]
            for scale in base.SCALES:
                stats = stat_cache[(scale, train_curve, lam)]
                kappa = final.final_kappa(stats, tau, cap=None)
                kappa_rows.append(
                    {
                        "estimator": estimator,
                        "scale": scale,
                        "train_curve": train_curve,
                        "train_label": train_label,
                        "tau": tau,
                        "kappa": kappa,
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
                            **base.score(scale, test_curve, kappa, feats),
                        }
                    )

    summary = summarize(details)
    comparison = compare(summary, kappa_rows)
    return details, kappa_rows, comparison


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


def compare(summary: list[dict[str, object]], kappa_rows: list[dict[str, object]]) -> list[dict[str, object]]:
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
        krows = [r for r in kappa_rows if r["estimator"] == estimator]
        cosine_krows = [r for r in krows if r["train_curve"] == "cosine_72000.csv"]
        rows.append(
            {
                "estimator": estimator,
                "lambda": float(krows[0]["lambda"]),
                "worst_offdiag": float(max(float(r["mean_delta_pct"]) for r in sub)),
                "median_offdiag": float(np.median([float(r["mean_delta_pct"]) for r in sub])),
                "mean_offdiag": float(np.mean([float(r["mean_delta_pct"]) for r in sub])),
                "cosine_to_wsd": float(cos_wsd["mean_delta_pct"]),
                "wsdcon9_to_wsd": float(w9_wsd["mean_delta_pct"]),
                "max_kappa": float(max(float(r["kappa"]) for r in krows)),
                "max_cosine_kappa": float(max(float(r["kappa"]) for r in cosine_krows)),
                "mean_retention": float(np.mean([float(r["orth_feature_retention"]) for r in krows])),
            }
        )
    return sorted(rows, key=lambda r: float(r["lambda"]))


def write_report(comparison: list[dict[str, object]]) -> None:
    best_worst = min(comparison, key=lambda r: float(r["worst_offdiag"]))
    best_mean = min(comparison, key=lambda r: float(r["mean_offdiag"]))
    best_cos = min(comparison, key=lambda r: float(r["cosine_to_wsd"]))
    pareto = [
        r
        for r in comparison
        if float(r["worst_offdiag"]) <= -2.0
        and float(r["mean_offdiag"]) <= -12.0
        and float(r["cosine_to_wsd"]) <= -10.0
    ]
    best_pareto = min(pareto, key=lambda r: (float(r["mean_offdiag"]), float(r["worst_offdiag"]))) if pareto else None
    lines = [
        "# Soft Spectral Kappa Audit\n\n",
        "This audit keeps the final nuisance-projected EB kappa formula fixed and replaces hard low-frequency projection with a soft DCT/Sobolev nuisance residualizer.\n\n",
        "For each curve, the nuisance drift is fit by `min_a ||y-Qa||^2 + lambda sum_j j^4 a_j^2` using DCT modes 0--12. "
        "The residualized feature and MPL residual are then passed to the same `sqrt(R)` EB estimator. "
        "This is a continuous version of the low-frequency nuisance assumption, not a polynomial fit and not a schedule-family classifier.\n\n",
        "## Sweep\n\n",
        "| lambda | worst offdiag | mean offdiag | cosine -> WSD | wsdcon_9 -> WSD | max cosine kappa | mean retention |\n",
        "|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in comparison:
        lines.append(
            f"| {float(row['lambda']):.3g} | {float(row['worst_offdiag']):+.1f}% | "
            f"{float(row['mean_offdiag']):+.1f}% | {float(row['cosine_to_wsd']):+.1f}% | "
            f"{float(row['wsdcon9_to_wsd']):+.1f}% | {float(row['max_cosine_kappa']):.4f} | "
            f"{float(row['mean_retention']):.3f} |\n"
        )
    lines += [
        "\n## Readout\n\n",
        f"- Best worst-offdiagonal setting: `lambda={float(best_worst['lambda']):.3g}` with worst `{float(best_worst['worst_offdiag']):+.1f}%`, mean `{float(best_worst['mean_offdiag']):+.1f}%`, cosine-to-WSD `{float(best_worst['cosine_to_wsd']):+.1f}%`.\n",
        f"- Best mean-offdiagonal setting: `lambda={float(best_mean['lambda']):.3g}` with worst `{float(best_mean['worst_offdiag']):+.1f}%`, mean `{float(best_mean['mean_offdiag']):+.1f}%`, cosine-to-WSD `{float(best_mean['cosine_to_wsd']):+.1f}%`.\n",
        f"- Best cosine-to-WSD setting: `lambda={float(best_cos['lambda']):.3g}` with worst `{float(best_cos['worst_offdiag']):+.1f}%`, mean `{float(best_cos['mean_offdiag']):+.1f}%`, cosine-to-WSD `{float(best_cos['cosine_to_wsd']):+.1f}%`.\n",
    ]
    if best_pareto is not None:
        lines.append(
            f"- Best conservative Pareto candidate (`worst <= -2%`, `mean <= -12%`, `cosine -> WSD <= -10%`): "
            f"`lambda={float(best_pareto['lambda']):.3g}` with worst `{float(best_pareto['worst_offdiag']):+.1f}%`, "
            f"mean `{float(best_pareto['mean_offdiag']):+.1f}%`, cosine-to-WSD `{float(best_pareto['cosine_to_wsd']):+.1f}%`, "
            f"and max cosine kappa `{float(best_pareto['max_cosine_kappa']):.4f}`.\n"
        )
    lines += [
        "\nA useful main-method replacement would dominate the legacy smooth basis (`worst -2.7%`, `mean -12.1%`, `cosine -> WSD -4.3%`) without relying on a hard cap or family label. "
        "The soft spectral sweep does not yet dominate worst-case behavior, but it exposes a theoretically cleaner Pareto frontier: "
        "lambda around `0.02--0.03` improves mean and cosine-to-WSD substantially while keeping every off-diagonal cell non-failing. "
        "Above `0.04`, the method starts to over-transfer amplitude and produces positive failures.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    details, kappa_rows, comparison = run()
    summary = summarize(details)
    write_csv(OUT_DIR / "details.csv", details)
    write_csv(OUT_DIR / "summary.csv", summary)
    write_csv(OUT_DIR / "kappa_diagnostics.csv", kappa_rows)
    write_csv(OUT_DIR / "comparison.csv", comparison)
    write_report(comparison)
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    for row in comparison:
        print(
            f"lambda={float(row['lambda']):9.1e} worst={float(row['worst_offdiag']):+7.1f}% "
            f"mean={float(row['mean_offdiag']):+7.1f}% cos->wsd={float(row['cosine_to_wsd']):+7.1f}% "
            f"w9->wsd={float(row['wsdcon9_to_wsd']):+7.1f}% maxcosk={float(row['max_cosine_kappa']):.4f}"
        )


if __name__ == "__main__":
    main()
