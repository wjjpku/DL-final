#!/usr/bin/env python3
"""Automatic lambda-selection audit for soft spectral kappa.

The previous soft spectral audit shows a useful Pareto frontier for fixed
lambda.  This script asks whether lambda can be selected from the calibration
curve itself, without test labels or schedule-family labels.

Candidate rules:
- residual_gcv: minimize GCV for smoothing the MPL residual.
- residual_bic: minimize a BIC-style residual smoother score.
- retention_r0p33: choose lambda whose identifiable feature retention is
  closest to 0.33.
- gcv_retention_band: minimize GCV subject to retention in [0.25, 0.36].
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


OUT_DIR = ROOT / "results" / "current_law_soft_spectral_selection_audit"


LAMBDAS = [
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
MAX_MODE = 12
SMOOTH_ORDER = 2


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


def smoother_parts(q: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    modes = np.arange(q.shape[1], dtype=np.float64)
    penalty = lam * np.power(modes, 2 * SMOOTH_ORDER)
    penalty[0] = 0.0
    lhs = q.T @ q + np.diag(penalty)
    inv_lhs = np.linalg.inv(lhs)
    a = inv_lhs @ q.T
    # H = Q (Q'Q + D)^-1 Q'; use trace(AB)=trace(BA).
    edf = float(np.trace((q.T @ q) @ inv_lhs))
    return a, edf


def soft_residualize_with_parts(y: np.ndarray, q: np.ndarray, a: np.ndarray) -> np.ndarray:
    return y - q @ (a @ y)


def smoother_scores(y: np.ndarray, q: np.ndarray, lam: float) -> dict[str, float]:
    a, edf = smoother_parts(q, lam)
    resid = soft_residualize_with_parts(y, q, a)
    n = len(y)
    rss = float(np.dot(resid, resid))
    denom = max(n - edf, 1e-9)
    gcv = rss / (denom * denom)
    bic = n * math.log(max(rss / max(n, 1), 1e-18)) + edf * math.log(max(n, 2))
    return {"gcv": gcv, "bic": bic, "edf": edf}


def stats_for(scale: str, curve_name: str, feats, lam: float) -> dict[str, float]:
    stats = amp.enriched_stats(scale, curve_name, feats)
    curve = base.load_curve(scale, curve_name)
    phi = feats[(scale, curve_name)]
    resid = curve.loss - base.mpl_predict(base.MPL_PRECOMPUTED_INIT[scale], curve)
    q = dct_basis(len(curve.step), MAX_MODE)
    a, edf = smoother_parts(q, lam)
    phi_o = soft_residualize_with_parts(phi, q, a)
    resid_o = soft_residualize_with_parts(resid, q, a)
    phi_o2 = max(float(np.dot(phi_o, phi_o)), 1e-18)
    dot_o = float(np.dot(phi_o, resid_o))
    retention = float(phi_o2 / max(float(np.dot(phi, phi)), 1e-18))
    corr_denom = float(np.linalg.norm(phi_o) * np.linalg.norm(resid_o))
    corr_o = 0.0 if corr_denom <= 1e-18 else float(np.dot(phi_o, resid_o) / corr_denom)
    scores = smoother_scores(resid, q, lam)
    return {
        **stats,
        "nuisance_family": "soft_dct_sobolev_auto",
        "max_mode": MAX_MODE,
        "lambda": lam,
        "smooth_order": SMOOTH_ORDER,
        "edf": edf,
        "gcv": scores["gcv"],
        "bic": scores["bic"],
        "orth_feature_l2": phi_o2,
        "orth_projection_dot": dot_o,
        "orth_raw_kappa": max(0.0, dot_o / phi_o2),
        "orth_feature_retention": retention,
        "orth_corr": corr_o,
        "orth_resid_scale": amp.robust_scale(resid_o),
    }


def select_lambda(rows: list[dict[str, float]], rule: str) -> dict[str, float]:
    if rule == "residual_gcv":
        return min(rows, key=lambda r: (float(r["gcv"]), float(r["lambda"])))
    if rule == "residual_bic":
        return min(rows, key=lambda r: (float(r["bic"]), float(r["lambda"])))
    if rule == "retention_r0p33":
        return min(rows, key=lambda r: (abs(float(r["orth_feature_retention"]) - 0.33), abs(float(r["lambda"]) - 0.025)))
    if rule == "gcv_retention_band":
        band = [r for r in rows if 0.25 <= float(r["orth_feature_retention"]) <= 0.36]
        candidates = band if band else rows
        return min(candidates, key=lambda r: (float(r["gcv"]), abs(float(r["orth_feature_retention"]) - 0.33)))
    if rule == "fixed_lam0p025":
        return min(rows, key=lambda r: abs(float(r["lambda"]) - 0.025))
    raise ValueError(rule)


def base_rows(feats) -> list[dict[str, object]]:
    rows = []
    for curve, label in base.CURVES:
        for scale in base.SCALES:
            rows.append({"scale": scale, "train_curve": curve, "train_label": label, **amp.enriched_stats(scale, curve, feats)})
    return rows


def run():
    feats = base.feature_cache()
    tau_rows = base_rows(feats)
    stat_cache = {
        (scale, curve, lam): stats_for(scale, curve, feats, lam)
        for curve, _ in base.CURVES
        for scale in base.SCALES
        for lam in LAMBDAS
    }
    rules = ["fixed_lam0p025", "residual_gcv", "residual_bic", "retention_r0p33", "gcv_retention_band"]
    details: list[dict[str, object]] = []
    kappa_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []

    for rule in rules:
        for train_curve, train_label in base.CURVES:
            pool = [r for r in tau_rows if r["train_curve"] != train_curve]
            tau = eb.estimate_tau(pool, "q75")["tau"]
            for scale in base.SCALES:
                candidates = [stat_cache[(scale, train_curve, lam)] for lam in LAMBDAS]
                selected = select_lambda(candidates, rule)
                kappa = final.final_kappa(selected, tau, cap=None)
                selection_rows.append(
                    {
                        "estimator": rule,
                        "scale": scale,
                        "train_curve": train_curve,
                        "train_label": train_label,
                        "selected_lambda": selected["lambda"],
                        "selected_retention": selected["orth_feature_retention"],
                        "selected_gcv": selected["gcv"],
                        "selected_bic": selected["bic"],
                        "selected_edf": selected["edf"],
                        "kappa": kappa,
                    }
                )
                kappa_rows.append(
                    {
                        "estimator": rule,
                        "scale": scale,
                        "train_curve": train_curve,
                        "train_label": train_label,
                        "tau": tau,
                        "kappa": kappa,
                        **selected,
                    }
                )
                for test_curve, test_label in base.CURVES:
                    details.append(
                        {
                            "estimator": rule,
                            "scale": scale,
                            "train_curve": train_curve,
                            "train_label": train_label,
                            "test_curve": test_curve,
                            "test_label": test_label,
                            "kappa": kappa,
                            **base.score(scale, test_curve, kappa, feats),
                        }
                    )
    return details, kappa_rows, selection_rows, summarize(details), compare(summarize(details), kappa_rows)


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
    order = ["fixed_lam0p025", "residual_gcv", "residual_bic", "retention_r0p33", "gcv_retention_band"]
    for estimator in order:
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
                "worst_offdiag": float(max(float(r["mean_delta_pct"]) for r in sub)),
                "median_offdiag": float(np.median([float(r["mean_delta_pct"]) for r in sub])),
                "mean_offdiag": float(np.mean([float(r["mean_delta_pct"]) for r in sub])),
                "cosine_to_wsd": float(cos_wsd["mean_delta_pct"]),
                "wsdcon9_to_wsd": float(w9_wsd["mean_delta_pct"]),
                "max_kappa": float(max(float(r["kappa"]) for r in krows)),
                "max_cosine_kappa": float(max(float(r["kappa"]) for r in cosine_krows)),
                "mean_retention": float(np.mean([float(r["orth_feature_retention"]) for r in krows])),
                "median_lambda": float(np.median([float(r["lambda"]) for r in krows])),
            }
        )
    return rows


def write_report(comparison: list[dict[str, object]], selection_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Soft Spectral Lambda-Selection Audit\n\n",
        "This audit tests whether the soft DCT/Sobolev nuisance prior can choose its smoothing strength from the calibration curve itself. "
        "No test curve labels or schedule-family labels are used in the selection rules.\n\n",
        "## Comparison\n\n",
        "| rule | worst offdiag | mean offdiag | cosine -> WSD | wsdcon_9 -> WSD | max cosine kappa | mean retention | median lambda |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in comparison:
        lines.append(
            f"| `{row['estimator']}` | {float(row['worst_offdiag']):+.1f}% | {float(row['mean_offdiag']):+.1f}% | "
            f"{float(row['cosine_to_wsd']):+.1f}% | {float(row['wsdcon9_to_wsd']):+.1f}% | "
            f"{float(row['max_cosine_kappa']):.4f} | {float(row['mean_retention']):.3f} | {float(row['median_lambda']):.3g} |\n"
        )
    lambdas_by_rule = {
        rule: [float(r["selected_lambda"]) for r in selection_rows if r["estimator"] == rule]
        for rule in sorted({str(r["estimator"]) for r in selection_rows})
    }
    lines += [
        "\n## Selection Behavior\n\n",
    ]
    for rule, values in lambdas_by_rule.items():
        lines.append(
            f"- `{rule}` selected lambda min/median/max = `{min(values):.3g}` / `{float(np.median(values)):.3g}` / `{max(values):.3g}`.\n"
        )
    lines += [
        "\n## Readout\n\n",
        "Pure residual GCV/BIC are calibration-only but may select smoothing strengths that optimize MPL-residual denoising rather than transfer-amplitude identifiability. "
        "The retention-target rule is also calibration-only, but uses the response feature geometry rather than the observed residual. "
        "The hybrid rule asks for both: a plausible identifiable-energy band and the best residual smoother inside that band.\n\n",
        "A rule can replace the current main method only if it is competitive with `final_no_cap` (`worst -2.7%`, `mean -12.1%`, `cosine -> WSD -4.3%`) without relying on test labels.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    details, kappa_rows, selection_rows, summary, comparison = run()
    write_csv(OUT_DIR / "details.csv", details)
    write_csv(OUT_DIR / "summary.csv", summary)
    write_csv(OUT_DIR / "kappa_diagnostics.csv", kappa_rows)
    write_csv(OUT_DIR / "selection.csv", selection_rows)
    write_csv(OUT_DIR / "comparison.csv", comparison)
    write_report(comparison, selection_rows)
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    for row in comparison:
        print(
            f"{row['estimator']:22s} worst={row['worst_offdiag']:+7.1f}% "
            f"mean={row['mean_offdiag']:+7.1f}% cos->wsd={row['cosine_to_wsd']:+7.1f}% "
            f"w9->wsd={row['wsdcon9_to_wsd']:+7.1f}% medlam={row['median_lambda']:.3g}"
        )


if __name__ == "__main__":
    main()
