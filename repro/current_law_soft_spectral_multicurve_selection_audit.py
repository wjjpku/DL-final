#!/usr/bin/env python3
"""Train-only multi-curve lambda selection for soft spectral kappa.

Single-curve GCV/BIC selection fails because it optimizes residual smoothing,
not transfer amplitude.  This audit tests a more relevant calibration-only
criterion: when multiple calibration curves are available, choose the soft
spectral nuisance strength by leave-one-curve-out transfer inside the training
set, then evaluate only on held-out curves.
"""
from __future__ import annotations

import csv
import itertools
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


OUT_DIR = ROOT / "results" / "current_law_soft_spectral_multicurve_selection_audit"
LAMBDAS = [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 1.5e-2, 2e-2, 2.5e-2, 3e-2, 4e-2, 5e-2, 7e-2, 1e-1]
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


def train_name(curves: tuple[str, ...]) -> str:
    labels = {curve: label for curve, label in base.CURVES}
    return " + ".join(labels[c] for c in curves)


def dct_basis(n: int, max_mode: int) -> np.ndarray:
    idx = np.arange(n, dtype=np.float64)
    cols = [np.ones(n, dtype=np.float64)]
    for k in range(1, max_mode + 1):
        cols.append(np.cos(math.pi * (idx + 0.5) * k / max(n, 1)))
    q = np.column_stack(cols)
    norms = np.linalg.norm(q, axis=0)
    return q / np.maximum(norms, 1e-12)


def smoother_matrix(q: np.ndarray, lam: float) -> np.ndarray:
    modes = np.arange(q.shape[1], dtype=np.float64)
    penalty = lam * np.power(modes, 2 * SMOOTH_ORDER)
    penalty[0] = 0.0
    lhs = q.T @ q + np.diag(penalty)
    return np.linalg.solve(lhs, q.T)


def soft_residualize(y: np.ndarray, q: np.ndarray, a: np.ndarray) -> np.ndarray:
    return y - q @ (a @ y)


def stats_for(scale: str, curve_name: str, feats, lam: float) -> dict[str, float]:
    stats = amp.enriched_stats(scale, curve_name, feats)
    curve = base.load_curve(scale, curve_name)
    phi = feats[(scale, curve_name)]
    resid = curve.loss - base.mpl_predict(base.MPL_PRECOMPUTED_INIT[scale], curve)
    q = dct_basis(len(curve.step), MAX_MODE)
    a = smoother_matrix(q, lam)
    phi_o = soft_residualize(phi, q, a)
    resid_o = soft_residualize(resid, q, a)
    phi_o2 = max(float(np.dot(phi_o, phi_o)), 1e-18)
    dot_o = float(np.dot(phi_o, resid_o))
    retention = float(phi_o2 / max(float(np.dot(phi, phi)), 1e-18))
    return {
        **stats,
        "lambda": lam,
        "orth_feature_l2": phi_o2,
        "orth_projection_dot": dot_o,
        "orth_feature_retention": retention,
    }


def pooled_kappa(stats_rows: list[dict[str, float]], tau: float) -> dict[str, float]:
    dot = float(sum(float(r["orth_projection_dot"]) for r in stats_rows))
    l2 = float(sum(float(r["orth_feature_l2"]) for r in stats_rows))
    full_l2 = float(sum(float(r["feature_l2"]) for r in stats_rows))
    raw = max(0.0, dot / max(l2 + tau * tau, 1e-18))
    retention = l2 / max(full_l2, 1e-18)
    kappa = (max(retention, 0.0) ** 0.5) * raw
    return {
        "kappa": kappa,
        "raw_map": raw,
        "pooled_dot": dot,
        "pooled_orth_l2": l2,
        "pooled_full_l2": full_l2,
        "pooled_retention": retention,
    }


def base_tau_rows(feats) -> list[dict[str, object]]:
    rows = []
    for curve, label in base.CURVES:
        for scale in base.SCALES:
            rows.append({"scale": scale, "train_curve": curve, "train_label": label, **amp.enriched_stats(scale, curve, feats)})
    return rows


def inner_cv_score(
    train_curves: tuple[str, ...],
    lam: float,
    stats_cache: dict[tuple[str, str, float], dict[str, float]],
    tau_rows: list[dict[str, object]],
    feats,
) -> dict[str, float]:
    deltas = []
    kappas = []
    retentions = []
    for val_curve in train_curves:
        fit_curves = tuple(c for c in train_curves if c != val_curve)
        if not fit_curves:
            continue
        tau_pool = [r for r in tau_rows if r["train_curve"] in fit_curves]
        tau = eb.estimate_tau(tau_pool, "q75")["tau"]
        for scale in base.SCALES:
            rows = [stats_cache[(scale, c, lam)] for c in fit_curves]
            estimate = pooled_kappa(rows, tau)
            scored = base.score(scale, val_curve, float(estimate["kappa"]), feats)
            deltas.append(float(scored["delta_pct"]))
            kappas.append(float(estimate["kappa"]))
            retentions.append(float(estimate["pooled_retention"]))
    return {
        "inner_mean": float(np.mean(deltas)) if deltas else 0.0,
        "inner_worst": float(max(deltas)) if deltas else 0.0,
        "inner_median": float(np.median(deltas)) if deltas else 0.0,
        "inner_max_kappa": float(max(kappas)) if kappas else 0.0,
        "inner_mean_retention": float(np.mean(retentions)) if retentions else 0.0,
    }


def select_lambda(
    train_curves: tuple[str, ...],
    rule: str,
    stats_cache: dict[tuple[str, str, float], dict[str, float]],
    inner_score_cache: dict[tuple[tuple[str, ...], float], dict[str, float]],
    tau_rows: list[dict[str, object]],
    feats,
) -> tuple[float, dict[str, float]]:
    def score_for(lam: float) -> dict[str, float]:
        key = (train_curves, lam)
        if key not in inner_score_cache:
            inner_score_cache[key] = inner_cv_score(train_curves, lam, stats_cache, tau_rows, feats)
        return inner_score_cache[key]

    if rule == "fixed_lam0p025" or len(train_curves) == 1:
        return 0.025, score_for(0.025)
    scored = [(lam, score_for(lam)) for lam in LAMBDAS]
    if rule == "inner_cv_mean":
        return min(scored, key=lambda x: (x[1]["inner_mean"], x[1]["inner_worst"], x[0]))
    if rule == "inner_cv_worst":
        return min(scored, key=lambda x: (x[1]["inner_worst"], x[1]["inner_mean"], x[0]))
    if rule == "inner_cv_safe":
        safe = [(lam, s) for lam, s in scored if s["inner_worst"] <= 0 and s["inner_max_kappa"] <= 0.06]
        candidates = safe if safe else scored
        return min(candidates, key=lambda x: (x[1]["inner_worst"], x[1]["inner_mean"], x[0]))
    if rule == "inner_cv_band_mean":
        band = [(lam, s) for lam, s in scored if 0.01 <= lam <= 0.03]
        return min(band, key=lambda x: (x[1]["inner_mean"], x[1]["inner_worst"], abs(x[0] - 0.025)))
    if rule == "inner_cv_band_worst":
        band = [(lam, s) for lam, s in scored if 0.01 <= lam <= 0.03]
        return min(band, key=lambda x: (x[1]["inner_worst"], x[1]["inner_mean"], abs(x[0] - 0.025)))
    raise ValueError(rule)


def run():
    feats = base.feature_cache()
    curves = tuple(curve for curve, _ in base.CURVES)
    labels = {curve: label for curve, label in base.CURVES}
    tau_rows = base_tau_rows(feats)
    stats_cache = {
        (scale, curve, lam): stats_for(scale, curve, feats, lam)
        for curve, _ in base.CURVES
        for scale in base.SCALES
        for lam in LAMBDAS
    }
    rules = [
        "fixed_lam0p025",
        "inner_cv_mean",
        "inner_cv_worst",
        "inner_cv_safe",
        "inner_cv_band_mean",
        "inner_cv_band_worst",
    ]
    details: list[dict[str, object]] = []
    kappa_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    inner_score_cache: dict[tuple[tuple[str, ...], float], dict[str, float]] = {}

    for train_size in range(1, len(curves)):
        for train_curves in itertools.combinations(curves, train_size):
            heldout = [curve for curve in curves if curve not in train_curves]
            train_id = "|".join(train_curves)
            train_label = train_name(train_curves)
            for rule in rules:
                selected_lam, inner = select_lambda(train_curves, rule, stats_cache, inner_score_cache, tau_rows, feats)
                tau_pool = [r for r in tau_rows if r["train_curve"] in train_curves]
                tau = eb.estimate_tau(tau_pool, "q75")["tau"]
                selection_rows.append(
                    {
                        "rule": rule,
                        "train_id": train_id,
                        "train_label": train_label,
                        "train_size": train_size,
                        "selected_lambda": selected_lam,
                        **inner,
                    }
                )
                for scale in base.SCALES:
                    rows = [stats_cache[(scale, curve, selected_lam)] for curve in train_curves]
                    estimate = pooled_kappa(rows, tau)
                    kappa_rows.append(
                        {
                            "rule": rule,
                            "train_id": train_id,
                            "train_label": train_label,
                            "train_size": train_size,
                            "scale": scale,
                            "selected_lambda": selected_lam,
                            "tau": tau,
                            **estimate,
                        }
                    )
                    for test_curve in heldout:
                        scored = base.score(scale, test_curve, float(estimate["kappa"]), feats)
                        details.append(
                            {
                                "rule": rule,
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
    return details, kappa_rows, selection_rows


def summarize_subsets(details: list[dict[str, object]], kappa_rows: list[dict[str, object]], selection_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for rule in sorted({str(r["rule"]) for r in details}):
        for train_id in sorted({str(r["train_id"]) for r in details if r["rule"] == rule}):
            sub = [r for r in details if r["rule"] == rule and r["train_id"] == train_id]
            ksub = [r for r in kappa_rows if r["rule"] == rule and r["train_id"] == train_id]
            srow = next(r for r in selection_rows if r["rule"] == rule and r["train_id"] == train_id)
            rows.append(
                {
                    "rule": rule,
                    "train_id": train_id,
                    "train_label": sub[0]["train_label"],
                    "train_size": int(sub[0]["train_size"]),
                    "selected_lambda": float(srow["selected_lambda"]),
                    "inner_worst": float(srow["inner_worst"]),
                    "inner_mean": float(srow["inner_mean"]),
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
    return rows


def summarize_sizes(subsets: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for rule in [
        "fixed_lam0p025",
        "inner_cv_mean",
        "inner_cv_worst",
        "inner_cv_safe",
        "inner_cv_band_mean",
        "inner_cv_band_worst",
    ]:
        for train_size in sorted({int(r["train_size"]) for r in subsets if r["rule"] == rule}):
            sub = [r for r in subsets if r["rule"] == rule and int(r["train_size"]) == train_size]
            rows.append(
                {
                    "rule": rule,
                    "train_size": train_size,
                    "subset_count": len(sub),
                    "mean_worst_heldout": float(np.mean([float(r["worst_heldout"]) for r in sub])),
                    "median_worst_heldout": float(np.median([float(r["worst_heldout"]) for r in sub])),
                    "best_worst_heldout": float(min(float(r["worst_heldout"]) for r in sub)),
                    "worst_worst_heldout": float(max(float(r["worst_heldout"]) for r in sub)),
                    "mean_heldout": float(np.mean([float(r["mean_heldout"]) for r in sub])),
                    "median_lambda": float(np.median([float(r["selected_lambda"]) for r in sub])),
                }
            )
    return rows


def write_report(size_rows: list[dict[str, object]]) -> None:
    def row_for(rule: str, train_size: int) -> dict[str, object]:
        return next(r for r in size_rows if r["rule"] == rule and int(r["train_size"]) == train_size)

    fixed4 = row_for("fixed_lam0p025", 4)
    fixed5 = row_for("fixed_lam0p025", 5)
    mean3 = row_for("inner_cv_mean", 3)
    safe3 = row_for("inner_cv_safe", 3)
    safe5 = row_for("inner_cv_safe", 5)
    band_mean3 = row_for("inner_cv_band_mean", 3)
    band_mean4 = row_for("inner_cv_band_mean", 4)
    band_mean5 = row_for("inner_cv_band_mean", 5)
    lines = [
        "# Soft Spectral Multi-Curve Lambda-Selection Audit\n\n",
        "This audit chooses the soft DCT/Sobolev nuisance strength using only the calibration curves. "
        "For train subsets with at least two curves, `inner_cv_*` rules select lambda by leave-one-curve-out transfer inside the train set, then evaluate only on held-out curves. "
        "The fixed-lambda row is included as the soft-prior Pareto reference.\n\n",
        "## Train-Size Summary\n\n",
        "| rule | train curves | median worst heldout | best worst heldout | worst worst heldout | mean heldout | median lambda |\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in size_rows:
        lines.append(
            f"| `{row['rule']}` | {int(row['train_size'])} | {float(row['median_worst_heldout']):+.1f}% | "
            f"{float(row['best_worst_heldout']):+.1f}% | {float(row['worst_worst_heldout']):+.1f}% | "
            f"{float(row['mean_heldout']):+.1f}% | {float(row['median_lambda']):.3g} |\n"
        )
    lines += [
        "\n## Readout\n\n",
        "A successful automatic rule should approach the fixed soft-prior Pareto reference while using only train curves. "
        "If inner-CV chooses overly large lambda values, it is overfitting the calibration-transfer matrix; if it chooses tiny lambda values, it collapses back to hard projection and loses amplitude.\n",
        "\nThe fixed soft prior becomes stable when calibration coverage is broad enough: with four train curves it has median worst held-out "
        f"`{float(fixed4['median_worst_heldout']):+.1f}%` and worst worst-heldout `{float(fixed4['worst_worst_heldout']):+.1f}%`; "
        f"with five train curves it has median worst held-out `{float(fixed5['median_worst_heldout']):+.1f}%` and worst worst-heldout "
        f"`{float(fixed5['worst_worst_heldout']):+.1f}%`. Thus multi-curve coverage can make the soft-prior candidate non-failing.\n\n",
        "The inner-CV rules are not yet reliable automatic selectors. With three train curves, `inner_cv_mean` still has worst worst-heldout "
        f"`{float(mean3['worst_worst_heldout']):+.1f}%`, and even `inner_cv_safe` has `{float(safe3['worst_worst_heldout']):+.1f}%`. "
        f"Only at five train curves does `inner_cv_safe` become non-failing (`{float(safe5['worst_worst_heldout']):+.1f}%`).\n\n",
        "Restricting inner-CV to the empirically identifiable soft-prior band `0.01 <= lambda <= 0.03` is a useful correction but not a complete solution. "
        f"At three train curves, `inner_cv_band_mean` improves worst worst-heldout to `{float(band_mean3['worst_worst_heldout']):+.1f}%`; "
        f"at four and five train curves it is non-failing (`{float(band_mean4['worst_worst_heldout']):+.1f}%` and `{float(band_mean5['worst_worst_heldout']):+.1f}%`). "
        "The practical conclusion is that soft spectral kappa is a promising multi-curve candidate, and band-limited calibration is the best automatic selector tested here, "
        "but a universally reliable small-train lambda selector remains unresolved.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    details, kappas, selections = run()
    subsets = summarize_subsets(details, kappas, selections)
    sizes = summarize_sizes(subsets)
    write_csv(OUT_DIR / "details.csv", details)
    write_csv(OUT_DIR / "kappa_diagnostics.csv", kappas)
    write_csv(OUT_DIR / "selection.csv", selections)
    write_csv(OUT_DIR / "subset_summary.csv", subsets)
    write_csv(OUT_DIR / "train_size_summary.csv", sizes)
    write_report(sizes)
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    for row in sizes:
        print(
            f"{row['rule']:16s} n={int(row['train_size'])} median_worst={float(row['median_worst_heldout']):+6.1f}% "
            f"worst_worst={float(row['worst_worst_heldout']):+6.1f}% mean={float(row['mean_heldout']):+6.1f}% "
            f"medlam={float(row['median_lambda']):.3g}"
        )


if __name__ == "__main__":
    main()
