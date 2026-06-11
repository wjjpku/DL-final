#!/usr/bin/env python3
"""MPL-tangent nuisance orthogonalization for DropRelaxS kappa.

Polynomial nuisance trends are useful, but the cleaner nuisance space is the
local tangent space of the baseline MPL predictor.  If the MPL residual is partly
caused by small parameter error, then

    loss - MPL(theta0) = kappa * phi + J(theta0) delta + eps

where J is the Jacobian of MPL predictions with respect to the MPL parameters.
By Frisch-Waugh-Lovell, the response amplitude should be estimated after
projecting both phi and the residual away from columns of J.
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


OUT_DIR = ROOT / "results" / "current_law_tangent_kappa_search"
FIG_DIR = OUT_DIR / "figs"

PARAM_NAMES = ["L0", "A", "alpha", "B", "C", "beta", "gamma"]
BASIS_VARIANTS = {
    "mpl_all7": list(range(7)),
    "mpl_core3": [0, 1, 2],
    "mpl_ld4": [3, 4, 5, 6],
    "mpl_no_L0": [1, 2, 3, 4, 5, 6],
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def standardize_columns(z: np.ndarray) -> np.ndarray:
    cols = []
    for j in range(z.shape[1]):
        col = z[:, j].astype(np.float64)
        col = col - float(np.mean(col))
        norm = float(np.linalg.norm(col))
        if norm > 1e-12:
            cols.append(col / norm)
    if not cols:
        return np.ones((z.shape[0], 1), dtype=np.float64)
    return np.column_stack(cols)


def mpl_jacobian(scale: str, curve_name: str, param_idx: list[int]) -> np.ndarray:
    curve = base.load_curve(scale, curve_name)
    p = np.array(base.MPL_PRECOMPUTED_INIT[scale], dtype=np.float64)
    pred0 = base.mpl_predict(p, curve)
    cols = [np.ones_like(pred0)]
    for idx in param_idx:
        step = 1e-4 * max(abs(float(p[idx])), 1.0)
        pp = p.copy()
        pm = p.copy()
        pp[idx] += step
        pm[idx] -= step
        if idx in [1, 3, 4, 5, 6]:
            pm[idx] = max(pm[idx], 1e-8)
        if idx == 2:
            pm[idx] = max(pm[idx], 1e-4)
        try:
            diff = (base.mpl_predict(pp, curve) - base.mpl_predict(pm, curve)) / (pp[idx] - pm[idx])
        except Exception:
            diff = (base.mpl_predict(pp, curve) - pred0) / step
        cols.append(diff)
    return standardize_columns(np.column_stack(cols))


def tangent_stats(scale: str, curve_name: str, feats, variant: str) -> dict[str, float]:
    stats = amp.enriched_stats(scale, curve_name, feats)
    curve = base.load_curve(scale, curve_name)
    phi = feats[(scale, curve_name)]
    resid = curve.loss - base.mpl_predict(base.MPL_PRECOMPUTED_INIT[scale], curve)
    z = mpl_jacobian(scale, curve_name, BASIS_VARIANTS[variant])
    phi_o = orth.residualize(phi, z)
    resid_o = orth.residualize(resid, z)
    phi_o2 = max(float(np.dot(phi_o, phi_o)), 1e-18)
    dot_o = float(np.dot(phi_o, resid_o))
    raw_o = max(0.0, dot_o / phi_o2)
    retention = float(phi_o2 / max(float(np.dot(phi, phi)), 1e-18))
    corr_denom = float(np.linalg.norm(phi_o) * np.linalg.norm(resid_o))
    corr_o = 0.0 if corr_denom <= 1e-18 else float(np.dot(phi_o, resid_o) / corr_denom)
    return {
        **stats,
        "basis": variant,
        "basis_dim": z.shape[1],
        "tangent_feature_l2": phi_o2,
        "tangent_projection_dot": dot_o,
        "tangent_raw_kappa": raw_o,
        "tangent_feature_retention": retention,
        "tangent_corr": corr_o,
        "tangent_resid_scale": amp.robust_scale(resid_o),
    }


def tangent_kappa(stats: dict[str, float], estimator: str, tau: float) -> float:
    if estimator == "current_smooth_cap":
        return base.estimate("smooth_weight_cap_0p03", stats)
    if estimator == "eb_q75":
        return eb.eb_kappa(stats, tau)
    if estimator == "poly_deg2":
        # Recompute using the already-tested polynomial nuisance degree 2 logic.
        denom = stats["orth_feature_l2"] + tau * tau
        k = max(0.0, stats["orth_projection_dot"] / max(denom, 1e-18))
        return min(k * math.sqrt(max(stats["orth_feature_retention"], 0.0)), 0.03)
    if estimator == "tangent_map":
        denom = stats["tangent_feature_l2"] + tau * tau
        return min(max(0.0, stats["tangent_projection_dot"] / max(denom, 1e-18)), 0.03)
    if estimator == "tangent_map_retention":
        denom = stats["tangent_feature_l2"] + tau * tau
        k = max(0.0, stats["tangent_projection_dot"] / max(denom, 1e-18))
        return min(k * math.sqrt(max(stats["tangent_feature_retention"], 0.0)), 0.03)
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

    estimators = ["current_smooth_cap", "eb_q75", "poly_deg2", "tangent_map", "tangent_map_retention"]
    variants = list(BASIS_VARIANTS)

    for variant in variants:
        tangent_cache = {
            (scale, curve): tangent_stats(scale, curve, feats, variant)
            for curve, _ in base.CURVES
            for scale in base.SCALES
        }
        poly_cache = {
            (scale, curve): orth.orthogonal_stats(scale, curve, feats, 2)
            for curve, _ in base.CURVES
            for scale in base.SCALES
        }
        for estimator in estimators:
            for train_curve, train_label in base.CURVES:
                pool = [r for r in all_base_rows if r["train_curve"] != train_curve]
                tau = eb.estimate_tau(pool, "q75")["tau"]
                for scale in base.SCALES:
                    if estimator == "poly_deg2":
                        stats = {**tangent_cache[(scale, train_curve)], **poly_cache[(scale, train_curve)]}
                    else:
                        stats = tangent_cache[(scale, train_curve)]
                    kappa = tangent_kappa(stats, estimator, tau)
                    kappa_rows.append(
                        {
                            "estimator": estimator,
                            "basis": variant,
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
                                "basis": variant,
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
    keys = sorted({(str(r["estimator"]), str(r["basis"])) for r in details})
    for estimator, basis in keys:
        for train_curve, train_label in base.CURVES:
            for test_curve, test_label in base.CURVES:
                subset = [
                    r
                    for r in details
                    if r["estimator"] == estimator
                    and r["basis"] == basis
                    and r["train_curve"] == train_curve
                    and r["test_curve"] == test_curve
                ]
                rows.append(
                    {
                        "estimator": estimator,
                        "basis": basis,
                        "name": f"{estimator}_{basis}",
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
    for estimator, basis in sorted({(str(r["estimator"]), str(r["basis"])) for r in summary}):
        sub = [r for r in summary if r["estimator"] == estimator and r["basis"] == basis and r["train_curve"] != r["test_curve"]]
        cos_wsd = next(
            r
            for r in summary
            if r["estimator"] == estimator
            and r["basis"] == basis
            and r["train_curve"] == "cosine_72000.csv"
            and r["test_curve"] == "wsd_20000_24000.csv"
        )
        w9_wsd = next(
            r
            for r in summary
            if r["estimator"] == estimator
            and r["basis"] == basis
            and r["train_curve"] == "wsdcon_9.csv"
            and r["test_curve"] == "wsd_20000_24000.csv"
        )
        rows.append(
            {
                "estimator": estimator,
                "basis": basis,
                "name": f"{estimator}_{basis}",
                "worst_offdiag": float(max(float(r["mean_delta_pct"]) for r in sub)),
                "median_offdiag": float(np.median([float(r["mean_delta_pct"]) for r in sub])),
                "mean_offdiag": float(np.mean([float(r["mean_delta_pct"]) for r in sub])),
                "cosine_to_wsd": float(cos_wsd["mean_delta_pct"]),
                "wsdcon9_to_wsd": float(w9_wsd["mean_delta_pct"]),
            }
        )
    rows.sort(key=lambda r: (max(float(r["worst_offdiag"]), 0.0), max(0.0, 10.0 + float(r["wsdcon9_to_wsd"])), float(r["mean_offdiag"])))
    return rows


def choose_cap_safe(comp, kappa_rows) -> dict[str, object]:
    for row in comp:
        rows = [
            kr
            for kr in kappa_rows
            if kr["estimator"] == row["estimator"]
            and kr["basis"] == row["basis"]
            and kr["train_curve"] == "cosine_72000.csv"
        ]
        if max(float(kr["kappa"]) for kr in rows) < 0.02 and float(row["wsdcon9_to_wsd"]) <= -15.0 and float(row["worst_offdiag"]) <= 0.1:
            return row
    return comp[0]


def plot_matrix(path: Path, summary: list[dict[str, object]], estimator: str, basis: str) -> None:
    labels = [label for _, label in base.CURVES]
    mat = np.full((len(base.CURVES), len(base.CURVES)), np.nan)
    wins = np.zeros_like(mat)
    for i, (train_curve, _) in enumerate(base.CURVES):
        for j, (test_curve, _) in enumerate(base.CURVES):
            row = next(
                r
                for r in summary
                if r["estimator"] == estimator
                and r["basis"] == basis
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
    ax.set_title(f"{estimator}, {basis}")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i,j]:+.1f}%\n{int(wins[i,j])}/3", ha="center", va="center", fontsize=8, weight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("MAE change vs MPL (negative is better)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def write_report(comp, kappa_rows) -> None:
    best_numeric = comp[0]
    best_defensible = choose_cap_safe(comp, kappa_rows)
    lines = [
        "# MPL-Tangent Kappa Search\n\n",
        "This experiment replaces polynomial nuisance trends with the local tangent space of the MPL predictor. "
        "It tests whether `kappa` should be estimated only from residual directions that cannot be explained by small MPL parameter perturbations.\n\n",
        "## Formula\n\n",
        "```text\n",
        "r = loss - MPL(theta0)\n",
        "J = d MPL(theta0) / d theta\n",
        "phi_perp = M_J phi,    r_perp = M_J r\n",
        "kappa_hat = min(0.03, max(0, <phi_perp,r_perp> / (||phi_perp||^2 + tau^2)))\n",
        "```\n\n",
        "This is the Frisch-Waugh-Lovell partial-regression estimator with an EB MAP prior. "
        "The nuisance directions now have a direct interpretation: they are exactly the loss-shape changes induced by local MPL parameter error.\n\n",
        "## Comparison\n\n",
        "| estimator | basis | worst offdiag | median offdiag | mean offdiag | cosine -> WSD | wsdcon_9 -> WSD |\n",
        "|---|---|---:|---:|---:|---:|---:|\n",
    ]
    for row in comp[:16]:
        lines.append(
            f"| `{row['estimator']}` | `{row['basis']}` | {float(row['worst_offdiag']):+.1f}% | "
            f"{float(row['median_offdiag']):+.1f}% | {float(row['mean_offdiag']):+.1f}% | "
            f"{float(row['cosine_to_wsd']):+.1f}% | {float(row['wsdcon9_to_wsd']):+.1f}% |\n"
        )
    lines += [
        "\n## Recommendation\n\n",
        f"Best numeric candidate: `{best_numeric['name']}`. Recommended cap-safe candidate: `{best_defensible['name']}`.\n\n",
        f"![matrix](figs/matrix_{best_defensible['name']}.png)\n\n",
        "If a tangent-basis candidate beats the polynomial nuisance formula without cap saturation, it should become the main paper formula. "
        "If not, the polynomial degree-2 nuisance formula remains preferable because it gives stronger transfer with simpler robustness.\n",
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
    matrix_rows = {(str(row["estimator"]), str(row["basis"]), str(row["name"])) for row in comp[:5]}
    matrix_rows.add((str(best_defensible["estimator"]), str(best_defensible["basis"]), str(best_defensible["name"])))
    for estimator, basis, name in sorted(matrix_rows):
        plot_matrix(FIG_DIR / f"matrix_{name}.png", summary, estimator, basis)
    write_report(comp, kappa_rows)
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    for row in comp[:16]:
        print(
            f"{row['name']:34s} worst={row['worst_offdiag']:+7.1f}% "
            f"median={row['median_offdiag']:+7.1f}% "
            f"cos->wsd={row['cosine_to_wsd']:+7.1f}% "
            f"w9->wsd={row['wsdcon9_to_wsd']:+7.1f}%"
        )


if __name__ == "__main__":
    main()
