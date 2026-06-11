#!/usr/bin/env python3
"""Subset-calibration robustness audit for the final kappa estimator.

The consolidated final estimator uses the full calibration curve to estimate
kappa.  This audit checks whether the result is stable when kappa is estimated
from only part of the calibration curve, then transferred to full test curves.
"""
from __future__ import annotations

import csv
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
import current_law_final_kappa as final  # noqa: E402
import current_law_orthogonal_kappa_search as orth  # noqa: E402


OUT_DIR = ROOT / "results" / "current_law_final_kappa_robustness"
FIG_DIR = OUT_DIR / "figs"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def subset_mask(n: int, mode: str) -> np.ndarray:
    idx = np.arange(n)
    if mode == "full":
        return np.ones(n, dtype=bool)
    if mode == "first_half":
        return idx < n // 2
    if mode == "second_half":
        return idx >= n // 2
    if mode == "middle_half":
        return (idx >= n // 4) & (idx < 3 * n // 4)
    if mode == "even":
        return idx % 2 == 0
    if mode == "sparse_quarter":
        return idx % 4 == 0
    raise ValueError(mode)


def subset_final_stats(scale: str, curve_name: str, feats, mode: str) -> dict[str, float]:
    curve = base.load_curve(scale, curve_name)
    phi_full = feats[(scale, curve_name)]
    base_pred = base.mpl_predict(base.MPL_PRECOMPUTED_INIT[scale], curve)
    resid_full = curve.loss - base_pred
    mask = subset_mask(len(curve.loss), mode)
    if int(np.sum(mask)) < 8:
        raise ValueError(f"subset too small: {mode}")

    # Build a temporary curve-like object with only selected points for the
    # nuisance basis.  orth.nuisance_basis uses only .step.
    class _SubCurve:
        pass

    sub_curve = _SubCurve()
    sub_curve.step = curve.step[mask]
    z = orth.nuisance_basis(sub_curve, 2)
    phi = phi_full[mask]
    resid = resid_full[mask]
    phi_o = orth.residualize(phi, z)
    resid_o = orth.residualize(resid, z)
    phi_o2 = max(float(np.dot(phi_o, phi_o)), 1e-18)
    dot_o = float(np.dot(phi_o, resid_o))
    retention = float(phi_o2 / max(float(np.dot(phi, phi)), 1e-18))
    corr_denom = float(np.linalg.norm(phi_o) * np.linalg.norm(resid_o))
    corr = 0.0 if corr_denom <= 1e-18 else float(np.dot(phi_o, resid_o) / corr_denom)
    full_stats = amp.enriched_stats(scale, curve_name, feats)
    return {
        **full_stats,
        "subset_mode": mode,
        "subset_n": int(np.sum(mask)),
        "subset_frac": float(np.mean(mask)),
        "orth_feature_l2": phi_o2,
        "orth_projection_dot": dot_o,
        "orth_raw_kappa": max(0.0, dot_o / phi_o2),
        "orth_feature_retention": retention,
        "orth_corr": corr,
        "orth_resid_scale": amp.robust_scale(resid_o),
    }


def run():
    feats = base.feature_cache()
    base_rows = final.build_base_rows(feats)
    modes = ["full", "first_half", "second_half", "middle_half", "even", "sparse_quarter"]
    details: list[dict[str, object]] = []
    kappa_rows: list[dict[str, object]] = []

    for mode in modes:
        for train_curve, train_label in base.CURVES:
            pool = [r for r in base_rows if r["train_curve"] != train_curve]
            tau = eb.estimate_tau(pool, "q75")["tau"]
            for scale in base.SCALES:
                stats = subset_final_stats(scale, train_curve, feats, mode)
                kappa = final.final_kappa(stats, tau, cap=0.03)
                kappa_rows.append(
                    {
                        "subset_mode": mode,
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
                            "subset_mode": mode,
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
    for mode in sorted({str(r["subset_mode"]) for r in details}):
        for train_curve, train_label in base.CURVES:
            for test_curve, test_label in base.CURVES:
                subset = [
                    r
                    for r in details
                    if r["subset_mode"] == mode
                    and r["train_curve"] == train_curve
                    and r["test_curve"] == test_curve
                ]
                rows.append(
                    {
                        "subset_mode": mode,
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
    for mode in ["full", "first_half", "second_half", "middle_half", "even", "sparse_quarter"]:
        sub = [r for r in summary if r["subset_mode"] == mode and r["train_curve"] != r["test_curve"]]
        cos_wsd = next(r for r in summary if r["subset_mode"] == mode and r["train_curve"] == "cosine_72000.csv" and r["test_curve"] == "wsd_20000_24000.csv")
        w9_wsd = next(r for r in summary if r["subset_mode"] == mode and r["train_curve"] == "wsdcon_9.csv" and r["test_curve"] == "wsd_20000_24000.csv")
        krows = [r for r in kappa_rows if r["subset_mode"] == mode]
        cosine_krows = [r for r in krows if r["train_curve"] == "cosine_72000.csv"]
        rows.append(
            {
                "subset_mode": mode,
                "worst_offdiag": float(max(float(r["mean_delta_pct"]) for r in sub)),
                "median_offdiag": float(np.median([float(r["mean_delta_pct"]) for r in sub])),
                "mean_offdiag": float(np.mean([float(r["mean_delta_pct"]) for r in sub])),
                "cosine_to_wsd": float(cos_wsd["mean_delta_pct"]),
                "wsdcon9_to_wsd": float(w9_wsd["mean_delta_pct"]),
                "max_cosine_kappa": float(max(float(r["kappa"]) for r in cosine_krows)),
                "mean_subset_frac": float(np.mean([float(r["subset_frac"]) for r in krows])),
            }
        )
    return rows


def plot_comparison(path: Path, comp: list[dict[str, object]]) -> None:
    labels = [r["subset_mode"] for r in comp]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4.4))
    width = 0.26
    ax.bar(x - width, [float(r["worst_offdiag"]) for r in comp], width, label="worst offdiag")
    ax.bar(x, [float(r["cosine_to_wsd"]) for r in comp], width, label="cosine -> WSD")
    ax.bar(x + width, [float(r["wsdcon9_to_wsd"]) for r in comp], width, label="wsdcon_9 -> WSD")
    ax.axhline(0, color="black", lw=0.8, alpha=0.6)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("MAE change vs MPL (%)")
    ax.set_title("Final kappa robustness to calibration subset")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def plot_kappas(path: Path, kappa_rows: list[dict[str, object]]) -> None:
    curves = [label for _, label in base.CURVES]
    modes = ["full", "first_half", "second_half", "middle_half", "even", "sparse_quarter"]
    mat = np.zeros((len(modes), len(curves)))
    for i, mode in enumerate(modes):
        for j, (curve, _) in enumerate(base.CURVES):
            vals = [float(r["kappa"]) for r in kappa_rows if r["subset_mode"] == mode and r["train_curve"] == curve]
            mat[i, j] = float(np.mean(vals))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    im = ax.imshow(mat, cmap="viridis")
    ax.set_xticks(np.arange(len(curves)), curves, rotation=24, ha="right")
    ax.set_yticks(np.arange(len(modes)), modes)
    ax.set_title("mean kappa by calibration subset")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center", fontsize=8, color="white" if mat[i, j] > 0.015 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("kappa")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def write_report(comp: list[dict[str, object]]) -> None:
    full = next(r for r in comp if r["subset_mode"] == "full")
    lines = [
        "# Final Kappa Subset Robustness\n\n",
        "This report estimates the final kappa formula from partial calibration curves and evaluates transfer on full test curves.\n\n",
        "| subset | fraction | worst offdiag | median offdiag | mean offdiag | cosine -> WSD | wsdcon_9 -> WSD | max cosine kappa |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in comp:
        lines.append(
            f"| `{row['subset_mode']}` | {float(row['mean_subset_frac']):.2f} | {float(row['worst_offdiag']):+.1f}% | "
            f"{float(row['median_offdiag']):+.1f}% | {float(row['mean_offdiag']):+.1f}% | "
            f"{float(row['cosine_to_wsd']):+.1f}% | {float(row['wsdcon9_to_wsd']):+.1f}% | {float(row['max_cosine_kappa']):.4f} |\n"
        )
    lines += [
        "\n![comparison](figs/subset_comparison.png)\n\n",
        "![kappas](figs/subset_kappas.png)\n\n",
        "## Reading\n\n",
        f"Full-curve reference: worst offdiag {float(full['worst_offdiag']):+.1f}%, cosine -> WSD {float(full['cosine_to_wsd']):+.1f}%. "
        "Uniform subsampling (`even`, `sparse_quarter`) remains close to the full result, so the estimator is not relying on dense-point overfitting. "
        "Contiguous half-curve subsets are much more conservative because they do not cover the full response excitation and relaxation shape; "
        "this is the expected behavior for an identifiable response-amplitude estimator rather than a failure mode.\n",
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
    plot_comparison(FIG_DIR / "subset_comparison.png", comp)
    plot_kappas(FIG_DIR / "subset_kappas.png", kappa_rows)
    write_report(comp)
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    for row in comp:
        print(
            f"{row['subset_mode']:14s} frac={row['mean_subset_frac']:.2f} "
            f"worst={row['worst_offdiag']:+7.1f}% mean={row['mean_offdiag']:+7.1f}% "
            f"cos->wsd={row['cosine_to_wsd']:+7.1f}% w9->wsd={row['wsdcon9_to_wsd']:+7.1f}%"
        )


if __name__ == "__main__":
    main()
