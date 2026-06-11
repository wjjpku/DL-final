#!/usr/bin/env python3
"""Spectral nuisance-subspace audit for the final kappa estimator.

The paper-facing law only needs a low-frequency nuisance subspace G.  This
audit replaces the legacy polynomial implementation with a discrete-cosine
low-frequency basis and checks whether the final kappa transfer behavior is
stable under that more implementation-neutral definition of G.
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
import current_law_final_kappa as final  # noqa: E402
import current_law_orthogonal_kappa_search as orth  # noqa: E402


OUT_DIR = ROOT / "results" / "current_law_spectral_nuisance_audit"
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


def spectral_stats(scale: str, curve_name: str, feats, modes: int) -> dict[str, float]:
    stats = amp.enriched_stats(scale, curve_name, feats)
    curve = base.load_curve(scale, curve_name)
    phi = feats[(scale, curve_name)]
    resid = curve.loss - base.mpl_predict(base.MPL_PRECOMPUTED_INIT[scale], curve)
    z = dct_low_frequency_basis(len(curve.step), modes)
    phi_o = residualize(phi, z)
    resid_o = residualize(resid, z)
    phi_o2 = max(float(np.dot(phi_o, phi_o)), 1e-18)
    dot_o = float(np.dot(phi_o, resid_o))
    raw_o = max(0.0, dot_o / phi_o2)
    retention = float(phi_o2 / max(float(np.dot(phi, phi)), 1e-18))
    corr_denom = float(np.linalg.norm(phi_o) * np.linalg.norm(resid_o))
    corr_o = 0.0 if corr_denom <= 1e-18 else float(np.dot(phi_o, resid_o) / corr_denom)
    return {
        **stats,
        "nuisance_family": "dct_low_frequency",
        "nuisance_modes": modes,
        "orth_feature_l2": phi_o2,
        "orth_projection_dot": dot_o,
        "orth_raw_kappa": raw_o,
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


def retention_target_name(target: float, min_modes: int = 1) -> str:
    suffix = f"r{str(target).replace('.', 'p')}"
    if min_modes > 1:
        suffix += f"_mmin{min_modes}"
    return f"dct_retention_target_G_{suffix}"


def choose_retention_target_stats(
    scale: str,
    curve: str,
    stats_by_mode: dict[tuple[str, str, int], dict[str, float]],
    target: float,
    min_modes: int = 1,
) -> dict[str, float]:
    candidates = [stats_by_mode[(scale, curve, modes)] for modes in range(min_modes, 13)]
    chosen = min(
        candidates,
        key=lambda row: (
            abs(float(row["orth_feature_retention"]) - target),
            int(row["nuisance_modes"]),
        ),
    )
    return {
        **chosen,
        "nuisance_family": "dct_retention_target",
        "retention_target": target,
        "min_modes": min_modes,
        "selected_modes": int(chosen["nuisance_modes"]),
    }


def run() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    feats = base.feature_cache()
    base_rows = build_base_rows(feats)
    details: list[dict[str, object]] = []
    kappa_rows: list[dict[str, object]] = []

    spectral_cache = {
        (scale, curve, modes): spectral_stats(scale, curve, feats, modes)
        for curve, _ in base.CURVES
        for scale in base.SCALES
        for modes in range(1, 13)
    }
    configs = (
        [("legacy_smooth_G", 2)]
        + [("dct_low_frequency_G", modes) for modes in range(1, 13)]
        + [("dct_retention_target_G", (target, 1)) for target in (0.20, 0.25, 0.30, 0.35)]
        + [("dct_retention_target_G", (target, 3)) for target in (0.20, 0.25, 0.30, 0.35)]
    )
    for family, complexity in configs:
        if family == "legacy_smooth_G":
            stats_cache = {
                (scale, curve): {
                    **orth.orthogonal_stats(scale, curve, feats, complexity),
                    "nuisance_family": family,
                    "nuisance_modes": complexity,
                }
                for curve, _ in base.CURVES
                for scale in base.SCALES
            }
            estimator = f"{family}_m{complexity}"
        else:
            if family == "dct_low_frequency_G":
                modes = int(complexity)
                stats_cache = {
                    (scale, curve): spectral_cache[(scale, curve, modes)]
                    for curve, _ in base.CURVES
                    for scale in base.SCALES
                }
                estimator = f"{family}_m{complexity}"
            elif family == "dct_retention_target_G":
                target, min_modes = complexity
                target = float(target)
                min_modes = int(min_modes)
                stats_cache = {
                    (scale, curve): choose_retention_target_stats(scale, curve, spectral_cache, target, min_modes)
                    for curve, _ in base.CURVES
                    for scale in base.SCALES
                }
                estimator = retention_target_name(target, min_modes)
            else:
                raise ValueError(family)

        for train_curve, train_label in base.CURVES:
            pool = [r for r in base_rows if r["train_curve"] != train_curve]
            tau = eb.estimate_tau(pool, "q75")["tau"]
            for scale in base.SCALES:
                stats = stats_cache[(scale, train_curve)]
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
    return details, kappa_rows


def summarize(details: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for estimator in sorted({str(r["estimator"]) for r in details}):
        for train_curve, train_label in base.CURVES:
            for test_curve, test_label in base.CURVES:
                subset = [
                    r
                    for r in details
                    if r["estimator"] == estimator and r["train_curve"] == train_curve and r["test_curve"] == test_curve
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
        cos_krows = [r for r in krows if r["train_curve"] == "cosine_72000.csv"]
        rows.append(
            {
                "estimator": estimator,
                "worst_offdiag": float(max(float(r["mean_delta_pct"]) for r in sub)),
                "median_offdiag": float(np.median([float(r["mean_delta_pct"]) for r in sub])),
                "mean_offdiag": float(np.mean([float(r["mean_delta_pct"]) for r in sub])),
                "cosine_to_wsd": float(cos_wsd["mean_delta_pct"]),
                "wsdcon9_to_wsd": float(w9_wsd["mean_delta_pct"]),
                "max_kappa": float(max(float(r["kappa"]) for r in krows)),
                "max_cosine_kappa": float(max(float(r["kappa"]) for r in cos_krows)),
                "mean_retention": float(np.mean([float(r["orth_feature_retention"]) for r in krows])),
            }
        )
    def order_key(row: dict[str, object]) -> tuple[int, int]:
        name = str(row["estimator"])
        if name == "legacy_smooth_G_m2":
            return (0, 0)
        if name.startswith("dct_low_frequency_G_m"):
            return (1, int(name.rsplit("m", 1)[1]))
        return (2, 0)

    rows.sort(key=order_key)
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
    im = ax.imshow(mat, cmap="RdYlGn_r", norm=TwoSlopeNorm(vmin=-60, vcenter=0, vmax=120))
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


def write_report(rows: list[dict[str, object]]) -> None:
    reference = next(r for r in rows if r["estimator"] == "dct_low_frequency_G_m4")
    adaptive = next(r for r in rows if r["estimator"] == "dct_retention_target_G_r0p35_mmin3")
    adaptive_bad = next(r for r in rows if r["estimator"] == "dct_retention_target_G_r0p35")
    undercovered = next(r for r in rows if r["estimator"] == "dct_low_frequency_G_m1")
    overcovered = next(r for r in rows if r["estimator"] == "dct_low_frequency_G_m12")
    legacy = next(r for r in rows if r["estimator"] == "legacy_smooth_G_m2")
    lines = [
        "# Spectral Nuisance-Subspace Audit\n\n",
        "This audit checks whether the final cap-free kappa estimator depends on the legacy smooth-basis implementation of `G`. ",
        "The alternative `G` is the span of the constant vector and the first few discrete-cosine low-frequency modes. ",
        "This is a more direct implementation of the paper-facing assumption that MPL residual drift is low frequency.\n\n",
        "| G implementation | worst offdiag | median offdiag | mean offdiag | cosine -> WSD | wsdcon_9 -> WSD | max cosine kappa | mean retention |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in rows:
        lines.append(
            f"| {row['estimator']} | {float(row['worst_offdiag']):+.1f}% | "
            f"{float(row['median_offdiag']):+.1f}% | {float(row['mean_offdiag']):+.1f}% | "
            f"{float(row['cosine_to_wsd']):+.1f}% | {float(row['wsdcon9_to_wsd']):+.1f}% | "
            f"{float(row['max_cosine_kappa']):.4f} | {float(row['mean_retention']):.3f} |\n"
        )
    lines.extend(
        [
            "\n![balanced spectral matrix](figs/matrix_balanced_spectral_G.png)\n\n",
            "![adaptive spectral matrix](figs/matrix_adaptive_spectral_G.png)\n\n",
            "## Readout\n\n",
            f"The balanced spectral reference is `{reference['estimator']}`: worst off-diagonal "
            f"{float(reference['worst_offdiag']):+.1f}%, cosine -> WSD {float(reference['cosine_to_wsd']):+.1f}%, and mean off-diagonal "
            f"{float(reference['mean_offdiag']):+.1f}%. ",
            f"The automatic constrained spectral rule `{adaptive['estimator']}` chooses the DCT bandwidth from the calibration feature by targeting "
            f"identifiable energy while enforcing `K_min=3`; it gives worst off-diagonal {float(adaptive['worst_offdiag']):+.1f}%, "
            f"cosine -> WSD {float(adaptive['cosine_to_wsd']):+.1f}%, and mean off-diagonal {float(adaptive['mean_offdiag']):+.1f}%. ",
            f"The current legacy implementation remains stronger on this matrix, with worst off-diagonal {float(legacy['worst_offdiag']):+.1f}% "
            f"and cosine -> WSD {float(legacy['cosine_to_wsd']):+.1f}%. ",
            f"The one-mode spectral `G` is under-covered and fails badly (worst {float(undercovered['worst_offdiag']):+.1f}%), while the "
            f"twelve-mode spectral `G` is over-covered and nearly erases the response (cosine -> WSD {float(overcovered['cosine_to_wsd']):+.1f}%). ",
            f"The unconstrained retention-target rule also fails (worst {float(adaptive_bad['worst_offdiag']):+.1f}%), confirming that retention alone cannot choose `G`. "
            "Thus the useful spectral window is not a polynomial artifact, but the estimator does require a nuisance bandwidth that removes MPL drift without absorbing the schedule-response feature.\n",
        ]
    )
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    details, kappa_rows = run()
    summary = summarize(details)
    rows = comparison(summary, kappa_rows)
    write_csv(OUT_DIR / "details.csv", details)
    write_csv(OUT_DIR / "summary.csv", summary)
    write_csv(OUT_DIR / "kappa_diagnostics.csv", kappa_rows)
    write_csv(OUT_DIR / "comparison.csv", rows)
    plot_matrix(FIG_DIR / "matrix_balanced_spectral_G.png", summary, "dct_low_frequency_G_m4")
    plot_matrix(FIG_DIR / "matrix_adaptive_spectral_G.png", summary, "dct_retention_target_G_r0p35_mmin3")
    write_report(rows)
    print("wrote", OUT_DIR)
    for row in rows:
        print(
            row["estimator"],
            f"worst={float(row['worst_offdiag']):+.1f}%",
            f"cos->wsd={float(row['cosine_to_wsd']):+.1f}%",
            f"mean={float(row['mean_offdiag']):+.1f}%",
        )


if __name__ == "__main__":
    main()
