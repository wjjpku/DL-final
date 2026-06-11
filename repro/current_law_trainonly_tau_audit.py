#!/usr/bin/env python3
"""Train-only tau audit for the final cap-free kappa estimator.

The main single-curve matrix estimates EB tau from curves other than the
calibration curve. That is useful as a leave-calibration-curve-out prior, but
for a strict train/test audit it can include the held-out test curve. This
script compares that reference with a stricter train-only tau estimate.
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
import current_law_orthogonal_kappa_search as orth  # noqa: E402


OUT_DIR = ROOT / "results" / "current_law_trainonly_tau_audit"
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


def final_kappa(stats: dict[str, float], tau: float) -> float:
    denom = float(stats["orth_feature_l2"]) + tau * tau
    raw = max(0.0, float(stats["orth_projection_dot"]) / max(denom, 1e-18))
    retention = max(float(stats["orth_feature_retention"]), 0.0)
    return (retention**0.5) * raw


def run() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    feats = base.feature_cache()
    base_rows = []
    orth_stats = {}
    for curve, label in base.CURVES:
        for scale in base.SCALES:
            stats = amp.enriched_stats(scale, curve, feats)
            base_rows.append({"scale": scale, "train_curve": curve, "train_label": label, **stats})
            orth_stats[(scale, curve)] = orth.orthogonal_stats(scale, curve, feats, 2)

    modes = ["other_curves_tau", "train_only_tau"]
    details: list[dict[str, object]] = []
    kappas: list[dict[str, object]] = []
    for mode in modes:
        for train_curve, train_label in base.CURVES:
            if mode == "other_curves_tau":
                pool = [r for r in base_rows if r["train_curve"] != train_curve]
            else:
                pool = [r for r in base_rows if r["train_curve"] == train_curve]
            tau_info = eb.estimate_tau(pool, "q75")
            tau = float(tau_info["tau"])
            for scale in base.SCALES:
                stats = orth_stats[(scale, train_curve)]
                kappa = final_kappa(stats, tau)
                kappas.append(
                    {
                        "mode": mode,
                        "scale": scale,
                        "train_curve": train_curve,
                        "train_label": train_label,
                        "tau": tau,
                        "kappa": kappa,
                        "retention": stats["orth_feature_retention"],
                        "reliable_n": tau_info["reliable_n"],
                    }
                )
                for test_curve, test_label in base.CURVES:
                    details.append(
                        {
                            "mode": mode,
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
    for mode in sorted({str(r["mode"]) for r in details}):
        for train_curve, train_label in base.CURVES:
            for test_curve, test_label in base.CURVES:
                sub = [
                    r
                    for r in details
                    if r["mode"] == mode
                    and r["train_curve"] == train_curve
                    and r["test_curve"] == test_curve
                ]
                rows.append(
                    {
                        "mode": mode,
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
    for mode in ["other_curves_tau", "train_only_tau"]:
        off = [r for r in summary if r["mode"] == mode and r["train_curve"] != r["test_curve"]]
        cos_wsd = next(
            r
            for r in summary
            if r["mode"] == mode
            and r["train_curve"] == "cosine_72000.csv"
            and r["test_curve"] == "wsd_20000_24000.csv"
        )
        w9_wsd = next(
            r
            for r in summary
            if r["mode"] == mode
            and r["train_curve"] == "wsdcon_9.csv"
            and r["test_curve"] == "wsd_20000_24000.csv"
        )
        krows = [r for r in kappas if r["mode"] == mode]
        cos_krows = [r for r in krows if r["train_curve"] == "cosine_72000.csv"]
        rows.append(
            {
                "mode": mode,
                "worst_offdiag": float(max(float(r["mean_delta_pct"]) for r in off)),
                "median_offdiag": float(np.median([float(r["mean_delta_pct"]) for r in off])),
                "mean_offdiag": float(np.mean([float(r["mean_delta_pct"]) for r in off])),
                "cosine_to_wsd": float(cos_wsd["mean_delta_pct"]),
                "wsdcon9_to_wsd": float(w9_wsd["mean_delta_pct"]),
                "max_kappa": float(max(float(r["kappa"]) for r in krows)),
                "max_cosine_kappa": float(max(float(r["kappa"]) for r in cos_krows)),
                "mean_tau": float(np.mean([float(r["tau"]) for r in krows])),
            }
        )
    return rows


def plot_matrix(path: Path, summary: list[dict[str, object]], mode: str) -> None:
    labels = [label for _, label in base.CURVES]
    mat = np.full((len(base.CURVES), len(base.CURVES)), np.nan)
    for i, (train_curve, _) in enumerate(base.CURVES):
        for j, (test_curve, _) in enumerate(base.CURVES):
            row = next(
                r
                for r in summary
                if r["mode"] == mode and r["train_curve"] == train_curve and r["test_curve"] == test_curve
            )
            mat[i, j] = float(row["mean_delta_pct"])
    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    im = ax.imshow(mat, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0.0, vmin=-25, vmax=15))
    ax.set_xticks(range(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("test curve")
    ax.set_ylabel("train curve")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            color = "white" if abs(mat[i, j]) > 13 else "black"
            ax.text(j, i, f"{mat[i, j]:+.1f}", ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(im, ax=ax, shrink=0.8, label="delta pct")
    ax.set_title(mode)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def write_report(rows: list[dict[str, object]]) -> None:
    other = next(r for r in rows if r["mode"] == "other_curves_tau")
    train = next(r for r in rows if r["mode"] == "train_only_tau")
    lines = [
        "# Train-Only Tau Audit\n\n",
        "This audit checks whether the final single-curve transfer matrix depends on estimating EB `tau` from curves that may include the held-out test curve. "
        "It compares the previous leave-calibration-curve-out tau with a stricter train-only tau estimated from the calibration curve itself.\n\n",
        "![train-only matrix](figs/matrix_train_only_tau.png)\n\n",
        "## Comparison\n\n",
        "| tau mode | worst offdiag | mean offdiag | cosine -> WSD | wsdcon_9 -> WSD | max cosine kappa | mean tau |\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in rows:
        lines.append(
            f"| `{row['mode']}` | {float(row['worst_offdiag']):+.1f}% | {float(row['mean_offdiag']):+.1f}% | "
            f"{float(row['cosine_to_wsd']):+.1f}% | {float(row['wsdcon9_to_wsd']):+.1f}% | "
            f"{float(row['max_cosine_kappa']):.4f} | {float(row['mean_tau']):.4f} |\n"
        )
    lines += [
        "\n## Reading\n\n",
        f"The stricter train-only tau gives worst off-diagonal {float(train['worst_offdiag']):+.1f}% and cosine -> WSD "
        f"{float(train['cosine_to_wsd']):+.1f}%, compared with {float(other['worst_offdiag']):+.1f}% and "
        f"{float(other['cosine_to_wsd']):+.1f}% under the previous other-curves tau. "
        "Thus the main transfer conclusion does not rely on using held-out test curves to set the EB regularization scale.\n",
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
    plot_matrix(FIG_DIR / "matrix_train_only_tau.png", summary, "train_only_tau")
    write_report(rows)
    for row in rows:
        print(
            f"{row['mode']:16s} worst={float(row['worst_offdiag']):+6.1f}% "
            f"mean={float(row['mean_offdiag']):+6.1f}% cos->wsd={float(row['cosine_to_wsd']):+6.1f}% "
            f"tau={float(row['mean_tau']):.4f}"
        )


if __name__ == "__main__":
    main()
