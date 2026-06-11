#!/usr/bin/env python3
"""Train-family by test-family matrix for the current DropRelaxS law.

The law is fixed:

    prediction = MPL + kappa * DropRelaxS_lambda

This script asks whether the fitted correction transfers between three genuinely
different LR schedule families: smooth cosine decay, sharp WSD cooldown, and
two-stage WSD-con step-to-constant probes.  The main figure uses lambda=10 and
fits only kappa per model scale from the chosen train family.  Cells report
held-family MAE change relative to the cosine-fit MPL baseline.
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
from reproduce_cosine_to_wsd import (  # noqa: E402
    MPL_PRECOMPUTED_INIT,
    SCALES,
    load_curve,
    metrics,
    mpl_predict,
)
from deep_stime import stime_feature  # noqa: E402
from nonadiabatic_theory import fit_origin  # noqa: E402


LAMBDA = 10.0
FAMILIES: list[tuple[str, list[str]]] = [
    ("Cosine decay", ["cosine_72000.csv"]),
    ("WSD sharp", ["wsd_20000_24000.csv"]),
    ("WSD-con step", ["wsdcon_3.csv", "wsdcon_9.csv", "wsdcon_18.csv"]),
]


def feature_cache() -> dict[tuple[str, str], np.ndarray]:
    cache = {}
    all_names = sorted({name for _, names in FAMILIES for name in names})
    for scale in SCALES:
        for name in all_names:
            cache[(scale, name)] = stime_feature(load_curve(scale, name), LAMBDA)
    return cache


def fit_kappa(scale: str, train_names: list[str], feats) -> float:
    p = MPL_PRECOMPUTED_INIT[scale]
    xs, ys = [], []
    for name in train_names:
        curve = load_curve(scale, name)
        xs.append(feats[(scale, name)])
        ys.append(curve.loss - mpl_predict(p, curve))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    if float(np.dot(x, x)) <= 1e-18:
        return 0.0
    return max(0.0, fit_origin(x, y)[0])


def score_family(scale: str, test_names: list[str], kappa: float, feats) -> list[dict[str, object]]:
    rows = []
    p = MPL_PRECOMPUTED_INIT[scale]
    for name in test_names:
        curve = load_curve(scale, name)
        base = mpl_predict(p, curve)
        pred = base + kappa * feats[(scale, name)]
        base_mae = metrics(curve.loss, base)["mae"]
        corr_mae = metrics(curve.loss, pred)["mae"]
        rows.append(
            {
                "scale": scale,
                "test_curve": name,
                "kappa": kappa,
                "base_mae": base_mae,
                "corr_mae": corr_mae,
                "delta_pct": 100.0 * (corr_mae / base_mae - 1.0),
                "win": int(corr_mae < base_mae),
            }
        )
    return rows


def run_matrix() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    feats = feature_cache()
    summary, details = [], []
    for train_label, train_names in FAMILIES:
        for test_label, test_names in FAMILIES:
            cell_rows = []
            for scale in SCALES:
                kappa = fit_kappa(scale, train_names, feats)
                rows = score_family(scale, test_names, kappa, feats)
                for row in rows:
                    row["train_family"] = train_label
                    row["test_family"] = test_label
                    row["train_curves"] = "+".join(x.replace(".csv", "") for x in train_names)
                    cell_rows.append(row)
                    details.append(row)

            base = np.array([float(r["base_mae"]) for r in cell_rows])
            corr = np.array([float(r["corr_mae"]) for r in cell_rows])
            wins = sum(int(r["win"]) for r in cell_rows)
            summary.append(
                {
                    "train_family": train_label,
                    "test_family": test_label,
                    "lambda": LAMBDA,
                    "base_mae": float(base.mean()),
                    "corr_mae": float(corr.mean()),
                    "delta_pct": 100.0 * (float(corr.mean()) / float(base.mean()) - 1.0),
                    "wins": wins,
                    "tests": len(cell_rows),
                    "mean_kappa": float(np.mean([float(r["kappa"]) for r in cell_rows])),
                }
            )
    return summary, details


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_matrix(path_png: Path, path_pdf: Path, summary: list[dict[str, object]]) -> None:
    labels = [x[0] for x in FAMILIES]
    matrix = np.full((len(labels), len(labels)), np.nan)
    wins: dict[tuple[int, int], str] = {}
    for row in summary:
        i = labels.index(str(row["train_family"]))
        j = labels.index(str(row["test_family"]))
        matrix[i, j] = float(row["delta_pct"])
        wins[(i, j)] = f"{int(row['wins'])}/{int(row['tests'])}"

    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    norm = TwoSlopeNorm(vmin=-55, vcenter=0, vmax=100)
    im = ax.imshow(matrix, cmap="RdYlGn_r", norm=norm)

    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel("Test decay family")
    ax.set_ylabel("Calibration decay family")
    ax.set_title("DropRelaxS cross-schedule transfer (lambda=10)")
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right", rotation_mode="anchor")

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = matrix[i, j]
            color = "white" if value > 35 else "black"
            ax.text(
                j,
                i,
                f"{value:+.1f}%\n{wins[(i, j)]}",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=color,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("MAE change vs MPL (negative is better)")
    fig.subplots_adjust(left=0.22, right=0.86, top=0.90, bottom=0.24)
    fig.text(
        0.50,
        0.06,
        "Cell text: mean MAE change, wins/tests across 25M/100M/400M and selected curves.",
        ha="center",
        fontsize=8.5,
        color="#333333",
    )
    fig.savefig(path_png, dpi=220, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(path_pdf, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def write_report(path: Path, summary: list[dict[str, object]]) -> None:
    lines = [
        "# Current-Law Decay-Family Matrix\n\n",
        "Fixed law: `MPL + kappa * DropRelaxS_lambda`, with `lambda=10`. "
        "Only `kappa` is fitted per scale from the calibration schedule family. "
        "Cosine is included as a smooth-decay diagnostic, but note that the MPL backbone was "
        "originally fit on cosine curves.\n\n",
        "| train family | test family | MAE | vs MPL | wins |\n",
        "|---|---|---:|---:|---:|\n",
    ]
    for row in summary:
        lines.append(
            f"| {row['train_family']} | {row['test_family']} | "
            f"{float(row['corr_mae']):.5f} | {float(row['delta_pct']):+.1f}% | "
            f"{int(row['wins'])}/{int(row['tests'])} |\n"
        )
    lines += [
        "\n## Reading\n\n",
        "- Cosine calibration learns almost no positive lag amplitude because the MPL backbone was already fit on smooth cosine schedules.\n",
        "- WSD-con step probes transfer to sharp WSD targets with a smaller but stable gain; this is the clean no-target-WSD calibration regime.\n",
        "- Sharp WSD calibration does not transfer to WSD-con tails. This confirms that the law should not be presented as a universal residual smoother for every LR schedule family.\n",
    ]
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    out_dir = ROOT / "results" / "current_law_decay_matrix"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary, details = run_matrix()
    write_csv(out_dir / "summary.csv", summary)
    write_csv(out_dir / "details.csv", details)
    write_report(out_dir / "REPORT.md", summary)
    plot_matrix(out_dir / "decay_family_matrix.png", out_dir / "decay_family_matrix.pdf", summary)

    paper_fig = ROOT / "paper" / "figs" / "fig_decay_family_matrix"
    slides_fig = ROOT / "slides" / "figs" / "fig_decay_family_matrix"
    plot_matrix(paper_fig.with_suffix(".png"), paper_fig.with_suffix(".pdf"), summary)
    plot_matrix(slides_fig.with_suffix(".png"), slides_fig.with_suffix(".pdf"), summary)

    print(f"wrote {out_dir / 'summary.csv'}")
    print(f"wrote {out_dir / 'details.csv'}")
    print(f"wrote {out_dir / 'REPORT.md'}")
    print(f"wrote {out_dir / 'decay_family_matrix.png'}")
    for row in summary:
        print(
            f"{row['train_family']:15s} -> {row['test_family']:15s} "
            f"delta={float(row['delta_pct']):+6.1f}% "
            f"wins={int(row['wins'])}/{int(row['tests'])}"
        )


if __name__ == "__main__":
    main()
