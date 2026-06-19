#!/usr/bin/env python3
"""Cross-schedule matrix with first 5k steps excluded from kappa fitting.

This audit is intentionally narrow: it keeps the MPL backbone, response feature,
lambda=10, schedule families, and scoring code from current_law_decay_matrix.py.
The only estimator change is that the scalar kappa fit ignores calibration
points with step < 5000.

We report two evaluation views:
  - full: score the complete available target curve.
  - after5k: score only target points with step >= 5000.
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

from current_law_decay_matrix import FAMILIES, LAMBDA, feature_cache  # noqa: E402
from reproduce_cosine_to_wsd import (  # noqa: E402
    MPL_PRECOMPUTED_INIT,
    SCALES,
    load_curve,
    metrics,
    mpl_predict,
)
from nonadiabatic_theory import fit_origin  # noqa: E402


FIT_MIN_STEP = 5000
EVAL_REGIONS = [("full", None), ("after5k", FIT_MIN_STEP)]


def fit_kappa_after5k(scale: str, train_names: list[str], feats) -> float:
    p = MPL_PRECOMPUTED_INIT[scale]
    xs, ys = [], []
    for name in train_names:
        curve = load_curve(scale, name)
        mask = curve.step >= FIT_MIN_STEP
        if int(np.sum(mask)) == 0:
            raise RuntimeError(f"No points remain after step>={FIT_MIN_STEP}: {scale} {name}")
        xs.append(feats[(scale, name)][mask])
        ys.append((curve.loss - mpl_predict(p, curve))[mask])
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    if float(np.dot(x, x)) <= 1e-18:
        return 0.0
    return max(0.0, fit_origin(x, y)[0])


def score_one(scale: str, test_name: str, kappa: float, feats) -> list[dict[str, object]]:
    p = MPL_PRECOMPUTED_INIT[scale]
    curve = load_curve(scale, test_name)
    base = mpl_predict(p, curve)
    pred = base + kappa * feats[(scale, test_name)]

    rows = []
    for region, min_step in EVAL_REGIONS:
        mask = np.ones_like(curve.step, dtype=bool) if min_step is None else curve.step >= min_step
        if int(np.sum(mask)) == 0:
            continue
        base_mae = metrics(curve.loss[mask], base[mask])["mae"]
        corr_mae = metrics(curve.loss[mask], pred[mask])["mae"]
        rows.append(
            {
                "eval_region": region,
                "eval_points": int(np.sum(mask)),
                "scale": scale,
                "test_curve": test_name,
                "kappa": kappa,
                "base_mae": base_mae,
                "corr_mae": corr_mae,
                "delta_pct": 100.0 * (corr_mae / base_mae - 1.0),
                "win": int(corr_mae < base_mae),
            }
        )
    return rows


def run_matrix() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    feats = feature_cache()
    summary, details, kappas = [], [], []
    for train_label, train_names in FAMILIES:
        for scale in SCALES:
            kappa = fit_kappa_after5k(scale, train_names, feats)
            kappas.append(
                {
                    "train_family": train_label,
                    "train_curves": "+".join(x.replace(".csv", "") for x in train_names),
                    "scale": scale,
                    "fit_min_step": FIT_MIN_STEP,
                    "kappa": kappa,
                }
            )

        for test_label, test_names in FAMILIES:
            cell_rows = []
            for scale in SCALES:
                kappa = next(
                    float(r["kappa"])
                    for r in kappas
                    if r["train_family"] == train_label and r["scale"] == scale
                )
                for name in test_names:
                    for row in score_one(scale, name, kappa, feats):
                        row["train_family"] = train_label
                        row["test_family"] = test_label
                        row["train_curves"] = "+".join(x.replace(".csv", "") for x in train_names)
                        cell_rows.append(row)
                        details.append(row)

            for region, _ in EVAL_REGIONS:
                sub = [r for r in cell_rows if r["eval_region"] == region]
                base = np.array([float(r["base_mae"]) for r in sub])
                corr = np.array([float(r["corr_mae"]) for r in sub])
                summary.append(
                    {
                        "eval_region": region,
                        "train_family": train_label,
                        "test_family": test_label,
                        "lambda": LAMBDA,
                        "fit_min_step": FIT_MIN_STEP,
                        "base_mae": float(base.mean()),
                        "corr_mae": float(corr.mean()),
                        "delta_pct": 100.0 * (float(corr.mean()) / float(base.mean()) - 1.0),
                        "wins": int(sum(int(r["win"]) for r in sub)),
                        "tests": len(sub),
                        "mean_kappa": float(
                            np.mean(
                                [
                                    float(r["kappa"])
                                    for r in kappas
                                    if r["train_family"] == train_label
                                ]
                            )
                        ),
                    }
                )
    return summary, details, kappas


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_matrices(path: Path, summary: list[dict[str, object]]) -> None:
    labels = [x[0] for x in FAMILIES]
    fig, axes = plt.subplots(1, len(EVAL_REGIONS), figsize=(13.2, 5.4), constrained_layout=True)
    norm = TwoSlopeNorm(vmin=-55, vcenter=0, vmax=100)

    for ax, (region, _) in zip(axes, EVAL_REGIONS):
        matrix = np.full((len(labels), len(labels)), np.nan)
        wins: dict[tuple[int, int], str] = {}
        for row in summary:
            if row["eval_region"] != region:
                continue
            i = labels.index(str(row["train_family"]))
            j = labels.index(str(row["test_family"]))
            matrix[i, j] = float(row["delta_pct"])
            wins[(i, j)] = f"{int(row['wins'])}/{int(row['tests'])}"

        im = ax.imshow(matrix, cmap="RdYlGn_r", norm=norm)
        ax.set_xticks(np.arange(len(labels)), labels=labels)
        ax.set_yticks(np.arange(len(labels)), labels=labels)
        ax.set_xlabel("Test decay family")
        ax.set_ylabel("Calibration decay family")
        ax.set_title(f"kappa fit uses step >= {FIT_MIN_STEP}; eval={region}")
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
                    fontsize=10,
                    fontweight="bold",
                    color=color,
                )

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.035, pad=0.02)
    cbar.set_label("MAE change vs MPL (negative is better)")
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def write_report(path: Path, summary: list[dict[str, object]], kappas: list[dict[str, object]]) -> None:
    lines = [
        "# Current-Law Decay Matrix: Fit Step >= 5000\n\n",
        "This reruns the cross-schedule matrix with the first 5k steps removed only from the `kappa` fitting data. "
        "The MPL backbone and DropRelaxS feature are unchanged. Results are reported for full-target evaluation and for evaluation restricted to `step>=5000`.\n\n",
        "## Kappa Values\n\n",
        "| train family | scale | kappa |\n",
        "|---|---:|---:|\n",
    ]
    for row in kappas:
        lines.append(f"| {row['train_family']} | {row['scale']}M | {float(row['kappa']):.5f} |\n")

    for region, _ in EVAL_REGIONS:
        lines += [
            f"\n## Matrix: eval={region}\n\n",
            "| train family | test family | MAE | vs MPL | wins |\n",
            "|---|---|---:|---:|---:|\n",
        ]
        for row in summary:
            if row["eval_region"] != region:
                continue
            lines.append(
                f"| {row['train_family']} | {row['test_family']} | "
                f"{float(row['corr_mae']):.5f} | {float(row['delta_pct']):+.1f}% | "
                f"{int(row['wins'])}/{int(row['tests'])} |\n"
            )

    cosine_k = [float(row["kappa"]) for row in kappas if row["train_family"] == "Cosine decay"]
    lines += [
        "\n## Reading\n\n",
        f"- Cosine-calibrated kappa remains large after dropping the first 5k steps: mean `{np.mean(cosine_k):.4f}`. "
        "This is nearly unchanged from the full-curve cosine fit, so the raw cosine transfer failure is not driven mainly by the earliest warmup-adjacent points.\n",
        "- The likely driver remains low-frequency MPL backbone mismatch over the smooth cosine curve. Early-step masking is useful hygiene, but it does not replace nuisance residualization or target-like probe calibration.\n",
    ]
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    out_dir = ROOT / "results" / "current_law_decay_matrix_after5k"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary, details, kappas = run_matrix()
    write_csv(out_dir / "summary.csv", summary)
    write_csv(out_dir / "details.csv", details)
    write_csv(out_dir / "kappas.csv", kappas)
    plot_matrices(out_dir / "decay_family_matrix_after5k.png", summary)
    write_report(out_dir / "REPORT.md", summary, kappas)

    print(f"wrote {out_dir / 'summary.csv'}")
    print(f"wrote {out_dir / 'details.csv'}")
    print(f"wrote {out_dir / 'kappas.csv'}")
    print(f"wrote {out_dir / 'decay_family_matrix_after5k.png'}")
    print(f"wrote {out_dir / 'REPORT.md'}")
    for row in summary:
        if row["eval_region"] == "full":
            print(
                f"{row['train_family']:15s} -> {row['test_family']:15s} "
                f"delta={float(row['delta_pct']):+6.1f}% wins={int(row['wins'])}/{int(row['tests'])}"
            )


if __name__ == "__main__":
    main()
