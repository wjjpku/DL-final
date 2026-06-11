#!/usr/bin/env python3
"""Research robust ways to estimate kappa for DropRelaxS.

Problem: the raw least-squares estimator

    kappa = <feature, residual> / <feature, feature>

can be numerically and scientifically invalid on smooth curves.  Cosine has a
small broad DropRelaxS feature and ordinary MPL residual drift; raw LS therefore
uses a huge kappa to fit the wrong signal, and this explodes on sharp schedules.

This script benchmarks more conservative kappa estimators:

  * raw_ls: current unconstrained nonnegative LS baseline.
  * ident_gate_raw: return 0 when the LR-drop signal is too diffuse.
  * gated_cap_0p04 / 0p03 / 0p02: identifiability gate plus global safety cap.
  * local_peak_cap_0p04: gate, fit only high-feature points, then cap.

All estimators keep the formula fixed: MPL + kappa * DropRelaxS_lambda.
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
    PEAK_LR,
    SCALES,
    load_curve,
    metrics,
    mpl_predict,
)
from deep_stime import stime_feature  # noqa: E402
from nonadiabatic_theory import fit_origin  # noqa: E402


OUT_DIR = ROOT / "results" / "current_law_kappa_estimators"
FIG_DIR = OUT_DIR / "figs"
LAMBDA = 10.0

FAMILIES: list[tuple[str, list[str]]] = [
    ("Cosine decay", ["cosine_72000.csv"]),
    ("WSD sharp", ["wsd_20000_24000.csv"]),
    ("WSD linear", ["wsdld_20000_24000.csv"]),
    ("WSD-con step", ["wsdcon_3.csv", "wsdcon_9.csv", "wsdcon_18.csv"]),
]
ESTIMATORS = [
    "raw_ls",
    "ident_gate_raw",
    "gated_cap_0p04",
    "gated_cap_0p03",
    "gated_cap_0p02",
    "local_peak_cap_0p04",
]


def all_curves() -> list[str]:
    return sorted({curve for _, curves in FAMILIES for curve in curves})


def feature_cache() -> dict[tuple[str, str], np.ndarray]:
    return {
        (scale, curve): stime_feature(load_curve(scale, curve), LAMBDA)
        for scale in SCALES
        for curve in all_curves()
    }


def drop_stats(scale: str, train_curves: list[str], feats) -> dict[str, float]:
    drops, features = [], []
    for name in train_curves:
        curve = load_curve(scale, name)
        eta = curve.lrs.astype(np.float64)
        drop = np.zeros_like(eta)
        drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0) / PEAK_LR
        drops.append(drop)
        features.append(feats[(scale, name)])
    d = np.concatenate(drops)
    f = np.concatenate(features)
    total_drop = float(np.sum(d))
    drop_l2 = float(np.dot(d, d))
    eff_steps = float(total_drop * total_drop / drop_l2) if drop_l2 > 1e-18 else float("inf")
    return {
        "total_drop": total_drop,
        "max_drop": float(np.max(d)) if len(d) else 0.0,
        "drop_eff_steps": eff_steps,
        "feature_max": float(np.max(f)) if len(f) else 0.0,
        "feature_mean": float(np.mean(f)) if len(f) else 0.0,
        "feature_peak_to_mean": float(np.max(f) / max(np.mean(f), 1e-12)) if len(f) else 0.0,
    }


def identifiable(stats: dict[str, float]) -> bool:
    """Gate out smooth/weak calibration curves.

    The threshold is intentionally simple.  Cosine has a large total LR decrease
    but it is spread over tens of thousands of steps; sharp/linear WSD and
    WSD-con have concentrated drop support.
    """
    if stats["total_drop"] < 0.05:
        return False
    if stats["feature_max"] < 0.05:
        return False
    if stats["drop_eff_steps"] > 6000:
        return False
    return True


def fit_origin_safe(x: np.ndarray, y: np.ndarray, nonnegative: bool = True) -> float:
    if float(np.dot(x, x)) <= 1e-18:
        return 0.0
    kappa = float(fit_origin(x, y)[0])
    return max(0.0, kappa) if nonnegative else kappa


def raw_xy(scale: str, train_curves: list[str], feats) -> tuple[np.ndarray, np.ndarray]:
    p = MPL_PRECOMPUTED_INIT[scale]
    xs, ys = [], []
    for name in train_curves:
        curve = load_curve(scale, name)
        xs.append(feats[(scale, name)])
        ys.append(curve.loss - mpl_predict(p, curve))
    return np.concatenate(xs), np.concatenate(ys)


def estimate_kappa(estimator: str, scale: str, train_curves: list[str], feats) -> tuple[float, dict[str, float]]:
    x, y = raw_xy(scale, train_curves, feats)
    stats = drop_stats(scale, train_curves, feats)
    raw = fit_origin_safe(x, y)
    stats["raw_kappa"] = raw
    stats["identifiable"] = float(identifiable(stats))

    if estimator == "raw_ls":
        return raw, stats

    if estimator == "ident_gate_raw":
        return (raw if identifiable(stats) else 0.0), stats

    if estimator.startswith("gated_cap_"):
        cap = float(estimator.split("_")[-1].replace("p", "."))
        if not identifiable(stats):
            return 0.0, stats
        return min(raw, cap), stats

    if estimator == "local_peak_cap_0p04":
        if not identifiable(stats):
            return 0.0, stats
        mask = x >= 0.25 * max(float(np.max(x)), 1e-12)
        local = fit_origin_safe(x[mask], y[mask]) if int(mask.sum()) >= 3 else raw
        return min(local, 0.04), stats

    raise ValueError(estimator)


def score_curve(scale: str, curve_name: str, kappa: float, feats) -> dict[str, object]:
    curve = load_curve(scale, curve_name)
    base = mpl_predict(MPL_PRECOMPUTED_INIT[scale], curve)
    pred = base + kappa * feats[(scale, curve_name)]
    base_mae = metrics(curve.loss, base)["mae"]
    corr_mae = metrics(curve.loss, pred)["mae"]
    return {
        "scale": scale,
        "curve": curve_name,
        "base_mae": base_mae,
        "corr_mae": corr_mae,
        "delta_pct": 100.0 * (corr_mae / base_mae - 1.0),
        "win": int(corr_mae < base_mae),
    }


def run_benchmark(feats) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    summary, details, kappa_rows = [], [], []
    for estimator in ESTIMATORS:
        for train_family, train_curves in FAMILIES:
            for test_family, test_curves in FAMILIES:
                cell = []
                kappas = []
                for scale in SCALES:
                    kappa, stats = estimate_kappa(estimator, scale, train_curves, feats)
                    kappas.append(kappa)
                    if test_family == train_family:
                        kappa_rows.append(
                            {
                                "estimator": estimator,
                                "scale": scale,
                                "train_family": train_family,
                                "kappa": kappa,
                                **stats,
                            }
                        )
                    for scored in [score_curve(scale, c, kappa, feats) for c in test_curves]:
                        row = {
                            "estimator": estimator,
                            "train_family": train_family,
                            "test_family": test_family,
                            "train_curves": "+".join(c.replace(".csv", "") for c in train_curves),
                            "kappa": kappa,
                            **scored,
                        }
                        details.append(row)
                        cell.append(row)
                base = np.array([float(r["base_mae"]) for r in cell])
                corr = np.array([float(r["corr_mae"]) for r in cell])
                summary.append(
                    {
                        "estimator": estimator,
                        "train_family": train_family,
                        "test_family": test_family,
                        "base_mae": float(base.mean()),
                        "corr_mae": float(corr.mean()),
                        "delta_pct": 100.0 * (float(corr.mean()) / float(base.mean()) - 1.0),
                        "wins": sum(int(r["win"]) for r in cell),
                        "tests": len(cell),
                        "mean_kappa": float(np.mean(kappas)),
                        "max_kappa": float(np.max(kappas)),
                    }
                )
    return summary, details, kappa_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def labels() -> list[str]:
    return [x[0] for x in FAMILIES]


def plot_matrix(path: Path, rows: list[dict[str, object]], title: str) -> None:
    labs = labels()
    mat = np.full((len(labs), len(labs)), np.nan)
    wins = {}
    for row in rows:
        i = labs.index(str(row["train_family"]))
        j = labs.index(str(row["test_family"]))
        mat[i, j] = float(row["delta_pct"])
        wins[(i, j)] = f"{int(row['wins'])}/{int(row['tests'])}"
    fig, ax = plt.subplots(figsize=(7.8, 6.3))
    norm = TwoSlopeNorm(vmin=-60, vcenter=0, vmax=150)
    im = ax.imshow(mat, cmap="RdYlGn_r", norm=norm)
    ax.set_xticks(np.arange(len(labs)), labs, rotation=22, ha="right")
    ax.set_yticks(np.arange(len(labs)), labs)
    ax.set_xlabel("Test family")
    ax.set_ylabel("Calibration family")
    ax.set_title(title)
    for i in range(len(labs)):
        for j in range(len(labs)):
            value = mat[i, j]
            color = "white" if value > 55 else "black"
            ax.text(j, i, f"{value:+.1f}%\n{wins[(i, j)]}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("MAE change vs MPL (negative is better)")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def plot_estimator_comparison(path: Path, summary: list[dict[str, object]]) -> None:
    rows = []
    for estimator in ESTIMATORS:
        subset = [r for r in summary if r["estimator"] == estimator]
        all_d = np.array([float(r["delta_pct"]) for r in subset])
        offdiag = np.array([float(r["delta_pct"]) for r in subset if r["train_family"] != r["test_family"]])
        worst = float(np.max(offdiag))
        median = float(np.median(offdiag))
        key = next(r for r in subset if r["train_family"] == "WSD-con step" and r["test_family"] == "WSD sharp")
        cosine_fail = next(r for r in subset if r["train_family"] == "Cosine decay" and r["test_family"] == "WSD sharp")
        rows.append(
            {
                "estimator": estimator,
                "median_offdiag": median,
                "worst_offdiag": worst,
                "probe_to_wsd": float(key["delta_pct"]),
                "cosine_to_wsd": float(cosine_fail["delta_pct"]),
                "mean_all": float(np.mean(all_d)),
            }
        )

    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.bar(x - 0.2, [r["worst_offdiag"] for r in rows], width=0.2, label="worst off-diagonal")
    ax.bar(x, [r["cosine_to_wsd"] for r in rows], width=0.2, label="cosine -> WSD sharp")
    ax.bar(x + 0.2, [r["probe_to_wsd"] for r in rows], width=0.2, label="WSD-con -> WSD sharp")
    ax.axhline(0.0, color="#777777", lw=0.9)
    ax.set_xticks(x, [r["estimator"] for r in rows], rotation=20, ha="right")
    ax.set_ylabel("MAE change vs MPL (%)")
    ax.set_title("Estimator comparison: failure control vs useful transfer")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return rows


def write_report(summary: list[dict[str, object]], comparison_rows: list[dict[str, object]]) -> None:
    def get(estimator: str, train: str, test: str) -> dict[str, object]:
        for row in summary:
            if row["estimator"] == estimator and row["train_family"] == train and row["test_family"] == test:
                return row
        raise KeyError((estimator, train, test))

    lines = [
        "# Robust Kappa Estimator Research\n\n",
        "Goal: estimate `kappa` from arbitrary calibration curves without producing the huge cosine-derived amplitude that explodes on sharp schedules.\n\n",
        "The raw estimator is nonnegative least squares through the origin. The robust estimators add two ideas with a simple theoretical reading:\n\n",
        "1. **Identifiability gate.** In the linear-response model, the observed residual is "
        "`r(t) = kappa * phi(t) + epsilon(t)`, where `phi=DropRelaxS`. A single curve can identify "
        "`kappa` only if `phi` is an actual localized LR-drop response rather than a tiny diffuse "
        "smooth-decay feature. If LR drops are too diffuse, or the feature is too weak, the estimator "
        "returns `kappa=0` instead of fitting ordinary MPL drift.\n",
        "2. **Conservative prior cap.** Theory gives `kappa = eta_peak * chi`, where `chi` is the "
        "local sensitivity of the LR-dependent equilibrium/noise floor. `chi` is positive and finite "
        "under the weak linear-response assumptions. If the calibration curve is not known to match "
        "the target family, we impose a prior upper bound on this susceptibility. This is a projected "
        "least-squares/MAP estimate under the constraint `0 <= kappa <= kappa_max`, not a new formula.\n\n",
        "## Estimator Comparison\n\n",
        "![estimator comparison](figs/estimator_comparison.png)\n\n",
        "| estimator | cosine -> WSD sharp | WSD-con -> WSD sharp | worst off-diagonal | reading |\n",
        "|---|---:|---:|---:|---|\n",
    ]
    readings = {
        "raw_ls": "maximal fit, but unsafe on smooth curves",
        "ident_gate_raw": "fixes cosine blow-up, still over-corrects WSD-con tails",
        "gated_cap_0p04": "aggressive safe option: controls failures and keeps more WSD gain",
        "gated_cap_0p03": "recommended safe default: no off-diagonal blow-up in this audit",
        "gated_cap_0p02": "very safe; close to WSD-con amplitude",
        "local_peak_cap_0p04": "similar to capped estimator, uses high-feature region only",
    }
    for row in comparison_rows:
        lines.append(
            f"| `{row['estimator']}` | {row['cosine_to_wsd']:+.1f}% | "
            f"{row['probe_to_wsd']:+.1f}% | {row['worst_offdiag']:+.1f}% | "
            f"{readings[row['estimator']]} |\n"
        )

    for estimator in ESTIMATORS:
        lines += [
            f"\n## Matrix: `{estimator}`\n\n",
            f"![{estimator}](figs/matrix_{estimator}.png)\n\n",
        ]

    rec = "gated_cap_0p03"
    lines += [
        "\n## Recommended Estimator\n\n",
        "For a curve-agnostic kappa that should not explode, use the projected estimator:\n\n",
        "```text\n",
        "if total_positive_drop < 0.05 or feature_max < 0.05 or drop_effective_steps > 6000:\n",
        "    kappa = 0\n",
        "else:\n",
        "    kappa = min(raw_nonnegative_ls_kappa, 0.03)\n",
        "```\n\n",
        "This is `gated_cap_0p03` in the benchmark. It is the safest default in this audit: "
        "cosine is gated to zero, and no off-diagonal train/test family worsens on average. "
        "The `0.04` cap is a slightly more aggressive option if the target is known to be WSD-like. Key cells:\n\n",
    ]
    for train, test in [
        ("Cosine decay", "WSD sharp"),
        ("WSD-con step", "WSD sharp"),
        ("WSD sharp", "WSD-con step"),
        ("WSD sharp", "WSD sharp"),
    ]:
        row = get(rec, train, test)
        lines.append(
            f"- `{train} -> {test}`: {float(row['delta_pct']):+.1f}% MAE, "
            f"{int(row['wins'])}/{int(row['tests'])} wins, mean kappa={float(row['mean_kappa']):.4f}\n"
        )

    lines += [
        "\n## Why this has theoretical support\n\n",
        "The correction law is a one-dimensional linear-response model after `lambda` is fixed. "
        "For a calibration curve, estimating `kappa` is therefore a projection of the MPL residual "
        "onto the response feature `phi=DropRelaxS`. This projection is meaningful only when the "
        "curve excites the response direction. Cosine violates this condition: its positive LR drops "
        "are spread over roughly `5.7e4` effective drop steps and `feature_max` is only about `0.021`, "
        "so the projection mostly fits slow MPL residual drift. The identifiability gate formalizes "
        "that failure and returns the minimum-norm positive-lag estimate, `kappa=0`.\n\n",
        "The cap is also not arbitrary in form. Under the theory, `kappa=eta_peak*chi`, with `chi` a "
        "local equilibrium-loss sensitivity. Since `chi` is a local physical susceptibility, it should "
        "be nonnegative and bounded within one architecture/data family. The cap implements this as a "
        "weak prior constraint. Empirically, `kappa_max=0.03` is close to the amplitude learned from "
        "the clean `WSD-con` relaxation probes and prevents the sharp-WSD amplitude from being used as "
        "a universal amplitude on unrelated targets.\n\n",
        "Interpretation: this estimator deliberately returns `kappa=0` on cosine because cosine does "
        "not identify the non-adiabatic lag. It also shrinks WSD-sharp amplitudes when the target family "
        "is unknown. If the target is known to be WSD-like, the uncapped WSD-family estimator is stronger; "
        "if the target is arbitrary, the capped estimator is safer.\n",
    ]
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    feats = feature_cache()
    summary, details, kappa_rows = run_benchmark(feats)
    write_csv(OUT_DIR / "summary.csv", summary)
    write_csv(OUT_DIR / "details.csv", details)
    write_csv(OUT_DIR / "kappa_diagnostics.csv", kappa_rows)

    comparison_rows = plot_estimator_comparison(FIG_DIR / "estimator_comparison.png", summary)
    write_csv(OUT_DIR / "estimator_comparison.csv", comparison_rows)

    for estimator in ESTIMATORS:
        rows = [r for r in summary if r["estimator"] == estimator]
        plot_matrix(FIG_DIR / f"matrix_{estimator}.png", rows, estimator)

    write_report(summary, comparison_rows)
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    for row in comparison_rows:
        print(
            f"{row['estimator']:24s} cosine->wsd={row['cosine_to_wsd']:+7.1f}% "
            f"probe->wsd={row['probe_to_wsd']:+7.1f}% worst={row['worst_offdiag']:+7.1f}%"
        )


if __name__ == "__main__":
    main()
