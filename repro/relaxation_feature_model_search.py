#!/usr/bin/env python3
"""Search residual-response features motivated by the residual-shape gallery.

The gallery suggests that the current S-time DropRelaxS feature is too diffuse
on smooth cosine schedules: its peak lags the MPL residual by tens of thousands
of steps.  This script compares alternative response kernels under two metrics:

1. self-fit: fit kappa on a curve and score the same curve;
2. generalization: fit kappa on calibration curves and score held-out curves.

No existing paper-facing artifacts are overwritten.
"""
from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
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


OUT_DIR = ROOT / "results" / "relaxation_feature_model_search"
FIG_DIR = OUT_DIR / "figs"

CURVES = [
    ("cosine_72000.csv", "Cosine 72k", "cosine"),
    ("wsd_20000_24000.csv", "WSD exp", "wsd"),
    ("wsdld_20000_24000.csv", "WSD linear", "wsd"),
    ("wsdcon_3.csv", "WSD-con 3e-5", "probe"),
    ("wsdcon_9.csv", "WSD-con 9e-5", "probe"),
    ("wsdcon_18.csv", "WSD-con 18e-5", "probe"),
]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    family: str
    lam_s: float | None = None
    tau_steps: float | None = None


FEATURES = [
    FeatureSpec("S10_current", "s_time", lam_s=10.0),
    FeatureSpec("S20", "s_time", lam_s=20.0),
    FeatureSpec("S50", "s_time", lam_s=50.0),
    FeatureSpec("S100", "s_time", lam_s=100.0),
    FeatureSpec("S200", "s_time", lam_s=200.0),
    FeatureSpec("S10_cap512", "s_time_cap", lam_s=10.0, tau_steps=512.0),
    FeatureSpec("S10_cap1024", "s_time_cap", lam_s=10.0, tau_steps=1024.0),
    FeatureSpec("S10_cap2048", "s_time_cap", lam_s=10.0, tau_steps=2048.0),
    FeatureSpec("S20_cap512", "s_time_cap", lam_s=20.0, tau_steps=512.0),
    FeatureSpec("S20_cap1024", "s_time_cap", lam_s=20.0, tau_steps=1024.0),
    FeatureSpec("step_tau256", "step_time", tau_steps=256.0),
    FeatureSpec("step_tau512", "step_time", tau_steps=512.0),
    FeatureSpec("step_tau1024", "step_time", tau_steps=1024.0),
    FeatureSpec("step_tau2048", "step_time", tau_steps=2048.0),
    FeatureSpec("local_drop", "local_drop"),
]

TRAIN_GROUPS = [
    ("cosine", ["cosine_72000.csv"]),
    ("wsd", ["wsd_20000_24000.csv", "wsdld_20000_24000.csv"]),
    ("probe", ["wsdcon_3.csv", "wsdcon_9.csv", "wsdcon_18.csv"]),
    ("probe3", ["wsdcon_3.csv"]),
    ("probe9", ["wsdcon_9.csv"]),
]


def response_feature(curve, spec: FeatureSpec) -> np.ndarray:
    eta = curve.lrs.astype(np.float64)
    drop = np.zeros_like(eta)
    drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0) / PEAK_LR

    if spec.family == "local_drop":
        return drop[curve.step]

    acc = 0.0
    values = np.zeros_like(eta)
    for t in range(len(eta)):
        if spec.family == "s_time":
            assert spec.lam_s is not None
            rate = spec.lam_s * eta[t]
        elif spec.family == "s_time_cap":
            assert spec.lam_s is not None and spec.tau_steps is not None
            rate = max(spec.lam_s * eta[t], 1.0 / spec.tau_steps)
        elif spec.family == "step_time":
            assert spec.tau_steps is not None
            rate = 1.0 / spec.tau_steps
        else:
            raise ValueError(spec)
        acc = acc * math.exp(-rate) + drop[t]
        values[t] = acc
    return values[curve.step]


def feature_cache() -> dict[tuple[str, str, str], np.ndarray]:
    cache = {}
    for spec in FEATURES:
        for scale in SCALES:
            for curve_name, _, _ in CURVES:
                cache[(spec.name, scale, curve_name)] = response_feature(load_curve(scale, curve_name), spec)
    return cache


def fit_origin_nonnegative(x: np.ndarray, y: np.ndarray) -> float:
    denom = float(np.dot(x, x))
    if denom <= 1e-18:
        return 0.0
    return max(0.0, float(np.dot(x, y) / denom))


def r2_origin(y: np.ndarray, yhat: np.ndarray) -> float:
    denom = float(np.dot(y, y))
    if denom <= 1e-18:
        return float("nan")
    return 1.0 - float(np.dot(y - yhat, y - yhat) / denom)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - float(np.mean(x))
    yc = y - float(np.mean(y))
    denom = float(np.linalg.norm(xc) * np.linalg.norm(yc))
    if denom <= 1e-18:
        return float("nan")
    return float(np.dot(xc, yc) / denom)


def fit_kappa(spec_name: str, scale: str, train_curves: list[str], feats) -> float:
    xs, ys = [], []
    p = MPL_PRECOMPUTED_INIT[scale]
    for curve_name in train_curves:
        curve = load_curve(scale, curve_name)
        xs.append(feats[(spec_name, scale, curve_name)])
        ys.append(curve.loss - mpl_predict(p, curve))
    return fit_origin_nonnegative(np.concatenate(xs), np.concatenate(ys))


def score_curve(spec_name: str, scale: str, curve_name: str, kappa: float, feats) -> dict[str, object]:
    curve = load_curve(scale, curve_name)
    base = mpl_predict(MPL_PRECOMPUTED_INIT[scale], curve)
    feature = feats[(spec_name, scale, curve_name)]
    pred = base + kappa * feature
    base_mae = metrics(curve.loss, base)["mae"]
    corr_mae = metrics(curve.loss, pred)["mae"]
    residual = curve.loss - base
    estimate = kappa * feature
    return {
        "base_mae": base_mae,
        "corr_mae": corr_mae,
        "delta_pct": 100.0 * (corr_mae / base_mae - 1.0),
        "win": int(corr_mae < base_mae),
        "origin_r2": r2_origin(residual, estimate),
        "pearson": pearson(residual, estimate),
    }


def self_fit(feats) -> list[dict[str, object]]:
    rows = []
    for spec in FEATURES:
        for scale in SCALES:
            for curve_name, label, group in CURVES:
                kappa = fit_kappa(spec.name, scale, [curve_name], feats)
                scored = score_curve(spec.name, scale, curve_name, kappa, feats)
                rows.append(
                    {
                        "feature": spec.name,
                        "scale": scale,
                        "curve": curve_name,
                        "label": label,
                        "group": group,
                        "kappa": kappa,
                        **scored,
                    }
                )
    return rows


def generalization(feats) -> list[dict[str, object]]:
    rows = []
    for spec in FEATURES:
        for train_id, train_curves in TRAIN_GROUPS:
            for scale in SCALES:
                kappa = fit_kappa(spec.name, scale, train_curves, feats)
                for test_curve, test_label, test_group in CURVES:
                    scored = score_curve(spec.name, scale, test_curve, kappa, feats)
                    rows.append(
                        {
                            "feature": spec.name,
                            "train_id": train_id,
                            "train_curves": "+".join(c.replace(".csv", "") for c in train_curves),
                            "scale": scale,
                            "test_curve": test_curve,
                            "test_label": test_label,
                            "test_group": test_group,
                            "kappa": kappa,
                            **scored,
                        }
                    )
    return rows


def mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else float("nan")


def summarize_self(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for spec in FEATURES:
        sub = [r for r in rows if r["feature"] == spec.name]
        out.append(
            {
                "feature": spec.name,
                "mean_delta": mean([float(r["delta_pct"]) for r in sub]),
                "median_delta": float(np.median([float(r["delta_pct"]) for r in sub])),
                "worst_delta": float(np.max([float(r["delta_pct"]) for r in sub])),
                "mean_r2": mean([float(r["origin_r2"]) for r in sub]),
                "wins": int(sum(int(r["win"]) for r in sub)),
                "tests": len(sub),
                "mean_kappa": mean([float(r["kappa"]) for r in sub]),
            }
        )
    return sorted(out, key=lambda r: (float(r["mean_delta"]), float(r["worst_delta"])))


def summarize_generalization(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for spec in FEATURES:
        sub_all = [r for r in rows if r["feature"] == spec.name]
        for train_id, _ in TRAIN_GROUPS:
            sub = [r for r in sub_all if r["train_id"] == train_id]
            off = [r for r in sub if r["test_group"] != train_id]
            wsd_targets = [r for r in sub if r["test_group"] == "wsd"]
            probe_to_wsd = [r for r in sub if train_id.startswith("probe") and r["test_group"] == "wsd"]
            cosine_to_wsd = [r for r in sub if train_id == "cosine" and r["test_group"] == "wsd"]
            out.append(
                {
                    "feature": spec.name,
                    "train_id": train_id,
                    "mean_all": mean([float(r["delta_pct"]) for r in sub]),
                    "mean_off_group": mean([float(r["delta_pct"]) for r in off]),
                    "worst_off_group": float(np.max([float(r["delta_pct"]) for r in off])) if off else float("nan"),
                    "wins_all": int(sum(int(r["win"]) for r in sub)),
                    "tests_all": len(sub),
                    "mean_wsd_targets": mean([float(r["delta_pct"]) for r in wsd_targets]),
                    "probe_to_wsd": mean([float(r["delta_pct"]) for r in probe_to_wsd]),
                    "cosine_to_wsd": mean([float(r["delta_pct"]) for r in cosine_to_wsd]),
                    "mean_kappa": mean([float(r["kappa"]) for r in sub]),
                    "max_kappa": float(np.max([float(r["kappa"]) for r in sub])) if sub else float("nan"),
                }
            )
    return sorted(out, key=lambda r: (float(r["mean_off_group"]), float(r["worst_off_group"])))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_self_summary(rows: list[dict[str, object]], path: Path) -> None:
    top = rows[:12]
    x = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(11.8, 4.8))
    ax.axhline(0.0, color="#333333", lw=0.8)
    ax.bar(x, [float(r["mean_delta"]) for r in top], color="#2563eb")
    ax.set_xticks(x, [str(r["feature"]) for r in top], rotation=25, ha="right")
    ax.set_ylabel("mean self-fit Delta MAE (%)")
    ax.set_title("Best response features by same-curve self-fit")
    ax.grid(axis="y", alpha=0.24)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def plot_generalization_summary(rows: list[dict[str, object]], path: Path) -> None:
    top = rows[:16]
    x = np.arange(len(top))
    labels = [f"{r['feature']}\\ntrain={r['train_id']}" for r in top]
    fig, ax = plt.subplots(figsize=(13.5, 5.6))
    ax.axhline(0.0, color="#333333", lw=0.8)
    ax.bar(x - 0.18, [float(r["mean_off_group"]) for r in top], width=0.36, label="mean off-group")
    ax.bar(x + 0.18, [float(r["worst_off_group"]) for r in top], width=0.36, label="worst off-group")
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("Delta MAE vs MPL (%)")
    ax.set_title("Best response features by calibration-to-held-group generalization")
    ax.grid(axis="y", alpha=0.24)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def family_matrix(rows: list[dict[str, object]], feature: str, path: Path) -> None:
    train_ids = [x[0] for x in TRAIN_GROUPS]
    test_groups = ["cosine", "wsd", "probe"]
    mat = np.full((len(train_ids), len(test_groups)), np.nan)
    wins = {}
    for i, train_id in enumerate(train_ids):
        for j, test_group in enumerate(test_groups):
            sub = [r for r in rows if r["feature"] == feature and r["train_id"] == train_id and r["test_group"] == test_group]
            mat[i, j] = mean([float(r["delta_pct"]) for r in sub])
            wins[(i, j)] = f"{sum(int(r['win']) for r in sub)}/{len(sub)}"

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    norm = TwoSlopeNorm(vmin=-55, vcenter=0, vmax=100)
    im = ax.imshow(mat, cmap="RdYlGn_r", norm=norm)
    ax.set_xticks(np.arange(len(test_groups)), test_groups)
    ax.set_yticks(np.arange(len(train_ids)), train_ids)
    ax.set_xlabel("test group")
    ax.set_ylabel("calibration group")
    ax.set_title(f"Generalization matrix: {feature}")
    for i in range(len(train_ids)):
        for j in range(len(test_groups)):
            value = mat[i, j]
            color = "white" if value > 35 else "black"
            ax.text(j, i, f"{value:+.1f}%\n{wins[(i, j)]}", ha="center", va="center", fontsize=10, fontweight="bold", color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Delta MAE vs MPL")
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def write_report(path: Path, self_summary: list[dict[str, object]], gen_summary: list[dict[str, object]]) -> None:
    lines = [
        "# Relaxation Feature Model Search\n\n",
        "The residual-shape gallery suggested that the current `S10_current` feature lags badly on diffuse cosine schedules. "
        "This search compares faster S-time kernels, step-time kernels, and capped-S kernels that bound the maximum step-time relaxation tail at low LR.\n\n",
        "## Best Self-Fit Features\n\n",
        "| feature | mean delta | worst delta | mean R2 | wins |\n",
        "|---|---:|---:|---:|---:|\n",
    ]
    for row in self_summary[:10]:
        lines.append(
            f"| {row['feature']} | {float(row['mean_delta']):+.1f}% | {float(row['worst_delta']):+.1f}% | "
            f"{float(row['mean_r2']):.3f} | {int(row['wins'])}/{int(row['tests'])} |\n"
        )

    lines += [
        "\n## Best Generalization Rows\n\n",
        "| feature | train | mean off-group | worst off-group | WSD targets | wins |\n",
        "|---|---|---:|---:|---:|---:|\n",
    ]
    for row in gen_summary[:14]:
        lines.append(
            f"| {row['feature']} | {row['train_id']} | {float(row['mean_off_group']):+.1f}% | "
            f"{float(row['worst_off_group']):+.1f}% | {float(row['mean_wsd_targets']):+.1f}% | "
            f"{int(row['wins_all'])}/{int(row['tests_all'])} |\n"
        )

    current_self = next(r for r in self_summary if r["feature"] == "S10_current")
    current_gen_probe = next(r for r in gen_summary if r["feature"] == "S10_current" and r["train_id"] == "probe")
    lines += [
        "\n## Reading\n\n",
        f"- Current `S10_current` self-fit mean delta is `{float(current_self['mean_delta']):+.1f}%`; "
        f"probe-calibrated off-group mean is `{float(current_gen_probe['mean_off_group']):+.1f}%`.\n",
        "- Capped-S kernels directly test the hypothesis that low-LR relaxation should not acquire an arbitrarily long step-time tail. "
        "If they move up the generalization table while preserving self-fit, they are the strongest candidate for replacing the current response feature.\n",
        "- Step-time kernels are included as an aggressive alternative; strong self-fit but poor transfer would indicate that pure step-time relaxation throws away the measured LR-dependent rate too aggressively.\n",
    ]
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    feats = feature_cache()
    self_rows = self_fit(feats)
    gen_rows = generalization(feats)
    self_summary = summarize_self(self_rows)
    gen_summary = summarize_generalization(gen_rows)

    write_csv(OUT_DIR / "self_fit_details.csv", self_rows)
    write_csv(OUT_DIR / "generalization_details.csv", gen_rows)
    write_csv(OUT_DIR / "self_fit_summary.csv", self_summary)
    write_csv(OUT_DIR / "generalization_summary.csv", gen_summary)
    plot_self_summary(self_summary, FIG_DIR / "self_fit_summary.png")
    plot_generalization_summary(gen_summary, FIG_DIR / "generalization_summary.png")
    for feature in ["S10_current", gen_summary[0]["feature"], "S10_cap1024", "step_tau512"]:
        family_matrix(gen_rows, str(feature), FIG_DIR / f"matrix_{feature}.png")
    write_report(OUT_DIR / "REPORT.md", self_summary, gen_summary)

    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    print("best self-fit:")
    for row in self_summary[:6]:
        print(f"  {row['feature']:14s} mean={float(row['mean_delta']):+6.1f}% worst={float(row['worst_delta']):+6.1f}%")
    print("best generalization:")
    for row in gen_summary[:10]:
        print(
            f"  {row['feature']:14s} train={row['train_id']:7s} "
            f"mean_off={float(row['mean_off_group']):+6.1f}% "
            f"worst={float(row['worst_off_group']):+6.1f}% "
            f"wsd={float(row['mean_wsd_targets']):+6.1f}%"
        )


if __name__ == "__main__":
    main()
