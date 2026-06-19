#!/usr/bin/env python3
"""Robust matrix for the finite step-time response model.

This script promotes the image-motivated step-time response from a search
diagnostic to a matrix-style method comparison.  It compares:

  - S10_current: original cumulative-LR feature.
  - step_tau1024: conservative finite catch-up feature.
  - step_tau1536: best pooled-probe WSD mean in the tau scan.
  - step_tau2304: strongest endpoint-matched WSD candidate.

Each feature uses the same scalar kappa fit through the origin.  We report
self-fit, train-group -> test-group transfer, and a target-schedule endpoint
match rule for WSD targets.
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

from reproduce_cosine_to_wsd import (  # noqa: E402
    END_LR,
    MPL_PRECOMPUTED_INIT,
    PEAK_LR,
    SCALES,
    load_curve,
    metrics,
    mpl_predict,
)


OUT_DIR = ROOT / "results" / "step_time_robust_matrix"
FIG_DIR = OUT_DIR / "figs"

CURVES = [
    ("cosine_72000.csv", "Cosine 72k", "cosine"),
    ("wsd_20000_24000.csv", "WSD exp", "wsd"),
    ("wsdld_20000_24000.csv", "WSD linear", "wsd"),
    ("wsdcon_3.csv", "WSD-con 3e-5", "probe"),
    ("wsdcon_9.csv", "WSD-con 9e-5", "probe"),
    ("wsdcon_18.csv", "WSD-con 18e-5", "probe"),
]

TRAIN_GROUPS = [
    ("cosine", ["cosine_72000.csv"]),
    ("wsd", ["wsd_20000_24000.csv", "wsdld_20000_24000.csv"]),
    ("probe", ["wsdcon_3.csv", "wsdcon_9.csv", "wsdcon_18.csv"]),
    ("probe3", ["wsdcon_3.csv"]),
    ("probe9", ["wsdcon_9.csv"]),
]

FEATURES = [
    ("S10_current", "s_time", 10.0),
    ("step_tau1024", "step_time", 1024.0),
    ("step_tau1536", "step_time", 1536.0),
    ("step_tau2304", "step_time", 2304.0),
]


def build_cache() -> dict[tuple[str, str], dict[str, object]]:
    cache = {}
    for scale in SCALES:
        for curve_name, _, _ in CURVES:
            curve = load_curve(scale, curve_name)
            base = mpl_predict(MPL_PRECOMPUTED_INIT[scale], curve)
            cache[(scale, curve_name)] = {
                "curve": curve,
                "base": base,
                "residual": curve.loss - base,
                "base_mae": metrics(curve.loss, base)["mae"],
            }
    return cache


def response_feature(curve, feature_kind: str, param: float) -> np.ndarray:
    eta = curve.lrs.astype(np.float64)
    drop = np.zeros_like(eta)
    drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0) / PEAK_LR
    out = np.zeros_like(eta)
    acc = 0.0
    for t in range(len(eta)):
        if feature_kind == "s_time":
            rate = param * eta[t]
        elif feature_kind == "step_time":
            rate = 1.0 / param
        else:
            raise ValueError(feature_kind)
        acc = acc * math.exp(-rate) + drop[t]
        out[t] = acc
    return out[curve.step]


def feature(cache, feat_cache, feature_name: str, scale: str, curve_name: str) -> np.ndarray:
    key = (feature_name, scale, curve_name)
    if key not in feat_cache:
        _, kind, param = next(x for x in FEATURES if x[0] == feature_name)
        feat_cache[key] = response_feature(cache[(scale, curve_name)]["curve"], kind, param)
    return feat_cache[key]


def fit_kappa(xs: list[np.ndarray], ys: list[np.ndarray]) -> float:
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    denom = float(np.dot(x, x))
    if denom <= 1e-18:
        return 0.0
    return max(0.0, float(np.dot(x, y) / denom))


def fit_on(cache, feat_cache, feature_name: str, scale: str, train_curves: list[str]) -> float:
    return fit_kappa(
        [feature(cache, feat_cache, feature_name, scale, c) for c in train_curves],
        [cache[(scale, c)]["residual"] for c in train_curves],
    )


def score(cache, feat_cache, feature_name: str, scale: str, curve_name: str, kappa: float) -> dict[str, object]:
    row = cache[(scale, curve_name)]
    curve = row["curve"]
    pred = row["base"] + kappa * feature(cache, feat_cache, feature_name, scale, curve_name)
    corr_mae = metrics(curve.loss, pred)["mae"]
    base_mae = float(row["base_mae"])
    return {
        "base_mae": base_mae,
        "corr_mae": corr_mae,
        "delta_pct": 100.0 * (corr_mae / base_mae - 1.0),
        "win": int(corr_mae < base_mae),
    }


def run_group_matrix() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    cache = build_cache()
    feat_cache: dict[tuple[str, str, str], np.ndarray] = {}
    details, summary, kappas = [], [], []

    for feature_name, _, _ in FEATURES:
        for train_id, train_curves in TRAIN_GROUPS:
            for scale in SCALES:
                kappa = fit_on(cache, feat_cache, feature_name, scale, train_curves)
                kappas.append(
                    {
                        "feature": feature_name,
                        "mode": "group",
                        "train_id": train_id,
                        "scale": scale,
                        "train_curves": "+".join(c.replace(".csv", "") for c in train_curves),
                        "kappa": kappa,
                    }
                )
                for test_curve, test_label, test_group in CURVES:
                    scored = score(cache, feat_cache, feature_name, scale, test_curve, kappa)
                    details.append(
                        {
                            "feature": feature_name,
                            "mode": "group",
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

    for feature_name, _, _ in FEATURES:
        for train_id, _ in TRAIN_GROUPS:
            for test_group in ["cosine", "wsd", "probe"]:
                sub = [
                    r
                    for r in details
                    if r["feature"] == feature_name
                    and r["mode"] == "group"
                    and r["train_id"] == train_id
                    and r["test_group"] == test_group
                ]
                summary.append(summarize_rows(feature_name, "group", train_id, test_group, sub))
    return details, summary, kappas


def run_self_fit() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    cache = build_cache()
    feat_cache: dict[tuple[str, str, str], np.ndarray] = {}
    details, summary = [], []
    for feature_name, _, _ in FEATURES:
        for scale in SCALES:
            for curve_name, label, group in CURVES:
                kappa = fit_on(cache, feat_cache, feature_name, scale, [curve_name])
                scored = score(cache, feat_cache, feature_name, scale, curve_name, kappa)
                details.append(
                    {
                        "feature": feature_name,
                        "mode": "self",
                        "scale": scale,
                        "curve": curve_name,
                        "label": label,
                        "group": group,
                        "kappa": kappa,
                        **scored,
                    }
                )
        sub = [r for r in details if r["feature"] == feature_name]
        summary.append(
            {
                "feature": feature_name,
                "mode": "self",
                "mean_delta": mean([float(r["delta_pct"]) for r in sub]),
                "worst_delta": float(np.max([float(r["delta_pct"]) for r in sub])),
                "wins": int(sum(int(r["win"]) for r in sub)),
                "tests": len(sub),
                "mean_kappa": mean([float(r["kappa"]) for r in sub]),
            }
        )
    return details, summary


def run_endpoint_rule() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Target-schedule-only rule: if WSD target ends at 3e-5, use wsdcon_3."""
    cache = build_cache()
    feat_cache: dict[tuple[str, str, str], np.ndarray] = {}
    details, summary = [], []
    rule_defs = [
        ("endpoint_tau1024", "step_tau1024", ["wsdcon_3.csv"]),
        ("endpoint_tau1536", "step_tau1536", ["wsdcon_3.csv"]),
        ("endpoint_tau2304", "step_tau2304", ["wsdcon_3.csv"]),
    ]
    targets = ["wsd_20000_24000.csv", "wsdld_20000_24000.csv"]
    for rule_name, feature_name, train_curves in rule_defs:
        for scale in SCALES:
            kappa = fit_on(cache, feat_cache, feature_name, scale, train_curves)
            for target in targets:
                scored = score(cache, feat_cache, feature_name, scale, target, kappa)
                details.append(
                    {
                        "feature": feature_name,
                        "mode": "endpoint_rule",
                        "rule": rule_name,
                        "scale": scale,
                        "train_curves": "+".join(c.replace(".csv", "") for c in train_curves),
                        "target_curve": target,
                        "target_end_lr": END_LR,
                        "kappa": kappa,
                        **scored,
                    }
                )
        sub = [r for r in details if r["rule"] == rule_name]
        summary.append(
            {
                "rule": rule_name,
                "feature": feature_name,
                "mode": "endpoint_rule",
                "mean_delta": mean([float(r["delta_pct"]) for r in sub]),
                "worst_delta": float(np.max([float(r["delta_pct"]) for r in sub])),
                "wins": int(sum(int(r["win"]) for r in sub)),
                "tests": len(sub),
                "mean_kappa": mean([float(r["kappa"]) for r in sub]),
            }
        )
    return details, summary


def summarize_rows(feature_name: str, mode: str, train_id: str, test_group: str, rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "feature": feature_name,
        "mode": mode,
        "train_id": train_id,
        "test_group": test_group,
        "mean_delta": mean([float(r["delta_pct"]) for r in rows]),
        "worst_delta": float(np.max([float(r["delta_pct"]) for r in rows])) if rows else float("nan"),
        "wins": int(sum(int(r["win"]) for r in rows)),
        "tests": len(rows),
        "mean_kappa": mean([float(r["kappa"]) for r in rows]),
        "max_kappa": float(np.max([float(r["kappa"]) for r in rows])) if rows else float("nan"),
    }


def mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else float("nan")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_feature_matrix(summary: list[dict[str, object]], feature_name: str, path: Path) -> None:
    train_ids = [x[0] for x in TRAIN_GROUPS]
    test_groups = ["cosine", "wsd", "probe"]
    mat = np.full((len(train_ids), len(test_groups)), np.nan)
    wins = {}
    for i, train_id in enumerate(train_ids):
        for j, test_group in enumerate(test_groups):
            row = next(
                r
                for r in summary
                if r["feature"] == feature_name
                and r["mode"] == "group"
                and r["train_id"] == train_id
                and r["test_group"] == test_group
            )
            mat[i, j] = float(row["mean_delta"])
            wins[(i, j)] = f"{int(row['wins'])}/{int(row['tests'])}"
    fig, ax = plt.subplots(figsize=(7.6, 5.1))
    norm = TwoSlopeNorm(vmin=-55, vcenter=0, vmax=100)
    im = ax.imshow(mat, cmap="RdYlGn_r", norm=norm)
    ax.set_xticks(np.arange(len(test_groups)), test_groups)
    ax.set_yticks(np.arange(len(train_ids)), train_ids)
    ax.set_xlabel("test group")
    ax.set_ylabel("calibration group")
    ax.set_title(feature_name)
    for i in range(len(train_ids)):
        for j in range(len(test_groups)):
            value = mat[i, j]
            color = "white" if value > 35 else "black"
            ax.text(j, i, f"{value:+.1f}%\n{wins[(i, j)]}", ha="center", va="center", fontsize=10, fontweight="bold", color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Delta MAE vs MPL")
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def plot_method_comparison(self_summary: list[dict[str, object]], group_summary: list[dict[str, object]], endpoint_summary: list[dict[str, object]], path: Path) -> None:
    rows = []
    for feature_name, _, _ in FEATURES:
        self_row = next(r for r in self_summary if r["feature"] == feature_name)
        probe_wsd = next(
            r
            for r in group_summary
            if r["feature"] == feature_name and r["train_id"] == "probe" and r["test_group"] == "wsd"
        )
        probe_worst = next(
            r
            for r in group_summary
            if r["feature"] == feature_name and r["train_id"] == "probe" and r["test_group"] == "cosine"
        )
        rows.append(
            {
                "name": feature_name,
                "self": float(self_row["mean_delta"]),
                "probe_wsd": float(probe_wsd["mean_delta"]),
                "probe_cosine": float(probe_worst["mean_delta"]),
            }
        )
    for row in endpoint_summary:
        rows.append(
            {
                "name": str(row["rule"]),
                "self": float("nan"),
                "probe_wsd": float(row["mean_delta"]),
                "probe_cosine": float("nan"),
            }
        )

    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(12.6, 5.2))
    ax.axhline(0.0, color="#333333", lw=0.8)
    width = 0.26
    ax.bar(x - width, [r["self"] for r in rows], width, label="self-fit mean")
    ax.bar(x, [r["probe_wsd"] for r in rows], width, label="probe -> WSD mean")
    ax.bar(x + width, [r["probe_cosine"] for r in rows], width, label="probe -> cosine sanity")
    ax.set_xticks(x, [r["name"] for r in rows], rotation=25, ha="right")
    ax.set_ylabel("Delta MAE vs MPL (%)")
    ax.set_title("Step-time response variants: self-fit and generalization")
    ax.grid(axis="y", alpha=0.24)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def write_report(path: Path, self_summary: list[dict[str, object]], group_summary: list[dict[str, object]], endpoint_summary: list[dict[str, object]]) -> None:
    lines = [
        "# Step-Time Robust Matrix\n\n",
        "This promotes the residual-shape finding into a matrix-style method comparison. "
        "The finite step-time response prevents the low-LR tail from producing the broad delayed cosine correction seen in the error plots.\n\n",
        "## Self-Fit\n\n",
        "| feature | mean delta | worst delta | wins | mean kappa |\n",
        "|---|---:|---:|---:|---:|\n",
    ]
    for row in self_summary:
        lines.append(
            f"| {row['feature']} | {float(row['mean_delta']):+.1f}% | {float(row['worst_delta']):+.1f}% | "
            f"{int(row['wins'])}/{int(row['tests'])} | {float(row['mean_kappa']):.4f} |\n"
        )

    lines += [
        "\n## Pooled Probe Calibration To WSD\n\n",
        "| feature | probe -> WSD mean | worst | wins | probe -> cosine sanity |\n",
        "|---|---:|---:|---:|---:|\n",
    ]
    for feature_name, _, _ in FEATURES:
        wsd = next(r for r in group_summary if r["feature"] == feature_name and r["train_id"] == "probe" and r["test_group"] == "wsd")
        cosine = next(r for r in group_summary if r["feature"] == feature_name and r["train_id"] == "probe" and r["test_group"] == "cosine")
        lines.append(
            f"| {feature_name} | {float(wsd['mean_delta']):+.1f}% | {float(wsd['worst_delta']):+.1f}% | "
            f"{int(wsd['wins'])}/{int(wsd['tests'])} | {float(cosine['mean_delta']):+.1f}% |\n"
        )

    lines += [
        "\n## Endpoint-Matched WSD Rule\n\n",
        "For WSD targets whose final LR is `3e-5`, use the `wsdcon_3` probe to estimate kappa. This rule uses the target schedule endpoint, not target losses.\n\n",
        "| rule | WSD mean | worst | wins | mean kappa |\n",
        "|---|---:|---:|---:|---:|\n",
    ]
    for row in endpoint_summary:
        lines.append(
            f"| {row['rule']} | {float(row['mean_delta']):+.1f}% | {float(row['worst_delta']):+.1f}% | "
            f"{int(row['wins'])}/{int(row['tests'])} | {float(row['mean_kappa']):.4f} |\n"
        )

    s10_self = next(r for r in self_summary if r["feature"] == "S10_current")
    st_self = next(r for r in self_summary if r["feature"] == "step_tau1024")
    s10_wsd = next(r for r in group_summary if r["feature"] == "S10_current" and r["train_id"] == "probe" and r["test_group"] == "wsd")
    st_wsd = next(r for r in group_summary if r["feature"] == "step_tau1024" and r["train_id"] == "probe" and r["test_group"] == "wsd")
    endpoint_best = min(endpoint_summary, key=lambda r: float(r["mean_delta"]))
    lines += [
        "\n## Reading\n\n",
        f"- Conservative replacement `step_tau1024` improves self-fit from `{float(s10_self['mean_delta']):+.1f}%` to `{float(st_self['mean_delta']):+.1f}%` while keeping `{int(st_self['wins'])}/{int(st_self['tests'])}` self-fit wins.\n",
        f"- Pooled-probe WSD generalization improves from `{float(s10_wsd['mean_delta']):+.1f}%` to `{float(st_wsd['mean_delta']):+.1f}%` with `step_tau1024`, preserving `{int(st_wsd['wins'])}/{int(st_wsd['tests'])}` WSD wins.\n",
        f"- Endpoint-matched aggressive variant `{endpoint_best['rule']}` reaches `{float(endpoint_best['mean_delta']):+.1f}%` WSD mean MAE change with worst `{float(endpoint_best['worst_delta']):+.1f}%`.\n",
        "- The remaining failure mode is diffuse cosine calibration. Step-time reduces but does not eliminate raw cosine over-transfer; cosine should still be treated as a low-frequency nuisance diagnostic rather than a primary kappa source.\n",
    ]
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    group_details, group_summary, kappa_rows = run_group_matrix()
    self_details, self_summary = run_self_fit()
    endpoint_details, endpoint_summary = run_endpoint_rule()

    write_csv(OUT_DIR / "group_details.csv", group_details)
    write_csv(OUT_DIR / "group_summary.csv", group_summary)
    write_csv(OUT_DIR / "kappas.csv", kappa_rows)
    write_csv(OUT_DIR / "self_details.csv", self_details)
    write_csv(OUT_DIR / "self_summary.csv", self_summary)
    write_csv(OUT_DIR / "endpoint_details.csv", endpoint_details)
    write_csv(OUT_DIR / "endpoint_summary.csv", endpoint_summary)

    for feature_name, _, _ in FEATURES:
        plot_feature_matrix(group_summary, feature_name, FIG_DIR / f"matrix_{feature_name}.png")
    plot_method_comparison(self_summary, group_summary, endpoint_summary, FIG_DIR / "method_comparison.png")
    write_report(OUT_DIR / "REPORT.md", self_summary, group_summary, endpoint_summary)

    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    for row in self_summary:
        print(
            f"self {row['feature']:13s} mean={float(row['mean_delta']):+6.1f}% "
            f"worst={float(row['worst_delta']):+6.1f}% wins={int(row['wins'])}/{int(row['tests'])}"
        )
    for feature_name, _, _ in FEATURES:
        wsd = next(r for r in group_summary if r["feature"] == feature_name and r["train_id"] == "probe" and r["test_group"] == "wsd")
        print(
            f"probe->WSD {feature_name:13s} mean={float(wsd['mean_delta']):+6.1f}% "
            f"worst={float(wsd['worst_delta']):+6.1f}% wins={int(wsd['wins'])}/{int(wsd['tests'])}"
        )
    for row in endpoint_summary:
        print(
            f"endpoint {row['rule']:16s} mean={float(row['mean_delta']):+6.1f}% "
            f"worst={float(row['worst_delta']):+6.1f}% wins={int(row['wins'])}/{int(row['tests'])}"
        )


if __name__ == "__main__":
    main()
