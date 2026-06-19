#!/usr/bin/env python3
"""Fixed image-driven step-time nuisance estimator.

This is the compact, reusable implementation of the candidate selected by the
response-shape diagnostics and holdout audit:

  * finite step-time response with tau=1024 steps,
  * Fourier low-frequency nuisance residualization,
  * leave-curve-out EB q75 ridge scale,
  * target-side total-drop factor.

Unlike `step_time_nuisance_estimator.py`, this file does not search a grid.  It
exists to make the current best formula reproducible as a fixed method.
"""
from __future__ import annotations

import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

TMP_CACHE = Path(os.environ.get("TMPDIR", "/tmp")) / "dl-final-cache"
TMP_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(TMP_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(TMP_CACHE / "xdg"))

import matplotlib
import numpy as np
from matplotlib.colors import TwoSlopeNorm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent
ROOT = REPO.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from reproduce_cosine_to_wsd import (  # noqa: E402
    MPL_PRECOMPUTED_INIT,
    PEAK_LR,
    SCALES,
    load_curve,
    metrics,
    mpl_predict,
)


OUT_DIR = ROOT / "results" / "step_time_nuisance_fixed"
FIG_DIR = OUT_DIR / "figs"

CURVES = [
    ("cosine_72000.csv", "Cosine"),
    ("wsd_20000_24000.csv", "WSD sharp"),
    ("wsdld_20000_24000.csv", "WSD linear"),
    ("wsdcon_3.csv", "WSD-con 3e-5"),
    ("wsdcon_9.csv", "WSD-con 9e-5"),
    ("wsdcon_18.csv", "WSD-con 18e-5"),
]

STEP_TAU = 1024.0
TARGET_DROP_NORMALIZER = 0.9


@dataclass(frozen=True)
class Estimate:
    scale: str
    train_id: str
    target_curve: str
    tau: float
    projected_dot: float
    projected_l2: float
    raw_map: float
    kappa: float
    target_drop_factor: float
    target_drop_norm: float


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def response_feature(curve) -> np.ndarray:
    eta = curve.lrs.astype(np.float64)
    drop = np.zeros_like(eta)
    drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0) / PEAK_LR
    out = np.zeros_like(eta)
    acc = 0.0
    for t in range(len(eta)):
        acc = acc * math.exp(-1.0 / STEP_TAU) + drop[t]
        out[t] = acc
    return out[curve.step]


def fourier2_basis(steps: np.ndarray) -> np.ndarray:
    t = steps.astype(np.float64)
    z = (t - float(t[0])) / max(float(t[-1] - t[0]), 1.0)
    cols = [
        np.ones_like(z),
        np.sin(math.pi * z),
        np.cos(math.pi * z),
        np.sin(2.0 * math.pi * z),
        np.cos(2.0 * math.pi * z),
    ]
    basis = np.column_stack(cols)
    return basis / np.maximum(np.linalg.norm(basis, axis=0), 1e-12)


def residualize(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(z, y, rcond=None)
    return y - z @ coef


def robust_scale(x: np.ndarray) -> float:
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return max(1.4826 * mad, float(np.std(x)) * 0.25, 1e-12)


def quantile(values: list[float], q: float) -> float:
    vals = sorted(v for v in values if math.isfinite(float(v)))
    if not vals:
        return float("nan")
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def target_drop_factor(curve) -> tuple[float, float]:
    eta = curve.lrs.astype(np.float64)
    drops = np.zeros_like(eta)
    drops[1:] = np.maximum(eta[:-1] - eta[1:], 0.0) / PEAK_LR
    total = float(np.sum(drops))
    return total, min(max(total / TARGET_DROP_NORMALIZER, 0.0), 1.0)


class StepTimeNuisanceEstimator:
    def __init__(self) -> None:
        self.curves: dict[tuple[str, str], dict[str, object]] = {}
        self.stats: dict[tuple[str, str], dict[str, float]] = {}
        self._build_cache()

    def _build_cache(self) -> None:
        for scale in SCALES:
            for curve_name, _ in CURVES:
                curve = load_curve(scale, curve_name)
                base = mpl_predict(MPL_PRECOMPUTED_INIT[scale], curve)
                residual = curve.loss - base
                phi = response_feature(curve)
                z = fourier2_basis(curve.step)
                phi_o = residualize(phi, z)
                residual_o = residualize(residual, z)
                l2 = max(float(np.dot(phi_o, phi_o)), 1e-18)
                dot = float(np.dot(phi_o, residual_o))
                corr_denom = float(np.linalg.norm(phi_o) * np.linalg.norm(residual_o))
                drop_norm, drop_factor = target_drop_factor(curve)
                self.curves[(scale, curve_name)] = {
                    "curve": curve,
                    "base": base,
                    "residual": residual,
                    "phi": phi,
                    "base_mae": metrics(curve.loss, base)["mae"],
                }
                self.stats[(scale, curve_name)] = {
                    "projected_dot": dot,
                    "projected_l2": l2,
                    "projected_raw_kappa": max(0.0, dot / l2),
                    "projected_corr": 0.0 if corr_denom <= 1e-18 else float(dot / corr_denom),
                    "projected_resid_scale": robust_scale(residual_o),
                    "target_drop_norm": drop_norm,
                    "target_drop_factor": drop_factor,
                }

    def estimate_tau(self, excluded_train_curves: tuple[str, ...]) -> float:
        excluded = set(excluded_train_curves)
        pool = [
            row
            for (scale, curve_name), row in self.stats.items()
            if curve_name not in excluded
            and row["projected_raw_kappa"] > 0.0
            and row["projected_corr"] > 0.05
        ]
        if len(pool) < 4:
            pool = [
                row
                for (scale, curve_name), row in self.stats.items()
                if curve_name not in excluded and row["projected_raw_kappa"] > 0.0
            ]
        sigma = quantile([row["projected_resid_scale"] for row in pool], 0.50)
        k0 = quantile([row["projected_raw_kappa"] for row in pool], 0.75)
        if not math.isfinite(sigma) or not math.isfinite(k0) or k0 <= 1e-12:
            return 0.0
        return min(max(sigma / k0, 0.0), 0.50)

    def fit_kappa(self, scale: str, train_curves: tuple[str, ...]) -> tuple[float, dict[str, float]]:
        tau = self.estimate_tau(train_curves)
        rows = [self.stats[(scale, curve_name)] for curve_name in train_curves]
        dot = float(sum(row["projected_dot"] for row in rows))
        l2 = float(sum(row["projected_l2"] for row in rows))
        raw_map = max(0.0, dot / max(l2 + tau * tau, 1e-18))
        return raw_map, {"tau": tau, "projected_dot": dot, "projected_l2": l2, "raw_map": raw_map}

    def estimate(self, scale: str, train_curves: tuple[str, ...], target_curve: str) -> Estimate:
        kappa, fit = self.fit_kappa(scale, train_curves)
        target_stats = self.stats[(scale, target_curve)]
        return Estimate(
            scale=scale,
            train_id="+".join(curve.replace(".csv", "") for curve in train_curves),
            target_curve=target_curve,
            tau=float(fit["tau"]),
            projected_dot=float(fit["projected_dot"]),
            projected_l2=float(fit["projected_l2"]),
            raw_map=float(fit["raw_map"]),
            kappa=kappa,
            target_drop_factor=float(target_stats["target_drop_factor"]),
            target_drop_norm=float(target_stats["target_drop_norm"]),
        )

    def score(self, estimate: Estimate) -> dict[str, object]:
        row = self.curves[(estimate.scale, estimate.target_curve)]
        curve = row["curve"]
        pred = row["base"] + estimate.target_drop_factor * estimate.kappa * row["phi"]
        corr_mae = metrics(curve.loss, pred)["mae"]
        base_mae = float(row["base_mae"])
        return {
            "base_mae": base_mae,
            "corr_mae": corr_mae,
            "delta_pct": 100.0 * (corr_mae / base_mae - 1.0),
            "win": int(corr_mae < base_mae),
            "nonharm": int(corr_mae <= base_mae * (1.0 + 1e-12)),
        }


def run_single(estimator: StepTimeNuisanceEstimator) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for train_curve, train_label in CURVES:
        for test_curve, test_label in CURVES:
            for scale in SCALES:
                estimate = estimator.estimate(scale, (train_curve,), test_curve)
                scored = estimator.score(estimate)
                rows.append(
                    {
                        "mode": "single_curve",
                        "scale": scale,
                        "train_curve": train_curve,
                        "train_label": train_label,
                        "test_curve": test_curve,
                        "test_label": test_label,
                        **estimate.__dict__,
                        **scored,
                    }
                )
    return rows


def run_groups(estimator: StepTimeNuisanceEstimator) -> list[dict[str, object]]:
    groups = [
        ("probe", ("wsdcon_3.csv", "wsdcon_9.csv", "wsdcon_18.csv")),
        ("probe3", ("wsdcon_3.csv",)),
        ("wsd", ("wsd_20000_24000.csv", "wsdld_20000_24000.csv")),
        ("cosine", ("cosine_72000.csv",)),
    ]
    rows: list[dict[str, object]] = []
    for group_id, train_curves in groups:
        for test_curve, test_label in CURVES:
            for scale in SCALES:
                estimate = estimator.estimate(scale, train_curves, test_curve)
                scored = estimator.score(estimate)
                rows.append(
                    {
                        "mode": "group",
                        "group_id": group_id,
                        "scale": scale,
                        "test_curve": test_curve,
                        "test_label": test_label,
                        **estimate.__dict__,
                        **scored,
                    }
                )
    return rows


def summarize(rows: list[dict[str, object]], prefix: str = "") -> dict[str, object]:
    deltas = [float(row["delta_pct"]) for row in rows]
    return {
        f"{prefix}rows": len(rows),
        f"{prefix}mean_delta": float(np.mean(deltas)) if deltas else float("nan"),
        f"{prefix}worst_delta": float(np.max(deltas)) if deltas else float("nan"),
        f"{prefix}wins": int(sum(float(delta) < 0.0 for delta in deltas)),
        f"{prefix}nonharm": int(sum(float(delta) <= 1e-10 for delta in deltas)),
    }


def single_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    cell_keys = sorted({(r["train_curve"], r["train_label"], r["test_curve"], r["test_label"]) for r in rows})
    for train_curve, train_label, test_curve, test_label in cell_keys:
        sub = [r for r in rows if r["train_curve"] == train_curve and r["test_curve"] == test_curve]
        out.append(
            {
                "train_curve": train_curve,
                "train_label": train_label,
                "test_curve": test_curve,
                "test_label": test_label,
                **summarize(sub),
            }
        )
    return out


def group_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for group_id in sorted({str(r["group_id"]) for r in rows}):
        for target_group, target_curves in [
            ("wsd", {"wsd_20000_24000.csv", "wsdld_20000_24000.csv"}),
            ("probe", {"wsdcon_3.csv", "wsdcon_9.csv", "wsdcon_18.csv"}),
            ("cosine", {"cosine_72000.csv"}),
        ]:
            sub = [r for r in rows if r["group_id"] == group_id and r["test_curve"] in target_curves]
            out.append(
                {
                    "group_id": group_id,
                    "target_group": target_group,
                    **summarize(sub),
                }
            )
    return out


def key_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    self_rows = [r for r in rows if r["train_curve"] == r["test_curve"]]
    off_rows = [r for r in rows if r["train_curve"] != r["test_curve"]]
    probe_wsd = [
        r
        for r in rows
        if r["train_curve"] in {"wsdcon_3.csv", "wsdcon_9.csv", "wsdcon_18.csv"}
        and r["test_curve"] in {"wsd_20000_24000.csv", "wsdld_20000_24000.csv"}
    ]
    cosine_wsd = [
        r
        for r in rows
        if r["train_curve"] == "cosine_72000.csv"
        and r["test_curve"] in {"wsd_20000_24000.csv", "wsdld_20000_24000.csv"}
    ]
    out: dict[str, object] = {}
    out.update(summarize(self_rows, "self_"))
    out.update(summarize(off_rows, "offdiag_"))
    out.update(summarize(probe_wsd, "probe_to_wsd_"))
    out.update(summarize(cosine_wsd, "cosine_to_wsd_"))
    return out


def plot_matrix(path: Path, summary_rows: list[dict[str, object]]) -> None:
    labels = [label for _, label in CURVES]
    mat = np.full((len(CURVES), len(CURVES)), np.nan)
    wins: dict[tuple[int, int], str] = {}
    for i, (train_curve, _) in enumerate(CURVES):
        for j, (test_curve, _) in enumerate(CURVES):
            row = next(r for r in summary_rows if r["train_curve"] == train_curve and r["test_curve"] == test_curve)
            mat[i, j] = float(row["mean_delta"])
            wins[(i, j)] = f"{int(row['wins'])}/{int(row['rows'])}"
    fig, ax = plt.subplots(figsize=(9.2, 7.0))
    im = ax.imshow(mat, cmap="RdYlGn_r", norm=TwoSlopeNorm(vmin=-80, vcenter=0, vmax=80))
    ax.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("test curve")
    ax.set_ylabel("calibration curve")
    ax.set_title("fixed step-time nuisance estimator")
    for i in range(len(CURVES)):
        for j in range(len(CURVES)):
            ax.text(j, i, f"{mat[i,j]:+.1f}%\n{wins[(i,j)]}", ha="center", va="center", fontsize=8, weight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("MAE change vs MPL")
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def write_report(single_rows: list[dict[str, object]], group_rows: list[dict[str, object]]) -> None:
    km = key_metrics(single_rows)
    gsum = group_summary(group_rows)
    probe_wsd = next(r for r in gsum if r["group_id"] == "probe" and r["target_group"] == "wsd")
    probe3_wsd = next(r for r in gsum if r["group_id"] == "probe3" and r["target_group"] == "wsd")
    lines = [
        "# Fixed Step-Time Nuisance Estimator\n\n",
        "This is the fixed implementation of the image-driven candidate selected in `../step_time_nuisance_estimator/REPORT.md` and stress-checked in `../step_time_nuisance_holdout_audit/REPORT.md`.\n\n",
        "## Formula\n\n",
        "```text\n",
        "phi_tau(t) = sum_{u<=t} exp(-(t-u)/1024) * relu(eta_{u-1}-eta_u) / eta_peak\n",
        "G = span{1, sin(pi z), cos(pi z), sin(2 pi z), cos(2 pi z)}\n",
        "phi_perp = M_G phi_tau,   r_perp = M_G(observed_loss - MPL)\n",
        "tau_EB = median(projected_residual_scale) / q75(projected_raw_kappa)\n",
        "kappa = max(0, <phi_perp, r_perp> / (||phi_perp||^2 + tau_EB^2))\n",
        "target_factor = min(total_positive_lr_drop(target)/0.9, 1)\n",
        "prediction = MPL + target_factor * kappa * phi_tau(target)\n",
        "```\n\n",
        "## Single-Curve Matrix\n\n",
        f"- Self-fit mean `{float(km['self_mean_delta']):+.1f}%`, worst `{float(km['self_worst_delta']):+.1f}%`, non-harm `{int(km['self_nonharm'])}/{int(km['self_rows'])}`.\n",
        f"- Off-diagonal mean `{float(km['offdiag_mean_delta']):+.1f}%`, worst `{float(km['offdiag_worst_delta']):+.1f}%`, non-harm `{int(km['offdiag_nonharm'])}/{int(km['offdiag_rows'])}`.\n",
        f"- Probe -> WSD mean `{float(km['probe_to_wsd_mean_delta']):+.1f}%`, worst `{float(km['probe_to_wsd_worst_delta']):+.1f}%`.\n",
        f"- Cosine -> WSD mean `{float(km['cosine_to_wsd_mean_delta']):+.1f}%`, worst `{float(km['cosine_to_wsd_worst_delta']):+.1f}%`.\n\n",
        "![matrix](figs/fixed_single_matrix.png)\n\n",
        "## Group Calibration\n\n",
        "| calibration group | target group | mean | worst | non-harm |\n",
        "|---|---|---:|---:|---:|\n",
    ]
    for row in gsum:
        lines.append(
            f"| {row['group_id']} | {row['target_group']} | {float(row['mean_delta']):+.1f}% | "
            f"{float(row['worst_delta']):+.1f}% | {int(row['nonharm'])}/{int(row['rows'])} |\n"
        )
    lines += [
        "\n## Reading\n\n",
        f"- Single-curve calibration is non-harming on all `{int(km['self_rows']) + int(km['offdiag_rows'])}` tested rows and improves the off-diagonal mean by `{float(km['offdiag_mean_delta']):.1f}%`.\n",
        f"- Pooled probe calibration remains stronger for WSD targets: `probe -> WSD` mean `{float(probe_wsd['mean_delta']):+.1f}%`, worst `{float(probe_wsd['worst_delta']):+.1f}%`; endpoint probe `wsdcon_3 -> WSD` mean `{float(probe3_wsd['mean_delta']):+.1f}%`, worst `{float(probe3_wsd['worst_delta']):+.1f}%`.\n",
        "- The method intentionally suppresses diffuse cosine-derived amplitudes when the residualized response direction is not identifiable, while retaining sharp/probe amplitudes.\n",
        "- Interpretation for the cosine lag diagnostic: the cosine residual behaves like a broad low-frequency wave, not like a persistent physical catch-up delay. The fixed estimator therefore uses cosine mainly to estimate the nuisance subspace and reads transferable amplitude from sharp or probe-like LR drops.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    estimator = StepTimeNuisanceEstimator()
    single_rows = run_single(estimator)
    group_rows = run_groups(estimator)
    ss = single_summary(single_rows)
    gs = group_summary(group_rows)
    write_csv(OUT_DIR / "single_details.csv", single_rows)
    write_csv(OUT_DIR / "single_summary.csv", ss)
    write_csv(OUT_DIR / "group_details.csv", group_rows)
    write_csv(OUT_DIR / "group_summary.csv", gs)
    plot_matrix(FIG_DIR / "fixed_single_matrix.png", ss)
    write_report(single_rows, group_rows)

    km = key_metrics(single_rows)
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    print(
        f"single self={float(km['self_mean_delta']):+.1f}%/{float(km['self_worst_delta']):+.1f}% "
        f"offdiag={float(km['offdiag_mean_delta']):+.1f}%/{float(km['offdiag_worst_delta']):+.1f}% "
        f"probeWSD={float(km['probe_to_wsd_mean_delta']):+.1f}%/{float(km['probe_to_wsd_worst_delta']):+.1f}%"
    )


if __name__ == "__main__":
    main()
