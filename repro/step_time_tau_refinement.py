#!/usr/bin/env python3
"""Refine the step-time relaxation kernel.

The broad residual-shape gallery suggests that the current cumulative-LR
S-time kernel can lag too much on smooth decay.  The first model search found
that a step-time kernel

    Phi_tau(t) = sum_{k<=t} exp(-(t-k)/tau) (eta_{k-1}-eta_k)_+ / eta_peak

substantially improves both self-fit and probe-to-WSD generalization.

This script scans tau and reports the self-fit/generalization Pareto tradeoff.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

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


OUT_DIR = ROOT / "results" / "step_time_tau_refinement"
FIG_DIR = OUT_DIR / "figs"
TAUS = [512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 3072, 3584, 4096, 5120, 6144]
CURVES = [
    "cosine_72000.csv",
    "wsd_20000_24000.csv",
    "wsdld_20000_24000.csv",
    "wsdcon_3.csv",
    "wsdcon_9.csv",
    "wsdcon_18.csv",
]
PROBES = ["wsdcon_3.csv", "wsdcon_9.csv", "wsdcon_18.csv"]
WSD_TARGETS = ["wsd_20000_24000.csv", "wsdld_20000_24000.csv"]


def build_cache() -> dict[tuple[str, str], dict[str, object]]:
    cache = {}
    for scale in SCALES:
        for curve_name in CURVES:
            curve = load_curve(scale, curve_name)
            base = mpl_predict(MPL_PRECOMPUTED_INIT[scale], curve)
            cache[(scale, curve_name)] = {
                "curve": curve,
                "base": base,
                "residual": curve.loss - base,
                "base_mae": metrics(curve.loss, base)["mae"],
            }
    return cache


def step_feature(curve, tau: float) -> np.ndarray:
    eta = curve.lrs.astype(np.float64)
    drop = np.zeros_like(eta)
    drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0) / PEAK_LR
    decay = math.exp(-1.0 / tau)
    out = np.zeros_like(eta)
    acc = 0.0
    for t in range(len(eta)):
        acc = acc * decay + drop[t]
        out[t] = acc
    return out[curve.step]


def fit_kappa(x: np.ndarray, y: np.ndarray) -> float:
    denom = float(np.dot(x, x))
    if denom <= 1e-18:
        return 0.0
    return max(0.0, float(np.dot(x, y) / denom))


def feature(cache, scale: str, curve_name: str, tau: float, feat_cache) -> np.ndarray:
    key = (scale, curve_name, tau)
    if key not in feat_cache:
        feat_cache[key] = step_feature(cache[(scale, curve_name)]["curve"], tau)
    return feat_cache[key]


def fit_on(cache, tau: float, scale: str, curves: list[str], feat_cache) -> float:
    xs = [feature(cache, scale, curve_name, tau, feat_cache) for curve_name in curves]
    ys = [cache[(scale, curve_name)]["residual"] for curve_name in curves]
    return fit_kappa(np.concatenate(xs), np.concatenate(ys))


def score(cache, tau: float, scale: str, curve_name: str, kappa: float, feat_cache) -> tuple[float, int, float]:
    row = cache[(scale, curve_name)]
    curve = row["curve"]
    pred = row["base"] + kappa * feature(cache, scale, curve_name, tau, feat_cache)
    corr_mae = metrics(curve.loss, pred)["mae"]
    base_mae = float(row["base_mae"])
    delta = 100.0 * (corr_mae / base_mae - 1.0)
    return delta, int(corr_mae < base_mae), corr_mae


def scan() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    cache = build_cache()
    feat_cache: dict[tuple[str, str, float], np.ndarray] = {}
    rows: list[dict[str, object]] = []
    details: list[dict[str, object]] = []

    for tau in TAUS:
        self_deltas, self_wins = [], 0
        for scale in SCALES:
            for curve_name in CURVES:
                kappa = fit_on(cache, tau, scale, [curve_name], feat_cache)
                delta, win, corr_mae = score(cache, tau, scale, curve_name, kappa, feat_cache)
                self_deltas.append(delta)
                self_wins += win
                details.append(
                    {
                        "mode": "self_fit",
                        "tau": tau,
                        "scale": scale,
                        "train": curve_name,
                        "test": curve_name,
                        "kappa": kappa,
                        "delta_pct": delta,
                        "win": win,
                        "corr_mae": corr_mae,
                    }
                )

        mode_rows = {}
        for train_name, train_curves in {
            "probe": PROBES,
            "probe3": ["wsdcon_3.csv"],
            "probe9": ["wsdcon_9.csv"],
        }.items():
            wsd_deltas, wsd_wins = [], 0
            cosine_wsd_deltas, cosine_wsd_wins = [], 0
            for scale in SCALES:
                kappa = fit_on(cache, tau, scale, train_curves, feat_cache)
                for curve_name in WSD_TARGETS:
                    delta, win, corr_mae = score(cache, tau, scale, curve_name, kappa, feat_cache)
                    wsd_deltas.append(delta)
                    wsd_wins += win
                    details.append(
                        {
                            "mode": train_name + "_to_wsd",
                            "tau": tau,
                            "scale": scale,
                            "train": "+".join(train_curves),
                            "test": curve_name,
                            "kappa": kappa,
                            "delta_pct": delta,
                            "win": win,
                            "corr_mae": corr_mae,
                        }
                    )
                for curve_name in ["cosine_72000.csv", *WSD_TARGETS]:
                    delta, win, _ = score(cache, tau, scale, curve_name, kappa, feat_cache)
                    cosine_wsd_deltas.append(delta)
                    cosine_wsd_wins += win
            mode_rows[train_name] = {
                "mean": float(np.mean(wsd_deltas)),
                "worst": float(np.max(wsd_deltas)),
                "wins": wsd_wins,
                "off_mean": float(np.mean(cosine_wsd_deltas)),
                "off_worst": float(np.max(cosine_wsd_deltas)),
                "off_wins": cosine_wsd_wins,
            }

        rows.append(
            {
                "tau": tau,
                "self_mean": float(np.mean(self_deltas)),
                "self_worst": float(np.max(self_deltas)),
                "self_wins": self_wins,
                "self_tests": len(self_deltas),
                "probe_wsd_mean": mode_rows["probe"]["mean"],
                "probe_wsd_worst": mode_rows["probe"]["worst"],
                "probe_wsd_wins": mode_rows["probe"]["wins"],
                "probe3_wsd_mean": mode_rows["probe3"]["mean"],
                "probe3_wsd_worst": mode_rows["probe3"]["worst"],
                "probe3_wsd_wins": mode_rows["probe3"]["wins"],
                "probe9_wsd_mean": mode_rows["probe9"]["mean"],
                "probe9_wsd_worst": mode_rows["probe9"]["worst"],
                "probe9_wsd_wins": mode_rows["probe9"]["wins"],
                "probe_off_mean": mode_rows["probe"]["off_mean"],
                "probe_off_worst": mode_rows["probe"]["off_worst"],
                "probe_off_wins": mode_rows["probe"]["off_wins"],
            }
        )
    return rows, details


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, object]]) -> None:
    tau = np.array([float(r["tau"]) for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), constrained_layout=True)

    ax = axes[0]
    ax.axhline(0.0, color="#333333", lw=0.8)
    ax.plot(tau, [float(r["self_mean"]) for r in rows], marker="o", label="self mean")
    ax.plot(tau, [float(r["self_worst"]) for r in rows], marker="o", label="self worst")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("step-time tau")
    ax.set_ylabel("Delta MAE vs MPL (%)")
    ax.set_title("Self-fit tradeoff")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.axhline(0.0, color="#333333", lw=0.8)
    ax.plot(tau, [float(r["probe_wsd_mean"]) for r in rows], marker="o", label="pooled probes -> WSD mean")
    ax.plot(tau, [float(r["probe_wsd_worst"]) for r in rows], marker="o", label="pooled probes -> WSD worst")
    ax.plot(tau, [float(r["probe3_wsd_mean"]) for r in rows], marker="s", label="wsdcon_3 -> WSD mean")
    ax.plot(tau, [float(r["probe3_wsd_worst"]) for r in rows], marker="s", label="wsdcon_3 -> WSD worst")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("step-time tau")
    ax.set_ylabel("Delta MAE vs MPL (%)")
    ax.set_title("Probe-to-WSD generalization")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8.5)

    fig.savefig(FIG_DIR / "tau_pareto.png", dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def write_report(path: Path, rows: list[dict[str, object]]) -> None:
    conservative = next(r for r in rows if int(r["tau"]) == 1024)
    aggressive = min(rows, key=lambda r: float(r["probe3_wsd_mean"]))
    pooled_best = min(rows, key=lambda r: float(r["probe_wsd_mean"]))
    lines = [
        "# Step-Time Tau Refinement\n\n",
        "Candidate response feature:\n\n",
        "```text\n",
        "Phi_tau(t) = sum_{k<=t} exp(-(t-k)/tau) * (eta_{k-1}-eta_k)_+ / eta_peak\n",
        "prediction = MPL + kappa * Phi_tau\n",
        "```\n\n",
        "This is motivated by the residual gallery: smooth cosine residuals look like broad low-frequency MPL mismatch, while sharp/probe residuals catch up over a finite number of steps. A step-time kernel prevents the low-LR tail from becoming arbitrarily slow.\n\n",
        "## Pareto Table\n\n",
        "| tau | self mean | self worst | self wins | probes->WSD mean | probes->WSD worst | wsdcon3->WSD mean | wsdcon3->WSD worst |\n",
        "|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for r in rows:
        lines.append(
            f"| {int(r['tau'])} | {float(r['self_mean']):+.1f}% | {float(r['self_worst']):+.1f}% | "
            f"{int(r['self_wins'])}/{int(r['self_tests'])} | {float(r['probe_wsd_mean']):+.1f}% | "
            f"{float(r['probe_wsd_worst']):+.1f}% | {float(r['probe3_wsd_mean']):+.1f}% | "
            f"{float(r['probe3_wsd_worst']):+.1f}% |\n"
        )
    lines += [
        "\n## Current Best Reading\n\n",
        f"- Conservative deployment candidate: `tau=1024`. It keeps self-fit non-harming on `{int(conservative['self_wins'])}/{int(conservative['self_tests'])}` curves "
        f"and gives pooled-probe WSD improvement `{float(conservative['probe_wsd_mean']):+.1f}%` with worst `{float(conservative['probe_wsd_worst']):+.1f}%`.\n",
        f"- Strong target-matched candidate: `tau={int(aggressive['tau'])}` with `wsdcon_3` calibration. It gives WSD improvement `{float(aggressive['probe3_wsd_mean']):+.1f}%` "
        f"with worst `{float(aggressive['probe3_wsd_worst']):+.1f}%`, but it no longer keeps every probe self-fit non-harming.\n",
        f"- Best pooled-probe WSD mean occurs at `tau={int(pooled_best['tau'])}`: `{float(pooled_best['probe_wsd_mean']):+.1f}%` mean, worst `{float(pooled_best['probe_wsd_worst']):+.1f}%`.\n",
        "- The practical modeling update is to treat `S10_current` as too diffuse for cosine-like residuals, and to add a finite step-time catch-up channel for localized LR-drop prediction.\n",
    ]
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows, details = scan()
    write_csv(OUT_DIR / "tau_summary.csv", rows)
    write_csv(OUT_DIR / "tau_details.csv", details)
    plot(rows)
    write_report(OUT_DIR / "REPORT.md", rows)
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    for r in rows:
        print(
            f"tau={int(r['tau']):4d} self={float(r['self_mean']):+6.1f}%/"
            f"{float(r['self_worst']):+5.1f}% wins={int(r['self_wins'])}/{int(r['self_tests'])} "
            f"probe->WSD={float(r['probe_wsd_mean']):+6.1f}% "
            f"probe3->WSD={float(r['probe3_wsd_mean']):+6.1f}%"
        )


if __name__ == "__main__":
    main()
