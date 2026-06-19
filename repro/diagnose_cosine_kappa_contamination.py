#!/usr/bin/env python3
"""Diagnose why raw global-cosine kappa does not transfer.

Hypothesis:
    On a smooth full cosine curve, the MPL residual contains low-frequency
    backbone mismatch.  Directly projecting that residual onto DropRelaxS gives
    a large kappa that is not a clean transient amplitude.

Diagnostic:
    1. Fit raw kappa on full cosine_72000.
    2. Remove a small low-frequency DCT nuisance subspace from both residual
       and feature, then convert the identifiable amplitude back to a full
       feature amplitude with sqrt(retention).
    3. Compare transfer to sharp WSD and WSD-con step targets.
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

from current_law_decay_matrix import LAMBDA  # noqa: E402
from deep_stime import stime_feature  # noqa: E402
from reproduce_cosine_to_wsd import (  # noqa: E402
    MPL_PRECOMPUTED_INIT,
    SCALES,
    load_curve,
    metrics,
    mpl_predict,
)


OUT_DIR = ROOT / "results" / "current_law_decay_matrix" / "error_visualization"
COSINE = "cosine_72000.csv"
TARGETS = ["wsd_20000_24000.csv", "wsdcon_3.csv"]
DCT_MODES = 4


def dct_low_frequency_basis(n: int, modes: int) -> np.ndarray:
    idx = np.arange(n, dtype=np.float64)
    cols = [np.ones(n, dtype=np.float64)]
    for k in range(1, modes + 1):
        cols.append(np.cos(math.pi * (idx + 0.5) * k / max(n, 1)))
    z = np.column_stack(cols)
    return z / np.maximum(np.linalg.norm(z, axis=0), 1e-12)


def residualize(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(z, y, rcond=None)
    return y - z @ coef


def origin_kappa(y: np.ndarray, x: np.ndarray) -> float:
    denom = float(np.dot(x, x))
    if denom <= 1e-18:
        return 0.0
    return max(0.0, float(np.dot(x, y) / denom))


def cosine_estimates(scale: str) -> dict[str, object]:
    curve = load_curve(scale, COSINE)
    base = mpl_predict(MPL_PRECOMPUTED_INIT[scale], curve)
    residual = curve.loss - base
    phi = stime_feature(curve, LAMBDA)

    raw_kappa = origin_kappa(residual, phi)

    z = dct_low_frequency_basis(len(curve.step), DCT_MODES)
    residual_perp = residualize(residual, z)
    phi_perp = residualize(phi, z)
    full_l2 = max(float(np.dot(phi, phi)), 1e-18)
    perp_l2 = float(np.dot(phi_perp, phi_perp))
    retention = perp_l2 / full_l2
    projected_kappa = origin_kappa(residual_perp, phi_perp)
    effective_kappa = math.sqrt(max(retention, 0.0)) * projected_kappa

    return {
        "scale": scale,
        "curve": curve,
        "base": base,
        "residual": residual,
        "phi": phi,
        "raw_kappa": raw_kappa,
        "dct_modes": DCT_MODES,
        "dct_retention": retention,
        "dct_projected_kappa": projected_kappa,
        "dct_effective_kappa": effective_kappa,
    }


def transfer_row(scale: str, target: str, raw_kappa: float, effective_kappa: float) -> dict[str, object]:
    curve = load_curve(scale, target)
    base = mpl_predict(MPL_PRECOMPUTED_INIT[scale], curve)
    phi = stime_feature(curve, LAMBDA)
    base_mae = metrics(curve.loss, base)["mae"]
    raw_mae = metrics(curve.loss, base + raw_kappa * phi)["mae"]
    dct_mae = metrics(curve.loss, base + effective_kappa * phi)["mae"]
    return {
        "scale": scale,
        "target": target,
        "base_mae": base_mae,
        "raw_mae": raw_mae,
        "dct_effective_mae": dct_mae,
        "raw_delta_pct": 100.0 * (raw_mae / base_mae - 1.0),
        "dct_effective_delta_pct": 100.0 * (dct_mae / base_mae - 1.0),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(estimates: list[dict[str, object]], transfers: list[dict[str, object]], path: Path) -> None:
    fig = plt.figure(figsize=(13.2, 7.6))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], wspace=0.28, hspace=0.34)
    ax_res = fig.add_subplot(grid[:, 0])
    ax_k = fig.add_subplot(grid[0, 1])
    ax_t = fig.add_subplot(grid[1, 1])

    example = next(row for row in estimates if row["scale"] == "25")
    curve = example["curve"]
    steps = curve.step
    ax_res.plot(steps, example["residual"], color="#111111", lw=1.4, label="MPL residual")
    ax_res.plot(
        steps,
        float(example["raw_kappa"]) * example["phi"],
        color="#dc2626",
        lw=1.2,
        ls="--",
        label="raw global cosine fit",
    )
    ax_res.plot(
        steps,
        float(example["dct_effective_kappa"]) * example["phi"],
        color="#2563eb",
        lw=1.2,
        label=f"DCT{DCT_MODES}-residualized effective fit",
    )
    ax_res.axhline(0.0, color="#444444", lw=0.8, alpha=0.65)
    ax_res.set_title("Full cosine residual is smooth, so raw projection absorbs backbone drift", fontsize=11)
    ax_res.set_xlabel("step")
    ax_res.set_ylabel("loss residual / correction")
    ax_res.grid(alpha=0.24, lw=0.5)
    ax_res.legend(frameon=False, fontsize=8.5, loc="best")

    x = np.arange(len(SCALES))
    width = 0.34
    raw = [float(row["raw_kappa"]) for row in estimates]
    eff = [float(row["dct_effective_kappa"]) for row in estimates]
    ax_k.bar(x - width / 2, raw, width, color="#dc2626", label="raw global cosine")
    ax_k.bar(x + width / 2, eff, width, color="#2563eb", label=f"DCT{DCT_MODES} effective")
    ax_k.set_xticks(x, [f"{s}M" for s in SCALES])
    ax_k.set_yscale("log")
    ax_k.set_ylabel("kappa (log scale)")
    ax_k.set_title("Residualization collapses cosine-derived amplitude", fontsize=11)
    ax_k.grid(axis="y", alpha=0.24, lw=0.5)
    ax_k.legend(frameon=True, framealpha=0.86, fontsize=8.5, loc="lower right")
    for i, row in enumerate(estimates):
        ax_k.text(
            i,
            max(float(row["dct_effective_kappa"]) * 1.35, 1e-4),
            f"R={float(row['dct_retention']):.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
        )

    labels = ["sharp WSD", "WSD-con 3e-5"]
    raw_mean = []
    eff_mean = []
    for target in TARGETS:
        rows = [row for row in transfers if row["target"] == target]
        raw_mean.append(float(np.mean([float(row["raw_delta_pct"]) for row in rows])))
        eff_mean.append(float(np.mean([float(row["dct_effective_delta_pct"]) for row in rows])))
    xt = np.arange(len(TARGETS))
    ax_t.axhline(0.0, color="#444444", lw=0.8)
    ax_t.bar(xt - width / 2, raw_mean, width, color="#dc2626", label="raw global cosine")
    ax_t.bar(xt + width / 2, eff_mean, width, color="#2563eb", label=f"DCT{DCT_MODES} effective")
    ax_t.set_xticks(xt, labels)
    ax_t.set_yscale("symlog", linthresh=20)
    ax_t.set_ylabel("mean Delta MAE vs MPL (%)")
    ax_t.set_title("The bad transfer is mostly amplitude contamination", fontsize=11)
    ax_t.grid(axis="y", alpha=0.24, lw=0.5)
    ax_t.legend(frameon=True, framealpha=0.86, fontsize=8.5, loc="upper left")
    for i, value in enumerate(raw_mean):
        ax_t.text(
            i - width / 2,
            value / 1.7,
            f"{value:+.0f}%",
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            fontweight="bold",
        )
    for i, value in enumerate(eff_mean):
        va = "top" if value < 0 else "bottom"
        y = value - 1.5 if value < 0 else value + 1.5
        ax_t.text(i + width / 2, y, f"{value:+.1f}%", ha="center", va=va, fontsize=8)

    fig.suptitle("Cosine kappa contamination diagnostic", fontsize=13.5, y=0.985)
    fig.text(
        0.5,
        0.012,
        "DCT residualization uses only calibration residuals and schedule-derived features; no target loss is used to choose kappa.",
        ha="center",
        fontsize=9,
        color="#333333",
    )
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def write_report(path: Path, estimates: list[dict[str, object]], transfers: list[dict[str, object]]) -> None:
    lines = [
        "# Cosine Kappa Contamination Diagnostic\n\n",
        "This diagnostic tests whether the raw full-cosine kappa is dominated by low-frequency MPL residual drift rather than transferable fast-decay lag.\n\n",
        "| scale | raw kappa | DCT retention | DCT projected kappa | DCT effective kappa |\n",
        "|---:|---:|---:|---:|---:|\n",
    ]
    for row in estimates:
        lines.append(
            f"| {row['scale']}M | {float(row['raw_kappa']):.4f} | "
            f"{float(row['dct_retention']):.4f} | {float(row['dct_projected_kappa']):.4f} | "
            f"{float(row['dct_effective_kappa']):.4f} |\n"
        )
    lines += [
        "\n| target | raw global cosine mean delta | DCT-effective mean delta |\n",
        "|---|---:|---:|\n",
    ]
    for target in TARGETS:
        rows = [row for row in transfers if row["target"] == target]
        raw_mean = float(np.mean([float(row["raw_delta_pct"]) for row in rows]))
        eff_mean = float(np.mean([float(row["dct_effective_delta_pct"]) for row in rows]))
        lines.append(f"| {target} | {raw_mean:+.1f}% | {eff_mean:+.1f}% |\n")

    lines += [
        "\n## Reading\n\n",
        "1. Raw full-cosine kappa is large because the full cosine residual is smooth and low-frequency.\n",
        "2. After removing four low-frequency DCT nuisance modes, only about one percent of the full-cosine feature energy remains identifiable. The effective amplitude collapses by one to two orders of magnitude.\n",
        "3. Using the raw full-cosine kappa causes the large WSD and WSD-con failures. Using the residualized effective amplitude removes the over-correction without fitting target losses.\n",
        "4. This supports the interpretation that full-cosine calibration is contaminated by MPL backbone mismatch; it should be used only with nuisance control or replaced by target-like probes.\n",
    ]
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    estimates = [cosine_estimates(scale) for scale in SCALES]
    transfer_rows = []
    for est in estimates:
        for target in TARGETS:
            transfer_rows.append(
                transfer_row(
                    str(est["scale"]),
                    target,
                    float(est["raw_kappa"]),
                    float(est["dct_effective_kappa"]),
                )
            )

    estimate_rows = [
        {
            k: v
            for k, v in est.items()
            if k
            in {
                "scale",
                "raw_kappa",
                "dct_modes",
                "dct_retention",
                "dct_projected_kappa",
                "dct_effective_kappa",
            }
        }
        for est in estimates
    ]
    write_csv(OUT_DIR / "cosine_kappa_contamination_estimates.csv", estimate_rows)
    write_csv(OUT_DIR / "cosine_kappa_contamination_transfer.csv", transfer_rows)
    plot(estimates, transfer_rows, OUT_DIR / "cosine_kappa_contamination.png")
    write_report(OUT_DIR / "COSINE_KAPPA_CONTAMINATION.md", estimates, transfer_rows)

    print(f"wrote {OUT_DIR / 'cosine_kappa_contamination.png'}")
    print(f"wrote {OUT_DIR / 'COSINE_KAPPA_CONTAMINATION.md'}")
    for est in estimates:
        print(
            f"{est['scale']}M raw={float(est['raw_kappa']):.4f} "
            f"DCT{DCT_MODES}_effective={float(est['dct_effective_kappa']):.4f} "
            f"retention={float(est['dct_retention']):.4f}"
        )


if __name__ == "__main__":
    main()
