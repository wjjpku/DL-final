#!/usr/bin/env python3
"""Visualize how source prefix dropping changes one target error estimate."""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

TMP_CACHE = Path(os.environ.get("TMPDIR", "/tmp")) / "dl-final-cache"
TMP_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(TMP_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(TMP_CACHE / "xdg"))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import interpretable_error_model as iem  # noqa: E402
import interpretable_theory_refinement_audit as tra  # noqa: E402
import source_data_drop_ablation as sda  # noqa: E402

OUT_DIR = iem.ROOT / "results" / "source_data_drop_ablation" / "single_experiment"
FIG_DIR = OUT_DIR / "figs"

SCALE = "100"
TARGET_CURVE = "wsdcon_3.csv"
TARGET_LABEL = "100M WSD-con 3e-5"
FIT_STARTS = [2160, 5000, 6500, 8000, 10000, 12000]

COLORS = {
    2160: "#7c3aed",
    5000: "#dc2626",
    6500: "#f97316",
    8000: "#2563eb",
    10000: "#059669",
    12000: "#6b7280",
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def smooth(y: np.ndarray) -> np.ndarray:
    if len(y) < 9:
        return y.copy()
    window = max(5, int(round(0.025 * len(y))))
    if window % 2 == 0:
        window += 1
    window = min(window, len(y) if len(y) % 2 == 1 else len(y) - 1)
    if window < 3:
        return y.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(y, kernel, mode="same")


def suffix_config(start: int) -> dict[str, object]:
    return {
        "config": f"suffix_ge_{start}",
        "experiment": "single_experiment_prefix_drop",
        "mode": "suffix",
        "start": start,
        "end": "",
        "block_label": "",
        "reference_config": "",
    }


def analyze() -> tuple[list[dict[str, object]], dict[str, object]]:
    cache: dict[tuple[str, str], iem.CurvePack] = {}
    tangent_cache: dict[str, np.ndarray] = {}
    basis_cache: dict[tuple[str, str], np.ndarray] = {}

    source = sda.load_pack(SCALE, iem.TRAIN_CURVE, cache)
    target = sda.load_pack(SCALE, TARGET_CURVE, cache)
    residual = target.curve.loss - target.baseline
    lam = tra.response_lambda(target.curve, "q2", "halflife")
    factor = tra.locality_factor(target.curve, "support_projection")
    feature = iem.causal_drop_response(target.curve, lam)

    rows: list[dict[str, object]] = []
    estimates: dict[int, dict[str, object]] = {}
    for start in FIT_STARTS:
        cfg = suffix_config(start)
        mask = sda.source_mask(source, cfg)
        coef, info = sda.fit_coefficient_mask(
            source,
            lam,
            mask,
            tangent_cache,
            basis_cache,
            str(cfg["config"]),
        )
        correction = factor * coef * feature
        remaining = residual - correction
        pred = target.baseline + correction
        mae = iem.mae(target.curve.loss, pred)
        delta = 100.0 * (mae / target.base_mae - 1.0)
        row = {
            "scale": SCALE,
            "target_curve": TARGET_CURVE,
            "target_label": TARGET_LABEL,
            "fit_start": start,
            "n_cal": info["n_cal"],
            "lambda": lam,
            "locality_factor": factor,
            "coef": coef,
            "base_mae": target.base_mae,
            "corr_mae": mae,
            "delta_pct": delta,
            "correction_l1": float(np.mean(np.abs(correction))),
            "true_residual_l1": float(np.mean(np.abs(residual))),
            "correction_to_residual_l1": float(np.mean(np.abs(correction)) / max(np.mean(np.abs(residual)), 1e-18)),
            "correction_max_abs": float(np.max(np.abs(correction))),
            "true_residual_max_abs": float(np.max(np.abs(residual))),
            "source_retention": info["source_retention"],
            "source_perp_norm": info["source_perp_norm"],
            "source_full_norm": info["source_full_norm"],
            "source_dot": info["source_dot"],
            "denominator": info["denominator"],
            "ridge": info["ridge"],
        }
        rows.append(row)
        estimates[start] = {
            "correction": correction,
            "remaining": remaining,
            "row": row,
        }

    panel = {
        "source": source,
        "target": target,
        "residual": residual,
        "lambda": lam,
        "locality_factor": factor,
        "estimates": estimates,
    }
    return rows, panel


def ylim_for(arrays: list[np.ndarray], pad: float = 1.12) -> tuple[float, float]:
    limit = max(float(np.max(np.abs(a))) for a in arrays)
    limit = max(limit, 1e-8)
    return -pad * limit, pad * limit


def plot_estimates(panel: dict[str, object], zoom: bool) -> None:
    target = panel["target"]
    residual = panel["residual"]
    estimates = panel["estimates"]
    steps = target.curve.step

    fig, ax = plt.subplots(figsize=(11.0, 4.8), constrained_layout=True)
    ax.axhline(0.0, color="#111827", lw=0.8, alpha=0.75)
    ax.plot(steps, smooth(residual), color="#111827", lw=1.65, label="true MPL residual")

    arrays = [residual]
    for start in FIT_STARTS:
        item = estimates[start]
        row = item["row"]
        correction = item["correction"]
        arrays.append(correction)
        ax.plot(
            steps,
            smooth(correction),
            color=COLORS[start],
            lw=1.45 if start == 8000 else 1.15,
            linestyle="-" if start == 8000 else "--",
            label=f"drop <{start/1000:g}k: {float(row['delta_pct']):+.1f}%, k={float(row['coef']):.3g}",
        )

    if zoom:
        arrays = [residual, estimates[8000]["correction"], estimates[10000]["correction"], estimates[12000]["correction"]]
        ax.set_ylim(*ylim_for(arrays, pad=1.45))
    else:
        ax.set_ylim(*ylim_for(arrays))

    ax2 = ax.twinx()
    ax2.plot(steps, target.curve.lrs[target.curve.step] / iem.PEAK_LR, color="#b45309", lw=0.9, alpha=0.24)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_yticks([])

    suffix = "zoom" if zoom else "full"
    ax.set_title(f"{TARGET_LABEL}: estimated residual after different source prefix drops ({suffix})")
    ax.set_xlabel("target step")
    ax.set_ylabel("estimated error / true residual")
    ax.legend(fontsize=7.2, loc="best")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"estimated_residual_by_fit_start_{suffix}.png", dpi=180)
    plt.close(fig)


def plot_remaining(panel: dict[str, object], zoom: bool) -> None:
    target = panel["target"]
    residual = panel["residual"]
    estimates = panel["estimates"]
    steps = target.curve.step

    fig, ax = plt.subplots(figsize=(11.0, 4.8), constrained_layout=True)
    ax.axhline(0.0, color="#111827", lw=0.8, alpha=0.75)
    ax.plot(steps, smooth(residual), color="#111827", lw=1.65, label="MPL residual")

    arrays = [residual]
    for start in FIT_STARTS:
        item = estimates[start]
        row = item["row"]
        remaining = item["remaining"]
        arrays.append(remaining)
        ax.plot(
            steps,
            smooth(remaining),
            color=COLORS[start],
            lw=1.45 if start == 8000 else 1.15,
            linestyle="-" if start == 8000 else "--",
            label=f"remaining drop <{start/1000:g}k: {float(row['delta_pct']):+.1f}%",
        )

    if zoom:
        arrays = [residual, estimates[8000]["remaining"], estimates[10000]["remaining"], estimates[12000]["remaining"]]
        ax.set_ylim(*ylim_for(arrays, pad=1.45))
    else:
        ax.set_ylim(*ylim_for(arrays))

    ax2 = ax.twinx()
    ax2.plot(steps, target.curve.lrs[target.curve.step] / iem.PEAK_LR, color="#b45309", lw=0.9, alpha=0.24)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_yticks([])

    suffix = "zoom" if zoom else "full"
    ax.set_title(f"{TARGET_LABEL}: remaining residual after correction ({suffix})")
    ax.set_xlabel("target step")
    ax.set_ylabel("remaining residual")
    ax.legend(fontsize=7.2, loc="best")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"remaining_residual_by_fit_start_{suffix}.png", dpi=180)
    plt.close(fig)


def write_report(rows: list[dict[str, object]]) -> None:
    lines = [
        "# Single-Experiment Prefix Drop Error Estimate\n\n",
        f"固定 target：`{TARGET_LABEL}`。只改变 cosine source 中参与 \\(\\kappa\\) 拟合的 prefix drop 边界。",
        "目标 loss 只用于画真实 residual 和计算 MAE。\n\n",
        "## Formula\n\n",
        "\\[\n",
        "\\widehat e_s(t)=a_s\\widehat\\kappa_s\\phi_{\\lambda_s,s}(t),\\qquad ",
        "\\widehat L_s(t)=L_{MPL,s}(t)+\\widehat e_s(t).\n",
        "\\]\n\n",
        "每个 `fit_start` 都重新在 source mask 上构造 MPL-LD tangent projection，并重新估计 ",
        "\\(\\widehat\\kappa_s\\)。\n\n",
        "## Metrics\n\n",
        "| fit_start | n_cal | kappa | source retention | correction / true residual L1 | MAE delta |\n",
        "| ---: | ---: | ---: | ---: | ---: | ---: |\n",
    ]
    for row in rows:
        lines.append(
            f"| {int(row['fit_start'])} | {float(row['n_cal']):.0f} | {float(row['coef']):.6g} | "
            f"{float(row['source_retention']):.4g} | {float(row['correction_to_residual_l1']):.2f}x | "
            f"{float(row['delta_pct']):+.2f}% |\n"
        )
    lines += [
        "\n## Figures\n\n",
        "- `figs/estimated_residual_by_fit_start_full.png`\n",
        "- `figs/estimated_residual_by_fit_start_zoom.png`\n",
        "- `figs/remaining_residual_by_fit_start_full.png`\n",
        "- `figs/remaining_residual_by_fit_start_zoom.png`\n",
        "\n## Reading\n\n",
        "`fit_start=2160/5000/6500` 会把误差估计放大成过大的 positive spike；",
        "`fit_start=8000` 的估计幅度与真实 MPL residual 最接近；",
        "`fit_start=10000/12000` 则开始欠估计，说明 8k-10k 附近是把幅度定准的关键 source 区间。\n",
    ]
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    rows, panel = analyze()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "metrics.csv", rows)
    plot_estimates(panel, zoom=False)
    plot_estimates(panel, zoom=True)
    plot_remaining(panel, zoom=False)
    plot_remaining(panel, zoom=True)
    write_report(rows)
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
