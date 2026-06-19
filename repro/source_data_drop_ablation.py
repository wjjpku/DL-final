#!/usr/bin/env python3
"""Ablate which cosine source-calibration points matter for kappa fitting."""
from __future__ import annotations

import csv
import math
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
import interpretable_nuisance_origin_audit as noa  # noqa: E402
import interpretable_theory_refinement_audit as tra  # noqa: E402

OUT_DIR = iem.ROOT / "results" / "source_data_drop_ablation"
FIG_DIR = OUT_DIR / "figs"

PREFIX_STARTS = [2160, 5000, 6500, 8000, 10000, 12000, 16000, 20000, 24000, 32000, 48000, 60000]
SOURCE_BLOCKS = [
    (2160, 5000, "2.16k-5k"),
    (5000, 8000, "5k-8k"),
    (8000, 12000, "8k-12k"),
    (12000, 20000, "12k-20k"),
    (20000, 32000, "20k-32k"),
    (32000, 48000, "32k-48k"),
    (48000, 60000, "48k-60k"),
    (60000, 72050, "60k-72k"),
]
CURRENT_SUFFIX_START = 8000
MIN_CAL = 12


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_pack(scale: str, curve_name: str, cache: dict[tuple[str, str], iem.CurvePack]) -> iem.CurvePack:
    key = (scale, curve_name)
    if key not in cache:
        cache[key] = tra.load_pack(scale, curve_name, cache)
    return cache[key]


def orthonormal_columns(z: np.ndarray) -> np.ndarray:
    cols: list[np.ndarray] = []
    for idx in range(z.shape[1]):
        col = z[:, idx].astype(np.float64)
        norm = float(np.linalg.norm(col))
        if norm > 1e-12:
            cols.append(col / norm)
    if not cols:
        return np.zeros((z.shape[0], 0), dtype=np.float64)
    q, r = np.linalg.qr(np.column_stack(cols))
    keep = np.abs(np.diag(r)) > 1e-8
    return q[:, keep]


def tangent_columns(pack: iem.CurvePack) -> np.ndarray:
    params = np.array(iem.MPL_PRECOMPUTED_INIT[pack.curve.scale], dtype=np.float64)
    curve = pack.curve
    cols: list[np.ndarray] = []
    eps = 1e-4
    idx_by_name = {
        "logB": 3,
        "logC": 4,
        "logBeta": 5,
        "logGamma": 6,
    }
    for name in ["logB", "logC", "logBeta", "logGamma"]:
        idx = idx_by_name[name]
        pp = params.copy()
        pm = params.copy()
        pp[idx] = params[idx] * math.exp(eps)
        pm[idx] = params[idx] * math.exp(-eps)
        cols.append((iem.mpl_predict(pp, curve) - iem.mpl_predict(pm, curve)) / (2.0 * eps))
    return np.column_stack(cols)


def fit_coefficient_mask(
    source: iem.CurvePack,
    lam: float,
    mask: np.ndarray,
    tangent_cache: dict[str, np.ndarray],
    basis_cache: dict[tuple[str, str], np.ndarray],
    config_name: str,
) -> tuple[float, dict[str, float]]:
    n_cal = int(np.sum(mask))
    if n_cal < MIN_CAL:
        return 0.0, {
            "basis_dim": 0,
            "n_cal": n_cal,
            "ridge": float("nan"),
            "source_dot": 0.0,
            "source_full_norm": 0.0,
            "source_perp_norm": 0.0,
            "source_retention": 0.0,
            "denominator": float("nan"),
            "too_few_points": 1.0,
        }

    if source.curve.scale not in tangent_cache:
        tangent_cache[source.curve.scale] = tangent_columns(source)
    basis_key = (source.curve.scale, config_name)
    if basis_key not in basis_cache:
        basis_cache[basis_key] = orthonormal_columns(tangent_cache[source.curve.scale][mask])
    q = basis_cache[basis_key]

    phi = iem.causal_drop_response(source.curve, lam)[mask]
    residual = source.residual[mask]
    phi_o = phi - q @ (q.T @ phi)
    residual_o = residual - q @ (q.T @ residual)

    dot = max(0.0, float(np.dot(phi_o, residual_o)))
    full_norm = float(np.linalg.norm(phi))
    perp_norm = float(np.linalg.norm(phi_o))
    perp_energy = perp_norm * perp_norm
    ridge = 1.0 / max(n_cal, 1)
    denom = perp_energy + ridge
    return dot / max(denom, 1e-18), {
        "basis_dim": float(q.shape[1]),
        "n_cal": float(n_cal),
        "ridge": float(ridge),
        "source_dot": float(dot),
        "source_full_norm": full_norm,
        "source_perp_norm": perp_norm,
        "source_retention": float(perp_energy / max(full_norm * full_norm, 1e-18)),
        "denominator": float(denom),
        "too_few_points": 0.0,
    }


def config_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for start in PREFIX_STARTS:
        rows.append(
            {
                "config": f"suffix_ge_{start}",
                "experiment": "prefix_start_sweep",
                "mode": "suffix",
                "start": start,
                "end": "",
                "block_label": "",
                "reference_config": "",
            }
        )
    for start, end, label in SOURCE_BLOCKS:
        if end <= CURRENT_SUFFIX_START:
            continue
        block_start = max(start, CURRENT_SUFFIX_START)
        rows.append(
            {
                "config": f"drop_{label}_from_suffix8k",
                "experiment": "leave_one_block_out_suffix8k",
                "mode": "drop_block_suffix",
                "start": block_start,
                "end": end,
                "block_label": label,
                "reference_config": f"suffix_ge_{CURRENT_SUFFIX_START}",
            }
        )
    for start, end, label in SOURCE_BLOCKS:
        rows.append(
            {
                "config": f"only_{label}",
                "experiment": "only_window",
                "mode": "only_window",
                "start": start,
                "end": end,
                "block_label": label,
                "reference_config": "",
            }
        )
    return rows


def source_mask(source: iem.CurvePack, cfg: dict[str, object]) -> np.ndarray:
    step = source.curve.step
    mode = str(cfg["mode"])
    start = int(cfg["start"])
    end_raw = cfg["end"]
    end = int(end_raw) if end_raw != "" else None
    if mode == "suffix":
        return step >= start
    if mode == "drop_block_suffix":
        assert end is not None
        return (step >= CURRENT_SUFFIX_START) & ~((step >= start) & (step < end))
    if mode == "only_window":
        assert end is not None
        return (step >= start) & (step < end)
    raise ValueError(f"unknown mode: {mode}")


def evaluate() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    configs = config_rows()
    detail_rows: list[dict[str, object]] = []
    coef_rows: list[dict[str, object]] = []
    cache: dict[tuple[str, str], iem.CurvePack] = {}
    tangent_cache: dict[str, np.ndarray] = {}
    basis_cache: dict[tuple[str, str], np.ndarray] = {}
    seen_coef: set[tuple[str, str, str, str]] = set()

    for cfg in configs:
        config_name = str(cfg["config"])
        for train_scale in iem.SCALES:
            source = load_pack(train_scale, iem.TRAIN_CURVE, cache)
            mask = source_mask(source, cfg)
            mask_step_min = int(np.min(source.curve.step[mask])) if np.any(mask) else -1
            mask_step_max = int(np.max(source.curve.step[mask])) if np.any(mask) else -1
            for test_scale in iem.SCALES:
                for group, curve_name, label in noa.ALL_TARGETS:
                    target = load_pack(test_scale, curve_name, cache)
                    lam = tra.response_lambda(target.curve, "q2", "halflife")
                    coef, info = fit_coefficient_mask(
                        source,
                        lam,
                        mask,
                        tangent_cache,
                        basis_cache,
                        config_name,
                    )
                    factor = tra.locality_factor(target.curve, "support_projection")
                    correction = factor * coef * iem.causal_drop_response(target.curve, lam)
                    pred = target.baseline + correction
                    corr_mae = iem.mae(target.curve.loss, pred)
                    delta = 100.0 * (corr_mae / target.base_mae - 1.0)
                    row = {
                        **cfg,
                        "train_scale": train_scale,
                        "test_scale": test_scale,
                        "group": group,
                        "test_curve": curve_name,
                        "test_label": label,
                        "lambda": lam,
                        "coef": coef,
                        "locality_factor": factor,
                        "base_mae": target.base_mae,
                        "corr_mae": corr_mae,
                        "delta_pct": delta,
                        "win": int(corr_mae < target.base_mae),
                        "nonharm": int(delta <= 1e-12),
                        "mask_step_min": mask_step_min,
                        "mask_step_max": mask_step_max,
                        **info,
                    }
                    detail_rows.append(row)
                    coef_key = (config_name, train_scale, curve_name, f"{lam:.12g}")
                    if coef_key not in seen_coef:
                        seen_coef.add(coef_key)
                        coef_rows.append(
                            {
                                key: row[key]
                                for key in [
                                    "config",
                                    "experiment",
                                    "mode",
                                    "start",
                                    "end",
                                    "block_label",
                                    "train_scale",
                                    "test_curve",
                                    "test_label",
                                    "lambda",
                                    "coef",
                                    "basis_dim",
                                    "n_cal",
                                    "ridge",
                                    "source_dot",
                                    "source_full_norm",
                                    "source_perp_norm",
                                    "source_retention",
                                    "denominator",
                                    "too_few_points",
                                    "mask_step_min",
                                    "mask_step_max",
                                ]
                            }
                        )
    return detail_rows, coef_rows


def aggregate(rows: list[dict[str, object]], cfg: dict[str, object], group: str, split: str) -> dict[str, object]:
    deltas = np.array([float(row["delta_pct"]) for row in rows], dtype=np.float64)
    return {
        "config": cfg["config"],
        "experiment": cfg["experiment"],
        "mode": cfg["mode"],
        "start": cfg["start"],
        "end": cfg["end"],
        "block_label": cfg["block_label"],
        "reference_config": cfg["reference_config"],
        "group": group,
        "split": split,
        "rows": len(rows),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "best_delta": float(np.min(deltas)),
        "worst_delta": float(np.max(deltas)),
        "wins": int(np.sum(deltas < 0.0)),
        "nonharm": int(np.sum(deltas <= 1e-12)),
        "mean_coef": float(np.mean([float(row["coef"]) for row in rows])),
        "mean_n_cal": float(np.mean([float(row["n_cal"]) for row in rows])),
        "mean_source_retention": float(np.mean([float(row["source_retention"]) for row in rows])),
    }


def summarize(detail_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    configs = config_rows()
    out: list[dict[str, object]] = []
    for cfg in configs:
        rows_c = [row for row in detail_rows if row["config"] == cfg["config"]]
        for group in ["core_wsd", "extra_control"]:
            rows_g = [row for row in rows_c if row["group"] == group]
            splits = {
                "all": rows_g,
                "same_scale": [row for row in rows_g if row["train_scale"] == row["test_scale"]],
                "cross_scale": [row for row in rows_g if row["train_scale"] != row["test_scale"]],
            }
            for split, rows in splits.items():
                if rows:
                    out.append(aggregate(rows, cfg, group, split))
    add_reference_deltas(out)
    return out


def add_reference_deltas(summary: list[dict[str, object]]) -> None:
    refs: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in summary:
        if row["config"] == f"suffix_ge_{CURRENT_SUFFIX_START}":
            refs[(str(row["group"]), str(row["split"]), str(row["experiment"]))] = row
            refs[(str(row["group"]), str(row["split"]), "leave_one_block_out_suffix8k")] = row
            refs[(str(row["group"]), str(row["split"]), "only_window")] = row

    for row in summary:
        key = (str(row["group"]), str(row["split"]), str(row["experiment"]))
        ref = refs.get(key)
        if ref is None:
            row["mean_delta_vs_suffix8k"] = ""
            row["worst_delta_vs_suffix8k"] = ""
            row["wins_vs_suffix8k"] = ""
            continue
        row["mean_delta_vs_suffix8k"] = float(row["mean_delta"]) - float(ref["mean_delta"])
        row["worst_delta_vs_suffix8k"] = float(row["worst_delta"]) - float(ref["worst_delta"])
        row["wins_vs_suffix8k"] = int(row["wins"]) - int(ref["wins"])


def find(summary: list[dict[str, object]], config: str, group: str, split: str) -> dict[str, object]:
    for row in summary:
        if row["config"] == config and row["group"] == group and row["split"] == split:
            return row
    raise KeyError((config, group, split))


def fmt(row: dict[str, object]) -> str:
    return (
        f"{float(row['mean_delta']):+.2f}% / {float(row['worst_delta']):+.2f}% / "
        f"{int(row['wins'])}/{int(row['rows'])}"
    )


def plot_prefix(summary: list[dict[str, object]]) -> None:
    rows = [
        row
        for row in summary
        if row["experiment"] == "prefix_start_sweep" and row["group"] == "core_wsd" and row["split"] == "all"
    ]
    rows = sorted(rows, key=lambda row: int(row["start"]))
    x = np.array([int(row["start"]) / 1000.0 for row in rows], dtype=np.float64)
    mean = np.array([float(row["mean_delta"]) for row in rows], dtype=np.float64)
    worst = np.array([float(row["worst_delta"]) for row in rows], dtype=np.float64)
    wins = np.array([int(row["wins"]) for row in rows], dtype=np.float64)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.6, 4.4), constrained_layout=True)
    ax.axhline(0.0, color="#111827", lw=0.8)
    ax.plot(x, mean, marker="o", color="#2563eb", lw=1.8, label="mean delta")
    ax.plot(x, worst, marker="o", color="#dc2626", lw=1.8, label="worst delta")
    ax.set_xlabel("source fit_start (k steps)")
    ax.set_ylabel("WSD delta vs MPL (%)")
    ax.set_title("Prefix drop sweep: cosine calibration suffix")
    ax.legend(loc="best")
    ax2 = ax.twinx()
    ax2.plot(x, wins, marker="s", color="#059669", lw=1.2, alpha=0.75, label="wins")
    ax2.set_ylim(0, 46)
    ax2.set_ylabel("wins out of 45")
    fig.savefig(FIG_DIR / "prefix_fit_start_sweep.png", dpi=180)
    plt.close(fig)

    zoom_rows = [row for row in rows if int(row["start"]) >= 6500]
    xz = np.array([int(row["start"]) / 1000.0 for row in zoom_rows], dtype=np.float64)
    mean_z = np.array([float(row["mean_delta"]) for row in zoom_rows], dtype=np.float64)
    wins_z = np.array([int(row["wins"]) for row in zoom_rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9.6, 4.2), constrained_layout=True)
    ax.axhline(0.0, color="#111827", lw=0.8)
    ax.plot(xz, mean_z, marker="o", color="#2563eb", lw=1.9, label="mean delta")
    ax.scatter([CURRENT_SUFFIX_START / 1000.0], [find(summary, f"suffix_ge_{CURRENT_SUFFIX_START}", "core_wsd", "all")["mean_delta"]], color="#dc2626", s=55, zorder=3, label="current 8k")
    ax.set_xlabel("source fit_start (k steps)")
    ax.set_ylabel("WSD mean delta vs MPL (%)")
    ax.set_title("Prefix drop sweep zoom: mean WSD transfer")
    ax.set_ylim(-34.0, 20.0)
    ax.legend(loc="best")
    ax2 = ax.twinx()
    ax2.plot(xz, wins_z, marker="s", color="#059669", lw=1.2, alpha=0.7)
    ax2.set_ylim(0, 46)
    ax2.set_ylabel("wins out of 45")
    fig.savefig(FIG_DIR / "prefix_fit_start_zoom.png", dpi=180)
    plt.close(fig)


def plot_leaveout(summary: list[dict[str, object]]) -> None:
    rows = [
        row
        for row in summary
        if row["experiment"] == "leave_one_block_out_suffix8k" and row["group"] == "core_wsd" and row["split"] == "all"
    ]
    rows = sorted(rows, key=lambda row: int(row["start"]))
    labels = [str(row["block_label"]) for row in rows]
    change = np.array([float(row["mean_delta_vs_suffix8k"]) for row in rows], dtype=np.float64)
    mean = np.array([float(row["mean_delta"]) for row in rows], dtype=np.float64)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.4, 4.4), constrained_layout=True)
    colors = ["#dc2626" if value > 0 else "#2563eb" for value in change]
    ax.axhline(0.0, color="#111827", lw=0.8)
    ax.bar(labels, change, color=colors, alpha=0.88)
    ax.set_xlabel("dropped source block")
    ax.set_ylabel("mean delta change vs suffix >= 8k (pp)")
    ax.set_title("Leave-one-block-out on current suffix calibration")
    ax.tick_params(axis="x", labelrotation=25)
    for idx, value in enumerate(mean):
        ax.text(idx, change[idx], f"{value:+.1f}", ha="center", va="bottom" if change[idx] >= 0 else "top", fontsize=8)
    fig.savefig(FIG_DIR / "leave_one_block_out_suffix8k.png", dpi=180)
    plt.close(fig)


def plot_only_window(summary: list[dict[str, object]]) -> None:
    rows = [
        row for row in summary if row["experiment"] == "only_window" and row["group"] == "core_wsd" and row["split"] == "all"
    ]
    rows = sorted(rows, key=lambda row: int(row["start"]))
    labels = [str(row["block_label"]) for row in rows]
    mean = np.array([float(row["mean_delta"]) for row in rows], dtype=np.float64)
    worst = np.array([float(row["worst_delta"]) for row in rows], dtype=np.float64)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.4, 4.4), constrained_layout=True)
    ax.axhline(0.0, color="#111827", lw=0.8)
    x = np.arange(len(rows))
    width = 0.38
    ax.bar(x - width / 2, mean, width, color="#2563eb", label="mean")
    ax.bar(x + width / 2, worst, width, color="#dc2626", label="worst")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("WSD delta vs MPL (%)")
    ax.set_title("Only-window calibration")
    ax.legend(loc="best")
    fig.savefig(FIG_DIR / "only_window_calibration.png", dpi=180)
    plt.close(fig)


def report_table_prefix(summary: list[dict[str, object]]) -> list[str]:
    lines = [
        "| fit_start | n_cal | same-scale | cross-scale | all WSD |\n",
        "| ---: | ---: | ---: | ---: | ---: |\n",
    ]
    for start in PREFIX_STARTS:
        cfg = f"suffix_ge_{start}"
        same = find(summary, cfg, "core_wsd", "same_scale")
        cross = find(summary, cfg, "core_wsd", "cross_scale")
        all_row = find(summary, cfg, "core_wsd", "all")
        lines.append(
            f"| {start} | {float(all_row['mean_n_cal']):.0f} | {fmt(same)} | {fmt(cross)} | {fmt(all_row)} |\n"
        )
    return lines


def report_table_leaveout(summary: list[dict[str, object]]) -> list[str]:
    rows = [
        row
        for row in summary
        if row["experiment"] == "leave_one_block_out_suffix8k" and row["group"] == "core_wsd" and row["split"] == "all"
    ]
    rows = sorted(rows, key=lambda row: int(row["start"]))
    lines = [
        "| dropped block | n_cal | all WSD | mean change vs suffix>=8k | interpretation |\n",
        "| --- | ---: | ---: | ---: | --- |\n",
    ]
    for row in rows:
        change = float(row["mean_delta_vs_suffix8k"])
        if change > 0.75:
            interp = "dropping hurts; useful block"
        elif change < -0.75:
            interp = "dropping helps; noisy/harmful block"
        else:
            interp = "low sensitivity"
        lines.append(
            f"| {row['block_label']} | {float(row['mean_n_cal']):.0f} | {fmt(row)} | {change:+.2f} pp | {interp} |\n"
        )
    return lines


def report_table_only(summary: list[dict[str, object]]) -> list[str]:
    rows = [
        row for row in summary if row["experiment"] == "only_window" and row["group"] == "core_wsd" and row["split"] == "all"
    ]
    rows = sorted(rows, key=lambda row: int(row["start"]))
    lines = [
        "| only window | n_cal | all WSD | source retention |\n",
        "| --- | ---: | ---: | ---: |\n",
    ]
    for row in rows:
        lines.append(
            f"| {row['block_label']} | {float(row['mean_n_cal']):.0f} | {fmt(row)} | "
            f"{float(row['mean_source_retention']):.4g} |\n"
        )
    return lines


def write_report(summary: list[dict[str, object]]) -> None:
    ref = find(summary, f"suffix_ge_{CURRENT_SUFFIX_START}", "core_wsd", "all")
    drop5 = find(summary, "suffix_ge_5000", "core_wsd", "all")
    prefix_rows = [
        row
        for row in summary
        if row["experiment"] == "prefix_start_sweep" and row["group"] == "core_wsd" and row["split"] == "all"
    ]
    best_mean = min(prefix_rows, key=lambda row: float(row["mean_delta"]))
    best_worst = min(prefix_rows, key=lambda row: float(row["worst_delta"]))

    leave_rows = [
        row
        for row in summary
        if row["experiment"] == "leave_one_block_out_suffix8k" and row["group"] == "core_wsd" and row["split"] == "all"
    ]
    most_useful = max(leave_rows, key=lambda row: float(row["mean_delta_vs_suffix8k"]))
    least_sensitive = min(leave_rows, key=lambda row: float(row["mean_delta_vs_suffix8k"]))
    least_change = float(least_sensitive["mean_delta_vs_suffix8k"])
    if least_change < -0.75:
        block_sensitivity_line = (
            f"- drop 后最改善的是 `{least_sensitive['block_label']}`，"
            f"mean change {least_change:+.2f} pp，说明这段可能 noisy。\n\n"
        )
    else:
        block_sensitivity_line = (
            f"- 没有发现丢掉后能改善的 current-suffix block；最不敏感的是 `{least_sensitive['block_label']}`，"
            f"但 mean change 仍为 {least_change:+.2f} pp。\n\n"
        )

    lines: list[str] = [
        "# Source Calibration Data Drop Ablation\n\n",
        "目标：检查从 cosine residual 拟合 \\(\\kappa\\) 时，丢弃 source 训练点是否能改善 cosine-to-WSD 迁移，",
        "并定位哪些 source 时间段最关键。目标 WSD/controls 的 loss 只用于评价，不参与拟合。\n\n",
        "固定公式：\n\n",
        "\\[\n",
        "\\widehat L_s(t)=L_{\\mathrm{MPL},s}(t)+a_s\\widehat\\kappa_s\\phi_{\\lambda_s,s}(t),\n",
        "\\quad q_s=\\sum_t(d_t/D)^2,\\quad \\lambda_s=\\lambda_{obs}/(2-q_s).\n",
        "\\]\n\n",
        "固定投影：\n\n",
        "\\[\n",
        "\\widehat\\kappa_s=\\frac{[((I-P_{LD})\\phi)^\\top ((I-P_{LD})r)]_+}{\\|(I-P_{LD})\\phi\\|_2^2+1/N_{cal}}.\n",
        "\\]\n\n",
        "唯一变化是 source calibration mask，即哪些 cosine 点参与上式中的内积、范数和 MPL-LD tangent projection。\n\n",
        "## Main Reading\n\n",
        f"- `fit_start=5000`: {fmt(drop5)}。\n",
        f"- 当前 `fit_start=8000`: {fmt(ref)}。\n",
        f"- prefix sweep 按 mean 最好的是 `{best_mean['config']}`: {fmt(best_mean)}。\n",
        f"- prefix sweep 按 worst 最好的是 `{best_worst['config']}`: {fmt(best_worst)}。\n",
        f"- 在当前 suffix>=8k 里，drop 后最伤性能的是 `{most_useful['block_label']}`，",
        f"mean change {float(most_useful['mean_delta_vs_suffix8k']):+.2f} pp。\n",
        block_sensitivity_line,
        "## Prefix Start Sweep\n\n",
        "表中每格为 `mean delta / worst delta / wins`。delta 越负越好。\n\n",
        *report_table_prefix(summary),
        "\n## Leave-One-Block-Out From Current Suffix >= 8k\n\n",
        "这里以当前 `suffix_ge_8000` 为 reference。`mean change` 为丢掉该 block 后 all-WSD mean delta 的变化：",
        "正数表示丢掉会变差，所以该 block 有用；负数表示丢掉反而变好，所以该 block 可能更 noisy。\n\n",
        *report_table_leaveout(summary),
        "\n## Only-Window Calibration\n\n",
        "只用单个 source window 拟合 \\(\\kappa\\)。这个实验检查每段数据单独是否足以支撑迁移。\n\n",
        *report_table_only(summary),
        "\n## Figures\n\n",
        "- `figs/prefix_fit_start_sweep.png`\n",
        "- `figs/prefix_fit_start_zoom.png`\n",
        "- `figs/leave_one_block_out_suffix8k.png`\n",
        "- `figs/only_window_calibration.png`\n",
    ]
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    details, coefs = evaluate()
    summary = summarize(details)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "details.csv", details)
    write_csv(OUT_DIR / "coefficients.csv", coefs)
    write_csv(OUT_DIR / "summary.csv", summary)
    plot_prefix(summary)
    plot_leaveout(summary)
    plot_only_window(summary)
    write_report(summary)
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
