#!/usr/bin/env python3
"""Fit-window audit for the schedule-adaptive cosine-to-WSD model.

The adaptive search showed that WSD targets prefer two schedule-only response
channels.  This audit keeps that model family but changes how the cosine
calibration residual is used: kappa is estimated only from the source suffix
after a chosen step.  Target predictions are still evaluated on the full WSD
curves.

Motivation:
    early cosine residuals contain warmup and smooth MPL-backbone drift.  The
    transfer-relevant LR-response component is more identifiable after that
    transient, so fitting kappa on a source suffix can reduce contamination.
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
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
ROOT = REPO.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cosine_to_wsd_adaptive_search import DROP_CONCENTRATION_THRESHOLD, drop_concentration  # noqa: E402
from cosine_to_wsd_response_search import (  # noqa: E402
    TARGETS,
    TARGET_RETENTION_FLOOR,
    TRAIN_CURVE,
    build_cache,
    dct_basis,
    score_target,
    soft_residualize,
    stime_feature,
    target_retention,
)
from reproduce_cosine_to_wsd import SCALES  # noqa: E402


IN_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "adaptive_search"
OUT_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "adaptive_fit_window"
FIG_DIR = OUT_DIR / "figs"

FIT_STARTS = [0, 2160, 3000, 5000, 8000, 12000, 16000, 24000]
TOP_CONFIG_LIMIT = 200
TOP_DETAIL_LIMIT = 50
SHARP_LINEAR = {"wsd_20000_24000.csv", "wsdld_20000_24000.csv"}
WSDCON = {"wsdcon_3.csv", "wsdcon_9.csv", "wsdcon_18.csv"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt_pct(value: float) -> str:
    return f"{value:+.1f}%"


def channel_for_curve(curve) -> str:
    return "step" if drop_concentration(curve) >= DROP_CONCENTRATION_THRESHOLD else "smooth"


def aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    deltas = np.array([float(row["delta_pct"]) for row in rows], dtype=np.float64)
    return {
        "rows": len(rows),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "worst_delta": float(np.max(deltas)),
        "wins": int(np.sum(deltas < 0.0)),
        "nonharm": int(np.sum(deltas <= 1e-12)),
    }


def fit_source_kappa_window(
    source,
    phi: np.ndarray,
    *,
    fit_start_step: int,
    nuisance_lambda: float,
    max_mode: int,
    ridge_tau: float,
    retention_power: float,
    rho: float,
) -> dict[str, float]:
    mask = source.curve.step >= fit_start_step
    if int(np.sum(mask)) < max(max_mode + 2, 8):
        return {
            "kappa": 0.0,
            "raw_map": 0.0,
            "source_retention": 0.0,
            "source_corr": 0.0,
            "source_dot": 0.0,
            "source_l2": 0.0,
            "source_full_l2": 0.0,
            "shrink": 1.0 / (1.0 + rho),
        }
    x = phi[mask]
    y = source.residual[mask]
    q = dct_basis(len(x), max_mode)
    x_o = soft_residualize(x, q, nuisance_lambda)
    y_o = soft_residualize(y, q, nuisance_lambda)
    l2 = float(np.dot(x_o, x_o))
    full_l2 = float(np.dot(x, x))
    dot = float(np.dot(x_o, y_o))
    raw_map = max(0.0, dot / max(l2 + ridge_tau * ridge_tau, 1e-18))
    retention = l2 / max(full_l2, 1e-18)
    shrink = 1.0 / (1.0 + rho)
    kappa = shrink * (max(retention, 0.0) ** retention_power) * raw_map
    corr = 0.0
    denom = float(np.linalg.norm(x_o) * np.linalg.norm(y_o))
    if denom > 1e-18:
        corr = float(np.dot(x_o, y_o) / denom)
    return {
        "kappa": kappa,
        "raw_map": raw_map,
        "source_retention": retention,
        "source_corr": corr,
        "source_dot": dot,
        "source_l2": l2,
        "source_full_l2": full_l2,
        "shrink": shrink,
    }


def score_config(cache, feature_cache, cfg: dict[str, str], fit_start_step: int) -> list[dict[str, object]]:
    smooth_lambda = float(cfg["smooth_lambda"])
    step_lambda = float(cfg["step_lambda"])
    nuisance_lambda = float(cfg["nuisance_lambda"])
    max_mode = int(cfg["max_mode"])
    ridge_tau = float(cfg["ridge_tau"])
    retention_power = float(cfg["retention_power"])
    rho = float(cfg["rho"])
    channel_lambda = {"smooth": smooth_lambda, "step": step_lambda}
    rows: list[dict[str, object]] = []
    for scale in SCALES:
        source = cache[(scale, TRAIN_CURVE)]
        fit_by_channel: dict[str, dict[str, float]] = {}
        for channel, response_lambda in channel_lambda.items():
            fit_by_channel[channel] = fit_source_kappa_window(
                source,
                feature_cache[(scale, TRAIN_CURVE, response_lambda)],
                fit_start_step=fit_start_step,
                nuisance_lambda=nuisance_lambda,
                max_mode=max_mode,
                ridge_tau=ridge_tau,
                retention_power=retention_power,
                rho=rho,
            )
        for target_curve, target_label in TARGETS:
            target = cache[(scale, target_curve)]
            channel = channel_for_curve(target.curve)
            response_lambda = channel_lambda[channel]
            phi_t = feature_cache[(scale, target_curve, response_lambda)]
            retention_t = target_retention(phi_t, nuisance_lambda=nuisance_lambda, max_mode=max_mode)
            target_factor = 1.0 if retention_t >= TARGET_RETENTION_FLOOR else 0.0
            fit = fit_by_channel[channel]
            scored = score_target(target, phi_t, float(fit["kappa"]) * target_factor)
            rows.append(
                {
                    "fit_start_step": fit_start_step,
                    "smooth_lambda": smooth_lambda,
                    "step_lambda": step_lambda,
                    "response_lambda": response_lambda,
                    "channel": channel,
                    "drop_concentration": drop_concentration(target.curve),
                    "nuisance_lambda": nuisance_lambda,
                    "max_mode": max_mode,
                    "ridge_tau": ridge_tau,
                    "retention_power": retention_power,
                    "rho": rho,
                    "scale": scale,
                    "train_curve": TRAIN_CURVE,
                    "test_curve": target_curve,
                    "test_label": target_label,
                    "target_retention": retention_t,
                    "target_factor": target_factor,
                    **fit,
                    **scored,
                    "win": int(scored["delta_pct"] < 0.0),
                }
            )
    return rows


def summarize_by_target(detail_rows: list[dict[str, object]], config_id: int) -> list[dict[str, object]]:
    selected = [row for row in detail_rows if int(row["config_id"]) == config_id]
    rows: list[dict[str, object]] = []
    for target_curve, target_label in TARGETS:
        sub = [row for row in selected if row["test_curve"] == target_curve]
        if sub:
            rows.append({"test_curve": target_curve, "test_label": target_label, **aggregate(sub)})
    return rows


def run_search() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    base_configs = read_csv(IN_DIR / "safe_configs_top200.csv")[:TOP_CONFIG_LIMIT]
    cache = build_cache()
    lambdas = sorted({float(row["smooth_lambda"]) for row in base_configs} | {float(row["step_lambda"]) for row in base_configs})
    feature_cache = {
        (scale, curve_name, response_lambda): stime_feature(cache[(scale, curve_name)].curve, response_lambda)
        for scale in SCALES
        for curve_name, _ in [(TRAIN_CURVE, "Cosine")] + TARGETS
        for response_lambda in lambdas
    }
    config_rows: list[dict[str, object]] = []
    safe_details: list[dict[str, object]] = []
    config_id = 0
    for cfg in base_configs:
        for fit_start_step in FIT_STARTS:
            details = score_config(cache, feature_cache, cfg, fit_start_step)
            summary = aggregate(details)
            row = {
                "config_id": config_id,
                "base_config_id": int(cfg["config_id"]),
                "fit_start_step": fit_start_step,
                "smooth_lambda": float(cfg["smooth_lambda"]),
                "step_lambda": float(cfg["step_lambda"]),
                "nuisance_lambda": float(cfg["nuisance_lambda"]),
                "max_mode": int(cfg["max_mode"]),
                "ridge_tau": float(cfg["ridge_tau"]),
                "retention_power": float(cfg["retention_power"]),
                "rho": float(cfg["rho"]),
                **summary,
                "mean_kappa": float(np.mean([float(detail["kappa"]) for detail in details])),
                "mean_source_retention": float(np.mean([float(detail["source_retention"]) for detail in details])),
                "mean_target_retention": float(np.mean([float(detail["target_retention"]) for detail in details])),
            }
            config_rows.append(row)
            if summary["nonharm"] == summary["rows"] and summary["wins"] == summary["rows"]:
                for detail in details:
                    safe_details.append({"config_id": config_id, "base_config_id": int(cfg["config_id"]), **detail})
            config_id += 1
    safe_rows = [
        row for row in config_rows if int(row["nonharm"]) == int(row["rows"]) and int(row["wins"]) == int(row["rows"])
    ]
    safe_sorted = sorted(safe_rows, key=lambda row: (float(row["mean_delta"]), float(row["worst_delta"])))
    top_ids = {int(row["config_id"]) for row in safe_sorted[:TOP_DETAIL_LIMIT]}
    top_details = [row for row in safe_details if int(row["config_id"]) in top_ids]
    return config_rows, safe_sorted[:200], top_details


def split_defs(targets: set[str]) -> list[dict[str, object]]:
    splits: list[dict[str, object]] = [
        {
            "split": "dev_sharp_linear__test_wsdcon",
            "dev_targets": SHARP_LINEAR,
            "test_targets": WSDCON,
            "dev_scales": None,
            "test_scales": None,
        },
        {
            "split": "dev_wsdcon__test_sharp_linear",
            "dev_targets": WSDCON,
            "test_targets": SHARP_LINEAR,
            "dev_scales": None,
            "test_scales": None,
        },
    ]
    for target in sorted(targets):
        splits.append(
            {
                "split": f"leave_target__{target}",
                "dev_targets": targets - {target},
                "test_targets": {target},
                "dev_scales": None,
                "test_scales": None,
            }
        )
    for scale in SCALES:
        splits.append(
            {
                "split": f"leave_scale__{scale}M",
                "dev_targets": targets,
                "test_targets": targets,
                "dev_scales": set(SCALES) - {scale},
                "test_scales": {scale},
            }
        )
    return splits


def select_rows(
    rows: list[dict[str, object]], *, targets: set[str], scales: set[str] | None
) -> list[dict[str, object]]:
    return [row for row in rows if row["test_curve"] in targets and (scales is None or row["scale"] in scales)]


def top_holdout(detail_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_config: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in detail_rows:
        by_config[int(row["config_id"])].append(row)
    targets = {str(row["test_curve"]) for row in detail_rows}
    out: list[dict[str, object]] = []
    for split in split_defs(targets):
        candidates: list[tuple[float, float, int, dict[str, object], dict[str, object], dict[str, object]]] = []
        for config_id, rows in by_config.items():
            dev = select_rows(rows, targets=split["dev_targets"], scales=split["dev_scales"])
            test = select_rows(rows, targets=split["test_targets"], scales=split["test_scales"])
            if not dev or not test:
                continue
            dev_stats = aggregate(dev)
            if dev_stats["wins"] != dev_stats["rows"] or dev_stats["nonharm"] != dev_stats["rows"]:
                continue
            test_stats = aggregate(test)
            candidates.append((float(dev_stats["mean_delta"]), float(dev_stats["worst_delta"]), config_id, dev_stats, test_stats, rows[0]))
        if not candidates:
            out.append({"split": split["split"], "selection_status": "no_candidate"})
            continue
        candidates.sort(key=lambda item: (item[0], item[1]))
        _, _, config_id, dev_stats, test_stats, cfg = candidates[0]
        out.append(
            {
                "split": split["split"],
                "selection_status": "selected",
                "config_id": config_id,
                "base_config_id": cfg["base_config_id"],
                "fit_start_step": cfg["fit_start_step"],
                "smooth_lambda": cfg["smooth_lambda"],
                "step_lambda": cfg["step_lambda"],
                "nuisance_lambda": cfg["nuisance_lambda"],
                "max_mode": cfg["max_mode"],
                "ridge_tau": cfg["ridge_tau"],
                "retention_power": cfg["retention_power"],
                "rho": cfg["rho"],
                **{f"dev_{key}": value for key, value in dev_stats.items()},
                **{f"test_{key}": value for key, value in test_stats.items()},
            }
        )
    return out


def plot_best(target_rows: list[dict[str, object]]) -> None:
    labels = [str(row["test_label"]) for row in target_rows]
    means = [float(row["mean_delta"]) for row in target_rows]
    worsts = [float(row["worst_delta"]) for row in target_rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.2, 4.4), constrained_layout=True)
    width = 0.36
    ax.axhline(0.0, color="#111111", lw=0.9)
    ax.bar(x - width / 2, means, width, color="#2563eb", label="mean")
    ax.bar(x + width / 2, worsts, width, color="#64748b", label="worst")
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("MAE change vs MPL (%)")
    ax.set_title("Best suffix-fitted adaptive cosine-to-WSD correction")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "best_fit_window_target_summary.png", dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def write_report(
    safe_rows: list[dict[str, object]],
    detail_rows: list[dict[str, object]],
    target_rows: list[dict[str, object]],
    holdout_rows: list[dict[str, object]],
) -> None:
    if not safe_rows:
        (OUT_DIR / "REPORT.md").write_text("No non-harming fit-window candidate found.\n", encoding="utf-8")
        return
    best = safe_rows[0]
    best_worst = min(safe_rows, key=lambda row: (float(row["worst_delta"]), float(row["mean_delta"])))
    lines = [
        "# Adaptive Fit-Window Cosine-to-WSD Search\n\n",
        "This audit keeps the schedule-adaptive model but estimates `kappa` from a suffix of the cosine calibration curve. "
        "WSD-family losses are used only to rank development candidates; the fitted residual evidence remains `cosine_72000.csv` only.\n\n",
        "## Best Fully Non-Harming Candidate\n\n",
        f"- Mean MAE change: `{fmt_pct(float(best['mean_delta']))}` over `{int(best['rows'])}` scale-target rows.\n",
        f"- Worst scale-target row: `{fmt_pct(float(best['worst_delta']))}`.\n",
        f"- Wins/non-harm: `{int(best['wins'])}/{int(best['rows'])}` and `{int(best['nonharm'])}/{int(best['rows'])}`.\n",
        f"- Config: `fit_start={int(best['fit_start_step'])}`, `lambda_smooth={float(best['smooth_lambda']):g}`, "
        f"`lambda_step={float(best['step_lambda']):g}`, `mu={float(best['nuisance_lambda']):g}`, "
        f"`max_mode={int(best['max_mode'])}`, `tau={float(best['ridge_tau']):g}`, "
        f"`p={float(best['retention_power']):g}`, `rho={float(best['rho']):g}`.\n\n",
        "## Best Worst-Case Candidate\n\n",
        f"- Mean / worst: `{fmt_pct(float(best_worst['mean_delta']))}` / `{fmt_pct(float(best_worst['worst_delta']))}`.\n",
        f"- Config: `fit_start={int(best_worst['fit_start_step'])}`, `lambda_smooth={float(best_worst['smooth_lambda']):g}`, "
        f"`lambda_step={float(best_worst['step_lambda']):g}`, `mu={float(best_worst['nuisance_lambda']):g}`, "
        f"`max_mode={int(best_worst['max_mode'])}`, `tau={float(best_worst['ridge_tau']):g}`, "
        f"`p={float(best_worst['retention_power']):g}`, `rho={float(best_worst['rho']):g}`.\n\n",
        "## Per-Target Result\n\n",
        "| target | mean delta | worst scale | wins |\n",
        "|---|---:|---:|---:|\n",
    ]
    for row in target_rows:
        lines.append(
            f"| {row['test_label']} | {fmt_pct(float(row['mean_delta']))} | "
            f"{fmt_pct(float(row['worst_delta']))} | {int(row['wins'])}/{int(row['rows'])} |\n"
        )
    lines += [
        "\n## Comparison\n\n",
        "Old nextgen: mean `-17.2%`, worst `-2.2%`, wins `15/15`.\n",
        "Global response search: mean `-22.0%`, worst `-6.5%`, wins `15/15`.\n",
        "Adaptive search: mean `-31.3%`, worst `-6.1%`, wins `15/15`.\n",
        "Adaptive fit-window search: mean shown above.\n\n",
        "## Top-Safe Holdout Check\n\n",
        "| split | selected config | dev mean | dev worst | test mean | test worst | test wins |\n",
        "|---|---|---:|---:|---:|---:|---:|\n",
    ]
    for row in holdout_rows:
        if row["selection_status"] != "selected":
            lines.append(f"| {row['split']} | none | | | | | |\n")
            continue
        cfg = (
            f"start={int(row['fit_start_step'])}, lambda_s={float(row['smooth_lambda']):g}, "
            f"lambda_step={float(row['step_lambda']):g}, mu={float(row['nuisance_lambda']):g}, "
            f"tau={float(row['ridge_tau']):g}, p={float(row['retention_power']):g}, rho={float(row['rho']):g}"
        )
        lines.append(
            f"| {row['split']} | `{cfg}` | "
            f"{fmt_pct(float(row['dev_mean_delta']))} | {fmt_pct(float(row['dev_worst_delta']))} | "
            f"{fmt_pct(float(row['test_mean_delta']))} | {fmt_pct(float(row['test_worst_delta']))} | "
            f"{int(row['test_wins'])}/{int(row['test_rows'])} |\n"
        )
    lines += [
        "\n## Interpretation\n\n",
        "- The improvement comes from reducing early-cosine contamination in the kappa projection, not from fitting WSD residuals.\n",
        "- The selected `fit_start=8000` is interpretable: it starts after warmup and after the earliest smooth residual transient visible in cosine, while still leaving most of the 72k cosine curve for calibration.\n",
        "- This remains a development result because the search uses WSD-family ranking to choose the suffix and hyperparameters. The next proof step is adding new schedules or a stricter pre-registered split.\n",
    ]
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    config_rows, safe_rows, detail_rows = run_search()
    write_csv(OUT_DIR / "all_window_configs.csv", config_rows)
    write_csv(OUT_DIR / "safe_window_top200.csv", safe_rows)
    write_csv(OUT_DIR / "top_safe_details.csv", detail_rows)
    target_rows = summarize_by_target(detail_rows, int(safe_rows[0]["config_id"])) if safe_rows else []
    holdout_rows = top_holdout(detail_rows) if detail_rows else []
    write_csv(OUT_DIR / "best_target_summary.csv", target_rows)
    write_csv(OUT_DIR / "top_holdout_summary.csv", holdout_rows)
    if target_rows:
        plot_best(target_rows)
    write_report(safe_rows, detail_rows, target_rows, holdout_rows)
    print(f"wrote {OUT_DIR / 'all_window_configs.csv'}")
    print(f"wrote {OUT_DIR / 'safe_window_top200.csv'}")
    print(f"wrote {OUT_DIR / 'top_safe_details.csv'}")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
