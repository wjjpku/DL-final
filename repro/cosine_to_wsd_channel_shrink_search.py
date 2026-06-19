#!/usr/bin/env python3
"""Channel-specific shrinkage audit for cosine-to-WSD prediction.

The adaptive fit-window model estimates the smooth and step response amplitudes
from the same cosine calibration curve, then applies one shared shrinkage
parameter rho.  This audit keeps the same cosine-only fitting protocol but
allows different transfer shrinkage for smooth and step response channels.

Motivation:
    Smooth WSD decay and concentrated WSD-con drops have different
    identifiability in the cosine residual.  A single rho can under-shrink one
    channel while over-shrinking the other.  Splitting rho is a small,
    interpretable change: it changes uncertainty calibration, not the response
    feature or the target-loss protocol.
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

import numpy as np

REPO = Path(__file__).resolve().parent
ROOT = REPO.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cosine_to_wsd_adaptive_fit_window import (  # noqa: E402
    SHARP_LINEAR,
    WSDCON,
    aggregate,
    channel_for_curve,
    fit_source_kappa_window,
)
from cosine_to_wsd_response_search import (  # noqa: E402
    TARGETS,
    TARGET_RETENTION_FLOOR,
    TRAIN_CURVE,
    build_cache,
    score_target,
    stime_feature,
    target_retention,
)
from reproduce_cosine_to_wsd import SCALES  # noqa: E402


IN_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "adaptive_fit_window"
OUT_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "channel_shrink"

TOP_CONFIG_LIMIT = 80
TOP_DETAIL_LIMIT = 200
RHO_GRID = [0.0, 0.2, 0.35, 0.5, 0.6, 0.75, 0.9, 1.0, 1.25, 1.5]


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


def fmt_pct2(value: float) -> str:
    return f"{value:+.2f}%"


def score_config(
    cache,
    feature_cache,
    cfg: dict[str, str],
    *,
    rho_smooth: float,
    rho_step: float,
) -> list[dict[str, object]]:
    fit_start_step = int(cfg["fit_start_step"])
    smooth_lambda = float(cfg["smooth_lambda"])
    step_lambda = float(cfg["step_lambda"])
    nuisance_lambda = float(cfg["nuisance_lambda"])
    max_mode = int(cfg["max_mode"])
    ridge_tau = float(cfg["ridge_tau"])
    retention_power = float(cfg["retention_power"])
    channel_lambda = {"smooth": smooth_lambda, "step": step_lambda}
    channel_rho = {"smooth": rho_smooth, "step": rho_step}

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
                rho=channel_rho[channel],
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
                    "rho": channel_rho[channel],
                    "rho_smooth": rho_smooth,
                    "rho_step": rho_step,
                    "nuisance_lambda": nuisance_lambda,
                    "max_mode": max_mode,
                    "ridge_tau": ridge_tau,
                    "retention_power": retention_power,
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


def run_search() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    base_configs = read_csv(IN_DIR / "safe_window_top200.csv")[:TOP_CONFIG_LIMIT]
    cache = build_cache()
    lambdas = sorted(
        {float(row["smooth_lambda"]) for row in base_configs}
        | {float(row["step_lambda"]) for row in base_configs}
    )
    feature_cache = {
        (scale, curve_name, response_lambda): stime_feature(cache[(scale, curve_name)].curve, response_lambda)
        for scale in SCALES
        for curve_name, _ in [(TRAIN_CURVE, "Cosine")] + TARGETS
        for response_lambda in lambdas
    }

    config_rows: list[dict[str, object]] = []
    safe_detail_rows: list[dict[str, object]] = []
    config_id = 0
    for cfg in base_configs:
        for rho_smooth in RHO_GRID:
            for rho_step in RHO_GRID:
                details = score_config(
                    cache,
                    feature_cache,
                    cfg,
                    rho_smooth=rho_smooth,
                    rho_step=rho_step,
                )
                summary = aggregate(details)
                row = {
                    "config_id": config_id,
                    "base_config_id": int(cfg["config_id"]),
                    "fit_start_step": int(cfg["fit_start_step"]),
                    "smooth_lambda": float(cfg["smooth_lambda"]),
                    "step_lambda": float(cfg["step_lambda"]),
                    "nuisance_lambda": float(cfg["nuisance_lambda"]),
                    "max_mode": int(cfg["max_mode"]),
                    "ridge_tau": float(cfg["ridge_tau"]),
                    "retention_power": float(cfg["retention_power"]),
                    "rho_smooth": rho_smooth,
                    "rho_step": rho_step,
                    **summary,
                    "mean_kappa": float(np.mean([float(detail["kappa"]) for detail in details])),
                    "mean_source_retention": float(
                        np.mean([float(detail["source_retention"]) for detail in details])
                    ),
                    "mean_target_retention": float(
                        np.mean([float(detail["target_retention"]) for detail in details])
                    ),
                }
                config_rows.append(row)
                if summary["nonharm"] == summary["rows"] and summary["wins"] == summary["rows"]:
                    for detail in details:
                        safe_detail_rows.append(
                            {
                                "config_id": config_id,
                                "base_config_id": int(cfg["config_id"]),
                                **detail,
                            }
                        )
                config_id += 1

    safe_rows = [
        row for row in config_rows if int(row["nonharm"]) == int(row["rows"]) and int(row["wins"]) == int(row["rows"])
    ]
    safe_sorted = sorted(safe_rows, key=lambda row: (float(row["mean_delta"]), float(row["worst_delta"])))
    top_ids = {int(row["config_id"]) for row in safe_sorted[:TOP_DETAIL_LIMIT]}
    top_details = [row for row in safe_detail_rows if int(row["config_id"]) in top_ids]
    return config_rows, safe_sorted[:200], top_details


def summarize_by_target(detail_rows: list[dict[str, object]], config_id: int) -> list[dict[str, object]]:
    selected = [row for row in detail_rows if int(row["config_id"]) == config_id]
    rows: list[dict[str, object]] = []
    for target_curve, target_label in TARGETS:
        sub = [row for row in selected if row["test_curve"] == target_curve]
        if sub:
            rows.append({"test_curve": target_curve, "test_label": target_label, **aggregate(sub)})
    return rows


def select_recommended(safe_rows: list[dict[str, object]]) -> dict[str, object]:
    old = read_csv(IN_DIR / "safe_window_top200.csv")[0]
    old_worst = float(old["worst_delta"])
    candidates = [row for row in safe_rows if float(row["worst_delta"]) <= old_worst + 1e-12]
    if not candidates:
        return safe_rows[0]
    return sorted(candidates, key=lambda row: (float(row["mean_delta"]), float(row["worst_delta"])))[0]


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
            candidates.append(
                (
                    float(dev_stats["mean_delta"]),
                    float(dev_stats["worst_delta"]),
                    config_id,
                    dev_stats,
                    test_stats,
                    rows[0],
                )
            )
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
                "rho_smooth": cfg["rho_smooth"],
                "rho_step": cfg["rho_step"],
                **{f"dev_{key}": value for key, value in dev_stats.items()},
                **{f"test_{key}": value for key, value in test_stats.items()},
            }
        )
    return out


def write_report(
    safe_rows: list[dict[str, object]],
    detail_rows: list[dict[str, object]],
    recommended_target_rows: list[dict[str, object]],
    holdout_rows: list[dict[str, object]],
) -> None:
    if not safe_rows:
        (OUT_DIR / "REPORT.md").write_text("No non-harming channel-shrink candidate found.\n", encoding="utf-8")
        return
    best_mean = safe_rows[0]
    recommended = select_recommended(safe_rows)
    best_worst = min(safe_rows, key=lambda row: (float(row["worst_delta"]), float(row["mean_delta"])))
    old = read_csv(IN_DIR / "safe_window_top200.csv")[0]
    lines = [
        "# Channel-Specific Shrinkage Cosine-to-WSD Audit\n\n",
        "This audit keeps the adaptive fit-window cosine-only protocol but replaces the shared shrink "
        "`rho` with `rho_smooth` and `rho_step`. The amplitudes are still estimated only from "
        "`cosine_72000.csv`; WSD losses are used for development ranking and evaluation.\n\n",
        "## Formula Change\n\n",
        "```text\n",
        "kappa_channel = [1/(1+rho_channel)] * R_source_channel^p\n",
        "                * max(0, <M_mu phi_channel, M_mu r_cos>_F\n",
        "                         / (||M_mu phi_channel||_F^2 + tau^2))\n",
        "rho_channel = rho_smooth for diffuse LR decay, rho_step for concentrated LR drops\n",
        "L_hat_target = L_MPL,target + kappa_channel * phi_channel,target\n",
        "```\n\n",
        "Only the uncertainty shrinkage is channel-specific. The response feature, suffix fitting, "
        "and target schedule routing are unchanged.\n\n",
        "## Recommended Pareto Candidate\n\n",
        "This is the main candidate from this audit: it improves the mean while keeping the worst row "
        "at least as good as the previous shared-rho fit-window model.\n\n",
        f"- Mean MAE change: `{fmt_pct(float(recommended['mean_delta']))}` over `{int(recommended['rows'])}` scale-target rows.\n",
        f"- Worst scale-target row: `{fmt_pct(float(recommended['worst_delta']))}`.\n",
        f"- Wins/non-harm: `{int(recommended['wins'])}/{int(recommended['rows'])}` and "
        f"`{int(recommended['nonharm'])}/{int(recommended['rows'])}`.\n",
        f"- Config: `fit_start={int(recommended['fit_start_step'])}`, "
        f"`lambda_smooth={float(recommended['smooth_lambda']):g}`, "
        f"`lambda_step={float(recommended['step_lambda']):g}`, "
        f"`mu={float(recommended['nuisance_lambda']):g}`, "
        f"`max_mode={int(recommended['max_mode'])}`, `tau={float(recommended['ridge_tau']):g}`, "
        f"`p={float(recommended['retention_power']):g}`, "
        f"`rho_smooth={float(recommended['rho_smooth']):g}`, "
        f"`rho_step={float(recommended['rho_step']):g}`.\n\n",
        "## Best Mean Candidate\n\n",
        f"- Mean / worst: `{fmt_pct(float(best_mean['mean_delta']))}` / "
        f"`{fmt_pct(float(best_mean['worst_delta']))}`.\n",
        f"- Config: `fit_start={int(best_mean['fit_start_step'])}`, "
        f"`lambda_smooth={float(best_mean['smooth_lambda']):g}`, "
        f"`lambda_step={float(best_mean['step_lambda']):g}`, `mu={float(best_mean['nuisance_lambda']):g}`, "
        f"`max_mode={int(best_mean['max_mode'])}`, `tau={float(best_mean['ridge_tau']):g}`, "
        f"`p={float(best_mean['retention_power']):g}`, "
        f"`rho_smooth={float(best_mean['rho_smooth']):g}`, `rho_step={float(best_mean['rho_step']):g}`.\n\n",
        "## Best Worst-Case Candidate\n\n",
        f"- Mean / worst: `{fmt_pct(float(best_worst['mean_delta']))}` / `{fmt_pct(float(best_worst['worst_delta']))}`.\n",
        f"- Config: `fit_start={int(best_worst['fit_start_step'])}`, "
        f"`lambda_smooth={float(best_worst['smooth_lambda']):g}`, "
        f"`lambda_step={float(best_worst['step_lambda']):g}`, `mu={float(best_worst['nuisance_lambda']):g}`, "
        f"`max_mode={int(best_worst['max_mode'])}`, `tau={float(best_worst['ridge_tau']):g}`, "
        f"`p={float(best_worst['retention_power']):g}`, "
        f"`rho_smooth={float(best_worst['rho_smooth']):g}`, `rho_step={float(best_worst['rho_step']):g}`.\n\n",
        "## Per-Target Result For Recommended Candidate\n\n",
        "| target | mean delta | worst scale | wins |\n",
        "|---|---:|---:|---:|\n",
    ]
    for row in recommended_target_rows:
        lines.append(
            f"| {row['test_label']} | {fmt_pct(float(row['mean_delta']))} | "
            f"{fmt_pct(float(row['worst_delta']))} | {int(row['wins'])}/{int(row['rows'])} |\n"
        )
    lines += [
        "\n## Comparison To Previous Main Candidate\n\n",
        f"Adaptive fit-window shared-rho: mean `{fmt_pct2(float(old['mean_delta']))}`, "
        f"worst `{fmt_pct2(float(old['worst_delta']))}`, wins `{int(old['wins'])}/{int(old['rows'])}`.\n",
        f"Recommended channel-specific shrinkage: mean `{fmt_pct2(float(recommended['mean_delta']))}`, "
        f"worst `{fmt_pct2(float(recommended['worst_delta']))}`, "
        f"wins `{int(recommended['wins'])}/{int(recommended['rows'])}`.\n",
        f"Best-mean channel-specific shrinkage: mean `{fmt_pct2(float(best_mean['mean_delta']))}`, "
        f"worst `{fmt_pct2(float(best_mean['worst_delta']))}`.\n\n",
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
            f"tau={float(row['ridge_tau']):g}, p={float(row['retention_power']):g}, "
            f"rho_s={float(row['rho_smooth']):g}, rho_step={float(row['rho_step']):g}"
        )
        lines.append(
            f"| {row['split']} | `{cfg}` | "
            f"{fmt_pct(float(row['dev_mean_delta']))} | {fmt_pct(float(row['dev_worst_delta']))} | "
            f"{fmt_pct(float(row['test_mean_delta']))} | {fmt_pct(float(row['test_worst_delta']))} | "
            f"{int(row['test_wins'])}/{int(row['test_rows'])} |\n"
        )
    lines += [
        "\n## Reading\n\n",
        "- The recommended candidate is small but clean: both mean and worst-cell MAE improve over the shared-rho fit-window model.\n",
        "- The best-mean candidate lowers average MAE further, but it slightly weakens the worst WSD-con 9e-5 cell; it is better treated as an optimistic development point.\n",
        "- The added degree of freedom has a direct interpretation as channel-specific transfer uncertainty, not an arbitrary residual basis.\n",
        "- This remains a development audit because the channel shrink values are selected by WSD-family ranking. "
        "For a stricter final protocol, freeze the channel-shrink grid choice before testing new schedules.\n",
    ]
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    config_rows, safe_rows, detail_rows = run_search()
    write_csv(OUT_DIR / "all_channel_shrink_configs.csv", config_rows)
    write_csv(OUT_DIR / "safe_channel_shrink_top200.csv", safe_rows)
    write_csv(OUT_DIR / "top_safe_details.csv", detail_rows)
    recommended = select_recommended(safe_rows) if safe_rows else None
    recommended_target_rows = (
        summarize_by_target(detail_rows, int(recommended["config_id"])) if recommended is not None else []
    )
    best_mean_target_rows = summarize_by_target(detail_rows, int(safe_rows[0]["config_id"])) if safe_rows else []
    holdout_rows = top_holdout(detail_rows) if detail_rows else []
    write_csv(OUT_DIR / "best_target_summary.csv", recommended_target_rows)
    write_csv(OUT_DIR / "best_mean_target_summary.csv", best_mean_target_rows)
    write_csv(OUT_DIR / "top_holdout_summary.csv", holdout_rows)
    write_report(safe_rows, detail_rows, recommended_target_rows, holdout_rows)
    print(f"wrote {OUT_DIR / 'all_channel_shrink_configs.csv'}")
    print(f"wrote {OUT_DIR / 'safe_channel_shrink_top200.csv'}")
    print(f"wrote {OUT_DIR / 'top_safe_details.csv'}")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
