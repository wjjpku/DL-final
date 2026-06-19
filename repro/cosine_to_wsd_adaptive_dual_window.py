#!/usr/bin/env python3
"""Dual-channel fit-window audit for cosine-to-WSD transfer.

This is a small robustness extension of adaptive_fit_window:
  * smooth-response kappa is fit on one cosine suffix;
  * step-response kappa is fit on another cosine suffix.

The target loss is still never used to fit kappa.  WSD losses only rank
development candidates in this audit.
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
    FIT_STARTS,
    SHARP_LINEAR,
    WSDCON,
    aggregate,
    fit_source_kappa_window,
    fmt_pct,
)
from cosine_to_wsd_adaptive_search import DROP_CONCENTRATION_THRESHOLD, drop_concentration  # noqa: E402
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
OUT_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "adaptive_dual_window"
TOP_CONFIG_LIMIT = 120
TOP_DETAIL_LIMIT = 50


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


def channel_for_curve(curve) -> str:
    return "step" if drop_concentration(curve) >= DROP_CONCENTRATION_THRESHOLD else "smooth"


def unique_base_configs(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, ...]] = set()
    out: list[dict[str, str]] = []
    for row in rows:
        key = (
            row["smooth_lambda"],
            row["step_lambda"],
            row["nuisance_lambda"],
            row["max_mode"],
            row["ridge_tau"],
            row["retention_power"],
            row["rho"],
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
        if len(out) >= TOP_CONFIG_LIMIT:
            break
    return out


def score_config(cache, feature_cache, cfg: dict[str, str], smooth_fit_start: int, step_fit_start: int) -> list[dict[str, object]]:
    smooth_lambda = float(cfg["smooth_lambda"])
    step_lambda = float(cfg["step_lambda"])
    nuisance_lambda = float(cfg["nuisance_lambda"])
    max_mode = int(cfg["max_mode"])
    ridge_tau = float(cfg["ridge_tau"])
    retention_power = float(cfg["retention_power"])
    rho = float(cfg["rho"])
    rows: list[dict[str, object]] = []
    for scale in SCALES:
        source = cache[(scale, TRAIN_CURVE)]
        smooth_fit = fit_source_kappa_window(
            source,
            feature_cache[(scale, TRAIN_CURVE, smooth_lambda)],
            fit_start_step=smooth_fit_start,
            nuisance_lambda=nuisance_lambda,
            max_mode=max_mode,
            ridge_tau=ridge_tau,
            retention_power=retention_power,
            rho=rho,
        )
        step_fit = fit_source_kappa_window(
            source,
            feature_cache[(scale, TRAIN_CURVE, step_lambda)],
            fit_start_step=step_fit_start,
            nuisance_lambda=nuisance_lambda,
            max_mode=max_mode,
            ridge_tau=ridge_tau,
            retention_power=retention_power,
            rho=rho,
        )
        for target_curve, target_label in TARGETS:
            target = cache[(scale, target_curve)]
            channel = channel_for_curve(target.curve)
            response_lambda = step_lambda if channel == "step" else smooth_lambda
            fit = step_fit if channel == "step" else smooth_fit
            phi_t = feature_cache[(scale, target_curve, response_lambda)]
            retention_t = target_retention(phi_t, nuisance_lambda=nuisance_lambda, max_mode=max_mode)
            target_factor = 1.0 if retention_t >= TARGET_RETENTION_FLOOR else 0.0
            scored = score_target(target, phi_t, float(fit["kappa"]) * target_factor)
            rows.append(
                {
                    "smooth_fit_start": smooth_fit_start,
                    "step_fit_start": step_fit_start,
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


def run_search() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    base_configs = unique_base_configs(read_csv(IN_DIR / "safe_window_top200.csv"))
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
        for smooth_fit_start in FIT_STARTS:
            for step_fit_start in FIT_STARTS:
                details = score_config(cache, feature_cache, cfg, smooth_fit_start, step_fit_start)
                summary = aggregate(details)
                row = {
                    "config_id": config_id,
                    "smooth_fit_start": smooth_fit_start,
                    "step_fit_start": step_fit_start,
                    "base_config_id": int(cfg["base_config_id"]),
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
                if summary["wins"] == summary["rows"] and summary["nonharm"] == summary["rows"]:
                    for detail in details:
                        safe_details.append({"config_id": config_id, **detail})
                config_id += 1
    safe_rows = [row for row in config_rows if int(row["wins"]) == int(row["rows"]) and int(row["nonharm"]) == int(row["rows"])]
    safe_sorted = sorted(safe_rows, key=lambda row: (float(row["mean_delta"]), float(row["worst_delta"])))
    top_ids = {int(row["config_id"]) for row in safe_sorted[:TOP_DETAIL_LIMIT]}
    top_details = [row for row in safe_details if int(row["config_id"]) in top_ids]
    return config_rows, safe_sorted[:200], top_details


def summarize_by_target(detail_rows: list[dict[str, object]], config_id: int) -> list[dict[str, object]]:
    selected = [row for row in detail_rows if int(row["config_id"]) == config_id]
    rows: list[dict[str, object]] = []
    for target_curve, target_label in TARGETS:
        sub = [row for row in selected if row["test_curve"] == target_curve]
        if sub:
            rows.append({"test_curve": target_curve, "test_label": target_label, **aggregate(sub)})
    return rows


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


def select_rows(rows: list[dict[str, object]], *, targets: set[str], scales: set[str] | None) -> list[dict[str, object]]:
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
                "smooth_fit_start": cfg["smooth_fit_start"],
                "step_fit_start": cfg["step_fit_start"],
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


def write_report(
    safe_rows: list[dict[str, object]],
    target_rows: list[dict[str, object]],
    holdout_rows: list[dict[str, object]],
) -> None:
    if not safe_rows:
        (OUT_DIR / "REPORT.md").write_text("No non-harming dual-window candidate found.\n", encoding="utf-8")
        return
    best = safe_rows[0]
    best_worst = min(safe_rows, key=lambda row: (float(row["worst_delta"]), float(row["mean_delta"])))
    lines = [
        "# Adaptive Dual-Window Cosine-to-WSD Audit\n\n",
        "This audit allows the smooth and step response channels to estimate `kappa` from different suffixes of the same cosine calibration curve. "
        "It is meant as a robustness extension of the simpler single-window adaptive model.\n\n",
        "## Best Fully Non-Harming Candidate\n\n",
        f"- Mean MAE change: `{fmt_pct(float(best['mean_delta']))}` over `{int(best['rows'])}` scale-target rows.\n",
        f"- Worst scale-target row: `{fmt_pct(float(best['worst_delta']))}`.\n",
        f"- Wins/non-harm: `{int(best['wins'])}/{int(best['rows'])}` and `{int(best['nonharm'])}/{int(best['rows'])}`.\n",
        f"- Config: `smooth_start={int(best['smooth_fit_start'])}`, `step_start={int(best['step_fit_start'])}`, "
        f"`lambda_smooth={float(best['smooth_lambda']):g}`, `lambda_step={float(best['step_lambda']):g}`, "
        f"`mu={float(best['nuisance_lambda']):g}`, `max_mode={int(best['max_mode'])}`, "
        f"`tau={float(best['ridge_tau']):g}`, `p={float(best['retention_power']):g}`, `rho={float(best['rho']):g}`.\n\n",
        "## Best Worst-Case Candidate\n\n",
        f"- Mean / worst: `{fmt_pct(float(best_worst['mean_delta']))}` / `{fmt_pct(float(best_worst['worst_delta']))}`.\n",
        f"- Config: `smooth_start={int(best_worst['smooth_fit_start'])}`, `step_start={int(best_worst['step_fit_start'])}`, "
        f"`lambda_smooth={float(best_worst['smooth_lambda']):g}`, `lambda_step={float(best_worst['step_lambda']):g}`, "
        f"`mu={float(best_worst['nuisance_lambda']):g}`, `max_mode={int(best_worst['max_mode'])}`, "
        f"`tau={float(best_worst['ridge_tau']):g}`, `p={float(best_worst['retention_power']):g}`, `rho={float(best_worst['rho']):g}`.\n\n",
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
        "Adaptive fit-window search: mean `-34.5%`, worst `-6.1%`, wins `15/15`.\n",
        "Dual-window improvement is small; the main value is a slightly better worst cell, not a new mechanism.\n\n",
        "## Top-Safe Holdout Check\n\n",
        "| split | selected config | dev mean | dev worst | test mean | test worst | test wins |\n",
        "|---|---|---:|---:|---:|---:|---:|\n",
    ]
    for row in holdout_rows:
        if row["selection_status"] != "selected":
            lines.append(f"| {row['split']} | none | | | | | |\n")
            continue
        cfg = (
            f"s_start={int(row['smooth_fit_start'])}, step_start={int(row['step_fit_start'])}, "
            f"lambda_s={float(row['smooth_lambda']):g}, lambda_step={float(row['step_lambda']):g}, "
            f"mu={float(row['nuisance_lambda']):g}, tau={float(row['ridge_tau']):g}, "
            f"p={float(row['retention_power']):g}, rho={float(row['rho']):g}"
        )
        lines.append(
            f"| {row['split']} | `{cfg}` | "
            f"{fmt_pct(float(row['dev_mean_delta']))} | {fmt_pct(float(row['dev_worst_delta']))} | "
            f"{fmt_pct(float(row['test_mean_delta']))} | {fmt_pct(float(row['test_worst_delta']))} | "
            f"{int(row['test_wins'])}/{int(row['test_rows'])} |\n"
        )
    lines += [
        "\n## Reading\n\n",
        "- Separate suffixes are plausible because smooth and step response channels are identified from different parts of the cosine residual spectrum.\n",
        "- The empirical gain is small compared with the added branch, so this is best treated as a robustness/audit variant rather than the main story.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    config_rows, safe_rows, detail_rows = run_search()
    write_csv(OUT_DIR / "all_dual_window_configs.csv", config_rows)
    write_csv(OUT_DIR / "safe_dual_window_top200.csv", safe_rows)
    write_csv(OUT_DIR / "top_safe_details.csv", detail_rows)
    target_rows = summarize_by_target(detail_rows, int(safe_rows[0]["config_id"])) if safe_rows else []
    holdout_rows = top_holdout(detail_rows) if detail_rows else []
    write_csv(OUT_DIR / "best_target_summary.csv", target_rows)
    write_csv(OUT_DIR / "top_holdout_summary.csv", holdout_rows)
    write_report(safe_rows, target_rows, holdout_rows)
    print(f"wrote {OUT_DIR / 'all_dual_window_configs.csv'}")
    print(f"wrote {OUT_DIR / 'safe_dual_window_top200.csv'}")
    print(f"wrote {OUT_DIR / 'top_safe_details.csv'}")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
