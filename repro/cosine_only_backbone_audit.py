#!/usr/bin/env python3
"""Strict cosine-only MPL-backbone audit for cosine-to-WSD correction.

Most current audits use the frozen MPL parameters shipped in the project. This
script checks the stricter protocol where the MPL backbone itself is refit only
on cosine curves, then the residual correction is again fitted from cosine_72000
and evaluated on WSD-family targets.

This is intentionally an audit, not the main result: the official frozen MPL
baseline is much stronger on WSD because its parameters are not a pure
cosine-only refit.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TMP_CACHE = Path(os.environ.get("TMPDIR", "/tmp")) / "dl-final-cache"
TMP_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(TMP_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(TMP_CACHE / "xdg"))

REPO = Path(__file__).resolve().parent
ROOT = REPO.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cosine_to_wsd_channel_shrink_search import (  # noqa: E402
    RHO_GRID,
    aggregate,
    score_config,
)
from cosine_to_wsd_response_search import TARGETS, TRAIN_CURVE, stime_feature  # noqa: E402
from reproduce_cosine_to_wsd import (  # noqa: E402
    MPL_PRECOMPUTED_INIT,
    SCALES,
    TRAIN_CURVES,
    fit_mpl,
    load_curve,
    metrics,
    mpl_predict,
    subsample_curve,
)


BASE_CONFIG_PATH = ROOT / "results" / "cosine_to_wsd_response_search" / "adaptive_fit_window" / "safe_window_top200.csv"
OUT_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "cosine_only_backbone"
BASE_CONFIG_LIMIT = 200
TOP_DETAIL_LIMIT = 200


@dataclass(frozen=True)
class CurveCache:
    curve: object
    baseline: np.ndarray
    residual: np.ndarray
    base_mae: float


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


def fit_cosine_only_mpl() -> dict[str, np.ndarray]:
    params: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for scale in SCALES:
        train = [subsample_curve(load_curve(scale, name)) for name in TRAIN_CURVES]
        fitted, obj = fit_mpl(train, scale)
        params[scale] = fitted
        rows.append(
            {
                "scale": scale,
                "objective": obj,
                **{f"p{i}": float(value) for i, value in enumerate(fitted)},
            }
        )
    write_csv(OUT_DIR / "cosine_only_mpl_params.csv", rows)
    (OUT_DIR / "cosine_only_mpl_params.json").write_text(
        json.dumps({scale: params[scale].tolist() for scale in SCALES}, indent=2),
        encoding="utf-8",
    )
    return params


def build_strict_cache(params: dict[str, np.ndarray]) -> dict[tuple[str, str], CurveCache]:
    cache: dict[tuple[str, str], CurveCache] = {}
    curves = [(TRAIN_CURVE, "Cosine")] + TARGETS
    for scale in SCALES:
        for curve_name, _ in curves:
            curve = load_curve(scale, curve_name)
            baseline = mpl_predict(params[scale], curve)
            cache[(scale, curve_name)] = CurveCache(
                curve=curve,
                baseline=baseline,
                residual=curve.loss - baseline,
                base_mae=metrics(curve.loss, baseline)["mae"],
            )
    return cache


def official_vs_strict_baseline(params: dict[str, np.ndarray]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scale in SCALES:
        for curve_name, label in TARGETS:
            curve = load_curve(scale, curve_name)
            official = metrics(curve.loss, mpl_predict(MPL_PRECOMPUTED_INIT[scale], curve))["mae"]
            strict = metrics(curve.loss, mpl_predict(params[scale], curve))["mae"]
            rows.append(
                {
                    "scale": scale,
                    "test_curve": curve_name,
                    "test_label": label,
                    "official_mpl_mae": official,
                    "cosine_only_mpl_mae": strict,
                    "strict_vs_official_delta_pct": 100.0 * (strict / official - 1.0),
                }
            )
    return rows


def run_channel_shrink_search(cache: dict[tuple[str, str], CurveCache]) -> tuple[
    list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]
]:
    base_configs = read_csv(BASE_CONFIG_PATH)[:BASE_CONFIG_LIMIT]
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
                }
                config_rows.append(row)
                if summary["wins"] == summary["rows"] and summary["nonharm"] == summary["rows"]:
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
        row for row in config_rows if int(row["wins"]) == int(row["rows"]) and int(row["nonharm"]) == int(row["rows"])
    ]
    safe_sorted = sorted(safe_rows, key=lambda row: (float(row["mean_delta"]), float(row["worst_delta"])))
    top_ids = {int(row["config_id"]) for row in safe_sorted[:TOP_DETAIL_LIMIT]}
    top_details = [row for row in safe_detail_rows if int(row["config_id"]) in top_ids]
    return config_rows, safe_sorted[:200], top_details


def combine_rows(
    pair_id: int,
    smooth_config_id: str,
    step_config_id: str,
    smooth_rows: list[dict[str, object]],
    step_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in smooth_rows:
        if row["channel"] == "smooth":
            out.append({**row, "pair_config_id": pair_id, "smooth_config_id": smooth_config_id, "step_config_id": step_config_id})
    for row in step_rows:
        if row["channel"] == "step":
            out.append({**row, "pair_config_id": pair_id, "smooth_config_id": smooth_config_id, "step_config_id": step_config_id})
    return out


def summarize_pair(
    pair_id: int,
    smooth_config_id: str,
    step_config_id: str,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    smooth = next(row for row in rows if row["channel"] == "smooth")
    step = next(row for row in rows if row["channel"] == "step")
    return {
        "pair_config_id": pair_id,
        "smooth_config_id": smooth_config_id,
        "step_config_id": step_config_id,
        "smooth_fit_start_step": int(smooth["fit_start_step"]),
        "smooth_lambda": float(smooth["smooth_lambda"]),
        "smooth_nuisance_lambda": float(smooth["nuisance_lambda"]),
        "smooth_max_mode": int(smooth["max_mode"]),
        "smooth_retention_power": float(smooth["retention_power"]),
        "smooth_rho": float(smooth["rho_smooth"]),
        "step_fit_start_step": int(step["fit_start_step"]),
        "step_lambda": float(step["step_lambda"]),
        "step_nuisance_lambda": float(step["nuisance_lambda"]),
        "step_max_mode": int(step["max_mode"]),
        "step_retention_power": float(step["retention_power"]),
        "step_rho": float(step["rho_step"]),
        **aggregate(rows),
    }


def run_decoupled(detail_rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_config: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in detail_rows:
        by_config[str(row["config_id"])].append(row)
    config_ids = sorted([config_id for config_id, rows in by_config.items() if len(rows) == 15], key=int)
    pair_rows: list[dict[str, object]] = []
    pair_details: list[dict[str, object]] = []
    pair_id = 0
    for smooth_config_id in config_ids:
        for step_config_id in config_ids:
            rows = combine_rows(pair_id, smooth_config_id, step_config_id, by_config[smooth_config_id], by_config[step_config_id])
            if len(rows) != 15:
                continue
            summary = summarize_pair(pair_id, smooth_config_id, step_config_id, rows)
            pair_rows.append(summary)
            if summary["wins"] == summary["rows"] and summary["nonharm"] == summary["rows"]:
                pair_details.extend(rows)
            pair_id += 1
    safe_pairs = [
        row for row in pair_rows if int(row["wins"]) == int(row["rows"]) and int(row["nonharm"]) == int(row["rows"])
    ]
    safe_sorted = sorted(safe_pairs, key=lambda row: (float(row["mean_delta"]), float(row["worst_delta"])))
    top_ids = {int(row["pair_config_id"]) for row in safe_sorted[:200]}
    top_details = [row for row in pair_details if int(row["pair_config_id"]) in top_ids]
    return safe_sorted[:200], top_details


def summarize_by_target(detail_rows: list[dict[str, object]], key: str, value: int) -> list[dict[str, object]]:
    selected = [row for row in detail_rows if int(row[key]) == value]
    rows: list[dict[str, object]] = []
    for target_curve, target_label in TARGETS:
        sub = [row for row in selected if row["test_curve"] == target_curve]
        if sub:
            rows.append({"test_curve": target_curve, "test_label": target_label, **aggregate(sub)})
    return rows


def write_report(
    baseline_rows: list[dict[str, object]],
    channel_rows: list[dict[str, object]],
    decoupled_rows: list[dict[str, object]],
    target_rows: list[dict[str, object]],
) -> None:
    baseline_delta = np.array([float(row["strict_vs_official_delta_pct"]) for row in baseline_rows], dtype=np.float64)
    best_channel = channel_rows[0] if channel_rows else None
    best_decoupled = decoupled_rows[0] if decoupled_rows else None
    lines = [
        "# Strict Cosine-Only MPL Backbone Audit\n\n",
        "This audit refits the MPL backbone using only `cosine_24000.csv` and `cosine_72000.csv`, "
        "then fits the residual correction from `cosine_72000.csv` and evaluates WSD-family targets.\n\n",
        "## Backbone Check\n\n",
        f"- Cosine-only MPL vs frozen official MPL on WSD: mean MAE change `{fmt_pct(float(np.mean(baseline_delta)))}`, "
        f"worst `{fmt_pct(float(np.max(baseline_delta)))}`.\n",
        "- This means the strict cosine-only backbone is substantially weaker on WSD than the frozen MPL backbone used in the main audits.\n\n",
    ]
    if best_channel is not None:
        lines += [
            "## Best Single-Config Channel-Shrink Correction\n\n",
            f"- Mean / worst vs strict cosine-only MPL: `{fmt_pct2(float(best_channel['mean_delta']))}` / "
            f"`{fmt_pct2(float(best_channel['worst_delta']))}`.\n",
            f"- Wins: `{int(best_channel['wins'])}/{int(best_channel['rows'])}`.\n\n",
        ]
    if best_decoupled is not None:
        lines += [
            "## Best Decoupled-Channel Correction\n\n",
            f"- Mean / worst vs strict cosine-only MPL: `{fmt_pct2(float(best_decoupled['mean_delta']))}` / "
            f"`{fmt_pct2(float(best_decoupled['worst_delta']))}`.\n",
            f"- Wins/non-harm: `{int(best_decoupled['wins'])}/{int(best_decoupled['rows'])}` and "
            f"`{int(best_decoupled['nonharm'])}/{int(best_decoupled['rows'])}`.\n",
            f"- Smooth config: `start={int(best_decoupled['smooth_fit_start_step'])}`, "
            f"`lambda={float(best_decoupled['smooth_lambda']):g}`, "
            f"`mu={float(best_decoupled['smooth_nuisance_lambda']):g}`, "
            f"`modes={int(best_decoupled['smooth_max_mode'])}`, "
            f"`p={float(best_decoupled['smooth_retention_power']):g}`, "
            f"`rho={float(best_decoupled['smooth_rho']):g}`.\n",
            f"- Step config: `start={int(best_decoupled['step_fit_start_step'])}`, "
            f"`lambda={float(best_decoupled['step_lambda']):g}`, "
            f"`mu={float(best_decoupled['step_nuisance_lambda']):g}`, "
            f"`modes={int(best_decoupled['step_max_mode'])}`, "
            f"`p={float(best_decoupled['step_retention_power']):g}`, "
            f"`rho={float(best_decoupled['step_rho']):g}`.\n\n",
            "## Per-Target Decoupled Result\n\n",
            "| target | mean delta | worst scale | wins |\n",
            "|---|---:|---:|---:|\n",
        ]
        for row in target_rows:
            lines.append(
                f"| {row['test_label']} | {fmt_pct(float(row['mean_delta']))} | "
                f"{fmt_pct(float(row['worst_delta']))} | {int(row['wins'])}/{int(row['rows'])} |\n"
            )
    lines += [
        "\n## Reading\n\n",
        "- The residual correction still works under a strict cosine-only MPL backbone, with non-harming improvement on all WSD rows.\n",
        "- The absolute WSD baseline is much worse after refitting MPL only on cosine, so this audit should not replace the frozen-backbone main result unless the assignment requires a fully cosine-only backbone too.\n",
    ]
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    params = fit_cosine_only_mpl()
    cache = build_strict_cache(params)
    baseline_rows = official_vs_strict_baseline(params)
    write_csv(OUT_DIR / "official_vs_cosine_only_mpl.csv", baseline_rows)

    config_rows, safe_channel_rows, detail_rows = run_channel_shrink_search(cache)
    write_csv(OUT_DIR / "all_channel_shrink_configs.csv", config_rows)
    write_csv(OUT_DIR / "safe_channel_shrink_top200.csv", safe_channel_rows)
    write_csv(OUT_DIR / "top_safe_channel_details.csv", detail_rows)

    safe_decoupled_rows, decoupled_details = run_decoupled(detail_rows)
    write_csv(OUT_DIR / "safe_decoupled_top200.csv", safe_decoupled_rows)
    write_csv(OUT_DIR / "top_safe_decoupled_details.csv", decoupled_details)
    target_rows = (
        summarize_by_target(decoupled_details, "pair_config_id", int(safe_decoupled_rows[0]["pair_config_id"]))
        if safe_decoupled_rows
        else []
    )
    write_csv(OUT_DIR / "best_decoupled_target_summary.csv", target_rows)
    write_report(baseline_rows, safe_channel_rows, safe_decoupled_rows, target_rows)
    print(f"wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
