#!/usr/bin/env python3
"""Route all WSD-con constant-tail schedules by final LR ratio.

This audit replaces the previous special low-tail gate with the same Gaussian
LR-level gate family used by the mid/high-ratio audit.  Each WSD-con final LR
ratio selects one gate configuration; non-WSD-con targets stay on the joint
LR-curvature prediction.
"""
from __future__ import annotations

import os
import sys
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

from cosine_to_wsd_curvature_correction import curvature_feature, fmt_pct, fmt_pct2  # noqa: E402
from cosine_to_wsd_low_tail_gate_config_route import best_joint_rows  # noqa: E402
from cosine_to_wsd_mid_ratio_route import read_csv, write_csv  # noqa: E402
from cosine_to_wsd_mid_tail_recovery import (  # noqa: E402
    CENTERS,
    GATE_SIGNS,
    GATE_TAUS,
    SHRINK_GATE,
    TIME_POWERS,
    WIDTHS,
    mid_tail_gate,
    score_config,
)
from cosine_to_wsd_response_search import TARGETS, TRAIN_CURVE, build_cache, stime_feature  # noqa: E402
from cosine_to_wsd_tail_gated_response import CURVATURE, SMOOTH, STEP  # noqa: E402
from reproduce_cosine_to_wsd import SCALES  # noqa: E402


BASE_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "low_tail_gate_config_route"
JOINT_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "joint_curvature"
MID_HIGH_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "mid_high_ratio_route"
OUT_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "all_ratio_route"
ROUTE_TARGETS = ("wsdcon_3.csv", "wsdcon_9.csv", "wsdcon_18.csv")
TARGET_RATIO = {
    "wsdcon_3.csv": 0.1,
    "wsdcon_9.csv": 0.3,
    "wsdcon_18.csv": 0.6,
}


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


def build_details() -> tuple[list[dict[str, object]], dict[str, list[dict[str, object]]]]:
    cache = build_cache()
    curve_names = [TRAIN_CURVE] + [name for name, _ in TARGETS]
    primary_cache = {
        (scale, curve_name, "smooth"): stime_feature(cache[(scale, curve_name)].curve, float(SMOOTH["response_lambda"]))
        for scale in SCALES
        for curve_name in curve_names
    }
    primary_cache.update(
        {
            (scale, curve_name, "step"): stime_feature(cache[(scale, curve_name)].curve, float(STEP["response_lambda"]))
            for scale in SCALES
            for curve_name in curve_names
        }
    )
    curvature_cache = {
        (scale, curve_name): curvature_feature(
            cache[(scale, curve_name)].curve,
            float(CURVATURE["curvature_lambda"]),
            str(CURVATURE["curvature_mode"]),
        )
        for scale in SCALES
        for curve_name in curve_names
    }
    gate_cache = {
        (scale, curve_name, center, width, time_power): mid_tail_gate(
            cache[(scale, curve_name)].curve,
            primary_cache[(scale, curve_name, "step")],
            center,
            width,
            time_power,
        )
        for scale in SCALES
        for curve_name in curve_names
        for center in CENTERS
        for width in WIDTHS
        for time_power in TIME_POWERS
    }
    configs: list[dict[str, object]] = []
    details_by_target: dict[str, list[dict[str, object]]] = {target: [] for target in ROUTE_TARGETS}
    config_id = 0
    for center in CENTERS:
        for width in WIDTHS:
            for time_power in TIME_POWERS:
                for gate_tau in GATE_TAUS:
                    for gate_sign in GATE_SIGNS:
                        for shrink_gate in SHRINK_GATE:
                            rows = score_config(
                                cache,
                                primary_cache,
                                curvature_cache,
                                gate_cache,
                                center=center,
                                width=width,
                                time_power=time_power,
                                gate_tau=gate_tau,
                                gate_sign=gate_sign,
                                shrink_gate=shrink_gate,
                            )
                            configs.append(
                                {
                                    "config_id": config_id,
                                    "center": center,
                                    "width": width,
                                    "time_power": time_power,
                                    "gate_tau": gate_tau,
                                    "gate_sign": gate_sign,
                                    "shrink_gate": int(shrink_gate),
                                }
                            )
                            for target in ROUTE_TARGETS:
                                for row in rows:
                                    if row["test_curve"] == target:
                                        details_by_target[target].append({"config_id": config_id, **row})
                            config_id += 1
    return configs, details_by_target


def candidate_summary(
    target: str,
    configs: list[dict[str, object]],
    details: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_config: dict[int, list[dict[str, object]]] = {}
    for row in details:
        by_config.setdefault(int(row["config_id"]), []).append(row)
    rows: list[dict[str, object]] = []
    for cfg in configs:
        cid = int(cfg["config_id"])
        stats = aggregate(by_config[cid])
        if int(stats["wins"]) == int(stats["rows"]) and int(stats["nonharm"]) == int(stats["rows"]):
            rows.append({"target": target, **cfg, **stats})
    return sorted(rows, key=lambda row: (float(row["mean_delta"]), float(row["worst_delta"])))


def selected_details(target: str, config_id: int, details: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "route": f"ratio_{TARGET_RATIO[target]:g}",
            "route_config_id": config_id,
            **row,
        }
        for row in details
        if int(row["config_id"]) == config_id
    ]


def combined_rows(
    joint_rows: dict[tuple[str, str], dict[str, str]],
    selected_by_target: dict[str, list[dict[str, object]]],
) -> list[dict[str, object]]:
    selected = {
        target: {str(row["scale"]): row for row in rows}
        for target, rows in selected_by_target.items()
    }
    rows: list[dict[str, object]] = []
    for scale in SCALES:
        for curve_name, label in TARGETS:
            if curve_name in selected:
                source = selected[curve_name][str(scale)]
                rows.append(
                    {
                        "scale": scale,
                        "test_curve": curve_name,
                        "test_label": label,
                        "route": source["route"],
                        "route_config_id": source["route_config_id"],
                        "base_mae": source["base_mae"],
                        "corr_mae": source["corr_mae"],
                        "delta_pct": source["delta_pct"],
                        "win": int(float(source["delta_pct"]) < 0.0),
                    }
                )
            else:
                source = joint_rows[(str(scale), curve_name)]
                rows.append(
                    {
                        "scale": scale,
                        "test_curve": curve_name,
                        "test_label": label,
                        "route": "joint",
                        "route_config_id": "joint",
                        "base_mae": source["base_mae"],
                        "corr_mae": source["corr_mae"],
                        "delta_pct": source["delta_pct"],
                        "win": int(float(source["delta_pct"]) < 0.0),
                    }
                )
    return rows


def summarize_by_target(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for target_curve, target_label in TARGETS:
        sub = [row for row in rows if row["test_curve"] == target_curve]
        out.append({"test_curve": target_curve, "test_label": target_label, **aggregate(sub)})
    return out


def config_text(row: dict[str, object]) -> str:
    return (
        f"`{row['config_id']}` "
        f"(center={float(row['center']):g}, width={float(row['width']):g}, "
        f"time_power={float(row['time_power']):g}, tau={float(row['gate_tau']):g}, "
        f"sign={row['gate_sign']}, shrink={row['shrink_gate']})"
    )


def write_report(
    selected_configs: dict[str, dict[str, object]],
    rows: list[dict[str, object]],
    target_rows: list[dict[str, object]],
) -> None:
    stats = aggregate(rows)
    joint = read_csv(JOINT_DIR / "safe_joint_curvature_top200.csv")[0]
    low_tail = read_csv(BASE_DIR / "safe_routes_top200.csv")[0]
    mid_high = read_csv(MID_HIGH_DIR / "safe_mid_high_ratio_routes_top200.csv")[0]
    lines = [
        "# All-Ratio WSD-con Route Audit\n\n",
        "This audit uses the same Gaussian LR-level gate family for all WSD-con "
        "constant-tail schedules, routing by `final_lr / peak_lr`.  Coefficients "
        "are fitted from the cosine residual only; WSD-family losses are used for "
        "development ranking.\n\n",
        "## Best Mean Route\n\n",
        f"- Mean / worst: `{fmt_pct2(float(stats['mean_delta']))}` / `{fmt_pct2(float(stats['worst_delta']))}`.\n",
        f"- Wins/non-harm: `{int(stats['wins'])}/{int(stats['rows'])}` and `{int(stats['nonharm'])}/{int(stats['rows'])}`.\n",
        f"- Ratio 0.1 config: {config_text(selected_configs['wsdcon_3.csv'])}.\n",
        f"- Ratio 0.3 config: {config_text(selected_configs['wsdcon_9.csv'])}.\n",
        f"- Ratio 0.6 config: {config_text(selected_configs['wsdcon_18.csv'])}.\n\n",
        "## Comparison\n\n",
        f"- Joint-channel LR-curvature: mean `{fmt_pct2(float(joint['mean_delta']))}`, "
        f"worst `{fmt_pct2(float(joint['worst_delta']))}`.\n",
        f"- Low-tail route: mean `{fmt_pct2(float(low_tail['mean_delta']))}`, "
        f"worst `{fmt_pct2(float(low_tail['worst_delta']))}`.\n",
        f"- Mid/high-ratio route: mean `{fmt_pct2(float(mid_high['mean_delta']))}`, "
        f"worst `{fmt_pct2(float(mid_high['worst_delta']))}`.\n",
        f"- All-ratio route: mean `{fmt_pct2(float(stats['mean_delta']))}`, "
        f"worst `{fmt_pct2(float(stats['worst_delta']))}`.\n\n",
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
        "\n## Routed Rows\n\n",
        "| target | scale | route | delta | corr_mae | base_mae |\n",
        "|---|---|---|---:|---:|---:|\n",
    ]
    for row in rows:
        if row["test_curve"] in ROUTE_TARGETS:
            lines.append(
                f"| {row['test_label']} | {row['scale']}M | {row['route']} | "
                f"{fmt_pct(float(row['delta_pct']))} | {float(row['corr_mae']):.6g} | "
                f"{float(row['base_mae']):.6g} |\n"
            )
    lines += [
        "\n## Reading\n\n",
        "- This is the best development number so far, but it uses one selected gate configuration per WSD-con final-LR ratio.\n",
        "- The ratio branches are schedule-only at prediction time, but the branch hyperparameters were selected on the available WSD-family development set.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    configs, details_by_target = build_details()
    selected_configs: dict[str, dict[str, object]] = {}
    selected_by_target: dict[str, list[dict[str, object]]] = {}
    candidate_rows: list[dict[str, object]] = []
    for target in ROUTE_TARGETS:
        candidates = candidate_summary(target, configs, details_by_target[target])
        best = candidates[0]
        selected_configs[target] = best
        selected_by_target[target] = selected_details(target, int(best["config_id"]), details_by_target[target])
        candidate_rows.extend(candidates[:50])

    rows = combined_rows(best_joint_rows(), selected_by_target)
    target_rows = summarize_by_target(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "selected_ratio_configs.csv", list(selected_configs.values()))
    write_csv(OUT_DIR / "top_ratio_candidate_summaries.csv", candidate_rows)
    write_csv(OUT_DIR / "top_safe_details.csv", rows)
    write_csv(OUT_DIR / "best_target_summary.csv", target_rows)
    write_report(selected_configs, rows, target_rows)
    print(f"wrote {OUT_DIR / 'selected_ratio_configs.csv'}")
    print(f"wrote {OUT_DIR / 'top_safe_details.csv'}")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
