#!/usr/bin/env python3
"""Two-ratio route for moderate/high WSD-con tails.

This extends the mid-ratio audit: keep the current low-tail route for
WSD-con 3e-5, then allow WSD-con 9e-5 and WSD-con 18e-5 to choose separate
mid-tail recovery candidates based only on their final LR ratio.
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
from cosine_to_wsd_mid_ratio_route import (  # noqa: E402
    JOINT_DIR,
    base_route_rows,
    read_csv,
    summarize_by_target,
    write_csv,
)
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
OUT_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "mid_high_ratio_route"
ROUTE_TARGETS = ("wsdcon_9.csv", "wsdcon_18.csv")
TARGET_RATIO = {
    "wsdcon_9.csv": 0.3,
    "wsdcon_18.csv": 0.6,
}
TOP_LIMIT = 200


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


def build_route_target_details() -> tuple[list[dict[str, object]], dict[str, list[dict[str, object]]]]:
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

    config_rows: list[dict[str, object]] = []
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
                            row = {
                                "config_id": config_id,
                                "center": center,
                                "width": width,
                                "time_power": time_power,
                                "gate_tau": gate_tau,
                                "gate_sign": gate_sign,
                                "shrink_gate": int(shrink_gate),
                            }
                            for target in ROUTE_TARGETS:
                                sub = [detail for detail in rows if detail["test_curve"] == target]
                                stats = aggregate(sub)
                                row.update({f"{target}_{key}": value for key, value in stats.items()})
                                for detail in sub:
                                    details_by_target[target].append({"config_id": config_id, **detail})
                            config_rows.append(row)
                            config_id += 1
    return config_rows, details_by_target


def target_candidates(
    target: str,
    config_rows: list[dict[str, object]],
    details: list[dict[str, object]],
    base_rows: dict[tuple[str, str], dict[str, object]],
) -> list[dict[str, object]]:
    base_details = [base_rows[(str(scale), target)] for scale in SCALES]
    out: list[dict[str, object]] = [
        {
            "target": target,
            "config_id": "base",
            "center": "",
            "width": "",
            "time_power": "",
            "gate_tau": "",
            "gate_sign": "",
            "shrink_gate": "",
            "details": base_details,
            **aggregate(base_details),
        }
    ]
    by_config: dict[int, list[dict[str, object]]] = {}
    for row in details:
        by_config.setdefault(int(row["config_id"]), []).append(row)
    for cfg in config_rows:
        cid = int(cfg["config_id"])
        rows = by_config[cid]
        stats = aggregate(rows)
        if int(stats["wins"]) == int(stats["rows"]) and int(stats["nonharm"]) == int(stats["rows"]):
            out.append(
                {
                    "target": target,
                    "config_id": cid,
                    "center": cfg["center"],
                    "width": cfg["width"],
                    "time_power": cfg["time_power"],
                    "gate_tau": cfg["gate_tau"],
                    "gate_sign": cfg["gate_sign"],
                    "shrink_gate": cfg["shrink_gate"],
                    "details": rows,
                    **stats,
                }
            )
    return out


def combined_rows(
    base_rows: dict[tuple[str, str], dict[str, object]],
    con9: dict[str, object],
    con18: dict[str, object],
) -> list[dict[str, object]]:
    selected = {
        "wsdcon_9.csv": {str(row["scale"]): row for row in con9["details"]},
        "wsdcon_18.csv": {str(row["scale"]): row for row in con18["details"]},
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
                        "route": f"ratio_{TARGET_RATIO[curve_name]:g}",
                        "route_config_id": con9["config_id"] if curve_name == "wsdcon_9.csv" else con18["config_id"],
                        "base_mae": source["base_mae"],
                        "corr_mae": source["corr_mae"],
                        "delta_pct": source["delta_pct"],
                        "win": int(float(source["delta_pct"]) < 0.0),
                    }
                )
            else:
                source = base_rows[(str(scale), curve_name)]
                rows.append(
                    {
                        "scale": scale,
                        "test_curve": curve_name,
                        "test_label": label,
                        "route": source["route"],
                        "route_config_id": "base",
                        "base_mae": source["base_mae"],
                        "corr_mae": source["corr_mae"],
                        "delta_pct": source["delta_pct"],
                        "win": source["win"],
                    }
                )
    return rows


def search_routes(
    base_rows: dict[tuple[str, str], dict[str, object]],
    con9_candidates: list[dict[str, object]],
    con18_candidates: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    combos: list[dict[str, object]] = []
    best_mean: dict[str, object] | None = None
    best_worst: dict[str, object] | None = None
    for con9 in con9_candidates:
        for con18 in con18_candidates:
            rows = combined_rows(base_rows, con9, con18)
            stats = aggregate(rows)
            combo = {
                "con9_config_id": con9["config_id"],
                "con9_mean_delta": con9["mean_delta"],
                "con9_worst_delta": con9["worst_delta"],
                "con9_center": con9["center"],
                "con9_width": con9["width"],
                "con9_time_power": con9["time_power"],
                "con9_gate_tau": con9["gate_tau"],
                "con9_gate_sign": con9["gate_sign"],
                "con9_shrink_gate": con9["shrink_gate"],
                "con18_config_id": con18["config_id"],
                "con18_mean_delta": con18["mean_delta"],
                "con18_worst_delta": con18["worst_delta"],
                "con18_center": con18["center"],
                "con18_width": con18["width"],
                "con18_time_power": con18["time_power"],
                "con18_gate_tau": con18["gate_tau"],
                "con18_gate_sign": con18["gate_sign"],
                "con18_shrink_gate": con18["shrink_gate"],
                **stats,
            }
            if int(stats["wins"]) == int(stats["rows"]) and int(stats["nonharm"]) == int(stats["rows"]):
                combos.append(combo)
                if best_mean is None or (float(combo["mean_delta"]), float(combo["worst_delta"])) < (
                    float(best_mean["mean_delta"]),
                    float(best_mean["worst_delta"]),
                ):
                    best_mean = combo
                if best_worst is None or (float(combo["worst_delta"]), float(combo["mean_delta"])) < (
                    float(best_worst["worst_delta"]),
                    float(best_worst["mean_delta"]),
                ):
                    best_worst = combo
    if best_mean is None or best_worst is None:
        raise RuntimeError("no safe route combination found")
    return sorted(combos, key=lambda row: (float(row["mean_delta"]), float(row["worst_delta"])))[:TOP_LIMIT], best_mean, best_worst


def candidate_by_id(candidates: list[dict[str, object]], config_id: object) -> dict[str, object]:
    return next(row for row in candidates if str(row["config_id"]) == str(config_id))


def config_text(prefix: str, row: dict[str, object]) -> str:
    if str(row[f"{prefix}_config_id"]) == "base":
        return "`base`"
    return (
        f"`{row[f'{prefix}_config_id']}` "
        f"(center={float(row[f'{prefix}_center']):g}, width={float(row[f'{prefix}_width']):g}, "
        f"time_power={float(row[f'{prefix}_time_power']):g}, tau={float(row[f'{prefix}_gate_tau']):g}, "
        f"sign={row[f'{prefix}_gate_sign']}, shrink={row[f'{prefix}_shrink_gate']})"
    )


def write_report(
    combo_rows: list[dict[str, object]],
    best_worst: dict[str, object],
    selected_rows: list[dict[str, object]],
    target_rows: list[dict[str, object]],
    base_stats: dict[str, object],
) -> None:
    best = combo_rows[0]
    joint = read_csv(JOINT_DIR / "safe_joint_curvature_top200.csv")[0]
    low_tail = read_csv(BASE_DIR / "safe_routes_top200.csv")[0]
    lines = [
        "# Mid/High-Ratio Route Audit\n\n",
        "This audit keeps the low-tail route for WSD-con 3e-5 and routes WSD-con 9e-5 "
        "and WSD-con 18e-5 by their final LR ratios.  Each routed candidate is fitted "
        "from the cosine residual only; WSD-family losses are used for development ranking.\n\n",
        "## Best Fully Non-Harming Route\n\n",
        f"- Mean / worst: `{fmt_pct2(float(best['mean_delta']))}` / `{fmt_pct2(float(best['worst_delta']))}`.\n",
        f"- Wins/non-harm: `{int(best['wins'])}/{int(best['rows'])}` and `{int(best['nonharm'])}/{int(best['rows'])}`.\n",
        f"- Ratio 0.3 config: {config_text('con9', best)}.\n",
        f"- Ratio 0.6 config: {config_text('con18', best)}.\n\n",
        "## Best Worst-Case Route\n\n",
        f"- Mean / worst: `{fmt_pct2(float(best_worst['mean_delta']))}` / "
        f"`{fmt_pct2(float(best_worst['worst_delta']))}`.\n\n",
        "## Comparison\n\n",
        f"- Joint-channel LR-curvature: mean `{fmt_pct2(float(joint['mean_delta']))}`, "
        f"worst `{fmt_pct2(float(joint['worst_delta']))}`.\n",
        f"- Low-tail route base: mean `{fmt_pct2(float(low_tail['mean_delta']))}`, "
        f"worst `{fmt_pct2(float(low_tail['worst_delta']))}`.\n",
        f"- Mid/high-ratio route: mean `{fmt_pct2(float(best['mean_delta']))}`, "
        f"worst `{fmt_pct2(float(best['worst_delta']))}`.\n",
        f"- Base recomputed in this script: mean `{fmt_pct2(float(base_stats['mean_delta']))}`, "
        f"worst `{fmt_pct2(float(base_stats['worst_delta']))}`.\n\n",
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
    for row in selected_rows:
        if row["test_curve"] in ROUTE_TARGETS:
            lines.append(
                f"| {row['test_label']} | {row['scale']}M | {row['route']} | "
                f"{fmt_pct(float(row['delta_pct']))} | {float(row['corr_mae']):.6g} | "
                f"{float(row['base_mae']):.6g} |\n"
            )
    lines += [
        "\n## Reading\n\n",
        "- This adds schedule-ratio routing, not per-scale routing.  All scales at the same final LR ratio use the same gate configuration.\n",
        "- The extra flexibility is selected on the available WSD-family development set, so it should be frozen before any stronger generalization claim.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    _base_cfg, base = base_route_rows()
    config_rows, details_by_target = build_route_target_details()
    con9_candidates = target_candidates("wsdcon_9.csv", config_rows, details_by_target["wsdcon_9.csv"], base)
    con18_candidates = target_candidates("wsdcon_18.csv", config_rows, details_by_target["wsdcon_18.csv"], base)
    base_rows = [
        base[(str(scale), curve_name)]
        for scale in SCALES
        for curve_name, _label in TARGETS
    ]
    base_stats = aggregate(base_rows)
    combo_rows, best_mean, best_worst = search_routes(base, con9_candidates, con18_candidates)
    con9 = candidate_by_id(con9_candidates, best_mean["con9_config_id"])
    con18 = candidate_by_id(con18_candidates, best_mean["con18_config_id"])
    selected_rows = combined_rows(base, con9, con18)
    target_rows = summarize_by_target(selected_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "safe_mid_high_ratio_routes_top200.csv", combo_rows)
    write_csv(OUT_DIR / "top_safe_details.csv", selected_rows)
    write_csv(OUT_DIR / "best_target_summary.csv", target_rows)
    write_report(combo_rows, best_worst, selected_rows, target_rows, base_stats)
    print(f"wrote {OUT_DIR / 'safe_mid_high_ratio_routes_top200.csv'}")
    print(f"wrote {OUT_DIR / 'top_safe_details.csv'}")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
