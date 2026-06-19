#!/usr/bin/env python3
"""Route moderate-tail WSD-con schedules to a mid-tail recovery candidate.

The current best development candidate routes only the lowest WSD-con final LR
to a tail-gated correction.  The remaining bottleneck is WSD-con 9e-5, whose
final LR ratio is 0.3.  This audit keeps the current low-tail route as the
base model and searches whether schedules with final_lr / peak_lr == 0.3
should switch to a mid-tail recovery feature fitted from cosine residuals only.
"""
from __future__ import annotations

import csv
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
OUT_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "mid_ratio_route"
MID_TARGET = "wsdcon_9.csv"
TOP_LIMIT = 200


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


def base_route_rows() -> tuple[dict[str, object], dict[tuple[str, str], dict[str, object]]]:
    best = read_csv(BASE_DIR / "safe_routes_top200.csv")[0]
    rows = read_csv(BASE_DIR / "details.csv")
    selected: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        if (
            abs(float(row["threshold"]) - float(best["threshold"])) < 1e-12
            and row["tail_config_id"] == best["tail_config_id"]
        ):
            selected[(row["scale"], row["test_curve"])] = {
                **row,
                "source": "base_low_tail",
                "mid_config_id": "",
                "center": "",
                "width": "",
                "time_power": "",
                "gate_tau": row.get("gate_tau", ""),
                "gate_sign": row.get("gate_sign", ""),
                "shrink_gate": row.get("shrink_gate", ""),
            }
    return best, selected


def build_mid_tail_details() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
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
    details: list[dict[str, object]] = []
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
                            mid_rows = [row for row in rows if row["test_curve"] == MID_TARGET]
                            stats = aggregate(mid_rows)
                            config_rows.append(
                                {
                                    "mid_config_id": config_id,
                                    "center": center,
                                    "width": width,
                                    "time_power": time_power,
                                    "gate_tau": gate_tau,
                                    "gate_sign": gate_sign,
                                    "shrink_gate": int(shrink_gate),
                                    **{f"mid_{key}": value for key, value in stats.items()},
                                }
                            )
                            for row in mid_rows:
                                details.append({"mid_config_id": config_id, **row})
                            config_id += 1
    return config_rows, details


def route_rows(
    base_rows: dict[tuple[str, str], dict[str, object]],
    mid_details: list[dict[str, object]],
    mid_config: dict[str, object] | None,
) -> list[dict[str, object]]:
    mid_by_key: dict[tuple[str, str], dict[str, object]] = {}
    if mid_config is not None:
        cid = int(mid_config["mid_config_id"])
        for row in mid_details:
            if int(row["mid_config_id"]) == cid:
                mid_by_key[(str(row["scale"]), str(row["test_curve"]))] = row

    rows: list[dict[str, object]] = []
    for scale in SCALES:
        for curve_name, label in TARGETS:
            key = (str(scale), curve_name)
            if mid_config is not None and curve_name == MID_TARGET:
                source = mid_by_key[key]
                rows.append(
                    {
                        "scale": scale,
                        "test_curve": curve_name,
                        "test_label": label,
                        "route": "mid_tail",
                        "mid_config_id": mid_config["mid_config_id"],
                        "center": mid_config["center"],
                        "width": mid_config["width"],
                        "time_power": mid_config["time_power"],
                        "gate_tau": mid_config["gate_tau"],
                        "gate_sign": mid_config["gate_sign"],
                        "shrink_gate": mid_config["shrink_gate"],
                        "base_mae": source["base_mae"],
                        "corr_mae": source["corr_mae"],
                        "delta_pct": source["delta_pct"],
                        "win": int(float(source["delta_pct"]) < 0.0),
                    }
                )
            else:
                source = base_rows[key]
                rows.append(
                    {
                        "scale": scale,
                        "test_curve": curve_name,
                        "test_label": label,
                        "route": source["route"],
                        "mid_config_id": mid_config["mid_config_id"] if mid_config else "base",
                        "center": mid_config["center"] if mid_config else "",
                        "width": mid_config["width"] if mid_config else "",
                        "time_power": mid_config["time_power"] if mid_config else "",
                        "gate_tau": mid_config["gate_tau"] if mid_config else "",
                        "gate_sign": mid_config["gate_sign"] if mid_config else "",
                        "shrink_gate": mid_config["shrink_gate"] if mid_config else "",
                        "base_mae": source["base_mae"],
                        "corr_mae": source["corr_mae"],
                        "delta_pct": source["delta_pct"],
                        "win": source["win"],
                    }
                )
    return rows


def summarize_by_target(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for target_curve, target_label in TARGETS:
        sub = [row for row in rows if row["test_curve"] == target_curve]
        out.append({"test_curve": target_curve, "test_label": target_label, **aggregate(sub)})
    return out


def write_report(
    route_rows_all: list[dict[str, object]],
    detail_rows: list[dict[str, object]],
    target_rows: list[dict[str, object]],
    base_stats: dict[str, object],
) -> None:
    safe = [
        row
        for row in route_rows_all
        if int(row["wins"]) == int(row["rows"]) and int(row["nonharm"]) == int(row["rows"])
    ]
    if not safe:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "REPORT.md").write_text("No safe mid-ratio route found.\n", encoding="utf-8")
        return
    best = safe[0]
    best_worst = min(safe, key=lambda row: (float(row["worst_delta"]), float(row["mean_delta"])))
    joint = read_csv(JOINT_DIR / "safe_joint_curvature_top200.csv")[0]
    low_tail = read_csv(BASE_DIR / "safe_routes_top200.csv")[0]
    if str(best["mid_config_id"]) == "base":
        config_line = "- Mid config: `base`; no moderate-tail route improved over the low-tail base.\n\n"
    else:
        config_line = (
            f"- Mid config: `{best['mid_config_id']}` with center `{float(best['center']):g}`, "
            f"width `{float(best['width']):g}`, time_power `{float(best['time_power']):g}`, "
            f"tau `{float(best['gate_tau']):g}`, sign `{best['gate_sign']}`, shrink `{best['shrink_gate']}`.\n\n"
        )
    lines = [
        "# Mid-Ratio Route Audit\n\n",
        "This audit keeps the current low-tail route and switches only WSD-con schedules with "
        "`final_lr / peak_lr = 0.3` to a mid-tail recovery candidate.  The mid-tail coefficient "
        "is still fitted from the cosine residual only; WSD-family losses are used here only for "
        "development ranking.\n\n",
        "## Best Fully Non-Harming Route\n\n",
        f"- Mean / worst: `{fmt_pct2(float(best['mean_delta']))}` / `{fmt_pct2(float(best['worst_delta']))}`.\n",
        f"- Wins/non-harm: `{int(best['wins'])}/{int(best['rows'])}` and `{int(best['nonharm'])}/{int(best['rows'])}`.\n",
        config_line,
        "## Best Worst-Case Route\n\n",
        f"- Mean / worst: `{fmt_pct2(float(best_worst['mean_delta']))}` / "
        f"`{fmt_pct2(float(best_worst['worst_delta']))}`.\n\n",
        "## Comparison\n\n",
        f"- Joint-channel LR-curvature: mean `{fmt_pct2(float(joint['mean_delta']))}`, "
        f"worst `{fmt_pct2(float(joint['worst_delta']))}`.\n",
        f"- Low-tail route base: mean `{fmt_pct2(float(low_tail['mean_delta']))}`, "
        f"worst `{fmt_pct2(float(low_tail['worst_delta']))}`.\n",
        f"- Mid-ratio route: mean `{fmt_pct2(float(best['mean_delta']))}`, "
        f"worst `{fmt_pct2(float(best['worst_delta']))}`.\n",
        f"- Base route recomputed in this script: mean `{fmt_pct2(float(base_stats['mean_delta']))}`, "
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
    selected = [
        row
        for row in detail_rows
        if str(row["mid_config_id"]) == str(best["mid_config_id"]) and row["test_curve"] == MID_TARGET
    ]
    lines += [
        "\n## Routed WSD-con 9e-5 Rows\n\n",
        "| scale | delta | corr_mae | base_mae |\n",
        "|---|---:|---:|---:|\n",
    ]
    for row in selected:
        lines.append(
            f"| {row['scale']}M | {fmt_pct(float(row['delta_pct']))} | "
            f"{float(row['corr_mae']):.6g} | {float(row['base_mae']):.6g} |\n"
        )
    lines += [
        "\n## Reading\n\n",
        "- This is a schedule-ratio route, not a per-scale route: all three scales of WSD-con 9e-5 use the same mid-tail configuration.\n",
        "- A useful improvement here would show that the remaining bottleneck is tied to moderate tail LR rather than to the low-tail case already handled by the previous route.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    _base_cfg, base = base_route_rows()
    mid_configs, mid_details = build_mid_tail_details()
    all_routes: list[dict[str, object]] = []
    all_details: list[dict[str, object]] = []

    base_rows = route_rows(base, mid_details, None)
    base_stats = aggregate(base_rows)
    all_routes.append({"mid_config_id": "base", **base_stats})
    all_details.extend(base_rows)

    for cfg in mid_configs:
        rows = route_rows(base, mid_details, cfg)
        stats = aggregate(rows)
        all_routes.append({**cfg, **stats})
        all_details.extend(rows)

    safe = [
        row
        for row in all_routes
        if int(row["wins"]) == int(row["rows"]) and int(row["nonharm"]) == int(row["rows"])
    ]
    safe_sorted = sorted(safe, key=lambda row: (float(row["mean_delta"]), float(row["worst_delta"])))
    selected_rows = route_rows(base, mid_details, safe_sorted[0] if safe_sorted[0]["mid_config_id"] != "base" else None)
    target_rows = summarize_by_target(selected_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "all_mid_ratio_routes.csv", all_routes)
    write_csv(OUT_DIR / "safe_mid_ratio_routes_top200.csv", safe_sorted[:TOP_LIMIT])
    write_csv(OUT_DIR / "top_safe_details.csv", selected_rows)
    write_csv(OUT_DIR / "best_target_summary.csv", target_rows)
    write_report(safe_sorted[:TOP_LIMIT], selected_rows, target_rows, base_stats)
    print(f"wrote {OUT_DIR / 'all_mid_ratio_routes.csv'}")
    print(f"wrote {OUT_DIR / 'safe_mid_ratio_routes_top200.csv'}")
    print(f"wrote {OUT_DIR / 'top_safe_details.csv'}")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
