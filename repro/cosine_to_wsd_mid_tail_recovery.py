#!/usr/bin/env python3
"""Mid-tail recovery feature audit for cosine-to-WSD prediction.

The current worst row is WSD-con 9e-5.  Its residual suggests the step response
persists too long when the post-drop LR is moderate.  This audit adds one
schedule-only recovery feature:

    recovery(t) = phi_step(t) * exp(-0.5 ((eta/eta_peak - center)/width)^2)
                  * (t / T)^power

with a non-positive or signed coefficient fitted from cosine residuals only.
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
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

from cosine_to_wsd_adaptive_fit_window import channel_for_curve  # noqa: E402
from cosine_to_wsd_curvature_correction import curvature_feature, fmt_pct, fmt_pct2  # noqa: E402
from cosine_to_wsd_response_search import (  # noqa: E402
    TARGETS,
    TARGET_RETENTION_FLOOR,
    TRAIN_CURVE,
    build_cache,
    stime_feature,
    target_retention,
)
from cosine_to_wsd_tail_gated_response import (  # noqa: E402
    CURVATURE,
    SMOOTH,
    STEP,
    fit_smooth,
    fit_step_with_gate,
)
from reproduce_cosine_to_wsd import PEAK_LR, SCALES, metrics  # noqa: E402


OUT_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "mid_tail_recovery"
JOINT_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "joint_curvature"

CENTERS = [0.2, 0.3, 0.4, 0.5]
WIDTHS = [0.05, 0.1, 0.15, 0.25]
TIME_POWERS = [0.0, 1.0, 2.0]
GATE_TAUS = [0.001, 0.003, 0.01, 0.03, 0.1]
GATE_SIGNS = ["nonpos", "signed"]
SHRINK_GATE = [True, False]
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


def mid_tail_gate(curve, phi: np.ndarray, center: float, width: float, time_power: float) -> np.ndarray:
    eta_ratio = np.clip(curve.lrs[curve.step].astype(np.float64) / PEAK_LR, 0.0, None)
    lr_gate = np.exp(-0.5 * np.square((eta_ratio - center) / max(width, 1e-12)))
    if time_power == 0.0:
        time_gate = 1.0
    else:
        time_gate = np.power(curve.step.astype(np.float64) / max(float(curve.step.max()), 1.0), time_power)
    return phi * lr_gate * time_gate


def score_config(
    cache,
    primary_cache,
    curvature_cache,
    gate_cache,
    *,
    center: float,
    width: float,
    time_power: float,
    gate_tau: float,
    gate_sign: str,
    shrink_gate: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scale in SCALES:
        source = cache[(scale, TRAIN_CURVE)]
        smooth_phi = primary_cache[(scale, TRAIN_CURVE, "smooth")]
        smooth_coef, smooth_fit = fit_smooth(source, smooth_phi)
        step_phi = primary_cache[(scale, TRAIN_CURVE, "step")]
        step_curv = curvature_cache[(scale, TRAIN_CURVE)]
        step_gate = gate_cache[(scale, TRAIN_CURVE, center, width, time_power)]
        step_coef, step_fit = fit_step_with_gate(
            source,
            step_phi,
            step_curv,
            step_gate,
            gate_tau=gate_tau,
            gate_sign=gate_sign,
            shrink_gate=shrink_gate,
        )
        for target_curve, target_label in TARGETS:
            target = cache[(scale, target_curve)]
            channel = channel_for_curve(target.curve)
            if channel == "smooth":
                phi = primary_cache[(scale, target_curve, "smooth")]
                retention = target_retention(
                    phi,
                    nuisance_lambda=float(SMOOTH["nuisance_lambda"]),
                    max_mode=int(SMOOTH["max_mode"]),
                )
                factor = 1.0 if retention >= TARGET_RETENTION_FLOOR else 0.0
                shape = smooth_coef * phi
                pred = target.baseline + factor * shape
                primary_coef = smooth_coef
                curvature_coef = 0.0
                gate_coef = 0.0
            else:
                phi = primary_cache[(scale, target_curve, "step")]
                curv = curvature_cache[(scale, target_curve)]
                gate = gate_cache[(scale, target_curve, center, width, time_power)]
                shape = step_coef[0] * phi + step_coef[1] * curv + step_coef[2] * gate
                retention = (
                    target_retention(
                        shape,
                        nuisance_lambda=float(STEP["nuisance_lambda"]),
                        max_mode=int(STEP["max_mode"]),
                    )
                    if float(np.dot(shape, shape)) > 1e-18
                    else 0.0
                )
                factor = 1.0 if retention >= TARGET_RETENTION_FLOOR else 0.0
                pred = target.baseline + factor * shape
                primary_coef = float(step_coef[0])
                curvature_coef = float(step_coef[1])
                gate_coef = float(step_coef[2])

            corr_mae = metrics(target.curve.loss, pred)["mae"]
            rows.append(
                {
                    "scale": scale,
                    "train_curve": TRAIN_CURVE,
                    "test_curve": target_curve,
                    "test_label": target_label,
                    "channel": channel,
                    "target_retention": retention,
                    "base_mae": target.base_mae,
                    "corr_mae": corr_mae,
                    "delta_pct": 100.0 * (corr_mae / target.base_mae - 1.0),
                    "win": int(corr_mae < target.base_mae),
                    "primary_coef": primary_coef,
                    "curvature_coef": curvature_coef,
                    "gate_coef": gate_coef,
                    "center": center,
                    "width": width,
                    "time_power": time_power,
                    "gate_tau": gate_tau,
                    "gate_sign": gate_sign,
                    "shrink_gate": int(shrink_gate),
                    "smooth_raw_primary": smooth_fit["raw_primary"] if channel == "smooth" else "",
                    "step_primary_retention": step_fit["primary_retention"] if channel == "step" else "",
                }
            )
    return rows


def run_search() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    cache = build_cache()
    primary_cache = {
        (scale, curve_name, "smooth"): stime_feature(cache[(scale, curve_name)].curve, float(SMOOTH["response_lambda"]))
        for scale in SCALES
        for curve_name, _ in [(TRAIN_CURVE, "Cosine")] + TARGETS
    }
    primary_cache.update(
        {
            (scale, curve_name, "step"): stime_feature(cache[(scale, curve_name)].curve, float(STEP["response_lambda"]))
            for scale in SCALES
            for curve_name, _ in [(TRAIN_CURVE, "Cosine")] + TARGETS
        }
    )
    curvature_cache = {
        (scale, curve_name): curvature_feature(
            cache[(scale, curve_name)].curve,
            float(CURVATURE["curvature_lambda"]),
            str(CURVATURE["curvature_mode"]),
        )
        for scale in SCALES
        for curve_name, _ in [(TRAIN_CURVE, "Cosine")] + TARGETS
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
        for curve_name, _ in [(TRAIN_CURVE, "Cosine")] + TARGETS
        for center in CENTERS
        for width in WIDTHS
        for time_power in TIME_POWERS
    }
    config_rows: list[dict[str, object]] = []
    safe_detail_rows: list[dict[str, object]] = []
    config_id = 0
    for center in CENTERS:
        for width in WIDTHS:
            for time_power in TIME_POWERS:
                for gate_tau in GATE_TAUS:
                    for gate_sign in GATE_SIGNS:
                        for shrink_gate in SHRINK_GATE:
                            details = score_config(
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
                            summary = aggregate(details)
                            step_rows = [row for row in details if row["channel"] == "step"]
                            row = {
                                "config_id": config_id,
                                "center": center,
                                "width": width,
                                "time_power": time_power,
                                "gate_tau": gate_tau,
                                "gate_sign": gate_sign,
                                "shrink_gate": int(shrink_gate),
                                **summary,
                                "mean_step_primary_coef": float(np.mean([float(row["primary_coef"]) for row in step_rows])),
                                "mean_step_curvature_coef": float(np.mean([float(row["curvature_coef"]) for row in step_rows])),
                                "mean_step_gate_coef": float(np.mean([float(row["gate_coef"]) for row in step_rows])),
                            }
                            config_rows.append(row)
                            if summary["wins"] == summary["rows"] and summary["nonharm"] == summary["rows"]:
                                for detail in details:
                                    safe_detail_rows.append({"config_id": config_id, **detail})
                            config_id += 1

    safe_rows = [
        row for row in config_rows if int(row["wins"]) == int(row["rows"]) and int(row["nonharm"]) == int(row["rows"])
    ]
    safe_sorted = sorted(safe_rows, key=lambda row: (float(row["mean_delta"]), float(row["worst_delta"])))
    top_ids = {int(row["config_id"]) for row in safe_sorted[:TOP_LIMIT]}
    top_details = [row for row in safe_detail_rows if int(row["config_id"]) in top_ids]
    return config_rows, safe_sorted[:TOP_LIMIT], top_details


def summarize_by_target(detail_rows: list[dict[str, object]], config_id: int) -> list[dict[str, object]]:
    selected = [row for row in detail_rows if int(row["config_id"]) == config_id]
    rows: list[dict[str, object]] = []
    for target_curve, target_label in TARGETS:
        sub = [row for row in selected if row["test_curve"] == target_curve]
        rows.append({"test_curve": target_curve, "test_label": target_label, **aggregate(sub)})
    return rows


def write_report(safe_rows: list[dict[str, object]], target_rows: list[dict[str, object]]) -> None:
    if not safe_rows:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "REPORT.md").write_text("No non-harming mid-tail recovery candidate found.\n", encoding="utf-8")
        return
    best = safe_rows[0]
    best_worst = min(safe_rows, key=lambda row: (float(row["worst_delta"]), float(row["mean_delta"])))
    joint = read_csv(JOINT_DIR / "safe_joint_curvature_top200.csv")[0]
    lines = [
        "# Mid-Tail Recovery Audit\n\n",
        "This audit adds a Gaussian LR-level gate to the step channel, intended to recover over-persistent lag "
        "on moderate final-LR WSD-con schedules.\n\n",
        "## Best Fully Non-Harming Candidate\n\n",
        f"- Mean / worst: `{fmt_pct2(float(best['mean_delta']))}` / `{fmt_pct2(float(best['worst_delta']))}`.\n",
        f"- Wins/non-harm: `{int(best['wins'])}/{int(best['rows'])}` and `{int(best['nonharm'])}/{int(best['rows'])}`.\n",
        f"- Gate: `center={float(best['center']):g}`, `width={float(best['width']):g}`, "
        f"`time_power={float(best['time_power']):g}`, `tau={float(best['gate_tau']):g}`, "
        f"`sign={best['gate_sign']}`, `shrink={int(best['shrink_gate'])}`.\n",
        f"- Mean step coefficients: primary `{float(best['mean_step_primary_coef']):.5f}`, "
        f"curvature `{float(best['mean_step_curvature_coef']):.5f}`, "
        f"gate `{float(best['mean_step_gate_coef']):.5f}`.\n\n",
        "## Best Worst-Case Candidate\n\n",
        f"- Mean / worst: `{fmt_pct2(float(best_worst['mean_delta']))}` / "
        f"`{fmt_pct2(float(best_worst['worst_delta']))}`.\n\n",
        "## Comparison\n\n",
        f"- Joint-channel LR-curvature: mean `{fmt_pct2(float(joint['mean_delta']))}`, "
        f"worst `{fmt_pct2(float(joint['worst_delta']))}`.\n",
        f"- Mid-tail recovery: mean `{fmt_pct2(float(best['mean_delta']))}`, "
        f"worst `{fmt_pct2(float(best['worst_delta']))}`.\n\n",
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
        "\n## Reading\n\n",
        "- If selected coefficients are zero, the cosine residual does not support this recovery shape.\n",
        "- If it improves mean but worsens worst-case, it may still be useful as a routed low- or mid-tail feature rather than a global step feature.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    config_rows, safe_rows, detail_rows = run_search()
    target_rows = summarize_by_target(detail_rows, int(safe_rows[0]["config_id"])) if safe_rows else []
    write_csv(OUT_DIR / "all_mid_tail_configs.csv", config_rows)
    write_csv(OUT_DIR / "safe_mid_tail_top200.csv", safe_rows)
    write_csv(OUT_DIR / "top_safe_details.csv", detail_rows)
    write_csv(OUT_DIR / "best_target_summary.csv", target_rows)
    write_report(safe_rows, target_rows)
    print(f"wrote {OUT_DIR / 'all_mid_tail_configs.csv'}")
    print(f"wrote {OUT_DIR / 'safe_mid_tail_top200.csv'}")
    print(f"wrote {OUT_DIR / 'top_safe_details.csv'}")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
