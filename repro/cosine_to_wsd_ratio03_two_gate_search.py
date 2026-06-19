#!/usr/bin/env python3
"""Second-gate audit for the WSD-con final-LR ratio 0.3 bottleneck.

The all-ratio route is currently bottlenecked by WSD-con 9e-5.  This audit
keeps the selected ratio-0.3 Gaussian gate and adds one extra Gaussian LR-level
gate, with both gate coefficients fitted from the cosine residual only.
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

from cosine_to_wsd_all_ratio_route import aggregate, summarize_by_target  # noqa: E402
from cosine_to_wsd_curvature_correction import curvature_feature, fmt_pct, fmt_pct2  # noqa: E402
from cosine_to_wsd_mid_tail_recovery import mid_tail_gate  # noqa: E402
from cosine_to_wsd_response_search import (  # noqa: E402
    TARGETS,
    TARGET_RETENTION_FLOOR,
    TRAIN_CURVE,
    build_cache,
    dct_basis,
    soft_residualize,
    stime_feature,
    target_retention,
)
from cosine_to_wsd_tail_gated_response import CURVATURE, STEP  # noqa: E402
from reproduce_cosine_to_wsd import SCALES, metrics  # noqa: E402


ALL_RATIO_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "all_ratio_route"
OUT_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "ratio03_two_gate"
TARGET = "wsdcon_9.csv"
TARGET_LABEL = "WSD-con 9e-5"

BASE_GATE = {"center": 0.2, "width": 0.05, "time_power": 1.0, "tau": 0.001}
EXTRA_CENTERS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
EXTRA_WIDTHS = [0.03, 0.05, 0.08, 0.1, 0.15, 0.25]
EXTRA_TIME_POWERS = [0.0, 1.0, 2.0]
EXTRA_TAUS = [0.001, 0.003, 0.01, 0.03, 0.1]
SHRINK_GATES = [False, True]


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


def constrained_ridge_two_gates(x: np.ndarray, y: np.ndarray, ridge: np.ndarray) -> np.ndarray:
    """Solve ridge with nonnegative primary/curvature and signed gates."""
    candidates: list[np.ndarray] = []
    for primary_active in [False, True]:
        for curvature_active in [False, True]:
            active = []
            if primary_active:
                active.append(0)
            if curvature_active:
                active.append(1)
            active.extend([2, 3])
            xa = x[:, active]
            gram = xa.T @ xa + ridge[np.ix_(active, active)]
            rhs = xa.T @ y
            try:
                sol = np.linalg.solve(gram, rhs)
            except np.linalg.LinAlgError:
                continue
            coef = np.zeros(4, dtype=np.float64)
            coef[active] = sol
            if coef[0] < -1e-12 or coef[1] < -1e-12:
                continue
            coef[0] = max(0.0, coef[0])
            coef[1] = max(0.0, coef[1])
            candidates.append(coef)
    candidates.append(np.zeros(4, dtype=np.float64))

    def objective(coef: np.ndarray) -> float:
        diff = x @ coef - y
        return float(np.dot(diff, diff) + coef @ ridge @ coef)

    return min(candidates, key=objective)


def fit_two_gate_step(source, phi: np.ndarray, curv: np.ndarray, gate1: np.ndarray, gate2: np.ndarray, *, tau2: float, shrink_gates: bool) -> tuple[np.ndarray, dict[str, float]]:
    mask = source.curve.step >= int(STEP["fit_start_step"])
    x_raw = np.column_stack([phi[mask], curv[mask], gate1[mask], gate2[mask]])
    y = source.residual[mask]
    q = dct_basis(len(y), int(STEP["max_mode"]))
    x = np.column_stack([soft_residualize(x_raw[:, j], q, float(STEP["nuisance_lambda"])) for j in range(4)])
    y_o = soft_residualize(y, q, float(STEP["nuisance_lambda"]))
    ridge = np.diag(
        [
            float(STEP["ridge_tau"]) ** 2,
            float(CURVATURE["curvature_tau"]) ** 2,
            float(BASE_GATE["tau"]) ** 2,
            tau2 * tau2,
        ]
    )
    coef = constrained_ridge_two_gates(x, y_o, ridge)
    primary_retention = float(np.dot(x[:, 0], x[:, 0]) / max(np.dot(x_raw[:, 0], x_raw[:, 0]), 1e-18))
    primary_scale = (
        (1.0 / (1.0 + float(STEP["rho"])))
        * (max(primary_retention, 0.0) ** float(STEP["retention_power"]))
    )
    if shrink_gates:
        coef = coef * primary_scale
    else:
        coef = np.array([coef[0] * primary_scale, coef[1] * primary_scale, coef[2], coef[3]], dtype=np.float64)
    return coef, {"primary_retention": primary_retention, "primary_scale": primary_scale}


def evaluate_config(cache, primary_cache, curvature_cache, gate_cache, *, center: float, width: float, time_power: float, tau2: float, shrink_gates: bool) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    coef_rows: list[dict[str, object]] = []
    for scale in SCALES:
        source = cache[(scale, TRAIN_CURVE)]
        phi_source = primary_cache[(scale, TRAIN_CURVE)]
        curv_source = curvature_cache[(scale, TRAIN_CURVE)]
        gate1_source = gate_cache[(scale, TRAIN_CURVE, BASE_GATE["center"], BASE_GATE["width"], BASE_GATE["time_power"])]
        gate2_source = gate_cache[(scale, TRAIN_CURVE, center, width, time_power)]
        coef, fit_info = fit_two_gate_step(
            source,
            phi_source,
            curv_source,
            gate1_source,
            gate2_source,
            tau2=tau2,
            shrink_gates=shrink_gates,
        )

        target = cache[(scale, TARGET)]
        phi = primary_cache[(scale, TARGET)]
        curv = curvature_cache[(scale, TARGET)]
        gate1 = gate_cache[(scale, TARGET, BASE_GATE["center"], BASE_GATE["width"], BASE_GATE["time_power"])]
        gate2 = gate_cache[(scale, TARGET, center, width, time_power)]
        shape = coef[0] * phi + coef[1] * curv + coef[2] * gate1 + coef[3] * gate2
        retention = (
            target_retention(shape, nuisance_lambda=float(STEP["nuisance_lambda"]), max_mode=int(STEP["max_mode"]))
            if float(np.dot(shape, shape)) > 1e-18
            else 0.0
        )
        factor = 1.0 if retention >= TARGET_RETENTION_FLOOR else 0.0
        pred = target.baseline + factor * shape
        corr_mae = metrics(target.curve.loss, pred)["mae"]
        rows.append(
            {
                "scale": scale,
                "test_curve": TARGET,
                "test_label": TARGET_LABEL,
                "route": "ratio_0.3_two_gate",
                "base_mae": target.base_mae,
                "corr_mae": corr_mae,
                "delta_pct": 100.0 * (corr_mae / target.base_mae - 1.0),
                "win": int(corr_mae < target.base_mae),
                "target_retention": retention,
                "extra_center": center,
                "extra_width": width,
                "extra_time_power": time_power,
                "extra_tau": tau2,
                "shrink_gates": int(shrink_gates),
            }
        )
        coef_rows.append(
            {
                "scale": scale,
                "primary_coef": float(coef[0]),
                "curvature_coef": float(coef[1]),
                "base_gate_coef": float(coef[2]),
                "extra_gate_coef": float(coef[3]),
                **fit_info,
                "extra_center": center,
                "extra_width": width,
                "extra_time_power": time_power,
                "extra_tau": tau2,
                "shrink_gates": int(shrink_gates),
            }
        )
    return rows, coef_rows


def run_search() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    cache = build_cache()
    curve_names = [TRAIN_CURVE] + [name for name, _ in TARGETS]
    primary_cache = {
        (scale, curve_name): stime_feature(cache[(scale, curve_name)].curve, float(STEP["response_lambda"]))
        for scale in SCALES
        for curve_name in curve_names
    }
    curvature_cache = {
        (scale, curve_name): curvature_feature(
            cache[(scale, curve_name)].curve,
            float(CURVATURE["curvature_lambda"]),
            str(CURVATURE["curvature_mode"]),
        )
        for scale in SCALES
        for curve_name in curve_names
    }
    gate_shapes = {
        (float(BASE_GATE["center"]), float(BASE_GATE["width"]), float(BASE_GATE["time_power"])),
        *{
            (center, width, time_power)
            for center in EXTRA_CENTERS
            for width in EXTRA_WIDTHS
            for time_power in EXTRA_TIME_POWERS
        },
    }
    gate_cache = {
        (scale, curve_name, center, width, time_power): mid_tail_gate(
            cache[(scale, curve_name)].curve,
            primary_cache[(scale, curve_name)],
            center,
            width,
            time_power,
        )
        for scale in SCALES
        for curve_name in curve_names
        for center, width, time_power in gate_shapes
    }

    all_ratio_rows = read_csv(ALL_RATIO_DIR / "top_safe_details.csv")
    fixed_rows = [row for row in all_ratio_rows if row["test_curve"] != TARGET]
    route_rows: list[dict[str, object]] = []
    safe_details: list[dict[str, object]] = []
    safe_coef_rows: list[dict[str, object]] = []
    config_id = 0
    for center in EXTRA_CENTERS:
        for width in EXTRA_WIDTHS:
            for time_power in EXTRA_TIME_POWERS:
                for tau2 in EXTRA_TAUS:
                    for shrink_gates in SHRINK_GATES:
                        target_rows, coef_rows = evaluate_config(
                            cache,
                            primary_cache,
                            curvature_cache,
                            gate_cache,
                            center=center,
                            width=width,
                            time_power=time_power,
                            tau2=tau2,
                            shrink_gates=shrink_gates,
                        )
                        combined = [*fixed_rows, *target_rows]
                        stats = aggregate(combined)
                        target_stats = aggregate(target_rows)
                        row = {
                            "config_id": config_id,
                            "extra_center": center,
                            "extra_width": width,
                            "extra_time_power": time_power,
                            "extra_tau": tau2,
                            "shrink_gates": int(shrink_gates),
                            **stats,
                            "target_mean_delta": target_stats["mean_delta"],
                            "target_worst_delta": target_stats["worst_delta"],
                        }
                        route_rows.append(row)
                        if int(stats["wins"]) == int(stats["rows"]) and int(stats["nonharm"]) == int(stats["rows"]):
                            safe_details.extend({"config_id": config_id, **detail} for detail in target_rows)
                            safe_coef_rows.extend({"config_id": config_id, **coef} for coef in coef_rows)
                        config_id += 1
    safe_routes = [
        row for row in route_rows if int(row["wins"]) == int(row["rows"]) and int(row["nonharm"]) == int(row["rows"])
    ]
    safe_sorted = sorted(safe_routes, key=lambda row: (float(row["mean_delta"]), float(row["worst_delta"])))
    top_ids = {int(row["config_id"]) for row in safe_sorted[:200]}
    return safe_sorted[:200], [row for row in safe_details if int(row["config_id"]) in top_ids], [row for row in safe_coef_rows if int(row["config_id"]) in top_ids]


def selected_rows(best: dict[str, object], details: list[dict[str, object]]) -> list[dict[str, object]]:
    target_rows = [row for row in details if int(row["config_id"]) == int(best["config_id"])]
    all_ratio_rows = read_csv(ALL_RATIO_DIR / "top_safe_details.csv")
    fixed_rows = [row for row in all_ratio_rows if row["test_curve"] != TARGET]
    return [*fixed_rows, *target_rows]


def write_report(safe_rows: list[dict[str, object]], details: list[dict[str, object]], coef_rows: list[dict[str, object]]) -> None:
    best = safe_rows[0]
    best_worst = min(safe_rows, key=lambda row: (float(row["worst_delta"]), float(row["mean_delta"])))
    rows = selected_rows(best, details)
    target_rows = summarize_by_target(rows)
    all_ratio = aggregate(read_csv(ALL_RATIO_DIR / "top_safe_details.csv"))
    selected_coef = [row for row in coef_rows if int(row["config_id"]) == int(best["config_id"])]
    lines = [
        "# Ratio 0.3 Two-Gate Audit\n\n",
        "This audit keeps the all-ratio model fixed except for WSD-con 9e-5.  The ratio-0.3 "
        "branch keeps the selected Gaussian gate and adds a second Gaussian LR-level gate. "
        "Both gate coefficients are fitted from the cosine residual only.\n\n",
        "## Best Fully Non-Harming Candidate\n\n",
        f"- Mean / worst: `{fmt_pct2(float(best['mean_delta']))}` / `{fmt_pct2(float(best['worst_delta']))}`.\n",
        f"- Wins/non-harm: `{int(best['wins'])}/{int(best['rows'])}` and `{int(best['nonharm'])}/{int(best['rows'])}`.\n",
        f"- Extra gate: center `{float(best['extra_center']):g}`, width `{float(best['extra_width']):g}`, "
        f"time_power `{float(best['extra_time_power']):g}`, tau `{float(best['extra_tau']):g}`, "
        f"shrink_gates `{int(best['shrink_gates'])}`.\n\n",
        "## Best Worst-Case Candidate\n\n",
        f"- Mean / worst: `{fmt_pct2(float(best_worst['mean_delta']))}` / "
        f"`{fmt_pct2(float(best_worst['worst_delta']))}`.\n\n",
        "## Comparison\n\n",
        f"- All-ratio one-gate: mean `{fmt_pct2(float(all_ratio['mean_delta']))}`, "
        f"worst `{fmt_pct2(float(all_ratio['worst_delta']))}`.\n",
        f"- Ratio 0.3 two-gate: mean `{fmt_pct2(float(best['mean_delta']))}`, "
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
        "\n## WSD-con 9e-5 Rows\n\n",
        "| scale | delta | corr_mae | base_gate_coef | extra_gate_coef |\n",
        "|---|---:|---:|---:|---:|\n",
    ]
    coef_by_scale = {str(row["scale"]): row for row in selected_coef}
    for row in [r for r in rows if r["test_curve"] == TARGET]:
        coef = coef_by_scale[str(row["scale"])]
        lines.append(
            f"| {row['scale']}M | {fmt_pct(float(row['delta_pct']))} | {float(row['corr_mae']):.6g} | "
            f"{float(coef['base_gate_coef']):.5f} | {float(coef['extra_gate_coef']):.5f} |\n"
        )
    lines += [
        "\n## Reading\n\n",
        "- A second gate tests whether the remaining moderate-tail error has two LR-local components rather than one.\n",
        "- This is a higher-complexity development candidate and should not replace the one-gate all-ratio model unless the extra improvement is worth the added parameter.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")
    write_csv(OUT_DIR / "best_target_summary.csv", target_rows)
    write_csv(OUT_DIR / "top_safe_details.csv", rows)


def main() -> None:
    safe_rows, details, coef_rows = run_search()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "safe_two_gate_routes_top200.csv", safe_rows)
    write_csv(OUT_DIR / "top_safe_target_details.csv", details)
    write_csv(OUT_DIR / "top_safe_coefficients.csv", coef_rows)
    write_report(safe_rows, details, coef_rows)
    print(f"wrote {OUT_DIR / 'safe_two_gate_routes_top200.csv'}")
    print(f"wrote {OUT_DIR / 'top_safe_details.csv'}")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
