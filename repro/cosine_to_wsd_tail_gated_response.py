#!/usr/bin/env python3
"""Tail-gated step response audit for cosine-to-WSD prediction.

The remaining WSD-con error is mostly tail-shaped, and changing a single fixed
step response rate did not beat the joint LR-curvature model.  This audit adds
one interpretable schedule-only feature:

    tail_gate(t) = phi_step(t) * g(eta_t / eta_peak)

with a signed or non-positive coefficient.  The intended role is a catch-up
term: if LR remains high after the drop, the non-adiabatic error should be
erased faster than the first-order drop response predicts.
"""
from __future__ import annotations

import csv
import math
import os
import sys
from collections import defaultdict
from itertools import combinations
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
    dct_basis,
    soft_residualize,
    stime_feature,
    target_retention,
)
from reproduce_cosine_to_wsd import PEAK_LR, SCALES, metrics  # noqa: E402


OUT_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "tail_gated_response"
JOINT_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "joint_curvature"

SMOOTH = {
    "fit_start_step": 12000,
    "response_lambda": 4.0,
    "nuisance_lambda": 0.05,
    "max_mode": 8,
    "ridge_tau": 0.05,
    "retention_power": 0.25,
    "rho": 0.2,
}
STEP = {
    "fit_start_step": 3000,
    "response_lambda": 20.0,
    "nuisance_lambda": 0.01,
    "max_mode": 8,
    "ridge_tau": 0.05,
    "retention_power": 0.0,
    "rho": 0.35,
}
CURVATURE = {
    "curvature_lambda": 10.0,
    "curvature_mode": "signed_d2_lr",
    "curvature_tau": 0.003,
}

GATE_MODES = ["lr", "sqrt_lr", "high_lr", "one_minus_lr", "late_lr"]
GATE_TAUS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
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


def fit_smooth(source, phi: np.ndarray) -> tuple[float, dict[str, float]]:
    mask = source.curve.step >= int(SMOOTH["fit_start_step"])
    x = phi[mask]
    y = source.residual[mask]
    q = dct_basis(len(x), int(SMOOTH["max_mode"]))
    x_o = soft_residualize(x, q, float(SMOOTH["nuisance_lambda"]))
    y_o = soft_residualize(y, q, float(SMOOTH["nuisance_lambda"]))
    l2 = float(np.dot(x_o, x_o))
    full_l2 = float(np.dot(x, x))
    dot = float(np.dot(x_o, y_o))
    raw = max(0.0, dot / max(l2 + float(SMOOTH["ridge_tau"]) ** 2, 1e-18))
    retention = l2 / max(full_l2, 1e-18)
    shrink = 1.0 / (1.0 + float(SMOOTH["rho"]))
    coef = shrink * (max(retention, 0.0) ** float(SMOOTH["retention_power"])) * raw
    return coef, {"raw_primary": raw, "source_retention": retention}


def gate_feature(curve, phi: np.ndarray, mode: str) -> np.ndarray:
    eta_ratio = curve.lrs[curve.step].astype(np.float64) / PEAK_LR
    eta_ratio = np.clip(eta_ratio, 0.0, None)
    if mode == "lr":
        gate = eta_ratio
    elif mode == "sqrt_lr":
        gate = np.sqrt(eta_ratio)
    elif mode == "high_lr":
        gate = np.maximum(eta_ratio - 0.1, 0.0)
    elif mode == "one_minus_lr":
        gate = 1.0 - eta_ratio
    elif mode == "late_lr":
        step_norm = curve.step.astype(np.float64) / max(float(curve.step.max()), 1.0)
        gate = eta_ratio * step_norm
    else:
        raise ValueError(f"unknown gate mode: {mode}")
    return phi * gate


def constraints_for(gate_sign: str) -> list[tuple[int, int]]:
    # (sign, index): sign=+1 means coef>=0; sign=-1 means coef<=0.
    constraints = [(1, 0), (1, 1)]
    if gate_sign == "nonpos":
        constraints.append((-1, 2))
    elif gate_sign == "signed":
        pass
    else:
        raise ValueError(gate_sign)
    return constraints


def feasible(coef: np.ndarray, gate_sign: str) -> bool:
    if coef[0] < -1e-12 or coef[1] < -1e-12:
        return False
    if gate_sign == "nonpos" and coef[2] > 1e-12:
        return False
    return True


def project_signs(coef: np.ndarray, gate_sign: str) -> np.ndarray:
    out = coef.copy()
    out[0] = max(0.0, out[0])
    out[1] = max(0.0, out[1])
    if gate_sign == "nonpos":
        out[2] = min(0.0, out[2])
    return out


def constrained_ridge(x: np.ndarray, y: np.ndarray, ridge: np.ndarray, gate_sign: str) -> np.ndarray:
    n = x.shape[1]
    candidates: list[np.ndarray] = [np.zeros(n, dtype=np.float64)]
    all_idx = range(n)
    # Enumerate free subsets. Constrained inactive variables are set to zero.
    for k in range(1, n + 1):
        for subset in combinations(all_idx, k):
            subset = tuple(subset)
            xs = x[:, subset]
            rs = ridge[np.ix_(subset, subset)]
            rhs = xs.T @ y
            gram = xs.T @ xs + rs
            try:
                sol = np.linalg.solve(gram, rhs)
            except np.linalg.LinAlgError:
                continue
            coef = np.zeros(n, dtype=np.float64)
            coef[list(subset)] = sol
            if feasible(coef, gate_sign):
                candidates.append(coef)
            candidates.append(project_signs(coef, gate_sign))

    def objective(coef: np.ndarray) -> float:
        diff = x @ coef - y
        return float(np.dot(diff, diff) + coef @ ridge @ coef)

    valid = [coef for coef in candidates if feasible(coef, gate_sign)]
    return min(valid, key=objective)


def fit_step_with_gate(
    source,
    phi: np.ndarray,
    curv: np.ndarray,
    gate: np.ndarray,
    *,
    gate_tau: float,
    gate_sign: str,
    shrink_gate: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    mask = source.curve.step >= int(STEP["fit_start_step"])
    x_raw = np.column_stack([phi[mask], curv[mask], gate[mask]])
    y = source.residual[mask]
    q = dct_basis(len(y), int(STEP["max_mode"]))
    x = np.column_stack([soft_residualize(x_raw[:, j], q, float(STEP["nuisance_lambda"])) for j in range(3)])
    y_o = soft_residualize(y, q, float(STEP["nuisance_lambda"]))
    ridge = np.diag([float(STEP["ridge_tau"]) ** 2, float(CURVATURE["curvature_tau"]) ** 2, gate_tau * gate_tau])
    coef = constrained_ridge(x, y_o, ridge, gate_sign)
    primary_retention = float(np.dot(x[:, 0], x[:, 0]) / max(np.dot(x_raw[:, 0], x_raw[:, 0]), 1e-18))
    primary_scale = (
        (1.0 / (1.0 + float(STEP["rho"])))
        * (max(primary_retention, 0.0) ** float(STEP["retention_power"]))
    )
    if shrink_gate:
        coef = coef * primary_scale
    else:
        coef = np.array([coef[0] * primary_scale, coef[1] * primary_scale, coef[2]], dtype=np.float64)
    return coef, {"primary_retention": primary_retention, "primary_scale": primary_scale}


def score_config(
    cache,
    primary_cache,
    curvature_cache,
    gate_cache,
    *,
    gate_mode: str,
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
        step_gate = gate_cache[(scale, TRAIN_CURVE, gate_mode)]
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
                gate = gate_cache[(scale, target_curve, gate_mode)]
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
                    "gate_mode": gate_mode,
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
        (scale, curve_name, mode): gate_feature(cache[(scale, curve_name)].curve, primary_cache[(scale, curve_name, "step")], mode)
        for scale in SCALES
        for curve_name, _ in [(TRAIN_CURVE, "Cosine")] + TARGETS
        for mode in GATE_MODES
    }
    config_rows: list[dict[str, object]] = []
    safe_detail_rows: list[dict[str, object]] = []
    config_id = 0
    for gate_mode in GATE_MODES:
        for gate_tau in GATE_TAUS:
            for gate_sign in GATE_SIGNS:
                for shrink_gate in SHRINK_GATE:
                    details = score_config(
                        cache,
                        primary_cache,
                        curvature_cache,
                        gate_cache,
                        gate_mode=gate_mode,
                        gate_tau=gate_tau,
                        gate_sign=gate_sign,
                        shrink_gate=shrink_gate,
                    )
                    summary = aggregate(details)
                    step_rows = [row for row in details if row["channel"] == "step"]
                    row = {
                        "config_id": config_id,
                        "gate_mode": gate_mode,
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


def split_defs(targets: set[str]) -> list[dict[str, object]]:
    sharp_linear = {"wsd_20000_24000.csv", "wsdld_20000_24000.csv"}
    wsdcon = {"wsdcon_3.csv", "wsdcon_9.csv", "wsdcon_18.csv"}
    splits: list[dict[str, object]] = [
        {
            "split": "dev_sharp_linear__test_wsdcon",
            "dev_targets": sharp_linear,
            "test_targets": wsdcon,
            "dev_scales": None,
            "test_scales": None,
        },
        {
            "split": "dev_wsdcon__test_sharp_linear",
            "dev_targets": wsdcon,
            "test_targets": sharp_linear,
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
                "gate_mode": cfg["gate_mode"],
                "gate_tau": cfg["gate_tau"],
                "gate_sign": cfg["gate_sign"],
                "shrink_gate": cfg["shrink_gate"],
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
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "REPORT.md").write_text("No non-harming tail-gated response candidate found.\n", encoding="utf-8")
        return
    best = safe_rows[0]
    best_worst = min(safe_rows, key=lambda row: (float(row["worst_delta"]), float(row["mean_delta"])))
    joint = read_csv(JOINT_DIR / "safe_joint_curvature_top200.csv")[0]
    lines = [
        "# Tail-Gated Step Response Audit\n\n",
        "This audit adds one schedule-only tail gate to the current joint LR-curvature step channel:\n\n",
        "```text\n",
        "step correction = a * phi_step + b * psi_curv + c * phi_step * g(eta / eta_peak)\n",
        "```\n\n",
        "The new coefficient is fitted only from `cosine_72000.csv` residuals.\n\n",
        "## Best Fully Non-Harming Candidate\n\n",
        f"- Mean / worst: `{fmt_pct2(float(best['mean_delta']))}` / `{fmt_pct2(float(best['worst_delta']))}`.\n",
        f"- Wins/non-harm: `{int(best['wins'])}/{int(best['rows'])}` and `{int(best['nonharm'])}/{int(best['rows'])}`.\n",
        f"- Gate: `mode={best['gate_mode']}`, `tau={float(best['gate_tau']):g}`, "
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
        f"- Tail-gated response: mean `{fmt_pct2(float(best['mean_delta']))}`, "
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
        "\n## Top-Safe Holdout Check\n\n",
        "| split | selected config | dev mean | dev worst | test mean | test worst | test wins |\n",
        "|---|---|---:|---:|---:|---:|---:|\n",
    ]
    for row in holdout_rows:
        if row["selection_status"] != "selected":
            lines.append(f"| {row['split']} | none | | | | | |\n")
            continue
        cfg = (
            f"mode={row['gate_mode']}, tau={float(row['gate_tau']):g}, "
            f"sign={row['gate_sign']}, shrink={int(row['shrink_gate'])}"
        )
        lines.append(
            f"| {row['split']} | `{cfg}` | "
            f"{fmt_pct(float(row['dev_mean_delta']))} | {fmt_pct(float(row['dev_worst_delta']))} | "
            f"{fmt_pct(float(row['test_mean_delta']))} | {fmt_pct(float(row['test_worst_delta']))} | "
            f"{int(row['test_wins'])}/{int(row['test_rows'])} |\n"
        )
    lines += [
        "\n## Reading\n\n",
        "- A negative high-LR gate would support the catch-up interpretation: higher post-drop LR erases the lag faster.\n",
        "- If selected gates are zero or do not beat the joint-channel model, the remaining error is not explained by a simple LR-level-gated response.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    config_rows, safe_rows, detail_rows = run_search()
    write_csv(OUT_DIR / "all_tail_gate_configs.csv", config_rows)
    write_csv(OUT_DIR / "safe_tail_gate_top200.csv", safe_rows)
    write_csv(OUT_DIR / "top_safe_details.csv", detail_rows)
    target_rows = summarize_by_target(detail_rows, int(safe_rows[0]["config_id"])) if safe_rows else []
    holdout_rows = top_holdout(detail_rows) if detail_rows else []
    write_csv(OUT_DIR / "best_target_summary.csv", target_rows)
    write_csv(OUT_DIR / "top_holdout_summary.csv", holdout_rows)
    write_report(safe_rows, target_rows, holdout_rows)
    print(f"wrote {OUT_DIR / 'all_tail_gate_configs.csv'}")
    print(f"wrote {OUT_DIR / 'safe_tail_gate_top200.csv'}")
    print(f"wrote {OUT_DIR / 'top_safe_details.csv'}")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
