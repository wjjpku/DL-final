#!/usr/bin/env python3
"""Joint channel-calibration + LR-curvature search for cosine-to-WSD.

The first LR-curvature audit fixed the smooth/step channel calibration at the
best decoupled-channel setting, then searched one curvature term.  This script
keeps the search constrained by using only the top non-harming decoupled-channel
settings, but lets the curvature term choose the channel calibration jointly.
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

from cosine_to_wsd_adaptive_fit_window import channel_for_curve  # noqa: E402
from cosine_to_wsd_curvature_correction import (  # noqa: E402
    CURVATURE_LAMBDAS,
    CURVATURE_MODES,
    CURVATURE_TAUS,
    SHRINK_CURVATURE,
    SIGNED_CURVATURE_COEF,
    curvature_feature,
    fmt_pct,
    fmt_pct2,
    top_holdout,
)
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
from reproduce_cosine_to_wsd import SCALES, metrics  # noqa: E402


DECOUPLED_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "decoupled_channel"
CURVATURE_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "curvature_correction"
OUT_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "joint_curvature"
PAIR_LIMIT = 200
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


def channel_params(row: dict[str, str], prefix: str) -> dict[str, float]:
    return {
        "fit_start_step": int(row[f"{prefix}_fit_start_step"]),
        "response_lambda": float(row[f"{prefix}_lambda"]),
        "nuisance_lambda": float(row[f"{prefix}_nuisance_lambda"]),
        "max_mode": int(row[f"{prefix}_max_mode"]),
        "ridge_tau": float(row[f"{prefix}_ridge_tau"]),
        "retention_power": float(row[f"{prefix}_retention_power"]),
        "rho": float(row[f"{prefix}_rho"]),
    }


def load_pair_candidates() -> list[dict[str, object]]:
    rows = read_csv(DECOUPLED_DIR / "safe_decoupled_channel_top200.csv")[:PAIR_LIMIT]
    seen: set[tuple[object, ...]] = set()
    out: list[dict[str, object]] = []
    for row in rows:
        smooth = channel_params(row, "smooth")
        step = channel_params(row, "step")
        key = (
            tuple(sorted(smooth.items())),
            tuple(sorted(step.items())),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "source_pair_config_id": int(row["pair_config_id"]),
                "source_decoupled_mean_delta": float(row["mean_delta"]),
                "source_decoupled_worst_delta": float(row["worst_delta"]),
                "smooth": smooth,
                "step": step,
                "smooth_config_id": int(row["smooth_config_id"]),
                "step_config_id": int(row["step_config_id"]),
            }
        )
    return out


def fit_primary(source, phi: np.ndarray, params: dict[str, float]) -> tuple[float, dict[str, float]]:
    mask = source.curve.step >= int(params["fit_start_step"])
    x = phi[mask]
    y = source.residual[mask]
    q = dct_basis(len(x), int(params["max_mode"]))
    x_o = soft_residualize(x, q, float(params["nuisance_lambda"]))
    y_o = soft_residualize(y, q, float(params["nuisance_lambda"]))
    l2 = float(np.dot(x_o, x_o))
    full_l2 = float(np.dot(x, x))
    dot = float(np.dot(x_o, y_o))
    raw = max(0.0, dot / max(l2 + float(params["ridge_tau"]) ** 2, 1e-18))
    retention = l2 / max(full_l2, 1e-18)
    shrink = 1.0 / (1.0 + float(params["rho"]))
    coef = shrink * (max(retention, 0.0) ** float(params["retention_power"])) * raw
    return coef, {
        "raw_primary": raw,
        "source_retention": retention,
        "source_dot": dot,
        "source_l2": l2,
        "source_full_l2": full_l2,
        "shrink": shrink,
    }


def fit_step_with_curvature(
    source,
    phi: np.ndarray,
    psi: np.ndarray,
    params: dict[str, float],
    *,
    curvature_tau: float,
    shrink_curvature: bool,
    signed_curvature_coef: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    mask = source.curve.step >= int(params["fit_start_step"])
    x_raw = np.column_stack([phi[mask], psi[mask]])
    y = source.residual[mask]
    q = dct_basis(len(y), int(params["max_mode"]))
    x = np.column_stack(
        [soft_residualize(x_raw[:, j], q, float(params["nuisance_lambda"])) for j in range(2)]
    )
    y_o = soft_residualize(y, q, float(params["nuisance_lambda"]))
    ridge = np.diag([float(params["ridge_tau"]) ** 2, curvature_tau * curvature_tau])
    gram = x.T @ x + ridge
    rhs = x.T @ y_o

    candidates: list[np.ndarray] = []
    try:
        coef = np.linalg.solve(gram, rhs)
        if coef[0] >= 0.0 and (signed_curvature_coef or coef[1] >= 0.0):
            candidates.append(coef)
    except np.linalg.LinAlgError:
        pass

    primary_only = np.zeros(2, dtype=np.float64)
    primary_only[0] = max(0.0, rhs[0] / max(gram[0, 0], 1e-18))
    candidates.append(primary_only)

    curvature_only = np.zeros(2, dtype=np.float64)
    curvature_only[1] = rhs[1] / max(gram[1, 1], 1e-18)
    if not signed_curvature_coef:
        curvature_only[1] = max(0.0, curvature_only[1])
    candidates.append(curvature_only)
    candidates.append(np.zeros(2, dtype=np.float64))

    def objective(coef: np.ndarray) -> float:
        residual = x @ coef - y_o
        return float(np.dot(residual, residual) + coef @ ridge @ coef)

    coef = min(candidates, key=objective)
    primary_retention = float(np.dot(x[:, 0], x[:, 0]) / max(np.dot(x_raw[:, 0], x_raw[:, 0]), 1e-18))
    primary_scale = (
        (1.0 / (1.0 + float(params["rho"])))
        * (max(primary_retention, 0.0) ** float(params["retention_power"]))
    )
    if shrink_curvature:
        coef = coef * primary_scale
    else:
        coef = np.array([coef[0] * primary_scale, coef[1]], dtype=np.float64)
    return coef, {
        "step_primary_coef": float(coef[0]),
        "step_curvature_coef": float(coef[1]),
        "step_primary_retention": primary_retention,
        "step_primary_shrink": primary_scale,
        "step_curvature_tau": curvature_tau,
        "step_curvature_shrunk": int(shrink_curvature),
        "step_curvature_signed_coef": int(signed_curvature_coef),
    }


def score_config(
    cache,
    primary_cache,
    curvature_cache,
    pair: dict[str, object],
    *,
    curvature_lambda: float,
    curvature_mode: str,
    curvature_tau: float,
    shrink_curvature: bool,
    signed_curvature_coef: bool,
) -> list[dict[str, object]]:
    smooth = pair["smooth"]
    step = pair["step"]
    assert isinstance(smooth, dict) and isinstance(step, dict)
    rows: list[dict[str, object]] = []
    for scale in SCALES:
        source = cache[(scale, TRAIN_CURVE)]
        smooth_phi = primary_cache[(scale, TRAIN_CURVE, float(smooth["response_lambda"]))]
        smooth_coef, smooth_fit = fit_primary(source, smooth_phi, smooth)
        step_phi = primary_cache[(scale, TRAIN_CURVE, float(step["response_lambda"]))]
        step_curv = curvature_cache[(scale, TRAIN_CURVE, curvature_lambda, curvature_mode)]
        step_coef, step_fit = fit_step_with_curvature(
            source,
            step_phi,
            step_curv,
            step,
            curvature_tau=curvature_tau,
            shrink_curvature=shrink_curvature,
            signed_curvature_coef=signed_curvature_coef,
        )

        for target_curve, target_label in TARGETS:
            target = cache[(scale, target_curve)]
            channel = channel_for_curve(target.curve)
            if channel == "smooth":
                phi = primary_cache[(scale, target_curve, float(smooth["response_lambda"]))]
                retention = target_retention(
                    phi,
                    nuisance_lambda=float(smooth["nuisance_lambda"]),
                    max_mode=int(smooth["max_mode"]),
                )
                factor = 1.0 if retention >= TARGET_RETENTION_FLOOR else 0.0
                pred = target.baseline + factor * smooth_coef * phi
                extra = {
                    "primary_coef": smooth_coef,
                    "curvature_coef": 0.0,
                    **smooth_fit,
                }
            else:
                phi = primary_cache[(scale, target_curve, float(step["response_lambda"]))]
                curv = curvature_cache[(scale, target_curve, curvature_lambda, curvature_mode)]
                shape = step_coef[0] * phi + step_coef[1] * curv
                retention = (
                    target_retention(
                        shape,
                        nuisance_lambda=float(step["nuisance_lambda"]),
                        max_mode=int(step["max_mode"]),
                    )
                    if float(np.dot(shape, shape)) > 1e-18
                    else 0.0
                )
                factor = 1.0 if retention >= TARGET_RETENTION_FLOOR else 0.0
                pred = target.baseline + factor * shape
                extra = {
                    "primary_coef": float(step_coef[0]),
                    "curvature_coef": float(step_coef[1]),
                    **step_fit,
                }

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
                    "curvature_lambda": curvature_lambda,
                    "curvature_mode": curvature_mode,
                    "curvature_tau": curvature_tau,
                    "shrink_curvature": int(shrink_curvature),
                    "signed_curvature_coef": int(signed_curvature_coef),
                    "source_pair_config_id": pair["source_pair_config_id"],
                    "smooth_config_id": pair["smooth_config_id"],
                    "step_config_id": pair["step_config_id"],
                    "smooth_fit_start_step": int(smooth["fit_start_step"]),
                    "smooth_lambda": float(smooth["response_lambda"]),
                    "smooth_nuisance_lambda": float(smooth["nuisance_lambda"]),
                    "smooth_max_mode": int(smooth["max_mode"]),
                    "smooth_ridge_tau": float(smooth["ridge_tau"]),
                    "smooth_retention_power": float(smooth["retention_power"]),
                    "smooth_rho": float(smooth["rho"]),
                    "step_fit_start_step": int(step["fit_start_step"]),
                    "step_lambda": float(step["response_lambda"]),
                    "step_nuisance_lambda": float(step["nuisance_lambda"]),
                    "step_max_mode": int(step["max_mode"]),
                    "step_ridge_tau": float(step["ridge_tau"]),
                    "step_retention_power": float(step["retention_power"]),
                    "step_rho": float(step["rho"]),
                    **extra,
                }
            )
    return rows


def run_search() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    pairs = load_pair_candidates()
    cache = build_cache()
    primary_lambdas = sorted(
        {
            float(pair["smooth"]["response_lambda"])
            for pair in pairs
            if isinstance(pair["smooth"], dict)
        }
        | {
            float(pair["step"]["response_lambda"])
            for pair in pairs
            if isinstance(pair["step"], dict)
        }
    )
    primary_cache = {
        (scale, curve_name, response_lambda): stime_feature(cache[(scale, curve_name)].curve, response_lambda)
        for scale in SCALES
        for curve_name, _ in [(TRAIN_CURVE, "Cosine")] + TARGETS
        for response_lambda in primary_lambdas
    }
    curvature_cache = {
        (scale, curve_name, curvature_lambda, curvature_mode): curvature_feature(
            cache[(scale, curve_name)].curve, curvature_lambda, curvature_mode
        )
        for scale in SCALES
        for curve_name, _ in [(TRAIN_CURVE, "Cosine")] + TARGETS
        for curvature_lambda in CURVATURE_LAMBDAS
        for curvature_mode in CURVATURE_MODES
    }

    config_rows: list[dict[str, object]] = []
    safe_detail_rows: list[dict[str, object]] = []
    config_id = 0
    for pair_index, pair in enumerate(pairs):
        for curvature_lambda in CURVATURE_LAMBDAS:
            for curvature_mode in CURVATURE_MODES:
                for curvature_tau in CURVATURE_TAUS:
                    for shrink_curvature in SHRINK_CURVATURE:
                        for signed_curvature_coef in SIGNED_CURVATURE_COEF:
                            details = score_config(
                                cache,
                                primary_cache,
                                curvature_cache,
                                pair,
                                curvature_lambda=curvature_lambda,
                                curvature_mode=curvature_mode,
                                curvature_tau=curvature_tau,
                                shrink_curvature=shrink_curvature,
                                signed_curvature_coef=signed_curvature_coef,
                            )
                            summary = aggregate(details)
                            step_details = [detail for detail in details if detail["channel"] == "step"]
                            row = {
                                "config_id": config_id,
                                "pair_index": pair_index,
                                "source_pair_config_id": pair["source_pair_config_id"],
                                "source_decoupled_mean_delta": pair["source_decoupled_mean_delta"],
                                "source_decoupled_worst_delta": pair["source_decoupled_worst_delta"],
                                "curvature_lambda": curvature_lambda,
                                "curvature_mode": curvature_mode,
                                "curvature_tau": curvature_tau,
                                "shrink_curvature": int(shrink_curvature),
                                "signed_curvature_coef": int(signed_curvature_coef),
                                **summary,
                                "mean_step_curvature_coef": float(
                                    np.mean([float(detail["curvature_coef"]) for detail in step_details])
                                ),
                                "mean_step_primary_coef": float(
                                    np.mean([float(detail["primary_coef"]) for detail in step_details])
                                ),
                                "smooth_fit_start_step": pair["smooth"]["fit_start_step"],
                                "smooth_lambda": pair["smooth"]["response_lambda"],
                                "smooth_nuisance_lambda": pair["smooth"]["nuisance_lambda"],
                                "smooth_max_mode": pair["smooth"]["max_mode"],
                                "smooth_ridge_tau": pair["smooth"]["ridge_tau"],
                                "smooth_retention_power": pair["smooth"]["retention_power"],
                                "smooth_rho": pair["smooth"]["rho"],
                                "step_fit_start_step": pair["step"]["fit_start_step"],
                                "step_lambda": pair["step"]["response_lambda"],
                                "step_nuisance_lambda": pair["step"]["nuisance_lambda"],
                                "step_max_mode": pair["step"]["max_mode"],
                                "step_ridge_tau": pair["step"]["ridge_tau"],
                                "step_retention_power": pair["step"]["retention_power"],
                                "step_rho": pair["step"]["rho"],
                            }
                            config_rows.append(row)
                            if summary["wins"] == summary["rows"] and summary["nonharm"] == summary["rows"]:
                                for detail in details:
                                    safe_detail_rows.append({"config_id": config_id, **detail})
                            config_id += 1

    safe_rows = [
        row
        for row in config_rows
        if int(row["wins"]) == int(row["rows"]) and int(row["nonharm"]) == int(row["rows"])
    ]
    safe_sorted = sorted(
        safe_rows,
        key=lambda row: (
            float(row["mean_delta"]),
            float(row["worst_delta"]),
            int(row["signed_curvature_coef"]),
            -int(row["shrink_curvature"]),
        ),
    )
    top_ids = {int(row["config_id"]) for row in safe_sorted[:TOP_LIMIT]}
    top_details = [row for row in safe_detail_rows if int(row["config_id"]) in top_ids]
    return config_rows, safe_sorted[:TOP_LIMIT], top_details


def summarize_by_target(detail_rows: list[dict[str, object]], config_id: int) -> list[dict[str, object]]:
    selected = [row for row in detail_rows if int(row["config_id"]) == config_id]
    rows: list[dict[str, object]] = []
    for target_curve, target_label in TARGETS:
        sub = [row for row in selected if row["test_curve"] == target_curve]
        if sub:
            rows.append({"test_curve": target_curve, "test_label": target_label, **aggregate(sub)})
    return rows


def write_report(
    safe_rows: list[dict[str, object]],
    target_rows: list[dict[str, object]],
    holdout_rows: list[dict[str, object]],
) -> None:
    if not safe_rows:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "REPORT.md").write_text("No non-harming joint curvature candidate found.\n", encoding="utf-8")
        return

    best = safe_rows[0]
    best_worst = min(safe_rows, key=lambda row: (float(row["worst_delta"]), float(row["mean_delta"])))
    decoupled = read_csv(DECOUPLED_DIR / "safe_decoupled_channel_top200.csv")[0]
    fixed_curvature = read_csv(CURVATURE_DIR / "safe_curvature_top200.csv")[0]
    lines = [
        "# Joint Channel + LR-Curvature Search\n\n",
        "This audit searches LR-curvature terms jointly with the top non-harming decoupled-channel "
        "calibrations. Coefficients are still fit only from the `cosine_72000.csv` residual; "
        "WSD-family losses are used for development ranking and evaluation.\n\n",
        "## Best Fully Non-Harming Candidate\n\n",
        f"- Mean / worst: `{fmt_pct2(float(best['mean_delta']))}` / `{fmt_pct2(float(best['worst_delta']))}`.\n",
        f"- Wins/non-harm: `{int(best['wins'])}/{int(best['rows'])}` and `{int(best['nonharm'])}/{int(best['rows'])}`.\n",
        f"- Source decoupled pair: `{int(best['source_pair_config_id'])}` "
        f"(mean `{fmt_pct2(float(best['source_decoupled_mean_delta']))}`, "
        f"worst `{fmt_pct2(float(best['source_decoupled_worst_delta']))}`).\n",
        f"- Curvature: `lambda2={float(best['curvature_lambda']):g}`, `mode={best['curvature_mode']}`, "
        f"`tau2={float(best['curvature_tau']):g}`, `shrink={int(best['shrink_curvature'])}`, "
        f"`signed={int(best['signed_curvature_coef'])}`.\n",
        f"- Mean step coefficients: primary `{float(best['mean_step_primary_coef']):.5f}`, "
        f"curvature `{float(best['mean_step_curvature_coef']):.5f}`.\n\n",
        "## Channel Calibration\n\n",
        f"- Smooth: `start={int(best['smooth_fit_start_step'])}`, `lambda={float(best['smooth_lambda']):g}`, "
        f"`mu={float(best['smooth_nuisance_lambda']):g}`, `modes={int(best['smooth_max_mode'])}`, "
        f"`tau={float(best['smooth_ridge_tau']):g}`, `p={float(best['smooth_retention_power']):g}`, "
        f"`rho={float(best['smooth_rho']):g}`.\n",
        f"- Step: `start={int(best['step_fit_start_step'])}`, `lambda={float(best['step_lambda']):g}`, "
        f"`mu={float(best['step_nuisance_lambda']):g}`, `modes={int(best['step_max_mode'])}`, "
        f"`tau={float(best['step_ridge_tau']):g}`, `p={float(best['step_retention_power']):g}`, "
        f"`rho={float(best['step_rho']):g}`.\n\n",
        "## Best Worst-Case Candidate\n\n",
        f"- Mean / worst: `{fmt_pct2(float(best_worst['mean_delta']))}` / "
        f"`{fmt_pct2(float(best_worst['worst_delta']))}`.\n",
        f"- Curvature: `lambda2={float(best_worst['curvature_lambda']):g}`, "
        f"`mode={best_worst['curvature_mode']}`, `tau2={float(best_worst['curvature_tau']):g}`, "
        f"`shrink={int(best_worst['shrink_curvature'])}`, "
        f"`signed={int(best_worst['signed_curvature_coef'])}`.\n\n",
        "## Comparison\n\n",
        f"- Decoupled-channel: mean `{fmt_pct2(float(decoupled['mean_delta']))}`, "
        f"worst `{fmt_pct2(float(decoupled['worst_delta']))}`.\n",
        f"- Fixed-channel LR-curvature: mean `{fmt_pct2(float(fixed_curvature['mean_delta']))}`, "
        f"worst `{fmt_pct2(float(fixed_curvature['worst_delta']))}`.\n",
        f"- Joint-channel LR-curvature: mean `{fmt_pct2(float(best['mean_delta']))}`, "
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
            f"lambda2={float(row['curvature_lambda']):g}, mode={row['curvature_mode']}, "
            f"tau2={float(row['curvature_tau']):g}, shrink={int(row['shrink_curvature'])}, "
            f"signed={int(row['signed_curvature_coef'])}"
        )
        lines.append(
            f"| {row['split']} | `{cfg}` | "
            f"{fmt_pct(float(row['dev_mean_delta']))} | {fmt_pct(float(row['dev_worst_delta']))} | "
            f"{fmt_pct(float(row['test_mean_delta']))} | {fmt_pct(float(row['test_worst_delta']))} | "
            f"{int(row['test_wins'])}/{int(row['test_rows'])} |\n"
        )
    lines += [
        "\n## Reading\n\n",
        "- Jointly selecting the channel calibration mainly tests whether the first curvature audit was limited by a fixed step-channel suffix/residualizer.\n",
        "- This remains a development search over the available WSD family. It should be frozen before claiming final generalization.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    config_rows, safe_rows, detail_rows = run_search()
    write_csv(OUT_DIR / "all_joint_curvature_configs.csv", config_rows)
    write_csv(OUT_DIR / "safe_joint_curvature_top200.csv", safe_rows)
    write_csv(OUT_DIR / "top_safe_details.csv", detail_rows)
    target_rows = summarize_by_target(detail_rows, int(safe_rows[0]["config_id"])) if safe_rows else []
    holdout_rows = top_holdout(detail_rows) if detail_rows else []
    write_csv(OUT_DIR / "best_target_summary.csv", target_rows)
    write_csv(OUT_DIR / "top_holdout_summary.csv", holdout_rows)
    write_report(safe_rows, target_rows, holdout_rows)
    print(f"wrote {OUT_DIR / 'all_joint_curvature_configs.csv'}")
    print(f"wrote {OUT_DIR / 'safe_joint_curvature_top200.csv'}")
    print(f"wrote {OUT_DIR / 'top_safe_details.csv'}")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
