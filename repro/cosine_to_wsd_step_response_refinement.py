#!/usr/bin/env python3
"""Step-channel response-shape refinement for cosine-to-WSD prediction.

The current joint LR-curvature model is mostly limited by WSD-con tail rows.
Oracle rescaling shows little headroom from changing only the step amplitude, so
this audit searches the step response shape while keeping the smooth channel
fixed.
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
from cosine_to_wsd_joint_curvature_search import (  # noqa: E402
    aggregate,
    fit_primary,
    fit_step_with_curvature,
)
from cosine_to_wsd_response_search import (  # noqa: E402
    TARGETS,
    TARGET_RETENTION_FLOOR,
    TRAIN_CURVE,
    build_cache,
    stime_feature,
    target_retention,
)
from reproduce_cosine_to_wsd import SCALES, metrics  # noqa: E402


OUT_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "step_response_refinement"
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

STEP_FIT_STARTS = [3000, 5000, 8000, 12000]
STEP_LAMBDAS = [10.0, 14.0, 20.0, 30.0, 50.0, 80.0, 120.0]
STEP_MUS = [0.005, 0.01, 0.015, 0.02, 0.03]
STEP_MODES = [8, 12, 16]
STEP_RHOS = [0.2, 0.35, 0.5, 0.75]
CURVATURE_LAMBDAS = [4.0, 7.0, 10.0, 14.0, 20.0, 30.0, 50.0]
CURVATURE_TAUS = [0.001, 0.003, 0.01, 0.03]
CURVATURE_MODES = ["signed_d2_lr", "diff_drop", "abs_diff_drop"]
SHRINK_CURVATURE = [True]
SIGNED_CURVATURE_COEF = [False]
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


def step_params(
    *,
    fit_start_step: int,
    response_lambda: float,
    nuisance_lambda: float,
    max_mode: int,
    rho: float,
) -> dict[str, float]:
    return {
        "fit_start_step": fit_start_step,
        "response_lambda": response_lambda,
        "nuisance_lambda": nuisance_lambda,
        "max_mode": max_mode,
        "ridge_tau": 0.05,
        "retention_power": 0.0,
        "rho": rho,
    }


def score_config(
    cache,
    primary_cache,
    curvature_cache,
    *,
    step: dict[str, float],
    curvature_lambda: float,
    curvature_mode: str,
    curvature_tau: float,
    shrink_curvature: bool,
    signed_curvature_coef: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scale in SCALES:
        source = cache[(scale, TRAIN_CURVE)]
        smooth_phi = primary_cache[(scale, TRAIN_CURVE, float(SMOOTH["response_lambda"]))]
        smooth_coef, smooth_fit = fit_primary(source, smooth_phi, SMOOTH)
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
                phi = primary_cache[(scale, target_curve, float(SMOOTH["response_lambda"]))]
                retention = target_retention(
                    phi,
                    nuisance_lambda=float(SMOOTH["nuisance_lambda"]),
                    max_mode=int(SMOOTH["max_mode"]),
                )
                factor = 1.0 if retention >= TARGET_RETENTION_FLOOR else 0.0
                pred = target.baseline + factor * smooth_coef * phi
                primary_coef = smooth_coef
                curvature_coef = 0.0
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
                primary_coef = float(step_coef[0])
                curvature_coef = float(step_coef[1])

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
                    "smooth_fit_start_step": int(SMOOTH["fit_start_step"]),
                    "smooth_lambda": float(SMOOTH["response_lambda"]),
                    "smooth_nuisance_lambda": float(SMOOTH["nuisance_lambda"]),
                    "smooth_max_mode": int(SMOOTH["max_mode"]),
                    "smooth_rho": float(SMOOTH["rho"]),
                    "step_fit_start_step": int(step["fit_start_step"]),
                    "step_lambda": float(step["response_lambda"]),
                    "step_nuisance_lambda": float(step["nuisance_lambda"]),
                    "step_max_mode": int(step["max_mode"]),
                    "step_rho": float(step["rho"]),
                    "curvature_lambda": curvature_lambda,
                    "curvature_mode": curvature_mode,
                    "curvature_tau": curvature_tau,
                    "shrink_curvature": int(shrink_curvature),
                    "signed_curvature_coef": int(signed_curvature_coef),
                    "smooth_raw_primary": smooth_fit["raw_primary"] if channel == "smooth" else "",
                    "step_primary_retention": step_fit["step_primary_retention"] if channel == "step" else "",
                }
            )
    return rows


def run_search() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    cache = build_cache()
    primary_lambdas = sorted({float(SMOOTH["response_lambda"])} | set(STEP_LAMBDAS))
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
    for fit_start_step in STEP_FIT_STARTS:
        for response_lambda in STEP_LAMBDAS:
            for nuisance_lambda in STEP_MUS:
                for max_mode in STEP_MODES:
                    for rho in STEP_RHOS:
                        step = step_params(
                            fit_start_step=fit_start_step,
                            response_lambda=response_lambda,
                            nuisance_lambda=nuisance_lambda,
                            max_mode=max_mode,
                            rho=rho,
                        )
                        for curvature_lambda in CURVATURE_LAMBDAS:
                            for curvature_mode in CURVATURE_MODES:
                                for curvature_tau in CURVATURE_TAUS:
                                    for shrink_curvature in SHRINK_CURVATURE:
                                        for signed_curvature_coef in SIGNED_CURVATURE_COEF:
                                            details = score_config(
                                                cache,
                                                primary_cache,
                                                curvature_cache,
                                                step=step,
                                                curvature_lambda=curvature_lambda,
                                                curvature_mode=curvature_mode,
                                                curvature_tau=curvature_tau,
                                                shrink_curvature=shrink_curvature,
                                                signed_curvature_coef=signed_curvature_coef,
                                            )
                                            summary = aggregate(details)
                                            step_rows = [row for row in details if row["channel"] == "step"]
                                            row = {
                                                "config_id": config_id,
                                                "step_fit_start_step": fit_start_step,
                                                "step_lambda": response_lambda,
                                                "step_nuisance_lambda": nuisance_lambda,
                                                "step_max_mode": max_mode,
                                                "step_rho": rho,
                                                "curvature_lambda": curvature_lambda,
                                                "curvature_mode": curvature_mode,
                                                "curvature_tau": curvature_tau,
                                                "shrink_curvature": int(shrink_curvature),
                                                "signed_curvature_coef": int(signed_curvature_coef),
                                                **summary,
                                                "mean_step_primary_coef": float(
                                                    np.mean([float(row["primary_coef"]) for row in step_rows])
                                                ),
                                                "mean_step_curvature_coef": float(
                                                    np.mean([float(row["curvature_coef"]) for row in step_rows])
                                                ),
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
                "step_lambda": cfg["step_lambda"],
                "step_nuisance_lambda": cfg["step_nuisance_lambda"],
                "step_max_mode": cfg["step_max_mode"],
                "step_rho": cfg["step_rho"],
                "curvature_lambda": cfg["curvature_lambda"],
                "curvature_mode": cfg["curvature_mode"],
                "curvature_tau": cfg["curvature_tau"],
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
        (OUT_DIR / "REPORT.md").write_text("No non-harming step response candidate found.\n", encoding="utf-8")
        return
    best = safe_rows[0]
    best_worst = min(safe_rows, key=lambda row: (float(row["worst_delta"]), float(row["mean_delta"])))
    joint = read_csv(JOINT_DIR / "safe_joint_curvature_top200.csv")[0]
    lines = [
        "# Step-Response Refinement Audit\n\n",
        "This audit keeps the smooth channel fixed and searches only step-channel response shape parameters. "
        "The goal is to reduce the WSD-con tail rows left by the joint LR-curvature model.\n\n",
        "## Best Fully Non-Harming Candidate\n\n",
        f"- Mean / worst: `{fmt_pct2(float(best['mean_delta']))}` / `{fmt_pct2(float(best['worst_delta']))}`.\n",
        f"- Wins/non-harm: `{int(best['wins'])}/{int(best['rows'])}` and `{int(best['nonharm'])}/{int(best['rows'])}`.\n",
        f"- Step: `start={int(best['step_fit_start_step'])}`, `lambda={float(best['step_lambda']):g}`, "
        f"`mu={float(best['step_nuisance_lambda']):g}`, `modes={int(best['step_max_mode'])}`, "
        f"`rho={float(best['step_rho']):g}`.\n",
        f"- Curvature: `lambda2={float(best['curvature_lambda']):g}`, `mode={best['curvature_mode']}`, "
        f"`tau2={float(best['curvature_tau']):g}`.\n",
        f"- Mean step coefficients: primary `{float(best['mean_step_primary_coef']):.5f}`, "
        f"curvature `{float(best['mean_step_curvature_coef']):.5f}`.\n\n",
        "## Best Worst-Case Candidate\n\n",
        f"- Mean / worst: `{fmt_pct2(float(best_worst['mean_delta']))}` / "
        f"`{fmt_pct2(float(best_worst['worst_delta']))}`.\n\n",
        "## Comparison\n\n",
        f"- Joint-channel LR-curvature: mean `{fmt_pct2(float(joint['mean_delta']))}`, "
        f"worst `{fmt_pct2(float(joint['worst_delta']))}`.\n",
        f"- Step-response refinement: mean `{fmt_pct2(float(best['mean_delta']))}`, "
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
            f"step_lambda={float(row['step_lambda']):g}, mu={float(row['step_nuisance_lambda']):g}, "
            f"modes={int(row['step_max_mode'])}, rho={float(row['step_rho']):g}, "
            f"lambda2={float(row['curvature_lambda']):g}, mode={row['curvature_mode']}, "
            f"tau2={float(row['curvature_tau']):g}"
        )
        lines.append(
            f"| {row['split']} | `{cfg}` | "
            f"{fmt_pct(float(row['dev_mean_delta']))} | {fmt_pct(float(row['dev_worst_delta']))} | "
            f"{fmt_pct(float(row['test_mean_delta']))} | {fmt_pct(float(row['test_worst_delta']))} | "
            f"{int(row['test_wins'])}/{int(row['test_rows'])} |\n"
        )
    lines += [
        "\n## Reading\n\n",
        "- If this search does not beat the joint-channel model, the remaining WSD-con error is probably not recoverable by a single fixed step response rate.\n",
        "- This is still a development search over WSD-family evaluation, not a frozen final protocol.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    config_rows, safe_rows, detail_rows = run_search()
    write_csv(OUT_DIR / "all_step_response_configs.csv", config_rows)
    write_csv(OUT_DIR / "safe_step_response_top200.csv", safe_rows)
    write_csv(OUT_DIR / "top_safe_details.csv", detail_rows)
    target_rows = summarize_by_target(detail_rows, int(safe_rows[0]["config_id"])) if safe_rows else []
    holdout_rows = top_holdout(detail_rows) if detail_rows else []
    write_csv(OUT_DIR / "best_target_summary.csv", target_rows)
    write_csv(OUT_DIR / "top_holdout_summary.csv", holdout_rows)
    write_report(safe_rows, target_rows, holdout_rows)
    print(f"wrote {OUT_DIR / 'all_step_response_configs.csv'}")
    print(f"wrote {OUT_DIR / 'safe_step_response_top200.csv'}")
    print(f"wrote {OUT_DIR / 'top_safe_details.csv'}")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
