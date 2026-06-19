#!/usr/bin/env python3
"""Holdout audit for the cosine-to-WSD response search.

The fitted correction amplitude is always estimated from cosine_72000.
This audit only tests whether the searched hyperparameters are brittle when
selected on one subset of WSD targets/scales and evaluated on another.
"""
from __future__ import annotations

import csv
import os
import sys
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

from cosine_to_wsd_response_search import (  # noqa: E402
    MAX_MODES,
    NUISANCE_LAMBDAS,
    RESPONSE_LAMBDAS,
    RETENTION_POWERS,
    RHOS,
    RIDGE_TAUS,
    TARGETS,
    TARGET_RETENTION_FLOOR,
    TRAIN_CURVE,
    build_cache,
    fit_source_kappa,
    score_target,
    stime_feature,
    target_retention,
)
from reproduce_cosine_to_wsd import SCALES  # noqa: E402


OUT_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "holdout_audit"

TARGET_GROUPS = {
    "sharp_linear": {"wsd_20000_24000.csv", "wsdld_20000_24000.csv"},
    "wsdcon_all": {"wsdcon_3.csv", "wsdcon_9.csv", "wsdcon_18.csv"},
}


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


def aggregate(rows: list[dict[str, object]], prefix: str) -> dict[str, object]:
    if not rows:
        return {
            f"{prefix}_rows": 0,
            f"{prefix}_mean_delta": float("nan"),
            f"{prefix}_median_delta": float("nan"),
            f"{prefix}_worst_delta": float("nan"),
            f"{prefix}_wins": 0,
            f"{prefix}_nonharm": 0,
        }
    deltas = np.array([float(row["delta_pct"]) for row in rows], dtype=np.float64)
    return {
        f"{prefix}_rows": len(rows),
        f"{prefix}_mean_delta": float(np.mean(deltas)),
        f"{prefix}_median_delta": float(np.median(deltas)),
        f"{prefix}_worst_delta": float(np.max(deltas)),
        f"{prefix}_wins": int(np.sum(deltas < 0.0)),
        f"{prefix}_nonharm": int(np.sum(deltas <= 1e-12)),
    }


def split_defs() -> list[dict[str, object]]:
    out: list[dict[str, object]] = [
        {
            "split": "dev_sharp_linear__test_wsdcon",
            "kind": "target_type",
            "dev_targets": TARGET_GROUPS["sharp_linear"],
            "test_targets": TARGET_GROUPS["wsdcon_all"],
            "dev_scales": set(SCALES),
            "test_scales": set(SCALES),
        },
        {
            "split": "dev_wsdcon__test_sharp_linear",
            "kind": "target_type",
            "dev_targets": TARGET_GROUPS["wsdcon_all"],
            "test_targets": TARGET_GROUPS["sharp_linear"],
            "dev_scales": set(SCALES),
            "test_scales": set(SCALES),
        },
    ]
    target_names = [curve for curve, _ in TARGETS]
    for held_curve, held_label in TARGETS:
        out.append(
            {
                "split": f"leave_target__{held_curve.replace('.csv', '')}",
                "kind": "leave_target_out",
                "held_label": held_label,
                "dev_targets": set(target_names) - {held_curve},
                "test_targets": {held_curve},
                "dev_scales": set(SCALES),
                "test_scales": set(SCALES),
            }
        )
    for held_scale in SCALES:
        out.append(
            {
                "split": f"leave_scale__{held_scale}M",
                "kind": "leave_scale_out",
                "held_scale": held_scale,
                "dev_targets": set(target_names),
                "test_targets": set(target_names),
                "dev_scales": set(SCALES) - {held_scale},
                "test_scales": {held_scale},
            }
        )
    return out


def select_rows(
    rows: list[dict[str, object]],
    *,
    targets: set[str],
    scales: set[str],
) -> list[dict[str, object]]:
    return [row for row in rows if row["test_curve"] in targets and row["scale"] in scales]


def score_config(
    cache,
    feature_cache,
    *,
    response_lambda: float,
    nuisance_lambda: float,
    max_mode: int,
    ridge_tau: float,
    retention_power: float,
    rho: float,
) -> list[dict[str, object]]:
    config_details: list[dict[str, object]] = []
    for scale in SCALES:
        source = cache[(scale, TRAIN_CURVE)]
        source_phi = feature_cache[(scale, TRAIN_CURVE, response_lambda)]
        fit = fit_source_kappa(
            source,
            source_phi,
            nuisance_lambda=nuisance_lambda,
            max_mode=max_mode,
            ridge_tau=ridge_tau,
            retention_power=retention_power,
            rho=rho,
        )
        for target_curve, target_label in TARGETS:
            target = cache[(scale, target_curve)]
            phi_t = feature_cache[(scale, target_curve, response_lambda)]
            retention_t = target_retention(phi_t, nuisance_lambda=nuisance_lambda, max_mode=max_mode)
            target_factor = 1.0 if retention_t >= TARGET_RETENTION_FLOOR else 0.0
            scored = score_target(target, phi_t, float(fit["kappa"]) * target_factor)
            config_details.append(
                {
                    "response_lambda": response_lambda,
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
    return config_details


def run() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    cache = build_cache()
    feature_cache = {
        (scale, curve_name, response_lambda): stime_feature(cache[(scale, curve_name)].curve, response_lambda)
        for scale in SCALES
        for curve_name, _ in [(TRAIN_CURVE, "Cosine")] + TARGETS
        for response_lambda in RESPONSE_LAMBDAS
    }
    splits = split_defs()
    best_by_split: dict[str, tuple[tuple[float, float], dict[str, object], list[dict[str, object]]]] = {}
    config_id = 0
    for response_lambda in RESPONSE_LAMBDAS:
        for nuisance_lambda in NUISANCE_LAMBDAS:
            for max_mode in MAX_MODES:
                for ridge_tau in RIDGE_TAUS:
                    for retention_power in RETENTION_POWERS:
                        for rho in RHOS:
                            details = score_config(
                                cache,
                                feature_cache,
                                response_lambda=response_lambda,
                                nuisance_lambda=nuisance_lambda,
                                max_mode=max_mode,
                                ridge_tau=ridge_tau,
                                retention_power=retention_power,
                                rho=rho,
                            )
                            for split in splits:
                                dev_rows = select_rows(
                                    details,
                                    targets=split["dev_targets"],
                                    scales=split["dev_scales"],
                                )
                                test_rows = select_rows(
                                    details,
                                    targets=split["test_targets"],
                                    scales=split["test_scales"],
                                )
                                dev = aggregate(dev_rows, "dev")
                                if int(dev["dev_nonharm"]) != int(dev["dev_rows"]):
                                    continue
                                if int(dev["dev_wins"]) != int(dev["dev_rows"]):
                                    continue
                                test = aggregate(test_rows, "test")
                                key = (float(dev["dev_mean_delta"]), float(dev["dev_worst_delta"]))
                                summary = {
                                    "config_id": config_id,
                                    "split": split["split"],
                                    "kind": split["kind"],
                                    "response_lambda": response_lambda,
                                    "nuisance_lambda": nuisance_lambda,
                                    "max_mode": max_mode,
                                    "ridge_tau": ridge_tau,
                                    "retention_power": retention_power,
                                    "rho": rho,
                                    **dev,
                                    **test,
                                }
                                current = best_by_split.get(str(split["split"]))
                                if current is None or key < current[0]:
                                    best_by_split[str(split["split"])] = (key, summary, details)
                            config_id += 1
    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    for split in splits:
        found = best_by_split.get(str(split["split"]))
        if found is None:
            summary_rows.append(
                {
                    "split": split["split"],
                    "kind": split["kind"],
                    "selection_status": "no_dev_nonharm_candidate",
                }
            )
            continue
        _, summary, details = found
        summary["selection_status"] = "selected"
        summary_rows.append(summary)
        for row in details:
            detail_rows.append({"selected_for_split": split["split"], **row})
    return summary_rows, detail_rows


def write_report(summary_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Cosine-to-WSD Holdout Audit\n\n",
        "This audit keeps the fitted amplitude source fixed to `cosine_72000.csv`. "
        "It only varies which WSD subset is allowed to select hyperparameters, then evaluates the selected configuration on the held-out subset.\n\n",
        "| split | selected config | dev mean | dev worst | test mean | test worst | test wins |\n",
        "|---|---|---:|---:|---:|---:|---:|\n",
    ]
    for row in summary_rows:
        if row.get("selection_status") != "selected":
            lines.append(f"| {row['split']} | none | | | | | |\n")
            continue
        config = (
            f"lambda={float(row['response_lambda']):g}, "
            f"mu={float(row['nuisance_lambda']):g}, "
            f"tau={float(row['ridge_tau']):g}, "
            f"p={float(row['retention_power']):g}, "
            f"rho={float(row['rho']):g}"
        )
        lines.append(
            f"| {row['split']} | `{config}` | "
            f"{fmt_pct(float(row['dev_mean_delta']))} | {fmt_pct(float(row['dev_worst_delta']))} | "
            f"{fmt_pct(float(row['test_mean_delta']))} | {fmt_pct(float(row['test_worst_delta']))} | "
            f"{int(row['test_wins'])}/{int(row['test_rows'])} |\n"
        )
    lines += [
        "\n## Reading\n\n",
        "- A healthy result is not that every split chooses the same hyperparameters; it is that held-out WSD targets remain below MPL after hyperparameters are chosen elsewhere.\n",
        "- Failures here would mean the cosine-derived correction is too sensitive to WSD-family hyperparameter selection, even though kappa itself is still fitted only on cosine.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    summary_rows, detail_rows = run()
    write_csv(OUT_DIR / "selection_summary.csv", summary_rows)
    write_csv(OUT_DIR / "selected_config_details.csv", detail_rows)
    write_report(summary_rows)
    print(f"wrote {OUT_DIR / 'selection_summary.csv'}")
    print(f"wrote {OUT_DIR / 'selected_config_details.csv'}")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
