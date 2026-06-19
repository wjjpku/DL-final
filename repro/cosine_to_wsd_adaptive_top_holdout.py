#!/usr/bin/env python3
"""Holdout check within the top safe adaptive-search neighborhood.

This is intentionally lightweight: it does not rerun the full adaptive search.
It asks whether the strongest safe configurations already found remain useful
when selected on one WSD subset and evaluated on another.  The fitted kappa in
all rows was still produced by cosine-only calibration in
cosine_to_wsd_adaptive_search.py.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "adaptive_search"
OUT_DIR = IN_DIR / "top_holdout"
SCALES = ["25", "100", "400"]
SHARP_LINEAR = {"wsd_20000_24000.csv", "wsdld_20000_24000.csv"}
WSDCON = {"wsdcon_3.csv", "wsdcon_9.csv", "wsdcon_18.csv"}


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


def aggregate(rows: list[dict[str, str]]) -> dict[str, object]:
    deltas = np.array([float(row["delta_pct"]) for row in rows], dtype=np.float64)
    return {
        "rows": len(rows),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "worst_delta": float(np.max(deltas)),
        "wins": int(np.sum(deltas < 0.0)),
        "nonharm": int(np.sum(deltas <= 1e-12)),
    }


def choose(rows_by_config: dict[str, list[dict[str, str]]], split: dict[str, object]) -> dict[str, object]:
    candidates: list[tuple[float, float, str, dict[str, object], dict[str, object], list[dict[str, str]]]] = []
    for config_id, rows in rows_by_config.items():
        dev = [
            row
            for row in rows
            if row["test_curve"] in split["dev_targets"] and (split["dev_scales"] is None or row["scale"] in split["dev_scales"])
        ]
        test = [
            row
            for row in rows
            if row["test_curve"] in split["test_targets"]
            and (split["test_scales"] is None or row["scale"] in split["test_scales"])
        ]
        if not dev or not test:
            continue
        dev_stats = aggregate(dev)
        if dev_stats["wins"] != dev_stats["rows"] or dev_stats["nonharm"] != dev_stats["rows"]:
            continue
        test_stats = aggregate(test)
        candidates.append((float(dev_stats["mean_delta"]), float(dev_stats["worst_delta"]), config_id, dev_stats, test_stats, rows))
    if not candidates:
        return {"split": split["split"], "selection_status": "no_candidate"}
    candidates.sort(key=lambda item: (item[0], item[1]))
    _, _, config_id, dev_stats, test_stats, rows = candidates[0]
    cfg = rows[0]
    return {
        "split": split["split"],
        "selection_status": "selected",
        "config_id": config_id,
        "smooth_lambda": float(cfg["smooth_lambda"]),
        "step_lambda": float(cfg["step_lambda"]),
        "nuisance_lambda": float(cfg["nuisance_lambda"]),
        "max_mode": int(cfg["max_mode"]),
        "ridge_tau": float(cfg["ridge_tau"]),
        "retention_power": float(cfg["retention_power"]),
        "rho": float(cfg["rho"]),
        **{f"dev_{key}": value for key, value in dev_stats.items()},
        **{f"test_{key}": value for key, value in test_stats.items()},
    }


def split_defs(targets: set[str]) -> list[dict[str, object]]:
    splits: list[dict[str, object]] = [
        {
            "split": "dev_sharp_linear__test_wsdcon",
            "dev_targets": SHARP_LINEAR,
            "test_targets": WSDCON,
            "dev_scales": None,
            "test_scales": None,
        },
        {
            "split": "dev_wsdcon__test_sharp_linear",
            "dev_targets": WSDCON,
            "test_targets": SHARP_LINEAR,
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


def write_report(rows: list[dict[str, object]]) -> None:
    lines = [
        "# Adaptive Top-Safe Holdout Check\n\n",
        "This check uses the per-target rows for the top safe adaptive-search configurations. "
        "It is not a full hyperparameter holdout search; it tests whether the high-performing safe neighborhood is brittle.\n\n",
        "| split | selected config | dev mean | dev worst | test mean | test worst | test wins |\n",
        "|---|---|---:|---:|---:|---:|---:|\n",
    ]
    for row in rows:
        if row["selection_status"] != "selected":
            lines.append(f"| {row['split']} | none | | | | | |\n")
            continue
        cfg = (
            f"lambda_s={float(row['smooth_lambda']):g}, lambda_step={float(row['step_lambda']):g}, "
            f"mu={float(row['nuisance_lambda']):g}, tau={float(row['ridge_tau']):g}, "
            f"p={float(row['retention_power']):g}, rho={float(row['rho']):g}"
        )
        lines.append(
            f"| {row['split']} | `{cfg}` | "
            f"{fmt_pct(float(row['dev_mean_delta']))} | {fmt_pct(float(row['dev_worst_delta']))} | "
            f"{fmt_pct(float(row['test_mean_delta']))} | {fmt_pct(float(row['test_worst_delta']))} | "
            f"{int(row['test_wins'])}/{int(row['test_rows'])} |\n"
        )
    lines += [
        "\n## Reading\n\n",
        "- Within the top safe adaptive-search neighborhood, target-type and scale holdouts remain below MPL.\n",
        "- This does not remove the need for new schedule families, but it reduces the concern that the best adaptive result is a single isolated WSD-family fit.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    detail_rows = read_csv(IN_DIR / "top_safe_details.csv")
    rows_by_config: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in detail_rows:
        rows_by_config[row["config_id"]].append(row)
    targets = {row["test_curve"] for row in detail_rows}
    rows = [choose(rows_by_config, split) for split in split_defs(targets)]
    write_csv(OUT_DIR / "selection_summary.csv", rows)
    write_report(rows)
    print(f"wrote {OUT_DIR / 'selection_summary.csv'}")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
