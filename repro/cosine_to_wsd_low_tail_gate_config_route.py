#!/usr/bin/env python3
"""Low-tail route with selectable tail-gate configuration.

The best global tail-gated model is conservative because it must also cover
WSD-con 9e-5 and 18e-5.  This audit routes only low-tail WSD-con schedules to a
tail-gate configuration and leaves all other schedules on the joint LR-curvature
model.
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

from cosine_to_wsd_curvature_correction import fmt_pct, fmt_pct2  # noqa: E402
from cosine_to_wsd_response_search import TARGETS  # noqa: E402
from reproduce_cosine_to_wsd import SCALES  # noqa: E402


JOINT_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "joint_curvature"
TAIL_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "tail_gated_response"
OUT_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "low_tail_gate_config_route"
THRESHOLDS = [0.0, 0.1, 0.3, 0.6]
FINAL_LR_RATIO = {
    "wsdcon_3.csv": 0.1,
    "wsdcon_9.csv": 0.3,
    "wsdcon_18.csv": 0.6,
}
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


def best_joint_rows() -> dict[tuple[str, str], dict[str, str]]:
    best = read_csv(JOINT_DIR / "safe_joint_curvature_top200.csv")[0]
    rows = read_csv(JOINT_DIR / "top_safe_details.csv")
    return {(row["scale"], row["test_curve"]): row for row in rows if row["config_id"] == best["config_id"]}


def tail_configs() -> tuple[list[dict[str, str]], dict[str, dict[tuple[str, str], dict[str, str]]]]:
    configs = read_csv(TAIL_DIR / "safe_tail_gate_top200.csv")
    details = read_csv(TAIL_DIR / "top_safe_details.csv")
    by_config: dict[str, dict[tuple[str, str], dict[str, str]]] = defaultdict(dict)
    for row in details:
        by_config[row["config_id"]][(row["scale"], row["test_curve"])] = row
    configs = [cfg for cfg in configs if cfg["config_id"] in by_config]
    return configs, by_config


def use_tail_gate(curve_name: str, threshold: float) -> bool:
    return curve_name in FINAL_LR_RATIO and FINAL_LR_RATIO[curve_name] <= threshold + 1e-12


def routed_rows(
    threshold: float,
    tail_config: dict[str, str] | None,
    joint_rows: dict[tuple[str, str], dict[str, str]],
    tail_by_config: dict[str, dict[tuple[str, str], dict[str, str]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    tail_rows = tail_by_config.get(tail_config["config_id"], {}) if tail_config else {}
    for scale in SCALES:
        for curve_name, label in TARGETS:
            key = (scale, curve_name)
            use_tail = tail_config is not None and use_tail_gate(curve_name, threshold)
            source = tail_rows[key] if use_tail else joint_rows[key]
            rows.append(
                {
                    "threshold": threshold,
                    "tail_config_id": tail_config["config_id"] if tail_config else "joint",
                    "gate_mode": tail_config["gate_mode"] if tail_config else "",
                    "gate_tau": tail_config["gate_tau"] if tail_config else "",
                    "gate_sign": tail_config["gate_sign"] if tail_config else "",
                    "shrink_gate": tail_config["shrink_gate"] if tail_config else "",
                    "scale": scale,
                    "test_curve": curve_name,
                    "test_label": label,
                    "route": "tail_gate" if use_tail else "joint",
                    "final_lr_ratio": FINAL_LR_RATIO.get(curve_name, ""),
                    "delta_pct": float(source["delta_pct"]),
                    "base_mae": float(source["base_mae"]),
                    "corr_mae": float(source["corr_mae"]),
                    "win": int(float(source["delta_pct"]) < 0.0),
                }
            )
    return rows


def run_search() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    joint_rows = best_joint_rows()
    configs, tail_by_config = tail_configs()
    config_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []

    # threshold 0 is exactly the joint baseline; include it once.
    rows = routed_rows(0.0, None, joint_rows, tail_by_config)
    detail_rows.extend(rows)
    config_rows.append(
        {
            "threshold": 0.0,
            "tail_config_id": "joint",
            "gate_mode": "",
            "gate_tau": "",
            "gate_sign": "",
            "shrink_gate": "",
            **aggregate(rows),
        }
    )
    for threshold in [t for t in THRESHOLDS if t > 0.0]:
        for cfg in configs:
            rows = routed_rows(threshold, cfg, joint_rows, tail_by_config)
            detail_rows.extend(rows)
            config_rows.append(
                {
                    "threshold": threshold,
                    "tail_config_id": cfg["config_id"],
                    "gate_mode": cfg["gate_mode"],
                    "gate_tau": cfg["gate_tau"],
                    "gate_sign": cfg["gate_sign"],
                    "shrink_gate": cfg["shrink_gate"],
                    **aggregate(rows),
                    "source_tail_mean_delta": float(cfg["mean_delta"]),
                    "source_tail_worst_delta": float(cfg["worst_delta"]),
                    "source_tail_gate_coef": float(cfg["mean_step_gate_coef"]),
                }
            )
    return config_rows, detail_rows


def summarize_by_target(detail_rows: list[dict[str, object]], config: dict[str, object]) -> list[dict[str, object]]:
    selected = [
        row
        for row in detail_rows
        if abs(float(row["threshold"]) - float(config["threshold"])) < 1e-12
        and row["tail_config_id"] == config["tail_config_id"]
    ]
    rows: list[dict[str, object]] = []
    for target_curve, target_label in TARGETS:
        sub = [row for row in selected if row["test_curve"] == target_curve]
        rows.append({"test_curve": target_curve, "test_label": target_label, **aggregate(sub)})
    return rows


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


def select_rows(rows: list[dict[str, object]], *, targets: set[str], scales: set[str] | None) -> list[dict[str, object]]:
    return [row for row in rows if row["test_curve"] in targets and (scales is None or row["scale"] in scales)]


def top_holdout(config_rows: list[dict[str, object]], detail_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_config: dict[tuple[float, str], list[dict[str, object]]] = defaultdict(list)
    for row in detail_rows:
        by_config[(float(row["threshold"]), str(row["tail_config_id"]))].append(row)
    targets = {str(row["test_curve"]) for row in detail_rows}
    out: list[dict[str, object]] = []
    for split in split_defs(targets):
        candidates: list[tuple[float, float, tuple[float, str], dict[str, object], dict[str, object], dict[str, object]]] = []
        for cfg in config_rows:
            key = (float(cfg["threshold"]), str(cfg["tail_config_id"]))
            rows = by_config[key]
            dev = select_rows(rows, targets=split["dev_targets"], scales=split["dev_scales"])
            test = select_rows(rows, targets=split["test_targets"], scales=split["test_scales"])
            if not dev or not test:
                continue
            dev_stats = aggregate(dev)
            if dev_stats["wins"] != dev_stats["rows"] or dev_stats["nonharm"] != dev_stats["rows"]:
                continue
            test_stats = aggregate(test)
            candidates.append((float(dev_stats["mean_delta"]), float(dev_stats["worst_delta"]), key, dev_stats, test_stats, cfg))
        if not candidates:
            out.append({"split": split["split"], "selection_status": "no_candidate"})
            continue
        candidates.sort(key=lambda item: (item[0], item[1]))
        _, _, _, dev_stats, test_stats, cfg = candidates[0]
        out.append(
            {
                "split": split["split"],
                "selection_status": "selected",
                "threshold": cfg["threshold"],
                "tail_config_id": cfg["tail_config_id"],
                "gate_mode": cfg["gate_mode"],
                "gate_tau": cfg["gate_tau"],
                "gate_sign": cfg["gate_sign"],
                "shrink_gate": cfg["shrink_gate"],
                **{f"dev_{key}": value for key, value in dev_stats.items()},
                **{f"test_{key}": value for key, value in test_stats.items()},
            }
        )
    return out


def write_report(config_rows: list[dict[str, object]], detail_rows: list[dict[str, object]], holdout_rows: list[dict[str, object]]) -> None:
    safe_rows = [
        row for row in config_rows if int(row["wins"]) == int(row["rows"]) and int(row["nonharm"]) == int(row["rows"])
    ]
    best = min(safe_rows, key=lambda row: (float(row["mean_delta"]), float(row["worst_delta"])))
    best_worst = min(safe_rows, key=lambda row: (float(row["worst_delta"]), float(row["mean_delta"])))
    target_rows = summarize_by_target(detail_rows, best)
    lines = [
        "# Low-Tail Gate Config Route Audit\n\n",
        "This audit keeps the joint LR-curvature prediction for all schedules except WSD-con targets whose "
        "`final_lr / peak_lr` is below a threshold. Those low-tail targets use one selected tail-gate "
        "configuration, fitted from cosine residuals only.\n\n",
        "## Best Fully Non-Harming Route\n\n",
        f"- Threshold / tail config: `{float(best['threshold']):g}` / `{best['tail_config_id']}`.\n",
        f"- Gate: `mode={best['gate_mode']}`, `tau={best['gate_tau']}`, "
        f"`sign={best['gate_sign']}`, `shrink={best['shrink_gate']}`.\n",
        f"- Mean / worst: `{fmt_pct2(float(best['mean_delta']))}` / `{fmt_pct2(float(best['worst_delta']))}`.\n",
        f"- Wins/non-harm: `{int(best['wins'])}/{int(best['rows'])}` and `{int(best['nonharm'])}/{int(best['rows'])}`.\n\n",
        "## Best Worst-Case Route\n\n",
        f"- Threshold / tail config: `{float(best_worst['threshold']):g}` / `{best_worst['tail_config_id']}`.\n",
        f"- Mean / worst: `{fmt_pct2(float(best_worst['mean_delta']))}` / "
        f"`{fmt_pct2(float(best_worst['worst_delta']))}`.\n\n",
        "## Per-Target Result For Best Route\n\n",
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
        "| split | selected route | dev mean | dev worst | test mean | test worst | test wins |\n",
        "|---|---|---:|---:|---:|---:|---:|\n",
    ]
    for row in holdout_rows:
        if row["selection_status"] != "selected":
            lines.append(f"| {row['split']} | none | | | | | |\n")
            continue
        cfg = (
            f"thr={float(row['threshold']):g}, cfg={row['tail_config_id']}, "
            f"mode={row['gate_mode']}, tau={row['gate_tau']}"
        )
        lines.append(
            f"| {row['split']} | `{cfg}` | "
            f"{fmt_pct(float(row['dev_mean_delta']))} | {fmt_pct(float(row['dev_worst_delta']))} | "
            f"{fmt_pct(float(row['test_mean_delta']))} | {fmt_pct(float(row['test_worst_delta']))} | "
            f"{int(row['test_wins'])}/{int(row['test_rows'])} |\n"
        )
    lines += [
        "\n## Reading\n\n",
        "- This improves the low-tail WSD-con row without applying the same gate to the mid/high-tail WSD-con rows that it slightly harms.\n",
        "- The route is schedule-only, but both the threshold and tail-gate configuration are selected in a development audit over available WSD-family targets.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    config_rows, detail_rows = run_search()
    safe_rows = [
        row for row in config_rows if int(row["wins"]) == int(row["rows"]) and int(row["nonharm"]) == int(row["rows"])
    ]
    safe_sorted = sorted(safe_rows, key=lambda row: (float(row["mean_delta"]), float(row["worst_delta"])))
    holdout_rows = top_holdout(safe_sorted[:200], detail_rows)
    write_csv(OUT_DIR / "all_routes.csv", config_rows)
    write_csv(OUT_DIR / "safe_routes_top200.csv", safe_sorted[:200])
    write_csv(OUT_DIR / "details.csv", detail_rows)
    write_csv(OUT_DIR / "top_holdout_summary.csv", holdout_rows)
    write_report(safe_sorted[:200], detail_rows, holdout_rows)
    print(f"wrote {OUT_DIR / 'all_routes.csv'}")
    print(f"wrote {OUT_DIR / 'safe_routes_top200.csv'}")
    print(f"wrote {OUT_DIR / 'details.csv'}")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
