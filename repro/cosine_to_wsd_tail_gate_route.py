#!/usr/bin/env python3
"""Route tail-gated correction by final LR ratio.

The tail-gated feature improves low-tail WSD-con targets but slightly weakens
the worst high-tail row.  This audit applies the tail gate only when the target
is a concentrated WSD-con schedule with final_lr / peak_lr below a threshold.
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
OUT_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "tail_gate_route"
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


def load_best_details(base_dir: Path, safe_name: str) -> list[dict[str, str]]:
    best = read_csv(base_dir / safe_name)[0]
    details = read_csv(base_dir / "top_safe_details.csv")
    return [row for row in details if row["config_id"] == best["config_id"]]


def use_tail_gate(curve_name: str, threshold: float) -> bool:
    return curve_name in FINAL_LR_RATIO and FINAL_LR_RATIO[curve_name] <= threshold + 1e-12


def routed_rows(threshold: float) -> list[dict[str, object]]:
    joint_rows = load_best_details(JOINT_DIR, "safe_joint_curvature_top200.csv")
    tail_rows = load_best_details(TAIL_DIR, "safe_tail_gate_top200.csv")
    joint = {(row["scale"], row["test_curve"]): row for row in joint_rows}
    tail = {(row["scale"], row["test_curve"]): row for row in tail_rows}
    rows: list[dict[str, object]] = []
    for scale in SCALES:
        for curve_name, label in TARGETS:
            key = (scale, curve_name)
            source = tail[key] if use_tail_gate(curve_name, threshold) else joint[key]
            row = {
                "threshold": threshold,
                "scale": scale,
                "test_curve": curve_name,
                "test_label": label,
                "route": "tail_gate" if use_tail_gate(curve_name, threshold) else "joint",
                "final_lr_ratio": FINAL_LR_RATIO.get(curve_name, ""),
                "delta_pct": float(source["delta_pct"]),
                "base_mae": float(source["base_mae"]),
                "corr_mae": float(source["corr_mae"]),
                "win": int(float(source["delta_pct"]) < 0.0),
            }
            rows.append(row)
    return rows


def run_search() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    config_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    for threshold in THRESHOLDS:
        rows = routed_rows(threshold)
        detail_rows.extend(rows)
        config_rows.append({"threshold": threshold, **aggregate(rows)})
    return config_rows, detail_rows


def summarize_by_target(detail_rows: list[dict[str, object]], threshold: float) -> list[dict[str, object]]:
    selected = [row for row in detail_rows if abs(float(row["threshold"]) - threshold) < 1e-12]
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


def top_holdout(detail_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_threshold: dict[float, list[dict[str, object]]] = defaultdict(list)
    for row in detail_rows:
        by_threshold[float(row["threshold"])].append(row)
    targets = {str(row["test_curve"]) for row in detail_rows}
    out: list[dict[str, object]] = []
    for split in split_defs(targets):
        candidates: list[tuple[float, float, float, dict[str, object], dict[str, object]]] = []
        for threshold, rows in by_threshold.items():
            dev = select_rows(rows, targets=split["dev_targets"], scales=split["dev_scales"])
            test = select_rows(rows, targets=split["test_targets"], scales=split["test_scales"])
            if not dev or not test:
                continue
            dev_stats = aggregate(dev)
            if dev_stats["wins"] != dev_stats["rows"] or dev_stats["nonharm"] != dev_stats["rows"]:
                continue
            test_stats = aggregate(test)
            candidates.append((float(dev_stats["mean_delta"]), float(dev_stats["worst_delta"]), threshold, dev_stats, test_stats))
        if not candidates:
            out.append({"split": split["split"], "selection_status": "no_candidate"})
            continue
        candidates.sort(key=lambda item: (item[0], item[1]))
        _, _, threshold, dev_stats, test_stats = candidates[0]
        out.append(
            {
                "split": split["split"],
                "selection_status": "selected",
                "threshold": threshold,
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
    target_rows = summarize_by_target(detail_rows, float(best["threshold"]))
    lines = [
        "# Tail-Gate Route Audit\n\n",
        "This audit routes between the joint LR-curvature model and the tail-gated model using only the target schedule:\n\n",
        "```text\n",
        "use tail gate if curve is WSD-con and final_lr / peak_lr <= threshold\n",
        "otherwise use the joint LR-curvature prediction\n",
        "```\n\n",
        "## Best Fully Non-Harming Route\n\n",
        f"- Threshold: `{float(best['threshold']):g}`.\n",
        f"- Mean / worst: `{fmt_pct2(float(best['mean_delta']))}` / `{fmt_pct2(float(best['worst_delta']))}`.\n",
        f"- Wins/non-harm: `{int(best['wins'])}/{int(best['rows'])}` and `{int(best['nonharm'])}/{int(best['rows'])}`.\n\n",
        "## Best Worst-Case Route\n\n",
        f"- Threshold: `{float(best_worst['threshold']):g}`.\n",
        f"- Mean / worst: `{fmt_pct2(float(best_worst['mean_delta']))}` / "
        f"`{fmt_pct2(float(best_worst['worst_delta']))}`.\n\n",
        "## Threshold Summary\n\n",
        "| threshold | mean delta | worst | wins |\n",
        "|---:|---:|---:|---:|\n",
    ]
    for row in config_rows:
        lines.append(
            f"| {float(row['threshold']):g} | {fmt_pct2(float(row['mean_delta']))} | "
            f"{fmt_pct2(float(row['worst_delta']))} | {int(row['wins'])}/{int(row['rows'])} |\n"
        )
    lines += [
        "\n## Per-Target Result For Best Route\n\n",
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
        "| split | selected threshold | dev mean | dev worst | test mean | test worst | test wins |\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in holdout_rows:
        if row["selection_status"] != "selected":
            lines.append(f"| {row['split']} | none | | | | | |\n")
            continue
        lines.append(
            f"| {row['split']} | `{float(row['threshold']):g}` | "
            f"{fmt_pct(float(row['dev_mean_delta']))} | {fmt_pct(float(row['dev_worst_delta']))} | "
            f"{fmt_pct(float(row['test_mean_delta']))} | {fmt_pct(float(row['test_worst_delta']))} | "
            f"{int(row['test_wins'])}/{int(row['test_rows'])} |\n"
        )
    lines += [
        "\n## Reading\n\n",
        "- The selected threshold applies the tail gate only to the lowest-tail WSD-con schedule, where the gate improves the residual without weakening the current worst row.\n",
        "- This route is schedule-only, but the threshold is still selected in a development audit over available WSD-family targets.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    config_rows, detail_rows = run_search()
    holdout_rows = top_holdout(detail_rows)
    write_csv(OUT_DIR / "summary.csv", config_rows)
    write_csv(OUT_DIR / "details.csv", detail_rows)
    write_csv(OUT_DIR / "top_holdout_summary.csv", holdout_rows)
    write_report(config_rows, detail_rows, holdout_rows)
    print(f"wrote {OUT_DIR / 'summary.csv'}")
    print(f"wrote {OUT_DIR / 'details.csv'}")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
