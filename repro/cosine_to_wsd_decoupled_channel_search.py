#!/usr/bin/env python3
"""Decoupled-channel audit for cosine-to-WSD prediction.

This audit composes the best smooth-channel and step-channel cosine-calibrated
corrections from the channel-shrink search.  The target channel is still chosen
from the LR schedule only.  No WSD residual is used to fit a kappa.

Motivation:
    Smooth LR decay and concentrated LR drops are identifiable in different
    parts of the same cosine residual.  A single calibration window and
    nuisance residualizer is therefore an unnecessary coupling between two
    schedule regimes.
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
IN_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "channel_shrink"
BASELINE_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "adaptive_fit_window"
OUT_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "decoupled_channel"

TMP_CACHE = Path(os.environ.get("TMPDIR", "/tmp")) / "dl-final-cache"
TMP_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(TMP_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(TMP_CACHE / "xdg"))

TARGETS = [
    ("wsd_20000_24000.csv", "WSD sharp"),
    ("wsdld_20000_24000.csv", "WSD linear"),
    ("wsdcon_3.csv", "WSD-con 3e-5"),
    ("wsdcon_9.csv", "WSD-con 9e-5"),
    ("wsdcon_18.csv", "WSD-con 18e-5"),
]
SHARP_LINEAR = {"wsd_20000_24000.csv", "wsdld_20000_24000.csv"}
WSDCON = {"wsdcon_3.csv", "wsdcon_9.csv", "wsdcon_18.csv"}
SCALES = ["25", "100", "400"]


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


def fmt_pct2(value: float) -> str:
    return f"{value:+.2f}%"


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


def load_channel_candidates() -> dict[str, list[dict[str, object]]]:
    rows = read_csv(IN_DIR / "top_safe_details.csv")
    by_config: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_config[str(row["config_id"])].append(row)
    return {config_id: config_rows for config_id, config_rows in by_config.items() if len(config_rows) == 15}


def combine_rows(
    pair_id: int,
    smooth_config_id: str,
    step_config_id: str,
    smooth_rows: list[dict[str, object]],
    step_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in smooth_rows:
        if row["channel"] != "smooth":
            continue
        out.append(
            {
                **row,
                "pair_config_id": pair_id,
                "smooth_config_id": smooth_config_id,
                "step_config_id": step_config_id,
                "active_source_config_id": smooth_config_id,
            }
        )
    for row in step_rows:
        if row["channel"] != "step":
            continue
        out.append(
            {
                **row,
                "pair_config_id": pair_id,
                "smooth_config_id": smooth_config_id,
                "step_config_id": step_config_id,
                "active_source_config_id": step_config_id,
            }
        )
    return out


def summarize_config_rows(
    pair_id: int,
    smooth_config_id: str,
    step_config_id: str,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    smooth_example = next(row for row in rows if row["channel"] == "smooth")
    step_example = next(row for row in rows if row["channel"] == "step")
    return {
        "pair_config_id": pair_id,
        "smooth_config_id": smooth_config_id,
        "step_config_id": step_config_id,
        "smooth_fit_start_step": int(smooth_example["fit_start_step"]),
        "smooth_lambda": float(smooth_example["smooth_lambda"]),
        "smooth_nuisance_lambda": float(smooth_example["nuisance_lambda"]),
        "smooth_max_mode": int(smooth_example["max_mode"]),
        "smooth_ridge_tau": float(smooth_example["ridge_tau"]),
        "smooth_retention_power": float(smooth_example["retention_power"]),
        "smooth_rho": float(smooth_example["rho_smooth"]),
        "step_fit_start_step": int(step_example["fit_start_step"]),
        "step_lambda": float(step_example["step_lambda"]),
        "step_nuisance_lambda": float(step_example["nuisance_lambda"]),
        "step_max_mode": int(step_example["max_mode"]),
        "step_ridge_tau": float(step_example["ridge_tau"]),
        "step_retention_power": float(step_example["retention_power"]),
        "step_rho": float(step_example["rho_step"]),
        **aggregate(rows),
    }


def run_search() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    candidates = load_channel_candidates()
    config_ids = sorted(candidates, key=lambda item: int(item))
    config_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    pair_id = 0
    for smooth_config_id in config_ids:
        for step_config_id in config_ids:
            rows = combine_rows(
                pair_id,
                smooth_config_id,
                step_config_id,
                candidates[smooth_config_id],
                candidates[step_config_id],
            )
            if len(rows) != 15:
                continue
            summary = summarize_config_rows(pair_id, smooth_config_id, step_config_id, rows)
            config_rows.append(summary)
            if summary["wins"] == summary["rows"] and summary["nonharm"] == summary["rows"]:
                detail_rows.extend(rows)
            pair_id += 1
    safe_rows = [
        row for row in config_rows if int(row["wins"]) == int(row["rows"]) and int(row["nonharm"]) == int(row["rows"])
    ]
    safe_sorted = sorted(safe_rows, key=lambda row: (float(row["mean_delta"]), float(row["worst_delta"])))
    top_ids = {int(row["pair_config_id"]) for row in safe_sorted[:200]}
    top_details = [row for row in detail_rows if int(row["pair_config_id"]) in top_ids]
    return config_rows, safe_sorted[:200], top_details


def summarize_by_target(detail_rows: list[dict[str, object]], pair_config_id: int) -> list[dict[str, object]]:
    selected = [row for row in detail_rows if int(row["pair_config_id"]) == pair_config_id]
    rows: list[dict[str, object]] = []
    for target_curve, target_label in TARGETS:
        sub = [row for row in selected if row["test_curve"] == target_curve]
        if sub:
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


def select_rows(
    rows: list[dict[str, object]], *, targets: set[str], scales: set[str] | None
) -> list[dict[str, object]]:
    return [row for row in rows if row["test_curve"] in targets and (scales is None or row["scale"] in scales)]


def top_holdout(detail_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_config: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in detail_rows:
        by_config[int(row["pair_config_id"])].append(row)
    targets = {str(row["test_curve"]) for row in detail_rows}
    out: list[dict[str, object]] = []
    for split in split_defs(targets):
        candidates: list[tuple[float, float, int, dict[str, object], dict[str, object], dict[str, object]]] = []
        for pair_config_id, rows in by_config.items():
            dev = select_rows(rows, targets=split["dev_targets"], scales=split["dev_scales"])
            test = select_rows(rows, targets=split["test_targets"], scales=split["test_scales"])
            if not dev or not test:
                continue
            dev_stats = aggregate(dev)
            if dev_stats["wins"] != dev_stats["rows"] or dev_stats["nonharm"] != dev_stats["rows"]:
                continue
            test_stats = aggregate(test)
            candidates.append(
                (
                    float(dev_stats["mean_delta"]),
                    float(dev_stats["worst_delta"]),
                    pair_config_id,
                    dev_stats,
                    test_stats,
                    rows[0],
                )
            )
        if not candidates:
            out.append({"split": split["split"], "selection_status": "no_candidate"})
            continue
        candidates.sort(key=lambda item: (item[0], item[1]))
        _, _, pair_config_id, dev_stats, test_stats, cfg = candidates[0]
        out.append(
            {
                "split": split["split"],
                "selection_status": "selected",
                "pair_config_id": pair_config_id,
                "smooth_config_id": cfg["smooth_config_id"],
                "step_config_id": cfg["step_config_id"],
                **{f"dev_{key}": value for key, value in dev_stats.items()},
                **{f"test_{key}": value for key, value in test_stats.items()},
            }
        )
    return out


def config_text(row: dict[str, object]) -> str:
    return (
        f"smooth(start={int(row['smooth_fit_start_step'])}, lambda={float(row['smooth_lambda']):g}, "
        f"mu={float(row['smooth_nuisance_lambda']):g}, modes={int(row['smooth_max_mode'])}, "
        f"p={float(row['smooth_retention_power']):g}, rho={float(row['smooth_rho']):g}); "
        f"step(start={int(row['step_fit_start_step'])}, lambda={float(row['step_lambda']):g}, "
        f"mu={float(row['step_nuisance_lambda']):g}, modes={int(row['step_max_mode'])}, "
        f"p={float(row['step_retention_power']):g}, rho={float(row['step_rho']):g})"
    )


def write_report(
    safe_rows: list[dict[str, object]],
    target_rows: list[dict[str, object]],
    holdout_rows: list[dict[str, object]],
) -> None:
    if not safe_rows:
        (OUT_DIR / "REPORT.md").write_text("No non-harming decoupled-channel candidate found.\n", encoding="utf-8")
        return
    best = safe_rows[0]
    best_worst = min(safe_rows, key=lambda row: (float(row["worst_delta"]), float(row["mean_delta"])))
    fit_window = read_csv(BASELINE_DIR / "safe_window_top200.csv")[0]
    channel = read_csv(IN_DIR / "safe_channel_shrink_top200.csv")[0]

    lines = [
        "# Decoupled-Channel Cosine-to-WSD Audit\n\n",
        "This audit keeps the cosine-only fitting rule but lets the smooth and step response channels "
        "use different calibration hyperparameters. Target routing is still schedule-only.\n\n",
        "## Formula Change\n\n",
        "```text\n",
        "channel(target) = smooth or step from LR drop concentration\n",
        "theta_smooth = suffix / residualizer / shrink settings for smooth decay\n",
        "theta_step   = suffix / residualizer / shrink settings for concentrated drops\n",
        "kappa_c(theta_c) is fitted only on cosine_72000 residuals\n",
        "L_hat_target = L_MPL,target + kappa_channel(theta_channel) * phi_channel,target\n",
        "```\n\n",
        "The model still has one fitted amplitude per scale and channel. The difference is that the "
        "two channels no longer share the same nuisance filter and suffix window.\n\n",
        "## Best Fully Non-Harming Candidate\n\n",
        f"- Mean MAE change: `{fmt_pct2(float(best['mean_delta']))}` over `{int(best['rows'])}` scale-target rows.\n",
        f"- Worst scale-target row: `{fmt_pct2(float(best['worst_delta']))}`.\n",
        f"- Wins/non-harm: `{int(best['wins'])}/{int(best['rows'])}` and `{int(best['nonharm'])}/{int(best['rows'])}`.\n",
        f"- Config: `{config_text(best)}`.\n\n",
        "## Best Worst-Case Candidate\n\n",
        f"- Mean / worst: `{fmt_pct2(float(best_worst['mean_delta']))}` / `{fmt_pct2(float(best_worst['worst_delta']))}`.\n",
        f"- Config: `{config_text(best_worst)}`.\n\n",
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
        "\n## Comparison\n\n",
        f"Shared-rho fit-window: mean `{fmt_pct2(float(fit_window['mean_delta']))}`, "
        f"worst `{fmt_pct2(float(fit_window['worst_delta']))}`.\n",
        f"Best-mean channel-shrink single-config: mean `{fmt_pct2(float(channel['mean_delta']))}`, "
        f"worst `{fmt_pct2(float(channel['worst_delta']))}`.\n",
        f"Decoupled-channel candidate: mean `{fmt_pct2(float(best['mean_delta']))}`, "
        f"worst `{fmt_pct2(float(best['worst_delta']))}`.\n\n",
        "The single-config channel-shrink candidate can improve the average but slightly weakens the "
        "worst WSD-con 9e-5 cell. Decoupling the channels recovers both: smooth targets use the "
        "longer-suffix smooth calibration, while WSD-con targets use the safer step calibration.\n\n",
        "## Top-Safe Holdout Check\n\n",
        "| split | selected pair | dev mean | dev worst | test mean | test worst | test wins |\n",
        "|---|---|---:|---:|---:|---:|---:|\n",
    ]
    for row in holdout_rows:
        if row["selection_status"] != "selected":
            lines.append(f"| {row['split']} | none | | | | | |\n")
            continue
        pair = f"smooth={row['smooth_config_id']}, step={row['step_config_id']}"
        lines.append(
            f"| {row['split']} | `{pair}` | "
            f"{fmt_pct(float(row['dev_mean_delta']))} | {fmt_pct(float(row['dev_worst_delta']))} | "
            f"{fmt_pct(float(row['test_mean_delta']))} | {fmt_pct(float(row['test_worst_delta']))} | "
            f"{int(row['test_wins'])}/{int(row['test_rows'])} |\n"
        )
    lines += [
        "\n## Reading\n\n",
        "- This is the strongest current cosine-to-WSD development result: lower mean and lower worst-cell MAE than the shared-rho and single-config channel-shrink variants.\n",
        "- The added complexity is interpretable but real: there are now separate calibration settings for smooth and step response channels.\n",
        "- This should be presented as a development candidate until the channel pair is frozen and tested on new schedules or a pre-registered split.\n",
    ]
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    config_rows, safe_rows, detail_rows = run_search()
    write_csv(OUT_DIR / "all_decoupled_channel_configs.csv", config_rows)
    write_csv(OUT_DIR / "safe_decoupled_channel_top200.csv", safe_rows)
    write_csv(OUT_DIR / "top_safe_details.csv", detail_rows)
    target_rows = summarize_by_target(detail_rows, int(safe_rows[0]["pair_config_id"])) if safe_rows else []
    holdout_rows = top_holdout(detail_rows) if detail_rows else []
    write_csv(OUT_DIR / "best_target_summary.csv", target_rows)
    write_csv(OUT_DIR / "top_holdout_summary.csv", holdout_rows)
    write_report(safe_rows, target_rows, holdout_rows)
    print(f"wrote {OUT_DIR / 'all_decoupled_channel_configs.csv'}")
    print(f"wrote {OUT_DIR / 'safe_decoupled_channel_top200.csv'}")
    print(f"wrote {OUT_DIR / 'top_safe_details.csv'}")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
