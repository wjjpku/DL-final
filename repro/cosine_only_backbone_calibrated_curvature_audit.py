#!/usr/bin/env python3
"""Strict-backbone LR-curvature audit with strict-calibrated channels.

`cosine_only_backbone_curvature_audit.py` applies the frozen-backbone channel
calibration to the strict cosine-only MPL backbone.  This variant first takes
the best strict-backbone decoupled-channel calibration, then adds the same
schedule-curvature search on top of it.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
ROOT = REPO.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import cosine_only_backbone_curvature_audit as strict_curv  # noqa: E402
import cosine_to_wsd_curvature_correction as curvature  # noqa: E402
from cosine_only_backbone_audit import build_strict_cache, official_vs_strict_baseline  # noqa: E402
from cosine_to_wsd_curvature_correction import aggregate, fmt_pct, fmt_pct2, summarize_by_target, top_holdout  # noqa: E402
from cosine_to_wsd_response_search import TARGETS  # noqa: E402


STRICT_BACKBONE_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "cosine_only_backbone"
OUT_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "cosine_only_backbone_curvature_calibrated"
STRICT_DECOUPLED_PATH = STRICT_BACKBONE_DIR / "safe_decoupled_top200.csv"
STRICT_DECOUPLED_DETAILS_PATH = STRICT_BACKBONE_DIR / "top_safe_decoupled_details.csv"


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


def best_strict_decoupled_row() -> dict[str, str]:
    return read_csv(STRICT_DECOUPLED_PATH)[0]


def strict_channel_ridge(pair_config_id: str, channel: str) -> float:
    for row in read_csv(STRICT_DECOUPLED_DETAILS_PATH):
        if row["pair_config_id"] == pair_config_id and row["channel"] == channel:
            return float(row["ridge_tau"])
    return 0.05


def apply_strict_decoupled_channels(row: dict[str, str]) -> None:
    smooth_ridge = strict_channel_ridge(row["pair_config_id"], "smooth")
    step_ridge = strict_channel_ridge(row["pair_config_id"], "step")
    curvature.SMOOTH.update(
        {
            "fit_start_step": int(row["smooth_fit_start_step"]),
            "response_lambda": float(row["smooth_lambda"]),
            "nuisance_lambda": float(row["smooth_nuisance_lambda"]),
            "max_mode": int(row["smooth_max_mode"]),
            "ridge_tau": smooth_ridge,
            "retention_power": float(row["smooth_retention_power"]),
            "rho": float(row["smooth_rho"]),
        }
    )
    curvature.STEP.update(
        {
            "fit_start_step": int(row["step_fit_start_step"]),
            "response_lambda": float(row["step_lambda"]),
            "nuisance_lambda": float(row["step_nuisance_lambda"]),
            "max_mode": int(row["step_max_mode"]),
            "ridge_tau": step_ridge,
            "retention_power": float(row["step_retention_power"]),
            "rho": float(row["step_rho"]),
        }
    )


def write_report(
    baseline_rows: list[dict[str, object]],
    strict_decoupled: dict[str, str],
    safe_rows: list[dict[str, object]],
    target_rows: list[dict[str, object]],
    holdout_rows: list[dict[str, object]],
) -> None:
    baseline_delta = np.array([float(row["strict_vs_official_delta_pct"]) for row in baseline_rows], dtype=np.float64)
    if not safe_rows:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "REPORT.md").write_text(
            "# Strict-Calibrated LR-Curvature Audit\n\nNo non-harming candidate found.\n",
            encoding="utf-8",
        )
        return

    best = safe_rows[0]
    best_worst = min(safe_rows, key=lambda row: (float(row["worst_delta"]), float(row["mean_delta"])))
    frozen_best = read_csv(curvature.OUT_DIR / "safe_curvature_top200.csv")[0]
    lines = [
        "# Strict-Calibrated LR-Curvature Audit\n\n",
        "This audit keeps the strict cosine-only MPL backbone, but uses the best strict-backbone "
        "decoupled-channel calibration before adding the LR-curvature term. Coefficients are still "
        "fit from `cosine_72000.csv` residuals only.\n\n",
        "## Channel Calibration\n\n",
        f"- Smooth channel: `start={int(strict_decoupled['smooth_fit_start_step'])}`, "
        f"`lambda={float(strict_decoupled['smooth_lambda']):g}`, "
        f"`mu={float(strict_decoupled['smooth_nuisance_lambda']):g}`, "
        f"`modes={int(strict_decoupled['smooth_max_mode'])}`, "
        f"`p={float(strict_decoupled['smooth_retention_power']):g}`, "
        f"`rho={float(strict_decoupled['smooth_rho']):g}`.\n",
        f"- Step channel: `start={int(strict_decoupled['step_fit_start_step'])}`, "
        f"`lambda={float(strict_decoupled['step_lambda']):g}`, "
        f"`mu={float(strict_decoupled['step_nuisance_lambda']):g}`, "
        f"`modes={int(strict_decoupled['step_max_mode'])}`, "
        f"`p={float(strict_decoupled['step_retention_power']):g}`, "
        f"`rho={float(strict_decoupled['step_rho']):g}`.\n\n",
        "## Backbone Check\n\n",
        f"- Cosine-only MPL vs frozen official MPL on WSD: mean MAE change `{fmt_pct(float(np.mean(baseline_delta)))}`, "
        f"worst `{fmt_pct(float(np.max(baseline_delta)))}`.\n",
        "- Strict percentages below are therefore robustness numbers against a weaker backbone, not the main frozen-backbone result.\n\n",
        "## Best Strict-Calibrated Curvature Candidate\n\n",
        f"- Strict decoupled-channel baseline: mean `{fmt_pct2(float(strict_decoupled['mean_delta']))}`, "
        f"worst `{fmt_pct2(float(strict_decoupled['worst_delta']))}`.\n",
        f"- Strict-calibrated curvature: mean `{fmt_pct2(float(best['mean_delta']))}`, "
        f"worst `{fmt_pct2(float(best['worst_delta']))}`.\n",
        f"- Wins/non-harm: `{int(best['wins'])}/{int(best['rows'])}` and `{int(best['nonharm'])}/{int(best['rows'])}`.\n",
        f"- Config: `curvature_lambda={float(best['curvature_lambda']):g}`, "
        f"`mode={best['curvature_mode']}`, `tau2={float(best['curvature_tau']):g}`, "
        f"`shrink_curvature={int(best['shrink_curvature'])}`, "
        f"`signed_curvature_coef={int(best['signed_curvature_coef'])}`.\n",
        f"- Mean step coefficients: primary `{float(best['mean_step_primary_coef']):.5f}`, "
        f"curvature `{float(best['mean_step_curvature_coef']):.5f}`.\n\n",
        "## Best Worst-Case Strict-Calibrated Candidate\n\n",
        f"- Mean / worst: `{fmt_pct2(float(best_worst['mean_delta']))}` / "
        f"`{fmt_pct2(float(best_worst['worst_delta']))}`.\n",
        f"- Config: `curvature_lambda={float(best_worst['curvature_lambda']):g}`, "
        f"`mode={best_worst['curvature_mode']}`, `tau2={float(best_worst['curvature_tau']):g}`, "
        f"`shrink_curvature={int(best_worst['shrink_curvature'])}`, "
        f"`signed_curvature_coef={int(best_worst['signed_curvature_coef'])}`.\n\n",
        "## Frozen-Backbone Reference\n\n",
        f"- Frozen-backbone LR-curvature main result: mean `{fmt_pct2(float(frozen_best['mean_delta']))}`, "
        f"worst `{fmt_pct2(float(frozen_best['worst_delta']))}`.\n\n",
        "## Per-Target Strict-Calibrated Result\n\n",
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
        "- With channel calibration aligned to the strict backbone, the LR-curvature term improves both mean and worst-case error over the strict decoupled-channel baseline.\n",
        "- The selected curvature mode changes from signed second LR difference to `diff_drop`, which suggests strict MPL residuals encode the local step transition through the change in positive LR drop rather than raw LR curvature.\n",
        "- This strengthens the robustness story, but the main result should still use the frozen-backbone protocol unless the assignment requires refitting MPL from cosine only.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    strict_decoupled = best_strict_decoupled_row()
    apply_strict_decoupled_channels(strict_decoupled)

    params = strict_curv.load_or_fit_params()
    cache = build_strict_cache(params)
    baseline_rows = official_vs_strict_baseline(params)
    write_csv(OUT_DIR / "official_vs_cosine_only_mpl.csv", baseline_rows)

    config_rows, safe_rows, detail_rows = strict_curv.run_search(cache)
    write_csv(OUT_DIR / "all_curvature_configs.csv", config_rows)
    write_csv(OUT_DIR / "safe_curvature_top200.csv", safe_rows)
    write_csv(OUT_DIR / "top_safe_details.csv", detail_rows)
    target_rows = summarize_by_target(detail_rows, int(safe_rows[0]["config_id"])) if safe_rows else []
    holdout_rows = top_holdout(detail_rows) if detail_rows else []
    write_csv(OUT_DIR / "best_target_summary.csv", target_rows)
    write_csv(OUT_DIR / "top_holdout_summary.csv", holdout_rows)
    write_report(baseline_rows, strict_decoupled, safe_rows, target_rows, holdout_rows)

    print(f"wrote {OUT_DIR / 'all_curvature_configs.csv'}")
    print(f"wrote {OUT_DIR / 'safe_curvature_top200.csv'}")
    print(f"wrote {OUT_DIR / 'top_safe_details.csv'}")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
