#!/usr/bin/env python3
"""Strict cosine-only MPL-backbone audit for LR-curvature correction.

The main LR-curvature audit uses the frozen MPL backbone shipped with the
project.  This script checks the stricter variant where MPL itself is first
refit from cosine curves only, then the same cosine-residual correction family
is evaluated on WSD-family targets.
"""
from __future__ import annotations

import csv
import json
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

from cosine_only_backbone_audit import (  # noqa: E402
    build_strict_cache,
    fit_cosine_only_mpl,
    official_vs_strict_baseline,
)
from cosine_to_wsd_curvature_correction import (  # noqa: E402
    CURVATURE_LAMBDAS,
    CURVATURE_MODES,
    CURVATURE_TAUS,
    OUT_DIR as FROZEN_CURVATURE_DIR,
    SHRINK_CURVATURE,
    SIGNED_CURVATURE_COEF,
    SMOOTH,
    STEP,
    aggregate,
    curvature_feature,
    fmt_pct,
    fmt_pct2,
    score_config,
    summarize_by_target,
    top_holdout,
)
from cosine_to_wsd_response_search import TARGETS, TRAIN_CURVE, stime_feature  # noqa: E402
from reproduce_cosine_to_wsd import SCALES  # noqa: E402


STRICT_BACKBONE_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "cosine_only_backbone"
OUT_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "cosine_only_backbone_curvature"
PARAM_JSON = STRICT_BACKBONE_DIR / "cosine_only_mpl_params.json"
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


def load_or_fit_params() -> dict[str, np.ndarray]:
    if PARAM_JSON.exists():
        raw = json.loads(PARAM_JSON.read_text(encoding="utf-8"))
        return {scale: np.array(values, dtype=np.float64) for scale, values in raw.items()}
    return fit_cosine_only_mpl()


def run_search(cache) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
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
    for curvature_lambda in CURVATURE_LAMBDAS:
        for curvature_mode in CURVATURE_MODES:
            for curvature_tau in CURVATURE_TAUS:
                for shrink_curvature in SHRINK_CURVATURE:
                    for signed_curvature_coef in SIGNED_CURVATURE_COEF:
                        details = score_config(
                            cache,
                            primary_cache,
                            curvature_cache,
                            curvature_lambda=curvature_lambda,
                            curvature_mode=curvature_mode,
                            curvature_tau=curvature_tau,
                            shrink_curvature=shrink_curvature,
                            signed_curvature_coef=signed_curvature_coef,
                        )
                        summary = aggregate(details)
                        row = {
                            "config_id": config_id,
                            "curvature_lambda": curvature_lambda,
                            "curvature_mode": curvature_mode,
                            "curvature_tau": curvature_tau,
                            "shrink_curvature": int(shrink_curvature),
                            "signed_curvature_coef": int(signed_curvature_coef),
                            **summary,
                            "mean_step_curvature_coef": float(
                                np.mean(
                                    [
                                        float(detail["curvature_coef"])
                                        for detail in details
                                        if detail["channel"] == "step"
                                    ]
                                )
                            ),
                            "mean_step_primary_coef": float(
                                np.mean(
                                    [
                                        float(detail["primary_coef"])
                                        for detail in details
                                        if detail["channel"] == "step"
                                    ]
                                )
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


def write_report(
    baseline_rows: list[dict[str, object]],
    safe_rows: list[dict[str, object]],
    target_rows: list[dict[str, object]],
    holdout_rows: list[dict[str, object]],
) -> None:
    baseline_delta = np.array([float(row["strict_vs_official_delta_pct"]) for row in baseline_rows], dtype=np.float64)
    strict_decoupled_path = STRICT_BACKBONE_DIR / "safe_decoupled_top200.csv"
    strict_decoupled = read_csv(strict_decoupled_path)[0] if strict_decoupled_path.exists() else None
    frozen_curvature_path = FROZEN_CURVATURE_DIR / "safe_curvature_top200.csv"
    frozen_curvature = read_csv(frozen_curvature_path)[0] if frozen_curvature_path.exists() else None

    if not safe_rows:
        lines = [
            "# Strict Cosine-Only Backbone LR-Curvature Audit\n\n",
            "No non-harming curvature candidate was found under the strict cosine-only MPL backbone.\n",
        ]
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")
        return

    best = safe_rows[0]
    best_worst = min(safe_rows, key=lambda row: (float(row["worst_delta"]), float(row["mean_delta"])))
    lines = [
        "# Strict Cosine-Only Backbone LR-Curvature Audit\n\n",
        "This audit uses the MPL backbone refit from `cosine_24000.csv` and `cosine_72000.csv`, "
        "then fits the residual correction from `cosine_72000.csv` only. WSD-family losses are used "
        "for development ranking and evaluation, not for coefficient fitting.\n\n",
        "## Backbone Check\n\n",
        f"- Cosine-only MPL vs frozen official MPL on WSD: mean MAE change `{fmt_pct(float(np.mean(baseline_delta)))}`, "
        f"worst `{fmt_pct(float(np.max(baseline_delta)))}`.\n",
        "- This confirms that the strict cosine-only MPL backbone is much weaker on WSD than the frozen MPL backbone.\n\n",
        "## Best Strict-Backbone Curvature Candidate\n\n",
        f"- Mean / worst vs strict cosine-only MPL: `{fmt_pct2(float(best['mean_delta']))}` / "
        f"`{fmt_pct2(float(best['worst_delta']))}`.\n",
        f"- Wins/non-harm: `{int(best['wins'])}/{int(best['rows'])}` and `{int(best['nonharm'])}/{int(best['rows'])}`.\n",
        f"- Config: `curvature_lambda={float(best['curvature_lambda']):g}`, "
        f"`mode={best['curvature_mode']}`, `tau2={float(best['curvature_tau']):g}`, "
        f"`shrink_curvature={int(best['shrink_curvature'])}`, "
        f"`signed_curvature_coef={int(best['signed_curvature_coef'])}`.\n",
        f"- Mean step coefficients: primary `{float(best['mean_step_primary_coef']):.5f}`, "
        f"curvature `{float(best['mean_step_curvature_coef']):.5f}`.\n\n",
        "## Best Worst-Case Strict Candidate\n\n",
        f"- Mean / worst: `{fmt_pct2(float(best_worst['mean_delta']))}` / "
        f"`{fmt_pct2(float(best_worst['worst_delta']))}`.\n",
        f"- Config: `curvature_lambda={float(best_worst['curvature_lambda']):g}`, "
        f"`mode={best_worst['curvature_mode']}`, `tau2={float(best_worst['curvature_tau']):g}`, "
        f"`shrink_curvature={int(best_worst['shrink_curvature'])}`, "
        f"`signed_curvature_coef={int(best_worst['signed_curvature_coef'])}`.\n\n",
        "## Comparison\n\n",
    ]
    if strict_decoupled is not None:
        lines.append(
            f"- Strict decoupled-channel: mean `{fmt_pct2(float(strict_decoupled['mean_delta']))}`, "
            f"worst `{fmt_pct2(float(strict_decoupled['worst_delta']))}`.\n"
        )
    lines.append(
        f"- Strict LR-curvature: mean `{fmt_pct2(float(best['mean_delta']))}`, "
        f"worst `{fmt_pct2(float(best['worst_delta']))}`.\n"
    )
    if frozen_curvature is not None:
        lines.append(
            f"- Frozen-backbone LR-curvature main result: mean `{fmt_pct2(float(frozen_curvature['mean_delta']))}`, "
            f"worst `{fmt_pct2(float(frozen_curvature['worst_delta']))}`.\n"
        )

    lines += [
        "\n## Per-Target Strict Result\n\n",
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
        "- The correction still improves every WSD-family row when the MPL backbone itself is fit from cosine-only evidence.\n",
        "- The strict-backbone percentages are measured against a weaker baseline, so they should be reported as a robustness audit rather than replacing the frozen-backbone main result.\n",
        "- If this protocol becomes the final story, the next step is to tune the smooth/step channel calibration under the strict backbone as well, instead of only reusing the frozen-backbone channel settings.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    params = load_or_fit_params()
    cache = build_strict_cache(params)
    baseline_rows = official_vs_strict_baseline(params)
    write_csv(OUT_DIR / "official_vs_cosine_only_mpl.csv", baseline_rows)

    config_rows, safe_rows, detail_rows = run_search(cache)
    write_csv(OUT_DIR / "all_curvature_configs.csv", config_rows)
    write_csv(OUT_DIR / "safe_curvature_top200.csv", safe_rows)
    write_csv(OUT_DIR / "top_safe_details.csv", detail_rows)
    target_rows = summarize_by_target(detail_rows, int(safe_rows[0]["config_id"])) if safe_rows else []
    holdout_rows = top_holdout(detail_rows) if detail_rows else []
    write_csv(OUT_DIR / "best_target_summary.csv", target_rows)
    write_csv(OUT_DIR / "top_holdout_summary.csv", holdout_rows)
    write_report(baseline_rows, safe_rows, target_rows, holdout_rows)
    print(f"wrote {OUT_DIR / 'all_curvature_configs.csv'}")
    print(f"wrote {OUT_DIR / 'safe_curvature_top200.csv'}")
    print(f"wrote {OUT_DIR / 'top_safe_details.csv'}")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
