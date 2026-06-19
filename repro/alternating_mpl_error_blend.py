#!/usr/bin/env python3
"""Under-relaxed alternating MPL/error correction audit.

The full alternating update can improve the mean but damage a few WSD rows.  A
standard way to stabilize fixed-point style updates is under-relaxation:

    prediction(w) = (1 - w) * prediction_first + w * prediction_alternating

where w=0 is the original two-stage correction and w=1 is the full alternating
result.  This script scans w without refitting anything.
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

from alternating_mpl_error_refit import ErrorFit, OUT_DIR as ALT_DIR, target_correction  # noqa: E402
from cosine_to_wsd_curvature_correction import fmt_pct, fmt_pct2  # noqa: E402
from cosine_to_wsd_response_search import TARGETS  # noqa: E402
from reproduce_cosine_to_wsd import SCALES, load_curve, metrics, mpl_predict  # noqa: E402


OUT_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "alternating_mpl_error_blend"
WEIGHTS = [round(float(x), 2) for x in np.linspace(0.0, 1.0, 21)]


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


def params_from_row(row: dict[str, str]) -> np.ndarray:
    return np.array([float(row[f"p{i}"]) for i in range(7)], dtype=np.float64)


def fit_from_row(row: dict[str, str]) -> ErrorFit:
    return ErrorFit(
        smooth_coef=float(row["smooth_coef"]),
        step_primary_coef=float(row["step_primary_coef"]),
        step_curvature_coef=float(row["step_curvature_coef"]),
        smooth_retention=0.0,
        step_primary_retention=0.0,
    )


def aggregate(rows: list[dict[str, object]], key: str = "delta_vs_pure_pct") -> dict[str, object]:
    deltas = np.array([float(row[key]) for row in rows], dtype=np.float64)
    return {
        "rows": len(rows),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "worst_delta": float(np.max(deltas)),
        "wins": int(np.sum(deltas < 0.0)),
        "nonharm": int(np.sum(deltas <= 1e-12)),
    }


def indexed_params() -> dict[tuple[str, str, str], dict[str, str]]:
    rows = read_csv(ALT_DIR / "params.csv")
    return {(row["variant"], row["scale"], row["stage"]): row for row in rows}


def score_variant_weight(
    index: dict[tuple[str, str, str], dict[str, str]],
    variant: str,
    weight: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scale in SCALES:
        pure_params = params_from_row(index[(variant, scale, "pure_mpl")])
        refit_params = params_from_row(index[(variant, scale, "refit_mpl")])
        first_fit = fit_from_row(index[(variant, scale, "first_error")])
        final_fit = fit_from_row(index[(variant, scale, "final_error")])
        for target_curve, target_label in TARGETS:
            curve = load_curve(scale, target_curve)
            pure_pred = mpl_predict(pure_params, curve)
            first_corr, _, channel = target_correction(curve, first_fit)
            final_corr, _, _ = target_correction(curve, final_fit)
            first_pred = pure_pred + first_corr
            final_pred = mpl_predict(refit_params, curve) + final_corr
            pred = (1.0 - weight) * first_pred + weight * final_pred

            pure_mae = metrics(curve.loss, pure_pred)["mae"]
            first_mae = metrics(curve.loss, first_pred)["mae"]
            corr_mae = metrics(curve.loss, pred)["mae"]
            rows.append(
                {
                    "variant": variant,
                    "weight": weight,
                    "scale": scale,
                    "test_curve": target_curve,
                    "test_label": target_label,
                    "channel": channel,
                    "pure_mpl_mae": pure_mae,
                    "first_error_mae": first_mae,
                    "blend_mae": corr_mae,
                    "delta_vs_pure_pct": 100.0 * (corr_mae / pure_mae - 1.0),
                    "delta_vs_first_error_pct": 100.0 * (corr_mae / first_mae - 1.0),
                    "win_vs_pure": int(corr_mae < pure_mae),
                    "win_vs_first_error": int(corr_mae < first_mae),
                }
            )
    return rows


def run_search() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    index = indexed_params()
    variants = sorted({variant for variant, _, _ in index})
    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for variant in variants:
        for weight in WEIGHTS:
            rows = score_variant_weight(index, variant, weight)
            detail_rows.extend(rows)
            summary_rows.append({"variant": variant, "weight": weight, "baseline": "pure_mpl", **aggregate(rows)})
            summary_rows.append(
                {
                    "variant": variant,
                    "weight": weight,
                    "baseline": "first_error",
                    **aggregate(rows, "delta_vs_first_error_pct"),
                }
            )
    return summary_rows, detail_rows


def summarize_targets(detail_rows: list[dict[str, object]], variant: str, weight: float) -> list[dict[str, object]]:
    selected = [
        row
        for row in detail_rows
        if row["variant"] == variant and abs(float(row["weight"]) - weight) < 1e-12
    ]
    rows: list[dict[str, object]] = []
    for target_curve, target_label in TARGETS:
        sub = [row for row in selected if row["test_curve"] == target_curve]
        rows.append({"test_curve": target_curve, "test_label": target_label, **aggregate(sub)})
    return rows


def write_report(summary_rows: list[dict[str, object]], detail_rows: list[dict[str, object]]) -> None:
    pure_rows = [row for row in summary_rows if row["baseline"] == "pure_mpl"]
    safe_rows = [
        row
        for row in pure_rows
        if int(row["wins"]) == int(row["rows"]) and int(row["nonharm"]) == int(row["rows"])
    ]
    best_safe = min(safe_rows, key=lambda row: (float(row["mean_delta"]), float(row["worst_delta"])))
    best_worst = min(safe_rows, key=lambda row: (float(row["worst_delta"]), float(row["mean_delta"])))
    best_mean = min(pure_rows, key=lambda row: (float(row["mean_delta"]), float(row["worst_delta"])))
    best_vs_first = next(
        row
        for row in summary_rows
        if row["baseline"] == "first_error"
        and row["variant"] == best_safe["variant"]
        and abs(float(row["weight"]) - float(best_safe["weight"])) < 1e-12
    )
    target_rows = summarize_targets(detail_rows, str(best_safe["variant"]), float(best_safe["weight"]))

    lines = [
        "# Under-Relaxed Alternating MPL/Error Audit\n\n",
        "This audit does not refit parameters. It blends the original two-stage prediction with the full alternating prediction:\n\n",
        "```text\n",
        "L_hat(w) = (1 - w) L_hat_first + w L_hat_alternating\n",
        "```\n\n",
        "`w=0` is the strict-calibrated two-stage correction. `w=1` is the full alternating refit.\n\n",
        "## Best Fully Non-Harming Blend\n\n",
        f"- Variant / weight: `{best_safe['variant']}`, `w={float(best_safe['weight']):.2f}`.\n",
        f"- Vs pure strict MPL: mean `{fmt_pct2(float(best_safe['mean_delta']))}`, "
        f"worst `{fmt_pct2(float(best_safe['worst_delta']))}`, wins "
        f"`{int(best_safe['wins'])}/{int(best_safe['rows'])}`.\n",
        f"- Vs first two-stage correction: mean `{fmt_pct2(float(best_vs_first['mean_delta']))}`, "
        f"worst `{fmt_pct2(float(best_vs_first['worst_delta']))}`, wins "
        f"`{int(best_vs_first['wins'])}/{int(best_vs_first['rows'])}`.\n\n",
        "## Best Worst-Case Blend\n\n",
        f"- Variant / weight: `{best_worst['variant']}`, `w={float(best_worst['weight']):.2f}`.\n",
        f"- Mean / worst: `{fmt_pct2(float(best_worst['mean_delta']))}` / "
        f"`{fmt_pct2(float(best_worst['worst_delta']))}`.\n\n",
        "## Best Mean-Only Blend\n\n",
        f"- Variant / weight: `{best_mean['variant']}`, `w={float(best_mean['weight']):.2f}`.\n",
        f"- Mean / worst / wins: `{fmt_pct2(float(best_mean['mean_delta']))}` / "
        f"`{fmt_pct2(float(best_mean['worst_delta']))}` / "
        f"`{int(best_mean['wins'])}/{int(best_mean['rows'])}`.\n\n",
        "## Per-Target Result For Best Non-Harming Blend\n\n",
        "| target | mean delta vs pure MPL | worst scale | wins |\n",
        "|---|---:|---:|---:|\n",
    ]
    for row in target_rows:
        lines.append(
            f"| {row['test_label']} | {fmt_pct(float(row['mean_delta']))} | "
            f"{fmt_pct(float(row['worst_delta']))} | {int(row['wins'])}/{int(row['rows'])} |\n"
        )
    lines += [
        "\n## Reading\n\n",
        "- This is an under-relaxed fixed-point interpretation of the alternating update. It is cheaper and more stable than another MPL refit.\n",
        "- A useful blend should improve over the first two-stage correction without reintroducing positive worst-case rows.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    summary_rows, detail_rows = run_search()
    write_csv(OUT_DIR / "summary.csv", summary_rows)
    write_csv(OUT_DIR / "details.csv", detail_rows)
    write_report(summary_rows, detail_rows)
    print(f"wrote {OUT_DIR / 'summary.csv'}")
    print(f"wrote {OUT_DIR / 'details.csv'}")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
