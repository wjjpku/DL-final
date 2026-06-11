#!/usr/bin/env python3
"""External sanity checks for the next-generation kappa candidate.

The main predictive-shrinkage matrix uses six WSD/cosine-72 schedules.  This
audit scores the same single-curve next-gen kappas on additional curves present
in the repo but not used in that matrix:

  * cosine_24000.csv
  * constant_24000.csv
  * constant_72000.csv

It also tests a schedule-only target-applicability safety gate.  If the target
response feature is too diffuse, the response direction is not identifiable
apart from low-frequency MPL drift, so the safest transfer action is to abstain.
"""
from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parent
ROOT = REPO.parent
sys.path.insert(0, str(REPO))

import current_law_continuous_kappa_search as base  # noqa: E402
from deep_stime import stime_feature  # noqa: E402


PRED_DIR = ROOT / "results" / "current_law_predictive_shrinkage_audit"
OUT_DIR = ROOT / "results" / "current_law_nextgen_external_holdout_audit"
EXTRA_CURVES = [
    ("cosine_24000.csv", "Cosine 24k"),
    ("constant_24000.csv", "Constant 24k"),
    ("constant_72000.csv", "Constant 72k"),
]
LOCALIZATION_THRESHOLD = 2.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def target_stats(scale: str, curve_name: str) -> dict[str, object]:
    curve = base.load_curve(scale, curve_name)
    phi = stime_feature(curve, base.LAMBDA)
    baseline = base.mpl_predict(base.MPL_PRECOMPUTED_INIT[scale], curve)
    base_mae = base.metrics(curve.loss, baseline)["mae"]
    mean = float(np.mean(phi))
    peak = float(np.max(phi))
    peak_to_mean = 0.0 if peak <= 0 else peak / max(mean, 1e-12)
    return {"curve": curve, "phi": phi, "baseline": baseline, "base_mae": base_mae, "peak_to_mean": peak_to_mean}


def score(stats: dict[str, object], kappa: float) -> dict[str, object]:
    curve = stats["curve"]
    pred = stats["baseline"] + kappa * stats["phi"]
    corr_mae = base.metrics(curve.loss, pred)["mae"]
    base_mae = float(stats["base_mae"])
    return {
        "base_mae": base_mae,
        "corr_mae": corr_mae,
        "delta_pct": 100.0 * (corr_mae / base_mae - 1.0),
        "win": int(corr_mae < base_mae),
    }


def run() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    krows = [
        r
        for r in read_csv(PRED_DIR / "kappa_diagnostics.csv")
        if r["candidate"] == "train_size_rho0p5" and int(r["train_size"]) == 1
    ]
    cached = {(scale, curve): target_stats(scale, curve) for scale in base.SCALES for curve, _ in EXTRA_CURVES}
    details: list[dict[str, object]] = []
    for kr in krows:
        scale = kr["scale"]
        kappa = float(kr["kappa"])
        for curve_name, label in EXTRA_CURVES:
            stats = cached[(scale, curve_name)]
            for mode in ["raw_nextgen", "target_localization_gate"]:
                factor = 1.0
                if mode == "target_localization_gate" and float(stats["peak_to_mean"]) < LOCALIZATION_THRESHOLD:
                    factor = 0.0
                scored = score(stats, kappa * factor)
                details.append(
                    {
                        "mode": mode,
                        "scale": scale,
                        "train_curve": kr["train_id"],
                        "train_label": kr["train_label"],
                        "test_curve": curve_name,
                        "test_label": label,
                        "kappa": kappa,
                        "target_factor": factor,
                        "target_peak_to_mean": float(stats["peak_to_mean"]),
                        **scored,
                    }
                )
    summary = []
    for mode in sorted({str(r["mode"]) for r in details}):
        for curve_name, label in EXTRA_CURVES:
            sub = [r for r in details if r["mode"] == mode and r["test_curve"] == curve_name]
            summary.append(
                {
                    "mode": mode,
                    "test_curve": curve_name,
                    "test_label": label,
                    "tests": len(sub),
                    "mean_delta_pct": float(np.mean([float(r["delta_pct"]) for r in sub])),
                    "worst_delta_pct": float(max(float(r["delta_pct"]) for r in sub)),
                    "wins": int(sum(int(r["win"]) for r in sub)),
                    "mean_target_factor": float(np.mean([float(r["target_factor"]) for r in sub])),
                    "mean_peak_to_mean": float(np.mean([float(r["target_peak_to_mean"]) for r in sub])),
                }
            )
    return details, summary


def write_report(summary: list[dict[str, object]]) -> None:
    raw_cos = next(r for r in summary if r["mode"] == "raw_nextgen" and r["test_curve"] == "cosine_24000.csv")
    gated_cos = next(r for r in summary if r["mode"] == "target_localization_gate" and r["test_curve"] == "cosine_24000.csv")
    lines = [
        "# Next-Gen External Holdout Sanity Audit\n\n",
        "This audit evaluates the next-generation single-curve `rho=0.5` kappa on repo curves not included in the main six-schedule matrix. "
        "`cosine_24000` is also one of the MPL baseline fitting curves, so this is a conservative sanity check rather than a clean independent benchmark.\n\n",
        "## Summary\n\n",
        "| mode | test curve | mean delta | worst delta | wins | target factor | peak/mean |\n",
        "|---|---|---:|---:|---:|---:|---:|\n",
    ]
    for r in summary:
        lines.append(
            f"| `{r['mode']}` | {r['test_label']} | {float(r['mean_delta_pct']):+.1f}% | "
            f"{float(r['worst_delta_pct']):+.1f}% | {int(r['wins'])}/{int(r['tests'])} | "
            f"{float(r['mean_target_factor']):.2f} | {float(r['mean_peak_to_mean']):.1f} |\n"
        )
    lines += [
        "\n## Readout\n\n",
        f"Raw next-gen transfer is not safe on `cosine_24000`: mean `{float(raw_cos['mean_delta_pct']):+.1f}%`, worst `{float(raw_cos['worst_delta_pct']):+.1f}%`. "
        "The constant schedules are unaffected because their response feature is zero.\n\n",
        f"A schedule-only target-localization gate with threshold `{LOCALIZATION_THRESHOLD:.1f}` abstains on diffuse targets such as `cosine_24000` and reduces that failure to mean `{float(gated_cos['mean_delta_pct']):+.1f}%`, worst `{float(gated_cos['worst_delta_pct']):+.1f}%`. "
        "This supports a theoretical limitation: if the target response feature is too diffuse, it is not identifiable apart from low-frequency MPL drift, so transfer should abstain unless target residual evidence is available.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    details, summary = run()
    write_csv(OUT_DIR / "details.csv", details)
    write_csv(OUT_DIR / "summary.csv", summary)
    write_report(summary)
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    for row in summary:
        print(
            f"{row['mode']:24s} {row['test_label']:12s} mean={float(row['mean_delta_pct']):+6.1f}% "
            f"worst={float(row['worst_delta_pct']):+6.1f}% wins={int(row['wins'])}/{int(row['tests'])}"
        )


if __name__ == "__main__":
    main()
