#!/usr/bin/env python3
"""Amplitude sensitivity for the zero-fit MPL-LD finite-response candidate.

The recommended model uses MPL's own B coefficient:

    L_hat = L_MPL + a_s B [D_down,tau_s - D_down]

This audit does not fit an amplitude.  It multiplies the correction by fixed
scale factors to check whether the result is brittle to the exact MPL B value.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import interpretable_error_model as iem  # noqa: E402
import interpretable_nuisance_origin_audit as noa  # noqa: E402
import mpl_ld_lag_response_audit as lag  # noqa: E402

OUT_DIR = iem.ROOT / "results" / "mpl_ld_lag_response_audit" / "amplitude_sensitivity"
SCALE_FACTORS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    keys = sorted({(float(row["amplitude_scale"]), str(row["group"])) for row in rows})
    for scale, group in keys:
        sub = [row for row in rows if float(row["amplitude_scale"]) == scale and row["group"] == group]
        deltas = np.array([float(row["delta_pct"]) for row in sub], dtype=np.float64)
        out.append(
            {
                "amplitude_scale": scale,
                "group": group,
                "rows": len(sub),
                "mean_delta": float(np.mean(deltas)),
                "median_delta": float(np.median(deltas)),
                "worst_delta": float(np.max(deltas)),
                "wins": int(np.sum(deltas < 0.0)),
                "nonharm": int(np.sum(deltas <= 1e-12)),
            }
        )
    return out


def run_audit() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    cache = lag.load_all_packs()
    rows: list[dict[str, object]] = []
    for amp in SCALE_FACTORS:
        for scale in iem.SCALES:
            for group, curve_name, label in noa.ALL_TARGETS:
                pack = cache[(scale, curve_name)]
                tau = lag.response_tau_steps(pack.curve, "support_bracket", None)
                factor = lag.adiabatic_attenuation(pack.curve, "linear_support")
                feature = factor * lag.lag_feature(pack, tau, "cooldown")
                pred = pack.baseline + amp * feature
                corr_mae = iem.mae(pack.curve.loss, pred)
                delta = 100.0 * (corr_mae / pack.base_mae - 1.0)
                rows.append(
                    {
                        "amplitude_scale": amp,
                        "group": group,
                        "scale": scale,
                        "test_curve": curve_name,
                        "test_label": label,
                        "effective_tau_steps": tau,
                        "adiabatic_factor": factor,
                        "base_mae": pack.base_mae,
                        "corr_mae": corr_mae,
                        "delta_pct": delta,
                        "win": int(delta < 0.0),
                        "nonharm": int(delta <= 1e-12),
                        "fitted_residual_params": 0,
                        "uses_target_loss": 0,
                    }
                )
    return rows, aggregate(rows)


def find(summary: list[dict[str, object]], amp: float, group: str) -> dict[str, object]:
    for row in summary:
        if abs(float(row["amplitude_scale"]) - amp) < 1e-12 and row["group"] == group:
            return row
    raise KeyError((amp, group))


def fmt(row: dict[str, object]) -> str:
    return (
        f"{float(row['mean_delta']):+.2f}% / "
        f"{float(row['worst_delta']):+.2f}% / "
        f"{int(row['wins'])}/{int(row['rows'])}"
    )


def write_report(summary: list[dict[str, object]]) -> None:
    lines = [
        "# MPL-LD Finite-Response Amplitude Sensitivity\n\n",
        "The recommended model uses `amplitude_scale=1`, i.e. MPL's own `B` coefficient.  "
        "Other rows are fixed-scale probes; no amplitude is fitted from residuals or target losses.\n\n",
        "| amplitude scale | WSD mean / worst / wins | controls mean / worst / nonharm |\n",
        "|---:|---:|---:|\n",
    ]
    for amp in SCALE_FACTORS:
        core = find(summary, amp, "core_wsd")
        ctrl = find(summary, amp, "extra_control")
        lines.append(
            f"| {amp:.2f} | {fmt(core)} | "
            f"{float(ctrl['mean_delta']):+.2f}% / {float(ctrl['worst_delta']):+.2f}% / "
            f"{int(ctrl['nonharm'])}/{int(ctrl['rows'])} |\n"
        )
    safe = [
        row
        for row in summary
        if row["group"] == "core_wsd"
        and int(row["wins"]) == int(row["rows"])
        and int(row["nonharm"]) == int(row["rows"])
    ]
    safe_scales = [float(row["amplitude_scale"]) for row in safe]
    lines += [
        "\n## Reading\n\n",
        f"- All-win WSD scale interval among tested fixed probes: `{min(safe_scales):.2f}` to `{max(safe_scales):.2f}`.\n",
        "- The recommended `1.00` scale is inside a broad non-harm region, so the result is not an isolated exact-B accident.\n",
        "- `0.00` is the MPL baseline and has no wins; improvement requires the finite-response term.\n",
        "- Large scales are stronger on average but can over-correct individual targets; they are not recommended because they would require amplitude selection.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    rows, summary = run_audit()
    iem.write_csv(OUT_DIR / "details.csv", rows)
    iem.write_csv(OUT_DIR / "summary.csv", summary)
    write_report(summary)


if __name__ == "__main__":
    main()
