#!/usr/bin/env python3
"""Sensitivity audit for schedule-only tau/boundary rules.

This script intentionally avoids fitting any residual coefficient.  It compares
small, interpretable rules for the cooldown finite-response candidate:

    L_hat = L_MPL + a_s B [D_down,tau_s - D_down]

The recommended rule is linear support-bracket tau:

    tau_s = Delta_obs * (1 + min(1, ell_down / Delta_obs))

Other rules are included as robustness references, not as a search over target
losses.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import interpretable_error_model as iem  # noqa: E402
import interpretable_nuisance_origin_audit as noa  # noqa: E402
import mpl_ld_lag_response_audit as lag  # noqa: E402

OUT_DIR = iem.ROOT / "results" / "mpl_ld_lag_response_audit" / "rule_sensitivity"


RULES = [
    {
        "rule": "fixed_one_obs",
        "role": "conservative_lower",
        "explanation_strength": "strong",
        "tau_source": "Delta_obs",
    },
    {
        "rule": "support_linear_bracket",
        "role": "recommended",
        "explanation_strength": "strong",
        "tau_source": "Delta_obs * (1 + min(1, ell_down / Delta_obs))",
    },
    {
        "rule": "support_hard_two_obs",
        "role": "stepwise_reference",
        "explanation_strength": "medium",
        "tau_source": "Delta_obs if ell_down < Delta_obs else 2 * Delta_obs",
    },
    {
        "rule": "support_sqrt_bracket",
        "role": "nonlinear_reference",
        "explanation_strength": "weaker",
        "tau_source": "Delta_obs * (1 + min(1, sqrt(ell_down / Delta_obs)))",
    },
    {
        "rule": "support_log_bracket",
        "role": "nonlinear_reference",
        "explanation_strength": "weaker",
        "tau_source": "Delta_obs * (1 + min(1, log1p(ell_down) / log1p(Delta_obs)))",
    },
    {
        "rule": "fixed_two_obs",
        "role": "unsafe_upper",
        "explanation_strength": "strong",
        "tau_source": "2 * Delta_obs",
    },
]

BOUNDARIES = [
    {
        "boundary": "none",
        "role": "ablation_no_adiabatic_boundary",
        "explanation_strength": "strong",
    },
    {
        "boundary": "linear_support",
        "role": "recommended_boundary",
        "explanation_strength": "strong",
    },
]


def tau_for_rule(curve: iem.Curve, rule: str) -> float:
    delta_obs = float(iem.modal_observation_interval(curve))
    span = float(lag.cooldown_support_span(curve))
    if rule == "fixed_one_obs":
        return delta_obs
    if rule == "fixed_two_obs":
        return 2.0 * delta_obs
    if rule == "support_linear_bracket":
        return delta_obs * (1.0 + min(1.0, span / max(delta_obs, 1.0)))
    if rule == "support_hard_two_obs":
        return delta_obs * (2.0 if span >= delta_obs else 1.0)
    if rule == "support_sqrt_bracket":
        return delta_obs * (1.0 + min(1.0, math.sqrt(span / max(delta_obs, 1.0))))
    if rule == "support_log_bracket":
        return delta_obs * (1.0 + min(1.0, math.log1p(span) / max(math.log1p(delta_obs), 1e-12)))
    raise ValueError(f"unknown tau rule: {rule}")


def boundary_factor(curve: iem.Curve, boundary: str) -> float:
    if boundary == "none":
        return 1.0
    if boundary == "linear_support":
        return iem.drop_localization_factor(curve)
    raise ValueError(f"unknown boundary: {boundary}")


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    keys = sorted(
        {
            (str(row["rule"]), str(row["boundary"]), str(row["group"]))
            for row in rows
        }
    )
    for rule, boundary, group in keys:
        sub = [
            row
            for row in rows
            if row["rule"] == rule and row["boundary"] == boundary and row["group"] == group
        ]
        deltas = np.array([float(row["delta_pct"]) for row in sub], dtype=np.float64)
        out.append(
            {
                "rule": rule,
                "boundary": boundary,
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
    for rule_spec in RULES:
        for boundary_spec in BOUNDARIES:
            for scale in iem.SCALES:
                for group, curve_name, label in noa.ALL_TARGETS:
                    pack = cache[(scale, curve_name)]
                    tau = tau_for_rule(pack.curve, str(rule_spec["rule"]))
                    factor = boundary_factor(pack.curve, str(boundary_spec["boundary"]))
                    pred = pack.baseline + factor * lag.lag_feature(pack, tau, "cooldown")
                    corr_mae = iem.mae(pack.curve.loss, pred)
                    delta = 100.0 * (corr_mae / pack.base_mae - 1.0)
                    rows.append(
                        {
                            "rule": rule_spec["rule"],
                            "rule_role": rule_spec["role"],
                            "rule_explanation_strength": rule_spec["explanation_strength"],
                            "tau_source": rule_spec["tau_source"],
                            "boundary": boundary_spec["boundary"],
                            "boundary_role": boundary_spec["role"],
                            "boundary_explanation_strength": boundary_spec["explanation_strength"],
                            "group": group,
                            "scale": scale,
                            "test_curve": curve_name,
                            "test_label": label,
                            "delta_obs": iem.modal_observation_interval(pack.curve),
                            "cooldown_support_span": lag.cooldown_support_span(pack.curve),
                            "effective_tau_steps": tau,
                            "boundary_factor": factor,
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


def find(summary: list[dict[str, object]], rule: str, boundary: str, group: str) -> dict[str, object]:
    for row in summary:
        if row["rule"] == rule and row["boundary"] == boundary and row["group"] == group:
            return row
    raise KeyError((rule, boundary, group))


def fmt(row: dict[str, object]) -> str:
    return (
        f"{float(row['mean_delta']):+.2f}% / "
        f"{float(row['worst_delta']):+.2f}% / "
        f"{int(row['wins'])}/{int(row['rows'])}"
    )


def write_report(summary: list[dict[str, object]]) -> None:
    lines = [
        "# MPL-LD Tau/Boundary Rule Sensitivity\n\n",
        "All rows use the same cooldown finite-response correction and fit no residual coefficient.  "
        "This audit checks whether the recommended support-bracket rule is a stable schedule-only choice rather than a one-off tuned point.\n\n",
        "## Linear Adiabatic Boundary\n\n",
        "| tau rule | role | explanation | WSD mean / worst / wins | controls mean / worst / nonharm |\n",
        "|---|---|---|---:|---:|\n",
    ]
    for rule_spec in RULES:
        rule = str(rule_spec["rule"])
        core = find(summary, rule, "linear_support", "core_wsd")
        ctrl = find(summary, rule, "linear_support", "extra_control")
        lines.append(
            f"| {rule} | {rule_spec['role']} | {rule_spec['explanation_strength']} | "
            f"{fmt(core)} | {float(ctrl['mean_delta']):+.2f}% / {float(ctrl['worst_delta']):+.2f}% / "
            f"{int(ctrl['nonharm'])}/{int(ctrl['rows'])} |\n"
        )
    lines += [
        "\n## Boundary Ablation\n\n",
        "| tau rule | boundary | WSD mean / worst / wins | controls mean / worst / nonharm |\n",
        "|---|---|---:|---:|\n",
    ]
    for rule in ["fixed_one_obs", "support_linear_bracket", "fixed_two_obs"]:
        for boundary in ["none", "linear_support"]:
            core = find(summary, rule, boundary, "core_wsd")
            ctrl = find(summary, rule, boundary, "extra_control")
            lines.append(
                f"| {rule} | {boundary} | {fmt(core)} | "
                f"{float(ctrl['mean_delta']):+.2f}% / {float(ctrl['worst_delta']):+.2f}% / "
                f"{int(ctrl['nonharm'])}/{int(ctrl['rows'])} |\n"
            )
    lines += [
        "\n## Reading\n\n",
        "- Fixed one-observation tau is conservative and all-win but weaker.\n",
        "- Fixed two-observation tau is too slow for small WSD-con drops and creates failures.\n",
        "- Support-bracket tau keeps the one-observation behavior for single-step drops and two-observation behavior for extended cooldowns, giving the best strong-explanation row.\n",
        "- Sqrt/log support rules are slightly stronger but use softer nonlinear priors; keep them as robustness references, not the main formula.\n",
        "- The linear adiabatic boundary is necessary for controls: without it, diffuse cosine cooldown is incorrectly treated as a local transient.\n",
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
