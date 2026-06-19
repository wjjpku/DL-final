#!/usr/bin/env python3
"""Pareto audit for step-time residual estimators.

The active modeling question has two axes:

1. same-curve explanatory power (self-fit);
2. target-holdout or stricter generalization.

This audit keeps those axes separate and also records whether a result is a
deployment rule, a diagnostic, or a retrospective ceiling.  It deliberately
does not edit paper or slides.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

TMP_CACHE = Path(os.environ.get("TMPDIR", "/tmp")) / "dl-final-cache"
TMP_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(TMP_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(TMP_CACHE / "xdg"))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent
ROOT = REPO.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from reproduce_cosine_to_wsd import SCALES  # noqa: E402
from step_time_decomposed_estimator import summarize  # noqa: E402
from step_time_shape_routed_estimator import (  # noqa: E402
    CORE_CURVES,
    build_cache,
    fit_kappa,
    route_for_target,
    schedule_stats,
    score_target,
)


OUT_DIR = ROOT / "results" / "step_time_pareto_audit"
FIG_DIR = OUT_DIR / "figs"


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


def fmt_pct(value: object) -> str:
    if value in {"", None}:
        return ""
    return f"{float(value):+.1f}%"


def row_by(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row[key] == value:
            return row
    raise KeyError(value)


def model_pareto_rows() -> list[dict[str, object]]:
    complexity = read_csv(ROOT / "results" / "step_time_model_complexity" / "variant_summary.csv")
    error_agg = read_csv(ROOT / "results" / "step_time_minimal_estimator" / "error_comparison" / "aggregate_metrics.csv")
    geometry = read_csv(ROOT / "results" / "step_time_geometry_tau" / "comparison.csv")
    core_error = row_by(error_agg, "group", "core")

    minimal = row_by(complexity, "mode", "shape_routed_no_nuisance")
    residualized = row_by(complexity, "mode", "shape_routed_residualized")
    self_min = row_by(complexity, "mode", "self_fit_no_nuisance")
    decomp = row_by(complexity, "mode", "decomposed_self_fit_reference")
    cross = row_by(complexity, "mode", "cross_family_residualized")
    strict = row_by(complexity, "mode", "cross_family_no_nuisance_p3")
    geometry_minimal = row_by(geometry, "model", "geometry_no_nuisance")
    geometry_residualized = row_by(geometry, "model", "geometry_shape_routed")
    geometry_cross = row_by(geometry, "model", "geometry_cross_family")
    geometry_self = row_by(geometry, "model", "geometry_self_fit_no_nuisance")

    return [
        {
            "model": "minimal_one_kappa",
            "role": "deployment_candidate",
            "self_fit_mean_delta": self_min["mean_delta"],
            "self_fit_worst_delta": self_min["worst_delta"],
            "generalization_mean_delta": minimal["mean_delta"],
            "generalization_worst_delta": minimal["worst_delta"],
            "nonharm": minimal["nonharm"],
            "target_residual_used": "no",
            "same_family_source_allowed": "yes",
            "loss_fitted_parameters": "1 kappa",
            "nuisance_coefficients": "none",
            "reading": "cleanest transferable model; strongest interpretability/overfit-control tradeoff",
        },
        {
            "model": "geometry_tau_one_kappa",
            "role": "deployment_candidate",
            "self_fit_mean_delta": geometry_self["mean_delta"],
            "self_fit_worst_delta": geometry_self["worst_delta"],
            "generalization_mean_delta": geometry_minimal["mean_delta"],
            "generalization_worst_delta": geometry_minimal["worst_delta"],
            "nonharm": geometry_minimal["nonharm"],
            "target_residual_used": "no",
            "same_family_source_allowed": "yes",
            "loss_fitted_parameters": "1 kappa; tau from LR geometry",
            "nuisance_coefficients": "none",
            "reading": "same one-kappa rule with fewer route-specific tau constants and slightly stronger strict worst-case behavior",
        },
        {
            "model": "shape_routed_residualized",
            "role": "strong_deployment_candidate",
            "self_fit_mean_delta": "",
            "self_fit_worst_delta": "",
            "generalization_mean_delta": residualized["mean_delta"],
            "generalization_worst_delta": residualized["worst_delta"],
            "nonharm": residualized["nonharm"],
            "target_residual_used": "no",
            "same_family_source_allowed": "yes",
            "loss_fitted_parameters": "1 kappa plus projection during calibration",
            "nuisance_coefficients": "projected, not transferred",
            "reading": "best current internal target-holdout result; projection reduces smooth-drift contamination",
        },
        {
            "model": "geometry_tau_residualized",
            "role": "strong_deployment_candidate",
            "self_fit_mean_delta": "",
            "self_fit_worst_delta": "",
            "generalization_mean_delta": geometry_residualized["mean_delta"],
            "generalization_worst_delta": geometry_residualized["worst_delta"],
            "nonharm": geometry_residualized["nonharm"],
            "target_residual_used": "no",
            "same_family_source_allowed": "yes",
            "loss_fitted_parameters": "1 kappa plus projection; tau from LR geometry",
            "nuisance_coefficients": "projected, not transferred",
            "reading": "matches the table-tau residualized result while replacing route tau constants with a geometry formula",
        },
        {
            "model": "cross_family_residualized",
            "role": "generalization_audit",
            "self_fit_mean_delta": "",
            "self_fit_worst_delta": "",
            "generalization_mean_delta": cross["mean_delta"],
            "generalization_worst_delta": cross["worst_delta"],
            "nonharm": cross["nonharm"],
            "target_residual_used": "no",
            "same_family_source_allowed": "no",
            "loss_fitted_parameters": "1 kappa plus projection during calibration",
            "nuisance_coefficients": "projected, not transferred",
            "reading": "removes same-family calibration; keeps most of the deployment gain",
        },
        {
            "model": "geometry_tau_cross_family",
            "role": "generalization_audit",
            "self_fit_mean_delta": "",
            "self_fit_worst_delta": "",
            "generalization_mean_delta": geometry_cross["mean_delta"],
            "generalization_worst_delta": geometry_cross["worst_delta"],
            "nonharm": geometry_cross["nonharm"],
            "target_residual_used": "no",
            "same_family_source_allowed": "no",
            "loss_fitted_parameters": "1 kappa plus projection; tau from LR geometry",
            "nuisance_coefficients": "projected, not transferred",
            "reading": "best current no-same-family audit and uses target LR geometry for response time",
        },
        {
            "model": "strict_cross_family_minimal",
            "role": "strict_audit",
            "self_fit_mean_delta": "",
            "self_fit_worst_delta": "",
            "generalization_mean_delta": strict["mean_delta"],
            "generalization_worst_delta": strict["worst_delta"],
            "nonharm": strict["nonharm"],
            "target_residual_used": "no",
            "same_family_source_allowed": "no",
            "loss_fitted_parameters": "1 kappa",
            "nuisance_coefficients": "none",
            "reading": "strictest current no-nuisance/no-same-family check; useful but conservative",
        },
        {
            "model": "decomposed_self_fit",
            "role": "diagnostic_self_fit",
            "self_fit_mean_delta": decomp["mean_delta"],
            "self_fit_worst_delta": decomp["worst_delta"],
            "generalization_mean_delta": "-14.8",
            "generalization_worst_delta": "0.0",
            "nonharm": "90/90 offdiag",
            "target_residual_used": "yes for same-curve diagnostic",
            "same_family_source_allowed": "diagnostic_matrix",
            "loss_fitted_parameters": "kappa plus low-frequency coefficients",
            "nuisance_coefficients": "fitted diagnostic component",
            "reading": "best explanation of residual plots; not the headline transferable mechanism",
        },
        {
            "model": "old_samefit_s_time",
            "role": "diagnostic_samefit_reference",
            "self_fit_mean_delta": core_error["old_mean_delta"],
            "self_fit_worst_delta": core_error["old_worst_delta"],
            "generalization_mean_delta": "",
            "generalization_worst_delta": "",
            "nonharm": f"{core_error['old_nonharm']}/{core_error['rows']}",
            "target_residual_used": "yes",
            "same_family_source_allowed": "not a transfer rule",
            "loss_fitted_parameters": "1 target amplitude per panel",
            "nuisance_coefficients": "none",
            "reading": "same-target shape diagnostic only; useful foil for minimal holdout",
        },
    ]


def route_tau_rows() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    cache = build_cache(CORE_CURVES)
    stats = schedule_stats(cache, CORE_CURVES)
    routes = {target: route_for_target(stats, target) for target, _ in CORE_CURVES}
    multipliers = [0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5, 1.75, 2.0]
    route_names = sorted({str(route["route"]) for route in routes.values()})

    def score_choice(choice: dict[str, float], scales: list[str] | tuple[str, ...]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for target_curve, target_label in CORE_CURVES:
            route = routes[target_curve]
            route_name = str(route["route"])
            mult = choice.get(route_name, 1.0)
            base_tau = float(route["tau"])
            tau = base_tau * mult if base_tau > 0.0 else 0.0
            train_curves = tuple(route["train_curves"])
            nuisance = str(route["nuisance"])
            for scale in scales:
                if train_curves and tau > 0.0:
                    kappa = fit_kappa(cache, scale, train_curves, tau, nuisance)
                else:
                    kappa = 0.0
                scored = score_target(cache, scale, target_curve, kappa, tau)
                rows.append(
                    {
                        "scale": scale,
                        "target_curve": target_curve,
                        "target_label": target_label,
                        "route": route_name,
                        "base_tau": base_tau,
                        "tau_multiplier": mult,
                        "tau": tau,
                        "train_curves": "+".join(c.replace(".csv", "") for c in train_curves)
                        if train_curves
                        else "none",
                        "nuisance": nuisance,
                        **scored,
                    }
                )
        return rows

    per_route: list[dict[str, object]] = []
    for route_name in route_names:
        for mult in multipliers:
            choice = {name: 1.0 for name in route_names}
            choice[route_name] = mult
            rows = [row for row in score_choice(choice, SCALES) if row["route"] == route_name]
            s = summarize(rows)
            per_route.append(
                {
                    "route": route_name,
                    "tau_multiplier": mult,
                    "mean_delta": s["mean_delta"],
                    "worst_delta": s["worst_delta"],
                    "nonharm": f"{int(s['nonharm'])}/{int(s['rows'])}",
                    "rows": s["rows"],
                }
            )

    def choose(scales: list[str] | tuple[str, ...], objective: str) -> dict[str, float]:
        choice: dict[str, float] = {}
        for route_name in route_names:
            candidates = []
            for mult in multipliers:
                rows = [
                    row
                    for row in score_choice({route_name: mult}, scales)
                    if row["route"] == route_name
                ]
                s = summarize(rows)
                objective_value = s["mean_delta"] if objective == "mean" else s["worst_delta"]
                candidates.append(
                    (
                        0 if s["nonharm"] == s["rows"] else 1,
                        objective_value,
                        s["worst_delta"],
                        mult,
                    )
                )
            choice[route_name] = sorted(candidates)[0][3]
        return choice

    current_choice = {name: 1.0 for name in route_names}
    best_mean_choice = choose(SCALES, "mean")
    best_worst_choice = choose(SCALES, "worst")

    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    for name, choice, protocol in [
        ("current_shape_routed", current_choice, "fixed current route taus"),
        ("retrospective_best_mean", best_mean_choice, "selected on all core targets/scales"),
        ("retrospective_best_worst", best_worst_choice, "selected on all core targets/scales"),
    ]:
        rows = score_choice(choice, SCALES)
        for row in rows:
            row["audit"] = name
        detail_rows.extend(rows)
        s = summarize(rows)
        summary_rows.append(
            {
                "audit": name,
                "protocol": protocol,
                "choice": "; ".join(f"{route}={choice[route]:g}" for route in route_names),
                "mean_delta": s["mean_delta"],
                "worst_delta": s["worst_delta"],
                "nonharm": f"{int(s['nonharm'])}/{int(s['rows'])}",
                "rows": s["rows"],
            }
        )

    for objective in ["mean", "worst"]:
        heldout_rows: list[dict[str, object]] = []
        choices = []
        for heldout_scale in SCALES:
            train_scales = [scale for scale in SCALES if scale != heldout_scale]
            choice = choose(train_scales, objective)
            choices.append(f"holdout_{heldout_scale}: " + "; ".join(f"{r}={choice[r]:g}" for r in route_names))
            rows = score_choice(choice, [heldout_scale])
            for row in rows:
                row["audit"] = f"leave_one_scale_select_{objective}"
                row["heldout_scale"] = heldout_scale
            heldout_rows.extend(rows)
        detail_rows.extend(heldout_rows)
        s = summarize(heldout_rows)
        summary_rows.append(
            {
                "audit": f"leave_one_scale_select_{objective}",
                "protocol": f"select route multipliers on two scales by {objective}, evaluate the held-out scale",
                "choice": " | ".join(choices),
                "mean_delta": s["mean_delta"],
                "worst_delta": s["worst_delta"],
                "nonharm": f"{int(s['nonharm'])}/{int(s['rows'])}",
                "rows": s["rows"],
            }
        )

    return summary_rows, detail_rows, per_route


def plot_route_tau_summary(summary_rows: list[dict[str, object]], path: Path) -> None:
    labels = [str(row["audit"]).replace("_", "\n") for row in summary_rows]
    means = np.array([float(row["mean_delta"]) for row in summary_rows])
    worst = np.array([float(row["worst_delta"]) for row in summary_rows])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.4, 4.7), constrained_layout=True)
    ax.axhline(0.0, color="#111111", lw=0.9)
    bars = ax.bar(x, means, width=0.62, color="#2563eb", label="mean")
    ax.scatter(x, worst, color="#dc2626", zorder=3, label="worst")
    ax.set_xticks(x, labels)
    ax.set_ylabel("MAE change vs MPL (%)")
    ax.set_title("Route-level tau refinement: ceiling vs held-out scale")
    ax.grid(axis="y", alpha=0.24)
    ax.legend(frameon=False)
    for bar in bars:
        value = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.2,
            f"{value:+.1f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="white" if value < -8.0 else "#111111",
        )
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def write_report(pareto: list[dict[str, object]], tau_summary: list[dict[str, object]]) -> None:
    by_model = {str(row["model"]): row for row in pareto}
    by_tau = {str(row["audit"]): row for row in tau_summary}
    lines = [
        "# Step-Time Pareto Audit\n\n",
        "This audit organizes the current residual-model evidence by the two axes that matter here: self-fit and generalization.  It also checks whether route-level tau tuning has meaningful unused headroom.\n\n",
        "## Main Takeaway\n\n",
        f"- Best same-curve explanation is still `decomposed_self_fit`: self-fit mean `{fmt_pct(by_model['decomposed_self_fit']['self_fit_mean_delta'])}`, worst `{fmt_pct(by_model['decomposed_self_fit']['self_fit_worst_delta'])}`.  This explains the residual figures but is not the transferable headline rule.\n",
        f"- Clean one-kappa deployment gives generalization mean `{fmt_pct(by_model['minimal_one_kappa']['generalization_mean_delta'])}`, worst `{fmt_pct(by_model['minimal_one_kappa']['generalization_worst_delta'])}`.\n",
        f"- Geometry-tau one-kappa deployment gives mean `{fmt_pct(by_model['geometry_tau_one_kappa']['generalization_mean_delta'])}`, worst `{fmt_pct(by_model['geometry_tau_one_kappa']['generalization_worst_delta'])}` while replacing route-specific tau constants with an LR-geometry formula.\n",
        f"- Strong residualized deployment gives generalization mean `{fmt_pct(by_model['shape_routed_residualized']['generalization_mean_delta'])}`, worst `{fmt_pct(by_model['shape_routed_residualized']['generalization_worst_delta'])}`.\n",
        f"- Geometry-tau no-same-family residualized audit gives mean `{fmt_pct(by_model['geometry_tau_cross_family']['generalization_mean_delta'])}`, worst `{fmt_pct(by_model['geometry_tau_cross_family']['generalization_worst_delta'])}`; this is the best current evidence against pure same-family transfer.\n",
        f"- Retrospective route-tau tuning only raises the internal target-holdout mean from `{fmt_pct(by_tau['current_shape_routed']['mean_delta'])}` to `{fmt_pct(by_tau['retrospective_best_mean']['mean_delta'])}`.  Leave-one-scale selection falls back to `{fmt_pct(by_tau['leave_one_scale_select_mean']['mean_delta'])}`, so this tuning is not strong enough to replace the current fixed route taus.\n\n",
        "## Model Table\n\n",
        "| model | role | self mean | self worst | generalization mean | generalization worst | non-harm | target residual? | parameters | reading |\n",
        "|---|---|---:|---:|---:|---:|---:|---|---|---|\n",
    ]
    for row in pareto:
        lines.append(
            f"| {row['model']} | {row['role']} | {fmt_pct(row['self_fit_mean_delta'])} | "
            f"{fmt_pct(row['self_fit_worst_delta'])} | {fmt_pct(row['generalization_mean_delta'])} | "
            f"{fmt_pct(row['generalization_worst_delta'])} | {row['nonharm']} | "
            f"{row['target_residual_used']} | {row['loss_fitted_parameters']} | {row['reading']} |\n"
        )

    lines += [
        "\n## Route-Tau Refinement Check\n\n",
        "The tau-refinement search is deliberately marked as retrospective.  It only changes route-level multipliers and does not add a new residual basis or extra fitted loss parameter.\n\n",
        "![route tau audit](figs/route_tau_refinement_audit.png)\n\n",
        "| audit | mean | worst | non-harm | protocol |\n",
        "|---|---:|---:|---:|---|\n",
    ]
    for row in tau_summary:
        lines.append(
            f"| {row['audit']} | {fmt_pct(row['mean_delta'])} | {fmt_pct(row['worst_delta'])} | "
            f"{row['nonharm']} | {row['protocol']} |\n"
        )
    lines += [
        "\n## Decision\n\n",
        "- Do not promote route-level tau refinement to the main model yet.  Its all-data ceiling is only about one MAE point better than the current shape-routed head, and the leave-one-scale audit does not preserve that mean gain.\n",
        "- Geometry tau is the cleaner default for future frozen-rule validation because it preserves the current gains while replacing several route-specific tau constants with an LR-shape formula.\n",
        "- The next modeling work should target genuinely new evidence, not more benchmark-shaped tau choices: either an external schedule family or a new training run.\n",
        "- For the current artifact set, keep three distinct claims: decomposed self-fit explains the residual images; geometry/minimal one-kappa is the clean transferable rule; residualized/cross-family audits show the clean rule is not relying only on same-family leakage.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pareto = model_pareto_rows()
    tau_summary, tau_details, per_route = route_tau_rows()
    write_csv(OUT_DIR / "model_pareto.csv", pareto)
    write_csv(OUT_DIR / "route_tau_summary.csv", tau_summary)
    write_csv(OUT_DIR / "route_tau_details.csv", tau_details)
    write_csv(OUT_DIR / "route_tau_per_route.csv", per_route)
    plot_route_tau_summary(tau_summary, FIG_DIR / "route_tau_refinement_audit.png")
    write_report(pareto, tau_summary)
    current = row_by([{k: str(v) for k, v in row.items()} for row in tau_summary], "audit", "current_shape_routed")
    ceiling = row_by([{k: str(v) for k, v in row.items()} for row in tau_summary], "audit", "retrospective_best_mean")
    loo = row_by([{k: str(v) for k, v in row.items()} for row in tau_summary], "audit", "leave_one_scale_select_mean")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    print(
        "route tau: current "
        f"{float(current['mean_delta']):+.1f}%/{float(current['worst_delta']):+.1f}%, "
        "retrospective "
        f"{float(ceiling['mean_delta']):+.1f}%/{float(ceiling['worst_delta']):+.1f}%, "
        "leave-one-scale "
        f"{float(loo['mean_delta']):+.1f}%/{float(loo['worst_delta']):+.1f}%"
    )


if __name__ == "__main__":
    main()
