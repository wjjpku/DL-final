#!/usr/bin/env python3
"""Validate the image-driven step-time error model artifacts.

This is a lightweight consistency gate for the current error-modeling line.
It checks the committed CSV/report artifacts produced by:

  python3 repro/step_time_decomposed_estimator.py
  python3 repro/step_time_shape_routed_estimator.py

The goal is to catch metric drift in the two quantities that matter for this
thread: same-curve self-fit and cross-schedule generalization.
"""
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DECOMP_DIR = ROOT / "results" / "step_time_decomposed_estimator"
ROUTED_DIR = ROOT / "results" / "step_time_shape_routed_estimator"
CROSS_FAMILY_DIR = ROOT / "results" / "step_time_cross_family_estimator"
COMPLEXITY_DIR = ROOT / "results" / "step_time_model_complexity"
MINIMAL_DIR = ROOT / "results" / "step_time_minimal_estimator"
ERROR_COMPARISON_DIR = MINIMAL_DIR / "error_comparison"
GEOMETRY_DIR = ROOT / "results" / "step_time_geometry_tau"
GEOMETRY_ERROR_DIR = GEOMETRY_DIR / "error_comparison"
FROZEN_MODEL_CARD = GEOMETRY_DIR / "FROZEN_MODEL_CARD.md"
FROZEN_MODEL_DIR = ROOT / "results" / "frozen_step_time_model"
PARETO_DIR = ROOT / "results" / "step_time_pareto_audit"
FINAL_DELIVERABLES = ROOT / "FINAL_DELIVERABLES.md"
PAPER = ROOT / "paper" / "main.tex"
SLIDES_ZH = ROOT / "slides" / "main_zh.tex"
SLIDES_EN = ROOT / "slides" / "main.tex"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise AssertionError(f"missing CSV: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_text(path: Path) -> str:
    if not path.exists():
        raise AssertionError(f"missing text file: {path}")
    return path.read_text(encoding="utf-8")


def summarize(rows: list[dict[str, str]]) -> dict[str, float | int]:
    if not rows:
        raise AssertionError("cannot summarize an empty row set")
    deltas = [float(row["delta_pct"]) for row in rows]
    return {
        "rows": len(rows),
        "mean": sum(deltas) / len(deltas),
        "worst": max(deltas),
        "nonharm": sum(delta <= 1e-10 for delta in deltas),
    }


def row_by(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    matches = [row for row in rows if row[key] == value]
    if len(matches) != 1:
        raise AssertionError(f"expected exactly one row where {key}={value}, got {len(matches)}")
    return matches[0]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def require_contains(text: str, needle: str, path: Path) -> None:
    if needle not in text:
        raise AssertionError(f"{path} does not contain expected text: {needle}")


def validate_decomposed() -> None:
    details = read_csv(DECOMP_DIR / "single_details.csv")
    long_rows = read_csv(DECOMP_DIR / "long_probe_to_wsd.csv")
    report = read_text(DECOMP_DIR / "REPORT.md")

    self_fit = summarize([row for row in details if row["train_curve"] == row["test_curve"]])
    offdiag = summarize([row for row in details if row["train_curve"] != row["test_curve"]])
    long = summarize(long_rows)

    require(self_fit["rows"] == 18, "decomposed self-fit should contain 18 scale-target rows")
    require(self_fit["nonharm"] == 18, "decomposed self-fit should be 18/18 non-harming")
    require(self_fit["mean"] <= -70.0, f"self-fit mean drifted above -70%: {self_fit['mean']:.3f}")
    require(self_fit["worst"] <= -38.0, f"self-fit worst drifted above -38%: {self_fit['worst']:.3f}")

    require(offdiag["rows"] == 90, "decomposed off-diagonal should contain 90 rows")
    require(offdiag["nonharm"] == 90, "decomposed off-diagonal should be 90/90 non-harming")
    require(offdiag["mean"] <= -14.0, f"off-diagonal mean drifted above -14%: {offdiag['mean']:.3f}")
    require(offdiag["worst"] <= 1e-10, f"off-diagonal worst became harmful: {offdiag['worst']:.3f}")

    require(long["rows"] == 6, "long probe-to-WSD head should contain 6 rows")
    require(long["nonharm"] == 6, "long probe-to-WSD head should be 6/6 non-harming")
    require(long["mean"] <= -40.0, f"long probe-to-WSD mean drifted above -40%: {long['mean']:.3f}")
    require(long["worst"] <= -25.0, f"long probe-to-WSD worst drifted above -25%: {long['worst']:.3f}")

    for needle in [
        "Decomposed self-fit: mean `-70.6%`",
        "Decomposed off-diagonal: mean `-14.8%`",
        "Long-memory pooled `probe -> WSD`: `-42.0%`",
    ]:
        require_contains(report, needle, DECOMP_DIR / "REPORT.md")


def validate_shape_routed() -> None:
    comparison = read_csv(ROUTED_DIR / "comparison.csv")
    ablation = read_csv(ROUTED_DIR / "ablation_summary.csv")
    extended_summary = read_csv(ROUTED_DIR / "extended_safety_summary.csv")
    protocol = read_csv(ROUTED_DIR / "protocol_audit.csv")
    overfit_summary = read_csv(ROUTED_DIR / "overfit_risk_summary.csv")
    report = read_text(ROUTED_DIR / "REPORT.md")
    protocol_report = read_text(ROUTED_DIR / "PROTOCOL_AUDIT.md")
    overfit_report = read_text(ROUTED_DIR / "OVERFIT_RISK_AUDIT.md")

    routed = row_by(comparison, "metric", "shape_routed_target_holdout")
    require(float(routed["mean_delta"]) <= -35.0, "shape-routed mean should stay below -35%")
    require(float(routed["worst_delta"]) <= -7.0, "shape-routed worst should stay at or below -7%")
    require(routed["nonharm"] == "18/18", "shape-routed core audit should be 18/18 non-harming")

    final_core = row_by(ablation, "audit", "final_core")
    final_extended = row_by(ablation, "audit", "final_extended")
    no_gate = row_by(ablation, "audit", "no_short_smooth_gate")
    fixed_tau = row_by(ablation, "audit", "fixed_tau_1024")
    tau_075 = row_by(ablation, "audit", "tau_x0.75")
    tau_125 = row_by(ablation, "audit", "tau_x1.25")

    require(float(final_core["mean_delta"]) <= -35.0, "final_core mean should stay below -35%")
    require(float(final_core["worst_delta"]) <= -7.0, "final_core worst should stay at or below -7%")
    require(final_core["nonharm"] == "18", "final_core should be 18/18 non-harming")

    require(float(final_extended["worst_delta"]) <= 1e-10, "extended audit should not harm")
    require(final_extended["nonharm"] == "27", "extended audit should be 27/27 non-harming")

    require(
        float(no_gate["worst_delta"]) >= 100.0,
        "short-smooth gate ablation should expose the cosine_24000 failure",
    )
    require(no_gate["nonharm"] != no_gate["rows"], "short-smooth gate ablation should not be non-harming")

    require(float(fixed_tau["worst_delta"]) > 0.0, "fixed tau=1024 ablation should expose a harming cell")
    require(tau_075["nonharm"] == "18", "tau x0.75 should remain 18/18 non-harming")
    require(tau_125["nonharm"] == "18", "tau x1.25 should remain 18/18 non-harming")

    for target in ["cosine_24000.csv", "constant_24000.csv", "constant_72000.csv"]:
        row = row_by(extended_summary, "target_curve", target)
        require(float(row["worst_delta"]) <= 1e-10, f"{target} safety control should not harm")
        require(row["nonharm"] == "3", f"{target} should be 3/3 non-harming")

    require(len(protocol) == 27, "protocol audit should contain 27 extended target-scale rows")
    require(
        all(row["status"] == "pass" for row in protocol),
        "all protocol audit rows should pass target-loss blindness checks",
    )
    require(
        all(row["target_not_in_train"] == "1" for row in protocol),
        "all protocol audit rows should exclude the target curve from calibration",
    )
    require(
        max(float(row["kappa_abs_diff"]) for row in protocol) <= 1e-12,
        "target residual scramble changed at least one kappa",
    )
    require(
        max(float(row["max_correction_abs_diff"]) for row in protocol) <= 1e-12,
        "target residual scramble changed at least one correction vector",
    )

    for needle in [
        "Shape-routed target-holdout: mean `-36.1%`",
        "Extended all-target audit: mean `-24.1%`, worst `+0.0%`, non-harm `27/27`",
        "| no_short_smooth_gate | -10.7% | +154.0%",
        "| fixed_tau_1024 | -24.4% | +5.5%",
    ]:
        require_contains(report, needle, ROUTED_DIR / "REPORT.md")

    for needle in [
        "Core route-table lock: `6/6`",
        "Extended route-table lock: `9/9`",
        "Target exclusion: `27/27`",
        "max `|delta kappa| = 0.000e+00`",
        "Overall protocol status: `27/27` rows pass",
    ]:
        require_contains(protocol_report, needle, ROUTED_DIR / "PROTOCOL_AUDIT.md")

    risk_by_metric = {row["metric"]: row for row in overfit_summary}
    require(
        risk_by_metric["target_loss_blindness"]["value"].startswith("27/27 pass"),
        "overfit audit should record target-loss blindness as 27/27 pass",
    )
    require(
        "scale_25 mean=-32.4%" in risk_by_metric["scale_slices"]["value"],
        "overfit audit scale slice summary drifted",
    )
    require(
        risk_by_metric["source_family_overlap"]["value"] == "4/6 target routes use a same-family source",
        "overfit audit should preserve source-family overlap warning",
    )
    for needle in [
        "Yes, the current estimator can still overfit the public-curve benchmark.",
        "Model-selection complexity is high",
        "This is target-holdout, not leave-family validation.",
        "Do not claim that it is fully validated on unseen regimes.",
    ]:
        require_contains(overfit_report, needle, ROUTED_DIR / "OVERFIT_RISK_AUDIT.md")


def validate_cross_family() -> None:
    details = read_csv(CROSS_FAMILY_DIR / "target_holdout_details.csv")
    extended = read_csv(CROSS_FAMILY_DIR / "extended_safety_details.csv")
    ablation = read_csv(CROSS_FAMILY_DIR / "ablation_summary.csv")
    routes = read_csv(CROSS_FAMILY_DIR / "route_table.csv")
    report = read_text(CROSS_FAMILY_DIR / "REPORT.md")

    core = summarize(details)
    safety = summarize(extended)
    require(core["rows"] == 18, "cross-family core audit should contain 18 rows")
    require(core["mean"] <= -32.0, f"cross-family mean drifted above -32%: {core['mean']:.3f}")
    require(core["worst"] <= -6.0, f"cross-family worst drifted above -6%: {core['worst']:.3f}")
    require(core["nonharm"] == 18, "cross-family core audit should be 18/18 non-harming")
    require(safety["rows"] == 27, "cross-family extended audit should contain 27 rows")
    require(safety["worst"] <= 1e-10, "cross-family extended audit should not harm")
    require(safety["nonharm"] == 27, "cross-family extended audit should be 27/27 non-harming")

    require(
        all(row["target_family"] not in row["source_families"].split("+") for row in routes),
        "cross-family routes should not use a same-family source",
    )
    no_drop = row_by(ablation, "audit", "no_drop_attenuation")
    linear = row_by(ablation, "audit", "linear_drop_attenuation")
    require(float(no_drop["worst_delta"]) >= 70.0, "no-attenuation ablation should expose a large failure")
    require(float(linear["worst_delta"]) > 0.0, "linear attenuation ablation should expose a harming cell")
    for needle in [
        "Cross-family target-holdout: mean `-32.7%`",
        "Same-family source routes: `0/6`",
        "| no_drop_attenuation | -21.8% | +75.3%",
        "does not replace external validation",
    ]:
        require_contains(report, needle, CROSS_FAMILY_DIR / "REPORT.md")


def validate_complexity_audit() -> None:
    variants = read_csv(COMPLEXITY_DIR / "variant_summary.csv")
    ledger = read_csv(COMPLEXITY_DIR / "parameter_ledger.csv")
    report = read_text(COMPLEXITY_DIR / "REPORT.md")

    by_mode = {row["mode"]: row for row in variants}
    minimal = by_mode["shape_routed_no_nuisance"]
    strict = by_mode["cross_family_no_nuisance_p3"]
    require(float(minimal["mean_delta"]) <= -32.0, "minimal no-nuisance transfer should stay below -32%")
    require(float(minimal["worst_delta"]) <= -0.3, "minimal no-nuisance transfer should remain non-harming")
    require(minimal["nonharm"] == "18/18", "minimal no-nuisance transfer should be 18/18")
    require(float(strict["mean_delta"]) <= -24.0, "strict no-nuisance cross-family audit should stay below -24%")
    require(float(strict["worst_delta"]) <= -3.0, "strict no-nuisance cross-family audit should remain non-harming")
    require(strict["nonharm"] == "18/18", "strict no-nuisance cross-family audit should be 18/18")

    by_component = {row["component"]: row for row in ledger}
    require(by_component["kappa"]["count_per_fit"] == "1", "kappa should remain the only headline fitted parameter")
    require(by_component["kappa"]["headline"] == "yes", "kappa should be the headline parameter")
    require(
        by_component["low-frequency nuisance coefficients"]["headline"] == "no",
        "nuisance coefficients should not be headline parameters",
    )
    for needle in [
        "The core transferable model does not need an interpretable sinusoidal component.",
        "Minimal shape-routed transfer",
        "Parameter Ledger",
        "schedule-only route and tau choices are still model-selection freedom",
    ]:
        require_contains(report, needle, COMPLEXITY_DIR / "REPORT.md")


def validate_minimal_estimator() -> None:
    details = read_csv(MINIMAL_DIR / "target_holdout_details.csv")
    extended = read_csv(MINIMAL_DIR / "extended_safety_details.csv")
    ablation = read_csv(MINIMAL_DIR / "ablation_summary.csv")
    routes = read_csv(MINIMAL_DIR / "route_table.csv")
    report = read_text(MINIMAL_DIR / "REPORT.md")
    decision = read_text(MINIMAL_DIR / "MODEL_DECISION.md")

    core = summarize(details)
    safety = summarize(extended)
    require(core["rows"] == 18, "minimal estimator core audit should contain 18 rows")
    require(core["mean"] <= -32.0, f"minimal estimator mean drifted above -32%: {core['mean']:.3f}")
    require(core["worst"] <= -0.3, f"minimal estimator worst drifted above -0.3%: {core['worst']:.3f}")
    require(core["nonharm"] == 18, "minimal estimator core audit should be 18/18 non-harming")
    require(safety["rows"] == 27, "minimal estimator extended audit should contain 27 rows")
    require(safety["worst"] <= 1e-10, "minimal estimator extended audit should not harm")
    require(safety["nonharm"] == 27, "minimal estimator extended audit should be 27/27 non-harming")
    require(
        all(row["nuisance"] == "none" for row in routes),
        "minimal estimator route table should not use nuisance projection",
    )

    fixed_tau = row_by(ablation, "audit", "fixed_tau_1024")
    no_gate = row_by(ablation, "audit", "no_short_smooth_gate")
    no_atten = row_by(ablation, "audit", "cross_family_no_attenuation")
    strict = row_by(ablation, "audit", "strict_cross_family_p3")
    require(float(fixed_tau["worst_delta"]) > 0.0, "fixed tau ablation should expose a harming cell")
    require(float(no_gate["worst_delta"]) >= 60.0, "no short-smooth gate should expose a large failure")
    require(float(no_atten["worst_delta"]) >= 70.0, "no attenuation should expose a large failure")
    require(float(strict["mean_delta"]) <= -24.0, "strict minimal cross-family audit should remain useful")
    require(strict["nonharm"] == "18", "strict minimal cross-family audit should be 18/18")
    for needle in [
        "Only `kappa` is fitted",
        "Minimal target-holdout: mean `-32.0%`",
        "Strict no-same-family/no-nuisance audit: mean `-24.6%`",
        "does not require a fitted sinusoidal or DCT nuisance component",
    ]:
        require_contains(report, needle, MINIMAL_DIR / "REPORT.md")
    for needle in [
        "Only `kappa` is fitted from calibration loss residuals.",
        "minimal target-holdout | `-32.0%`",
        "comparison is that minimal reaches essentially the same average correction",
        "The next decisive validation is to freeze this minimal rule",
    ]:
        require_contains(decision, needle, MINIMAL_DIR / "MODEL_DECISION.md")


def validate_error_comparison() -> None:
    aggregate = read_csv(ERROR_COMPARISON_DIR / "aggregate_metrics.csv")
    details = read_csv(ERROR_COMPARISON_DIR / "error_metrics.csv")
    report = read_text(ERROR_COMPARISON_DIR / "REPORT.md")

    core = row_by(aggregate, "group", "core")
    extended = row_by(aggregate, "group", "extended")
    safety = row_by(aggregate, "group", "safety_controls")
    require(float(core["minimal_mean_delta"]) <= -32.0, "minimal core error plot mean drifted")
    require(float(core["minimal_worst_delta"]) <= -0.3, "minimal core error plot worst drifted")
    require(core["minimal_nonharm"] == "18", "minimal core error plot should be 18/18 non-harming")
    require(float(core["old_mean_delta"]) <= -32.0, "old same-fit core mean should stay near the plotted reference")
    require(core["old_nonharm"] == "18", "old same-fit core should be 18/18 in this diagnostic")
    require(extended["minimal_nonharm"] == "27", "minimal extended error plot should be 27/27")
    require(safety["minimal_nonharm"] == "9", "minimal safety controls should be 9/9")
    require(
        all(row["minimal_uses_target_residual"] == "0" for row in details),
        "minimal error comparison should not use target residuals",
    )
    require(
        all(row["old_uses_target_residual"] == "1" for row in details),
        "old error comparison should be labeled as target-residual same-fit",
    )
    for scale in ["25", "100", "400"]:
        require((ERROR_COMPARISON_DIR / f"error_comparison_{scale}M.png").exists(), f"missing {scale}M error plot")
    for needle in [
        "`MPL+old` uses the previous cumulative-LR / S-time response feature",
        "`MPL+minimal` is the current one-kappa target-holdout rule",
        "old same-curve feature gives mean `-32.3%`",
        "minimal holdout rule gives mean `-32.0%`",
    ]:
        require_contains(report, needle, ERROR_COMPARISON_DIR / "REPORT.md")


def validate_geometry_tau() -> None:
    comparison = read_csv(GEOMETRY_DIR / "comparison.csv")
    route_table = read_csv(GEOMETRY_DIR / "route_table.csv")
    extended = read_csv(GEOMETRY_DIR / "extended_safety_details.csv")
    scale_slices = read_csv(GEOMETRY_DIR / "scale_slices.csv")
    stability = read_csv(GEOMETRY_DIR / "stability_summary.csv")
    report = read_text(GEOMETRY_DIR / "REPORT.md")

    geometry_shape = row_by(comparison, "model", "geometry_shape_routed")
    geometry_minimal = row_by(comparison, "model", "geometry_no_nuisance")
    geometry_cross = row_by(comparison, "model", "geometry_cross_family")
    geometry_self = row_by(comparison, "model", "geometry_self_fit_no_nuisance")

    require(float(geometry_shape["mean_delta"]) <= -36.1, "geometry residualized mean drifted")
    require(float(geometry_shape["worst_delta"]) <= -7.0, "geometry residualized worst drifted")
    require(geometry_shape["nonharm"] == "18/18", "geometry residualized should be 18/18 non-harming")
    require(float(geometry_minimal["mean_delta"]) <= -32.2, "geometry no-nuisance mean drifted")
    require(float(geometry_minimal["worst_delta"]) <= -1.0, "geometry no-nuisance worst drifted")
    require(geometry_minimal["nonharm"] == "18/18", "geometry no-nuisance should be 18/18 non-harming")
    require(float(geometry_cross["mean_delta"]) <= -33.5, "geometry cross-family mean drifted")
    require(float(geometry_cross["worst_delta"]) <= -6.0, "geometry cross-family worst drifted")
    require(geometry_cross["nonharm"] == "18/18", "geometry cross-family should be 18/18 non-harming")
    require(float(geometry_self["mean_delta"]) <= -40.0, "geometry self-fit mean drifted")

    safety = summarize(extended)
    require(safety["rows"] == 27, "geometry extended safety audit should contain 27 rows")
    require(safety["nonharm"] == 27, "geometry extended safety should be 27/27 non-harming")
    require(safety["worst"] <= 1e-10, "geometry extended safety should not harm")

    finite_tail = row_by(route_table, "target_curve", "wsd_20000_24000.csv")
    medium_step = row_by(route_table, "target_curve", "wsdcon_9.csv")
    require(abs(float(finite_tail["geometry_tau"]) - 4998.75) <= 1e-9, "finite-tail geometry tau drifted")
    require(abs(float(medium_step["geometry_tau"]) - 733.184) <= 1e-6, "medium-step geometry tau drifted")
    require((GEOMETRY_DIR / "figs" / "geometry_tau_comparison.png").exists(), "missing geometry tau comparison figure")

    require(len(scale_slices) == 12, "geometry scale-slice audit should contain 4 models x 3 scales")
    require(
        all(row["nonharm"] == "6/6" for row in scale_slices),
        "all geometry scale slices should stay 6/6 non-harming",
    )

    stability_by_key = {(row["model"], row["variant"]): row for row in stability}
    require(stability_by_key[("residualized", "baseline")]["nonharm"] == "18/18", "residualized baseline unstable")
    require(stability_by_key[("no_nuisance", "baseline")]["nonharm"] == "18/18", "no-nuisance baseline unstable")
    require(stability_by_key[("cross_family", "baseline")]["nonharm"] == "18/18", "cross-family baseline unstable")
    require(
        all(row["nonharm"] == "18/18" for row in stability if row["model"] == "residualized"),
        "residualized geometry tau should be non-harming under all perturbations",
    )
    require(
        all(row["nonharm"] == "18/18" for row in stability if row["model"] == "cross_family"),
        "cross-family geometry tau should be non-harming under all perturbations",
    )
    require(
        stability_by_key[("no_nuisance", "step_base_x1.25")]["nonharm"] == "17/18",
        "no-nuisance large-step-base perturbation should expose one harming cell",
    )
    require(
        stability_by_key[("no_nuisance", "step_power_2")]["nonharm"] == "17/18",
        "no-nuisance low-power perturbation should expose one harming cell",
    )

    for needle in [
        "tau = min(8192, 1.25 * positive_drop_span)",
        "tau = 512 * (1 + 2 q^3)",
        "## Formula Stability",
        "Geometry tau is a better default than the raw discrete tau table",
        "supports keeping the conservative cubic single-step rule",
    ]:
        require_contains(report, needle, GEOMETRY_DIR / "REPORT.md")


def validate_geometry_error_comparison() -> None:
    aggregate = read_csv(GEOMETRY_ERROR_DIR / "aggregate_metrics.csv")
    details = read_csv(GEOMETRY_ERROR_DIR / "error_metrics.csv")
    target_summary = read_csv(GEOMETRY_ERROR_DIR / "target_summary.csv")
    report = read_text(GEOMETRY_ERROR_DIR / "REPORT.md")

    core = row_by(aggregate, "group", "core")
    extended = row_by(aggregate, "group", "extended")
    safety = row_by(aggregate, "group", "safety_controls")
    require(float(core["table_mean_delta"]) <= -32.0, "table-tau core mean drifted")
    require(float(core["geometry_mean_delta"]) <= -32.2, "geometry-tau core mean drifted")
    require(float(core["geometry_worst_delta"]) <= -1.0, "geometry-tau core worst drifted")
    require(core["geometry_nonharm"] == "18", "geometry-tau core residual plot should be 18/18 non-harming")
    require(extended["geometry_nonharm"] == "27", "geometry-tau extended residual plot should be 27/27")
    require(safety["geometry_nonharm"] == "9", "geometry-tau safety residual plot should be 9/9")
    require(
        all(row["uses_target_residual"] == "0" for row in details),
        "geometry residual comparison should not use target residuals",
    )

    wsdcon9 = row_by(target_summary, "curve", "wsdcon_9.csv")
    require(
        float(wsdcon9["geometry_mean_delta"]) < float(wsdcon9["table_mean_delta"]),
        "geometry tau should improve WSD-con 9e-5 mean in residual comparison",
    )
    require(
        float(wsdcon9["geometry_worst_delta"]) < float(wsdcon9["table_worst_delta"]),
        "geometry tau should improve WSD-con 9e-5 worst in residual comparison",
    )

    for scale in ["25", "100", "400"]:
        require((GEOMETRY_ERROR_DIR / f"core_residuals_{scale}M.png").exists(), f"missing {scale}M geometry core plot")
        require(
            (GEOMETRY_ERROR_DIR / f"safety_residuals_{scale}M.png").exists(),
            f"missing {scale}M geometry safety plot",
        )
    require((GEOMETRY_ERROR_DIR / "mae_bar_summary.png").exists(), "missing geometry MAE bar summary")
    for needle in [
        "previous discrete route-tau table and the schedule-geometry tau formula",
        "Core target-holdout changes from table mean `-32.0%`",
        "geometry tau leaves most residual shapes unchanged",
    ]:
        require_contains(report, needle, GEOMETRY_ERROR_DIR / "REPORT.md")


def validate_frozen_model_card() -> None:
    card = read_text(FROZEN_MODEL_CARD)
    for needle in [
        "Use the one-kappa geometry-tau rule as the clean transferable model",
        "L_hat(t) = L_MPL(t) + kappa_hat * phi_tau(t)",
        "tau = min(8192, 1.25 * positive_drop_span)",
        "tau = 512 * (1 + 2 q^3)",
        "| geometry-tau one-kappa | primary transferable rule | `-32.3%` | `-1.5%` | `18/18` | no |",
        "| decomposed self-fit | residual explanation diagnostic | `-70.6%` | `-38.9%` | `18/18` | yes |",
        "| geometry-tau cross-family residualized | no-same-family audit | `-33.8%` | `-6.5%` | `18/18` | no |",
        "Do not claim prospective validation on unseen schedules.",
        "Do not promote retrospective route-tau tuning.",
        "PYTHONDONTWRITEBYTECODE=1 python3 repro/validate_step_time_model.py",
        "repro/frozen_step_time_model.py",
        "Freeze the geometry-tau one-kappa rule as the main transferable model.",
    ]:
        require_contains(card, needle, FROZEN_MODEL_CARD)


def validate_frozen_estimator() -> None:
    details = read_csv(FROZEN_MODEL_DIR / "target_holdout_details.csv")
    self_details = read_csv(FROZEN_MODEL_DIR / "self_fit_details.csv")
    extended = read_csv(FROZEN_MODEL_DIR / "extended_safety_details.csv")
    routes = read_csv(FROZEN_MODEL_DIR / "route_table.csv")
    geometry_details = read_csv(GEOMETRY_DIR / "no_nuisance_details.csv")
    report = read_text(FROZEN_MODEL_DIR / "REPORT.md")

    core = summarize(details)
    self_fit = summarize(self_details)
    safety = summarize(extended)
    require(core["rows"] == 18, "frozen estimator core audit should contain 18 rows")
    require(core["mean"] <= -32.2, "frozen estimator core mean drifted")
    require(core["worst"] <= -1.0, "frozen estimator core worst drifted")
    require(core["nonharm"] == 18, "frozen estimator core should be 18/18 non-harming")
    require(self_fit["rows"] == 18, "frozen self-fit audit should contain 18 rows")
    require(self_fit["mean"] <= -40.0, "frozen self-fit mean drifted")
    require(self_fit["worst"] <= -6.0, "frozen self-fit worst drifted")
    require(safety["rows"] == 27, "frozen extended safety audit should contain 27 rows")
    require(safety["nonharm"] == 27, "frozen extended safety audit should be 27/27 non-harming")
    require(safety["worst"] <= 1e-10, "frozen extended safety audit should not harm")

    require(
        all(row["target_residual_used_for_kappa"] == "0" for row in details),
        "frozen target-holdout must not use target residuals for kappa",
    )
    require(
        all(row["target_residual_used_for_kappa"] == "1" for row in self_details),
        "frozen self-fit diagnostic should be explicitly labeled as using target residuals",
    )
    require(all(row["nuisance"] == "none" for row in routes), "frozen primary route should use no nuisance")

    frozen_by_key = {(row["scale"], row["target_curve"]): row for row in details}
    geometry_by_key = {(row["scale"], row["target_curve"]): row for row in geometry_details}
    require(frozen_by_key.keys() == geometry_by_key.keys(), "frozen and geometry no-nuisance row keys differ")
    for key, row in frozen_by_key.items():
        geometry = geometry_by_key[key]
        for field in ["geometry_tau", "kappa", "delta_pct", "corr_mae"]:
            require(
                abs(float(row[field]) - float(geometry[field])) <= 1e-12,
                f"frozen estimator drifted from geometry no-nuisance for {key} field {field}",
            )

    for needle in [
        "Frozen Geometry-Tau One-Kappa Model",
        "Target-holdout primary rule: mean `-32.3%`, worst `-1.5%`, non-harm `18/18`",
        "Same-curve one-kappa diagnostic: mean `-40.7%`, worst `-6.6%`, non-harm `18/18`",
        "Use this module as the source of truth for the frozen primary transferable rule.",
    ]:
        require_contains(report, needle, FROZEN_MODEL_DIR / "REPORT.md")


def validate_final_deliverables_index() -> None:
    text = read_text(FINAL_DELIVERABLES)
    for needle in [
        "2026-06-18 残差建模冻结补充",
        "repro/frozen_step_time_model.py",
        "target-holdout `-32.3% / -1.5%`, `18/18`",
        "decomposed self-fit `-70.6% / -38.9%`, `18/18`",
        "geometry-tau one-kappa` 是当前最干净的 transferable rule",
        "不能声称已经在未见 schedule 上 prospective validated",
        "PYTHONDONTWRITEBYTECODE=1 python3 repro/validate_step_time_model.py",
    ]:
        require_contains(text, needle, FINAL_DELIVERABLES)


def validate_pareto_audit() -> None:
    pareto = read_csv(PARETO_DIR / "model_pareto.csv")
    tau = read_csv(PARETO_DIR / "route_tau_summary.csv")
    report = read_text(PARETO_DIR / "REPORT.md")

    minimal = row_by(pareto, "model", "minimal_one_kappa")
    geometry_minimal = row_by(pareto, "model", "geometry_tau_one_kappa")
    residualized = row_by(pareto, "model", "shape_routed_residualized")
    geometry_residualized = row_by(pareto, "model", "geometry_tau_residualized")
    cross = row_by(pareto, "model", "cross_family_residualized")
    geometry_cross = row_by(pareto, "model", "geometry_tau_cross_family")
    decomp = row_by(pareto, "model", "decomposed_self_fit")
    old = row_by(pareto, "model", "old_samefit_s_time")

    require(float(minimal["self_fit_mean_delta"]) <= -40.0, "minimal self-fit should stay below -40%")
    require(float(minimal["generalization_mean_delta"]) <= -32.0, "minimal generalization mean drifted")
    require(float(geometry_minimal["generalization_mean_delta"]) <= -32.2, "geometry minimal mean drifted")
    require(float(residualized["generalization_mean_delta"]) <= -36.0, "residualized generalization mean drifted")
    require(float(geometry_residualized["generalization_mean_delta"]) <= -36.1, "geometry residualized mean drifted")
    require(float(cross["generalization_mean_delta"]) <= -32.0, "cross-family generalization mean drifted")
    require(float(geometry_cross["generalization_mean_delta"]) <= -33.5, "geometry cross-family mean drifted")
    require(float(decomp["self_fit_mean_delta"]) <= -70.0, "decomposed self-fit mean drifted")
    require(old["target_residual_used"] == "yes", "old same-fit diagnostic must be labeled target-residual")

    current = row_by(tau, "audit", "current_shape_routed")
    ceiling = row_by(tau, "audit", "retrospective_best_mean")
    loo = row_by(tau, "audit", "leave_one_scale_select_mean")
    require(float(current["mean_delta"]) <= -36.0, "current route-tau baseline drifted")
    require(
        float(ceiling["mean_delta"]) < float(current["mean_delta"]),
        "retrospective route-tau ceiling should beat the current mean",
    )
    require(
        float(loo["mean_delta"]) > float(current["mean_delta"]),
        "leave-one-scale route-tau selection should not be promoted as a mean improvement",
    )
    require(ceiling["nonharm"] == "18/18", "retrospective route-tau ceiling should stay non-harming")
    require(loo["nonharm"] == "18/18", "leave-one-scale route-tau audit should stay non-harming")
    require((PARETO_DIR / "figs" / "route_tau_refinement_audit.png").exists(), "missing route tau audit figure")

    for needle in [
        "Best same-curve explanation is still `decomposed_self_fit`",
        "Geometry-tau one-kappa deployment gives mean",
        "Geometry-tau no-same-family residualized audit gives mean",
        "Retrospective route-tau tuning only raises the internal target-holdout mean",
        "Geometry tau is the cleaner default for future frozen-rule validation",
    ]:
        require_contains(report, needle, PARETO_DIR / "REPORT.md")


def validate_paper_and_slides() -> None:
    paper = read_text(PAPER)
    zh = read_text(SLIDES_ZH)
    en = read_text(SLIDES_EN)

    for needle in [
        "$70.6\\%$ MAE reduction",
        "$36.1\\%$ mean reduction",
        "$+154.0\\%$ MAE failure",
    ]:
        require_contains(paper, needle, PAPER)

    for text, path in [(zh, SLIDES_ZH), (en, SLIDES_EN)]:
        require_contains(text, "$-70.6\\%$", path)
        require_contains(text, "$-36.1\\%$", path)
        require_contains(text, "27/27", path)


def main() -> None:
    validate_decomposed()
    validate_shape_routed()
    validate_cross_family()
    validate_complexity_audit()
    validate_minimal_estimator()
    validate_error_comparison()
    validate_geometry_tau()
    validate_geometry_error_comparison()
    validate_frozen_model_card()
    validate_frozen_estimator()
    validate_final_deliverables_index()
    validate_pareto_audit()
    validate_paper_and_slides()
    print("step-time model artifacts validated")


if __name__ == "__main__":
    main()
