#!/usr/bin/env python3
"""Consistency checks for the interpretable cosine-to-WSD checkpoint."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def close(value: float, expected: float, tol: float = 0.02) -> None:
    if abs(value - expected) > tol:
        raise AssertionError(f"expected {expected}, got {value}")


def find_row(rows: list[dict[str, str]], **match: object) -> dict[str, str]:
    for row in rows:
        if all(str(row[key]) == str(value) for key, value in match.items()):
            return row
    raise AssertionError(f"missing row: {match}")


def must_contain(path: Path, text: str) -> None:
    body = path.read_text(encoding="utf-8")
    if text not in body:
        raise AssertionError(f"{path} does not contain expected text: {text}")


def check_main_summary() -> None:
    rows = read_csv(ROOT / "results/interpretable_error_model/summary.csv")

    sqrt_ablation = find_row(
        rows,
        candidate="obs_half_life_sqrtlocalized_projected_2p5_roundfast20",
        fit_start="8000",
        nuisance_lambda="0.01",
    )
    close(float(sqrt_ablation["mean_delta"]), -34.149)
    close(float(sqrt_ablation["worst_delta"]), -5.298)
    if sqrt_ablation["wins"] != "15" or sqrt_ablation["nonharm"] != "15":
        raise AssertionError(f"sqrt ablation should be 15/15: {sqrt_ablation}")

    upper = find_row(
        rows,
        candidate="obs_half_life_projected_2p5_roundfast20",
        fit_start="8000",
        nuisance_lambda="0.01",
    )
    close(float(upper["mean_delta"]), -34.560)
    close(float(upper["worst_delta"]), -5.296)
    if upper["wins"] != "15" or upper["nonharm"] != "15":
        raise AssertionError(f"upper variant should be 15/15: {upper}")


def check_core_decision() -> None:
    rows = read_csv(ROOT / "results/interpretable_error_model/core_decision_summary.csv")

    fixed = find_row(rows, variant="fixed_lambda_obs", group="core_wsd")
    close(float(fixed["mean_delta"]), -20.553)
    close(float(fixed["worst_delta"]), -1.086)
    if fixed["wins"] != "15" or fixed["nonharm"] != "15":
        raise AssertionError(f"fixed observed half-life should be 15/15: {fixed}")

    strict = find_row(rows, variant="strict_exact", group="core_wsd")
    close(float(strict["mean_delta"]), -31.972)
    close(float(strict["worst_delta"]), -1.086)
    if strict["wins"] != "15" or strict["nonharm"] != "15":
        raise AssertionError(f"strict exact should be 15/15: {strict}")

    rounded = find_row(rows, variant="rounded_fast20", group="core_wsd")
    close(float(rounded["mean_delta"]), -34.560)
    close(float(rounded["worst_delta"]), -5.296)
    if rounded["wins"] != "15" or rounded["nonharm"] != "15":
        raise AssertionError(f"rounded fast20 should be 15/15: {rounded}")

    linear_control = find_row(rows, variant="rounded_fast20_localized", group="extra_control")
    close(float(linear_control["mean_delta"]), 0.0, tol=0.001)
    close(float(linear_control["worst_delta"]), 0.0, tol=0.001)
    if linear_control["nonharm"] != "9":
        raise AssertionError(f"linear locality controls should be 9/9: {linear_control}")


def check_shrinkage_origin() -> None:
    rows = read_csv(ROOT / "results/interpretable_shrinkage_origin_audit/summary.csv")

    hard = find_row(
        rows,
        response_rule="fixed_lambda_20",
        shrinkage="tau_free_sqrt_retention",
        locality="linear",
        group="core_wsd",
    )
    close(float(hard["mean_delta"]), -20.772)
    close(float(hard["worst_delta"]), -5.863)
    if hard["wins"] != "15" or hard["nonharm"] != "15":
        raise AssertionError(f"tau-free hard baseline should be 15/15: {hard}")

    hard_controls = find_row(
        rows,
        response_rule="fixed_lambda_20",
        shrinkage="tau_free_sqrt_retention",
        locality="linear",
        group="extra_control",
    )
    close(float(hard_controls["mean_delta"]), 0.0, tol=0.001)
    close(float(hard_controls["worst_delta"]), 0.0, tol=0.001)
    if hard_controls["nonharm"] != "9":
        raise AssertionError(f"tau-free hard baseline controls should be 9/9: {hard_controls}")

    lower = find_row(
        rows,
        response_rule="fixed_lambda_20",
        shrinkage="tau_free_full_energy",
        locality="linear",
        group="core_wsd",
    )
    close(float(lower["mean_delta"]), -3.716)
    if lower["wins"] != "15" or lower["nonharm"] != "15":
        raise AssertionError(f"tau-free full-energy lower bound should be 15/15: {lower}")


def check_nuisance_origin() -> None:
    rows = read_csv(ROOT / "results/interpretable_nuisance_origin_audit/summary.csv")

    no_nuisance = find_row(
        rows,
        response_rule="fixed_lambda_20",
        nuisance="none",
        shrinkage="tau_free_sqrt_retention",
        locality="linear",
        group="core_wsd",
    )
    close(float(no_nuisance["mean_delta"]), 672.314)
    close(float(no_nuisance["worst_delta"]), 2585.942)
    if no_nuisance["wins"] != "0" or no_nuisance["nonharm"] != "0":
        raise AssertionError(f"raw no-nuisance projection should fail badly: {no_nuisance}")

    tangent_ld = find_row(
        rows,
        response_rule="two_point_five_roundfast20",
        nuisance="mpl_ld4",
        shrinkage="ridge_tau_0p05",
        locality="linear",
        group="core_wsd",
    )
    close(float(tangent_ld["mean_delta"]), -27.250)
    close(float(tangent_ld["worst_delta"]), -3.002)
    if tangent_ld["wins"] != "15" or tangent_ld["nonharm"] != "15":
        raise AssertionError(f"MPL-LD tangent nuisance should be 15/15: {tangent_ld}")

    tangent_ld_controls = find_row(
        rows,
        response_rule="two_point_five_roundfast20",
        nuisance="mpl_ld4",
        shrinkage="ridge_tau_0p05",
        locality="linear",
        group="extra_control",
    )
    close(float(tangent_ld_controls["mean_delta"]), 0.0, tol=0.001)
    if tangent_ld_controls["nonharm"] != "9":
        raise AssertionError(f"MPL-LD tangent controls should be 9/9: {tangent_ld_controls}")

    tangent_ld_no_locality = find_row(
        rows,
        response_rule="two_point_five_roundfast20",
        nuisance="mpl_ld4",
        shrinkage="ridge_tau_0p05",
        locality="none",
        group="core_wsd",
    )
    close(float(tangent_ld_no_locality["mean_delta"]), -24.856)
    if tangent_ld_no_locality["wins"] != "15" or tangent_ld_no_locality["nonharm"] != "15":
        raise AssertionError(f"MPL-LD tangent should help WSD even without locality: {tangent_ld_no_locality}")

    tangent_ld_no_locality_controls = find_row(
        rows,
        response_rule="two_point_five_roundfast20",
        nuisance="mpl_ld4",
        shrinkage="ridge_tau_0p05",
        locality="none",
        group="extra_control",
    )
    close(float(tangent_ld_no_locality_controls["worst_delta"]), 84.551)
    if tangent_ld_no_locality_controls["nonharm"] != "6":
        raise AssertionError(f"no-locality controls should expose the boundary condition: {tangent_ld_no_locality_controls}")

    bad_core = find_row(
        rows,
        response_rule="two_point_five_roundfast20",
        nuisance="mpl_core3",
        shrinkage="ridge_tau_0p05",
        locality="linear",
        group="core_wsd",
    )
    if float(bad_core["worst_delta"]) < 100.0:
        raise AssertionError(f"MPL-core tangent should remain documented negative evidence: {bad_core}")


def check_scale_stability() -> None:
    rows = read_csv(ROOT / "results/interpretable_scale_stability_audit/summary.csv")

    same = find_row(rows, method="mpl_ld_tangent", group="core_wsd", split="same_scale")
    close(float(same["mean_delta"]), -27.250)
    close(float(same["worst_delta"]), -3.002)
    if same["wins"] != "15" or same["nonharm"] != "15":
        raise AssertionError(f"MPL-LD same-scale should be 15/15: {same}")

    cross = find_row(rows, method="mpl_ld_tangent", group="core_wsd", split="cross_scale")
    close(float(cross["mean_delta"]), -23.068)
    close(float(cross["worst_delta"]), -2.071)
    if cross["wins"] != "30" or cross["nonharm"] != "30":
        raise AssertionError(f"MPL-LD cross-scale should be 30/30: {cross}")

    dct_cross = find_row(rows, method="dct_performance", group="core_wsd", split="cross_scale")
    close(float(dct_cross["mean_delta"]), -18.978)
    close(float(dct_cross["worst_delta"]), 26.678)
    if dct_cross["wins"] != "26" or dct_cross["nonharm"] != "26":
        raise AssertionError(f"DCT cross-scale bad cases should be documented: {dct_cross}")

    tau_cross = find_row(rows, method="tau_free_dct", group="core_wsd", split="cross_scale")
    close(float(tau_cross["worst_delta"]), 9.037)
    if tau_cross["wins"] != "27":
        raise AssertionError(f"tau-free DCT cross-scale should document partial failures: {tau_cross}")


def check_observation_bracket() -> None:
    rows = read_csv(ROOT / "results/interpretable_observation_bracket_audit/summary.csv")

    main_same = find_row(
        rows,
        variant="observation_bracket_mplld_neff",
        group="core_wsd",
        split="same_scale",
    )
    close(float(main_same["mean_delta"]), -29.875)
    close(float(main_same["worst_delta"]), -4.665)
    if main_same["wins"] != "15" or main_same["nonharm"] != "15":
        raise AssertionError(f"observation bracket same-scale should be 15/15: {main_same}")

    main_cross = find_row(
        rows,
        variant="observation_bracket_mplld_neff",
        group="core_wsd",
        split="cross_scale",
    )
    close(float(main_cross["mean_delta"]), -24.948)
    close(float(main_cross["worst_delta"]), -3.153)
    if main_cross["wins"] != "30" or main_cross["nonharm"] != "30":
        raise AssertionError(f"observation bracket cross-scale should be 30/30: {main_cross}")

    controls = find_row(
        rows,
        variant="observation_bracket_mplld_neff",
        group="extra_control",
        split="same_scale",
    )
    close(float(controls["mean_delta"]), 0.0, tol=0.001)
    if controls["nonharm"] != "9":
        raise AssertionError(f"observation bracket controls should be 9/9: {controls}")

    no_nuisance = find_row(
        rows,
        variant="observation_bracket_no_nuisance_neff",
        group="core_wsd",
        split="same_scale",
    )
    close(float(no_nuisance["mean_delta"]), 602.171)
    close(float(no_nuisance["worst_delta"]), 2366.348)
    if no_nuisance["wins"] != "0":
        raise AssertionError(f"observation bracket no-nuisance failure should remain documented: {no_nuisance}")

    rule_rows = read_csv(ROOT / "results/interpretable_observation_bracket_audit/fit_start_rule.csv")
    selected = [row for row in rule_rows if row["passes"] == "1"]
    if not selected or selected[0]["fit_start"] != "8000":
        raise AssertionError(f"source-only fit-start rule should select 8000 first: {rule_rows}")
    if selected[0].get("lambda_grid_points") != "2" or selected[0].get("rows") != "6":
        raise AssertionError(f"fit-start rule should check the source-only lambda-bracket endpoints: {selected[0]}")
    row_6500 = find_row(rule_rows, fit_start="6500")
    if row_6500["passes"] != "0":
        raise AssertionError(f"fit start 6500 should fail the source-only retention rule: {row_6500}")

    ledger = read_csv(ROOT / "results/interpretable_observation_bracket_audit/parameter_ledger.csv")
    fitted = [row for row in ledger if row["fitted_in_error_model"] == "1"]
    if [row["quantity"] for row in fitted] != ["kappa_hat_s"]:
        raise AssertionError(f"only kappa_hat_s should be fitted in the error model: {fitted}")
    for row in ledger:
        if row["quantity"] != "MPL parameters" and row["uses_target_loss"] != "0":
            raise AssertionError(f"error-model quantity should not use target loss: {row}")

    locality_rows = read_csv(ROOT / "results/interpretable_observation_bracket_audit/locality_boundary.csv")
    wsd_sharp = find_row(locality_rows, test_curve="wsd_20000_24000.csv")
    close(float(wsd_sharp["median_locality_factor"]), 0.81685, tol=0.001)
    cosine_short = find_row(locality_rows, test_curve="cosine_24000.csv")
    close(float(cosine_short["median_locality_factor"]), 0.0, tol=0.001)
    constant_short = find_row(locality_rows, test_curve="constant_24000.csv")
    close(float(constant_short["median_locality_factor"]), 0.0, tol=0.001)


def check_theory_refinement() -> None:
    rows = read_csv(ROOT / "results/interpretable_theory_refinement/summary.csv")

    main_same = find_row(
        rows,
        variant="hhi_q2_halflife_support_projection",
        group="core_wsd",
        split="same_scale",
    )
    close(float(main_same["mean_delta"]), -29.882)
    close(float(main_same["worst_delta"]), -4.665)
    if main_same["wins"] != "15" or main_same["nonharm"] != "15":
        raise AssertionError(f"q2 half-life same-scale should be 15/15: {main_same}")

    main_cross = find_row(
        rows,
        variant="hhi_q2_halflife_support_projection",
        group="core_wsd",
        split="cross_scale",
    )
    close(float(main_cross["mean_delta"]), -24.948)
    close(float(main_cross["worst_delta"]), -3.153)
    if main_cross["wins"] != "30" or main_cross["nonharm"] != "30":
        raise AssertionError(f"q2 half-life cross-scale should be 30/30: {main_cross}")

    controls = find_row(
        rows,
        variant="hhi_q2_halflife_support_projection",
        group="extra_control",
        split="same_scale",
    )
    close(float(controls["mean_delta"]), 0.0, tol=0.001)
    close(float(controls["worst_delta"]), 0.0, tol=0.001)
    if controls["nonharm"] != "9":
        raise AssertionError(f"q2 half-life controls should be 9/9 non-harm: {controls}")

    density_controls = find_row(
        rows,
        variant="hhi_q2_density_projection",
        group="extra_control",
        split="same_scale",
    )
    close(float(density_controls["worst_delta"]), 8.254)
    if density_controls["nonharm"] != "6":
        raise AssertionError(f"density projection should document control harm: {density_controls}")

    no_locality = find_row(
        rows,
        variant="hhi_q2_no_locality",
        group="extra_control",
        split="same_scale",
    )
    close(float(no_locality["worst_delta"]), 56.993)
    if no_locality["nonharm"] != "6":
        raise AssertionError(f"no-locality q2 should expose control failure: {no_locality}")

    diagnostics = read_csv(ROOT / "results/interpretable_theory_refinement/schedule_diagnostics.csv")
    sharp = find_row(diagnostics, label="WSD sharp")
    close(float(sharp["q2"]), 0.000352, tol=0.00001)
    close(float(sharp["support_projection"]), 0.81685, tol=0.001)
    cosine = find_row(diagnostics, label="Cosine 24k")
    close(float(cosine["support_projection"]), 0.0, tol=0.001)
    if float(cosine["density_projection"]) <= 0.1:
        raise AssertionError(f"cosine density projection should show why density boundary is unsafe: {cosine}")


def check_error_comparison_figures() -> None:
    rows = read_csv(ROOT / "results/interpretable_error_model/error_comparison/aggregate_metrics.csv")
    core = find_row(rows, group="core_wsd")
    close(float(core["observation_bracket_mean_delta"]), -29.875)
    close(float(core["old_mpl_ld_mean_delta"]), -27.250)
    close(float(core["dct_performance_mean_delta"]), -32.826)
    for key in ["observation_bracket", "old_mpl_ld", "dct_performance"]:
        if core[f"{key}_wins"] != "15" or core[f"{key}_nonharm"] != "15":
            raise AssertionError(f"{key} should be all-win in core error comparison: {core}")

    controls = find_row(rows, group="extra_control")
    for key in ["observation_bracket", "old_mpl_ld", "dct_performance"]:
        close(float(controls[f"{key}_mean_delta"]), 0.0, tol=0.001)
        if controls[f"{key}_nonharm"] != "9":
            raise AssertionError(f"{key} controls should be 9/9 non-harm: {controls}")

    fig_dir = ROOT / "results/interpretable_error_model/error_comparison/figs"
    for name in [
        "error_curves_25M.png",
        "error_curves_100M.png",
        "error_curves_400M.png",
        "target_mae_summary.png",
    ]:
        path = fig_dir / name
        if not path.exists() or path.stat().st_size <= 0:
            raise AssertionError(f"missing or empty figure: {path}")


def check_parameter_origin() -> None:
    rows = read_csv(ROOT / "results/interpretable_parameter_origin_audit/summary.csv")
    bad_lambda = find_row(rows, method="cosine_source_selected_lambda")
    close(float(bad_lambda["mean_delta"]), 200.290)
    close(float(bad_lambda["worst_delta"]), 452.794)
    if bad_lambda["wins"] != "0":
        raise AssertionError(f"cosine-selected lambda should fail: {bad_lambda}")

    obs = find_row(rows, method="obs_half_life_2p5_roundfast20")
    close(float(obs["mean_delta"]), -34.560)
    if obs["wins"] != "15":
        raise AssertionError(f"observation-derived endpoint should be 15/15: {obs}")


def check_controls_and_localization() -> None:
    rows = read_csv(ROOT / "results/interpretable_strict_vs_rounded/summary.csv")
    safe = find_row(rows, variant="rounded_fast20_sqrtlocalized", group="extra_control")
    close(float(safe["mean_delta"]), 0.0, tol=0.001)
    close(float(safe["worst_delta"]), 0.0, tol=0.001)
    if safe["nonharm"] != "9":
        raise AssertionError(f"sqrt-localized controls should be 9/9 non-harm: {safe}")

    unsafe = find_row(rows, variant="rounded_fast20", group="extra_control")
    close(float(unsafe["mean_delta"]), 14.020)
    close(float(unsafe["worst_delta"]), 56.430)
    if unsafe["nonharm"] != "6":
        raise AssertionError(f"unlocalized controls should expose the cosine failure: {unsafe}")

    loc_rows = read_csv(ROOT / "results/interpretable_localization_sensitivity/summary.csv")
    sqrt_row = find_row(loc_rows, power="0.5", group="core_wsd")
    close(float(sqrt_row["mean_delta"]), -34.149)
    if sqrt_row["wins"] != "15" or sqrt_row["nonharm"] != "15":
        raise AssertionError(f"sqrt-localized WSD should be 15/15: {sqrt_row}")


def check_protocol() -> None:
    fit_rows = read_csv(ROOT / "results/interpretable_protocol_sensitivity/fit_start_sensitivity.csv")
    for fit_start in ["5000", "6500", "8000", "10000", "12000"]:
        row = find_row(fit_rows, fit_start=fit_start)
        if row["wins"] != "15" or row["nonharm"] != "15":
            raise AssertionError(f"fit_start {fit_start} should be all-win: {row}")

    ridge_rows = read_csv(ROOT / "results/interpretable_protocol_sensitivity/ridge_sensitivity.csv")
    row_004 = find_row(ridge_rows, ridge_tau="0.04")
    if float(row_004["worst_delta"]) <= 0.0:
        raise AssertionError("ridge tau 0.04 should remain a documented unsafe boundary")
    row_005 = find_row(ridge_rows, ridge_tau="0.05")
    close(float(row_005["mean_delta"]), -34.149)
    if row_005["wins"] != "15":
        raise AssertionError(f"ridge tau 0.05 should be all-win: {row_005}")


def check_mpl_ld_lag_response() -> None:
    rows = read_csv(ROOT / "results/mpl_ld_lag_response_audit/summary.csv")

    direct_64 = find_row(
        rows,
        variant="direct_mpl_ld_lag",
        tau_steps="64",
        split="direct_no_source_fit",
        group="core_wsd",
    )
    close(float(direct_64["mean_delta"]), -3.113)
    close(float(direct_64["worst_delta"]), -2.376)
    if direct_64["wins"] != "15" or direct_64["nonharm"] != "15":
        raise AssertionError(f"direct tau=64 should be WSD all-win: {direct_64}")

    direct_128 = find_row(
        rows,
        variant="direct_mpl_ld_lag",
        tau_steps="128",
        split="direct_no_source_fit",
        group="core_wsd",
    )
    close(float(direct_128["mean_delta"]), -9.522)
    close(float(direct_128["worst_delta"]), -5.951)
    if direct_128["wins"] != "15" or direct_128["nonharm"] != "15":
        raise AssertionError(f"direct tau=128 should be WSD all-win: {direct_128}")

    direct_128_ctrl = find_row(
        rows,
        variant="direct_mpl_ld_lag",
        tau_steps="128",
        split="direct_no_source_fit",
        group="extra_control",
    )
    close(float(direct_128_ctrl["worst_delta"]), 6.008)
    if direct_128_ctrl["nonharm"] != "6":
        raise AssertionError(f"direct tau=128 should document control harm: {direct_128_ctrl}")

    direct_256 = find_row(
        rows,
        variant="direct_mpl_ld_lag",
        tau_steps="256",
        split="direct_no_source_fit",
        group="core_wsd",
    )
    close(float(direct_256["worst_delta"]), 18.573)
    if direct_256["wins"] != "14":
        raise AssertionError(f"direct tau=256 should document over-correction: {direct_256}")

    cooldown_safe = find_row(
        rows,
        variant="cooldown_adiabatic_mpl_ld_lag",
        tau_steps="128",
        split="direct_no_source_fit",
        group="core_wsd",
    )
    close(float(cooldown_safe["mean_delta"]), -8.730)
    close(float(cooldown_safe["worst_delta"]), -6.220)
    if cooldown_safe["wins"] != "15" or cooldown_safe["nonharm"] != "15":
        raise AssertionError(f"cooldown adiabatic tau=128 should be WSD all-win: {cooldown_safe}")

    cooldown_safe_ctrl = find_row(
        rows,
        variant="cooldown_adiabatic_mpl_ld_lag",
        tau_steps="128",
        split="direct_no_source_fit",
        group="extra_control",
    )
    close(float(cooldown_safe_ctrl["mean_delta"]), 0.0, tol=0.001)
    close(float(cooldown_safe_ctrl["worst_delta"]), 0.0, tol=0.001)
    if cooldown_safe_ctrl["nonharm"] != "9":
        raise AssertionError(f"cooldown adiabatic controls should be 9/9: {cooldown_safe_ctrl}")

    cooldown_256 = find_row(
        rows,
        variant="cooldown_adiabatic_mpl_ld_lag",
        tau_steps="256",
        split="direct_no_source_fit",
        group="core_wsd",
    )
    close(float(cooldown_256["worst_delta"]), 14.957)
    if cooldown_256["wins"] != "14":
        raise AssertionError(f"cooldown adiabatic tau=256 should document tau instability: {cooldown_256}")

    support = find_row(
        rows,
        variant="cooldown_support_bracket_mpl_ld_lag",
        tau_steps="support_bracket",
        split="direct_no_source_fit",
        group="core_wsd",
    )
    close(float(support["mean_delta"]), -13.767)
    close(float(support["worst_delta"]), -6.292)
    if support["wins"] != "15" or support["nonharm"] != "15":
        raise AssertionError(f"support-bracket cooldown lag should be WSD all-win: {support}")

    support_ctrl = find_row(
        rows,
        variant="cooldown_support_bracket_mpl_ld_lag",
        tau_steps="support_bracket",
        split="direct_no_source_fit",
        group="extra_control",
    )
    close(float(support_ctrl["mean_delta"]), 0.0, tol=0.001)
    close(float(support_ctrl["worst_delta"]), 0.0, tol=0.001)
    if support_ctrl["nonharm"] != "9":
        raise AssertionError(f"support-bracket controls should be 9/9: {support_ctrl}")

    ledger = read_csv(ROOT / "results/mpl_ld_lag_response_audit/parameter_ledger.csv")
    fitted = [row for row in ledger if row["fitted_in_recommended_error_model"] != "0"]
    if fitted:
        raise AssertionError(f"recommended finite-response model should fit no residual parameters: {fitted}")
    residual_amp = find_row(ledger, quantity="residual amplitude")
    if "negative control" not in residual_amp["notes"]:
        raise AssertionError(f"residual amplitude should only be a negative control: {residual_amp}")

    features = read_csv(ROOT / "results/mpl_ld_lag_response_audit/schedule_features.csv")
    feature_sharp = find_row(features, test_label="WSD sharp", scale="100")
    close(float(feature_sharp["effective_tau_steps"]), 256.0, tol=0.001)
    close(float(feature_sharp["adiabatic_factor"]), 0.81685, tol=0.001)
    feature_wsdcon = find_row(features, test_label="WSD-con 9e-5", scale="100")
    close(float(feature_wsdcon["effective_tau_steps"]), 130.0, tol=0.001)
    feature_cos = find_row(features, test_label="Cosine 24k", scale="100")
    close(float(feature_cos["adiabatic_factor"]), 0.0, tol=0.001)

    details = read_csv(ROOT / "results/mpl_ld_lag_response_audit/details.csv")
    sharp = find_row(
        details,
        variant="cooldown_support_bracket_mpl_ld_lag",
        tau_steps="support_bracket",
        test_label="WSD sharp",
        test_scale="100",
    )
    close(float(sharp["effective_tau_steps"]), 256.0, tol=0.001)
    wsdcon = find_row(
        details,
        variant="cooldown_support_bracket_mpl_ld_lag",
        tau_steps="support_bracket",
        test_label="WSD-con 9e-5",
        test_scale="100",
    )
    close(float(wsdcon["effective_tau_steps"]), 130.0, tol=0.001)

    amp_128 = find_row(
        rows,
        variant="cosine_fit_amplitude_mpl_ld_lag",
        tau_steps="128",
        split="same_scale",
        group="core_wsd",
    )
    close(float(amp_128["mean_delta"]), 565.160)
    if amp_128["wins"] != "0" or amp_128["nonharm"] != "0":
        raise AssertionError(f"cosine-fitted amplitude should fail: {amp_128}")

    amp_128_cross = find_row(
        rows,
        variant="cosine_fit_amplitude_mpl_ld_lag",
        tau_steps="128",
        split="cross_scale",
        group="core_wsd",
    )
    close(float(amp_128_cross["mean_delta"]), 548.272)
    if amp_128_cross["wins"] != "0":
        raise AssertionError(f"cosine-fitted amplitude should fail cross-scale: {amp_128_cross}")

    plot_rows = read_csv(ROOT / "results/mpl_ld_lag_response_audit/error_curve_aggregate.csv")
    plot_core = find_row(plot_rows, group="core_wsd")
    close(float(plot_core["cooldown_support_bracket_mean_delta"]), -13.767)
    if plot_core["cooldown_support_bracket_wins"] != "15":
        raise AssertionError(f"support-bracket plot aggregate should be all-win: {plot_core}")
    for scale in ["25", "100", "400"]:
        fig = ROOT / f"results/mpl_ld_lag_response_audit/figs/finite_response_errors_{scale}M.png"
        if not fig.exists() or fig.stat().st_size <= 0:
            raise AssertionError(f"missing finite-response error figure: {fig}")

    sens = read_csv(ROOT / "results/mpl_ld_lag_response_audit/rule_sensitivity/summary.csv")
    sens_rec = find_row(sens, rule="support_linear_bracket", boundary="linear_support", group="core_wsd")
    close(float(sens_rec["mean_delta"]), -13.767)
    if sens_rec["wins"] != "15" or sens_rec["nonharm"] != "15":
        raise AssertionError(f"rule sensitivity recommended row should be all-win: {sens_rec}")
    sens_rec_ctrl = find_row(sens, rule="support_linear_bracket", boundary="linear_support", group="extra_control")
    close(float(sens_rec_ctrl["worst_delta"]), 0.0, tol=0.001)
    if sens_rec_ctrl["nonharm"] != "9":
        raise AssertionError(f"rule sensitivity recommended controls should be 9/9: {sens_rec_ctrl}")
    sens_unsafe = find_row(sens, rule="fixed_two_obs", boundary="linear_support", group="core_wsd")
    close(float(sens_unsafe["worst_delta"]), 14.957)
    if sens_unsafe["wins"] != "14":
        raise AssertionError(f"fixed two-observation row should document WSD-con failure: {sens_unsafe}")

    amp_sens = read_csv(ROOT / "results/mpl_ld_lag_response_audit/amplitude_sensitivity/summary.csv")
    amp_one = find_row(amp_sens, amplitude_scale="1.0", group="core_wsd")
    close(float(amp_one["mean_delta"]), -13.767)
    if amp_one["wins"] != "15" or amp_one["nonharm"] != "15":
        raise AssertionError(f"amplitude scale 1.0 should match recommended all-win result: {amp_one}")
    amp_low = find_row(amp_sens, amplitude_scale="0.25", group="core_wsd")
    if amp_low["wins"] != "15":
        raise AssertionError(f"amplitude scale 0.25 should show broad non-brittle improvement: {amp_low}")
    amp_high = find_row(amp_sens, amplitude_scale="1.5", group="core_wsd")
    if amp_high["wins"] != "15":
        raise AssertionError(f"amplitude scale 1.5 should remain all-win: {amp_high}")
    amp_too_high = find_row(amp_sens, amplitude_scale="2.0", group="core_wsd")
    close(float(amp_too_high["worst_delta"]), 3.765)
    if amp_too_high["wins"] != "14":
        raise AssertionError(f"amplitude scale 2.0 should document over-correction: {amp_too_high}")

    strict = read_csv(ROOT / "results/mpl_ld_lag_response_audit/strict_cosine_backbone/summary.csv")
    strict_core = find_row(strict, protocol="cosine_only", group="core_wsd")
    close(float(strict_core["mean_delta_vs_own_baseline"]), -11.437)
    close(float(strict_core["worst_delta_vs_own_baseline"]), -6.395)
    close(float(strict_core["mean_base_vs_official_baseline"]), 55.045)
    close(float(strict_core["mean_corr_vs_official_baseline"]), 37.342)
    if strict_core["wins_vs_own_baseline"] != "15" or strict_core["nonharm_vs_own_baseline"] != "15":
        raise AssertionError(f"strict cosine-only backbone should be WSD all-win vs itself: {strict_core}")

    strict_ctrl = find_row(strict, protocol="cosine_only", group="extra_control")
    close(float(strict_ctrl["mean_delta_vs_own_baseline"]), 0.0, tol=0.001)
    if strict_ctrl["nonharm_vs_own_baseline"] != "9":
        raise AssertionError(f"strict cosine-only controls should be 9/9 non-harm: {strict_ctrl}")


def check_docs() -> None:
    base = ROOT / "results/interpretable_error_model"
    current_checks = [
        (base / "FORMULA_CARD.md", "q2 Half-Life MPL-LD Response"),
        (base / "FORMULA_CARD.md", "Use a half-life bracket"),
        (base / "FORMULA_CARD.md", "2-q_s"),
        (base / "FORMULA_CARD.md", "support-projection locality"),
        (base / "FORMULA_CARD.md", "q2 + half-life bracket + support projection"),
        (base / "THEORY.md", "Herfindahl concentration"),
        (base / "THEORY.md", "直接插值 half-life"),
        (base / "THEORY.md", "Support-Projection Locality"),
        (base / "MODEL_DECISION.md", "q2 half-life MPL-LD response"),
        (base / "MODEL_DECISION.md", "support-projection locality"),
        (base / "MANIFEST.md", "hhi_q2_halflife_support_projection"),
        (base / "MANIFEST.md", "interpretable_theory_refinement_audit.py"),
        (ROOT / "results/interpretable_theory_refinement/REPORT.md", "q2 concentration + half-life bracket + support-projection locality"),
        (ROOT / "results/interpretable_theory_refinement/REPORT.md", "density projection"),
        (ROOT / "results/interpretable_theory_refinement/REPORT.md", "仍然只有一个 \\(\\widehat\\kappa_s\\)"),
    ]
    for path, text in current_checks:
        must_contain(path, text)
    return
    must_contain(base / "MODEL_DECISION.md", "MPL-LD tangent")
    must_contain(base / "MODEL_DECISION.md", "observation-bracket MPL-LD")
    must_contain(base / "MODEL_DECISION.md", "当前主线恢复为")
    must_contain(base / "MODEL_DECISION.md", "零参数 finite-response 保留为机制下界")
    must_contain(base / "MODEL_DECISION.md", "1/N_{\\mathrm{cal}}")
    must_contain(base / "MODEL_DECISION.md", "不枚举当前 WSD target schedule")
    must_contain(base / "MODEL_DECISION.md", "no nuisance raw projection")
    must_contain(base / "MODEL_DECISION.md", "DCT 仍只作为数值上限")
    must_contain(base / "INTERPRETABILITY_RESET.md", "当前不再推荐把 gate")
    must_contain(base / "INTERPRETABILITY_RESET.md", "observation-bracket MPL-LD")
    must_contain(base / "INTERPRETABILITY_RESET.md", "finite-response")
    must_contain(base / "INTERPRETABILITY_RESET.md", "DCT-based ablation")
    must_contain(base / "FORMULA_CARD.md", "主公式不再使用 gate")
    must_contain(base / "FORMULA_CARD.md", "observation-bracket MPL-LD")
    must_contain(base / "FORMULA_CARD.md", "current main candidate")
    must_contain(base / "FORMULA_CARD.md", "-29.87%")
    must_contain(base / "FORMULA_CARD.md", "-24.95%, 30/30")
    must_contain(base / "FORMULA_CARD.md", "DCT 版本仍保留为 performance reference")
    must_contain(base / "FORMULA_CARD.md", "Tau-Free Retention Shrinkage")
    must_contain(base / "FORMULA_CARD.md", "The previous square-root variant")
    must_contain(base / "THEORY.md", "observation-bracket MPL-LD")
    must_contain(base / "THEORY.md", "当前主公式")
    must_contain(base / "MANIFEST.md", "retained only as an ablation")
    must_contain(base / "MANIFEST.md", "direct_mpl_ld_lag")
    must_contain(base / "MANIFEST.md", "cooldown adiabatic MPL-LD lag")
    must_contain(base / "MANIFEST.md", "cooldown support-bracket MPL-LD lag")
    must_contain(base / "MANIFEST.md", "current main candidate")
    must_contain(base / "MANIFEST.md", "tau_free_sqrt_retention")
    must_contain(base / "MANIFEST.md", "MPL-LD tangent nuisance")
    must_contain(base / "MANIFEST.md", "interpretable_observation_bracket_audit.py")
    must_contain(base / "MANIFEST.md", "mpl_ld_lag_response_audit.py")
    must_contain(base / "MANIFEST.md", "parameter_ledger.csv")
    must_contain(base / "MANIFEST.md", "locality_boundary.csv")
    must_contain(base / "MANIFEST.md", "interpretable_scale_stability_audit.py")
    must_contain(base / "MANIFEST.md", "plot_interpretable_error_model.py")
    must_contain(base / "MANIFEST.md", "plot_mpl_ld_lag_response.py")
    must_contain(base / "MANIFEST.md", "mpl_ld_lag_rule_sensitivity.py")
    must_contain(base / "MANIFEST.md", "mpl_ld_lag_amplitude_sensitivity.py")
    must_contain(base / "RESEARCH_LOG.md", "撤回 sqrt-localized 主公式建议")
    must_contain(base / "RESEARCH_LOG.md", "tau-free shrinkage baseline")
    must_contain(base / "RESEARCH_LOG.md", "nuisance origin audit")
    must_contain(base / "RESEARCH_LOG.md", "current interpretable error-curve visualization")
    must_contain(base / "RESEARCH_LOG.md", "interpretability repair, DCT demoted")
    must_contain(base / "RESEARCH_LOG.md", "observation-bracket MPL-LD")
    must_contain(base / "RESEARCH_LOG.md", "解释性审计后降级 observation-bracket")
    must_contain(base / "RESEARCH_LOG.md", "signed MPL-LD decomposition")
    must_contain(base / "RESEARCH_LOG.md", "support-bracket")
    must_contain(base / "RESEARCH_LOG.md", "rule sensitivity")
    must_contain(base / "RESEARCH_LOG.md", "amplitude sensitivity")
    must_contain(base / "OPEN_LIMITATIONS.md", "外部泛化仍缺失")
    must_contain(base / "OPEN_LIMITATIONS.md", "Locality 是边界条件")
    must_contain(base / "OPEN_LIMITATIONS.md", "finite-response")
    must_contain(base / "OPEN_LIMITATIONS.md", "D_\\downarrow")
    must_contain(base / "OPEN_LIMITATIONS.md", "-13.77%")
    must_contain(base / "MANIFEST.md", "OPEN_LIMITATIONS.md")
    must_contain(base / "INTERPRETABILITY_AUDIT_2026_06_19.md", "cosine residual 中确实混入")
    must_contain(base / "INTERPRETABILITY_AUDIT_2026_06_19.md", "cooldown + adiabatic boundary")
    must_contain(base / "INTERPRETABILITY_AUDIT_2026_06_19.md", "support-bracket")
    must_contain(ROOT / "results/interpretable_shrinkage_origin_audit/REPORT.md", "tau-free hard baseline")
    must_contain(ROOT / "results/interpretable_nuisance_origin_audit/REPORT.md", "no-nuisance raw projection")
    must_contain(ROOT / "results/interpretable_nuisance_origin_audit/REPORT.md", "MPL-LD tangent main candidate")
    must_contain(ROOT / "results/interpretable_scale_stability_audit/REPORT.md", "Cross-scale `mpl_ld_tangent`")
    must_contain(ROOT / "results/interpretable_observation_bracket_audit/REPORT.md", "Selected fit start: `8000`")
    must_contain(ROOT / "results/interpretable_observation_bracket_audit/REPORT.md", "without WSD losses or target schedule enumeration")
    must_contain(ROOT / "results/interpretable_observation_bracket_audit/REPORT.md", "two endpoints of the observation bracket")
    must_contain(ROOT / "results/interpretable_observation_bracket_audit/REPORT.md", "The only residual-fitted quantity")
    must_contain(ROOT / "results/interpretable_observation_bracket_audit/REPORT.md", "not a learned gate")
    must_contain(ROOT / "results/interpretable_error_model/error_comparison/REPORT.md", "restored observation-bracket MPL-LD main candidate")
    must_contain(ROOT / "results/interpretable_error_model/RESTORE_OBSERVATION_BRACKET_ZH.md", "恢复后的判断")
    must_contain(ROOT / "results/interpretable_error_model/RESTORE_OBSERVATION_BRACKET_ZH.md", "zero-param finite-response")
    must_contain(ROOT / "results/mpl_ld_lag_response_audit/REPORT.md", "not yet a final method")
    must_contain(ROOT / "results/mpl_ld_lag_response_audit/REPORT.md", "Cooldown + Adiabatic Boundary Results")
    must_contain(ROOT / "results/mpl_ld_lag_response_audit/REPORT.md", "Cooldown + Support-Bracket Tau Results")
    must_contain(ROOT / "results/mpl_ld_lag_response_audit/ERROR_CURVES.md", "Cooldown support tau")
    must_contain(ROOT / "results/mpl_ld_lag_response_audit/MODEL_CARD_ZH.md", "fitted residual params")
    must_contain(ROOT / "results/mpl_ld_lag_response_audit/MODEL_CARD_ZH.md", "0 |")
    must_contain(ROOT / "results/mpl_ld_lag_response_audit/MODEL_CARD_ZH.md", "Strict cosine-only MPL backbone")
    must_contain(ROOT / "results/mpl_ld_lag_response_audit/MODEL_CARD_ZH.md", "+37.34%")
    must_contain(ROOT / "results/mpl_ld_lag_response_audit/rule_sensitivity/REPORT.md", "Support-bracket tau")
    must_contain(ROOT / "results/mpl_ld_lag_response_audit/amplitude_sensitivity/REPORT.md", "0.25` to `1.50")
    must_contain(ROOT / "results/mpl_ld_lag_response_audit/ARCHIVE_MANIFEST.md", "support-bracket cooldown finite-response")
    must_contain(ROOT / "results/mpl_ld_lag_response_audit/ARCHIVE_MANIFEST.md", "strict cosine-only MPL backbone")
    must_contain(ROOT / "results/mpl_ld_lag_response_audit/strict_cosine_backbone/REPORT.md", "cosine-only MPL")
    must_contain(ROOT / "results/mpl_ld_lag_response_audit/strict_cosine_backbone/REPORT.md", "+37.34%")
    must_contain(ROOT / "results/mpl_ld_lag_response_audit/INTERPRETABILITY_RESET_ZH.md", "不能说已经完整解决 cosine-to-WSD")
    must_contain(ROOT / "results/mpl_ld_lag_response_audit/INTERPRETABILITY_RESET_ZH.md", "gate、channel 选择、正弦")
    must_contain(ROOT / "results/mpl_ld_lag_response_audit/THEORY_ZH.md", "一阶响应方程")
    must_contain(ROOT / "results/mpl_ld_lag_response_audit/THEORY_ZH.md", "residual amplitude | not used")


def main() -> None:
    check_main_summary()
    check_core_decision()
    check_shrinkage_origin()
    check_nuisance_origin()
    check_scale_stability()
    check_observation_bracket()
    check_theory_refinement()
    check_error_comparison_figures()
    check_parameter_origin()
    check_controls_and_localization()
    check_protocol()
    check_mpl_ld_lag_response()
    check_docs()
    print("interpretable error model validation passed")


if __name__ == "__main__":
    main()
