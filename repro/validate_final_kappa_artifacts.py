#!/usr/bin/env python3
"""Validate paper-facing final kappa artifacts.

This is a lightweight consistency check. It does not rerun experiments; it
verifies that generated reports, method text, and key CSV metrics agree with
the final cap-free estimator narrative.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FINAL_DIR = ROOT / "results" / "current_law_final_kappa"
MULTI_DIR = ROOT / "results" / "current_law_multicurve_kappa_audit"
TRAINONLY_DIR = ROOT / "results" / "current_law_trainonly_tau_audit"
SPECTRAL_DIR = ROOT / "results" / "current_law_spectral_nuisance_audit"
SHRINK_DIR = ROOT / "results" / "current_law_predictive_shrinkage_audit"
EXTERNAL_DIR = ROOT / "results" / "current_law_nextgen_external_holdout_audit"
SAFETY_DIR = ROOT / "results" / "current_law_nextgen_safety_gate_audit"
TARGET_ID_DIR = ROOT / "results" / "current_law_target_identifiability_audit"
NEXTGEN_VS_FINAL_DIR = ROOT / "results" / "current_law_nextgen_vs_final_audit"
LAMBDA_STABILITY_DIR = ROOT / "results" / "current_law_nextgen_lambda_stability_audit"
COMPONENT_DIR = ROOT / "results" / "current_law_nextgen_component_ablation_audit"
MARGIN_DIR = ROOT / "results" / "current_law_target_retention_margin_audit"
STRESS_DIR = ROOT / "results" / "current_law_nextgen_stress_slice_audit"
RHO_DIR = ROOT / "results" / "current_law_nextgen_rho_margin_audit"
DEPLOY_DIR = ROOT / "results" / "current_law_nextgen_deployment_audit"
BLIND_DIR = ROOT / "results" / "current_law_nextgen_target_loss_blindness_audit"
SCALE_HOLDOUT_DIR = ROOT / "results" / "current_law_nextgen_scale_holdout_audit"


def read_text(path: Path) -> str:
    if not path.exists():
        raise AssertionError(f"missing file: {path}")
    return path.read_text(encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise AssertionError(f"missing file: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt_pct(x: float) -> str:
    return f"{x:+.1f}%"


def must_contain(text: str, needle: str, path: Path) -> None:
    if needle not in text:
        raise AssertionError(f"{path} does not contain expected text: {needle}")


def must_not_contain(text: str, needle: str, path: Path) -> None:
    if needle in text:
        raise AssertionError(f"{path} contains forbidden text: {needle}")


def validate_final_metrics(report: str, paper: str) -> None:
    rows = read_csv(FINAL_DIR / "comparison.csv")
    by_name = {row["estimator"]: row for row in rows}
    final = by_name["final_no_cap"]
    spectral = by_name["final_spectral_G4_no_cap"]
    capped = by_name["final_cap_0p03"]
    no_ret = by_name["no_retention_cap_0p03"]

    expected = {
        "worst": fmt_pct(float(final["worst_offdiag"])),
        "mean": fmt_pct(float(final["mean_offdiag"])),
        "cos_wsd": fmt_pct(float(final["cosine_to_wsd"])),
        "w9_wsd": fmt_pct(float(final["wsdcon9_to_wsd"])),
        "no_ret_worst": fmt_pct(float(no_ret["worst_offdiag"])),
    }
    spectral_expected = {
        "spectral_worst": fmt_pct(float(spectral["worst_offdiag"])),
        "spectral_mean": fmt_pct(float(spectral["mean_offdiag"])),
        "spectral_cos_wsd": fmt_pct(float(spectral["cosine_to_wsd"])),
        "spectral_w9_wsd": fmt_pct(float(spectral["wsdcon9_to_wsd"])),
    }
    for value in [*expected.values(), *spectral_expected.values()]:
        must_contain(paper, value, FINAL_DIR / "PAPER_METHOD.md")

    if fmt_pct(float(capped["worst_offdiag"])) != expected["worst"]:
        raise AssertionError("capped and cap-free worst offdiag no longer match")

    must_contain(report, "| `final_no_cap` |", FINAL_DIR / "REPORT.md")
    must_contain(report, "| `final_spectral_G4_no_cap` |", FINAL_DIR / "REPORT.md")
    must_contain(report, "matrix_final_spectral_G4_no_cap.png", FINAL_DIR / "REPORT.md")
    must_contain(report, "matrix_final_no_cap.png", FINAL_DIR / "REPORT.md")
    must_contain(report, "PAPER_METHOD.md", FINAL_DIR / "REPORT.md")


def validate_multicurve_metrics(paper: str, theory: str) -> None:
    rows = read_csv(MULTI_DIR / "train_size_summary.csv")
    by_size = {int(row["train_size"]): row for row in rows}
    one = fmt_pct(float(by_size[1]["median_worst_heldout"]))
    five = fmt_pct(float(by_size[5]["median_worst_heldout"]))
    for text, path in [(paper, FINAL_DIR / "PAPER_METHOD.md"), (theory, FINAL_DIR / "THEORY.md")]:
        must_contain(text, one, path)
        must_contain(text, five, path)
        must_contain(text, "training curves only", path)

    multi_report = read_text(MULTI_DIR / "REPORT.md")
    must_contain(multi_report, "tau` is estimated from the training curves only", MULTI_DIR / "REPORT.md")


def validate_trainonly_tau_metrics(paper: str, theory: str) -> None:
    rows = read_csv(TRAINONLY_DIR / "comparison.csv")
    by_mode = {row["mode"]: row for row in rows}
    other = by_mode["other_curves_tau"]
    train = by_mode["train_only_tau"]
    values = [
        fmt_pct(float(train["worst_offdiag"])),
        fmt_pct(float(train["mean_offdiag"])),
        fmt_pct(float(other["cosine_to_wsd"])),
        fmt_pct(float(train["cosine_to_wsd"])),
    ]
    for text, path in [(paper, FINAL_DIR / "PAPER_METHOD.md"), (theory, FINAL_DIR / "THEORY.md")]:
        for value in values:
            must_contain(text, value, path)
        must_contain(text, "train-only", path)

    report = read_text(TRAINONLY_DIR / "REPORT.md")
    must_contain(report, "train-only tau", report_path := TRAINONLY_DIR / "REPORT.md")
    must_contain(report, "does not rely on using held-out test curves", report_path)


def validate_spectral_nuisance_metrics(report: str, paper: str, theory: str) -> None:
    rows = read_csv(SPECTRAL_DIR / "comparison.csv")
    by_name = {row["estimator"]: row for row in rows}
    spectral = by_name["dct_low_frequency_G_m4"]
    legacy = by_name["legacy_smooth_G_m2"]
    adaptive = by_name["dct_retention_target_G_r0p35_mmin3"]
    adaptive_bad = by_name["dct_retention_target_G_r0p35"]
    expected = [
        fmt_pct(float(spectral["worst_offdiag"])),
        fmt_pct(float(spectral["cosine_to_wsd"])),
        fmt_pct(float(legacy["worst_offdiag"])),
        fmt_pct(float(legacy["cosine_to_wsd"])),
        fmt_pct(float(adaptive["worst_offdiag"])),
        fmt_pct(float(adaptive["cosine_to_wsd"])),
        fmt_pct(float(adaptive_bad["worst_offdiag"])),
    ]
    spectral_report = read_text(SPECTRAL_DIR / "REPORT.md")
    for value in expected:
        must_contain(spectral_report, value, SPECTRAL_DIR / "REPORT.md")
    for text, path in [
        (report, FINAL_DIR / "REPORT.md"),
        (paper, FINAL_DIR / "PAPER_METHOD.md"),
        (theory, FINAL_DIR / "THEORY.md"),
    ]:
        must_contain(text, "Spectral nuisance-subspace audit", path)
        must_contain(text, fmt_pct(float(spectral["worst_offdiag"])), path)
        must_contain(text, fmt_pct(float(spectral["cosine_to_wsd"])), path)
        must_contain(text, fmt_pct(float(adaptive["worst_offdiag"])), path)
        must_contain(text, fmt_pct(float(adaptive["cosine_to_wsd"])), path)
    must_contain(spectral_report, "discrete-cosine low-frequency", SPECTRAL_DIR / "REPORT.md")
    must_contain(spectral_report, "matrix_balanced_spectral_G.png", SPECTRAL_DIR / "REPORT.md")
    must_contain(spectral_report, "matrix_adaptive_spectral_G.png", SPECTRAL_DIR / "REPORT.md")
    must_contain(spectral_report, "K_min=3", SPECTRAL_DIR / "REPORT.md")


def validate_predictive_shrinkage(report: str, paper: str, theory: str) -> None:
    shrink_report = read_text(SHRINK_DIR / "REPORT.md")
    rows = read_csv(SHRINK_DIR / "train_size_summary.csv")
    by_key = {(row["candidate"], int(row["train_size"])): row for row in rows}
    expected = [
        fmt_pct(float(by_key[("train_size_rho0p5", n)]["worst_worst_heldout"]))
        for n in (1, 2, 3)
    ]
    for value in expected:
        if not value.startswith("-"):
            raise AssertionError("rho=0.5 predictive shrinkage is no longer non-failing for n<=3")
        must_contain(shrink_report, value, SHRINK_DIR / "REPORT.md")
    for candidate in ["train_size_rho0p5", "train_size_rho0p75", "train_size_rho1p0"]:
        worst = max(float(row["worst_worst_heldout"]) for row in rows if row["candidate"] == candidate)
        if worst >= 0:
            raise AssertionError(f"{candidate} is no longer non-failing across train sizes")
    rho025_worst = max(float(row["worst_worst_heldout"]) for row in rows if row["candidate"] == "train_size_rho0p25")
    if rho025_worst <= 0:
        raise AssertionError("rho=0.25 unexpectedly became non-failing; update the shrinkage sensitivity narrative")
    selected_worst = max(float(row["worst_worst_heldout"]) for row in rows if row["candidate"] == "train_selected_rho")
    if selected_worst <= 0:
        raise AssertionError("automatic train_selected_rho became non-failing; update the cautionary rho-selection narrative")

    key_rows = read_csv(SHRINK_DIR / "key_transfer_cells.csv")
    for fig in [
        SHRINK_DIR / "figs" / "rho_sensitivity.png",
        SHRINK_DIR / "figs" / "key_transfer_cells.png",
        SHRINK_DIR / "figs" / "single_curve_matrix_rho0p5.png",
        SHRINK_DIR / "figs" / "single_curve_matrix_by_scale_rho0p5.png",
    ]:
        if not fig.exists() or fig.stat().st_size < 10_000:
            raise AssertionError(f"missing or suspiciously small predictive shrinkage figure: {fig}")
    matrix_rows = read_csv(SHRINK_DIR / "single_curve_matrix_rho0p5.csv")
    if len(matrix_rows) != 30:
        raise AssertionError("single-curve rho=0.5 matrix should have 30 off-diagonal cells")
    matrix_worst = max(float(row["mean_delta_pct"]) for row in matrix_rows)
    if matrix_worst >= 0:
        raise AssertionError("single-curve rho=0.5 matrix is no longer fully non-failing")
    scale_matrix_rows = read_csv(SHRINK_DIR / "single_curve_matrix_by_scale_rho0p5.csv")
    if len(scale_matrix_rows) != 90:
        raise AssertionError("scale-specific single-curve rho=0.5 matrix should have 90 off-diagonal cells")
    scale_worst = max(float(row["delta_pct"]) for row in scale_matrix_rows)
    if scale_worst >= 0:
        raise AssertionError("scale-specific single-curve rho=0.5 matrix is no longer fully non-failing")
    cos_wsd = next(
        row
        for row in key_rows
        if row["candidate"] == "train_size_rho0p5"
        and row["train_label"] == "Cosine"
        and row["test_label"] == "WSD sharp"
    )
    if float(cos_wsd["mean_delta_pct"]) > -15:
        raise AssertionError("rho=0.5 no longer preserves substantial cosine-to-WSD transfer")
    must_contain(shrink_report, "Key Transfer Cells", SHRINK_DIR / "REPORT.md")
    must_contain(shrink_report, "figs/rho_sensitivity.png", SHRINK_DIR / "REPORT.md")
    must_contain(shrink_report, "figs/key_transfer_cells.png", SHRINK_DIR / "REPORT.md")
    must_contain(shrink_report, "figs/single_curve_matrix_rho0p5.png", SHRINK_DIR / "REPORT.md")
    must_contain(shrink_report, "figs/single_curve_matrix_by_scale_rho0p5.png", SHRINK_DIR / "REPORT.md")
    must_contain(shrink_report, "rho=0.75", SHRINK_DIR / "REPORT.md")
    must_contain(shrink_report, "train_selected_rho", SHRINK_DIR / "REPORT.md")
    must_contain(shrink_report, "cautionary check", SHRINK_DIR / "REPORT.md")
    for text, path in [
        (report, FINAL_DIR / "REPORT.md"),
        (paper, FINAL_DIR / "PAPER_METHOD.md"),
        (theory, FINAL_DIR / "THEORY.md"),
    ]:
        must_contain(text, "Predictive shrinkage", path)
        must_contain(text, "c_n = n/(n+0.5)", path)
    must_contain(report, "Target-identifiability attenuation audit", FINAL_DIR / "REPORT.md")
    must_contain(report, "Next-gen lambda stability audit", FINAL_DIR / "REPORT.md")
    must_contain(report, "186/186", FINAL_DIR / "REPORT.md")
    must_contain(report, "R_target(lambda) >= 0.01", FINAL_DIR / "REPORT.md")
    must_contain(report, "1116/1116", FINAL_DIR / "REPORT.md")
    must_contain(report, "Next-gen vs final audit", FINAL_DIR / "REPORT.md")
    must_contain(report, "90/90", FINAL_DIR / "REPORT.md")
    must_contain(report, "87/90", FINAL_DIR / "REPORT.md")
    must_contain(report, "Next-gen component ablation audit", FINAL_DIR / "REPORT.md")
    must_contain(report, "+32.6%", FINAL_DIR / "REPORT.md")
    must_contain(report, "Target-retention margin audit", FINAL_DIR / "REPORT.md")
    must_contain(report, "1.75x", FINAL_DIR / "REPORT.md")
    must_contain(report, "1.48x", FINAL_DIR / "REPORT.md")
    must_contain(report, "Next-gen stress-slice audit", FINAL_DIR / "REPORT.md")
    must_contain(report, "0` slice failures", FINAL_DIR / "REPORT.md")
    must_contain(report, "Next-gen rho margin audit", FINAL_DIR / "REPORT.md")
    must_contain(report, "rho=0.40", FINAL_DIR / "REPORT.md")
    must_contain(report, "rho=2.00", FINAL_DIR / "REPORT.md")
    must_contain(report, "Next-gen deployment estimator audit", FINAL_DIR / "REPORT.md")
    must_contain(report, "NextGenKappaEstimator", FINAL_DIR / "REPORT.md")
    must_contain(report, "Next-gen target-loss blindness audit", FINAL_DIR / "REPORT.md")
    must_contain(report, "Next-gen scale-holdout constant audit", FINAL_DIR / "REPORT.md")
    must_contain(theory, "posterior-predictive", FINAL_DIR / "THEORY.md")
    must_contain(theory, "rho=0.5", FINAL_DIR / "THEORY.md")
    must_contain(theory, "rho=0.25", FINAL_DIR / "THEORY.md")
    must_contain(theory, "fully automatic inner-CV rho selector", FINAL_DIR / "THEORY.md")
    for needle in [
        "R_target(lambda) = ||M_lambda phi_target||^2 / ||phi_target||^2",
        "a_target = 0 if R_target(lambda) < 0.01",
        "kappa_safe",
        "1116/1116",
        "+22.5%",
        "0.014797",
        "0.005721",
        "0.009201",
        "1.75x",
        "1.48x",
        "0.005",
        "non-knife-edge",
        "0` slice failures",
        "372/372",
        "rho=0.40",
        "rho=2.00",
        "stable safe range",
        "NextGenKappaEstimator",
        "0.000e+00",
        "target loss is used only for evaluation",
        "Scale-holdout constant audit",
        "3/3",
        "186/186",
        "The rule remains schedule-agnostic.",
    ]:
        must_contain(theory, needle, FINAL_DIR / "THEORY.md")


def validate_nextgen_method() -> None:
    path = FINAL_DIR / "NEXTGEN_METHOD.md"
    text = read_text(path)
    for needle in [
        "Next-Generation Kappa Formula Candidate",
        "M_lambda y = y - Q A_lambda y",
        "lambda in [0.01, 0.03]",
        "lambda stability audit reports `186/186`",
        "median `0.030`",
        "kappa_pool = sqrt(R_S)",
        "c_n = n / (n + 0.5)",
        "kappa_transfer = c_n * kappa_pool",
        "Proposition-Style Derivation",
        "r_c = kappa_c phi_c + g_c + eps_c",
        "kappa_c = theta + u_c",
        "kappa_transfer = [n / (n + 0.5)] * sqrt(l2_S / full_l2_S)",
        "The derivation uses no schedule-family label.",
        "Cosine -> WSD sharp `-20.5%`",
        "WSD-con 9e-5 -> WSD sharp `-8.7%`",
        "30/30` improving cells",
        "worst mean cell `-1.5%`",
        "90/90` improving cells",
        "worst cell `-1.0%`",
        "Target Identifiability And External Holdout Limitation",
        "Raw next-gen transfer is unsafe on `cosine_24000`",
        "R_target(lambda) = ||M_lambda phi_target||^2 / ||phi_target||^2",
        "a_target = 0 if R_target(lambda) < 0.01",
        "a_target = 0 if peak(phi_target) / mean(phi_target) < 2",
        "kappa_transfer_safe = a_target * kappa_transfer",
        "1116/1116` non-harming cells",
        "mean `-5.9%`",
        "144/144`, `315/315`, `360/360`, `225/225`, and `72/72`",
        "0.0075` through `0.015`",
        "geometric midpoint is `0.009201`",
        "without using held-out loss values",
        "train_relative_gate_0p05",
        "train_relative_gate_0p5",
        "absolute target-identifiability floor",
        "fully automatic train-only rho selector is currently unreliable",
    ]:
        must_contain(text, needle, path)


def validate_nextgen_formula_card() -> None:
    path = FINAL_DIR / "NEXTGEN_FORMULA_CARD.md"
    text = read_text(path)
    for needle in [
        "Next-Gen Kappa Formula Card",
        "not the paper-facing main estimator",
        "M_lambda y = y - Q (Q^T Q + lambda D)^(-1) Q^T y",
        "lambda in [0.01, 0.03]",
        "186/186",
        "selected lambda `0.030`",
        "kappa_transfer",
        "[n / (n + 0.5)]",
        "sqrt(l2_S / full_l2_S)",
        "R_target(lambda) = ||M_lambda phi_target||^2 / ||phi_target||^2",
        "a_target = 1{R_target(lambda) >= 0.01}",
        "kappa_safe = a_target * kappa_transfer",
        "No schedule-family labels are used.",
        "1116/1116",
        "+22.5%",
        "+32.6%",
        "Component ablation isolates the two stabilizers",
        "0.005721 < 0.01 < 0.014797",
        "1.75x",
        "1.48x",
        "-5.9%",
        "additional independent",
        "schedule families or runs",
    ]:
        must_contain(text, needle, path)


def validate_nextgen_external_holdout() -> None:
    report_path = EXTERNAL_DIR / "REPORT.md"
    report = read_text(report_path)
    rows = read_csv(EXTERNAL_DIR / "summary.csv")
    by_key = {(row["mode"], row["test_curve"]): row for row in rows}
    raw_cos = by_key[("raw_nextgen", "cosine_24000.csv")]
    gated_cos = by_key[("target_localization_gate", "cosine_24000.csv")]
    if float(raw_cos["worst_delta_pct"]) <= 0:
        raise AssertionError("raw next-gen no longer fails on cosine_24000; update external limitation narrative")
    if abs(float(gated_cos["worst_delta_pct"])) > 1e-9:
        raise AssertionError("target localization gate no longer abstains cleanly on cosine_24000")
    for needle in [
        "cosine_24000",
        "+21.8%",
        "target-localization gate",
        "0.0%",
    ]:
        must_contain(report, needle, report_path)


def validate_nextgen_safety_gate() -> None:
    report_path = SAFETY_DIR / "REPORT.md"
    report = read_text(report_path)
    rows = read_csv(SAFETY_DIR / "summary.csv")
    threshold_rows = read_csv(SAFETY_DIR / "threshold_sensitivity.csv")
    by_key = {(row["mode"], row["group"]): row for row in rows}
    by_threshold = {(float(row["threshold"]), row["group"]): row for row in threshold_rows}
    gated_all = by_key[("target_localization_gate", "all")]
    raw_all = by_key[("raw_nextgen", "all")]
    if int(gated_all["non_harm_cells"]) != int(gated_all["tests"]):
        raise AssertionError("target localization gate no longer makes all cells non-harming")
    if float(gated_all["worst_delta_pct"]) > 1e-9:
        raise AssertionError("target localization gate worst delta is no longer non-harming")
    if float(raw_all["worst_delta_pct"]) <= 0:
        raise AssertionError("raw next-gen no longer exposes the external holdout failure; update safety narrative")
    if int(by_threshold[(1.5, "all")]["non_harm_cells"]) == int(by_threshold[(1.5, "all")]["tests"]):
        raise AssertionError("threshold 1.5 unexpectedly became safe; update threshold sensitivity narrative")
    if int(by_threshold[(2.0, "all")]["non_harm_cells"]) != int(by_threshold[(2.0, "all")]["tests"]):
        raise AssertionError("threshold 2.0 is no longer safe")
    for needle in [
        "144/144",
        "+21.8%",
        "target-localization gate",
        "threshold sweep",
        "90/90",
    ]:
        must_contain(report, needle, report_path)


def validate_target_identifiability_audit() -> None:
    report_path = TARGET_ID_DIR / "REPORT.md"
    report = read_text(report_path)
    rows = read_csv(TARGET_ID_DIR / "summary.csv")
    train_size_rows = read_csv(TARGET_ID_DIR / "train_size_summary.csv")
    curve_rows = read_csv(TARGET_ID_DIR / "target_retention_by_curve.csv")
    by_key = {(row["mode"], row["group"]): row for row in rows}
    by_size = {(row["mode"], int(row["train_size"]), row["group"]): row for row in train_size_rows}
    by_curve = {row["test_curve"]: row for row in curve_rows}
    raw_all = by_key[("raw_nextgen", "all")]
    peak_all = by_key[("peak_mean_gate", "all")]
    ret001_all = by_key[("retention_gate_0p01", "all")]
    ret0005_all = by_key[("retention_gate_0p005", "all")]
    train_rel_weak = by_key[("train_relative_gate_0p05", "all")]
    train_rel_safe = by_key[("train_relative_gate_0p5", "all")]
    floor_rel = by_key[("floor_train_relative_gate_0p05", "all")]
    main_positive_floor = min(
        float(row["min_retention"])
        for row in curve_rows
        if row["group"] == "main_matrix" and float(row["min_retention"]) > 0
    )
    extra_positive_ceiling = max(
        float(row["max_retention"])
        for row in curve_rows
        if row["group"] == "extra_holdout" and float(row["max_retention"]) > 0
    )
    if not (extra_positive_ceiling < 0.01 < main_positive_floor):
        raise AssertionError("retention threshold 0.01 no longer separates diffuse extra targets from main targets")
    midpoint = (main_positive_floor * extra_positive_ceiling) ** 0.5
    if abs(midpoint - 0.009201) > 1e-4:
        raise AssertionError("retention margin midpoint changed; update target-identifiability narrative")
    if float(by_curve["cosine_24000.csv"]["max_retention"]) >= 0.01:
        raise AssertionError("cosine_24000 is no longer below the retention gate")
    if float(by_curve["cosine_72000.csv"]["min_retention"]) <= 0.01:
        raise AssertionError("cosine_72000 is no longer above the retention gate")
    if int(ret001_all["non_harm_cells"]) != int(ret001_all["tests"]):
        raise AssertionError("retention_gate_0p01 is no longer fully non-harming")
    if int(ret001_all["tests"]) != 1116:
        raise AssertionError("target-identifiability audit no longer covers the expected all-train-size matrix")
    if float(ret001_all["worst_delta_pct"]) > 1e-9:
        raise AssertionError("retention_gate_0p01 worst delta is no longer non-harming")
    if float(ret001_all["mean_delta_pct"]) >= float(peak_all["mean_delta_pct"]):
        raise AssertionError("retention_gate_0p01 no longer improves mean gain over peak_mean_gate")
    if int(ret0005_all["non_harm_cells"]) == int(ret0005_all["tests"]):
        raise AssertionError("retention threshold 0.005 unexpectedly became safe; update threshold narrative")
    if float(raw_all["worst_delta_pct"]) <= 0:
        raise AssertionError("raw next-gen no longer exposes the target-identifiability failure")
    if int(train_rel_weak["non_harm_cells"]) == int(train_rel_weak["tests"]):
        raise AssertionError("weak train-relative gate unexpectedly became safe; update relative-threshold narrative")
    if int(train_rel_safe["non_harm_cells"]) != int(train_rel_safe["tests"]):
        raise AssertionError("train_relative_gate_0p5 is no longer safe; update relative-threshold narrative")
    if float(train_rel_safe["mean_delta_pct"]) <= float(ret001_all["mean_delta_pct"]):
        raise AssertionError("train_relative_gate_0p5 is no longer more conservative than retention_gate_0p01")
    if int(floor_rel["non_harm_cells"]) != int(floor_rel["tests"]):
        raise AssertionError("floor_train_relative_gate_0p05 is no longer safe")
    expected_tests_by_size = {1: 144, 2: 315, 3: 360, 4: 225, 5: 72}
    for train_size, expected_tests in expected_tests_by_size.items():
        row = by_size[("retention_gate_0p01", train_size, "all")]
        if int(row["tests"]) != expected_tests:
            raise AssertionError(f"unexpected target-identifiability test count for train size {train_size}")
        if int(row["non_harm_cells"]) != int(row["tests"]):
            raise AssertionError(f"retention_gate_0p01 no longer non-harming for train size {train_size}")
        if float(row["worst_delta_pct"]) > 1e-9:
            raise AssertionError(f"retention_gate_0p01 worst delta no longer safe for train size {train_size}")
    for needle in [
        "R_target(lambda)",
        "retention_gate_0p01",
        "train_relative_gate_0p05",
        "train_relative_gate_0p5",
        "floor_train_relative_gate_0p05",
        "1116/1116",
        "Train-Size Breakdown",
        "+22.5%",
        "-5.9%",
        "-5.4%",
        "absolute target-identifiability floor",
        "Target Retention By Curve",
        "0.014797",
        "0.005721",
        "0.009201",
        "without using held-out loss values",
    ]:
        must_contain(report, needle, report_path)


def validate_nextgen_vs_final_audit() -> None:
    report_path = NEXTGEN_VS_FINAL_DIR / "REPORT.md"
    report = read_text(report_path)
    cell_rows = read_csv(NEXTGEN_VS_FINAL_DIR / "cell_summary.csv")
    scale_rows = read_csv(NEXTGEN_VS_FINAL_DIR / "scale_summary.csv")
    paired_rows = read_csv(NEXTGEN_VS_FINAL_DIR / "paired_cells.csv")
    by_cell = {row["method"]: row for row in cell_rows}
    by_scale = {row["method"]: row for row in scale_rows}
    final_cell = by_cell["final_no_cap"]
    next_cell = by_cell["nextgen_safe_rho0p5_Rtarget0p01"]
    final_scale = by_scale["final_no_cap"]
    next_scale = by_scale["nextgen_safe_rho0p5_Rtarget0p01"]
    if int(final_cell["winning_cells"]) != 30 or int(next_cell["winning_cells"]) != 30:
        raise AssertionError("final/nextgen common cell matrix is no longer fully improving")
    if int(next_scale["non_harm_rows"]) != 90:
        raise AssertionError("nextgen safe no longer has 90/90 scale-level non-harm")
    if int(final_scale["non_harm_rows"]) >= int(next_scale["non_harm_rows"]):
        raise AssertionError("nextgen safe no longer improves scale-level non-harm over final")
    if float(final_cell["worst_cell_delta_pct"]) >= float(next_cell["worst_cell_delta_pct"]):
        raise AssertionError("final_no_cap no longer has the stronger worst cell; update comparison narrative")
    better = sum(int(row["nextgen_better"]) for row in paired_rows)
    if better >= len(paired_rows):
        raise AssertionError("nextgen unexpectedly dominates final on all cells; update comparison narrative")
    for fig in [
        NEXTGEN_VS_FINAL_DIR / "figs" / "method_matrices.png",
        NEXTGEN_VS_FINAL_DIR / "figs" / "paired_difference_heatmap.png",
    ]:
        if not fig.exists() or fig.stat().st_size < 50_000:
            raise AssertionError(f"missing or suspiciously small nextgen-vs-final figure: {fig}")
    for needle in [
        "Next-Gen vs Final Kappa Audit",
        "figs/method_matrices.png",
        "figs/paired_difference_heatmap.png",
        "`final_no_cap`",
        "`nextgen_safe_rho0p5_Rtarget0p01`",
        "-12.0%",
        "-12.1%",
        "90/90",
        "87/90",
        "not a strict dominance result",
        "not as a replacement for the conservative paper-facing estimator",
    ]:
        must_contain(report, needle, report_path)


def validate_nextgen_component_ablation_audit() -> None:
    report_path = COMPONENT_DIR / "REPORT.md"
    report = read_text(report_path)
    rows = read_csv(COMPONENT_DIR / "summary.csv")
    train_rows = read_csv(COMPONENT_DIR / "train_size_summary.csv")
    by_key = {(row["mode"], row["group"]): row for row in rows}
    no_shrink_all = by_key[("no_predictive_shrinkage", "all")]
    shrink_all = by_key[("rho0p5_shrinkage", "all")]
    safe_all = by_key[("rho0p5_plus_Rtarget_gate", "all")]
    if int(safe_all["tests"]) != 1116:
        raise AssertionError("component ablation no longer covers the expected all-train-size matrix")
    if int(safe_all["non_harm_cells"]) != int(safe_all["tests"]):
        raise AssertionError("component ablation safe mode is no longer fully non-harming")
    if float(safe_all["worst_delta_pct"]) > 1e-9:
        raise AssertionError("component ablation safe mode worst delta is no longer non-harming")
    if float(no_shrink_all["worst_delta_pct"]) <= float(shrink_all["worst_delta_pct"]):
        raise AssertionError("rho=0.5 shrinkage no longer improves worst case over no predictive shrinkage")
    if float(shrink_all["worst_delta_pct"]) <= 0:
        raise AssertionError("rho=0.5 shrinkage unexpectedly became safe without target gate; update ablation narrative")
    if int(shrink_all["non_harm_cells"]) >= int(safe_all["non_harm_cells"]):
        raise AssertionError("target gate no longer improves non-harm count over shrinkage-only mode")
    expected_tests_by_size = {1: 144, 2: 315, 3: 360, 4: 225, 5: 72}
    by_size = {
        (row["mode"], int(row["train_size"])): row
        for row in train_rows
    }
    for train_size, expected_tests in expected_tests_by_size.items():
        row = by_size[("rho0p5_plus_Rtarget_gate", train_size)]
        if int(row["tests"]) != expected_tests:
            raise AssertionError(f"unexpected component ablation test count for train size {train_size}")
        if int(row["non_harm_cells"]) != int(row["tests"]):
            raise AssertionError(f"component ablation safe mode is no longer non-harming for train size {train_size}")
    for needle in [
        "Next-Gen Component Ablation Audit",
        "no_predictive_shrinkage",
        "rho0p5_shrinkage",
        "rho0p5_plus_Rtarget_gate",
        "+32.6%",
        "+22.5%",
        "1116/1116",
        "shrinkage controls finite-calibration amplitude over-transfer",
        "target gate controls non-identifiable target directions",
    ]:
        must_contain(report, needle, report_path)


def validate_target_retention_margin_audit() -> None:
    report_path = MARGIN_DIR / "REPORT.md"
    report = read_text(report_path)
    margins = read_csv(MARGIN_DIR / "margin_summary.csv")
    thresholds = read_csv(MARGIN_DIR / "threshold_sweep.csv")
    curves = read_csv(MARGIN_DIR / "curve_retention_summary.csv")
    if len(margins) != 1:
        raise AssertionError("target-retention margin audit should have one margin summary row")
    margin = margins[0]
    harmful_max = float(margin["max_raw_harmful_retention"])
    chosen = float(margin["chosen_threshold"])
    main_min = float(margin["min_main_matrix_retention"])
    midpoint = float(margin["geometric_midpoint"])
    if not (harmful_max < chosen < main_min):
        raise AssertionError("chosen target-retention threshold is no longer inside the empirical margin")
    if abs(chosen - 0.01) > 1e-12:
        raise AssertionError("chosen target-retention threshold changed; update formula narrative")
    if abs(harmful_max - 0.005721165724107469) > 1e-9:
        raise AssertionError("raw harmful retention boundary changed; update margin narrative")
    if abs(main_min - 0.01479727140302661) > 1e-9:
        raise AssertionError("main retention boundary changed; update margin narrative")
    if abs(midpoint - 0.009200958752288343) > 1e-9:
        raise AssertionError("retention margin midpoint changed; update margin narrative")
    by_threshold = {round(float(row["threshold"]), 12): row for row in thresholds}
    weak = by_threshold[0.005]
    selected = by_threshold[0.01]
    if float(weak["all_worst_delta_pct"]) <= 0:
        raise AssertionError("threshold 0.005 no longer exposes the diffuse-cosine failure")
    if int(selected["all_non_harm_cells"]) != int(selected["all_tests"]):
        raise AssertionError("threshold 0.01 is no longer fully non-harming")
    if int(selected["main_matrix_wins"]) != int(selected["main_matrix_tests"]):
        raise AssertionError("threshold 0.01 no longer preserves all main-matrix wins")
    by_curve = {row["test_label"]: row for row in curves}
    if float(by_curve["Cosine 24k"]["max_retention"]) >= chosen:
        raise AssertionError("diffuse Cosine 24k target is no longer below the chosen threshold")
    if float(by_curve["Cosine"]["min_retention"]) <= chosen:
        raise AssertionError("main cosine target is no longer above the chosen threshold")
    for needle in [
        "Target-Retention Margin Audit",
        "0.005721",
        "0.014797",
        "0.009201",
        "1.75x",
        "1.48x",
        "0.005",
        "+22.5%",
        "1116/1116",
        "margin-based identifiability floor",
        "not a tuned loss-optimal threshold",
    ]:
        must_contain(report, needle, report_path)


def validate_nextgen_stress_slice_audit() -> None:
    report_path = STRESS_DIR / "REPORT.md"
    report = read_text(report_path)
    mode_rows = read_csv(STRESS_DIR / "mode_summary.csv")
    axis_rows = read_csv(STRESS_DIR / "safe_axis_summary.csv")
    interaction_rows = read_csv(STRESS_DIR / "safe_scale_train_size_summary.csv")
    failures_path = STRESS_DIR / "safe_slice_failures.csv"
    failures = read_csv(failures_path)
    by_mode = {row["mode"]: row for row in mode_rows}
    safe = by_mode["rho0p5_plus_Rtarget_gate"]
    shrink = by_mode["rho0p5_shrinkage"]
    raw = by_mode["no_predictive_shrinkage"]
    if failures:
        raise AssertionError("stress-slice audit found safe formula slice failures")
    if int(safe["non_harm_rows"]) != int(safe["rows"]) or int(safe["rows"]) != 1116:
        raise AssertionError("stress-slice safe mode is no longer globally non-harming")
    if float(safe["worst_delta_pct"]) > 1e-12:
        raise AssertionError("stress-slice safe worst delta is no longer non-harming")
    if float(shrink["worst_delta_pct"]) <= 0 or float(raw["worst_delta_pct"]) <= float(shrink["worst_delta_pct"]):
        raise AssertionError("stress-slice ablation contrast changed; update narrative")
    for row in axis_rows:
        if int(row["non_harm_rows"]) != int(row["rows"]):
            raise AssertionError(f"safe formula hidden failure in axis slice {row['slice']}={row['value']}")
        if float(row["worst_delta_pct"]) > 1e-12:
            raise AssertionError(f"safe formula positive worst delta in axis slice {row['slice']}={row['value']}")
    for row in interaction_rows:
        if int(row["non_harm_rows"]) != int(row["rows"]):
            raise AssertionError(
                f"safe formula hidden failure in scale x train-size slice {row['scale']} x {row['train_size']}"
            )
        if float(row["worst_delta_pct"]) > 1e-12:
            raise AssertionError(
                f"safe formula positive worst delta in scale x train-size slice {row['scale']} x {row['train_size']}"
            )
    by_axis = {(row["slice"], row["value"]): row for row in axis_rows}
    if int(by_axis[("scale", "25")]["non_harm_rows"]) != 372:
        raise AssertionError("stress-slice scale=25 count changed")
    if int(by_axis[("scale", "100")]["non_harm_rows"]) != 372:
        raise AssertionError("stress-slice scale=100 count changed")
    if int(by_axis[("scale", "400")]["non_harm_rows"]) != 372:
        raise AssertionError("stress-slice scale=400 count changed")
    if int(by_axis[("target_curve", "WSD sharp")]["wins"]) != 93:
        raise AssertionError("stress-slice WSD sharp target no longer fully improves")
    if int(by_axis[("target_curve", "Cosine 24k")]["wins"]) != 0:
        raise AssertionError("stress-slice Cosine 24k should be an abstention target")
    for needle in [
        "Next-Gen Stress-Slice Audit",
        "1116/1116",
        "slice failures",
        "scale x train-size",
        "372/372",
        "144/144",
        "315/315",
        "360/360",
        "225/225",
        "72/72",
        "not hidden positive failures",
    ]:
        must_contain(report, needle, report_path)


def validate_nextgen_rho_margin_audit() -> None:
    report_path = RHO_DIR / "REPORT.md"
    report = read_text(report_path)
    rows = read_csv(RHO_DIR / "summary.csv")
    train_rows = read_csv(RHO_DIR / "train_size_summary.csv")
    by_key = {(round(float(row["rho"]), 12), row["group"]): row for row in rows}
    selected = by_key[(0.5, "all")]
    unsafe_035 = by_key[(0.35, "all")]
    first_safe = min(
        rho
        for rho, group in by_key
        if group == "all"
        and int(by_key[(rho, group)]["non_harm_cells"]) == int(by_key[(rho, group)]["tests"])
        and float(by_key[(rho, group)]["worst_delta_pct"]) <= 1e-12
    )
    if abs(first_safe - 0.4) > 1e-12:
        raise AssertionError("rho-margin first safe grid value changed; update rho narrative")
    if float(unsafe_035["worst_delta_pct"]) <= 0:
        raise AssertionError("rho=0.35 unexpectedly became safe; update rho-margin narrative")
    if int(selected["non_harm_cells"]) != int(selected["tests"]) or int(selected["tests"]) != 1116:
        raise AssertionError("rho=0.5 no longer fully non-harming in rho-margin audit")
    if float(selected["worst_delta_pct"]) > 1e-12:
        raise AssertionError("rho=0.5 rho-margin worst delta is no longer non-harming")
    if abs(float(selected["mean_delta_pct"]) - -5.895534667958595) > 1e-9:
        raise AssertionError("rho=0.5 rho-margin mean changed; update report")
    for rho in [0.4, 0.45, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0]:
        row = by_key[(rho, "all")]
        main = by_key[(rho, "main_matrix")]
        if int(row["non_harm_cells"]) != int(row["tests"]):
            raise AssertionError(f"rho={rho} no longer fully non-harming")
        if int(main["wins"]) != 558:
            raise AssertionError(f"rho={rho} no longer preserves all main-matrix wins")
    expected_tests_by_size = {1: 144, 2: 315, 3: 360, 4: 225, 5: 72}
    selected_train = [row for row in train_rows if abs(float(row["rho"]) - 0.5) < 1e-12]
    for row in selected_train:
        train_size = int(row["train_size"])
        if int(row["tests"]) != expected_tests_by_size[train_size]:
            raise AssertionError(f"unexpected rho-margin train-size count for {train_size}")
        if int(row["non_harm_cells"]) != int(row["tests"]):
            raise AssertionError(f"rho=0.5 no longer safe for train size {train_size}")
    for needle in [
        "Next-Gen Rho Margin Audit",
        "rho=0.40",
        "rho=2.00",
        "rho=0.50",
        "1116/1116",
        "558/558",
        "-5.9%",
        "+0.0%",
        "not a knife-edge",
        "stable non-harming range",
    ]:
        must_contain(report, needle, report_path)


def validate_nextgen_deployment_audit() -> None:
    report_path = DEPLOY_DIR / "REPORT.md"
    report = read_text(report_path)
    summary_rows = read_csv(DEPLOY_DIR / "summary.csv")
    diff_rows = read_csv(DEPLOY_DIR / "reference_diffs.csv")
    by_group = {row["group"]: row for row in summary_rows}
    all_row = by_group["all"]
    main_row = by_group["main_matrix"]
    extra_row = by_group["extra_holdout"]
    if int(all_row["tests"]) != 1116 or int(all_row["non_harm_cells"]) != 1116:
        raise AssertionError("deployment estimator no longer reproduces 1116/1116 non-harming result")
    if float(all_row["worst_delta_pct"]) > 1e-12:
        raise AssertionError("deployment estimator worst delta is no longer non-harming")
    if int(main_row["wins"]) != 558 or int(main_row["non_harm_cells"]) != 558:
        raise AssertionError("deployment estimator no longer preserves all main-matrix wins")
    if int(extra_row["wins"]) != 0 or float(extra_row["mean_kappa_safe"]) != 0.0:
        raise AssertionError("deployment estimator extra-holdout abstention changed")
    if len(diff_rows) != 1116:
        raise AssertionError("deployment reference comparison should cover 1116 rows")
    for key in [
        "delta_abs_diff",
        "kappa_abs_diff",
        "target_retention_abs_diff",
        "lambda_abs_diff",
        "target_factor_diff",
    ]:
        if max(float(row[key]) for row in diff_rows) > 1e-12:
            raise AssertionError(f"deployment estimator reference mismatch for {key}")
    for needle in [
        "Next-Gen Deployment Estimator Audit",
        "NextGenKappaEstimator",
        "1116/1116",
        "-5.9%",
        "+0.0%",
        "0.000e+00",
        "reusable estimator",
        "not relying on report-specific glue code",
    ]:
        must_contain(report, needle, report_path)


def validate_nextgen_target_loss_blindness_audit() -> None:
    report_path = BLIND_DIR / "REPORT.md"
    report = read_text(report_path)
    rows = read_csv(BLIND_DIR / "diffs.csv")
    if len(rows) != 1116:
        raise AssertionError("target-loss blindness audit should cover 1116 rows")
    for key in [
        "retention_abs_diff",
        "deployment_retention_abs_diff",
        "factor_abs_diff",
        "deployment_factor_abs_diff",
        "kappa_safe_abs_diff",
        "deployment_kappa_abs_diff",
    ]:
        if max(float(row[key]) for row in rows) > 1e-12:
            raise AssertionError(f"target-loss blindness failed for {key}")
    for needle in [
        "Next-Gen Target-Loss Blindness Audit",
        "target loss is not used for deployment",
        "1116",
        "0.000e+00",
        "target-loss blind",
        "target loss is used only for evaluation",
        "training residuals plus target schedule features",
    ]:
        must_contain(report, needle, report_path)


def validate_nextgen_scale_holdout_audit() -> None:
    report_path = SCALE_HOLDOUT_DIR / "REPORT.md"
    report = read_text(report_path)
    retention_rows = read_csv(SCALE_HOLDOUT_DIR / "retention_scale_holdout.csv")
    rho_rows = read_csv(SCALE_HOLDOUT_DIR / "rho_scale_holdout.csv")
    if len(retention_rows) != 3 or len(rho_rows) != 3:
        raise AssertionError("scale-holdout audit should contain three held-out scales")
    for row in retention_rows:
        if int(row["threshold_inside_train_margin"]) != 1:
            raise AssertionError(f"retention threshold not inside train-scale margin for heldout {row['heldout_scale']}")
        if int(row["heldout_non_harm_cells"]) != int(row["heldout_tests"]) or int(row["heldout_tests"]) != 372:
            raise AssertionError(f"retention heldout scale no longer fully non-harming for {row['heldout_scale']}")
        if int(row["heldout_main_wins"]) != int(row["heldout_main_tests"]) or int(row["heldout_main_tests"]) != 186:
            raise AssertionError(f"retention heldout scale no longer preserves main wins for {row['heldout_scale']}")
        if not (float(row["train_max_harmful_retention"]) < 0.01 < float(row["train_min_main_retention"])):
            raise AssertionError(f"retention margin inequality failed for heldout {row['heldout_scale']}")
    for row in rho_rows:
        if int(row["selected_inside_safe_side"]) != 1:
            raise AssertionError(f"rho=0.5 no longer on safe side for heldout {row['heldout_scale']}")
        if float(row["train_first_safe_rho"]) > 0.5:
            raise AssertionError(f"train-scale first safe rho exceeds selected rho for heldout {row['heldout_scale']}")
        if int(row["heldout_non_harm_cells"]) != int(row["heldout_tests"]) or int(row["heldout_tests"]) != 372:
            raise AssertionError(f"rho heldout scale no longer fully non-harming for {row['heldout_scale']}")
        if int(row["heldout_main_wins"]) != int(row["heldout_main_tests"]) or int(row["heldout_main_tests"]) != 186:
            raise AssertionError(f"rho heldout scale no longer preserves main wins for {row['heldout_scale']}")
    for needle in [
        "Next-Gen Scale-Holdout Constant Audit",
        "R_target >= 0.01",
        "rho=0.5",
        "3/3",
        "372/372",
        "186/186",
        "scale-stable",
        "not replacing true external scale or schedule-family validation",
    ]:
        must_contain(report, needle, report_path)


def validate_nextgen_lambda_stability_audit() -> None:
    report_path = LAMBDA_STABILITY_DIR / "REPORT.md"
    report = read_text(report_path)
    rows = read_csv(LAMBDA_STABILITY_DIR / "summary.csv")
    by_size = {row["train_size"]: row for row in rows}
    all_row = by_size["all"]
    if int(all_row["within_band_rows"]) != int(all_row["rows"]):
        raise AssertionError("next-gen lambda stability audit has out-of-band rows")
    if int(all_row["rows"]) != 186:
        raise AssertionError("next-gen lambda stability audit no longer covers 186 kappa rows")
    if abs(float(all_row["min_lambda"]) - 0.01) > 1e-12:
        raise AssertionError("next-gen lambda min changed; update lambda stability narrative")
    if abs(float(all_row["max_lambda"]) - 0.03) > 1e-12:
        raise AssertionError("next-gen lambda max changed; update lambda stability narrative")
    if abs(float(all_row["median_lambda"]) - 0.03) > 1e-12:
        raise AssertionError("next-gen lambda median changed; update lambda stability narrative")
    if int(by_size["1"]["rows"]) != 18 or abs(float(by_size["1"]["median_lambda"]) - 0.025) > 1e-12:
        raise AssertionError("single-curve lambda fallback changed; update lambda stability narrative")
    for needle in [
        "Next-Gen Lambda Stability Audit",
        "lambda in [0.01, 0.03]",
        "186/186",
        "median `0.030`",
        "Single-curve calibration uses the fixed fallback `0.025`",
        "high-drift-control side",
    ]:
        must_contain(report, needle, report_path)


def validate_no_old_paper_narrative(report: str, paper: str, theory: str) -> None:
    forbidden = [
        "Z = [1, t, t^2] over normalized training step",
        "matrix_final_cap_0p03.png",
        "numeric oracle",
        "degree selection",
        "sqrt(retention)` shrinkage",
        "geometric identifiability factor",
    ]
    for text, path in [
        (report, FINAL_DIR / "REPORT.md"),
        (paper, FINAL_DIR / "PAPER_METHOD.md"),
        (theory, FINAL_DIR / "THEORY.md"),
    ]:
        for needle in forbidden:
            must_not_contain(text, needle, path)

    must_contain(report, "G = low-frequency MPL-residual nuisance subspace", FINAL_DIR / "REPORT.md")
    must_contain(report, "MANIFEST.md", FINAL_DIR / "REPORT.md")
    must_contain(theory, "The theoretical object is the nuisance subspace `G`, not a polynomial fit.", FINAL_DIR / "THEORY.md")
    must_contain(paper, "identifiable-amplitude conversion", FINAL_DIR / "PAPER_METHOD.md")


def validate_manifest() -> None:
    manifest_path = FINAL_DIR / "MANIFEST.md"
    manifest = read_text(manifest_path)
    by_name = {row["estimator"]: row for row in read_csv(FINAL_DIR / "comparison.csv")}
    final = by_name["final_no_cap"]
    spectral = by_name["final_spectral_G4_no_cap"]
    expected_values = [
        fmt_pct(float(spectral["worst_offdiag"])),
        fmt_pct(float(spectral["mean_offdiag"])),
        fmt_pct(float(spectral["cosine_to_wsd"])),
        fmt_pct(float(spectral["wsdcon9_to_wsd"])),
        f"{float(spectral['max_cosine_kappa']):.4f}",
        f"{100 * float(spectral['cap_saturation_rate']):.1f}%",
        fmt_pct(float(final["worst_offdiag"])),
        fmt_pct(float(final["mean_offdiag"])),
        fmt_pct(float(final["cosine_to_wsd"])),
        fmt_pct(float(final["wsdcon9_to_wsd"])),
    ]
    for needle in [
        "final_spectral_G4_no_cap",
        "final_no_cap",
        "python3 repro/current_law_final_kappa.py",
        "python3 repro/validate_final_kappa_artifacts.py",
        "Do Not Use As Main Claim",
        "numeric_oracle_deg1",
        "final kappa artifacts validated",
        "PAPER_METHOD.md",
        "REPORT.md",
        "THEORY.md",
        "NEXTGEN_FORMULA_CARD.md",
        "NEXTGEN_METHOD.md",
        "APPENDIX_LATEX.md",
        "Generation note:",
        "`THEORY.md` and `APPENDIX_LATEX.md` are maintained derivation artifacts",
        "current_law_predictive_shrinkage_audit",
        "current_law_nextgen_lambda_stability_audit",
        "current_law_nextgen_rho_margin_audit",
        "current_law_nextgen_component_ablation_audit",
        "current_law_target_retention_margin_audit",
        "current_law_nextgen_stress_slice_audit",
        "current_law_nextgen_deployment_audit",
        "current_law_nextgen_target_loss_blindness_audit",
        "current_law_nextgen_scale_holdout_audit",
        "target-identifiability gating",
        "current_law_target_identifiability_audit",
        "python3 repro/current_law_predictive_shrinkage_audit.py",
        "python3 repro/current_law_nextgen_lambda_stability_audit.py",
        "python3 repro/current_law_nextgen_rho_margin_audit.py",
        "python3 repro/current_law_nextgen_external_holdout_audit.py",
        "python3 repro/current_law_nextgen_safety_gate_audit.py",
        "python3 repro/current_law_target_identifiability_audit.py",
        "python3 repro/current_law_target_retention_margin_audit.py",
        "python3 repro/current_law_nextgen_component_ablation_audit.py",
        "python3 repro/current_law_nextgen_stress_slice_audit.py",
        "python3 repro/current_law_nextgen_deployment_audit.py",
        "python3 repro/current_law_nextgen_target_loss_blindness_audit.py",
        "python3 repro/current_law_nextgen_scale_holdout_audit.py",
        "python3 repro/current_law_nextgen_vs_final_audit.py",
        "current_law_nextgen_vs_final_audit",
    ]:
        must_contain(manifest, needle, manifest_path)
    for value in expected_values:
        must_contain(manifest, value, manifest_path)

    links = re.findall(r"\]\(([^)]+)\)", manifest)
    for link in links:
        if link.startswith("http"):
            continue
        target = (FINAL_DIR / link).resolve()
        if not target.exists():
            raise AssertionError(f"broken manifest link: {link} -> {target}")


def validate_links() -> None:
    report = read_text(FINAL_DIR / "REPORT.md")
    links = re.findall(r"\]\(([^)]+)\)", report)
    for link in links:
        if link.startswith("http"):
            continue
        target = (FINAL_DIR / link).resolve()
        if not target.exists():
            raise AssertionError(f"broken report link: {link} -> {target}")


def validate_appendix() -> None:
    appendix_path = FINAL_DIR / "APPENDIX_LATEX.md"
    appendix = read_text(appendix_path)
    for needle in [
        "\\widehat{\\kappa}",
        "M_{\\mathcal{G}}",
        "Frisch--Waugh--Lovell",
        "\\tau = \\sigma / k_0",
        "cap-free estimator",
        "\\mathcal{G}_K",
        "K_{\\min}",
        "\\rho=0.35",
        "two-stage bandwidth rule",
        "\\widehat{\\kappa}_{\\mathrm{transfer}}",
        "\\rho=0.5",
        "Target-side identifiability",
        "R_{\\mathrm{tar}}(\\lambda)",
        "\\widehat{\\kappa}_{\\mathrm{safe}}",
        "\\mathbf{1}\\{R_{\\mathrm{tar}}(\\lambda)\\ge 0.01\\}",
        "\\frac{n}{n+0.5}",
        "0.005721 < 0.01 < 0.014797",
        "1.75\\times",
        "1.48\\times",
    ]:
        must_contain(appendix, needle, appendix_path)


def main() -> None:
    report = read_text(FINAL_DIR / "REPORT.md")
    paper = read_text(FINAL_DIR / "PAPER_METHOD.md")
    theory = read_text(FINAL_DIR / "THEORY.md")
    validate_final_metrics(report, paper)
    validate_multicurve_metrics(paper, theory)
    validate_trainonly_tau_metrics(paper, theory)
    validate_spectral_nuisance_metrics(report, paper, theory)
    validate_predictive_shrinkage(report, paper, theory)
    validate_nextgen_formula_card()
    validate_nextgen_method()
    validate_nextgen_lambda_stability_audit()
    validate_nextgen_rho_margin_audit()
    validate_nextgen_external_holdout()
    validate_nextgen_safety_gate()
    validate_target_identifiability_audit()
    validate_nextgen_vs_final_audit()
    validate_nextgen_component_ablation_audit()
    validate_target_retention_margin_audit()
    validate_nextgen_stress_slice_audit()
    validate_nextgen_deployment_audit()
    validate_nextgen_target_loss_blindness_audit()
    validate_nextgen_scale_holdout_audit()
    validate_no_old_paper_narrative(report, paper, theory)
    validate_manifest()
    validate_links()
    validate_appendix()
    print("final kappa artifacts validated")


if __name__ == "__main__":
    main()
