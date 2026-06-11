# Final Kappa Artifact Manifest

This directory is the paper-facing entry point for the final `kappa` estimator. Use this manifest to avoid mixing the final method with earlier exploratory variants.

## Final Estimator

The paper-facing estimator is the cap-free nuisance-projected empirical-Bayes amplitude estimator. The strongest current implementation uses the legacy smooth low-frequency nuisance basis; the balanced spectral `G_4` version is the basis-neutral robustness audit.

```text
r = observed_loss - MPL
G = low-frequency MPL-residual nuisance subspace
phi_perp = M_G phi
r_perp = M_G r
tau = sigma / k0
R = ||phi_perp||^2 / ||phi||^2

kappa_hat =
sqrt(R) * max(0, <phi_perp, r_perp> / (||phi_perp||^2 + tau^2))
```

The optional capped variant `kappa <= 0.03` should be described only as a truncated-prior variant. It is not the main mechanism.

## Main Result

From `comparison.csv`, the paper-facing `final_no_cap` row is:

| metric | value |
|---|---:|
| worst off-diagonal | `-2.7%` |
| mean off-diagonal | `-12.1%` |
| cosine -> WSD | `-4.3%` |
| WSD-con 9e-5 -> WSD | `-16.0%` |
| max cosine kappa | `0.0089` |
| cap saturation | `0.0%` |

The spectral `final_spectral_G4_no_cap` audit row is:

| metric | value |
|---|---:|
| worst off-diagonal | `-1.8%` |
| mean off-diagonal | `-10.0%` |
| cosine -> WSD | `-3.6%` |
| WSD-con 9e-5 -> WSD | `-11.0%` |
| max cosine kappa | `0.0070` |
| cap saturation | `0.0%` |

## Required Reading Order

1. [`PAPER_METHOD.md`](PAPER_METHOD.md): concise paper-ready method text.
2. [`REPORT.md`](REPORT.md): main matrix, comparison table, and audit links.
3. [`THEORY.md`](THEORY.md): assumptions, proposition-style derivation, and limitations.
4. [`NEXTGEN_FORMULA_CARD.md`](NEXTGEN_FORMULA_CARD.md): compact next-generation formula card for writing slides or notes.
5. [`NEXTGEN_METHOD.md`](NEXTGEN_METHOD.md): full next-generation research candidate with predictive transfer shrinkage and target-identifiability gating.
6. [`APPENDIX_LATEX.md`](APPENDIX_LATEX.md): LaTeX-ready appendix derivation.

## Supporting Audits

- Subset robustness: `../current_law_final_kappa_robustness/REPORT.md`
- Bootstrap uncertainty: `../current_law_final_kappa_bootstrap/REPORT.md`
- Retention exponent sweep: `../current_law_retention_power_audit/REPORT.md`
- Tau multiplier sweep: `../current_law_tau_sensitivity_audit/REPORT.md`
- Train-only tau audit: `../current_law_trainonly_tau_audit/REPORT.md`
- Multi-curve calibration: `../current_law_multicurve_kappa_audit/REPORT.md`
- Spectral nuisance-subspace audit: `../current_law_spectral_nuisance_audit/REPORT.md`
- Soft spectral nuisance-prior audit: `../current_law_soft_spectral_kappa_audit/REPORT.md`
- Soft spectral lambda-selection audit: `../current_law_soft_spectral_selection_audit/REPORT.md`
- Soft spectral multi-curve selection audit: `../current_law_soft_spectral_multicurve_selection_audit/REPORT.md`
- Predictive shrinkage audit: `../current_law_predictive_shrinkage_audit/REPORT.md`

- Next-gen lambda stability audit: `../current_law_nextgen_lambda_stability_audit/REPORT.md`
- Next-gen rho margin audit: `../current_law_nextgen_rho_margin_audit/REPORT.md`
- Next-gen external holdout sanity audit: `../current_law_nextgen_external_holdout_audit/REPORT.md`

- Next-gen target safety gate audit: `../current_law_nextgen_safety_gate_audit/REPORT.md`

- Next-gen target-identifiability attenuation audit: `../current_law_target_identifiability_audit/REPORT.md`
- Next-gen target-retention margin audit: `../current_law_target_retention_margin_audit/REPORT.md`
- Next-gen component ablation audit: `../current_law_nextgen_component_ablation_audit/REPORT.md`
- Next-gen stress-slice audit: `../current_law_nextgen_stress_slice_audit/REPORT.md`
- Next-gen deployment estimator audit: `../current_law_nextgen_deployment_audit/REPORT.md`
- Next-gen target-loss blindness audit: `../current_law_nextgen_target_loss_blindness_audit/REPORT.md`
- Next-gen vs final common-matrix audit: `../current_law_nextgen_vs_final_audit/REPORT.md`

## Reproduction Commands

Generation note: `current_law_final_kappa.py` regenerates the main CSVs, figures, `REPORT.md`, `PAPER_METHOD.md`, `NEXTGEN_FORMULA_CARD.md`, `NEXTGEN_METHOD.md`, and `MANIFEST.md`. `THEORY.md` and `APPENDIX_LATEX.md` are maintained derivation artifacts and are checked by `validate_final_kappa_artifacts.py`.

Regenerate the main final artifacts:

```bash
python3 repro/current_law_final_kappa.py
```

Regenerate the train-only tau audit:

```bash
python3 repro/current_law_trainonly_tau_audit.py
```

Regenerate the multi-curve calibration audit:

```bash
python3 repro/current_law_multicurve_kappa_audit.py
```

Regenerate the spectral nuisance-subspace audit:

```bash
python3 repro/current_law_spectral_nuisance_audit.py
```

Regenerate the soft spectral nuisance-prior audit:

```bash
python3 repro/current_law_soft_spectral_kappa_audit.py
```

Regenerate the soft spectral lambda-selection audit:

```bash
python3 repro/current_law_soft_spectral_selection_audit.py
```

Regenerate the soft spectral multi-curve selection audit:

```bash
python3 repro/current_law_soft_spectral_multicurve_selection_audit.py
```

Regenerate the predictive shrinkage audit:

```bash
python3 repro/current_law_predictive_shrinkage_audit.py
```

Regenerate the next-gen lambda stability audit:

```bash
python3 repro/current_law_nextgen_lambda_stability_audit.py
```

Regenerate the next-gen rho margin audit:

```bash
python3 repro/current_law_nextgen_rho_margin_audit.py
```

Regenerate the next-gen external holdout sanity audit:

```bash
python3 repro/current_law_nextgen_external_holdout_audit.py
```

Regenerate the next-gen target safety gate audit:

```bash
python3 repro/current_law_nextgen_safety_gate_audit.py
```

Regenerate the next-gen target-identifiability attenuation audit:

```bash
python3 repro/current_law_target_identifiability_audit.py
```

Regenerate the next-gen target-retention margin audit:

```bash
python3 repro/current_law_target_retention_margin_audit.py
```

Regenerate the next-gen component ablation audit:

```bash
python3 repro/current_law_nextgen_component_ablation_audit.py
```

Regenerate the next-gen stress-slice audit:

```bash
python3 repro/current_law_nextgen_stress_slice_audit.py
```

Regenerate the next-gen deployment estimator audit:

```bash
python3 repro/current_law_nextgen_deployment_audit.py
```

Regenerate the next-gen target-loss blindness audit:

```bash
python3 repro/current_law_nextgen_target_loss_blindness_audit.py
```

Regenerate the next-gen vs final common-matrix audit:

```bash
python3 repro/current_law_nextgen_vs_final_audit.py
```

Validate the final paper-facing artifacts:

```bash
python3 repro/validate_final_kappa_artifacts.py
```

Expected validator output:

```text
final kappa artifacts validated
```

## Do Not Use As Main Claim

The following are useful diagnostics but should not be presented as the main method:

- `numeric_oracle_deg1`: internal warning/diagnostic only.
- `final_cap_0p03`: optional truncated-prior variant, not the main estimator.
- Any degree-selection narrative for `G`: the theoretical object is the low-frequency nuisance subspace, not a polynomial fitting law.
