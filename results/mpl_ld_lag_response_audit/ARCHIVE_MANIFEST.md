# MPL-LD Cooldown Finite-Response Archive

Date: 2026-06-19

## Current Candidate

\[
\hat L_s(t)
=
L_{\mathrm{MPL},s}(t)
+
a_sB_s[D_{\downarrow,\tau_s,s}(t)-D_{\downarrow,s}(t)].
\]

\[
\tau_s
=
\Delta_{\mathrm{obs}}
\left(1+\min(1,\ell_\downarrow/\Delta_{\mathrm{obs}})\right),
\qquad
a_s=\left[1-\ell_\downarrow/(T-W)\right]_+.
\]

This model uses zero residual-fitted parameters.  All added quantities come from the existing MPL formula, the LR schedule, and the logging interval.

## Main Result

Frozen official MPL backbone:

| candidate | WSD mean | WSD worst | wins | controls |
|---|---:|---:|---:|---:|
| support-bracket cooldown finite-response | -13.77% | -6.29% | 15/15 | 9/9 non-harm |

Strict cosine-only MPL backbone:

| candidate | WSD mean | WSD worst | wins | controls | corrected vs official MPL |
|---|---:|---:|---:|---:|---:|
| support-bracket cooldown finite-response | -11.44% | -6.40% | 15/15 | 9/9 non-harm | +37.34% mean |

The strict backbone result confirms the finite-response signal is not only an artifact of the frozen official backbone, but it also shows the current method is not a complete cosine-to-WSD solution.

Negative control:

| negative control | WSD mean | WSD worst | wins |
|---|---:|---:|---:|
| cosine-fitted cooldown amplitude, tau=128 | +525.54% | +1166.45% | 0/15 |

## Reproduction Commands

```bash
python3 repro/mpl_ld_lag_response_audit.py
python3 repro/mpl_ld_lag_strict_backbone_audit.py
python3 repro/plot_mpl_ld_lag_response.py
python3 repro/mpl_ld_lag_rule_sensitivity.py
python3 repro/mpl_ld_lag_amplitude_sensitivity.py
python3 repro/validate_interpretable_error_model.py
```

Syntax gate:

```bash
python3 -m py_compile \
  repro/mpl_ld_lag_response_audit.py \
  repro/mpl_ld_lag_strict_backbone_audit.py \
  repro/plot_mpl_ld_lag_response.py \
  repro/mpl_ld_lag_rule_sensitivity.py \
  repro/mpl_ld_lag_amplitude_sensitivity.py \
  repro/validate_interpretable_error_model.py
```

## Artifacts

- `repro/mpl_ld_lag_response_audit.py`: main finite-response audit.
- `repro/mpl_ld_lag_strict_backbone_audit.py`: same formula under a strict cosine-only MPL backbone.
- `repro/plot_mpl_ld_lag_response.py`: error-curve visualization.
- `repro/mpl_ld_lag_rule_sensitivity.py`: schedule-only tau/boundary sensitivity audit.
- `repro/mpl_ld_lag_amplitude_sensitivity.py`: fixed amplitude-scale robustness audit for MPL's \(B\).
- `results/mpl_ld_lag_response_audit/MODEL_CARD_ZH.md`: Chinese model card.
- `results/mpl_ld_lag_response_audit/THEORY_ZH.md`: Chinese derivation from MPL's cooldown term and a first-order response equation.
- `results/mpl_ld_lag_response_audit/parameter_ledger.csv`: parameter-source ledger; recommended model has 0 residual-fitted parameters.
- `results/mpl_ld_lag_response_audit/schedule_features.csv`: schedule-derived \(\tau_s\), \(a_s\), and cooldown support span.
- `results/mpl_ld_lag_response_audit/REPORT.md`: main audit report.
- `results/mpl_ld_lag_response_audit/strict_cosine_backbone/REPORT.md`: protocol audit separating frozen official MPL from strict cosine-only MPL.
- `results/mpl_ld_lag_response_audit/INTERPRETABILITY_RESET_ZH.md`: Chinese reset note demoting weakly interpretable residual models.
- `results/mpl_ld_lag_response_audit/ERROR_CURVES.md`: error-curve plot report.
- `results/mpl_ld_lag_response_audit/rule_sensitivity/REPORT.md`: rule sensitivity report.
- `results/mpl_ld_lag_response_audit/amplitude_sensitivity/REPORT.md`: fixed-scale amplitude sensitivity report.
- `results/mpl_ld_lag_response_audit/figs/finite_response_errors_25M.png`
- `results/mpl_ld_lag_response_audit/figs/finite_response_errors_100M.png`
- `results/mpl_ld_lag_response_audit/figs/finite_response_errors_400M.png`

## Interpretation Boundary

- This is the current cleanest interpretable candidate, not a final CCF-A-ready result.
- The frozen-official result is mechanism evidence, not a strict cosine-to-WSD deployment claim.
- Under strict cosine-only MPL, the formula improves all WSD-family rows but remains behind the official MPL baseline.
- The remaining weak point is the schedule-level adiabatic boundary \(a_s\); it is not fitted, but it is still a modeling prior.
- External validation on new LR schedules or new training runs is still missing.
- Observation-bracket MPL-LD remains a stronger numerical diagnostic, but not the main interpretable formula.
