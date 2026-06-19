# Interpretable Error Model Checkpoint

Date: 2026-06-19

## Current Status

2026-06-19 theory-refined 状态：`q2 half-life MPL-LD response` 是当前推荐主候选。它保持核心架构不变：MPL baseline + 一个 causal LR-drop response + 一个 cosine-fitted nonnegative amplitude。相比恢复版 observation-bracket，它只改 schedule-only 规则：用 Herfindahl concentration \(q_s=\sum_t(d_t/\sum_u d_u)^2\) 表示 effective drop count，用 \(H_s=(2-q_s)\Delta_{\mathrm{obs}}\) 给出 half-life bracket，并把 \(a_s\) 解释为 support-projection locality。零参数 MPL-LD finite-response 版本解释性更干净但性能太弱，现在只作为机制下界和负控。

Current main candidate:

```text
L_hat_s(t) = L_MPL,s(t) + a_s * kappa_hat_s * phi_{lambda_s,s}(t)
```

where `phi_{lambda_s,s}` is a causal LR-drop response, `lambda_s = lambda_obs / (2 - q2_s)`, `kappa_hat_s` is the only residual-fitted scalar, and `a_s` is a schedule-only support-projection locality factor.

Conservative finite-response lower bound:

```text
L_hat_tau(t) = L_MPL(t) + a_s * B * (D_down_tau(t) - D_down(t))
```

This lower-bound formula is useful but not strong enough: it has cleaner semantics but weaker WSD gains and poor strict-backbone absolute performance.

Recommended writing choices:

- `hhi_q2_halflife_support_projection`: current recommended formula.
  It removes only MPL LR-dependent tangent directions, uses
  `lambda_obs / (2 - q2_s)`, and replaces fixed `tau=0.05` with
  `1 / N_cal`.  Its locality factor is support-projection energy, not a
  learned gate.
- `observation_bracket_mplld_neff`: previous restored diagnostic reference.
  Numerically almost identical, but its max-drop concentration is less
  interpretable than q2.
- `direct_mpl_ld_lag`: conservative finite-response lower bound. It changes only
  MPL's existing LR-dependent term and adds no residual amplitude in fixed-tau
  rows, but it is not strong enough as the main candidate.
- `mpl_ld_tangent`: previous mechanism-native reference with fixed `tau=0.05`
  and old `2.5`/`20` response endpoints.
- `dct_performance`: stronger same-scale numerical reference, but not the main
  explanation because the nuisance basis is generic low-frequency DCT.
- `tau_free_dct` / `tau_free_sqrt_retention`: useful ablation showing that a response mechanism can work
  without a fixed ridge constant, but still not the preferred core story because
  it depends on DCT nuisance filtering.
- `rounded_fast20_sqrtlocalized`: retained only as an ablation; no longer
  recommended as the paper/slides main formula.

## Reproduction Commands

```bash
python3 repro/interpretable_core_decision.py
python3 repro/interpretable_error_model.py
python3 repro/interpretable_shrinkage_origin_audit.py
python3 repro/interpretable_nuisance_origin_audit.py
python3 repro/interpretable_scale_stability_audit.py
python3 repro/interpretable_observation_bracket_audit.py
python3 repro/interpretable_theory_refinement_audit.py
python3 repro/plot_interpretable_error_model.py
python3 repro/interpretable_parameter_origin_audit.py
python3 repro/interpretable_protocol_sensitivity.py
python3 repro/interpretable_strict_vs_rounded_audit.py
python3 repro/interpretable_localization_sensitivity.py
python3 repro/mpl_ld_lag_response_audit.py
python3 repro/plot_mpl_ld_lag_response.py
python3 repro/mpl_ld_lag_rule_sensitivity.py
python3 repro/mpl_ld_lag_amplitude_sensitivity.py
```

Syntax gate:

```bash
python3 -m py_compile \
  repro/interpretable_core_decision.py \
  repro/interpretable_error_model.py \
  repro/interpretable_shrinkage_origin_audit.py \
  repro/interpretable_nuisance_origin_audit.py \
  repro/interpretable_scale_stability_audit.py \
  repro/interpretable_observation_bracket_audit.py \
  repro/interpretable_theory_refinement_audit.py \
  repro/plot_interpretable_error_model.py \
  repro/interpretable_parameter_origin_audit.py \
  repro/interpretable_protocol_sensitivity.py \
  repro/interpretable_strict_vs_rounded_audit.py \
  repro/interpretable_localization_sensitivity.py \
  repro/mpl_ld_lag_response_audit.py \
  repro/plot_mpl_ld_lag_response.py \
  repro/mpl_ld_lag_rule_sensitivity.py \
  repro/mpl_ld_lag_amplitude_sensitivity.py
```

## Key Artifacts

- `repro/interpretable_core_decision.py`: interpretability-first decision report.
- `repro/interpretable_error_model.py`: fixed-candidate main audit.
- `repro/interpretable_shrinkage_origin_audit.py`: tau-free shrinkage and ridge-origin audit.
- `repro/interpretable_nuisance_origin_audit.py`: DCT-vs-MPL-tangent nuisance audit.
- `repro/interpretable_scale_stability_audit.py`: cross-scale stability audit for MPL-LD tangent vs DCT references.
- `repro/interpretable_observation_bracket_audit.py`: restored parameter-light observation-bracket MPL-LD reference.
- `repro/interpretable_theory_refinement_audit.py`: q2 half-life and support-projection refinement audit.
- `repro/plot_interpretable_error_model.py`: residual-error visualization for current interpretable variants.
- `repro/interpretable_parameter_origin_audit.py`: endpoint-origin and negative-source-selection audit.
- `repro/interpretable_protocol_sensitivity.py`: fit-start, nuisance, ridge sensitivity audit.
- `repro/interpretable_strict_vs_rounded_audit.py`: exact-vs-rounded endpoint and extra-control audit.
- `repro/mpl_ld_lag_response_audit.py`: direct MPL-LD finite-response audit.
- `repro/plot_mpl_ld_lag_response.py`: error-curve visualization for the finite-response candidates.
- `repro/mpl_ld_lag_rule_sensitivity.py`: schedule-only tau/boundary rule sensitivity audit.
- `repro/mpl_ld_lag_amplitude_sensitivity.py`: fixed amplitude-scale robustness audit for the use of MPL's \(B\).
- `results/interpretable_error_model/INTERPRETABILITY_RESET.md`: Chinese reset memo kept as historical context; superseded by the 2026-06-19 interpretability audit.
- `results/interpretable_error_model/MODEL_DECISION.md`: current Chinese model decision note; this supersedes earlier DCT-first wording.
- `results/interpretable_error_model/INTERPRETABILITY_AUDIT_2026_06_19.md`: Chinese audit that downgrades observation-bracket MPL-LD and defines the cleaner finite-response direction.
- `results/interpretable_error_model/REPORT.md`: main result summary.
- `results/interpretable_error_model/FORMULA_CARD.md`: current formula card.
- `results/interpretable_error_model/THEORY.md`: proposition-style theory note.
- `results/interpretable_error_model/RESEARCH_LOG.md`: Chinese research log and decision trail.
- `results/interpretable_error_model/OPEN_LIMITATIONS.md`: current unresolved limits and safe claim boundary.
- `results/interpretable_shrinkage_origin_audit/REPORT.md`: tau-free baseline audit.
- `results/interpretable_nuisance_origin_audit/REPORT.md`: nuisance-origin audit.
- `results/interpretable_scale_stability_audit/REPORT.md`: scale-transfer/stability audit.
- `results/interpretable_observation_bracket_audit/REPORT.md`: restored observation-bracket MPL-LD reference audit.
- `results/interpretable_theory_refinement/REPORT.md`: current q2 half-life formula decision and ablation.
- `results/interpretable_observation_bracket_audit/parameter_ledger.csv`: fitted/derived quantity ledger; only `kappa_hat_s` is fitted by the residual model.
- `results/interpretable_observation_bracket_audit/locality_boundary.csv`: schedule-only locality factors and control-boundary audit.
- `results/interpretable_error_model/error_comparison/REPORT.md`: residual-error plots and per-target visual summary.
- `results/interpretable_parameter_origin_audit/REPORT.md`: endpoint-origin audit.
- `results/interpretable_protocol_sensitivity/REPORT.md`: protocol sensitivity audit.
- `results/interpretable_localization_sensitivity/REPORT.md`: locality-shape sensitivity audit.
- `results/mpl_ld_lag_response_audit/REPORT.md`: fixed-tau direct MPL-LD lag results and cosine-amplitude contamination check.
- `results/mpl_ld_lag_response_audit/ERROR_CURVES.md`: finite-response error-curve plot summary.
- `results/mpl_ld_lag_response_audit/ARCHIVE_MANIFEST.md`: reproducibility archive for the current finite-response checkpoint.
- `results/mpl_ld_lag_response_audit/THEORY_ZH.md`: derivation from MPL's cooldown term and a first-order response equation.
- `results/mpl_ld_lag_response_audit/rule_sensitivity/REPORT.md`: support-bracket tau and boundary sensitivity audit.
- `results/mpl_ld_lag_response_audit/amplitude_sensitivity/REPORT.md`: robustness audit showing `amplitude_scale=1` is not an isolated exact-B accident.

## Current Numbers

Current finite-response baseline on 15 WSD-family scale-target rows:

| variant | fitted residual coefficient | mean | worst | wins | control worst |
|---|---:|---:|---:|---:|---:|
| cooldown support-bracket MPL-LD lag | 0 | -13.77% | -6.29% | 15/15 | +0.00% |
| cooldown adiabatic MPL-LD lag, tau=64 | 0 | -2.91% | -1.96% | 15/15 | +0.00% |
| cooldown adiabatic MPL-LD lag, tau=128 | 0 | -8.73% | -6.22% | 15/15 | +0.00% |
| direct MPL-LD lag, tau=64 | 0 | -3.11% | -2.38% | 15/15 | +1.62% |
| direct MPL-LD lag, tau=128 | 0 | -9.52% | -5.95% | 15/15 | +6.01% |
| direct MPL-LD lag, tau=256 | 0 | -15.09% | +18.57% | 14/15 | +15.90% |
| MPL-LD lag, tau=128, cosine-fitted amplitude | 1 | +565.16% | +1314.46% | 0/15 | +656.40% |

Observation-bracket diagnostic variants on 15 WSD-family scale-target rows:

| variant | role | mean | worst | wins |
|---|---|---:|---:|---:|
| q2 half-life MPL-LD + support projection | current recommended formula | -29.88% | -4.67% | 15/15 |
| observation-bracket MPL-LD + sample-size ridge | previous restored reference | -29.87% | -4.67% | 15/15 |
| no-nuisance raw projection | failure mode | +602.17% | +2366.35% | 0/15 |
| MPL-LD tangent nuisance + fixed ridge | previous mechanism-native reference | -27.25% | -3.00% | 15/15 |
| MPL-LD tangent nuisance + ridge, no locality | WSD-only core without boundary term | -24.86% | -3.00% | 15/15 |
| DCT performance extension + ridge | numerical reference, not core explanation | -32.83% | -5.30% | 15/15 |
| fixed20 tau-free sqrt-retention + DCT | tau-free DCT ablation | -20.77% | -5.86% | 15/15 |
| fixed20 tau-free full-energy + DCT | conservative DCT lower bound | -3.72% | -1.30% | 15/15 |
| fixed_lambda_obs | minimal sanity | -20.55% | -1.09% | 15/15 |
| fixed_lambda_20 | minimal rounded | -22.06% | -5.30% | 15/15 |
| strict_exact | historical DCT-based reference | -31.97% | -1.09% | 15/15 |
| rounded_fast20 | performance variant | -34.56% | -5.30% | 15/15 |
| rounded_fast20_localized | optional control-safety | -32.83% | -5.30% | 15/15 |

Theory-refinement controls and cross-scale:

| variant | cross-scale WSD | controls |
|---|---:|---:|
| q2 half-life MPL-LD + support projection | -24.95%, 30/30 | 9/9 non-harm |
| q2 density projection | -24.93%, 30/30 | worst +8.25% |
| q2 no locality | -24.60%, 30/30 | worst +56.99% |

Scale-stability audit:

| method | same-scale WSD | cross-scale WSD | cross-scale worst |
|---|---:|---:|---:|
| observation-bracket MPL-LD | -29.87%, 15/15 | -24.95%, 30/30 | -3.15% |
| MPL-LD tangent fixed-tau reference | -27.25%, 15/15 | -23.07%, 30/30 | -2.07% |
| DCT performance | -32.83%, 15/15 | -18.98%, 26/30 | +26.68% |
| tau-free DCT | -20.77%, 15/15 | -13.27%, 27/30 | +9.04% |

Extra controls:

| variant | control behavior |
|---|---:|
| fixed20 tau-free sqrt-retention + linear locality | 9/9 non-harm |
| observation-bracket MPL-LD + sample-size ridge | 9/9 non-harm |
| observation-bracket MPL-LD, no locality | +13.39% mean, +56.99% worst |
| MPL-LD tangent nuisance + ridge | 9/9 non-harm |
| MPL-LD tangent nuisance + ridge, no locality | +18.92% mean, +84.55% worst |
| DCT performance extension + linear locality | 9/9 non-harm |
| rounded_fast20 | +14.02% mean, +56.43% worst |
| rounded_fast20_localized | 9/9 non-harm |
| rounded_fast20_sqrtlocalized | 9/9 non-harm, ablation only |

Important negative evidence:

| rejected direction | result | reason |
|---|---:|---|
| cosine-source objective selects lambda | +200.29% mean, 0/15 wins | learns low-frequency MPL drift |
| step-time geometry tau with cosine-only source | +30.39% mean, 1/15 wins | over-transfers long memory |
| ridge tau too small | fails up to +408.55% worst | raw projection over-amplifies weakly identifiable response |
| unlocalized transfer to Cosine 24k | +42.06% mean, +56.43% worst | treats full-run diffuse cosine decay as a local cooldown transient |
| MPL-LD lag amplitude fitted from cosine residual | +565.16% mean, 0/15 wins | cosine residual amplitude absorbs global MPL drift |

## Remaining Work Before Slides

- Freeze the q2 half-life MPL-LD response protocol before any paper/slides update.
- Keep direct MPL-LD finite-response as a clean lower bound unless its performance can be improved without losing interpretability.
- Keep `sqrt-localized`, gate, channel, sine, and curvature variants out of the main story unless new evidence gives a cleaner derivation.
- Tighten the theory wording for MPL-LD finite response, controls harm, and response-time selection.
- Add external schedule or new-run validation if available.
