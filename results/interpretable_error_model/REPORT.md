# Interpretable Error Model Audit

This audit intentionally evaluates a small fixed set of mechanism candidates. All coefficients are fit from `cosine_72000.csv` residuals only; WSD-family curves are evaluation only.

## Interpretability Reset

After the finite-response contraction audit, the zero-parameter MPL-LD variant is now treated as a conservative mechanism lower bound, not the main model.  The current research-facing formula is restored to observation-bracket MPL-LD with MPL-LD tangent projection and sample-size ridge.  The DCT-projected rows in this report remain numerical references and ablations, not the core theory.

## Best WSD-Only Candidate

- Candidate: `obs_half_life_projected_2p5_roundfast20`.
- Mean / worst: `-34.56%` / `-5.30%`.
- Wins/non-harm: `15/15` / `15/15`.
- Fit start: `8000`, nuisance lambda: `0.01`.

## Candidate Summary

| candidate | kind | lambdas | fit start | mu | mean | worst | wins |
|---|---|---:|---:|---:|---:|---:|---:|
| obs_half_life_projected_2p5_roundfast20 | adaptive_observed_projected_raw | 2.5,20 | 8000 | 0.01 | -34.56% | -5.30% | 15/15 |
| continuous_lambda_projected_7_20 | adaptive_projected_raw | 7,20 | 8000 | 0.01 | -34.45% | -5.30% | 15/15 |
| obs_half_life_sqrtlocalized_projected_2p5_roundfast20 | adaptive_observed_sqrtlocalized_projected_raw | 2.5,20 | 8000 | 0.01 | -34.15% | -5.30% | 15/15 |
| obs_half_life_localized_projected_2p5_roundfast20 | adaptive_observed_localized_projected_raw | 2.5,20 | 8000 | 0.01 | -32.83% | -5.30% | 15/15 |
| obs_half_life_sqrtlocalized_projected_2p5_roundfast20 | adaptive_observed_sqrtlocalized_projected_raw | 2.5,20 | 5000 | 0.01 | -32.18% | -4.09% | 15/15 |
| obs_half_life_projected_2p5_roundfast20 | adaptive_observed_projected_raw | 2.5,20 | 5000 | 0.01 | -31.72% | -4.09% | 15/15 |
| obs_half_life_localized_projected_2p5_roundfast20 | adaptive_observed_localized_projected_raw | 2.5,20 | 5000 | 0.01 | -31.54% | -4.10% | 15/15 |
| continuous_lambda_projected_7_20 | adaptive_projected_raw | 7,20 | 5000 | 0.01 | -31.43% | -4.09% | 15/15 |
| continuous_lambda_projected_4_20 | adaptive_projected_raw | 4,20 | 8000 | 0.01 | -24.76% | -5.30% | 15/15 |
| continuous_lambda_projected_4_20 | adaptive_projected_raw | 4,20 | 5000 | 0.01 | -22.30% | -2.63% | 15/15 |
| raw_drop_l20 | raw | 20 | 8000 | 0.01 | -22.06% | -5.30% | 15/15 |
| raw_drop_l20 | raw | 20 | 5000 | 0.01 | -20.99% | -4.09% | 15/15 |
| lag_rawslope_l20 | lag_raw | 20 | 5000 | 0.01 | -0.96% | -0.14% | 15/15 |
| lag_rawslope_l20 | lag_raw | 20 | 8000 | 0.01 | -0.90% | -0.12% | 15/15 |
| continuous_lambda_raw_7_20 | adaptive_raw | 7,20 | 8000 | 0.01 | -26.02% | +4.65% | 13/15 |
| continuous_lambda_raw_7_20 | adaptive_raw | 7,20 | 5000 | 0.01 | -22.77% | +3.81% | 13/15 |
| continuous_lambda_raw_4_20 | adaptive_raw | 4,20 | 8000 | 0.01 | -15.33% | +5.43% | 13/15 |
| obs_half_life_localized_projected_2p5_roundfast20 | adaptive_observed_localized_projected_raw | 2.5,20 | 8000 | 0.02 | -13.88% | +16.36% | 10/15 |
| continuous_lambda_raw_4_20 | adaptive_raw | 4,20 | 5000 | 0.01 | -13.81% | +2.23% | 13/15 |
| obs_half_life_sqrtlocalized_projected_2p5_roundfast20 | adaptive_observed_sqrtlocalized_projected_raw | 2.5,20 | 8000 | 0.02 | -10.92% | +16.37% | 10/15 |

## Per-Target Result For Best Candidate

| target | mean delta | worst scale | wins |
|---|---:|---:|---:|
| WSD sharp | -51.51% | -36.50% | 3/3 |
| WSD linear | -46.06% | -33.84% | 3/3 |
| WSD-con 3e-5 | -53.08% | -41.40% | 3/3 |
| WSD-con 9e-5 | -13.04% | -5.30% | 3/3 |
| WSD-con 18e-5 | -9.11% | -6.43% | 3/3 |

## Localized Control-Safety Ablation

`obs_half_life_projected_2p5_roundfast20` is the WSD-only upper variant.  `obs_half_life_sqrtlocalized_projected_2p5_roundfast20` multiplies the transferred correction by the square root of a continuous LR-drop localization factor, which removes the short-cosine control failure without adding fitted parameters.  It is useful evidence, but no longer recommended as the main formula.

| variant | mean | worst | wins |
|---|---:|---:|---:|
| sqrt-localized ablation | -34.15% | -5.30% | 15/15 |

See `results/interpretable_strict_vs_rounded/REPORT.md` for the constant and short-cosine control audit.

## Holdout Selection Check

| split | selected candidate | dev mean | dev worst | test mean | test worst | test wins |
|---|---|---:|---:|---:|---:|---:|
| dev_sharp_linear__test_wsdcon | `obs_half_life_projected_2p5_roundfast20 / start=8000, mu=0.01` | -48.79% | -33.84% | -25.07% | -5.30% | 9/9 |
| dev_wsdcon__test_sharp_linear | `obs_half_life_localized_projected_2p5_roundfast20 / start=8000, mu=0.01` | -25.08% | -5.30% | -44.45% | -30.83% | 6/6 |
| leave_target__WSD sharp | `continuous_lambda_projected_7_20 / start=8000, mu=0.01` | -30.35% | -5.30% | -50.85% | -35.17% | 3/3 |
| leave_target__WSD linear | `obs_half_life_sqrtlocalized_projected_2p5_roundfast20 / start=8000, mu=0.01` | -31.74% | -5.30% | -43.80% | -32.99% | 3/3 |
| leave_target__WSD-con 3e-5 | `obs_half_life_projected_2p5_roundfast20 / start=8000, mu=0.01` | -29.93% | -5.30% | -53.08% | -41.40% | 3/3 |
| leave_target__WSD-con 9e-5 | `obs_half_life_projected_2p5_roundfast20 / start=8000, mu=0.01` | -39.94% | -6.43% | -13.04% | -5.30% | 3/3 |
| leave_target__WSD-con 18e-5 | `obs_half_life_projected_2p5_roundfast20 / start=8000, mu=0.01` | -40.92% | -5.30% | -9.11% | -6.43% | 3/3 |
| leave_scale__25M | `obs_half_life_projected_2p5_roundfast20 / start=8000, mu=0.01` | -35.85% | -5.30% | -31.98% | -12.44% | 5/5 |
| leave_scale__100M | `obs_half_life_sqrtlocalized_projected_2p5_roundfast20 / start=8000, mu=0.01` | -35.57% | -6.43% | -31.31% | -5.30% | 5/5 |
| leave_scale__400M | `obs_half_life_projected_2p5_roundfast20 / start=8000, mu=0.01` | -32.34% | -5.30% | -39.00% | -6.43% | 5/5 |

## Reading

- This file is now a historical fixed-candidate audit.  Use `MODEL_DECISION.md` for the current interpretable formula.
- `raw_drop` is the minimal causal LR-drop response baseline inside the old DCT-projected audit.
- `obs_half_life_projected` replaces the unexplained exact `7/20` endpoints with observable response half-life anchors.  The modal loss-curve interval is 128 steps; the slow endpoint is a 2.5-interval half-life, and the fast endpoint is rounded to the one-interval response rate.
- The target LR schedule determines a continuous response rate through drop concentration, and the matching response operator is calibrated only on cosine residuals.
- The projected continuous model recovers most of the previous adaptive-fit-window gain while avoiding discrete smooth/step routing.
- The sqrt-localized variant adds a parameter-free schedule-locality amplitude factor `sqrt(1 - drop_support_span / post_warmup_span)`; it is now treated as a control-safety ablation rather than the main formula.
- DCT nuisance is not mechanism-native enough for the main story.  The stronger current interpretation projects out MPL LR-dependent tangent directions instead.
- `lag_rawslope` and unrelated MPL-internal sensitivity directions do not beat the response baseline; they should stay as negative evidence, not main-method components.
- The remaining weak points are ridge identifiability, the `2.5` slow response prior, locality as a boundary condition, and external validation.
