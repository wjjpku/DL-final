# Parameter Origin Audit

This audit asks whether the response-time constants can be explained from observable quantities rather than WSD target loss.  Every coefficient is still fitted from `cosine_72000.csv` residuals only.

## Observed-Time Anchor

- Modal loss-curve observation interval: `128` training steps.
- One-observation half-life in LR time: `lambda_obs = ln(2) / (eta_peak * 128) = 18.0507`.
- This gives a direct interpretation of the old fast endpoint: `20` is close to one observable-interval half-life.
- A slower smooth-decay endpoint can be read as a 2.5-observation half-life, giving `lambda_obs / 2.5`.

## Summary

| method | mean | worst | wins | note |
|---|---:|---:|---:|---|
| obs_half_life_2p5_roundfast20 | -34.56% | -5.30% | 15/15 | slow endpoint from 2.5 observed intervals; fast endpoint rounded to 20 |
| current_q_7_20 | -34.45% | -5.30% | 15/15 | current development endpoint rule |
| obs_half_life_2p5_exact | -31.97% | -1.09% | 15/15 | endpoints from 2.5 and 1 observed-interval half-lives |
| fixed_lr_lambda_20 | -22.06% | -5.30% | 15/15 | one universal fast LR-time response |
| fixed_lr_lambda_7 | +22.34% | +137.85% | 7/15 | one universal slow LR-time response |
| step_time_geometry_tau | +30.39% | +58.56% | 1/15 | step-time contrast rule; target tau from LR geometry, coefficient from cosine only |
| cosine_source_selected_lambda | +200.29% | +452.79% | 0/15 | lambda chosen by cosine residual fit objective only |

## Lambda Source Diagnostics

| source | scale | selected lambda | source objective | source coef |
|---|---:|---:|---:|---:|
| cosine_source_fit_grid | 25 | 1 | 0.000184605 | 0.0448823 |
| cosine_source_fit_grid | 100 | 1 | 0.000208175 | 0.0575191 |
| cosine_source_fit_grid | 400 | 1 | 0.000267773 | 0.0840224 |

## Observation-Half-Life Sensitivity

| slow half-life multiplier | fast endpoint | slow lambda | fast lambda | mean | worst | wins |
|---:|---|---:|---:|---:|---:|---:|
| 2.00 | exact_lambda_obs | 9.0254 | 18.0507 | -30.08% | -1.09% | 15/15 |
| 2.00 | rounded_20 | 9.0254 | 20.0000 | -32.66% | -5.30% | 15/15 |
| 2.25 | exact_lambda_obs | 8.0225 | 18.0507 | -31.55% | -1.09% | 15/15 |
| 2.25 | rounded_20 | 8.0225 | 20.0000 | -34.14% | -5.30% | 15/15 |
| 2.50 | exact_lambda_obs | 7.2203 | 18.0507 | -31.97% | -1.09% | 15/15 |
| 2.50 | rounded_20 | 7.2203 | 20.0000 | -34.56% | -5.30% | 15/15 |
| 2.75 | exact_lambda_obs | 6.5639 | 18.0507 | -31.04% | -1.09% | 15/15 |
| 2.75 | rounded_20 | 6.5639 | 20.0000 | -33.63% | -5.30% | 15/15 |
| 3.00 | exact_lambda_obs | 6.0169 | 18.0507 | -29.67% | -1.09% | 15/15 |
| 3.00 | rounded_20 | 6.0169 | 20.0000 | -32.26% | -5.30% | 15/15 |
| 3.50 | exact_lambda_obs | 5.1573 | 18.0507 | -26.86% | -1.09% | 15/15 |
| 3.50 | rounded_20 | 5.1573 | 20.0000 | -29.45% | -5.30% | 15/15 |
| 4.00 | exact_lambda_obs | 4.5127 | 18.0507 | -24.39% | -1.09% | 15/15 |
| 4.00 | rounded_20 | 4.5127 | 20.0000 | -26.98% | -5.30% | 15/15 |

## Per-Target For Observation-Derived Rule

| target | mean | worst | wins |
|---|---:|---:|---:|
| WSD linear | -46.06% | -33.84% | 3/3 |
| WSD sharp | -51.51% | -36.50% | 3/3 |
| WSD-con 18e-5 | -9.11% | -6.43% | 3/3 |
| WSD-con 3e-5 | -53.08% | -41.40% | 3/3 |
| WSD-con 9e-5 | -13.04% | -5.30% | 3/3 |

## Reading

- Source-loss selection of `lambda` is not trustworthy: it selects a very slow cosine response and fails on every WSD-family target.  This is direct evidence of low-frequency MPL-drift contamination.
- A universal LR-time response is safer only around the fast endpoint, but it leaves large smooth-decay gains on the table.
- Step-time geometry is interpretable in same-family audits, but with cosine as the only calibration source it over-transfers long-memory corrections to WSD and fails this specific task.
- The best current explanation for the remaining endpoint constants is observable response half-life: fast step corrections should be resolvable within roughly one logged interval, while smooth-decay corrections need a few intervals to be identifiable.
- The rounded observation-derived rule remains all-win, so the main formula no longer depends on an unexplained exact `7/20` choice.
