# Next-Gen Target Safety Gate Audit

This audit evaluates a deployment-style target-localization gate for the next-generation `rho=0.5` kappa. The gate is target-only and schedule-label-free: abstain when `peak(phi_target) / mean(phi_target) < 2`.

## Summary

| mode | group | mean delta | worst delta | non-harm cells | wins | target factor |
|---|---|---:|---:|---:|---:|---:|
| `raw_nextgen` | main_matrix | -12.0% | -1.0% | 90/90 | 90/90 | 1.00 |
| `raw_nextgen` | extra_holdout | +2.4% | +21.8% | 36/54 | 0/54 | 1.00 |
| `raw_nextgen` | all | -6.6% | +21.8% | 126/144 | 90/144 | 1.00 |
| `target_localization_gate` | main_matrix | -11.6% | +0.0% | 90/90 | 75/90 | 0.83 |
| `target_localization_gate` | extra_holdout | +0.0% | +0.0% | 54/54 | 0/54 | 0.00 |
| `target_localization_gate` | all | -7.2% | +0.0% | 144/144 | 75/144 | 0.52 |

## Threshold Sensitivity

| threshold | group | mean delta | worst delta | non-harm cells | target factor |
|---:|---|---:|---:|---:|---:|
| 1.5 | all | -6.6% | +21.8% | 126/144 | 0.75 |
| 2.0 | all | -7.2% | +0.0% | 144/144 | 0.52 |
| 2.5 | all | -7.2% | +0.0% | 144/144 | 0.52 |
| 3.0 | all | -7.2% | +0.0% | 144/144 | 0.52 |
| 3.5 | all | -7.2% | +0.0% | 144/144 | 0.52 |
| 4.0 | all | -7.2% | +0.0% | 144/144 | 0.52 |
| 5.0 | all | -4.3% | +0.0% | 144/144 | 0.42 |
| 6.0 | all | -4.3% | +0.0% | 144/144 | 0.42 |

## Readout

Raw next-gen remains strong on the main matrix but fails the extra holdout group (worst `+21.8%`) because of `cosine_24000`. The target-localization gate makes the combined main-plus-extra audit non-harming (`144/144` cells, worst `+0.0%`). On the main matrix it preserves non-harm (`90/90` cells) but abstains on diffuse cosine targets, so wins are lower than raw transfer.

The threshold sweep shows that `1.5` does not remove the external holdout failure, while thresholds from `2.0` through `4.0` all give `144/144` non-harming cells with the same mean gain. Larger thresholds such as `5.0` and `6.0` are also safe but more conservative.

Interpretation: the raw next-gen formula is a strong transfer estimator when the target response direction is identifiable. The safety gate is a conservative deployment rule for target schedules whose response feature is too diffuse to distinguish from low-frequency MPL drift.
