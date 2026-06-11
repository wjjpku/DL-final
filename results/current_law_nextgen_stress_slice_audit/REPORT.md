# Next-Gen Stress-Slice Audit

This audit checks whether the next-generation safe formula only looks good after aggregation. It slices the same all-train-size matrix by scale, train-size, target curve, train group, and scale x train-size interaction.

## Mode Summary

| mode | rows | mean delta | worst delta | non-harm rows | wins | target factor |
|---|---:|---:|---:|---:|---:|---:|
| `no_predictive_shrinkage` | 1116 | -4.9% | +32.6% | 909/1116 | 537/1116 | 1.000 |
| `rho0p5_plus_Rtarget_gate` | 1116 | -5.9% | +0.0% | 1116/1116 | 558/1116 | 0.500 |
| `rho0p5_shrinkage` | 1116 | -4.8% | +22.5% | 930/1116 | 558/1116 | 1.000 |

## Safe Formula Slices

| slice | value | rows | mean delta | worst delta | non-harm rows | wins | target factor |
|---|---|---:|---:|---:|---:|---:|---:|
| scale | `100` | 372 | -4.9% | +0.0% | 372/372 | 186/372 | 0.500 |
| scale | `25` | 372 | -6.4% | +0.0% | 372/372 | 186/372 | 0.500 |
| scale | `400` | 372 | -6.4% | +0.0% | 372/372 | 186/372 | 0.500 |
| train_size | `1` | 144 | -7.5% | +0.0% | 144/144 | 90/144 | 0.625 |
| train_size | `2` | 315 | -6.4% | +0.0% | 315/315 | 180/315 | 0.571 |
| train_size | `3` | 360 | -5.9% | +0.0% | 360/360 | 180/360 | 0.500 |
| train_size | `4` | 225 | -5.0% | +0.0% | 225/225 | 90/225 | 0.400 |
| train_size | `5` | 72 | -3.1% | +0.0% | 72/72 | 18/72 | 0.250 |
| target_curve | `Constant 24k` | 186 | +0.0% | +0.0% | 186/186 | 0/186 | 0.000 |
| target_curve | `Constant 72k` | 186 | +0.0% | +0.0% | 186/186 | 0/186 | 0.000 |
| target_curve | `Cosine` | 93 | -2.3% | -1.0% | 93/93 | 93/93 | 1.000 |
| target_curve | `Cosine 24k` | 186 | +0.0% | +0.0% | 186/186 | 0/186 | 0.000 |
| target_curve | `WSD linear` | 93 | -9.9% | -4.8% | 93/93 | 93/93 | 1.000 |
| target_curve | `WSD sharp` | 93 | -11.8% | -6.1% | 93/93 | 93/93 | 1.000 |
| target_curve | `WSD-con 18e-5` | 93 | -8.1% | -2.8% | 93/93 | 93/93 | 1.000 |
| target_curve | `WSD-con 3e-5` | 93 | -28.6% | -16.3% | 93/93 | 93/93 | 1.000 |
| target_curve | `WSD-con 9e-5` | 93 | -10.1% | -1.4% | 93/93 | 93/93 | 1.000 |
| group | `extra_holdout` | 558 | +0.0% | +0.0% | 558/558 | 0/558 | 0.000 |
| group | `main_matrix` | 558 | -11.8% | -1.0% | 558/558 | 558/558 | 1.000 |

## Scale x Train-Size Interaction

| scale | train size | rows | mean delta | worst delta | non-harm rows | wins |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 1 | 48 | -6.1% | +0.0% | 48/48 | 30/48 |
| 100 | 2 | 105 | -5.3% | +0.0% | 105/105 | 60/105 |
| 100 | 3 | 120 | -4.9% | +0.0% | 120/120 | 60/120 |
| 100 | 4 | 75 | -4.2% | +0.0% | 75/75 | 30/75 |
| 100 | 5 | 24 | -2.6% | +0.0% | 24/24 | 6/24 |
| 25 | 1 | 48 | -8.2% | +0.0% | 48/48 | 30/48 |
| 25 | 2 | 105 | -7.2% | +0.0% | 105/105 | 60/105 |
| 25 | 3 | 120 | -6.4% | +0.0% | 120/120 | 60/120 |
| 25 | 4 | 75 | -5.3% | +0.0% | 75/75 | 30/75 |
| 25 | 5 | 24 | -3.4% | +0.0% | 24/24 | 6/24 |
| 400 | 1 | 48 | -8.1% | +0.0% | 48/48 | 30/48 |
| 400 | 2 | 105 | -6.9% | +0.0% | 105/105 | 60/105 |
| 400 | 3 | 120 | -6.4% | +0.0% | 120/120 | 60/120 |
| 400 | 4 | 75 | -5.5% | +0.0% | 75/75 | 30/75 |
| 400 | 5 | 24 | -3.4% | +0.0% | 24/24 | 6/24 |

## Worst Safe Rows

| delta | scale | train size | train | test | group | R_target | factor |
|---:|---:|---:|---|---|---|---:|---:|
| +0.0% | 100 | 1 | `Cosine` | `Cosine 24k` | extra_holdout | 0.005450 | 0.0 |
| +0.0% | 100 | 1 | `Cosine` | `Constant 24k` | extra_holdout | 0.000000 | 0.0 |
| +0.0% | 100 | 1 | `Cosine` | `Constant 72k` | extra_holdout | 0.000000 | 0.0 |
| +0.0% | 25 | 1 | `Cosine` | `Cosine 24k` | extra_holdout | 0.005450 | 0.0 |
| +0.0% | 25 | 1 | `Cosine` | `Constant 24k` | extra_holdout | 0.000000 | 0.0 |
| +0.0% | 25 | 1 | `Cosine` | `Constant 72k` | extra_holdout | 0.000000 | 0.0 |
| +0.0% | 400 | 1 | `Cosine` | `Cosine 24k` | extra_holdout | 0.005450 | 0.0 |
| +0.0% | 400 | 1 | `Cosine` | `Constant 24k` | extra_holdout | 0.000000 | 0.0 |
| +0.0% | 400 | 1 | `Cosine` | `Constant 72k` | extra_holdout | 0.000000 | 0.0 |
| +0.0% | 100 | 2 | `Cosine + WSD sharp` | `Cosine 24k` | extra_holdout | 0.005721 | 0.0 |
| +0.0% | 100 | 2 | `Cosine + WSD sharp` | `Constant 24k` | extra_holdout | 0.000000 | 0.0 |
| +0.0% | 100 | 2 | `Cosine + WSD sharp` | `Constant 72k` | extra_holdout | 0.000000 | 0.0 |

## Readout

The safe formula has `1116/1116` non-harming rows overall, worst `+0.0%`, and mean `-5.9%`. The audit found `0` slice failures: every audited one-dimensional safe slice is non-harming, with worst slice value `+0.0%`; every scale x train-size interaction is also non-harming, with worst `+0.0%`. By contrast, shrinkage without target gating has worst `+22.5%`, and no predictive shrinkage has worst `+32.6%`. The remaining zero-delta rows are deliberate abstentions on non-identifiable extra targets, not hidden positive failures.
