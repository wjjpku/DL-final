# Target Identifiability Attenuation Audit

This audit asks whether the next-generation `rho=0.5` kappa can use a continuous target-side identifiability factor instead of a binary peak/mean abstention gate. For each target schedule, the factor is based on the response-energy retention after applying the same soft DCT/Sobolev nuisance residualizer used during kappa estimation.

Candidate factor:

```text
R_target(lambda) = ||M_lambda phi_target||^2 / ||phi_target||^2
kappa_safe = R_target(lambda)^alpha * kappa_transfer
```

## Summary

| mode | group | mean delta | worst delta | non-harm cells | wins | target factor | retention |
|---|---|---:|---:|---:|---:|---:|---:|
| `floor_train_relative_gate_0p05` | main_matrix | -11.8% | +0.0% | 558/558 | 549/558 | 0.984 | 0.321 |
| `floor_train_relative_gate_0p05` | extra_holdout | +0.0% | +0.0% | 558/558 | 0/558 | 0.000 | 0.002 |
| `floor_train_relative_gate_0p05` | all | -5.9% | +0.0% | 1116/1116 | 549/1116 | 0.492 | 0.162 |
| `floor_train_relative_gate_0p1` | main_matrix | -11.5% | +0.0% | 558/558 | 482/558 | 0.864 | 0.321 |
| `floor_train_relative_gate_0p1` | extra_holdout | +0.0% | +0.0% | 558/558 | 0/558 | 0.000 | 0.002 |
| `floor_train_relative_gate_0p1` | all | -5.7% | +0.0% | 1116/1116 | 482/1116 | 0.432 | 0.162 |
| `floor_train_relative_gate_0p2` | main_matrix | -11.4% | +0.0% | 558/558 | 465/558 | 0.833 | 0.321 |
| `floor_train_relative_gate_0p2` | extra_holdout | +0.0% | +0.0% | 558/558 | 0/558 | 0.000 | 0.002 |
| `floor_train_relative_gate_0p2` | all | -5.7% | +0.0% | 1116/1116 | 465/1116 | 0.417 | 0.162 |
| `floor_train_relative_gate_0p5` | main_matrix | -10.7% | +0.0% | 558/558 | 438/558 | 0.785 | 0.321 |
| `floor_train_relative_gate_0p5` | extra_holdout | +0.0% | +0.0% | 558/558 | 0/558 | 0.000 | 0.002 |
| `floor_train_relative_gate_0p5` | all | -5.4% | +0.0% | 1116/1116 | 438/1116 | 0.392 | 0.162 |
| `floor_train_relative_gate_1p0` | main_matrix | -9.7% | +0.0% | 558/558 | 389/558 | 0.697 | 0.321 |
| `floor_train_relative_gate_1p0` | extra_holdout | +0.0% | +0.0% | 558/558 | 0/558 | 0.000 | 0.002 |
| `floor_train_relative_gate_1p0` | all | -4.8% | +0.0% | 1116/1116 | 389/1116 | 0.349 | 0.162 |
| `peak_mean_gate` | main_matrix | -11.4% | +0.0% | 558/558 | 465/558 | 0.833 | 0.321 |
| `peak_mean_gate` | extra_holdout | +0.0% | +0.0% | 558/558 | 0/558 | 0.000 | 0.002 |
| `peak_mean_gate` | all | -5.7% | +0.0% | 1116/1116 | 465/1116 | 0.417 | 0.162 |
| `raw_nextgen` | main_matrix | -11.8% | -1.0% | 558/558 | 558/558 | 1.000 | 0.321 |
| `raw_nextgen` | extra_holdout | +2.2% | +22.5% | 372/558 | 0/558 | 1.000 | 0.002 |
| `raw_nextgen` | all | -4.8% | +22.5% | 930/1116 | 558/1116 | 1.000 | 0.162 |
| `retention_alpha_0p25` | main_matrix | -9.3% | -0.4% | 558/558 | 558/558 | 0.698 | 0.321 |
| `retention_alpha_0p25` | extra_holdout | +0.6% | +6.2% | 372/558 | 0/558 | 0.090 | 0.002 |
| `retention_alpha_0p25` | all | -4.4% | +6.2% | 930/1116 | 558/1116 | 0.394 | 0.162 |
| `retention_alpha_0p5` | main_matrix | -7.3% | -0.1% | 558/558 | 558/558 | 0.519 | 0.321 |
| `retention_alpha_0p5` | extra_holdout | +0.2% | +1.7% | 372/558 | 0/558 | 0.024 | 0.002 |
| `retention_alpha_0p5` | all | -3.6% | +1.7% | 930/1116 | 558/1116 | 0.272 | 0.162 |
| `retention_alpha_0p75` | main_matrix | -5.8% | -0.0% | 558/558 | 558/558 | 0.402 | 0.321 |
| `retention_alpha_0p75` | extra_holdout | +0.0% | +0.5% | 372/558 | 0/558 | 0.007 | 0.002 |
| `retention_alpha_0p75` | all | -2.9% | +0.5% | 930/1116 | 558/1116 | 0.204 | 0.162 |
| `retention_alpha_1p0` | main_matrix | -4.6% | -0.0% | 558/558 | 558/558 | 0.321 | 0.321 |
| `retention_alpha_1p0` | extra_holdout | +0.0% | +0.1% | 372/558 | 0/558 | 0.002 | 0.002 |
| `retention_alpha_1p0` | all | -2.3% | +0.1% | 930/1116 | 558/1116 | 0.162 | 0.162 |
| `retention_alpha_1p5` | main_matrix | -3.0% | -0.0% | 558/558 | 558/558 | 0.219 | 0.321 |
| `retention_alpha_1p5` | extra_holdout | +0.0% | +0.0% | 372/558 | 0/558 | 0.000 | 0.002 |
| `retention_alpha_1p5` | all | -1.5% | +0.0% | 930/1116 | 558/1116 | 0.109 | 0.162 |
| `retention_alpha_2p0` | main_matrix | -2.1% | -0.0% | 558/558 | 558/558 | 0.158 | 0.321 |
| `retention_alpha_2p0` | extra_holdout | +0.0% | +0.0% | 372/558 | 0/558 | 0.000 | 0.002 |
| `retention_alpha_2p0` | all | -1.1% | +0.0% | 930/1116 | 558/1116 | 0.079 | 0.162 |
| `retention_gate_0p0025` | main_matrix | -11.8% | -1.0% | 558/558 | 558/558 | 1.000 | 0.321 |
| `retention_gate_0p0025` | extra_holdout | +2.2% | +22.5% | 372/558 | 0/558 | 0.333 | 0.002 |
| `retention_gate_0p0025` | all | -4.8% | +22.5% | 930/1116 | 558/1116 | 0.667 | 0.162 |
| `retention_gate_0p005` | main_matrix | -11.8% | -1.0% | 558/558 | 558/558 | 1.000 | 0.321 |
| `retention_gate_0p005` | extra_holdout | +1.7% | +22.5% | 423/558 | 0/558 | 0.242 | 0.002 |
| `retention_gate_0p005` | all | -5.0% | +22.5% | 981/1116 | 558/1116 | 0.621 | 0.162 |
| `retention_gate_0p0075` | main_matrix | -11.8% | -1.0% | 558/558 | 558/558 | 1.000 | 0.321 |
| `retention_gate_0p0075` | extra_holdout | +0.0% | +0.0% | 558/558 | 0/558 | 0.000 | 0.002 |
| `retention_gate_0p0075` | all | -5.9% | +0.0% | 1116/1116 | 558/1116 | 0.500 | 0.162 |
| `retention_gate_0p01` | main_matrix | -11.8% | -1.0% | 558/558 | 558/558 | 1.000 | 0.321 |
| `retention_gate_0p01` | extra_holdout | +0.0% | +0.0% | 558/558 | 0/558 | 0.000 | 0.002 |
| `retention_gate_0p01` | all | -5.9% | +0.0% | 1116/1116 | 558/1116 | 0.500 | 0.162 |
| `retention_gate_0p015` | main_matrix | -11.7% | +0.0% | 558/558 | 540/558 | 0.968 | 0.321 |
| `retention_gate_0p015` | extra_holdout | +0.0% | +0.0% | 558/558 | 0/558 | 0.000 | 0.002 |
| `retention_gate_0p015` | all | -5.9% | +0.0% | 1116/1116 | 540/1116 | 0.484 | 0.162 |
| `retention_gate_0p02` | main_matrix | -11.6% | +0.0% | 558/558 | 513/558 | 0.919 | 0.321 |
| `retention_gate_0p02` | extra_holdout | +0.0% | +0.0% | 558/558 | 0/558 | 0.000 | 0.002 |
| `retention_gate_0p02` | all | -5.8% | +0.0% | 1116/1116 | 513/1116 | 0.460 | 0.162 |
| `retention_gate_0p05` | main_matrix | -11.4% | +0.0% | 558/558 | 465/558 | 0.833 | 0.321 |
| `retention_gate_0p05` | extra_holdout | +0.0% | +0.0% | 558/558 | 0/558 | 0.000 | 0.002 |
| `retention_gate_0p05` | all | -5.7% | +0.0% | 1116/1116 | 465/1116 | 0.417 | 0.162 |
| `retention_gate_0p1` | main_matrix | -11.4% | +0.0% | 558/558 | 465/558 | 0.833 | 0.321 |
| `retention_gate_0p1` | extra_holdout | +0.0% | +0.0% | 558/558 | 0/558 | 0.000 | 0.002 |
| `retention_gate_0p1` | all | -5.7% | +0.0% | 1116/1116 | 465/1116 | 0.417 | 0.162 |
| `train_relative_gate_0p05` | main_matrix | -11.8% | +0.0% | 558/558 | 549/558 | 0.984 | 0.321 |
| `train_relative_gate_0p05` | extra_holdout | +1.2% | +22.5% | 465/558 | 0/558 | 0.167 | 0.002 |
| `train_relative_gate_0p05` | all | -5.3% | +22.5% | 1023/1116 | 549/1116 | 0.575 | 0.162 |
| `train_relative_gate_0p1` | main_matrix | -11.5% | +0.0% | 558/558 | 482/558 | 0.864 | 0.321 |
| `train_relative_gate_0p1` | extra_holdout | +1.2% | +22.5% | 465/558 | 0/558 | 0.167 | 0.002 |
| `train_relative_gate_0p1` | all | -5.1% | +22.5% | 1023/1116 | 482/1116 | 0.515 | 0.162 |
| `train_relative_gate_0p2` | main_matrix | -11.4% | +0.0% | 558/558 | 465/558 | 0.833 | 0.321 |
| `train_relative_gate_0p2` | extra_holdout | +1.2% | +22.5% | 465/558 | 0/558 | 0.167 | 0.002 |
| `train_relative_gate_0p2` | all | -5.1% | +22.5% | 1023/1116 | 465/1116 | 0.500 | 0.162 |
| `train_relative_gate_0p5` | main_matrix | -10.7% | +0.0% | 558/558 | 438/558 | 0.785 | 0.321 |
| `train_relative_gate_0p5` | extra_holdout | +0.0% | +0.0% | 558/558 | 0/558 | 0.000 | 0.002 |
| `train_relative_gate_0p5` | all | -5.4% | +0.0% | 1116/1116 | 438/1116 | 0.392 | 0.162 |
| `train_relative_gate_1p0` | main_matrix | -9.7% | +0.0% | 558/558 | 389/558 | 0.697 | 0.321 |
| `train_relative_gate_1p0` | extra_holdout | +0.0% | +0.0% | 558/558 | 0/558 | 0.000 | 0.002 |
| `train_relative_gate_1p0` | all | -4.8% | +0.0% | 1116/1116 | 389/1116 | 0.349 | 0.162 |

## Train-Size Breakdown

| mode | train curves | group | mean delta | worst delta | non-harm cells | wins | target factor |
|---|---:|---|---:|---:|---:|---:|---:|
| `raw_nextgen` | 1 | all | -6.6% | +21.8% | 126/144 | 90/144 | 1.000 |
| `raw_nextgen` | 2 | all | -5.5% | +22.2% | 270/315 | 180/315 | 1.000 |
| `raw_nextgen` | 3 | all | -4.8% | +22.5% | 300/360 | 180/360 | 1.000 |
| `raw_nextgen` | 4 | all | -3.6% | +18.7% | 180/225 | 90/225 | 1.000 |
| `raw_nextgen` | 5 | all | -1.4% | +15.0% | 54/72 | 18/72 | 1.000 |
| `retention_gate_0p01` | 1 | all | -7.5% | +0.0% | 144/144 | 90/144 | 0.625 |
| `retention_gate_0p01` | 2 | all | -6.4% | +0.0% | 315/315 | 180/315 | 0.571 |
| `retention_gate_0p01` | 3 | all | -5.9% | +0.0% | 360/360 | 180/360 | 0.500 |
| `retention_gate_0p01` | 4 | all | -5.0% | +0.0% | 225/225 | 90/225 | 0.400 |
| `retention_gate_0p01` | 5 | all | -3.1% | +0.0% | 72/72 | 18/72 | 0.250 |
| `train_relative_gate_0p5` | 1 | all | -5.6% | +0.0% | 144/144 | 57/144 | 0.396 |
| `train_relative_gate_0p5` | 2 | all | -5.8% | +0.0% | 315/315 | 141/315 | 0.448 |
| `train_relative_gate_0p5` | 3 | all | -5.7% | +0.0% | 360/360 | 150/360 | 0.417 |
| `train_relative_gate_0p5` | 4 | all | -4.8% | +0.0% | 225/225 | 75/225 | 0.333 |
| `train_relative_gate_0p5` | 5 | all | -3.0% | +0.0% | 72/72 | 15/72 | 0.208 |

## Target Retention By Curve

| target | group | min retention | mean retention | max retention | peak/mean | raw worst |
|---|---|---:|---:|---:|---:|---:|
| Constant 24k | extra_holdout | 0.000000 | 0.000000 | 0.000000 | 0.00 | +0.0% |
| Constant 72k | extra_holdout | 0.000000 | 0.000000 | 0.000000 | 0.00 | +0.0% |
| Cosine 24k | extra_holdout | 0.004180 | 0.005295 | 0.005721 | 1.83 | +22.5% |
| Cosine | main_matrix | 0.014797 | 0.018888 | 0.020411 | 1.87 | -1.0% |
| WSD sharp | main_matrix | 0.127387 | 0.182541 | 0.211331 | 6.11 | -6.1% |
| WSD-con 18e-5 | main_matrix | 0.654885 | 0.716253 | 0.750869 | 21.47 | -2.8% |
| WSD-con 3e-5 | main_matrix | 0.192332 | 0.245165 | 0.272263 | 4.41 | -16.3% |
| WSD-con 9e-5 | main_matrix | 0.453438 | 0.530465 | 0.571272 | 11.37 | -1.4% |
| WSD linear | main_matrix | 0.159492 | 0.234384 | 0.264453 | 11.15 | -4.8% |

## Readout

The best non-harming target-identifiability candidate is `retention_gate_0p0075`: mean `-5.9%`, worst `+0.0%`, non-harm `1116/1116`. Retention-gated candidates are more theory-native than peak/mean because they measure identifiability after the exact nuisance residualizer used by the estimator.

A pure train-relative threshold can be made safe only by becoming more conservative: `train_relative_gate_0p5` has mean `-5.4%` and non-harm `1116/1116`. Weaker train-relative thresholds preserve more transfer but let the diffuse external cosine target through. This supports using an absolute target-identifiability floor rather than only a relative-to-calibration threshold.

The threshold `0.01` has a margin interpretation on the current artifacts: the lowest positive main-matrix target retention is `0.014797`, the highest positive extra-holdout diffuse retention is `0.005721`, and their geometric midpoint is `0.009201`. Thus `0.01` separates the retained response directions from the diffuse external cosine target with a visible log-scale gap without using held-out loss values.
