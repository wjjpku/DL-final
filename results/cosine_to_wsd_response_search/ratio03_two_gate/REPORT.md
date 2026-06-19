# Ratio 0.3 Two-Gate Audit

This audit keeps the all-ratio model fixed except for WSD-con 9e-5.  The ratio-0.3 branch keeps the selected Gaussian gate and adds a second Gaussian LR-level gate. Both gate coefficients are fitted from the cosine residual only.

## Best Fully Non-Harming Candidate

- Mean / worst: `-37.90%` / `-11.81%`.
- Wins/non-harm: `15/15` and `15/15`.
- Extra gate: center `0.15`, width `0.25`, time_power `2`, tau `0.001`, shrink_gates `1`.

## Best Worst-Case Candidate

- Mean / worst: `-37.88%` / `-11.82%`.

## Comparison

- All-ratio one-gate: mean `-37.88%`, worst `-11.80%`.
- Ratio 0.3 two-gate: mean `-37.90%`, worst `-11.81%`.

## Per-Target Result

| target | mean delta | worst scale | wins |
|---|---:|---:|---:|
| WSD sharp | -54.3% | -40.5% | 3/3 |
| WSD linear | -46.3% | -33.1% | 3/3 |
| WSD-con 3e-5 | -57.9% | -48.9% | 3/3 |
| WSD-con 9e-5 | -17.7% | -11.8% | 3/3 |
| WSD-con 18e-5 | -13.3% | -12.3% | 3/3 |

## WSD-con 9e-5 Rows

| scale | delta | corr_mae | base_gate_coef | extra_gate_coef |
|---|---:|---:|---:|---:|
| 25M | -22.3% | 0.0022746 | 0.03032 | -0.00001 |
| 100M | -11.8% | 0.00435177 | 0.03864 | 0.00148 |
| 400M | -18.9% | 0.0054233 | 0.04740 | 0.02698 |

## Reading

- A second gate tests whether the remaining moderate-tail error has two LR-local components rather than one.
- This is a higher-complexity development candidate and should not replace the one-gate all-ratio model unless the extra improvement is worth the added parameter.
