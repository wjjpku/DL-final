# Mid-Tail Recovery Audit

This audit adds a Gaussian LR-level gate to the step channel, intended to recover over-persistent lag on moderate final-LR WSD-con schedules.

## Best Fully Non-Harming Candidate

- Mean / worst: `-37.59%` / `-10.48%`.
- Wins/non-harm: `15/15` and `15/15`.
- Gate: `center=0.2`, `width=0.1`, `time_power=1`, `tau=0.1`, `sign=signed`, `shrink=1`.
- Mean step coefficients: primary `0.04001`, curvature `0.01872`, gate `0.00394`.

## Best Worst-Case Candidate

- Mean / worst: `-37.53%` / `-10.80%`.

## Comparison

- Joint-channel LR-curvature: mean `-37.53%`, worst `-10.80%`.
- Mid-tail recovery: mean `-37.59%`, worst `-10.48%`.

## Per-Target Result

| target | mean delta | worst scale | wins |
|---|---:|---:|---:|
| WSD sharp | -54.3% | -40.5% | 3/3 |
| WSD linear | -46.3% | -33.1% | 3/3 |
| WSD-con 3e-5 | -57.5% | -47.0% | 3/3 |
| WSD-con 9e-5 | -16.8% | -10.5% | 3/3 |
| WSD-con 18e-5 | -13.0% | -12.0% | 3/3 |

## Reading

- If selected coefficients are zero, the cosine residual does not support this recovery shape.
- If it improves mean but worsens worst-case, it may still be useful as a routed low- or mid-tail feature rather than a global step feature.
