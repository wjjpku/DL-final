# Tail-Gate Route Audit

This audit routes between the joint LR-curvature model and the tail-gated model using only the target schedule:

```text
use tail gate if curve is WSD-con and final_lr / peak_lr <= threshold
otherwise use the joint LR-curvature prediction
```

## Best Fully Non-Harming Route

- Threshold: `0.1`.
- Mean / worst: `-37.62%` / `-10.80%`.
- Wins/non-harm: `15/15` and `15/15`.

## Best Worst-Case Route

- Threshold: `0.1`.
- Mean / worst: `-37.62%` / `-10.80%`.

## Threshold Summary

| threshold | mean delta | worst | wins |
|---:|---:|---:|---:|
| 0 | -37.53% | -10.80% | 15/15 |
| 0.1 | -37.62% | -10.80% | 15/15 |
| 0.3 | -37.59% | -10.47% | 15/15 |
| 0.6 | -37.59% | -10.47% | 15/15 |

## Per-Target Result For Best Route

| target | mean delta | worst scale | wins |
|---|---:|---:|---:|
| WSD sharp | -54.3% | -40.5% | 3/3 |
| WSD linear | -46.3% | -33.1% | 3/3 |
| WSD-con 3e-5 | -57.5% | -47.3% | 3/3 |
| WSD-con 9e-5 | -17.0% | -10.8% | 3/3 |
| WSD-con 18e-5 | -13.0% | -12.1% | 3/3 |

## Top-Safe Holdout Check

| split | selected threshold | dev mean | dev worst | test mean | test worst | test wins |
|---|---:|---:|---:|---:|---:|---:|
| dev_sharp_linear__test_wsdcon | `0` | -50.3% | -33.1% | -29.0% | -10.8% | 9/9 |
| dev_wsdcon__test_sharp_linear | `0.1` | -29.2% | -10.8% | -50.3% | -33.1% | 6/6 |
| leave_target__wsd_20000_24000.csv | `0.1` | -33.5% | -10.8% | -54.3% | -40.5% | 3/3 |
| leave_target__wsdcon_18.csv | `0.1` | -43.8% | -10.8% | -13.0% | -12.1% | 3/3 |
| leave_target__wsdcon_3.csv | `0` | -32.6% | -10.8% | -57.1% | -46.7% | 3/3 |
| leave_target__wsdcon_9.csv | `0.1` | -42.8% | -12.1% | -17.0% | -10.8% | 3/3 |
| leave_target__wsdld_20000_24000.csv | `0.1` | -35.4% | -10.8% | -46.3% | -33.1% | 3/3 |
| leave_scale__25M | `0.1` | -39.8% | -10.8% | -33.3% | -12.1% | 5/5 |
| leave_scale__100M | `0.1` | -38.4% | -12.1% | -36.0% | -10.8% | 5/5 |
| leave_scale__400M | `0.1` | -34.7% | -10.8% | -43.5% | -13.6% | 5/5 |

## Reading

- The selected threshold applies the tail gate only to the lowest-tail WSD-con schedule, where the gate improves the residual without weakening the current worst row.
- This route is schedule-only, but the threshold is still selected in a development audit over available WSD-family targets.
