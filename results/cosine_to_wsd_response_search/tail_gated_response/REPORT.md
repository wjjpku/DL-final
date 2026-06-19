# Tail-Gated Step Response Audit

This audit adds one schedule-only tail gate to the current joint LR-curvature step channel:

```text
step correction = a * phi_step + b * psi_curv + c * phi_step * g(eta / eta_peak)
```

The new coefficient is fitted only from `cosine_72000.csv` residuals.

## Best Fully Non-Harming Candidate

- Mean / worst: `-37.59%` / `-10.47%`.
- Wins/non-harm: `15/15` and `15/15`.
- Gate: `mode=one_minus_lr`, `tau=0.3`, `sign=signed`, `shrink=0`.
- Mean step coefficients: primary `0.04064`, curvature `0.01871`, gate `0.00089`.

## Best Worst-Case Candidate

- Mean / worst: `-37.53%` / `-10.80%`.

## Comparison

- Joint-channel LR-curvature: mean `-37.53%`, worst `-10.80%`.
- Tail-gated response: mean `-37.59%`, worst `-10.47%`.

## Per-Target Result

| target | mean delta | worst scale | wins |
|---|---:|---:|---:|
| WSD sharp | -54.3% | -40.5% | 3/3 |
| WSD linear | -46.3% | -33.1% | 3/3 |
| WSD-con 3e-5 | -57.5% | -47.3% | 3/3 |
| WSD-con 9e-5 | -16.8% | -10.5% | 3/3 |
| WSD-con 18e-5 | -13.0% | -12.1% | 3/3 |

## Top-Safe Holdout Check

| split | selected config | dev mean | dev worst | test mean | test worst | test wins |
|---|---|---:|---:|---:|---:|---:|
| dev_sharp_linear__test_wsdcon | `mode=lr, tau=0.001, sign=nonpos, shrink=1` | -50.3% | -33.1% | -29.0% | -10.8% | 9/9 |
| dev_wsdcon__test_sharp_linear | `mode=one_minus_lr, tau=0.3, sign=signed, shrink=0` | -29.1% | -10.5% | -50.3% | -33.1% | 6/6 |
| leave_target__wsd_20000_24000.csv | `mode=one_minus_lr, tau=0.3, sign=signed, shrink=0` | -33.4% | -10.5% | -54.3% | -40.5% | 3/3 |
| leave_target__wsdcon_18.csv | `mode=one_minus_lr, tau=0.3, sign=signed, shrink=0` | -43.7% | -10.5% | -13.0% | -12.1% | 3/3 |
| leave_target__wsdcon_3.csv | `mode=lr, tau=0.001, sign=nonpos, shrink=1` | -32.6% | -10.8% | -57.1% | -46.7% | 3/3 |
| leave_target__wsdcon_9.csv | `mode=one_minus_lr, tau=0.3, sign=signed, shrink=0` | -42.8% | -12.1% | -16.8% | -10.5% | 3/3 |
| leave_target__wsdld_20000_24000.csv | `mode=one_minus_lr, tau=0.3, sign=signed, shrink=0` | -35.4% | -10.5% | -46.3% | -33.1% | 3/3 |
| leave_scale__25M | `mode=one_minus_lr, tau=0.3, sign=signed, shrink=0` | -39.7% | -10.5% | -33.3% | -12.1% | 5/5 |
| leave_scale__100M | `mode=one_minus_lr, tau=0.3, sign=signed, shrink=0` | -38.4% | -12.1% | -36.0% | -10.5% | 5/5 |
| leave_scale__400M | `mode=one_minus_lr, tau=0.3, sign=signed, shrink=0` | -34.6% | -10.5% | -43.5% | -13.7% | 5/5 |

## Reading

- A negative high-LR gate would support the catch-up interpretation: higher post-drop LR erases the lag faster.
- If selected gates are zero or do not beat the joint-channel model, the remaining error is not explained by a simple LR-level-gated response.
