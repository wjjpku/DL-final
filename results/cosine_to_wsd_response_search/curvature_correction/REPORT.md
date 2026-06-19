# LR-Curvature Cosine-to-WSD Audit

This audit extends the decoupled-channel model with one step-channel feature: a causal relaxation of the second finite difference of the LR schedule. Coefficients are fitted only from `cosine_72000.csv` residuals.

## Formula Change

```text
psi_lambda(t) = causal_relax_lambda(eta_{t-2} - 2 eta_{t-1} + eta_t) / eta_peak
step correction = a * phi_step(t) + b * psi_lambda(t)
smooth correction = kappa_smooth * phi_smooth(t)
L_hat_target = L_MPL,target + correction_channel(target)
```

The curvature term is schedule-only. On WSD-con schedules it acts near the abrupt LR transition, which directly targets the tail overshoot left by the first-order response model.

## Best Fully Non-Harming Candidate

- Mean MAE change: `-37.47%` over `15` scale-target rows.
- Worst scale-target row: `-9.43%`.
- Wins/non-harm: `15/15` and `15/15`.
- Config: `curvature_lambda=4`, `mode=signed_d2_lr`, `tau2=0.01`, `shrink_curvature=0`, `signed_curvature_coef=0`.
- Mean step coefficients: primary `0.04145`, curvature `0.01797`.

## Best Worst-Case Candidate

- Mean / worst: `-37.40%` / `-9.73%`.
- Config: `curvature_lambda=10`, `mode=signed_d2_lr`, `tau2=0.003`, `shrink_curvature=0`, `signed_curvature_coef=0`.

## Per-Target Result

| target | mean delta | worst scale | wins |
|---|---:|---:|---:|
| WSD sharp | -54.3% | -40.5% | 3/3 |
| WSD linear | -46.3% | -33.1% | 3/3 |
| WSD-con 3e-5 | -57.7% | -47.7% | 3/3 |
| WSD-con 9e-5 | -16.5% | -9.4% | 3/3 |
| WSD-con 18e-5 | -12.6% | -11.9% | 3/3 |

## Comparison

Decoupled-channel: mean `-36.18%`, worst `-6.29%`.
Curvature correction: mean `-37.47%`, worst `-9.43%`.

## Top-Safe Holdout Check

| split | selected config | dev mean | dev worst | test mean | test worst | test wins |
|---|---|---:|---:|---:|---:|---:|
| dev_sharp_linear__test_wsdcon | `lambda2=4, mode=signed_d2_lr, tau2=0.01, shrink=1, signed=0` | -50.3% | -33.1% | -28.0% | -9.3% | 9/9 |
| dev_wsdcon__test_sharp_linear | `lambda2=4, mode=signed_d2_lr, tau2=0.01, shrink=0, signed=0` | -28.9% | -9.4% | -50.3% | -33.1% | 6/6 |
| leave_target__wsd_20000_24000.csv | `lambda2=4, mode=signed_d2_lr, tau2=0.01, shrink=0, signed=0` | -33.3% | -9.4% | -54.3% | -40.5% | 3/3 |
| leave_target__wsdcon_18.csv | `lambda2=4, mode=signed_d2_lr, tau2=0.01, shrink=0, signed=0` | -43.7% | -9.4% | -12.6% | -11.9% | 3/3 |
| leave_target__wsdcon_3.csv | `lambda2=4, mode=signed_d2_lr, tau2=0.01, shrink=0, signed=0` | -32.4% | -9.4% | -57.7% | -47.7% | 3/3 |
| leave_target__wsdcon_9.csv | `lambda2=4, mode=signed_d2_lr, tau2=0.01, shrink=0, signed=0` | -42.7% | -11.9% | -16.5% | -9.4% | 3/3 |
| leave_target__wsdld_20000_24000.csv | `lambda2=4, mode=signed_d2_lr, tau2=0.01, shrink=0, signed=0` | -35.3% | -9.4% | -46.3% | -33.1% | 3/3 |
| leave_scale__25M | `lambda2=4, mode=signed_d2_lr, tau2=0.01, shrink=0, signed=0` | -39.5% | -9.4% | -33.5% | -12.4% | 5/5 |
| leave_scale__100M | `lambda2=4, mode=signed_d2_lr, tau2=0.01, shrink=0, signed=0` | -38.4% | -12.4% | -35.6% | -9.4% | 5/5 |
| leave_scale__400M | `lambda2=7, mode=signed_d2_lr, tau2=0.003, shrink=1, signed=0` | -34.8% | -10.3% | -40.3% | -6.9% | 5/5 |

## Reading

- The gain is concentrated on WSD-con targets, especially the high-tail-LR settings where the first-order lag leaves a long tail residual.
- This is more interpretable than a sinusoidal residual basis: the added variable is the LR schedule curvature, not a free time-series basis.
- It is still a development result because `tau2` and the curvature kernel were selected by WSD-family ranking. A frozen-protocol test on new schedules is the next proof step.
