# Step-Response Refinement Audit

This audit keeps the smooth channel fixed and searches only step-channel response shape parameters. The goal is to reduce the WSD-con tail rows left by the joint LR-curvature model.

## Best Fully Non-Harming Candidate

- Mean / worst: `-37.53%` / `-10.80%`.
- Wins/non-harm: `15/15` and `15/15`.
- Step: `start=3000`, `lambda=20`, `mu=0.01`, `modes=8`, `rho=0.35`.
- Curvature: `lambda2=10`, `mode=signed_d2_lr`, `tau2=0.003`.
- Mean step coefficients: primary `0.04074`, curvature `0.01870`.

## Best Worst-Case Candidate

- Mean / worst: `-36.53%` / `-11.26%`.

## Comparison

- Joint-channel LR-curvature: mean `-37.53%`, worst `-10.80%`.
- Step-response refinement: mean `-37.53%`, worst `-10.80%`.

## Per-Target Result

| target | mean delta | worst scale | wins |
|---|---:|---:|---:|
| WSD sharp | -54.3% | -40.5% | 3/3 |
| WSD linear | -46.3% | -33.1% | 3/3 |
| WSD-con 3e-5 | -57.1% | -46.7% | 3/3 |
| WSD-con 9e-5 | -17.0% | -10.8% | 3/3 |
| WSD-con 18e-5 | -13.0% | -12.1% | 3/3 |

## Top-Safe Holdout Check

| split | selected config | dev mean | dev worst | test mean | test worst | test wins |
|---|---|---:|---:|---:|---:|---:|
| dev_sharp_linear__test_wsdcon | `step_lambda=20, mu=0.01, modes=8, rho=0.2, lambda2=4, mode=signed_d2_lr, tau2=0.01` | -50.3% | -33.1% | -27.2% | -7.7% | 9/9 |
| dev_wsdcon__test_sharp_linear | `step_lambda=20, mu=0.01, modes=8, rho=0.35, lambda2=10, mode=signed_d2_lr, tau2=0.003` | -29.0% | -10.8% | -50.3% | -33.1% | 6/6 |
| leave_target__wsd_20000_24000.csv | `step_lambda=20, mu=0.01, modes=8, rho=0.35, lambda2=10, mode=signed_d2_lr, tau2=0.003` | -33.3% | -10.8% | -54.3% | -40.5% | 3/3 |
| leave_target__wsdcon_18.csv | `step_lambda=20, mu=0.01, modes=8, rho=0.35, lambda2=10, mode=signed_d2_lr, tau2=0.003` | -43.7% | -10.8% | -13.0% | -12.1% | 3/3 |
| leave_target__wsdcon_3.csv | `step_lambda=20, mu=0.01, modes=12, rho=0.35, lambda2=10, mode=signed_d2_lr, tau2=0.003` | -32.6% | -10.8% | -57.1% | -46.6% | 3/3 |
| leave_target__wsdcon_9.csv | `step_lambda=20, mu=0.01, modes=8, rho=0.35, lambda2=10, mode=signed_d2_lr, tau2=0.003` | -42.7% | -12.1% | -17.0% | -10.8% | 3/3 |
| leave_target__wsdld_20000_24000.csv | `step_lambda=20, mu=0.01, modes=8, rho=0.35, lambda2=10, mode=signed_d2_lr, tau2=0.003` | -35.3% | -10.8% | -46.3% | -33.1% | 3/3 |
| leave_scale__25M | `step_lambda=20, mu=0.01, modes=8, rho=0.35, lambda2=10, mode=signed_d2_lr, tau2=0.003` | -39.7% | -10.8% | -33.2% | -12.1% | 5/5 |
| leave_scale__100M | `step_lambda=20, mu=0.01, modes=8, rho=0.35, lambda2=10, mode=signed_d2_lr, tau2=0.003` | -38.3% | -12.1% | -35.9% | -10.8% | 5/5 |
| leave_scale__400M | `step_lambda=20, mu=0.015, modes=12, rho=0.75, lambda2=7, mode=signed_d2_lr, tau2=0.003` | -34.8% | -10.3% | -40.3% | -6.9% | 5/5 |

## Reading

- If this search does not beat the joint-channel model, the remaining WSD-con error is probably not recoverable by a single fixed step response rate.
- This is still a development search over WSD-family evaluation, not a frozen final protocol.
