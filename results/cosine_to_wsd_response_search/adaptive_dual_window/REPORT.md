# Adaptive Dual-Window Cosine-to-WSD Audit

This audit allows the smooth and step response channels to estimate `kappa` from different suffixes of the same cosine calibration curve. It is meant as a robustness extension of the simpler single-window adaptive model.

## Best Fully Non-Harming Candidate

- Mean MAE change: `-34.6%` over `15` scale-target rows.
- Worst scale-target row: `-5.9%`.
- Wins/non-harm: `15/15` and `15/15`.
- Config: `smooth_start=3000`, `step_start=12000`, `lambda_smooth=4`, `lambda_step=20`, `mu=0.05`, `max_mode=8`, `tau=0.05`, `p=0.25`, `rho=0.4`.

## Best Worst-Case Candidate

- Mean / worst: `-33.6%` / `-6.4%`.
- Config: `smooth_start=3000`, `step_start=3000`, `lambda_smooth=4`, `lambda_step=20`, `mu=0.025`, `max_mode=12`, `tau=0.05`, `p=0.25`, `rho=0`.

## Per-Target Result

| target | mean delta | worst scale | wins |
|---|---:|---:|---:|
| WSD sharp | -52.9% | -41.3% | 3/3 |
| WSD linear | -43.8% | -32.8% | 3/3 |
| WSD-con 3e-5 | -53.2% | -45.1% | 3/3 |
| WSD-con 9e-5 | -13.9% | -6.5% | 3/3 |
| WSD-con 18e-5 | -9.1% | -5.9% | 3/3 |

## Comparison

Adaptive fit-window search: mean `-34.5%`, worst `-6.1%`, wins `15/15`.
Dual-window improvement is small; the main value is a slightly better worst cell, not a new mechanism.

## Top-Safe Holdout Check

| split | selected config | dev mean | dev worst | test mean | test worst | test wins |
|---|---|---:|---:|---:|---:|---:|
| dev_sharp_linear__test_wsdcon | `s_start=3000, step_start=12000, lambda_s=4, lambda_step=20, mu=0.05, tau=0.05, p=0.25, rho=0.4` | -48.3% | -32.8% | -25.4% | -5.9% | 9/9 |
| dev_wsdcon__test_sharp_linear | `s_start=3000, step_start=3000, lambda_s=7, lambda_step=20, mu=0.015, tau=0.05, p=0, rho=0.75` | -26.8% | -6.3% | -45.6% | -32.1% | 6/6 |
| leave_target__wsd_20000_24000.csv | `s_start=3000, step_start=3000, lambda_s=7, lambda_step=20, mu=0.015, tau=0.05, p=0, rho=0.75` | -30.5% | -6.3% | -49.6% | -40.8% | 3/3 |
| leave_target__wsdcon_18.csv | `s_start=3000, step_start=12000, lambda_s=4, lambda_step=20, mu=0.05, tau=0.05, p=0.25, rho=0.4` | -40.9% | -6.5% | -9.1% | -5.9% | 3/3 |
| leave_target__wsdcon_3.csv | `s_start=3000, step_start=16000, lambda_s=4, lambda_step=20, mu=0.05, tau=0.05, p=0.25, rho=0.4` | -30.1% | -5.2% | -48.6% | -45.9% | 3/3 |
| leave_target__wsdcon_9.csv | `s_start=8000, step_start=8000, lambda_s=7, lambda_step=20, mu=0.02, tau=0.05, p=0, rho=0.75` | -39.8% | -6.1% | -13.6% | -6.3% | 3/3 |
| leave_target__wsdld_20000_24000.csv | `s_start=3000, step_start=3000, lambda_s=7, lambda_step=20, mu=0.015, tau=0.05, p=0, rho=0.75` | -32.5% | -6.3% | -41.6% | -32.1% | 3/3 |
| leave_scale__25M | `s_start=5000, step_start=8000, lambda_s=7, lambda_step=20, mu=0.02, tau=0.05, p=0, rho=0.75` | -35.5% | -6.1% | -32.6% | -12.3% | 5/5 |
| leave_scale__100M | `s_start=3000, step_start=8000, lambda_s=4, lambda_step=20, mu=0.04, tau=0.05, p=0.25, rho=0.2` | -36.2% | -6.3% | -29.6% | -4.5% | 5/5 |
| leave_scale__400M | `s_start=8000, step_start=12000, lambda_s=7, lambda_step=20, mu=0.02, tau=0.05, p=0, rho=0.75` | -33.2% | -9.1% | -36.0% | -5.5% | 5/5 |

## Reading

- Separate suffixes are plausible because smooth and step response channels are identified from different parts of the cosine residual spectrum.
- The empirical gain is small compared with the added branch, so this is best treated as a robustness/audit variant rather than the main story.
