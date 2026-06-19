# Adaptive Fit-Window Cosine-to-WSD Search

This audit keeps the schedule-adaptive model but estimates `kappa` from a suffix of the cosine calibration curve. WSD-family losses are used only to rank development candidates; the fitted residual evidence remains `cosine_72000.csv` only.

## Best Fully Non-Harming Candidate

- Mean MAE change: `-34.5%` over `15` scale-target rows.
- Worst scale-target row: `-6.1%`.
- Wins/non-harm: `15/15` and `15/15`.
- Config: `fit_start=8000`, `lambda_smooth=7`, `lambda_step=20`, `mu=0.02`, `max_mode=12`, `tau=0.05`, `p=0`, `rho=0.75`.

## Best Worst-Case Candidate

- Mean / worst: `-33.6%` / `-6.4%`.
- Config: `fit_start=3000`, `lambda_smooth=4`, `lambda_step=20`, `mu=0.025`, `max_mode=12`, `tau=0.05`, `p=0.25`, `rho=0`.

## Per-Target Result

| target | mean delta | worst scale | wins |
|---|---:|---:|---:|
| WSD sharp | -51.9% | -40.9% | 3/3 |
| WSD linear | -43.9% | -33.0% | 3/3 |
| WSD-con 3e-5 | -54.0% | -44.5% | 3/3 |
| WSD-con 9e-5 | -13.6% | -6.3% | 3/3 |
| WSD-con 18e-5 | -9.2% | -6.1% | 3/3 |

## Comparison

Old nextgen: mean `-17.2%`, worst `-2.2%`, wins `15/15`.
Global response search: mean `-22.0%`, worst `-6.5%`, wins `15/15`.
Adaptive search: mean `-31.3%`, worst `-6.1%`, wins `15/15`.
Adaptive fit-window search: mean shown above.

## Top-Safe Holdout Check

| split | selected config | dev mean | dev worst | test mean | test worst | test wins |
|---|---|---:|---:|---:|---:|---:|
| dev_sharp_linear__test_wsdcon | `start=8000, lambda_s=7, lambda_step=20, mu=0.02, tau=0.05, p=0, rho=0.75` | -47.9% | -33.0% | -25.6% | -6.1% | 9/9 |
| dev_wsdcon__test_sharp_linear | `start=3000, lambda_s=7, lambda_step=20, mu=0.015, tau=0.05, p=0, rho=0.75` | -26.8% | -6.3% | -45.6% | -32.1% | 6/6 |
| leave_target__wsd_20000_24000.csv | `start=3000, lambda_s=7, lambda_step=20, mu=0.015, tau=0.05, p=0, rho=0.75` | -30.5% | -6.3% | -49.6% | -40.8% | 3/3 |
| leave_target__wsdcon_18.csv | `start=8000, lambda_s=7, lambda_step=20, mu=0.02, tau=0.05, p=0, rho=0.75` | -40.9% | -6.3% | -9.2% | -6.1% | 3/3 |
| leave_target__wsdcon_3.csv | `start=8000, lambda_s=7, lambda_step=20, mu=0.02, tau=0.05, p=0, rho=0.75` | -29.7% | -6.1% | -54.0% | -44.5% | 3/3 |
| leave_target__wsdcon_9.csv | `start=8000, lambda_s=7, lambda_step=20, mu=0.02, tau=0.05, p=0, rho=0.75` | -39.8% | -6.1% | -13.6% | -6.3% | 3/3 |
| leave_target__wsdld_20000_24000.csv | `start=3000, lambda_s=7, lambda_step=20, mu=0.015, tau=0.05, p=0, rho=0.75` | -32.5% | -6.3% | -41.6% | -32.1% | 3/3 |
| leave_scale__25M | `start=8000, lambda_s=7, lambda_step=20, mu=0.02, tau=0.05, p=0, rho=0.75` | -35.2% | -6.1% | -33.2% | -12.3% | 5/5 |
| leave_scale__100M | `start=3000, lambda_s=7, lambda_step=20, mu=0.015, tau=0.05, p=0, rho=0.75` | -36.0% | -6.3% | -31.0% | -9.0% | 5/5 |
| leave_scale__400M | `start=8000, lambda_s=7, lambda_step=20, mu=0.02, tau=0.05, p=0, rho=0.75` | -32.9% | -6.3% | -37.8% | -6.1% | 5/5 |

## Interpretation

- The improvement comes from reducing early-cosine contamination in the kappa projection, not from fitting WSD residuals.
- The selected `fit_start=8000` is interpretable: it starts after warmup and after the earliest smooth residual transient visible in cosine, while still leaving most of the 72k cosine curve for calibration.
- This remains a development result because the search uses WSD-family ranking to choose the suffix and hyperparameters. The next proof step is adding new schedules or a stricter pre-registered split.
