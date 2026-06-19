# Decoupled-Channel Cosine-to-WSD Audit

This audit keeps the cosine-only fitting rule but lets the smooth and step response channels use different calibration hyperparameters. Target routing is still schedule-only.

## Formula Change

```text
channel(target) = smooth or step from LR drop concentration
theta_smooth = suffix / residualizer / shrink settings for smooth decay
theta_step   = suffix / residualizer / shrink settings for concentrated drops
kappa_c(theta_c) is fitted only on cosine_72000 residuals
L_hat_target = L_MPL,target + kappa_channel(theta_channel) * phi_channel,target
```

The model still has one fitted amplitude per scale and channel. The difference is that the two channels no longer share the same nuisance filter and suffix window.

## Best Fully Non-Harming Candidate

- Mean MAE change: `-36.18%` over `15` scale-target rows.
- Worst scale-target row: `-6.29%`.
- Wins/non-harm: `15/15` and `15/15`.
- Config: `smooth(start=12000, lambda=4, mu=0.05, modes=8, p=0.25, rho=0.2); step(start=3000, lambda=20, mu=0.015, modes=16, p=0, rho=0.75)`.

## Best Worst-Case Candidate

- Mean / worst: `-36.18%` / `-6.32%`.
- Config: `smooth(start=12000, lambda=4, mu=0.05, modes=8, p=0.25, rho=0.2); step(start=3000, lambda=20, mu=0.01, modes=12, p=0, rho=0.35)`.

## Per-Target Result

| target | mean delta | worst scale | wins |
|---|---:|---:|---:|
| WSD sharp | -54.3% | -40.5% | 3/3 |
| WSD linear | -46.3% | -33.1% | 3/3 |
| WSD-con 3e-5 | -55.8% | -47.3% | 3/3 |
| WSD-con 9e-5 | -14.7% | -9.0% | 3/3 |
| WSD-con 18e-5 | -9.9% | -6.3% | 3/3 |

## Comparison

Shared-rho fit-window: mean `-34.53%`, worst `-6.08%`.
Best-mean channel-shrink single-config: mean `-35.26%`, worst `-5.53%`.
Decoupled-channel candidate: mean `-36.18%`, worst `-6.29%`.

The single-config channel-shrink candidate can improve the average but slightly weakens the worst WSD-con 9e-5 cell. Decoupling the channels recovers both: smooth targets use the longer-suffix smooth calibration, while WSD-con targets use the safer step calibration.

## Top-Safe Holdout Check

| split | selected pair | dev mean | dev worst | test mean | test worst | test wins |
|---|---|---:|---:|---:|---:|---:|
| dev_sharp_linear__test_wsdcon | `smooth=2112, step=235` | -50.3% | -33.1% | -26.8% | -6.3% | 9/9 |
| dev_wsdcon__test_sharp_linear | `smooth=1401, step=335` | -26.8% | -6.3% | -49.8% | -33.5% | 6/6 |
| leave_target__wsd_20000_24000.csv | `smooth=1401, step=335` | -31.8% | -6.3% | -52.8% | -39.1% | 3/3 |
| leave_target__wsdcon_18.csv | `smooth=2112, step=235` | -42.8% | -9.0% | -9.9% | -6.3% | 3/3 |
| leave_target__wsdcon_3.csv | `smooth=2112, step=7422` | -31.5% | -6.3% | -55.1% | -45.8% | 3/3 |
| leave_target__wsdcon_9.csv | `smooth=2112, step=235` | -41.6% | -6.3% | -14.7% | -9.0% | 3/3 |
| leave_target__wsdld_20000_24000.csv | `smooth=2112, step=335` | -33.6% | -6.3% | -46.3% | -33.1% | 3/3 |
| leave_scale__25M | `smooth=2112, step=7322` | -37.7% | -6.3% | -33.2% | -12.1% | 5/5 |
| leave_scale__100M | `smooth=2112, step=235` | -36.6% | -6.3% | -35.3% | -9.0% | 5/5 |
| leave_scale__400M | `smooth=2112, step=335` | -34.3% | -9.0% | -40.0% | -6.3% | 5/5 |

## Reading

- This is the strongest current cosine-to-WSD development result: lower mean and lower worst-cell MAE than the shared-rho and single-config channel-shrink variants.
- The added complexity is interpretable but real: there are now separate calibration settings for smooth and step response channels.
- This should be presented as a development candidate until the channel pair is frozen and tested on new schedules or a pre-registered split.
