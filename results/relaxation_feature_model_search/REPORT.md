# Relaxation Feature Model Search

The residual-shape gallery suggested that the current `S10_current` feature lags badly on diffuse cosine schedules. This search compares faster S-time kernels, step-time kernels, and capped-S kernels that bound the maximum step-time relaxation tail at low LR.

## Best Self-Fit Features

| feature | mean delta | worst delta | mean R2 | wins |
|---|---:|---:|---:|---:|
| step_tau2048 | -38.7% | +4.6% | 0.603 | 16/18 |
| step_tau1024 | -36.8% | -1.7% | 0.606 | 18/18 |
| S10_cap1024 | -36.4% | -6.4% | 0.611 | 18/18 |
| S10_cap2048 | -35.6% | -5.6% | 0.605 | 18/18 |
| S20 | -34.1% | -3.5% | 0.587 | 18/18 |
| S20_cap1024 | -32.9% | -3.5% | 0.579 | 18/18 |
| S10_current | -32.3% | -5.6% | 0.582 | 18/18 |
| S10_cap512 | -31.6% | -6.6% | 0.544 | 18/18 |
| S20_cap512 | -30.9% | -3.5% | 0.545 | 18/18 |
| step_tau512 | -30.5% | -6.6% | 0.535 | 18/18 |

## Best Generalization Rows

| feature | train | mean off-group | worst off-group | WSD targets | wins |
|---|---|---:|---:|---:|---:|
| step_tau2048 | probe | -24.5% | -6.5% | -32.6% | 13/18 |
| step_tau1024 | probe | -22.1% | -4.5% | -30.2% | 16/18 |
| S10_cap2048 | probe3 | -18.8% | +3.2% | -23.7% | 17/18 |
| S20 | probe3 | -18.1% | -1.9% | -14.5% | 18/18 |
| S20_cap1024 | probe3 | -16.8% | -2.2% | -17.0% | 18/18 |
| S10_cap1024 | probe3 | -16.6% | +11.9% | -28.5% | 14/18 |
| step_tau1024 | probe3 | -16.6% | +48.2% | -37.4% | 14/18 |
| S10_cap1024 | probe | -16.5% | -3.0% | -22.6% | 17/18 |
| S10_current | probe3 | -15.7% | -2.7% | -17.7% | 18/18 |
| S10_cap2048 | probe | -15.5% | -3.2% | -21.1% | 18/18 |
| S10_cap2048 | probe9 | -14.6% | -2.3% | -14.1% | 18/18 |
| S10_current | probe9 | -14.4% | -2.4% | -14.2% | 18/18 |
| S20 | probe9 | -14.3% | -1.6% | -10.9% | 18/18 |
| step_tau512 | probe | -14.0% | -2.5% | -19.3% | 18/18 |

## Reading

- Current `S10_current` self-fit mean delta is `-32.3%`; probe-calibrated off-group mean is `-12.6%`.
- Capped-S kernels directly test the hypothesis that low-LR relaxation should not acquire an arbitrarily long step-time tail. If they move up the generalization table while preserving self-fit, they are the strongest candidate for replacing the current response feature.
- Step-time kernels are included as an aggressive alternative; strong self-fit but poor transfer would indicate that pure step-time relaxation throws away the measured LR-dependent rate too aggressively.
