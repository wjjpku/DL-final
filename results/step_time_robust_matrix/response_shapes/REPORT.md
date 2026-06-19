# Response Shape Comparison

This diagnostic compares the original cumulative-LR response with finite step-time responses. It focuses on whether the residual behaves like a local LR-drop catch-up transient or like a broad low-frequency MPL mismatch.

## Figures

- `25M` response shape: `response_shape_comparison_25M.png`
- `25M` remaining error: `response_remaining_error_25M.png`
- `100M` response shape: `response_shape_comparison_100M.png`
- `100M` remaining error: `response_remaining_error_100M.png`
- `400M` response shape: `response_shape_comparison_400M.png`
- `400M` remaining error: `response_remaining_error_400M.png`
- Cross-scale metric summary: `response_shape_metric_summary.png`

## Mean Metrics Across Scales

| schedule | S10 lag | tau1024 lag | S10 width | tau1024 width | raw low-freq R2 | S10 delta | tau1024 delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| constant 72k | n/a | n/a | n/a% | n/a% | 0.93 | +0.0% | +0.0% |
| cosine 72k | +19029 | -2688 | 71.7% | 71.4% | 0.98 | -38.2% | -71.7% |
| cosine 24k | +10880 | +4224 | n/a% | n/a% | 0.89 | +0.0% | +0.0% |
| WSD exp cooldown | +597 | -1963 | 15.3% | 14.7% | 0.88 | -50.8% | -39.3% |
| WSD linear cooldown | +256 | +256 | 15.3% | 14.7% | 0.91 | -45.7% | -42.7% |
| step to 3e-5 | -384 | -384 | 46.2% | 22.5% | 0.72 | -32.7% | -40.5% |
| step to 9e-5 | -341 | -341 | 24.4% | 22.5% | 0.82 | -12.9% | -13.8% |
| step to 18e-5 | -299 | -299 | 12.7% | 22.5% | 0.63 | -13.4% | -13.0% |

## Reading

- Cosine is the outlier: its old cumulative-LR response is both late and wide, so a same-curve kappa can absorb a global sinusoidal MPL residual rather than a local decay transient.
- WSD schedules have localized cooldowns. The finite step-time response follows those changes with much shorter memory, which matches the idea that the loss should catch up after a finite delay.
- Step probes are useful calibration curves because their LR perturbation is identifiable. They do not let a smooth low-frequency residual masquerade as a schedule response as easily as cosine does.
- Therefore, cosine should remain a diagnostic for nuisance structure, while kappa calibration should prefer sharp or endpoint-matched decay probes for WSD-style targets.
