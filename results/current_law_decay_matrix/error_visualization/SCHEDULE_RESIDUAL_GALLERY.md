# Schedule Residual Gallery

This diagnostic compares MPL residuals with same-curve DropRelaxS self-fits across schedule types. Because `kappa` is fitted on the same curve, the plots are shape diagnostics rather than transfer evidence.

## Mean Metrics Across Scales

| schedule | mean delta | mean kappa | mean peak lag | mean feature width | mean drop N_eff |
|---|---:|---:|---:|---:|---:|
| constant 72k | +0.0% | 0.0000 | n/a | n/a% | 0.0 |
| cosine 72k | -38.2% | 0.4532 | +19029 | 71.7% | 56610.2 |
| cosine 24k | +0.0% | 0.0000 | +10880 | 71.8% | 17702.8 |
| WSD exp cooldown | -50.8% | 0.0748 | +597 | 15.3% | 2842.3 |
| WSD linear cooldown | -45.7% | 0.0771 | +256 | 15.3% | 3999.0 |
| step to 3e-5 | -32.7% | 0.0251 | -384 | 46.2% | 1.0 |
| step to 9e-5 | -12.9% | 0.0200 | -341 | 24.4% | 1.0 |
| step to 18e-5 | -13.4% | 0.0248 | -299 | 12.7% | 1.0 |

## Reading

- Constant schedules are a useful control: they can have MPL residual structure even though the positive-drop feature is zero, so not every residual is a non-adiabatic LR-drop lag.
- Cosine schedules produce a very diffuse DropRelaxS feature. The feature peak is late relative to the broad MPL residual, matching the visual impression that the correction is lagging and trying to fit a low-frequency wave.
- WSD and WSD-con schedules have more localized LR changes; their residuals are better suited for estimating a transient response amplitude.
- This supports treating diffuse cosine correction as non-transferable unless a nuisance projection or target-localization gate removes the low-frequency component.
