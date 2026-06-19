# Cosine Self-Fit Error Comparison

This compares the MPL residual on `cosine_72000.csv` with `kappa * DropRelaxS(lambda=10)` fitted on the same cosine curve. It is a self-fit diagnostic, not a transfer result.

| scale | kappa | MPL MAE | +ours MAE | delta | R2_origin | pearson |
|---:|---:|---:|---:|---:|---:|---:|
| 25M | 0.4814 | 0.00745 | 0.00435 | -41.6% | 0.629 | -0.038 |
| 100M | 0.5359 | 0.00727 | 0.00371 | -48.9% | 0.742 | 0.425 |
| 400M | 0.3424 | 0.00561 | 0.00427 | -24.0% | 0.427 | 0.552 |

## Reading

- On the cosine curve itself, the correction reduces MPL MAE by about 24--49%, because the fitted DropRelaxS feature tracks a large smooth positive residual component.
- The estimated residual is broad and low-frequency, not a localized fast-cooldown transient. This explains why the same raw cosine kappa does not transfer to sharp WSD/WSD-con targets.
