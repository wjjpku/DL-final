# LR-Curvature Error Comparison

Residual plots compare MPL, the decoupled-channel model, and the LR-curvature model. Both corrected models fit coefficients only from cosine residuals.

- Decoupled-channel: mean `-36.2%`, worst `-6.3%`.
- LR-curvature: mean `-37.5%`, worst `-9.4%`.

| target | decoupled mean | curvature mean | curvature worst |
|---|---:|---:|---:|
| WSD sharp | -54.3% | -54.3% | -40.5% |
| WSD linear | -46.3% | -46.3% | -33.1% |
| WSD-con 3e-5 | -55.8% | -57.7% | -47.7% |
| WSD-con 9e-5 | -14.7% | -16.5% | -9.4% |
| WSD-con 18e-5 | -9.9% | -12.6% | -11.9% |
