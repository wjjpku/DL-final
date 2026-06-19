# Decoupled-Channel Error Comparison

Residual plots compare MPL, the previous shared fit-window model, and the decoupled-channel model. Both corrected models fit kappa only from cosine residuals.

- Shared fit-window: mean `-34.5%`, worst `-6.1%`.
- Decoupled-channel: mean `-36.2%`, worst `-6.3%`.

| target | shared mean | decoupled mean | decoupled worst |
|---|---:|---:|---:|
| WSD sharp | -51.9% | -54.3% | -40.5% |
| WSD linear | -43.9% | -46.3% | -33.1% |
| WSD-con 3e-5 | -54.0% | -55.8% | -47.3% |
| WSD-con 9e-5 | -13.6% | -14.7% | -9.0% |
| WSD-con 18e-5 | -9.2% | -9.9% | -6.3% |
