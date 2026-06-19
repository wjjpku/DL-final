# Adaptive Fit-Window Error Comparison

Residual plots compare the previous adaptive model and the suffix-fitted adaptive model. Both fit kappa only from cosine residuals.

- Adaptive: mean `-31.3%`, worst `-6.1%`.
- Fit-window adaptive: mean `-34.5%`, worst `-6.1%`.

| target | adaptive mean | fit-window mean | fit-window worst |
|---|---:|---:|---:|
| WSD sharp | -43.9% | -51.9% | -40.9% |
| WSD linear | -35.8% | -43.9% | -33.0% |
| WSD-con 3e-5 | -53.5% | -54.0% | -44.5% |
| WSD-con 9e-5 | -14.1% | -13.6% | -6.3% |
| WSD-con 18e-5 | -9.0% | -9.2% | -6.1% |
