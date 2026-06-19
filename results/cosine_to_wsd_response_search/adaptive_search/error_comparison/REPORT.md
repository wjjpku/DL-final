# Adaptive Error Comparison

Residual plots compare MPL, old nextgen, the global response candidate, and the best schedule-adaptive candidate. All correction amplitudes are fitted from cosine residuals only.

## Aggregate

- Old: mean `-17.2%`, worst `-2.2%`.
- Global response: mean `-22.0%`, worst `-6.5%`.
- Adaptive response: mean `-31.3%`, worst `-6.1%`.

| target | old mean | global mean | adaptive mean | adaptive worst |
|---|---:|---:|---:|---:|
| WSD sharp | -20.5% | -17.2% | -43.9% | -25.2% |
| WSD linear | -17.3% | -15.6% | -35.8% | -19.0% |
| WSD-con 3e-5 | -30.2% | -53.9% | -53.5% | -46.6% |
| WSD-con 9e-5 | -9.8% | -14.2% | -14.1% | -7.0% |
| WSD-con 18e-5 | -8.3% | -9.1% | -9.0% | -6.1% |
