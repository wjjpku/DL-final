# Soft Spectral Kappa Audit

This audit keeps the final nuisance-projected EB kappa formula fixed and replaces hard low-frequency projection with a soft DCT/Sobolev nuisance residualizer.

For each curve, the nuisance drift is fit by `min_a ||y-Qa||^2 + lambda sum_j j^4 a_j^2` using DCT modes 0--12. The residualized feature and MPL residual are then passed to the same `sqrt(R)` EB estimator. This is a continuous version of the low-frequency nuisance assumption, not a polynomial fit and not a schedule-family classifier.

## Sweep

| lambda | worst offdiag | mean offdiag | cosine -> WSD | wsdcon_9 -> WSD | max cosine kappa | mean retention |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | -0.1% | -4.3% | -0.2% | -6.2% | 0.0005 | 0.107 |
| 1e-08 | -0.1% | -4.3% | -0.2% | -6.2% | 0.0005 | 0.107 |
| 3e-08 | -0.1% | -4.3% | -0.2% | -6.2% | 0.0005 | 0.107 |
| 1e-07 | -0.1% | -4.3% | -0.2% | -6.2% | 0.0005 | 0.107 |
| 3e-07 | -0.1% | -4.3% | -0.2% | -6.2% | 0.0005 | 0.107 |
| 1e-06 | -0.1% | -4.3% | -0.2% | -6.2% | 0.0005 | 0.107 |
| 3e-06 | -0.1% | -4.3% | -0.2% | -6.2% | 0.0005 | 0.107 |
| 1e-05 | -0.1% | -4.3% | -0.2% | -6.2% | 0.0005 | 0.107 |
| 3e-05 | -0.2% | -4.5% | -0.2% | -6.3% | 0.0007 | 0.110 |
| 0.0001 | -0.4% | -4.9% | -0.4% | -6.9% | 0.0014 | 0.120 |
| 0.0003 | -0.6% | -5.9% | -1.0% | -7.9% | 0.0032 | 0.141 |
| 0.001 | -1.2% | -7.6% | -2.3% | -9.3% | 0.0072 | 0.176 |
| 0.003 | -1.7% | -9.9% | -4.7% | -10.7% | 0.0135 | 0.219 |
| 0.01 | -2.0% | -12.5% | -10.7% | -12.3% | 0.0259 | 0.278 |
| 0.015 | -2.2% | -13.2% | -14.3% | -12.9% | 0.0324 | 0.301 |
| 0.02 | -2.2% | -13.4% | -17.5% | -13.3% | 0.0382 | 0.318 |
| 0.025 | -2.3% | -13.5% | -20.6% | -13.7% | 0.0435 | 0.332 |
| 0.03 | -2.4% | -13.3% | -23.5% | -14.0% | 0.0486 | 0.343 |
| 0.04 | -1.0% | -12.6% | -28.8% | -14.5% | 0.0581 | 0.362 |
| 0.05 | +7.0% | -11.8% | -33.4% | -15.0% | 0.0667 | 0.377 |
| 0.07 | +37.2% | -10.0% | -40.7% | -15.8% | 0.0815 | 0.401 |
| 0.1 | +76.8% | -8.0% | -47.4% | -16.7% | 0.0989 | 0.428 |
| 0.3 | +177.7% | -0.9% | -38.3% | -19.0% | 0.1504 | 0.503 |
| 1 | +198.3% | +2.8% | -24.0% | -20.4% | 0.1883 | 0.575 |

## Readout

- Best worst-offdiagonal setting: `lambda=0.03` with worst `-2.4%`, mean `-13.3%`, cosine-to-WSD `-23.5%`.
- Best mean-offdiagonal setting: `lambda=0.025` with worst `-2.3%`, mean `-13.5%`, cosine-to-WSD `-20.6%`.
- Best cosine-to-WSD setting: `lambda=0.1` with worst `+76.8%`, mean `-8.0%`, cosine-to-WSD `-47.4%`.
- Best conservative Pareto candidate (`worst <= -2%`, `mean <= -12%`, `cosine -> WSD <= -10%`): `lambda=0.025` with worst `-2.3%`, mean `-13.5%`, cosine-to-WSD `-20.6%`, and max cosine kappa `0.0435`.

A useful main-method replacement would dominate the legacy smooth basis (`worst -2.7%`, `mean -12.1%`, `cosine -> WSD -4.3%`) without relying on a hard cap or family label. The soft spectral sweep does not yet dominate worst-case behavior, but it exposes a theoretically cleaner Pareto frontier: lambda around `0.02--0.03` improves mean and cosine-to-WSD substantially while keeping every off-diagonal cell non-failing. Above `0.04`, the method starts to over-transfer amplitude and produces positive failures.
