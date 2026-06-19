# Interpretable Protocol Sensitivity

This audit keeps the current sqrt-localized observation-half-life response formula fixed and varies only protocol-level choices.  WSD-family target losses are used only for evaluation.

## Fit-Start Sensitivity

| fit start | mean | worst | wins |
|---:|---:|---:|---:|
| 3000 | -28.17% | -2.15% | 15/15 |
| 5000 | -32.18% | -4.09% | 15/15 |
| 6500 | -33.68% | -4.95% | 15/15 |
| 8000 | -34.15% | -5.30% | 15/15 |
| 10000 | -34.08% | -6.06% | 15/15 |
| 12000 | -33.37% | -5.75% | 15/15 |

## Nuisance Bandwidth Sensitivity

| DCT modes | mu | mean | worst | wins |
|---:|---:|---:|---:|---:|
| 4 | 0.005 | -26.31% | -4.73% | 15/15 |
| 4 | 0.01 | -33.98% | -5.08% | 15/15 |
| 4 | 0.02 | -11.32% | +16.25% | 10/15 |
| 6 | 0.005 | -26.00% | -4.54% | 15/15 |
| 6 | 0.01 | -34.14% | -5.26% | 15/15 |
| 6 | 0.02 | -10.95% | +16.39% | 10/15 |
| 8 | 0.005 | -25.94% | -4.50% | 15/15 |
| 8 | 0.01 | -34.15% | -5.30% | 15/15 |
| 8 | 0.02 | -10.92% | +16.37% | 10/15 |
| 10 | 0.005 | -25.91% | -4.50% | 15/15 |
| 10 | 0.01 | -34.16% | -5.33% | 15/15 |
| 10 | 0.02 | -10.93% | +16.34% | 10/15 |
| 12 | 0.005 | -25.90% | -4.50% | 15/15 |
| 12 | 0.01 | -34.16% | -5.34% | 15/15 |
| 12 | 0.02 | -10.93% | +16.33% | 10/15 |

## Ridge Sensitivity

| ridge tau | mean | worst | wins |
|---:|---:|---:|---:|
| 0 | +133.64% | +408.51% | 1/15 |
| 0.01 | +100.53% | +314.40% | 1/15 |
| 0.02 | +44.92% | +162.26% | 5/15 |
| 0.035 | -10.99% | +21.97% | 9/15 |
| 0.04 | -22.43% | +6.55% | 12/15 |
| 0.045 | -30.66% | -0.11% | 15/15 |
| 0.05 | -34.15% | -5.30% | 15/15 |
| 0.055 | -32.86% | -5.62% | 15/15 |
| 0.06 | -29.83% | -4.97% | 15/15 |
| 0.07 | -23.95% | -4.01% | 15/15 |
| 0.08 | -19.58% | -3.36% | 15/15 |
| 0.1 | -13.63% | -2.55% | 15/15 |
| 0.2 | -3.85% | -1.02% | 15/15 |

## Identifiable Feature Norms

- Source response features have residualized L2 norm from `0.0216` to `0.0423` after the DCT nuisance projection.
- The current ridge `tau=0.05` is therefore a round conservative threshold slightly above the largest identifiable source-feature norm, preventing raw cosine drift from dominating the one-coefficient fit.

| target | lambda | full norm | residualized norm | identifiable fraction |
|---|---:|---:|---:|---:|
| WSD sharp | 7.2285 | 0.4274 | 0.0422 | 0.0098 |
| WSD linear | 7.2235 | 0.4277 | 0.0422 | 0.0097 |
| WSD-con 3e-5 | 20.0000 | 0.1530 | 0.0216 | 0.0200 |
| WSD-con 9e-5 | 20.0000 | 0.1530 | 0.0216 | 0.0200 |
| WSD-con 18e-5 | 20.0000 | 0.1530 | 0.0216 | 0.0200 |

## Reading

- The response formula is not tied to a single fit-start value: `5000` and `8000` both give all-win transfer, with `8000` stronger.  This supports treating early steps as a transient-removal protocol rather than a fitted model term.
- Nuisance projection is necessary but not arbitrary.  Too little residualization leaves cosine drift in the amplitude; too much or too strong regularization can become conservative or harmful.
- Ridge `tau` has a useful all-win plateau once it is at least comparable to the residualized source-feature norm.  Values below that threshold fail because the raw cosine projection is allowed to over-amplify weakly identifiable features.
- Remaining work: replace these protocol choices with pre-registered defaults before changing slides or paper claims.
