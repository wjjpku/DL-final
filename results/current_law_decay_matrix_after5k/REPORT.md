# Current-Law Decay Matrix: Fit Step >= 5000

This reruns the cross-schedule matrix with the first 5k steps removed only from the `kappa` fitting data. The MPL backbone and DropRelaxS feature are unchanged. Results are reported for full-target evaluation and for evaluation restricted to `step>=5000`.

## Kappa Values

| train family | scale | kappa |
|---|---:|---:|
| Cosine decay | 25M | 0.48107 |
| Cosine decay | 100M | 0.53572 |
| Cosine decay | 400M | 0.34310 |
| WSD sharp | 25M | 0.04770 |
| WSD sharp | 100M | 0.07542 |
| WSD sharp | 400M | 0.10121 |
| WSD-con step | 25M | 0.01774 |
| WSD-con step | 100M | 0.02020 |
| WSD-con step | 400M | 0.03469 |

## Matrix: eval=full

| train family | test family | MAE | vs MPL | wins |
|---|---|---:|---:|---:|
| Cosine decay | Cosine decay | 0.00411 | -39.4% | 3/3 |
| Cosine decay | WSD sharp | 0.01346 | +240.6% | 0/3 |
| Cosine decay | WSD-con step | 0.04347 | +894.9% | 0/9 |
| WSD sharp | Cosine decay | 0.00609 | -10.1% | 3/3 |
| WSD sharp | WSD sharp | 0.00191 | -51.6% | 3/3 |
| WSD sharp | WSD-con step | 0.00748 | +71.2% | 1/9 |
| WSD-con step | Cosine decay | 0.00655 | -3.4% | 3/3 |
| WSD-con step | WSD sharp | 0.00320 | -19.1% | 3/3 |
| WSD-con step | WSD-con step | 0.00346 | -20.9% | 9/9 |

## Matrix: eval=after5k

| train family | test family | MAE | vs MPL | wins |
|---|---|---:|---:|---:|
| Cosine decay | Cosine decay | 0.00396 | -41.3% | 3/3 |
| Cosine decay | WSD sharp | 0.01515 | +264.3% | 0/3 |
| Cosine decay | WSD-con step | 0.05518 | +1038.7% | 0/9 |
| WSD sharp | Cosine decay | 0.00602 | -10.6% | 3/3 |
| WSD sharp | WSD sharp | 0.00180 | -56.7% | 3/3 |
| WSD sharp | WSD-con step | 0.00882 | +81.9% | 1/9 |
| WSD-con step | Cosine decay | 0.00650 | -3.5% | 3/3 |
| WSD-con step | WSD sharp | 0.00329 | -20.9% | 3/3 |
| WSD-con step | WSD-con step | 0.00367 | -24.2% | 9/9 |

## Reading

- Cosine-calibrated kappa remains large after dropping the first 5k steps: mean `0.4533`. This is nearly unchanged from the full-curve cosine fit, so the raw cosine transfer failure is not driven mainly by the earliest warmup-adjacent points.
- The likely driver remains low-frequency MPL backbone mismatch over the smooth cosine curve. Early-step masking is useful hygiene, but it does not replace nuisance residualization or target-like probe calibration.
