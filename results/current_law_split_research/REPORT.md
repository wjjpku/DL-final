# Current-Law Split Research
Fixed law: `residual = kappa * DropRelaxS_lambda`. No new residual feature.
## all split train/test
- mean test delta: `-6.8%`
- median test delta: `-14.1%`
- wins: `182/225`
- selected lambda median/IQR: `20` / `14-20`
## probe-only calibration -> final wsd/wsdld
- mean test delta: `-14.8%`
- median test delta: `-13.3%`
- wins: `42/42`
- selected lambda median/IQR: `20` / `14-20`
## Best probe-only final protocols
| rank | scale | train probes | lambda | MAE | vs MPL | wins | kappa |
|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 25M | `wsdcon_18` | 2 | 0.00211 | -35.6% | 2/2 | 0.01775 |
| 2 | 25M | `wsdcon_9+wsdcon_18` | 14 | 0.00284 | -13.1% | 2/2 | 0.02116 |
| 3 | 25M | `wsdcon_9` | 14 | 0.00289 | -11.7% | 2/2 | 0.01895 |
| 4 | 25M | `wsdcon_3+wsdcon_18` | 20 | 0.00289 | -11.7% | 2/2 | 0.02582 |
| 5 | 25M | `wsdcon_3` | 20 | 0.00289 | -11.5% | 2/2 | 0.02544 |
| 6 | 25M | `wsdcon_3+wsdcon_9+wsdcon_18` | 20 | 0.00290 | -11.3% | 2/2 | 0.02496 |
| 7 | 25M | `wsdcon_3+wsdcon_9` | 20 | 0.00291 | -11.2% | 2/2 | 0.02462 |
| 8 | 100M | `wsdcon_3` | 20 | 0.00301 | -14.1% | 2/2 | 0.03344 |
| 9 | 100M | `wsdcon_3+wsdcon_18` | 20 | 0.00301 | -13.9% | 2/2 | 0.03293 |
| 10 | 100M | `wsdcon_3+wsdcon_9` | 20 | 0.00303 | -13.3% | 2/2 | 0.03156 |
| 11 | 100M | `wsdcon_3+wsdcon_9+wsdcon_18` | 20 | 0.00304 | -13.2% | 2/2 | 0.03121 |
| 12 | 100M | `wsdcon_18` | 14 | 0.00308 | -11.9% | 2/2 | 0.02065 |

## Worst all-split failures
| rank | scale | train | test | lambda | vs MPL | wins |
|---:|---:|---|---|---:|---:|---:|
| 1 | 100M | `wsd_20000_24000+wsdld_20000_24000` | `wsdcon_3+wsdcon_9+wsdcon_18` | 5 | +96.3% | 0/3 |
| 2 | 100M | `wsd_20000_24000+wsdld_20000_24000+wsdcon_18` | `wsdcon_3+wsdcon_9` | 7 | +77.8% | 0/2 |
| 3 | 400M | `wsd_20000_24000+wsdld_20000_24000` | `wsdcon_3+wsdcon_9+wsdcon_18` | 2 | +73.0% | 0/3 |
| 4 | 100M | `wsdld_20000_24000` | `wsd_20000_24000+wsdcon_3+wsdcon_9+wsdcon_18` | 5 | +67.7% | 1/4 |
| 5 | 100M | `wsd_20000_24000` | `wsdld_20000_24000+wsdcon_3+wsdcon_9+wsdcon_18` | 5 | +60.9% | 1/4 |
| 6 | 400M | `wsd_20000_24000+wsdld_20000_24000+wsdcon_18` | `wsdcon_3+wsdcon_9` | 7 | +49.4% | 0/2 |
| 7 | 25M | `wsd_20000_24000+wsdld_20000_24000+wsdcon_18` | `wsdcon_3+wsdcon_9` | 3 | +48.7% | 0/2 |
| 8 | 400M | `wsdld_20000_24000` | `wsd_20000_24000+wsdcon_3+wsdcon_9+wsdcon_18` | 2 | +47.1% | 1/4 |
| 9 | 400M | `wsd_20000_24000` | `wsdld_20000_24000+wsdcon_3+wsdcon_9+wsdcon_18` | 3 | +43.7% | 1/4 |
| 10 | 100M | `wsd_20000_24000+wsdcon_18` | `wsdld_20000_24000+wsdcon_3+wsdcon_9` | 7 | +39.2% | 1/3 |
| 11 | 25M | `wsd_20000_24000+wsdld_20000_24000` | `wsdcon_3+wsdcon_9+wsdcon_18` | 5 | +35.0% | 1/3 |
| 12 | 25M | `wsdld_20000_24000` | `wsd_20000_24000+wsdcon_3+wsdcon_9+wsdcon_18` | 5 | +23.7% | 2/4 |

## Interpretation
- The law is not a universal residual patch: single-probe calibration can overfit specific two-stage tails and fail on other held-out curves.
- Probe-only calibration still improves the final cosine->WSD target on average, but the gain is smaller than the cross-scale amplitude rule.
- Lambda selected by pure training fit is often larger than 10; the measured `lambda=10` is therefore a conservative shape prior, not a flexible split optimum.
