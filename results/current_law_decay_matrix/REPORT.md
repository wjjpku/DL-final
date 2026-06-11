# Current-Law Decay-Family Matrix

Fixed law: `MPL + kappa * DropRelaxS_lambda`, with `lambda=10`. Only `kappa` is fitted per scale from the calibration schedule family. Cosine is included as a smooth-decay diagnostic, but note that the MPL backbone was originally fit on cosine curves.

| train family | test family | MAE | vs MPL | wins |
|---|---|---:|---:|---:|
| Cosine decay | Cosine decay | 0.00411 | -39.4% | 3/3 |
| Cosine decay | WSD sharp | 0.01346 | +240.6% | 0/3 |
| Cosine decay | WSD-con step | 0.04347 | +894.8% | 0/9 |
| WSD sharp | Cosine decay | 0.00609 | -10.1% | 3/3 |
| WSD sharp | WSD sharp | 0.00191 | -51.6% | 3/3 |
| WSD sharp | WSD-con step | 0.00748 | +71.2% | 1/9 |
| WSD-con step | Cosine decay | 0.00655 | -3.4% | 3/3 |
| WSD-con step | WSD sharp | 0.00320 | -19.1% | 3/3 |
| WSD-con step | WSD-con step | 0.00346 | -20.9% | 9/9 |

## Reading

- Cosine calibration learns almost no positive lag amplitude because the MPL backbone was already fit on smooth cosine schedules.
- WSD-con step probes transfer to sharp WSD targets with a smaller but stable gain; this is the clean no-target-WSD calibration regime.
- Sharp WSD calibration does not transfer to WSD-con tails. This confirms that the law should not be presented as a universal residual smoother for every LR schedule family.
