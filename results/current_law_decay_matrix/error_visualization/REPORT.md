# Cross-Family Error Visualization

The plots diagnose the largest cross-schedule gaps in the current-law matrix. Errors are plotted as `prediction - true loss`; positive error means the model predicts a loss that is too high.

| case | train -> test | scale | curve | kappa | MPL MAE | corrected MAE | delta |
|---|---|---:|---|---:|---:|---:|---:|
| Worst: cosine -> WSD-con | Cosine decay -> WSD-con step | 25M | wsdcon_3.csv | 0.4814 | 0.00449 | 0.10073 | +2144.8% |
| Worst: cosine -> sharp WSD | Cosine decay -> WSD sharp | 25M | wsd_20000_24000.csv | 0.4814 | 0.00341 | 0.01547 | +353.9% |
| Worst: sharp WSD -> WSD-con | WSD sharp -> WSD-con step | 100M | wsdcon_3.csv | 0.0754 | 0.00567 | 0.01361 | +140.1% |
| Contrast: WSD-con -> sharp WSD | WSD-con step -> WSD sharp | 400M | wsd_20000_24000.csv | 0.0347 | 0.00470 | 0.00362 | -23.0% |

## Reading

1. The largest failures are correction-amplitude failures.  The MPL baseline is often close enough, but the transferred correction term has the wrong magnitude for the target schedule family.
2. Cosine-calibrated kappa is large because cosine fitting treats a smooth long-horizon residual as positive lag.  When the same kappa multiplies a step-like WSD-con feature, it creates a large positive post-drop error.
3. Sharp-WSD calibration also over-transfers to WSD-con tails, but less severely.  This indicates that the residual shape learned from a terminal cooldown is not identical to the long constant tail after a step drop.
4. WSD-con probes transfer back to sharp WSD because their fitted kappa is much smaller and the correction aligns with the late-cooldown residual instead of dominating it.

## Files

- Combined grid: `/Users/jiaju/Documents/github/DL-final/results/current_law_decay_matrix/error_visualization/cross_family_error_cases.png`
- Individual case: `/Users/jiaju/Documents/github/DL-final/results/current_law_decay_matrix/error_visualization/cosine_to_wsdcon_worst.png`
- Individual case: `/Users/jiaju/Documents/github/DL-final/results/current_law_decay_matrix/error_visualization/cosine_to_wsd_worst.png`
- Individual case: `/Users/jiaju/Documents/github/DL-final/results/current_law_decay_matrix/error_visualization/sharp_to_wsdcon_worst.png`
- Individual case: `/Users/jiaju/Documents/github/DL-final/results/current_law_decay_matrix/error_visualization/probe_to_wsd_best.png`
