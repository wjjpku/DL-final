# Soft Spectral Lambda-Selection Audit

This audit tests whether the soft DCT/Sobolev nuisance prior can choose its smoothing strength from the calibration curve itself. No test curve labels or schedule-family labels are used in the selection rules.

## Comparison

| rule | worst offdiag | mean offdiag | cosine -> WSD | wsdcon_9 -> WSD | max cosine kappa | mean retention | median lambda |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fixed_lam0p025` | -2.3% | -13.5% | -20.6% | -13.7% | 0.0435 | 0.332 | 0.025 |
| `residual_gcv` | -0.1% | -4.4% | -0.2% | -6.3% | 0.0005 | 0.108 | 1e-05 |
| `residual_bic` | -0.3% | -4.6% | -0.4% | -6.6% | 0.0010 | 0.116 | 3e-05 |
| `retention_r0p33` | +198.3% | +1.5% | -24.0% | -9.3% | 0.1883 | 0.290 | 0.06 |
| `gcv_retention_band` | -0.1% | -8.6% | -0.2% | -8.2% | 0.0005 | 0.235 | 0.013 |

## Selection Behavior

- `fixed_lam0p025` selected lambda min/median/max = `0.025` / `0.025` / `0.025`.
- `gcv_retention_band` selected lambda min/median/max = `3e-06` / `0.013` / `0.05`.
- `residual_bic` selected lambda min/median/max = `1e-05` / `3e-05` / `0.0003`.
- `residual_gcv` selected lambda min/median/max = `3e-06` / `1e-05` / `3e-05`.
- `retention_r0p33` selected lambda min/median/max = `0` / `0.06` / `1`.

## Readout

Pure residual GCV/BIC are calibration-only but may select smoothing strengths that optimize MPL-residual denoising rather than transfer-amplitude identifiability. The retention-target rule is also calibration-only, but uses the response feature geometry rather than the observed residual. The hybrid rule asks for both: a plausible identifiable-energy band and the best residual smoother inside that band.

A rule can replace the current main method only if it is competitive with `final_no_cap` (`worst -2.7%`, `mean -12.1%`, `cosine -> WSD -4.3%`) without relying on test labels.
