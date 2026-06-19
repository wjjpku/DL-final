# Cosine Kappa Contamination Diagnostic

This diagnostic tests whether the raw full-cosine kappa is dominated by low-frequency MPL residual drift rather than transferable fast-decay lag.

| scale | raw kappa | DCT retention | DCT projected kappa | DCT effective kappa |
|---:|---:|---:|---:|---:|
| 25M | 0.4814 | 0.0104 | 0.0300 | 0.0031 |
| 100M | 0.5359 | 0.0104 | 0.1056 | 0.0108 |
| 400M | 0.3424 | 0.0105 | 0.1343 | 0.0138 |

| target | raw global cosine mean delta | DCT-effective mean delta |
|---|---:|---:|
| wsd_20000_24000.csv | +256.3% | -7.0% |
| wsdcon_3.csv | +1530.8% | -16.1% |

## Reading

1. Raw full-cosine kappa is large because the full cosine residual is smooth and low-frequency.
2. After removing four low-frequency DCT nuisance modes, only about one percent of the full-cosine feature energy remains identifiable. The effective amplitude collapses by one to two orders of magnitude.
3. Using the raw full-cosine kappa causes the large WSD and WSD-con failures. Using the residualized effective amplitude removes the over-correction without fitting target losses.
4. This supports the interpretation that full-cosine calibration is contaminated by MPL backbone mismatch; it should be used only with nuisance control or replaced by target-like probes.
