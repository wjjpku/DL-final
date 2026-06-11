# Next-Gen Lambda Stability Audit

This audit checks the selected soft DCT/Sobolev nuisance strength used by the next-generation `rho=0.5` kappa. The method restricts train-only inner-CV selection to the identifiable band `lambda in [0.01, 0.03]`.

## Summary

| train curves | rows | min | median | max | mean | in band | lower edge | upper edge | counts |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 18 | 0.025 | 0.025 | 0.025 | 0.025 | 18/18 | 0 | 0 | `0.025:18` |
| 2 | 45 | 0.010 | 0.010 | 0.030 | 0.018 | 45/45 | 24 | 15 | `0.01:24 0.015:3 0.025:3 0.03:15` |
| 3 | 60 | 0.010 | 0.030 | 0.030 | 0.023 | 60/60 | 12 | 33 | `0.01:12 0.015:9 0.02:3 0.025:3 0.03:33` |
| 4 | 45 | 0.015 | 0.030 | 0.030 | 0.029 | 45/45 | 0 | 39 | `0.015:3 0.025:3 0.03:39` |
| 5 | 18 | 0.030 | 0.030 | 0.030 | 0.030 | 18/18 | 0 | 18 | `0.03:18` |
| all | 186 | 0.010 | 0.030 | 0.030 | 0.024 | 186/186 | 36 | 105 | `0.01:36 0.015:15 0.02:3 0.025:27 0.03:105` |

## Readout

All next-generation `rho=0.5` kappa rows remain inside the identifiable band: `186/186`. The selected values span `0.010` to `0.030`, with median `0.030`. Single-curve calibration uses the fixed fallback `0.025`; larger train sets often choose the upper band edge `0.030`, which indicates that the useful region is the high-drift-control side of the identifiable band rather than an unconstrained smoothing optimum.
