# Next-Gen External Holdout Sanity Audit

This audit evaluates the next-generation single-curve `rho=0.5` kappa on repo curves not included in the main six-schedule matrix. `cosine_24000` is also one of the MPL baseline fitting curves, so this is a conservative sanity check rather than a clean independent benchmark.

## Summary

| mode | test curve | mean delta | worst delta | wins | target factor | peak/mean |
|---|---|---:|---:|---:|---:|---:|
| `raw_nextgen` | Cosine 24k | +7.2% | +21.8% | 0/18 | 1.00 | 1.8 |
| `raw_nextgen` | Constant 24k | +0.0% | +0.0% | 0/18 | 1.00 | 0.0 |
| `raw_nextgen` | Constant 72k | +0.0% | +0.0% | 0/18 | 1.00 | 0.0 |
| `target_localization_gate` | Cosine 24k | +0.0% | +0.0% | 0/18 | 0.00 | 1.8 |
| `target_localization_gate` | Constant 24k | +0.0% | +0.0% | 0/18 | 0.00 | 0.0 |
| `target_localization_gate` | Constant 72k | +0.0% | +0.0% | 0/18 | 0.00 | 0.0 |

## Readout

Raw next-gen transfer is not safe on `cosine_24000`: mean `+7.2%`, worst `+21.8%`. The constant schedules are unaffected because their response feature is zero.

A schedule-only target-localization gate with threshold `2.0` abstains on diffuse targets such as `cosine_24000` and reduces that failure to mean `+0.0%`, worst `+0.0%`. This supports a theoretical limitation: if the target response feature is too diffuse, it is not identifiable apart from low-frequency MPL drift, so transfer should abstain unless target residual evidence is available.
