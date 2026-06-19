# Localization Sensitivity Audit

This audit keeps the observation-half-life response and cosine-only coefficient fixed, and varies only the shape of the schedule-locality factor.  `power=0` is the unlocalized upper variant; `power=1` is the current deployable linear rule.

## Summary

| power | interpretation | group | mean | worst | wins | non-harm |
|---:|---|---|---:|---:|---:|---:|
| 0.0 | no localization | core_wsd | -34.56% | -5.30% | 15/15 | 15/15 |
| 0.0 | no localization | extra_control | +14.02% | +56.43% | 0/9 | 6/9 |
| 0.5 | milder localization | core_wsd | -34.15% | -5.30% | 15/15 | 15/15 |
| 0.5 | milder localization | extra_control | +0.00% | +0.00% | 0/9 | 9/9 |
| 1.0 | linear default | core_wsd | -32.83% | -5.30% | 15/15 | 15/15 |
| 1.0 | linear default | extra_control | +0.00% | +0.00% | 0/9 | 9/9 |
| 2.0 | strong localization | core_wsd | -29.84% | -5.30% | 15/15 | 15/15 |
| 2.0 | strong localization | extra_control | +0.00% | +0.00% | 0/9 | 9/9 |
| 3.0 | very strong localization | core_wsd | -27.14% | -5.31% | 15/15 | 15/15 |
| 3.0 | very strong localization | extra_control | +0.00% | +0.00% | 0/9 | 9/9 |

## Locality Values

| curve | base locality |
|---|---:|
| WSD sharp | 0.8168 |
| WSD linear | 0.8168 |
| WSD-con 3e-5 | 0.9999 |
| WSD-con 9e-5 | 0.9999 |
| WSD-con 18e-5 | 0.9999 |
| Cosine 24k | 0.0000 |
| Constant 24k | 0.0000 |
| Constant 72k | 0.0000 |

## Reading

- Any positive localization power removes the short-cosine failure while keeping constant schedules unchanged.
- Stronger powers are safer but increasingly conservative on WSD sharp/linear, because their finite cooldown occupies about 18% of the post-warmup horizon.
- The linear default is the least additional structure that removes the control failure while preserving all WSD-family wins and most of the WSD-only gain.
