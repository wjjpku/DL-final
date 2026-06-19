# Strict vs Rounded Endpoint Audit

This audit compares the fully derived observation-half-life endpoint with the rounded fast endpoint.  It keeps the same one-coefficient cosine-only projected estimator and adds constant / short-cosine controls.

## Summary

| variant | group | mean | worst | wins | non-harm |
|---|---|---:|---:|---:|---:|
| strict_exact | core_wsd | -31.97% | -1.09% | 15/15 | 15/15 |
| strict_exact | extra_control | +14.02% | +56.43% | 0/9 | 6/9 |
| strict_exact_sqrtlocalized | core_wsd | -31.56% | -1.09% | 15/15 | 15/15 |
| strict_exact_sqrtlocalized | extra_control | +0.00% | +0.00% | 0/9 | 9/9 |
| strict_exact_localized | core_wsd | -30.24% | -1.09% | 15/15 | 15/15 |
| strict_exact_localized | extra_control | +0.00% | +0.00% | 0/9 | 9/9 |
| rounded_fast20 | core_wsd | -34.56% | -5.30% | 15/15 | 15/15 |
| rounded_fast20 | extra_control | +14.02% | +56.43% | 0/9 | 6/9 |
| rounded_fast20_sqrtlocalized | core_wsd | -34.15% | -5.30% | 15/15 | 15/15 |
| rounded_fast20_sqrtlocalized | extra_control | +0.00% | +0.00% | 0/9 | 9/9 |
| rounded_fast20_localized | core_wsd | -32.83% | -5.30% | 15/15 | 15/15 |
| rounded_fast20_localized | extra_control | +0.00% | +0.00% | 0/9 | 9/9 |
| legacy_7_20 | core_wsd | -34.45% | -5.30% | 15/15 | 15/15 |
| legacy_7_20 | extra_control | +14.59% | +58.26% | 0/9 | 6/9 |

## Extra Controls By Curve

| variant | control | mean | worst | non-harm |
|---|---|---:|---:|---:|
| strict_exact | Cosine 24k | +42.06% | +56.43% | 0/3 |
| strict_exact | Constant 24k | +0.00% | +0.00% | 3/3 |
| strict_exact | Constant 72k | +0.00% | +0.00% | 3/3 |
| strict_exact_sqrtlocalized | Cosine 24k | +0.00% | +0.00% | 3/3 |
| strict_exact_sqrtlocalized | Constant 24k | +0.00% | +0.00% | 3/3 |
| strict_exact_sqrtlocalized | Constant 72k | +0.00% | +0.00% | 3/3 |
| strict_exact_localized | Cosine 24k | +0.00% | +0.00% | 3/3 |
| strict_exact_localized | Constant 24k | +0.00% | +0.00% | 3/3 |
| strict_exact_localized | Constant 72k | +0.00% | +0.00% | 3/3 |
| rounded_fast20 | Cosine 24k | +42.06% | +56.43% | 0/3 |
| rounded_fast20 | Constant 24k | +0.00% | +0.00% | 3/3 |
| rounded_fast20 | Constant 72k | +0.00% | +0.00% | 3/3 |
| rounded_fast20_sqrtlocalized | Cosine 24k | +0.00% | +0.00% | 3/3 |
| rounded_fast20_sqrtlocalized | Constant 24k | +0.00% | +0.00% | 3/3 |
| rounded_fast20_sqrtlocalized | Constant 72k | +0.00% | +0.00% | 3/3 |
| rounded_fast20_localized | Cosine 24k | +0.00% | +0.00% | 3/3 |
| rounded_fast20_localized | Constant 24k | +0.00% | +0.00% | 3/3 |
| rounded_fast20_localized | Constant 72k | +0.00% | +0.00% | 3/3 |
| legacy_7_20 | Cosine 24k | +43.77% | +58.26% | 0/3 |
| legacy_7_20 | Constant 24k | +0.00% | +0.00% | 3/3 |
| legacy_7_20 | Constant 72k | +0.00% | +0.00% | 3/3 |

## Reading

- The strict exact endpoint is the cleanest endpoint formula: `lambda_fast = lambda_obs`, with no rounded constant.  It keeps all WSD-family rows improving, but is more conservative than the rounded fast endpoint.
- The rounded fast endpoint gives stronger WSD-family gains, especially on step-to-constant targets, but it should be described as a rounded observable-resolution prior rather than as a fitted parameter.
- The localized variants multiply the correction by a continuous schedule-locality factor, not a fitted gate: full-run diffuse cosine decay receives no local transient correction, while finite WSD cooldown and single-step WSD-con remain active.  The sqrt-localized variant is the current deployable amplitude rule.
- Constant controls are unaffected because the positive LR-drop feature is exactly zero after warmup.
- Without localization, the short-cosine control is the main limitation.  With localization, this failure is removed at the cost of a smaller but still all-win WSD-family gain.
