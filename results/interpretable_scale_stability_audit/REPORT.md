# Scale-Stability Audit

This audit tests whether the cosine-fitted response amplitude is stable across model scales.  For every method, the source loss is still only `cosine_72000.csv`; the difference is that the source scale can be 25M, 100M, or 400M while the target scale is independently varied.

## Main Reading

The mechanism-native candidate is `mpl_ld_tangent`: before estimating the LR-drop response amplitude, it projects the cosine residual and response feature away from the local tangent space of MPL's LR-dependent parameters \((B,C,\beta,\gamma)\).  This is more interpretable than a generic DCT low-frequency filter because the nuisance directions are exactly MPL parameter-error directions.

## Summary

| method | split | group | mean | worst | wins/non-harm |
|---|---|---|---:|---:|---:|
| mpl_ld_tangent | same_scale | core_wsd | -27.25% | -3.00% | 15/15 wins, 15/15 non-harm |
| mpl_ld_tangent | same_scale | extra_control | +0.00% | +0.00% | 0/9 wins, 9/9 non-harm |
| mpl_ld_tangent | cross_scale | core_wsd | -23.07% | -2.07% | 30/30 wins, 30/30 non-harm |
| mpl_ld_tangent | cross_scale | extra_control | +0.00% | +0.00% | 0/18 wins, 18/18 non-harm |
| mpl_ld_tangent | holdout_test_25 | core_wsd | -19.80% | -2.07% | 10/10 wins, 10/10 non-harm |
| mpl_ld_tangent | holdout_test_25 | extra_control | +0.00% | +0.00% | 0/6 wins, 6/6 non-harm |
| mpl_ld_tangent | holdout_test_100 | core_wsd | -26.67% | -6.20% | 10/10 wins, 10/10 non-harm |
| mpl_ld_tangent | holdout_test_100 | extra_control | +0.00% | +0.00% | 0/6 wins, 6/6 non-harm |
| mpl_ld_tangent | holdout_test_400 | core_wsd | -22.73% | -2.15% | 10/10 wins, 10/10 non-harm |
| mpl_ld_tangent | holdout_test_400 | extra_control | +0.00% | +0.00% | 0/6 wins, 6/6 non-harm |
| dct_performance | same_scale | core_wsd | -32.83% | -5.30% | 15/15 wins, 15/15 non-harm |
| dct_performance | same_scale | extra_control | +0.00% | +0.00% | 0/9 wins, 9/9 non-harm |
| dct_performance | cross_scale | core_wsd | -18.98% | +26.68% | 26/30 wins, 26/30 non-harm |
| dct_performance | cross_scale | extra_control | +0.00% | +0.00% | 0/18 wins, 18/18 non-harm |
| dct_performance | holdout_test_25 | core_wsd | -10.19% | +26.68% | 8/10 wins, 8/10 non-harm |
| dct_performance | holdout_test_25 | extra_control | +0.00% | +0.00% | 0/6 wins, 6/6 non-harm |
| dct_performance | holdout_test_100 | core_wsd | -23.20% | +5.77% | 8/10 wins, 8/10 non-harm |
| dct_performance | holdout_test_100 | extra_control | +0.00% | +0.00% | 0/6 wins, 6/6 non-harm |
| dct_performance | holdout_test_400 | core_wsd | -23.55% | -3.74% | 10/10 wins, 10/10 non-harm |
| dct_performance | holdout_test_400 | extra_control | +0.00% | +0.00% | 0/6 wins, 6/6 non-harm |
| tau_free_dct | same_scale | core_wsd | -20.77% | -5.86% | 15/15 wins, 15/15 non-harm |
| tau_free_dct | same_scale | extra_control | +0.00% | +0.00% | 0/9 wins, 9/9 non-harm |
| tau_free_dct | cross_scale | core_wsd | -13.27% | +9.04% | 27/30 wins, 27/30 non-harm |
| tau_free_dct | cross_scale | extra_control | +0.00% | +0.00% | 0/18 wins, 18/18 non-harm |
| tau_free_dct | holdout_test_25 | core_wsd | -13.07% | +9.04% | 8/10 wins, 8/10 non-harm |
| tau_free_dct | holdout_test_25 | extra_control | +0.00% | +0.00% | 0/6 wins, 6/6 non-harm |
| tau_free_dct | holdout_test_100 | core_wsd | -13.55% | +1.77% | 9/10 wins, 9/10 non-harm |
| tau_free_dct | holdout_test_100 | extra_control | +0.00% | +0.00% | 0/6 wins, 6/6 non-harm |
| tau_free_dct | holdout_test_400 | core_wsd | -13.18% | -3.46% | 10/10 wins, 10/10 non-harm |
| tau_free_dct | holdout_test_400 | extra_control | +0.00% | +0.00% | 0/6 wins, 6/6 non-harm |

## Interpretation

- Same-scale `mpl_ld_tangent`: -27.25% mean, -3.00% worst, 15/15 wins.
- Cross-scale `mpl_ld_tangent`: -23.07% mean, -2.07% worst, 30/30 wins.  This means the most mechanism-native version does not require choosing a separate scale-specific story to stay beneficial.
- Cross-scale `dct_performance`: -18.98% mean, +26.68% worst, 26/30 wins.  It has strong mean gains but a positive worst case, so it should remain a performance reference rather than the main explanation.
- Cross-scale `tau_free_dct`: -13.27% mean, +9.04% worst, 27/30 wins.  It is safer than the DCT performance reference but still inherits the interpretability cost of a generic low-frequency projection.

## Coefficient Range

| method | min coef | median coef | max coef | min retention | median retention |
|---|---:|---:|---:|---:|---:|
| mpl_ld_tangent | 0.0129384 | 0.0506566 | 0.0872278 | 0.00118207 | 0.00139133 |
| dct_performance | 0.0304352 | 0.0585902 | 0.0829027 | 0.00974113 | 0.00978924 |
| tau_free_dct | 0.0272992 | 0.0386121 | 0.0542326 | 0.0199661 | 0.0199661 |

## Decision

For a rigorous presentation, foreground `mpl_ld_tangent` as the interpretable candidate.  It gives slightly weaker same-scale WSD performance than DCT, but it is cleaner: the nuisance projection has a direct MPL-error meaning, and the cross-scale audit stays non-harmful on all WSD rows.  DCT-based variants should be described as diagnostic or performance extensions, not as the core theory.
