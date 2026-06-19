# Direct MPL Backbone Shape-Projection Audit

This no-optimization audit projects LD-kernel shape parameters from independent cosine-only MPL fits to cross-scale medians.  It tests whether simple shape stabilization is enough before introducing any more residual modeling.

| variant | group | backbone params | correction vs own MPL | wins / non-harm | own MPL vs official | corrected vs official |
|---|---|---:|---:|---:|---:|---:|
| frozen_official | WSD-family | 21 | -13.77% mean / -6.29% worst | 15/15 / 15/15 | +0.00% mean | -13.77% mean |
| frozen_official | controls | 21 | +0.00% mean / +0.00% worst | 0/9 / 9/9 | +0.00% mean | +0.00% mean |
| cosine_independent | WSD-family | 21 | -11.44% mean / -6.40% worst | 15/15 / 15/15 | +55.05% mean | +37.34% mean |
| cosine_independent | controls | 21 | +0.00% mean / +0.00% worst | 0/9 / 9/9 | +79.78% mean | +79.78% mean |
| median_beta_gamma_projected | WSD-family | 17 | -11.34% mean / -6.92% worst | 15/15 / 15/15 | +54.23% mean | +36.83% mean |
| median_beta_gamma_projected | controls | 17 | +0.00% mean / +0.00% worst | 0/9 / 9/9 | +79.74% mean | +79.74% mean |
| median_c_beta_gamma_projected | WSD-family | 15 | -11.32% mean / -6.95% worst | 15/15 / 15/15 | +54.09% mean | +36.74% mean |
| median_c_beta_gamma_projected | controls | 15 | +0.00% mean / +0.00% worst | 0/9 / 9/9 | +79.74% mean | +79.74% mean |

## Reading

- This audit has zero optimization after the original cosine-only MPL fits.
- If median projection hurts, simple cross-scale LD-shape stabilization is not enough and should not be used as a main result.
- If it helps, the next step is a controlled, efficient backbone refit around the projected shape.
