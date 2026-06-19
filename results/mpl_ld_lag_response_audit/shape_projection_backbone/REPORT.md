# Fast MPL Backbone Shape-Projection Audit

This audit keeps the error model fixed and tests whether strict cosine-only MPL is weak because its LD-kernel shape parameters are poorly identified by smooth cosine curves.

The downstream correction remains:

\[
\hat L_s(t)=L_{\mathrm{MPL},s}(t)+a_sB_s[D_{\downarrow,\tau_s,s}(t)-D_{\downarrow,s}(t)].
\]

No residual-fitted parameter is introduced.

## Summary

| variant | group | backbone params | correction vs own MPL | wins / non-harm | own MPL vs official | corrected vs official |
|---|---|---:|---:|---:|---:|---:|
| frozen_official | WSD-family | 21 | -13.77% mean / -6.29% worst | 15/15 / 15/15 | +0.00% mean | -13.77% mean |
| frozen_official | controls | 21 | +0.00% mean / +0.00% worst | 0/9 / 9/9 | +0.00% mean | +0.00% mean |
| cosine_independent | WSD-family | 21 | -11.44% mean / -6.40% worst | 15/15 / 15/15 | +55.05% mean | +37.34% mean |
| cosine_independent | controls | 21 | +0.00% mean / +0.00% worst | 0/9 / 9/9 | +79.78% mean | +79.78% mean |
| median_beta_gamma_refit | WSD-family | 17 | -11.33% mean / -6.92% worst | 15/15 / 15/15 | +54.37% mean | +36.96% mean |
| median_beta_gamma_refit | controls | 17 | +0.00% mean / +0.00% worst | 0/9 / 9/9 | +79.90% mean | +79.90% mean |
| median_c_beta_gamma_refit | WSD-family | 15 | -11.31% mean / -6.95% worst | 15/15 / 15/15 | +54.24% mean | +36.89% mean |
| median_c_beta_gamma_refit | controls | 15 | +0.00% mean / +0.00% worst | 0/9 / 9/9 | +79.89% mean | +79.89% mean |

## Cosine Train Objective

| variant | scale | objective | shared C | shared beta | shared gamma |
|---|---:|---:|---:|---:|---:|
| cosine_independent | 100 | 0.0001049732 |  |  |  |
| cosine_independent | 25 | 0.00010151417 |  |  |  |
| cosine_independent | 400 | 0.00015197608 |  |  |  |
| frozen_official | 100 | 0.00029445552 |  |  |  |
| frozen_official | 25 | 0.00028411853 |  |  |  |
| frozen_official | 400 | 0.00030771001 |  |  |  |
| median_beta_gamma_refit | 100 | 0.00010497282 |  | 0.9600541420381892 | 0.9918905329304987 |
| median_beta_gamma_refit | 25 | 0.00010165542 |  | 0.9600541420381892 | 0.9918905329304987 |
| median_beta_gamma_refit | 400 | 0.00015203277 |  | 0.9600541420381892 | 0.9918905329304987 |
| median_c_beta_gamma_refit | 100 | 0.00010497282 | 2.1489760011145105 | 0.9600541420381892 | 0.9918905329304987 |
| median_c_beta_gamma_refit | 25 | 0.00010167743 | 2.1489760011145105 | 0.9600541420381892 | 0.9918905329304987 |
| median_c_beta_gamma_refit | 400 | 0.00015203944 | 2.1489760011145105 | 0.9600541420381892 | 0.9918905329304987 |

## Reading

- This is a backbone identifiability probe, not a new residual estimator.
- A good outcome would reduce corrected strict-cosine WSD error versus `cosine_independent` without losing 15/15 non-harm against its own MPL baseline.
- If the projection worsens WSD, it means simple cross-scale shape sharing is too crude; the finite-response error formula should remain unchanged while the backbone problem is handled separately.
