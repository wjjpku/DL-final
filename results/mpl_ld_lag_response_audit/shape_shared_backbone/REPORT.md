# MPL Backbone Shape-Sharing Audit

This audit targets the current bottleneck: strict cosine-only MPL is weak on WSD before any error correction.  The tested repair is not a new residual formula; it reduces MPL backbone freedom by sharing LD-kernel shape parameters across scales.

Downstream correction is unchanged:

\[
\hat L_s(t)=L_{\mathrm{MPL},s}(t)+a_sB_s[D_{\downarrow,\tau_s,s}(t)-D_{\downarrow,s}(t)].
\]

## Summary

| variant | group | backbone params | correction vs own MPL | wins / non-harm | own MPL vs official | corrected vs official |
|---|---|---:|---:|---:|---:|---:|
| frozen_official | WSD-family | 21 | -13.77% mean / -6.29% worst | 15/15 / 15/15 | +0.00% mean | -13.77% mean |
| frozen_official | controls | 21 | +0.00% mean / +0.00% worst | 0/9 / 9/9 | +0.00% mean | +0.00% mean |
| cosine_independent | WSD-family | 21 | -11.44% mean / -6.40% worst | 15/15 / 15/15 | +55.05% mean | +37.34% mean |
| cosine_independent | controls | 21 | +0.00% mean / +0.00% worst | 0/9 / 9/9 | +79.78% mean | +79.78% mean |
| shared_beta_gamma | WSD-family | 17 | -11.33% mean / -6.92% worst | 15/15 / 15/15 | +54.31% mean | +36.91% mean |
| shared_beta_gamma | controls | 17 | +0.00% mean / +0.00% worst | 0/9 / 9/9 | +79.88% mean | +79.88% mean |
| shared_c_beta_gamma | WSD-family | 15 | -11.31% mean / -6.95% worst | 15/15 / 15/15 | +54.17% mean | +36.82% mean |
| shared_c_beta_gamma | controls | 15 | +0.00% mean / +0.00% worst | 0/9 / 9/9 | +79.87% mean | +79.87% mean |

## Train Objective

| variant | cosine train objective |
|---|---:|
| cosine_independent | 0.00035846346 |
| frozen_official | 0.00088628406 |
| shared_beta_gamma | 0.00035866233 |
| shared_c_beta_gamma | 0.00035869112 |

## WSD Details After Correction

| variant | scale | target | correction delta | corrected vs official |
|---|---:|---|---:|---:|
| frozen_official | 25 | WSD sharp | -17.48% | -17.48% |
| frozen_official | 25 | WSD linear | -19.38% | -19.38% |
| frozen_official | 25 | WSD-con 3e-5 | -12.04% | -12.04% |
| frozen_official | 25 | WSD-con 9e-5 | -9.74% | -9.74% |
| frozen_official | 25 | WSD-con 18e-5 | -8.45% | -8.45% |
| frozen_official | 100 | WSD sharp | -21.08% | -21.08% |
| frozen_official | 100 | WSD linear | -22.85% | -22.85% |
| frozen_official | 100 | WSD-con 3e-5 | -11.08% | -11.08% |
| frozen_official | 100 | WSD-con 9e-5 | -6.29% | -6.29% |
| frozen_official | 100 | WSD-con 18e-5 | -8.42% | -8.42% |
| frozen_official | 400 | WSD sharp | -19.62% | -19.62% |
| frozen_official | 400 | WSD linear | -22.47% | -22.47% |
| frozen_official | 400 | WSD-con 3e-5 | -9.07% | -9.07% |
| frozen_official | 400 | WSD-con 9e-5 | -8.31% | -8.31% |
| frozen_official | 400 | WSD-con 18e-5 | -10.22% | -10.22% |
| cosine_independent | 25 | WSD sharp | -10.12% | +13.48% |
| cosine_independent | 25 | WSD linear | -8.04% | +13.21% |
| cosine_independent | 25 | WSD-con 3e-5 | -11.75% | +68.97% |
| cosine_independent | 25 | WSD-con 9e-5 | -12.97% | +79.99% |
| cosine_independent | 25 | WSD-con 18e-5 | -10.85% | +47.15% |
| cosine_independent | 100 | WSD sharp | -14.62% | -1.03% |
| cosine_independent | 100 | WSD linear | -13.98% | -4.19% |
| cosine_independent | 100 | WSD-con 3e-5 | -11.53% | +23.72% |
| cosine_independent | 100 | WSD-con 9e-5 | -11.37% | +12.38% |
| cosine_independent | 100 | WSD-con 18e-5 | -15.22% | +63.92% |
| cosine_independent | 400 | WSD sharp | -13.09% | +34.49% |
| cosine_independent | 400 | WSD linear | -13.95% | +35.44% |
| cosine_independent | 400 | WSD-con 3e-5 | -6.40% | +76.11% |
| cosine_independent | 400 | WSD-con 9e-5 | -8.55% | +23.52% |
| cosine_independent | 400 | WSD-con 18e-5 | -9.13% | +72.96% |
| shared_beta_gamma | 25 | WSD sharp | -9.97% | +13.00% |
| shared_beta_gamma | 25 | WSD linear | -7.92% | +12.75% |
| shared_beta_gamma | 25 | WSD-con 3e-5 | -10.73% | +63.95% |
| shared_beta_gamma | 25 | WSD-con 9e-5 | -11.91% | +74.04% |
| shared_beta_gamma | 25 | WSD-con 18e-5 | -9.90% | +43.08% |
| shared_beta_gamma | 100 | WSD sharp | -14.62% | -1.02% |
| shared_beta_gamma | 100 | WSD linear | -13.98% | -4.19% |
| shared_beta_gamma | 100 | WSD-con 3e-5 | -11.52% | +23.74% |
| shared_beta_gamma | 100 | WSD-con 9e-5 | -11.36% | +12.39% |
| shared_beta_gamma | 100 | WSD-con 18e-5 | -15.21% | +63.95% |
| shared_beta_gamma | 400 | WSD sharp | -13.09% | +35.46% |
| shared_beta_gamma | 400 | WSD linear | -13.97% | +36.40% |
| shared_beta_gamma | 400 | WSD-con 3e-5 | -6.92% | +78.77% |
| shared_beta_gamma | 400 | WSD-con 9e-5 | -9.15% | +25.61% |
| shared_beta_gamma | 400 | WSD-con 18e-5 | -9.72% | +75.64% |
| shared_c_beta_gamma | 25 | WSD sharp | -9.95% | +12.93% |
| shared_c_beta_gamma | 25 | WSD linear | -7.90% | +12.68% |
| shared_c_beta_gamma | 25 | WSD-con 3e-5 | -10.62% | +63.40% |
| shared_c_beta_gamma | 25 | WSD-con 9e-5 | -11.79% | +73.38% |
| shared_c_beta_gamma | 25 | WSD-con 18e-5 | -9.79% | +42.62% |
| shared_c_beta_gamma | 100 | WSD sharp | -14.62% | -1.02% |
| shared_c_beta_gamma | 100 | WSD linear | -13.98% | -4.18% |
| shared_c_beta_gamma | 100 | WSD-con 3e-5 | -11.52% | +23.74% |
| shared_c_beta_gamma | 100 | WSD-con 9e-5 | -11.36% | +12.39% |
| shared_c_beta_gamma | 100 | WSD-con 18e-5 | -15.21% | +63.95% |
| shared_c_beta_gamma | 400 | WSD sharp | -13.09% | +35.49% |
| shared_c_beta_gamma | 400 | WSD linear | -13.97% | +36.43% |
| shared_c_beta_gamma | 400 | WSD-con 3e-5 | -6.95% | +78.90% |
| shared_c_beta_gamma | 400 | WSD-con 9e-5 | -9.18% | +25.73% |
| shared_c_beta_gamma | 400 | WSD-con 18e-5 | -9.75% | +75.81% |

## Reading

- Sharing LD-shape parameters is interpretable because it treats \(\beta,\gamma\) or \(C,\beta,\gamma\) as scale-invariant schedule-response shape, while keeping loss scale parameters per model size.
- This reduces backbone parameters from 21 to 17 or 15 and adds no residual-fitted parameters.
- If shape sharing improves corrected WSD absolute error versus independent cosine-only MPL, it points to an identifiability issue in the cosine-only backbone rather than a need for a larger residual model.
- If it hurts, the result is still useful negative evidence: stronger interpretability constraints on the backbone are not sufficient by themselves.
