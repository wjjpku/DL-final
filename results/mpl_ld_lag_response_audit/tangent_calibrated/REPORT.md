# Tangent-Calibrated MPL-LD Finite-Response Audit

This audit estimates one scalar finite-response amplitude from cosine residuals after projecting out the local MPL tangent space.  It is designed to test whether cosine residual contamination can be removed without gate/channel/DCT/sinusoidal terms.

Calibration formula:

\[
\hat\kappa_s=\frac{\langle P_\perp\phi_s, P_\perp r_s\rangle_+}{\|P_\perp\phi_s\|^2},\qquad \phi_s(t)=B_s[D_{\downarrow,\tau_s,s}(t)-D_{\downarrow,s}(t)].
\]

Prediction formula:

\[
\hat L_s(t)=L_{\mathrm{MPL},s}(t)+a_s\hat\kappa_s\phi_s(t).
\]

## Main Rows

| protocol | fit start | tangent | shrinkage | WSD correction | wins / non-harm | corrected vs official | controls non-harm |
|---|---:|---|---|---:|---:|---:|---:|
| frozen_official | 5000 | none | orth_ols | +14.58% mean / +65.63% worst | 6/15 / 6/15 | +14.58% mean | 9/9 |
| frozen_official | 5000 | none | sqrt_retention | +14.58% mean / +65.63% worst | 6/15 / 6/15 | +14.58% mean | 9/9 |
| frozen_official | 5000 | none | full_energy | +14.58% mean / +65.63% worst | 6/15 / 6/15 | +14.58% mean | 9/9 |
| frozen_official | 5000 | ld4 | orth_ols | +7.47% mean / +100.92% worst | 9/15 / 9/15 | +7.47% mean | 9/9 |
| frozen_official | 5000 | ld4 | sqrt_retention | -8.29% mean / -1.48% worst | 15/15 / 15/15 | -8.29% mean | 9/9 |
| frozen_official | 5000 | ld4 | full_energy | -1.69% mean / -0.26% worst | 15/15 / 15/15 | -1.69% mean | 9/9 |
| frozen_official | 5000 | all7 | orth_ols | +0.00% mean / +0.00% worst | 0/15 / 15/15 | +0.00% mean | 9/9 |
| frozen_official | 5000 | all7 | sqrt_retention | +0.00% mean / +0.00% worst | 0/15 / 15/15 | +0.00% mean | 9/9 |
| frozen_official | 5000 | all7 | full_energy | +0.00% mean / +0.00% worst | 0/15 / 15/15 | +0.00% mean | 9/9 |
| cosine_only | 5000 | none | orth_ols | -0.02% mean / +0.00% worst | 5/15 / 15/15 | +55.02% mean | 9/9 |
| cosine_only | 5000 | none | sqrt_retention | -0.02% mean / +0.00% worst | 5/15 / 15/15 | +55.02% mean | 9/9 |
| cosine_only | 5000 | none | full_energy | -0.02% mean / +0.00% worst | 5/15 / 15/15 | +55.02% mean | 9/9 |
| cosine_only | 5000 | ld4 | orth_ols | +245.75% mean / +935.42% worst | 0/15 / 5/15 | +393.19% mean | 9/9 |
| cosine_only | 5000 | ld4 | sqrt_retention | -7.59% mean / +0.00% worst | 10/15 / 15/15 | +43.96% mean | 9/9 |
| cosine_only | 5000 | ld4 | full_energy | -0.30% mean / +0.00% worst | 10/15 / 15/15 | +54.61% mean | 9/9 |
| cosine_only | 5000 | all7 | orth_ols | +362.07% mean / +1520.29% worst | 0/15 / 5/15 | +541.90% mean | 9/9 |
| cosine_only | 5000 | all7 | sqrt_retention | -4.94% mean / +0.00% worst | 10/15 / 15/15 | +46.66% mean | 9/9 |
| cosine_only | 5000 | all7 | full_energy | -3.88% mean / +0.00% worst | 10/15 / 15/15 | +48.52% mean | 9/9 |

## Fit-Start Sensitivity

| protocol | tangent | shrinkage | fit start | WSD mean | WSD worst | wins |
|---|---|---|---:|---:|---:|---:|
| frozen_official | all7 | sqrt_retention | 2160 | -3.51% | +0.00% | 5/15 |
| frozen_official | all7 | sqrt_retention | 5000 | +0.00% | +0.00% | 0/15 |
| frozen_official | all7 | sqrt_retention | 8000 | +0.00% | +0.00% | 0/15 |
| frozen_official | all7 | full_energy | 2160 | -0.60% | +0.00% | 5/15 |
| frozen_official | all7 | full_energy | 5000 | +0.00% | +0.00% | 0/15 |
| frozen_official | all7 | full_energy | 8000 | +0.00% | +0.00% | 0/15 |
| frozen_official | all7 | orth_ols | 2160 | +2.14% | +20.03% | 0/15 |
| frozen_official | all7 | orth_ols | 5000 | +0.00% | +0.00% | 0/15 |
| frozen_official | all7 | orth_ols | 8000 | +0.00% | +0.00% | 0/15 |
| frozen_official | ld4 | sqrt_retention | 2160 | +0.00% | +0.00% | 0/15 |
| frozen_official | ld4 | sqrt_retention | 5000 | -8.29% | -1.48% | 15/15 |
| frozen_official | ld4 | sqrt_retention | 8000 | -4.56% | +0.00% | 10/15 |
| frozen_official | ld4 | full_energy | 2160 | +0.00% | +0.00% | 0/15 |
| frozen_official | ld4 | full_energy | 5000 | -1.69% | -0.26% | 15/15 |
| frozen_official | ld4 | full_energy | 8000 | -0.65% | +0.00% | 10/15 |
| frozen_official | ld4 | orth_ols | 2160 | +0.00% | +0.00% | 0/15 |
| frozen_official | ld4 | orth_ols | 5000 | +7.47% | +100.92% | 9/15 |
| frozen_official | ld4 | orth_ols | 8000 | -4.03% | +16.93% | 7/15 |
| frozen_official | none | sqrt_retention | 2160 | +14.31% | +65.61% | 6/15 |
| frozen_official | none | sqrt_retention | 5000 | +14.58% | +65.63% | 6/15 |
| frozen_official | none | sqrt_retention | 8000 | +15.19% | +69.41% | 5/15 |
| frozen_official | none | full_energy | 2160 | +14.31% | +65.61% | 6/15 |
| frozen_official | none | full_energy | 5000 | +14.58% | +65.63% | 6/15 |
| frozen_official | none | full_energy | 8000 | +15.19% | +69.41% | 5/15 |
| frozen_official | none | orth_ols | 2160 | +14.31% | +65.61% | 6/15 |
| frozen_official | none | orth_ols | 5000 | +14.58% | +65.63% | 6/15 |
| frozen_official | none | orth_ols | 8000 | +15.19% | +69.41% | 5/15 |
| cosine_only | all7 | sqrt_retention | 2160 | -7.50% | -1.47% | 15/15 |
| cosine_only | all7 | sqrt_retention | 5000 | -4.94% | +0.00% | 10/15 |
| cosine_only | all7 | sqrt_retention | 8000 | -7.07% | +0.00% | 10/15 |
| cosine_only | all7 | full_energy | 2160 | -5.08% | -0.36% | 15/15 |
| cosine_only | all7 | full_energy | 5000 | -3.88% | +0.00% | 10/15 |
| cosine_only | all7 | full_energy | 8000 | -4.19% | +0.00% | 10/15 |
| cosine_only | all7 | orth_ols | 2160 | -5.12% | +28.73% | 11/15 |
| cosine_only | all7 | orth_ols | 5000 | +362.07% | +1520.29% | 0/15 |
| cosine_only | all7 | orth_ols | 8000 | +3276.25% | +13690.11% | 0/15 |
| cosine_only | ld4 | sqrt_retention | 2160 | -7.89% | +0.00% | 10/15 |
| cosine_only | ld4 | sqrt_retention | 5000 | -7.59% | +0.00% | 10/15 |
| cosine_only | ld4 | sqrt_retention | 8000 | -6.43% | +0.00% | 10/15 |
| cosine_only | ld4 | full_energy | 2160 | -3.56% | +0.00% | 10/15 |
| cosine_only | ld4 | full_energy | 5000 | -0.30% | +0.00% | 10/15 |
| cosine_only | ld4 | full_energy | 8000 | -0.21% | +0.00% | 10/15 |
| cosine_only | ld4 | orth_ols | 2160 | -8.68% | +0.00% | 10/15 |
| cosine_only | ld4 | orth_ols | 5000 | +245.75% | +935.42% | 0/15 |
| cosine_only | ld4 | orth_ols | 8000 | +263.33% | +1088.58% | 0/15 |
| cosine_only | none | sqrt_retention | 2160 | -0.28% | +0.00% | 5/15 |
| cosine_only | none | sqrt_retention | 5000 | -0.02% | +0.00% | 5/15 |
| cosine_only | none | sqrt_retention | 8000 | +0.00% | +0.00% | 0/15 |
| cosine_only | none | full_energy | 2160 | -0.28% | +0.00% | 5/15 |
| cosine_only | none | full_energy | 5000 | -0.02% | +0.00% | 5/15 |
| cosine_only | none | full_energy | 8000 | +0.00% | +0.00% | 0/15 |
| cosine_only | none | orth_ols | 2160 | -0.28% | +0.00% | 5/15 |
| cosine_only | none | orth_ols | 5000 | -0.02% | +0.00% | 5/15 |
| cosine_only | none | orth_ols | 8000 | +0.00% | +0.00% | 0/15 |

## Reading

- `none` is the known bad raw cosine-amplitude path if it over-transfers.
- `ld4` removes only MPL's LR-dependent tangent directions; `all7` removes the full local MPL parameter tangent.
- `orth_ols` is the direct projected least-squares estimator; `sqrt_retention` and `full_energy` are no-hyperparameter energy normalizations that prevent tiny projected feature energy from creating a huge amplitude.
- This still has one fitted scalar per scale.  It is more flexible than the zero-parameter finite-response row, so it must be judged by strict cosine-only performance and sensitivity, not just the best frozen-backbone number.
