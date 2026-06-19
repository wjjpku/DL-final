# Shrinkage-Origin Audit

This audit isolates the role of the ridge constant.  All variants keep the one-response formula and fit only one nonnegative coefficient from `cosine_72000.csv` residuals.  WSD-family and control losses are evaluation only.

## Coefficient Rules

Let \(x=M_\mu\phi_{\lambda,\cos}\), \(y=M_\mu r_{\cos}\), and \(\phi=\phi_{\lambda,\cos}\) on the source suffix.

Current ridge rule:

\[
\hat\kappa=\frac{\langle x,y\rangle_+}{\|x\|^2+0.05^2}.
\]

Tau-free sqrt-retention rule:

\[
\hat\kappa=\frac{\langle x,y\rangle_+}{\|x\|\,\|\phi\|}.
\]

Tau-free full-energy rule:

\[
\hat\kappa=\frac{\langle x,y\rangle_+}{\|\phi\|^2}.
\]

The tau-free rules shrink automatically when most feature energy is removed by the nuisance projection.  They introduce no fitted parameter and no fixed ridge constant.

## Highlight Results

| role | response | shrinkage | locality | group | mean | worst | wins/non-harm |
|---|---|---|---|---|---:|---:|---:|
| tau-free hard baseline | fixed_lambda_20 | tau_free_sqrt_retention | linear | core_wsd | -20.77% | -5.86% | 15/15 wins, 15/15 non-harm |
| tau-free hard baseline | fixed_lambda_20 | tau_free_sqrt_retention | linear | extra_control | +0.00% | +0.00% | 0/9 wins, 9/9 non-harm |
| most conservative tau-free baseline | fixed_lambda_20 | tau_free_full_energy | linear | core_wsd | -3.72% | -1.30% | 15/15 wins, 15/15 non-harm |
| most conservative tau-free baseline | fixed_lambda_20 | tau_free_full_energy | linear | extra_control | +0.00% | +0.00% | 0/9 wins, 9/9 non-harm |
| two-observation performance variant | two_observation_roundfast20 | ridge_tau_0p05 | linear | core_wsd | -29.82% | -5.30% | 15/15 wins, 15/15 non-harm |
| two-observation performance variant | two_observation_roundfast20 | ridge_tau_0p05 | linear | extra_control | +0.00% | +0.00% | 0/9 wins, 9/9 non-harm |
| old high-performance reference | two_point_five_roundfast20 | ridge_tau_0p05 | linear | core_wsd | -32.83% | -5.30% | 15/15 wins, 15/15 non-harm |
| old high-performance reference | two_point_five_roundfast20 | ridge_tau_0p05 | linear | extra_control | +0.00% | +0.00% | 0/9 wins, 9/9 non-harm |

## All Summary Rows

| response | shrinkage | locality | group | mean | worst | wins | non-harm |
|---|---|---|---|---:|---:|---:|---:|
| fixed_lambda_20 | ridge_tau_0p05 | linear | core_wsd | -20.78% | -5.30% | 15/15 | 15/15 |
| fixed_lambda_20 | tau_free_full_energy | linear | core_wsd | -3.72% | -1.30% | 15/15 | 15/15 |
| fixed_lambda_20 | tau_free_sqrt_retention | linear | core_wsd | -20.77% | -5.86% | 15/15 | 15/15 |
| fixed_lambda_20 | ridge_tau_0p05 | none | core_wsd | -22.06% | -5.30% | 15/15 | 15/15 |
| fixed_lambda_20 | tau_free_full_energy | none | core_wsd | -3.88% | -1.59% | 15/15 | 15/15 |
| fixed_lambda_20 | tau_free_sqrt_retention | none | core_wsd | -21.92% | -5.86% | 15/15 | 15/15 |
| fixed_lambda_obs | ridge_tau_0p05 | linear | core_wsd | -19.08% | -1.09% | 15/15 | 15/15 |
| fixed_lambda_obs | tau_free_full_energy | linear | core_wsd | -3.53% | -1.25% | 15/15 | 15/15 |
| fixed_lambda_obs | tau_free_sqrt_retention | linear | core_wsd | -20.41% | -6.63% | 15/15 | 15/15 |
| fixed_lambda_obs | ridge_tau_0p05 | none | core_wsd | -20.55% | -1.09% | 15/15 | 15/15 |
| fixed_lambda_obs | tau_free_full_energy | none | core_wsd | -3.69% | -1.52% | 15/15 | 15/15 |
| fixed_lambda_obs | tau_free_sqrt_retention | none | core_wsd | -21.54% | -6.63% | 15/15 | 15/15 |
| two_observation_roundfast20 | ridge_tau_0p05 | linear | core_wsd | -29.82% | -5.30% | 15/15 | 15/15 |
| two_observation_roundfast20 | tau_free_full_energy | linear | core_wsd | -3.48% | -0.85% | 15/15 | 15/15 |
| two_observation_roundfast20 | tau_free_sqrt_retention | linear | core_wsd | -20.17% | -5.86% | 15/15 | 15/15 |
| two_observation_roundfast20 | ridge_tau_0p05 | none | core_wsd | -32.66% | -5.30% | 15/15 | 15/15 |
| two_observation_roundfast20 | tau_free_full_energy | none | core_wsd | -3.59% | -1.04% | 15/15 | 15/15 |
| two_observation_roundfast20 | tau_free_sqrt_retention | none | core_wsd | -21.18% | -5.86% | 15/15 | 15/15 |
| two_point_five_roundfast20 | ridge_tau_0p05 | linear | core_wsd | -32.83% | -5.30% | 15/15 | 15/15 |
| two_point_five_roundfast20 | tau_free_full_energy | linear | core_wsd | -3.41% | -0.72% | 15/15 | 15/15 |
| two_point_five_roundfast20 | tau_free_sqrt_retention | linear | core_wsd | -19.93% | -5.86% | 15/15 | 15/15 |
| two_point_five_roundfast20 | ridge_tau_0p05 | none | core_wsd | -34.56% | -5.30% | 15/15 | 15/15 |
| two_point_five_roundfast20 | tau_free_full_energy | none | core_wsd | -3.51% | -0.89% | 15/15 | 15/15 |
| two_point_five_roundfast20 | tau_free_sqrt_retention | none | core_wsd | -20.90% | -5.86% | 15/15 | 15/15 |
| fixed_lambda_20 | ridge_tau_0p05 | linear | extra_control | +0.00% | +0.00% | 0/9 | 9/9 |
| fixed_lambda_20 | tau_free_full_energy | linear | extra_control | +0.00% | +0.00% | 0/9 | 9/9 |
| fixed_lambda_20 | tau_free_sqrt_retention | linear | extra_control | +0.00% | +0.00% | 0/9 | 9/9 |
| fixed_lambda_20 | ridge_tau_0p05 | none | extra_control | +3.04% | +15.43% | 0/9 | 6/9 |
| fixed_lambda_20 | tau_free_full_energy | none | extra_control | +0.36% | +1.96% | 0/9 | 6/9 |
| fixed_lambda_20 | tau_free_sqrt_retention | none | extra_control | +2.69% | +13.82% | 0/9 | 6/9 |
| fixed_lambda_obs | ridge_tau_0p05 | linear | extra_control | +0.00% | +0.00% | 0/9 | 9/9 |
| fixed_lambda_obs | tau_free_full_energy | linear | extra_control | +0.00% | +0.00% | 0/9 | 9/9 |
| fixed_lambda_obs | tau_free_sqrt_retention | linear | extra_control | +0.00% | +0.00% | 0/9 | 9/9 |
| fixed_lambda_obs | ridge_tau_0p05 | none | extra_control | +3.62% | +18.01% | 0/9 | 6/9 |
| fixed_lambda_obs | tau_free_full_energy | none | extra_control | +0.35% | +1.91% | 0/9 | 6/9 |
| fixed_lambda_obs | tau_free_sqrt_retention | none | extra_control | +2.69% | +13.80% | 0/9 | 6/9 |
| two_observation_roundfast20 | ridge_tau_0p05 | linear | extra_control | +0.00% | +0.00% | 0/9 | 9/9 |
| two_observation_roundfast20 | tau_free_full_energy | linear | extra_control | +0.00% | +0.00% | 0/9 | 9/9 |
| two_observation_roundfast20 | tau_free_sqrt_retention | linear | extra_control | +0.00% | +0.00% | 0/9 | 9/9 |
| two_observation_roundfast20 | ridge_tau_0p05 | none | extra_control | +10.42% | +44.35% | 0/9 | 6/9 |
| two_observation_roundfast20 | tau_free_full_energy | none | extra_control | +0.28% | +1.48% | 0/9 | 6/9 |
| two_observation_roundfast20 | tau_free_sqrt_retention | none | extra_control | +2.68% | +13.53% | 0/9 | 6/9 |
| two_point_five_roundfast20 | ridge_tau_0p05 | linear | extra_control | +0.00% | +0.00% | 0/9 | 9/9 |
| two_point_five_roundfast20 | tau_free_full_energy | linear | extra_control | +0.00% | +0.00% | 0/9 | 9/9 |
| two_point_five_roundfast20 | tau_free_sqrt_retention | linear | extra_control | +0.00% | +0.00% | 0/9 | 9/9 |
| two_point_five_roundfast20 | ridge_tau_0p05 | none | extra_control | +14.02% | +56.43% | 0/9 | 6/9 |
| two_point_five_roundfast20 | tau_free_full_energy | none | extra_control | +0.25% | +1.32% | 0/9 | 6/9 |
| two_point_five_roundfast20 | tau_free_sqrt_retention | none | extra_control | +2.65% | +13.37% | 0/9 | 6/9 |

## Tau-Free Hard Baseline Per Target

| target | mean | worst | wins |
|---|---:|---:|---:|
| WSD linear | -12.19% | -9.21% | 3/3 |
| WSD sharp | -13.53% | -10.94% | 3/3 |
| WSD-con 18e-5 | -9.46% | -5.86% | 3/3 |
| WSD-con 3e-5 | -54.28% | -48.23% | 3/3 |
| WSD-con 9e-5 | -14.40% | -8.11% | 3/3 |

## Reading

- The one-response mechanism does not require the fixed ridge constant to be useful: `fixed_lambda_20 + tau_free_sqrt_retention + linear locality` improves all 15 WSD-family rows and keeps all 9 controls non-harm.
- The price of removing `tau` and schedule-geometry tuning is lower mean gain: about `-20.77%` instead of the old `-32%` to `-34%` performance variants.
- The full-energy rule is the most conservative and still improves every WSD row, but the gain is small.  This is a useful lower-bound sanity check rather than a competitive model.
- The ridge-based variants should now be described as performance extensions over a tau-free identifiable-response baseline, not as the only evidence that the formula works.
- This gives a cleaner research story: first prove the mechanism with a tau-free estimator, then separately justify whether the ridge performance extension is worth the extra protocol assumption.
