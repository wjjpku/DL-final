# Nuisance-Origin Audit

This audit compares the current soft DCT nuisance projection with exact projections onto MPL tangent spaces.  All variants keep the one-response formula, fit only one nonnegative coefficient from `cosine_72000.csv`, and use WSD/control losses only for evaluation.

## Nuisance Spaces

- `none`: no nuisance removal; this is the raw one-dimensional projection and should fail if cosine residual is contaminated by MPL drift.
- `dct_soft`: current low-frequency residualizer, with `8` DCT modes and `mu=0.01`.
- `mpl_core3`: tangent space of MPL backbone parameters \(L_0,A,\alpha\).
- `mpl_ld4`: tangent space of MPL LR-dependent parameters \(B,C,\beta,\gamma\).
- `mpl_all7`: all local MPL parameter directions.

The tangent variants remove residual directions that could be explained by local MPL parameter error, which is more mechanism-native than generic low-frequency filtering.

## Highlight Results

| role | response | nuisance | shrinkage | group | mean | worst | wins/non-harm |
|---|---|---|---|---|---:|---:|---:|
| no-nuisance raw projection | fixed_lambda_20 | none | tau_free_sqrt_retention | core_wsd | +672.31% | +2585.94% | 0/15 wins, 0/15 non-harm |
| no-nuisance raw projection | fixed_lambda_20 | none | tau_free_sqrt_retention | extra_control | +0.00% | +0.00% | 0/9 wins, 9/9 non-harm |
| DCT tau-free baseline | fixed_lambda_20 | dct_soft | tau_free_sqrt_retention | core_wsd | -20.77% | -5.86% | 15/15 wins, 15/15 non-harm |
| DCT tau-free baseline | fixed_lambda_20 | dct_soft | tau_free_sqrt_retention | extra_control | +0.00% | +0.00% | 0/9 wins, 9/9 non-harm |
| MPL-all tangent lower bound | fixed_lambda_20 | mpl_all7 | tau_free_sqrt_retention | core_wsd | -4.92% | -1.47% | 15/15 wins, 15/15 non-harm |
| MPL-all tangent lower bound | fixed_lambda_20 | mpl_all7 | tau_free_sqrt_retention | extra_control | +0.00% | +0.00% | 0/9 wins, 9/9 non-harm |
| MPL-LD tangent main candidate | two_point_five_roundfast20 | mpl_ld4 | ridge_tau_0p05 | core_wsd | -27.25% | -3.00% | 15/15 wins, 15/15 non-harm |
| MPL-LD tangent main candidate | two_point_five_roundfast20 | mpl_ld4 | ridge_tau_0p05 | extra_control | +0.00% | +0.00% | 0/9 wins, 9/9 non-harm |
| DCT performance reference | two_point_five_roundfast20 | dct_soft | ridge_tau_0p05 | core_wsd | -32.83% | -5.30% | 15/15 wins, 15/15 non-harm |
| DCT performance reference | two_point_five_roundfast20 | dct_soft | ridge_tau_0p05 | extra_control | +0.00% | +0.00% | 0/9 wins, 9/9 non-harm |
| MPL-core negative evidence | two_point_five_roundfast20 | mpl_core3 | ridge_tau_0p05 | core_wsd | +54.21% | +202.73% | 4/15 wins, 6/15 non-harm |
| MPL-core negative evidence | two_point_five_roundfast20 | mpl_core3 | ridge_tau_0p05 | extra_control | +0.00% | +0.00% | 0/9 wins, 9/9 non-harm |

## Core WSD Summary

| response | nuisance | shrinkage | locality | mean | worst | wins |
|---|---|---|---|---:|---:|---:|
| fixed_lambda_20 | dct_soft | ridge_tau_0p05 | none | -22.06% | -5.30% | 15/15 |
| fixed_lambda_20 | dct_soft | ridge_tau_0p05 | linear | -20.78% | -5.30% | 15/15 |
| fixed_lambda_20 | mpl_ld4 | ridge_tau_0p05 | none | -12.23% | -3.00% | 15/15 |
| fixed_lambda_20 | mpl_ld4 | ridge_tau_0p05 | linear | -11.66% | -3.00% | 15/15 |
| fixed_lambda_20 | mpl_all7 | ridge_tau_0p05 | none | -0.14% | -0.05% | 15/15 |
| fixed_lambda_20 | mpl_all7 | ridge_tau_0p05 | linear | -0.14% | -0.04% | 15/15 |
| fixed_lambda_20 | mpl_core3 | ridge_tau_0p05 | linear | +41.20% | +202.73% | 6/15 |
| fixed_lambda_20 | mpl_core3 | ridge_tau_0p05 | none | +43.27% | +202.78% | 6/15 |
| fixed_lambda_20 | constant_only | ridge_tau_0p05 | linear | +96.98% | +471.93% | 2/15 |
| fixed_lambda_20 | constant_only | ridge_tau_0p05 | none | +103.71% | +472.02% | 2/15 |
| fixed_lambda_20 | none | ridge_tau_0p05 | linear | +598.84% | +2322.74% | 0/15 |
| fixed_lambda_20 | none | ridge_tau_0p05 | none | +624.26% | +2323.10% | 0/15 |
| fixed_lambda_20 | dct_soft | tau_free_sqrt_retention | none | -21.92% | -5.86% | 15/15 |
| fixed_lambda_20 | dct_soft | tau_free_sqrt_retention | linear | -20.77% | -5.86% | 15/15 |
| fixed_lambda_20 | mpl_ld4 | tau_free_sqrt_retention | none | -16.30% | +14.55% | 12/15 |
| fixed_lambda_20 | mpl_ld4 | tau_free_sqrt_retention | linear | -14.60% | +14.53% | 12/15 |
| fixed_lambda_20 | mpl_all7 | tau_free_sqrt_retention | none | -5.15% | -1.80% | 15/15 |
| fixed_lambda_20 | mpl_all7 | tau_free_sqrt_retention | linear | -4.92% | -1.47% | 15/15 |
| fixed_lambda_20 | mpl_core3 | tau_free_sqrt_retention | none | +12.93% | +111.40% | 7/15 |
| fixed_lambda_20 | mpl_core3 | tau_free_sqrt_retention | linear | +15.19% | +111.37% | 7/15 |
| fixed_lambda_20 | constant_only | tau_free_sqrt_retention | linear | +47.99% | +277.54% | 4/15 |
| fixed_lambda_20 | constant_only | tau_free_sqrt_retention | none | +52.30% | +277.60% | 4/15 |
| fixed_lambda_20 | none | tau_free_sqrt_retention | linear | +672.31% | +2585.94% | 0/15 |
| fixed_lambda_20 | none | tau_free_sqrt_retention | none | +700.44% | +2586.34% | 0/15 |
| two_point_five_roundfast20 | dct_soft | ridge_tau_0p05 | none | -34.56% | -5.30% | 15/15 |
| two_point_five_roundfast20 | dct_soft | ridge_tau_0p05 | linear | -32.83% | -5.30% | 15/15 |
| two_point_five_roundfast20 | mpl_ld4 | ridge_tau_0p05 | linear | -27.25% | -3.00% | 15/15 |
| two_point_five_roundfast20 | mpl_ld4 | ridge_tau_0p05 | none | -24.86% | -3.00% | 15/15 |
| two_point_five_roundfast20 | mpl_all7 | ridge_tau_0p05 | none | -0.27% | -0.05% | 15/15 |
| two_point_five_roundfast20 | mpl_all7 | ridge_tau_0p05 | linear | -0.24% | -0.05% | 15/15 |
| two_point_five_roundfast20 | mpl_core3 | ridge_tau_0p05 | none | +53.09% | +202.78% | 4/15 |
| two_point_five_roundfast20 | mpl_core3 | ridge_tau_0p05 | linear | +54.21% | +202.73% | 4/15 |
| two_point_five_roundfast20 | constant_only | ridge_tau_0p05 | linear | +94.43% | +471.93% | 3/15 |
| two_point_five_roundfast20 | constant_only | ridge_tau_0p05 | none | +100.56% | +472.02% | 2/15 |
| two_point_five_roundfast20 | none | ridge_tau_0p05 | linear | +585.27% | +2322.74% | 0/15 |
| two_point_five_roundfast20 | none | ridge_tau_0p05 | none | +607.64% | +2323.10% | 0/15 |
| two_point_five_roundfast20 | dct_soft | tau_free_sqrt_retention | none | -20.90% | -5.86% | 15/15 |
| two_point_five_roundfast20 | dct_soft | tau_free_sqrt_retention | linear | -19.93% | -5.86% | 15/15 |
| two_point_five_roundfast20 | mpl_ld4 | tau_free_sqrt_retention | none | -17.83% | +14.55% | 12/15 |
| two_point_five_roundfast20 | mpl_ld4 | tau_free_sqrt_retention | linear | -15.85% | +14.53% | 12/15 |
| two_point_five_roundfast20 | mpl_all7 | tau_free_sqrt_retention | none | -5.25% | -1.88% | 15/15 |
| two_point_five_roundfast20 | mpl_all7 | tau_free_sqrt_retention | linear | -5.00% | -1.54% | 15/15 |
| two_point_five_roundfast20 | mpl_core3 | tau_free_sqrt_retention | none | +28.86% | +111.40% | 5/15 |
| two_point_five_roundfast20 | mpl_core3 | tau_free_sqrt_retention | linear | +29.12% | +111.37% | 5/15 |
| two_point_five_roundfast20 | constant_only | tau_free_sqrt_retention | none | +46.22% | +277.60% | 4/15 |
| two_point_five_roundfast20 | constant_only | tau_free_sqrt_retention | linear | +46.89% | +277.54% | 4/15 |
| two_point_five_roundfast20 | none | tau_free_sqrt_retention | linear | +648.04% | +2585.94% | 0/15 |
| two_point_five_roundfast20 | none | tau_free_sqrt_retention | none | +670.72% | +2586.34% | 0/15 |

## Reading

- Best tangent nuisance row: `two_point_five_roundfast20 / mpl_ld4 / ridge_tau_0p05`, mean `-27.25%`, worst `-3.00%`, wins `15/15`.
- The no-nuisance row is intentionally included as a failure mode: raw projection lets smooth MPL residual drift masquerade as the LR-drop response.
- The MPL-LD tangent nuisance (`mpl_ld4`) is now the mechanism-native main candidate.  It removes only local MPL LR-term error directions before estimating the response amplitude.
- DCT remains numerically stronger, but its generic low-frequency basis is less defensible as a core theory term.  It should be treated as a performance reference or diagnostic upper bound.
- `mpl_core3` fails, which is useful negative evidence: the removable drift is not just an error in the backbone trend \((L_0,A,\alpha)\).
