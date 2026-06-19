# Observation-Bracket MPL-LD Audit

This audit removes the two weakest protocol constants from the previous MPL-LD response model.

## Formula

Let \(\Delta_{\mathrm{obs}}\) be the modal logging interval and

\[
\lambda_{\mathrm{obs}}=\frac{\log 2}{\eta_{\max}\Delta_{\mathrm{obs}}}.
\]

For a target schedule, define drop concentration

\[
q_s=\frac{\max_t [\eta_{t-1}-\eta_t]_+}{\sum_t [\eta_{t-1}-\eta_t]_+}.
\]

The new response rate is

\[
\lambda_s=\lambda_{\mathrm{obs}}\frac{1+q_s}{2}.
\]

Equivalently, the response half-life is \(2\Delta_{\mathrm{obs}}/(1+q_s)\): diffuse LR decay receives a two-observation half-life, while a single sharp drop receives a one-observation half-life.  No target loss is used.

After projecting both the cosine response feature and cosine residual away from the MPL-LD tangent space, the coefficient is

\[
\hat\kappa_s=\frac{\langle x_s,y\rangle_+}{\|x_s\|_2^2+1/N_{\mathrm{cal}}}.
\]

Here \(N_{\mathrm{cal}}\) is the number of source cosine points in the calibration suffix.  This replaces the fixed `tau=0.05` ridge with a finite-sample identifiability floor.

## Parameter Ledger

| quantity | role | source | fitted? | target loss? |
|---|---|---|---:|---:|
| MPL parameters | baseline predictor | precomputed MPL fit already used by baseline | 0 | outside_error_model |
| drop concentration q_s | response-rate interpolation | target LR schedule | 0 | 0 |
| lambda_obs | observation-scale response unit | modal logging interval and peak LR | 0 | 0 |
| lambda_s | target response rate | lambda_obs * (1 + q_s) / 2 | 0 | 0 |
| MPL-LD tangent projection | nuisance removal | finite differences of MPL LR-dependent parameters B,C,beta,gamma | 0 | 0 |
| fit_start | calibration suffix boundary | earliest source-only lambda-bracket retention pass | 0 | 0 |
| 1/N_cal ridge | finite-sample identifiability floor | number of source calibration points | 0 | 0 |
| kappa_hat_s | response amplitude | one nonnegative projection from cosine residual | 1 | 0 |
| locality factor a_s | schedule-boundary condition for controls | LR-drop support span | 0 | 0 |

The only residual-fitted quantity introduced by the error model is the nonnegative scalar \(\hat\kappa_s\).  Every other term is derived from the LR schedule, the logging resolution, the source suffix size, or the existing MPL formula.

## Locality Boundary

The locality factor is a schedule-support boundary condition, not a learned gate:

\[
a_s=\mathbf{1}\{\sum_t d_t>0\}\left[1-\frac{\ell_s}{T_s-W}\right]_+,
\]

where \(\ell_s\) is the support span of positive LR drops after warmup.  It uses only the LR schedule and is never fit from loss values.

| curve | group | median factor | support span | post-warmup span |
|---|---|---:|---:|---:|
| WSD sharp | core_wsd | 0.8168 | 4000 | 21840 |
| WSD linear | core_wsd | 0.8168 | 4000 | 21840 |
| WSD-con 3e-5 | core_wsd | 0.9999 | 2 | 13840 |
| WSD-con 9e-5 | core_wsd | 0.9999 | 2 | 13840 |
| WSD-con 18e-5 | core_wsd | 0.9999 | 2 | 13840 |
| Cosine 24k | extra_control | 0.0000 | 21840 | 21840 |
| Constant 24k | extra_control | 0.0000 | 0 | 21840 |
| Constant 72k | extra_control | 0.0000 | 0 | 69840 |

Locality tradeoff:

- Without locality, WSD remains all-win: `-30.89% / -4.67% / 15/15`.
- Without locality, controls fail: `+13.39% / +56.99% / 0/9`.
- Linear locality changes same-scale WSD mean by `+1.01` percentage points while restoring all controls to non-harm.

## Source-Only Suffix Rule

The calibration suffix is selected without WSD losses or target schedule enumeration.  For candidate suffix starts, evaluate the two endpoints of the observation bracket, \(\lambda_{\mathrm{obs}}/2\) and \(\lambda_{\mathrm{obs}}\), on the source cosine curve and compute the retained response-feature energy after MPL-LD projection:

\[
\rho=\frac{\|(I-P_{\mathrm{LD}})\phi\|_2^2}{\|\phi\|_2^2}.
\]

Choose the earliest suffix start whose maximum endpoint \(\rho\) over source scales is below the finite-sample floor \(1/N_{\mathrm{cal}}\).  A dense grid check with 2, 3, 5, 9, 17, 33, 65, and 129 points selects the same suffix, so this endpoint rule is not a grid-resolution artifact.  This avoids early cosine segments where the response direction is still too entangled with MPL-LD drift.

Selected fit start: `8000`.

| fit start | lambda points | max retention | median retention | floor | passes |
|---:|---:|---:|---:|---:|---:|
| 5000 | 2 | 0.0137479 | 0.0117008 | 0.00191205 | 0 |
| 6500 | 2 | 0.00422538 | 0.00382383 | 0.00195312 | 0 |
| 8000 | 2 | 0.00147784 | 0.00131267 | 0.002 | 1 |
| 10000 | 2 | 0.000489295 | 0.000462062 | 0.00206612 | 1 |
| 12000 | 2 | 0.00035334 | 0.000270008 | 0.0021322 | 1 |

## Key Results

| variant | split | group | mean / worst / wins |
|---|---|---|---:|
| observation_bracket_mplld_neff | same_scale | core_wsd | -29.87% / -4.67% / 15/15 |
| observation_bracket_mplld_neff | cross_scale | core_wsd | -24.95% / -3.15% / 30/30 |
| observation_bracket_mplld_neff | same_scale | extra_control | +0.00% / +0.00% / 0/9 |
| old_mplld_fixedtau | same_scale | core_wsd | -27.25% / -3.00% / 15/15 |
| no-nuisance failure | same_scale | core_wsd | +602.17% / +2366.35% / 0/15 |
| no-locality control boundary | same_scale | extra_control | +13.39% / +56.99% / 0/9 |

## Fit-Start Sensitivity

| fit start | group | mean | worst | wins/non-harm |
|---:|---|---:|---:|---:|
| 5000 | core_wsd | +123.32% | +399.52% | 0/15, 0/15 |
| 5000 | extra_control | +0.00% | +0.00% | 0/9, 9/9 |
| 6500 | core_wsd | +11.88% | +90.39% | 5/15, 5/15 |
| 6500 | extra_control | +0.00% | +0.00% | 0/9, 9/9 |
| 8000 | core_wsd | -29.87% | -4.67% | 15/15, 15/15 |
| 8000 | extra_control | +0.00% | +0.00% | 0/9, 9/9 |
| 10000 | core_wsd | -8.53% | -1.37% | 15/15, 15/15 |
| 10000 | extra_control | +0.00% | +0.00% | 0/9, 9/9 |
| 12000 | core_wsd | -1.49% | +0.00% | 6/15, 15/15 |
| 12000 | extra_control | +0.00% | +0.00% | 0/9, 9/9 |

## Reading

- The observation-bracket rule is stronger than the previous MPL-LD reference while removing the fixed `2.5` slow endpoint, rounded fast endpoint `20`, and fixed `tau=0.05`.
- Raw projection still fails badly, so the MPL-LD nuisance projection remains essential.
- No-locality WSD performance remains positive, but controls fail; locality should be written as a schedule-boundary condition, not as the mechanism itself.
- The fit-start scan is a protocol audit.  A result is research-safe only if the main conclusion is not tied to a single suffix boundary.
