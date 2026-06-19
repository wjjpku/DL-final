# MPL-LD Tau/Boundary Rule Sensitivity

All rows use the same cooldown finite-response correction and fit no residual coefficient.  This audit checks whether the recommended support-bracket rule is a stable schedule-only choice rather than a one-off tuned point.

## Linear Adiabatic Boundary

| tau rule | role | explanation | WSD mean / worst / wins | controls mean / worst / nonharm |
|---|---|---|---:|---:|
| fixed_one_obs | conservative_lower | strong | -8.73% / -6.22% / 15/15 | +0.00% / +0.00% / 9/9 |
| support_linear_bracket | recommended | strong | -13.77% / -6.29% / 15/15 | +0.00% / +0.00% / 9/9 |
| support_hard_two_obs | stepwise_reference | medium | -13.70% / -6.22% / 15/15 | +0.00% / +0.00% / 9/9 |
| support_sqrt_bracket | nonlinear_reference | weaker | -14.15% / -6.79% / 15/15 | +0.00% / +0.00% / 9/9 |
| support_log_bracket | nonlinear_reference | weaker | -14.36% / -6.65% / 15/15 | +0.00% / +0.00% / 9/9 |
| fixed_two_obs | unsafe_upper | strong | -13.52% / +14.96% / 14/15 | +0.00% / +0.00% / 9/9 |

## Boundary Ablation

| tau rule | boundary | WSD mean / worst / wins | controls mean / worst / nonharm |
|---|---|---:|---:|
| fixed_one_obs | none | -9.44% / -6.22% / 15/15 | +0.26% / +6.53% / 8/9 |
| fixed_one_obs | linear_support | -8.73% / -6.22% / 15/15 | +0.00% / +0.00% / 9/9 |
| support_linear_bracket | none | -15.15% / -6.29% / 15/15 | +1.52% / +17.28% / 8/9 |
| support_linear_bracket | linear_support | -13.77% / -6.29% / 15/15 | +0.00% / +0.00% / 9/9 |
| fixed_two_obs | none | -14.90% / +14.96% / 14/15 | +1.52% / +17.28% / 8/9 |
| fixed_two_obs | linear_support | -13.52% / +14.96% / 14/15 | +0.00% / +0.00% / 9/9 |

## Reading

- Fixed one-observation tau is conservative and all-win but weaker.
- Fixed two-observation tau is too slow for small WSD-con drops and creates failures.
- Support-bracket tau keeps the one-observation behavior for single-step drops and two-observation behavior for extended cooldowns, giving the best strong-explanation row.
- Sqrt/log support rules are slightly stronger but use softer nonlinear priors; keep them as robustness references, not the main formula.
- The linear adiabatic boundary is necessary for controls: without it, diffuse cosine cooldown is incorrectly treated as a local transient.
