# Alternating Forced-Step Refit Audit

This audit removes smooth/step routing. Every target uses the same step-response correction:

```text
C(t) = a * phi_20(t) + b * psi(t)
```

The optimization is the frozen alternating procedure: fit MPL, freeze MPL and fit residual, freeze residual and refit MPL on corrected cosine losses, then freeze MPL and fit residual again.

## Best Final Alternating Result

- Config: `forced_step_best_mean`.
- First two-stage vs pure MPL: mean `-17.85%`, worst `-7.41%`, wins `15/15`.
- Alternating final vs pure MPL: mean `-17.82%`, worst `-5.01%`, wins `15/15`.
- Alternating final vs first two-stage: mean `+0.18%`, worst `+6.00%`, wins `8/15`.

## Config Summary

| config | baseline | mean delta | worst | wins |
|---|---|---:|---:|---:|
| forced_step_best_mean | pure_mpl | -17.82% | -5.01% | 15/15 |
| forced_step_best_mean | first_error | +0.18% | +6.00% | 8/15 |
| forced_step_best_mean | first_vs_pure | -17.85% | -7.41% | 15/15 |
| forced_step_best_worst | pure_mpl | -17.04% | -4.25% | 15/15 |
| forced_step_best_worst | first_error | -0.11% | +5.67% | 9/15 |
| forced_step_best_worst | first_vs_pure | -16.87% | -7.30% | 15/15 |
| forced_step_current_core | pure_mpl | -17.33% | -4.34% | 15/15 |
| forced_step_current_core | first_error | -0.31% | +5.73% | 10/15 |
| forced_step_current_core | first_vs_pure | -17.00% | -7.70% | 15/15 |

## Per-Target Result For Best Final Config

| target | first two-stage mean | alternating final mean | final worst | final wins |
|---|---:|---:|---:|---:|
| WSD sharp | -11.8% | -12.9% | -11.1% | 3/3 |
| WSD linear | -10.1% | -11.7% | -10.5% | 3/3 |
| WSD-con 3e-5 | -40.0% | -38.7% | -33.5% | 3/3 |
| WSD-con 9e-5 | -17.0% | -16.5% | -15.4% | 3/3 |
| WSD-con 18e-5 | -10.4% | -9.3% | -5.0% | 3/3 |

## Reading

- The comparison that matters is `alternating final vs first two-stage`. If this row is positive, the frozen iteration did not improve the forced-step estimator.
- This audit still uses only cosine losses for fitting MPL and residual coefficients. WSD-family curves are used for evaluation and development ranking only.
