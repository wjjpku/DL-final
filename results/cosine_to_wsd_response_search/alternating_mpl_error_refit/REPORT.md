# Alternating MPL/Error Refit Audit

This audit implements the requested alternating procedure under the strict cosine-only protocol. Pure MPL is fit on `cosine_24000.csv` and `cosine_72000.csv`; the error coefficients are fit only from `cosine_72000.csv` residuals.

## Fixed Error Model Used

The error model is the strict-calibrated LR-curvature correction:

```text
smooth correction = k_smooth * phi_lambda=4
step correction   = a_step * phi_lambda=20 + b_curv * psi_diff_drop,lambda=30
```

Channel calibration:

```text
smooth: start=12000, mu=0.05, modes=8, tau=0.05, p=0.25, rho=0.2
step:   start=12000, mu=0.02, modes=12, tau=0.05, p=0,    rho=0
curv:   tau=0.001, shrink_curvature=true, nonnegative coefficient
```

## Best Fully Non-Harming Alternating Variant

- Subtraction variant: `step_only`.
- Final vs pure strict MPL: mean `-35.29%`, worst `-9.73%`, wins `15/15`.
- Final vs first two-stage correction: mean `-2.42%`, worst `+6.23%`, wins `12/15`.
- Reference strict-calibrated two-stage correction: mean `-33.68%`, worst `-14.27%`.

Best mean-only variant:

- `smooth_plus_step` reaches mean `-36.33%`, but worst `+12.67%` and wins `14/15`.

## Variant Summary

| variant | baseline | mean delta | worst | wins |
|---|---|---:|---:|---:|
| smooth_only | pure_mpl | -13.00% | +52.17% | 9/15 |
| smooth_only | first_error | +30.40% | +145.76% | 6/15 |
| step_only | pure_mpl | -35.29% | -9.73% | 15/15 |
| step_only | first_error | -2.42% | +6.23% | 12/15 |
| smooth_plus_step | pure_mpl | -36.33% | +12.67% | 14/15 |
| smooth_plus_step | first_error | -4.01% | +32.59% | 10/15 |

## Per-Target Result For Best Variant

| target | mean delta vs pure MPL | worst scale | wins |
|---|---:|---:|---:|
| WSD sharp | -41.5% | -34.1% | 3/3 |
| WSD linear | -30.8% | -22.6% | 3/3 |
| WSD-con 3e-5 | -57.1% | -55.1% | 3/3 |
| WSD-con 9e-5 | -29.2% | -28.5% | 3/3 |
| WSD-con 18e-5 | -17.9% | -9.7% | 3/3 |

## Coefficient Drift

| scale | first smooth | final smooth | first step | final step | first curv | final curv |
|---|---:|---:|---:|---:|---:|---:|
| 25M | 0.02421 | 0.02426 | 0.04770 | 0.04775 | 0.01103 | 0.01103 |
| 100M | 0.03800 | 0.03812 | 0.06854 | 0.06869 | 0.01525 | 0.01525 |
| 400M | 0.04898 | 0.04918 | 0.09121 | 0.09157 | 0.02024 | 0.02027 |

## Reading

- The useful test is the `final vs first two-stage correction` row. If it is positive, the alternating refit made WSD transfer worse even if it still beats pure MPL.
- `smooth_only` subtracts the correction that would actually be applied to cosine schedules. `step_only` and `smooth_plus_step` are diagnostics for whether the WSD-con channel should also be treated as part of the backbone-refit residual.
