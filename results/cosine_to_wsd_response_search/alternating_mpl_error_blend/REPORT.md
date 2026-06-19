# Under-Relaxed Alternating MPL/Error Audit

This audit does not refit parameters. It blends the original two-stage prediction with the full alternating prediction:

```text
L_hat(w) = (1 - w) L_hat_first + w L_hat_alternating
```

`w=0` is the strict-calibrated two-stage correction. `w=1` is the full alternating refit.

## Best Fully Non-Harming Blend

- Variant / weight: `smooth_plus_step`, `w=0.60`.
- Vs pure strict MPL: mean `-36.41%`, worst `-1.67%`, wins `15/15`.
- Vs first two-stage correction: mean `-4.17%`, worst `+15.72%`, wins `12/15`.

## Best Worst-Case Blend

- Variant / weight: `step_only`, `w=0.15`.
- Mean / worst: `-33.96%` / `-14.43%`.

## Best Mean-Only Blend

- Variant / weight: `smooth_plus_step`, `w=0.75`.
- Mean / worst / wins: `-36.49%` / `+3.71%` / `14/15`.

## Per-Target Result For Best Non-Harming Blend

| target | mean delta vs pure MPL | worst scale | wins |
|---|---:|---:|---:|
| WSD sharp | -43.7% | -37.7% | 3/3 |
| WSD linear | -34.2% | -27.6% | 3/3 |
| WSD-con 3e-5 | -57.5% | -56.5% | 3/3 |
| WSD-con 9e-5 | -29.9% | -27.9% | 3/3 |
| WSD-con 18e-5 | -16.7% | -1.7% | 3/3 |

## Reading

- This is an under-relaxed fixed-point interpretation of the alternating update. It is cheaper and more stable than another MPL refit.
- A useful blend should improve over the first two-stage correction without reintroducing positive worst-case rows.
