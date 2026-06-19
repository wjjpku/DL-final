# Schedule-Adaptive Cosine-to-WSD Audit

This audit tests a schedule-only extension of the cosine-calibrated response model. The target loss curve is never used to fit kappa; the target LR schedule only selects the response rate.

```text
drop_concentration = max_t relu(lr_{t-1}-lr_t) / sum_t relu(lr_{t-1}-lr_t)
lambda_target = 20 if drop_concentration >= 0.2 else 7
```

The shared estimator hyperparameters are borrowed from the global mean candidate: `mu=0.07`, `max_mode=8`, `tau=0.05`, `p=0.5`, `rho=0`.

## Aggregate Comparison

| method | mean delta | worst delta | wins |
|---|---:|---:|---:|
| `old_nextgen` | -17.2% | -2.2% | 15/15 |
| `global_best_mean` | -22.0% | -6.5% | 15/15 |
| `global_best_worst` | -19.3% | -8.3% | 15/15 |
| `schedule_adaptive` | -26.8% | -6.5% | 15/15 |

## Candidate Configs

- Global mean: `lambda=20`, `mu=0.07`, `tau=0.05`, `p=0.5`, `rho=0`.
- Global worst: `lambda=14`, `mu=0.03`, `tau=0.03`, `p=0.5`, `rho=0.35`.

## Reading

- The adaptive rule improves the mean result because smooth WSD decays prefer a slower response channel, while WSD-con step schedules need a faster channel to avoid long tail mismatch.
- This is a promising hypothesis but it adds a schedule-dependent branch. It should be presented as an analysis/extension until validated on more schedules.
