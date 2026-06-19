# Schedule-Adaptive Cosine-to-WSD Search

This search keeps the original assignment protocol: `kappa` is fitted from `cosine_72000.csv` residuals only. WSD-family losses are used only to rank development candidates in this audit. The target schedule contributes only LR-derived features and a schedule-shape channel choice.

## Searched Formula

```text
drop_concentration = max_t relu(lr_{t-1}-lr_t) / sum_t relu(lr_{t-1}-lr_t)
channel = step if drop_concentration >= 0.2 else smooth
phi_channel(t) = sum_{u <= t} exp(-lambda_channel (S_t-S_u)) * relu(lr_{u-1}-lr_u)/lr_peak
r = L_true_cosine - L_MPL_cosine
kappa_channel = [1/(1+rho)] * R_source_channel^p * max(0, <M_mu phi_channel, M_mu r> / (||M_mu phi_channel||^2 + tau^2))
L_hat_target = L_MPL_target + kappa_channel(target) * phi_channel,target
```

## Best Fully Non-Harming Candidate

- Mean MAE change: `-31.3%` over `15` scale-target rows.
- Worst scale-target row: `-6.1%`.
- Wins/non-harm: `15/15` and `15/15`.
- Config: `lambda_smooth=4`, `lambda_step=20`, `mu=0.04`, `max_mode=8`, `tau=0.05`, `p=0.25`, `rho=0.5`.

## Best Worst-Case Candidate

- Mean / worst: `-29.7%` / `-6.3%`.
- Config: `lambda_smooth=4`, `lambda_step=20`, `mu=0.02`, `max_mode=16`, `tau=0.05`, `p=0.25`, `rho=0`.

## Per-Target Result For Best Mean Candidate

| target | mean delta | worst scale | wins |
|---|---:|---:|---:|
| WSD sharp | -43.9% | -25.2% | 3/3 |
| WSD linear | -35.8% | -19.0% | 3/3 |
| WSD-con 3e-5 | -53.5% | -46.6% | 3/3 |
| WSD-con 9e-5 | -14.1% | -7.0% | 3/3 |
| WSD-con 18e-5 | -9.0% | -6.1% | 3/3 |

## Comparison

Previous old nextgen: mean `-17.2%`, worst `-2.2%`, wins `15/15`.
Previous global response search: mean `-22.0%`, worst `-6.5%`, wins `15/15`.
Manual schedule-adaptive audit: mean `-26.8%`, worst `-6.5%`, wins `15/15`.

## Interpretation

- Smooth WSD decays and WSD-con step decays have different identifiable response time scales. A single global response rate is therefore conservative.
- The adaptive branch is schedule-only: it separates diffuse decay from concentrated LR drops using `drop_concentration`, not target loss.
- Because the best hyperparameters are selected on the WSD family, this remains a development result; it should be checked with held-out WSD types or additional schedules before becoming the final paper/slides model.
