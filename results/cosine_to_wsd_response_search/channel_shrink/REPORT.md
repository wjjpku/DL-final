# Channel-Specific Shrinkage Cosine-to-WSD Audit

This audit keeps the adaptive fit-window cosine-only protocol but replaces the shared shrink `rho` with `rho_smooth` and `rho_step`. The amplitudes are still estimated only from `cosine_72000.csv`; WSD losses are used for development ranking and evaluation.

## Formula Change

```text
kappa_channel = [1/(1+rho_channel)] * R_source_channel^p
                * max(0, <M_mu phi_channel, M_mu r_cos>_F
                         / (||M_mu phi_channel||_F^2 + tau^2))
rho_channel = rho_smooth for diffuse LR decay, rho_step for concentrated LR drops
L_hat_target = L_MPL,target + kappa_channel * phi_channel,target
```

Only the uncertainty shrinkage is channel-specific. The response feature, suffix fitting, and target schedule routing are unchanged.

## Recommended Pareto Candidate

This is the main candidate from this audit: it improves the mean while keeping the worst row at least as good as the previous shared-rho fit-window model.

- Mean MAE change: `-35.1%` over `15` scale-target rows.
- Worst scale-target row: `-6.1%`.
- Wins/non-harm: `15/15` and `15/15`.
- Config: `fit_start=3000`, `lambda_smooth=4`, `lambda_step=20`, `mu=0.04`, `max_mode=16`, `tau=0.05`, `p=0.25`, `rho_smooth=0.2`, `rho_step=0.5`.

## Best Mean Candidate

- Mean / worst: `-35.3%` / `-5.5%`.
- Config: `fit_start=12000`, `lambda_smooth=4`, `lambda_step=20`, `mu=0.05`, `max_mode=8`, `tau=0.05`, `p=0.25`, `rho_smooth=0.2`, `rho_step=0.35`.

## Best Worst-Case Candidate

- Mean / worst: `-34.4%` / `-6.3%`.
- Config: `fit_start=3000`, `lambda_smooth=7`, `lambda_step=20`, `mu=0.01`, `max_mode=12`, `tau=0.05`, `p=0`, `rho_smooth=0.35`, `rho_step=0.35`.

## Per-Target Result For Recommended Candidate

| target | mean delta | worst scale | wins |
|---|---:|---:|---:|
| WSD sharp | -52.1% | -40.4% | 3/3 |
| WSD linear | -44.1% | -33.1% | 3/3 |
| WSD-con 3e-5 | -55.5% | -48.2% | 3/3 |
| WSD-con 9e-5 | -14.2% | -7.6% | 3/3 |
| WSD-con 18e-5 | -9.5% | -6.1% | 3/3 |

## Comparison To Previous Main Candidate

Adaptive fit-window shared-rho: mean `-34.53%`, worst `-6.08%`, wins `15/15`.
Recommended channel-specific shrinkage: mean `-35.07%`, worst `-6.12%`, wins `15/15`.
Best-mean channel-specific shrinkage: mean `-35.26%`, worst `-5.53%`.

## Top-Safe Holdout Check

| split | selected config | dev mean | dev worst | test mean | test worst | test wins |
|---|---|---:|---:|---:|---:|---:|
| dev_sharp_linear__test_wsdcon | `start=12000, lambda_s=4, lambda_step=20, mu=0.05, tau=0.05, p=0.25, rho_s=0.2, rho_step=0.35` | -50.3% | -33.1% | -25.2% | -5.5% | 9/9 |
| dev_wsdcon__test_sharp_linear | `start=3000, lambda_s=7, lambda_step=20, mu=0.015, tau=0.05, p=0, rho_s=0.5, rho_step=0.75` | -26.8% | -6.3% | -45.9% | -33.9% | 6/6 |
| leave_target__wsd_20000_24000.csv | `start=5000, lambda_s=7, lambda_step=20, mu=0.02, tau=0.05, p=0, rho_s=0.6, rho_step=1` | -31.0% | -5.9% | -49.6% | -32.2% | 3/3 |
| leave_target__wsdcon_18.csv | `start=12000, lambda_s=4, lambda_step=20, mu=0.05, tau=0.05, p=0.25, rho_s=0.2, rho_step=0.35` | -41.8% | -5.5% | -8.9% | -6.1% | 3/3 |
| leave_target__wsdcon_3.csv | `start=12000, lambda_s=4, lambda_step=20, mu=0.05, tau=0.05, p=0.25, rho_s=0.2, rho_step=0.6` | -31.1% | -5.3% | -49.4% | -46.3% | 3/3 |
| leave_target__wsdcon_9.csv | `start=12000, lambda_s=4, lambda_step=20, mu=0.05, tau=0.05, p=0.25, rho_s=0.2, rho_step=0.35` | -40.7% | -6.1% | -13.5% | -5.5% | 3/3 |
| leave_target__wsdld_20000_24000.csv | `start=3000, lambda_s=4, lambda_step=20, mu=0.04, tau=0.05, p=0.25, rho_s=0.2, rho_step=0.5` | -32.8% | -6.1% | -44.1% | -33.1% | 3/3 |
| leave_scale__25M | `start=12000, lambda_s=4, lambda_step=20, mu=0.05, tau=0.05, p=0.25, rho_s=0.2, rho_step=0.5` | -36.6% | -5.5% | -32.5% | -11.5% | 5/5 |
| leave_scale__100M | `start=12000, lambda_s=4, lambda_step=20, mu=0.05, tau=0.05, p=0.25, rho_s=0.2, rho_step=0.35` | -36.4% | -6.1% | -33.0% | -5.5% | 5/5 |
| leave_scale__400M | `start=12000, lambda_s=4, lambda_step=20, mu=0.05, tau=0.05, p=0.25, rho_s=0.2, rho_step=0.5` | -33.8% | -8.2% | -38.0% | -5.5% | 5/5 |

## Reading

- The recommended candidate is small but clean: both mean and worst-cell MAE improve over the shared-rho fit-window model.
- The best-mean candidate lowers average MAE further, but it slightly weakens the worst WSD-con 9e-5 cell; it is better treated as an optimistic development point.
- The added degree of freedom has a direct interpretation as channel-specific transfer uncertainty, not an arbitrary residual basis.
- This remains a development audit because the channel shrink values are selected by WSD-family ranking. For a stricter final protocol, freeze the channel-shrink grid choice before testing new schedules.
