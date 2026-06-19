# Mid-Ratio Route Audit

This audit keeps the current low-tail route and switches only WSD-con schedules with `final_lr / peak_lr = 0.3` to a mid-tail recovery candidate.  The mid-tail coefficient is still fitted from the cosine residual only; WSD-family losses are used here only for development ranking.

## Best Fully Non-Harming Route

- Mean / worst: `-37.78%` / `-11.80%`.
- Wins/non-harm: `15/15` and `15/15`.
- Mid config: `23` with center `0.2`, width `0.05`, time_power `1`, tau `0.001`, sign `signed`, shrink `0`.

## Best Worst-Case Route

- Mean / worst: `-37.75%` / `-11.81%`.

## Comparison

- Joint-channel LR-curvature: mean `-37.53%`, worst `-10.80%`.
- Low-tail route base: mean `-37.67%`, worst `-10.80%`.
- Mid-ratio route: mean `-37.78%`, worst `-11.80%`.
- Base route recomputed in this script: mean `-37.67%`, worst `-10.80%`.

## Per-Target Result

| target | mean delta | worst scale | wins |
|---|---:|---:|---:|
| WSD sharp | -54.3% | -40.5% | 3/3 |
| WSD linear | -46.3% | -33.1% | 3/3 |
| WSD-con 3e-5 | -57.8% | -47.7% | 3/3 |
| WSD-con 9e-5 | -17.6% | -11.8% | 3/3 |
| WSD-con 18e-5 | -13.0% | -12.1% | 3/3 |

## Routed WSD-con 9e-5 Rows

| scale | delta | corr_mae | base_mae |
|---|---:|---:|---:|
| 25M | -22.2% | 0.00227633 | 0.00292648 |
| 100M | -11.8% | 0.00435231 | 0.00493462 |
| 400M | -18.7% | 0.00543904 | 0.00668731 |

## Reading

- This is a schedule-ratio route, not a per-scale route: all three scales of WSD-con 9e-5 use the same mid-tail configuration.
- A useful improvement here would show that the remaining bottleneck is tied to moderate tail LR rather than to the low-tail case already handled by the previous route.
