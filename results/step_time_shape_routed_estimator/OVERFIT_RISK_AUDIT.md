# Step-Time Overfit-Risk Audit

Yes, the current estimator can still overfit the public-curve benchmark.  The protocol audit rules out target-residual leakage, but it does not prove that the route thresholds, tau values, and safety gates were selected prospectively.  This file separates what the evidence supports from what remains a risk.

## What Reduces The Concern

- Target-loss blindness: `27/27` protocol rows pass; scrambling a target residual changes max `kappa` by `0.000e+00` and max correction by `0.000e+00`.
- Target exclusion: `27/27` predictions exclude the target curve from calibration.
- Final core target-holdout: mean `-36.1%`, worst `-7.0%`, non-harm `18/18`.
- Removing nuisance projection still stays non-harming: mean `-32.0%`, worst `-0.4%`.
- Local tau perturbations stay non-harming: `0.75x` worst `-7.7%`, `1.25x` worst `-3.2%`.

## Scale Slices

| slice | mean | worst | non-harm |
|---|---:|---:|---:|
| scale_25 | -32.4% | -16.3% | 6/6 |
| scale_100 | -36.2% | -8.8% | 6/6 |
| scale_400 | -39.8% | -7.0% | 6/6 |

## Route Slices

| slice | mean | worst | non-harm |
|---|---:|---:|---:|
| route_finite_tail | -50.5% | -33.9% | 6/6 |
| route_full_step_drop | -53.1% | -47.3% | 3/3 |
| route_medium_step_drop | -15.9% | -8.8% | 3/3 |
| route_smooth_decay | -35.0% | -23.7% | 3/3 |
| route_weak_step_drop | -11.7% | -7.0% | 3/3 |

## What Still Looks Like Overfitting Risk

- Model-selection complexity is high for the available benchmark: `5` route classes, `5` nonzero tau values, and `3` nuisance choices for only `18` core target-scale cells.
- Wide tau perturbations are not safe: `1.5x` gives `16/18` non-harming cells and `2x` gives `14/18`.
- The short-smooth safety gate is necessary: removing it gives worst `+154.0%`.
- `4/6` target routes use a same-family source.  This is target-holdout, not leave-family validation.
- There is no new external schedule family in the current repository that was untouched during model design.

## Recommended Claim

Use the shape-routed estimator as the strongest current internal evidence and as a candidate deployment rule.  Do not claim that it is fully validated on unseen regimes.  The next decisive experiment is to freeze the route rule and evaluate it on a new schedule family or a new training run that was not used while designing the residual model.
