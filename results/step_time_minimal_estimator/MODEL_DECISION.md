# Minimal Step-Time Model Decision

This note records the current model decision after the residual-image audits.
It is intentionally separate from the paper and slides.

## Recommended Headline Model

```text
r(t) = L_true(t) - L_MPL(t)

phi_tau(t) =
  sum_{u<=t} exp(-(t-u)/tau)
  * relu(eta_{u-1}-eta_u) / eta_peak

kappa_hat = max(0, <phi_tau, r_source> / ||phi_tau||^2)

L_hat_target(t) =
  L_MPL,target(t) + kappa_hat * phi_tau,target(t)
```

Only `kappa` is fitted from calibration loss residuals.

## What Changed From The Earlier Model

The earlier correction let a global schedule-response feature absorb both local
cooldown lag and smooth cosine-shaped MPL drift.  The new model restricts the
transferable term to positive LR drops with finite step-time memory.

```text
old risk:       cosine low-frequency drift -> kappa -> WSD transfer
new principle: only LR-drop transient      -> kappa -> target schedule
```

The route and tau are chosen from the LR schedule only.  Target residuals are
not used when choosing the source curve, tau, or amplitude.

## What Is Not Headline

- Low-frequency `G_low` / DCT / sinusoidal coefficients are not a physical
  mechanism and are not needed for the main generalization result.
- They remain useful as a diagnostic/self-fit residualizer, not as the primary
  transferable model.
- Cross-family and overfit-risk audits are supporting evidence, not extra
  model terms.

## Current Evidence

| model/audit | mean MAE change | worst | non-harm | reading |
|---|---:|---:|---:|---|
| minimal target-holdout | `-32.0%` | `-0.4%` | `18/18` | clean one-kappa headline candidate |
| residualized target-holdout | `-36.1%` | `-7.0%` | `18/18` | stronger but uses nuisance projection |
| minimal self-fit | `-40.7%` | `-6.6%` | `18/18` | one-kappa same-curve diagnostic |
| decomposed self-fit | `-70.6%` | `-38.9%` | `18/18` | fitted nuisance explains smooth drift |
| conservative cross-family | `-32.7%` | `-6.5%` | `18/18` | no same-family source calibration |
| strict no-nuisance cross-family | `-24.6%` | `-3.5%` | `18/18` | no same-family source and no nuisance projection |

## Old-vs-Minimal Error Plot Reading

`MPL+old` is a target-residual same-fit diagnostic.  `MPL+minimal` is the
target-holdout deployment rule.

| group | old same-fit mean | old worst | minimal holdout mean | minimal worst |
|---|---:|---:|---:|---:|
| core schedules | `-32.3%` | `-5.6%` | `-32.0%` | `-0.4%` |
| extended controls | `-21.5%` | `+0.0%` | `-21.4%` | `+0.0%` |
| safety controls | `+0.0%` | `+0.0%` | `+0.0%` | `+0.0%` |

The important comparison is not that minimal beats old same-fit.  The important
comparison is that minimal reaches essentially the same average correction
without reading target residuals and without fitting a nuisance component.

## Necessary Safety Rules

| ablation | result | interpretation |
|---|---|---|
| fixed `tau=1024` | worst `+6.8%` | one universal response time is unsafe |
| no short-smooth gate | worst `+67.1%` | short smooth cosine should not receive the transient correction |
| no cross-family weak-step attenuation | worst `+75.3%` | weaker single-step targets need schedule-only amplitude shrinkage |

## Remaining Risk

The current evidence is still an internal public-curve audit.  The route and
tau rules are schedule-only, but they were designed after inspecting these
curves.  The next decisive validation is to freeze this minimal rule and test it
on a new schedule family or a new training run.
