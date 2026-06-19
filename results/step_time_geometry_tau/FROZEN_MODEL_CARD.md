# Frozen Step-Time Error Model Card

This file records the current frozen residual-model decision.  It separates the
deployable rule from diagnostics and audits so later paper or slide edits do
not mix target self-fit evidence with target-holdout evidence.

## Primary Rule

Use the one-kappa geometry-tau rule as the clean transferable model:

```text
L_hat(t) = L_MPL(t) + kappa_hat * phi_tau(t)
```

where `phi_tau` is a finite response to positive LR drops:

```text
drop_u = max(lr_{u-1} - lr_u, 0) / lr_peak
phi_tau(t) = sum_{u <= t} exp(-(t-u)/tau) * drop_u
```

The only loss-fitted parameter is the nonnegative source amplitude:

```text
kappa_hat = max(0, <phi_tau(source), residual_source> / ||phi_tau(source)||^2)
residual_source = L_true(source) - L_MPL(source)
```

The target residual is not used.  The target LR schedule only selects the
route/source and the geometry-derived `tau`.

## Geometry Tau

Safety gates:

```text
if total_positive_drop <= 0.05:
    tau = 0
if positive_drop_span > 16000 and schedule_len <= 30000:
    tau = 0
```

Otherwise:

```text
if positive_drop_span > 100:
    tau = min(8192, 1.25 * positive_drop_span)
else:
    q = clip((total_positive_drop - 0.40) / (0.90 - 0.40), 0, 1)
    tau = 512 * (1 + 2 q^3)
```

Interpretation:

- Long or finite LR decay needs a longer response memory.
- Weak single-step drops should use shorter memory to avoid over-correction.
- The cubic single-step rule is deliberately conservative; the stability audit
  shows that a larger step base or a quadratic exponent can harm the
  no-nuisance head.

## Route Table

| target shape | source calibration | geometry tau | primary nuisance |
|---|---|---:|---|
| long smooth cosine | strongest full-step probe | `8192.0` | none for primary rule |
| finite WSD tail | paired finite-tail schedule | `4998.8` | none |
| full single-step drop | finite-tail WSD schedules | `1536.0` | none for primary rule |
| medium single-step drop | neighboring step probes | `733.2` | none for primary rule |
| weak single-step drop | nearest stronger step probe | `512.0` | none |
| no-drop / short-smooth safety controls | none | `0` | none |

The residualized audit may project out low-frequency nuisance while fitting
`kappa`, but the primary one-kappa rule does not fit or transfer nuisance
coefficients.

## Evidence Summary

| model/audit | role | mean MAE change | worst | non-harm | target residual used? |
|---|---|---:|---:|---:|---|
| geometry-tau one-kappa | primary transferable rule | `-32.3%` | `-1.5%` | `18/18` | no |
| geometry-tau one-kappa self-fit | primary rule, same-curve diagnostic | `-40.7%` | `-6.6%` | `18/18` | yes |
| decomposed self-fit | residual explanation diagnostic | `-70.6%` | `-38.9%` | `18/18` | yes |
| geometry-tau residualized | strong target-holdout audit | `-36.1%` | `-7.0%` | `18/18` | no |
| geometry-tau cross-family residualized | no-same-family audit | `-33.8%` | `-6.5%` | `18/18` | no |
| extended safety controls | no-drop / short-smooth controls | `-21.5%` overall | `+0.0%` | `27/27` | no |

Key residual-plot evidence:

- `error_comparison/core_residuals_100M.png` shows that geometry tau leaves most
  residual shapes almost unchanged and mainly tightens the medium single-step
  correction.
- `error_comparison/mae_bar_summary.png` shows the core one-kappa comparison:
  table tau `-32.0% / -0.4%`, geometry tau `-32.3% / -1.5%`.
- The safety plots remain unchanged: the correction abstains on no-drop and
  short-smooth controls.

## What Not To Claim

- Do not claim prospective validation on unseen schedules.  The current
  evidence is internal target-holdout plus no-same-family audit.
- Do not claim that low-frequency nuisance is the transferable mechanism.  It
  explains self-fit residuals but is not the primary rule.
- Do not promote retrospective route-tau tuning.  It improves the all-data
  target-holdout mean to about `-37.1%`, but leave-one-scale selection falls
  back to about `-35.2%`.
- Do not present the old S-time same-fit curve as deployment evidence; it uses
  target residuals.

## Reproducible Evidence

Run the full consistency gate:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 repro/validate_step_time_model.py
```

Important source artifacts:

- `repro/frozen_step_time_model.py`
- `repro/audit_step_time_geometry_tau.py`
- `repro/plot_geometry_tau_errors.py`
- `repro/audit_step_time_pareto.py`
- `repro/step_time_decomposed_estimator.py`
- `repro/validate_step_time_model.py`

Important result artifacts:

- `results/step_time_geometry_tau/REPORT.md`
- `results/step_time_geometry_tau/error_comparison/REPORT.md`
- `results/step_time_geometry_tau/error_comparison/core_residuals_100M.png`
- `results/step_time_geometry_tau/error_comparison/mae_bar_summary.png`
- `results/step_time_pareto_audit/REPORT.md`
- `results/step_time_decomposed_estimator/REPORT.md`

## Current Decision

Freeze the geometry-tau one-kappa rule as the main transferable model.  Use the
decomposed estimator only to explain why MPL residuals contain a broad
low-frequency component.  Use residualized and cross-family audits as
supporting evidence that the transferable correction is not just same-family
or nuisance leakage.
