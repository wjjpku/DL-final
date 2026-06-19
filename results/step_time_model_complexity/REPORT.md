# Step-Time Model Complexity Audit

This report answers which additions are necessary.  It treats fitted loss-dependent parameters separately from schedule-only routing choices.

## Main Takeaway

- The core transferable model does not need an interpretable sinusoidal component.  The one-kappa route, with nuisance projection disabled, still gives strong target-holdout generalization.
- Minimal shape-routed transfer (`L_MPL + kappa phi_tau`, no nuisance): mean `-32.0%`, worst `-0.4%`, non-harm `18/18`.
- Residualized shape-routed transfer: mean `-36.1%`, worst `-7.0%`, non-harm `18/18`.
- Conservative cross-family transfer: mean `-32.7%`, worst `-6.5%`, non-harm `18/18`.
- Strict one-kappa/no-nuisance/no-same-family transfer remains useful but more conservative: mean `-24.6%`, worst `-3.5%`, non-harm `18/18`.

## Variant Table

| variant | role | mean | worst | non-harm | wins | reading |
|---|---|---:|---:|---:|---:|---|
| shape_routed_no_nuisance | minimal target-holdout | -32.0% | -0.4% | 18/18 | 18/18 | one fitted kappa; no nuisance coefficients used |
| shape_routed_residualized | strong target-holdout | -36.1% | -7.0% | 18/18 | 18/18 | adds projection to reduce smooth-drift contamination |
| self_fit_no_nuisance | minimal self-fit | -40.7% | -6.6% | 18/18 | 18/18 | same target curve fits only kappa against phi_tau |
| decomposed_self_fit_reference | diagnostic self-fit | -70.6% | -38.9% | 18/18 | 18/18 | uses fitted low-frequency nuisance for same-curve explanation |
| cross_family_residualized | no-same-family audit | -32.7% | -6.5% | 18/18 | 18/18 | forbids calibration from the target schedule family |
| cross_family_no_nuisance_p3 | strict audit | -24.6% | -3.5% | 18/18 | 18/18 | no same-family source, no nuisance projection, stronger drop attenuation |

## Parameter Ledger

| component | fitted from loss? | count per calibration fit | should be headline? | explanation |
|---|---|---:|---|---|
| kappa | yes | 1 | yes | the only transferable amplitude fitted from calibration residuals |
| low-frequency nuisance coefficients | yes | 0, 3, or 5 depending on audit | no | residualization/self-fit diagnostic only; not a physical sinusoidal mechanism |
| tau / route / safety gate | no | 0 | no | schedule-only model-selection rule; ablated separately because it can overfit benchmarks |
| drop attenuation | no | 0 | no | schedule-only conservative shrinkage for weaker single-step targets |

## Route Complexity

| variant | route classes | nonzero tau values | nuisance choices | source sets |
|---|---:|---:|---:|---:|
| shape_routed_no_nuisance | 5 | 5 | 1 | 6 |
| self_fit_no_nuisance | 5 | 5 | 1 | 6 |
| cross_family_residualized | 5 | 5 | 3 | 3 |
| cross_family_no_nuisance_p3 | 5 | 5 | 1 | 3 |

## Interpretation

- Self-fit improves from `-40.7%` with one-kappa/no-nuisance to `-70.6%` with the diagnostic low-frequency component.  That makes the nuisance useful for explanation, but not the main transferable mechanism.
- For generalization, the cleanest primary claim is the minimal route: one fitted amplitude kappa, schedule-only tau/route choices, and no transferred nuisance coefficient.
- The residualized and cross-family versions are best presented as audits: residualization checks that smooth MPL drift does not contaminate kappa; cross-family checks that the result is not just same-family calibration.
- The schedule-only route and tau choices are still model-selection freedom, not learned parameters.  Their necessity is supported by ablations, but external frozen-rule validation is still the strongest missing evidence.
