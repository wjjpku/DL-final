# Frozen Geometry-Tau One-Kappa Model

This is the executable implementation of the frozen primary transferable rule.  It fits only `kappa` from source residuals; target residuals are excluded in target-holdout mode.

## Formula

```text
L_hat(t) = L_MPL(t) + kappa_hat * phi_tau(t)
phi_tau(t) = sum_{u <= t} exp(-(t-u)/tau) * max(lr_{u-1} - lr_u, 0) / lr_peak
```

Geometry tau:

```text
if positive_drop_span > 100:
    tau = min(8192, 1.25 * positive_drop_span)
else:
    q = clip((total_positive_drop - 0.40) / (0.90 - 0.40), 0, 1)
    tau = 512 * (1 + 2 q^3)
```

Safety gates set `tau=0` for zero-positive-drop and short-smooth controls.

## Metrics

- Target-holdout primary rule: mean `-32.3%`, worst `-1.5%`, non-harm `18/18`.
- Same-curve one-kappa diagnostic: mean `-40.7%`, worst `-6.6%`, non-harm `18/18`.
- Extended safety audit: mean `-21.5%`, worst `+0.0%`, non-harm `27/27`.
- Safety controls only: mean `+0.0%`, worst `+0.0%`, non-harm `9/9`.

## Route Table

| target | route | source | geometry tau | table tau | target residual used for kappa? |
|---|---|---|---:|---:|---:|
| Cosine | smooth_decay | `wsdcon_3` | 8192.0 | 8192.0 | 0 |
| WSD sharp | finite_tail | `wsdld_20000_24000` | 4998.8 | 5120.0 | 0 |
| WSD linear | finite_tail | `wsd_20000_24000` | 4998.8 | 5120.0 | 0 |
| WSD-con 3e-5 | full_step_drop | `wsd_20000_24000+wsdld_20000_24000` | 1536.0 | 1536.0 | 0 |
| WSD-con 9e-5 | medium_step_drop | `wsdcon_3+wsdcon_18` | 733.2 | 768.0 | 0 |
| WSD-con 18e-5 | weak_step_drop | `wsdcon_9` | 512.0 | 512.0 | 0 |

## Decision

- Use this module as the source of truth for the frozen primary transferable rule.
- Use decomposed self-fit only as a residual explanation diagnostic, not as the deployment rule.
- Use residualized and cross-family results as audits around this primary rule.
