# Staged Joint MPL-Lag Audit

This is the first, fast joint-fitting audit.  It does not freeze MPL predictions.  For each held-out target, both the baseline and the lag model refit on the same remaining curves.

## Fixed-Shape Formula

MPL fixed-shape baseline:

```text
L_base(t) = L0 + A S(t)^(-alpha) + B LD_fixed(t; C0,beta0,gamma0)
```

Joint lag model:

```text
L_lag(t) = L0 + A S(t)^(-alpha) + B LD_fixed(t; C0,beta0,gamma0) + K Lag_tau(t)
Lag_tau(t) = sum_{u <= t} max(lr_{u-1}-lr_u,0)/lr_peak * exp(-(t-u)/tau(schedule))
```

`C0,beta0,gamma0` are the scale-specific published MPL values.  `L0,A,alpha,B` and `K` are fitted jointly in one log-Huber objective.

## Leave-One-Curve-Out Result

- Overall: mean `-10.1%`, worst `+69.4%`, non-harm `12/18`.

| held-out target | mean delta | worst | non-harm |
|---|---:|---:|---:|
| Cosine | +38.9% | +69.4% | 0/3 |
| WSD sharp | -8.3% | +7.7% | 1/3 |
| WSD linear | -3.2% | +43.6% | 2/3 |
| WSD-con 3e-5 | -22.8% | -6.4% | 3/3 |
| WSD-con 9e-5 | -39.9% | -38.3% | 3/3 |
| WSD-con 18e-5 | -25.6% | -5.4% | 3/3 |

## Interpretation

- This directly addresses the two-stage objection at a first-order level: the lag amplitude competes with the MPL backbone during fitting.
- Because `C,beta,gamma` are still fixed, this is not yet the final full-MPL joint fit.  It is a fast diagnostic for whether the lag term remains useful before paying for the full eight-parameter optimization.
