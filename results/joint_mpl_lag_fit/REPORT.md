# Joint MPL-Lag Fit Audit

This audit modifies MPL's LR-drop term and optimizes the new lag amplitude in the same fitting objective as the original MPL parameters.

## Formula

Original MPL:

```text
L(t) = L0 + A S(t)^(-alpha) + B * sum_k delta_eta_k * G(C eta_k^(-gamma) (S(t)-S(k)))
G(x) = 1 - (1 + x)^(-beta)
```

Joint lag variant:

```text
L(t) = L0 + A S(t)^(-alpha) + B * LD_ad(t; C,beta,gamma) + K * Lag_tau(t)
Lag_tau(t) = sum_{u <= t} max(lr_{u-1} - lr_u, 0) * exp(-(t-u)/tau(schedule))
```

The fitted parameters are `L0,A,alpha,B,C,beta,gamma,K`.  `K` is optimized jointly with MPL, not fitted from frozen-MPL residuals.  `tau` is computed from schedule geometry.

## Leave-One-Curve-Out Result

- Overall: mean `-7.5%`, worst `+132.3%`, non-harm `14/18`.

| held-out target | mean delta vs jointly refit MPL | worst | non-harm |
|---|---:|---:|---:|
| Cosine | -31.3% | -7.9% | 3/3 |
| WSD sharp | -15.2% | +17.6% | 2/3 |
| WSD linear | -27.4% | -3.2% | 3/3 |
| WSD-con 3e-5 | +82.1% | +132.3% | 0/3 |
| WSD-con 9e-5 | -26.3% | -22.4% | 3/3 |
| WSD-con 18e-5 | -27.2% | -12.3% | 3/3 |

## Reading

- This is the right fitting protocol for testing whether the lag term is still useful after MPL can re-optimize around it.
- The comparison baseline is not frozen MPL; it is MPL refit on the same train curves for each held-out target.
- A weak or unstable result here means the post-hoc residual story is confounded by MPL fitting error and should not be the main claim without this joint objective.
