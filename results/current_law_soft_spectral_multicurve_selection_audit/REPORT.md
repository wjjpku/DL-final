# Soft Spectral Multi-Curve Lambda-Selection Audit

This audit chooses the soft DCT/Sobolev nuisance strength using only the calibration curves. For train subsets with at least two curves, `inner_cv_*` rules select lambda by leave-one-curve-out transfer inside the train set, then evaluate only on held-out curves. The fixed-lambda row is included as the soft-prior Pareto reference.

## Train-Size Summary

| rule | train curves | median worst heldout | best worst heldout | worst worst heldout | mean heldout | median lambda |
|---|---:|---:|---:|---:|---:|---:|
| `fixed_lam0p025` | 1 | -0.7% | -2.3% | +13.2% | -12.4% | 0.025 |
| `fixed_lam0p025` | 2 | -1.8% | -6.6% | +3.9% | -13.0% | 0.025 |
| `fixed_lam0p025` | 3 | -2.3% | -8.6% | +1.7% | -13.1% | 0.025 |
| `fixed_lam0p025` | 4 | -5.6% | -11.8% | -1.7% | -13.1% | 0.025 |
| `fixed_lam0p025` | 5 | -7.2% | -21.9% | -1.7% | -13.1% | 0.025 |
| `inner_cv_mean` | 1 | -0.7% | -2.3% | +13.2% | -12.4% | 0.025 |
| `inner_cv_mean` | 2 | -1.9% | -6.2% | +14.1% | -11.2% | 0.01 |
| `inner_cv_mean` | 3 | -2.1% | -9.3% | +14.2% | -12.4% | 0.03 |
| `inner_cv_mean` | 4 | -5.4% | -12.1% | +4.4% | -13.5% | 0.04 |
| `inner_cv_mean` | 5 | -8.4% | -21.9% | -2.2% | -14.5% | 0.1 |
| `inner_cv_worst` | 1 | -0.7% | -2.3% | +13.2% | -12.4% | 0.025 |
| `inner_cv_worst` | 2 | -1.6% | -6.2% | +19.5% | -10.3% | 0.015 |
| `inner_cv_worst` | 3 | -2.1% | -11.0% | +14.2% | -12.3% | 0.02 |
| `inner_cv_worst` | 4 | -4.7% | -13.3% | -1.3% | -13.8% | 0.05 |
| `inner_cv_worst` | 5 | -8.4% | -22.0% | -1.4% | -14.4% | 0.1 |
| `inner_cv_safe` | 1 | -0.7% | -2.3% | +13.2% | -12.4% | 0.025 |
| `inner_cv_safe` | 2 | -1.6% | -6.2% | +5.9% | -11.9% | 0.015 |
| `inner_cv_safe` | 3 | -2.1% | -11.0% | +5.7% | -12.7% | 0.02 |
| `inner_cv_safe` | 4 | -4.7% | -13.3% | -1.3% | -13.8% | 0.05 |
| `inner_cv_safe` | 5 | -8.4% | -22.0% | -1.4% | -14.4% | 0.1 |
| `inner_cv_band_mean` | 1 | -0.7% | -2.3% | +13.2% | -12.4% | 0.025 |
| `inner_cv_band_mean` | 2 | -1.6% | -6.2% | +5.6% | -12.2% | 0.01 |
| `inner_cv_band_mean` | 3 | -2.2% | -8.9% | +3.3% | -12.8% | 0.03 |
| `inner_cv_band_mean` | 4 | -5.5% | -12.1% | -1.7% | -13.3% | 0.03 |
| `inner_cv_band_mean` | 5 | -7.4% | -22.0% | -1.8% | -13.3% | 0.03 |
| `inner_cv_band_worst` | 1 | -0.7% | -2.3% | +13.2% | -12.4% | 0.025 |
| `inner_cv_band_worst` | 2 | -1.6% | -6.2% | +5.6% | -12.2% | 0.015 |
| `inner_cv_band_worst` | 3 | -1.9% | -8.9% | +3.3% | -12.6% | 0.02 |
| `inner_cv_band_worst` | 4 | -5.5% | -12.1% | -1.4% | -13.0% | 0.03 |
| `inner_cv_band_worst` | 5 | -7.4% | -22.0% | -1.4% | -13.3% | 0.03 |

## Readout

A successful automatic rule should approach the fixed soft-prior Pareto reference while using only train curves. If inner-CV chooses overly large lambda values, it is overfitting the calibration-transfer matrix; if it chooses tiny lambda values, it collapses back to hard projection and loses amplitude.

The fixed soft prior becomes stable when calibration coverage is broad enough: with four train curves it has median worst held-out `-5.6%` and worst worst-heldout `-1.7%`; with five train curves it has median worst held-out `-7.2%` and worst worst-heldout `-1.7%`. Thus multi-curve coverage can make the soft-prior candidate non-failing.

The inner-CV rules are not yet reliable automatic selectors. With three train curves, `inner_cv_mean` still has worst worst-heldout `+14.2%`, and even `inner_cv_safe` has `+5.7%`. Only at five train curves does `inner_cv_safe` become non-failing (`-1.4%`).

Restricting inner-CV to the empirically identifiable soft-prior band `0.01 <= lambda <= 0.03` is a useful correction but not a complete solution. At three train curves, `inner_cv_band_mean` improves worst worst-heldout to `+3.3%`; at four and five train curves it is non-failing (`-1.7%` and `-1.8%`). The practical conclusion is that soft spectral kappa is a promising multi-curve candidate, and band-limited calibration is the best automatic selector tested here, but a universally reliable small-train lambda selector remains unresolved.
