# Interpretable Error-Comparison Figures

Residual curves compare MPL, the restored observation-bracket MPL-LD main candidate, the previous fixed-tau MPL-LD reference, and the DCT performance extension.  After the finite-response contraction proved too weak, observation-bracket MPL-LD is restored as the current research-facing model.  All corrected variants fit one nonnegative coefficient from `cosine_72000.csv` only.

## Aggregate

| group | method | mean | worst | wins | non-harm |
|---|---|---:|---:|---:|---:|
| core_wsd | Obs-bracket MPL-LD | -29.87% | -4.67% | 15/15 | 15/15 |
| core_wsd | Old MPL-LD | -27.25% | -3.00% | 15/15 | 15/15 |
| core_wsd | DCT performance | -32.83% | -5.30% | 15/15 | 15/15 |
| extra_control | Obs-bracket MPL-LD | +0.00% | +0.00% | 0/9 | 9/9 |
| extra_control | Old MPL-LD | +0.00% | +0.00% | 0/9 | 9/9 |
| extra_control | DCT performance | +0.00% | +0.00% | 0/9 | 9/9 |
| all | Obs-bracket MPL-LD | -18.67% | +0.00% | 15/24 | 24/24 |
| all | Old MPL-LD | -17.03% | +0.00% | 15/24 | 24/24 |
| all | DCT performance | -20.52% | +0.00% | 15/24 | 24/24 |

## Per Target

| target | observation-bracket mean/worst | old MPL-LD mean/worst | DCT perf mean/worst |
|---|---:|---:|---:|
| WSD sharp | -46.24% / -37.10% | -46.73% / -37.20% | -48.57% / -39.35% |
| WSD linear | -39.73% / -29.80% | -43.79% / -33.80% | -40.32% / -30.83% |
| WSD-con 3e-5 | -41.12% / -35.97% | -28.90% / -23.64% | -53.08% / -41.41% |
| WSD-con 9e-5 | -13.40% / -9.17% | -10.18% / -6.09% | -13.04% / -5.30% |
| WSD-con 18e-5 | -8.88% / -4.67% | -6.65% / -3.00% | -9.11% / -6.43% |
| Cosine 24k | +0.00% / +0.00% | +0.00% / +0.00% | +0.00% / +0.00% |
| Constant 24k | +0.00% / +0.00% | +0.00% / +0.00% | +0.00% / +0.00% |
| Constant 72k | +0.00% / +0.00% | +0.00% / +0.00% | +0.00% / +0.00% |

## Figures

- `25M`: `figs/error_curves_25M.png`
- `100M`: `figs/error_curves_100M.png`
- `400M`: `figs/error_curves_400M.png`
- Target MAE summary: `figs/target_mae_summary.png`

## Reading

- The observation-bracket MPL-LD curve is the current main candidate: it removes the old fixed ridge and response-rate endpoints while improving every WSD-family row.
- The old MPL-LD curve is retained as a reference to show that the newer observation-bracket rule improves the mechanism-native baseline.
- The DCT performance extension is still a useful numerical reference, but its generic low-frequency nuisance basis should not be presented as the core explanation.
