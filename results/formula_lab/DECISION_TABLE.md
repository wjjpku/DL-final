# Formula-upgrade decision table (public MPL curves, frozen backbone)

All numbers MAE change vs frozen-MPL baseline (negative better), 6/6 = wins.
Baselines from the paper / engineering map are reproduced exactly by the lab
(`repro/formula_lab/`).

## Protocols

- **R2**: in-sample pooled origin-LS R^2 on sharp residuals (wsd+wsdld), per scale.
- **LOS**: leave-one-sharp; kappa from the other sharp curve, same scale (paper -49.0%).
- **T1lin**: Table-1 zero-fit cross-scale chain, probe-linear amplitude (paper -44.0%).
- **T1-B**: same chain, amplitude from MPL's own B (zero-probe; NEW).
- **probes**: kappa from 3 wsdcon probes pooled -> sharp (paper law -17.1%).
- **dilut**: kappa from [other sharp + 3 probes] -> held-out sharp (paper law -19.0%).
- **Matrix**: 6x6 transfer matrix, final_no_cap estimator (paper worst -2.72 / mean -12.08 / cos->wsd -4.27).

## Sweep (lam = 10 everywhere)

| variant | R2(100/400) | LOS | T1lin | T1-B | probes | dilut | M.worst | M.mean | M.cos->wsd | maxCosK |
|---|---|---|---|---|---|---|---|---|---|---|
| lr@10 [paper] | .804/.874 | -49.0 | -44.0 | -43.1 | -17.1 | -19.0 | -2.72 | -12.08 | -4.27 | .0089 |
| pow d=0.25 | .785/.851 | -46.3 | -42.5 | -41.5 | -23.0 | -25.8 | **-2.85** | **-13.51** | -8.44 | .0203 |
| pow d=0.5 | .744/.816 | -42.4 | -40.0 | -39.2 | -28.6 | -31.6 | +0.02 | -13.14 | **-12.31** | .0327 |
| pow d=per-scale (p-1 from floors: .06/.49/.25) | n/a | ~-44 | -41.1 | n/a | -23.3 | n/a | -1.97 | -12.98 | -7.37 | .0246 |
| affine r=0.5 | .774/.842 | -45.2 | -41.7 | -40.7 | -22.3 | -25.0 | -2.83 | -13.08 | -4.72 | .0123 |
| affine r=0.65 | .750/.821 | -42.8 | -40.2 | -39.4 | -25.3 | -28.1 | -2.67 | -12.91 | -4.81 | .0135 |
| affine r=0.75 | .725/.801 | -40.9 | -38.9 | -38.2 | -27.7 | -30.5 | -2.55 | -12.49 | -4.73 | .0138 |
| affine r=0.9 | .671/.758 | -37.5 | -36.3 | -35.8 | -31.1 | -33.3 | +13.59 FAIL | -10.58 | -2.38 | .0074 |

Probes->sharp (delta, lam) grid optimum: delta 0.75-1.0 @ lam 5-7: -34.5% (vs -17.1).
Kernel shape (two-exp / Lomax / stretched): collapses to one-pole with and
without the weight -- dead end, retested after the weight fix (probe_grid.py).

## Supporting measurements

- Floor power law `F_eq ~ eta^p` from settled wsdcon floors: p = 1.06 / 1.49 / 1.25
  (25/100/400M); real ~10M transformer floors confounded by backbone (S differs).
- Amplitude-from-B: B matches independently measured dL_eq/deta within 5-15%
  (363.8/381.7, 437.9/499.3, 523.4/561.5); kappa_fit/(eta_peak*B) CV 14.7% vs
  probe chain 13.9% -> zero-probe amplitude chain viable.
- Per-curve kappa across families (lr@10): probe/sharp = 0.24-0.44, cosine 5-7x
  too high; pow d=0.5 lifts probe/sharp to 0.36-0.71.
- Early-window probe kappa (tail-contamination check): minor effect; the
  deficit is real amplitude. pow d=0.5 fixes the deepest drop (wsdcon_3 ->
  0.80-0.96 of sharp kappa) but mid-eta probes keep a ~3-4x uniform deficit
  -> concentration-dependent spectral compression (fast-mode absorption),
  documented as remaining structure.
- lam does NOT unify across families (probes ~15-19 vs sharp ~0.5-5 fitted);
  amplitude weight and memory are entangled on sharp curves; lam=10 stays the
  best compromise (independently measured).

## Real-model out-of-family results (added after GPU runs)

10.7M transformer, probe-only calibration (kappa+lam from suite wsdcon probes,
lam* ~ 1 = measured flat tau ~ 850 steps). MAE change vs MPL backbone fit on
[constant, cosine, wsdcon_20]:

| target | d=0 | d=0.5 | d=0.75 | affine r=0.5 |
|---|---|---|---|---|
| sharp600 | -18.1 | -61.3 | -71.2 | -28.5 |
| wsd | -7.6 | -26.5 | -36.1 | -12.3 |
| wsdld | -6.2 | -22.2 | -32.7 | -10.1 |
| twodrop | -11.2 | -13.5 | -8.8 | -14.4 |
| invsqrt | +6.1 | +25.1 | +40.1 | +10.5 |
| cyclic (excl.: MPL re-warm backbone failure +0.07) | -1.8 | -2.0 | -1.5 | -1.9 |

Showcase bed (peak 2.5e-3): wsd_sharp -13.8/-24.5/-38.0/-52.4 and wsd_grad
-6.3/-11.0/-16.1/-20.2 for d=0/0.25/0.5/0.75.

twodrop G1 (deposit ratio A2/A1; predictions d=0: 0.700, p=1.25: 0.564,
d=0.5: 0.383, p=2: 0.303): UNMEASURABLE at 10.7M.
- kernel decomposition through MPL residual: 0.000 [0, 0.131] -- but K1,K2
  nearly collinear at lam*~1 (identifiability artifact);
- seed-paired difference (twodrop - onedrop, bitwise-matched trunks):
  specification-dependent 0.86-2.06 (window x tau-fixed grid), r2 <= 0.3,
  eval-noise/smoothing dominated.
Needs seed averaging or larger model; reported honestly, no estimator selected.
Gate portability: smooth_id_weight does NOT switch off invsqrt (w_id=0.90);
NextGen retention floor 0.01 abstains on invsqrt AND sharp600 here ->
thresholds are suite-specific.

## Equal-S floor ladder (10.7M, curves_floor/, confound-free)

- G4a: floors strictly MONOTONE in eta2 (1.0468/1.0544/1.0678/1.0863/1.1081/
  1.1580 for eta2 = 0.5/1/2/4/8/15 e-4) -- the old U-shape was the backbone
  confound, removed by the equal-S design.
- G4b: floor = 1.0385 + 0.1178*(eta/peak)^0.731, p_real = 0.731
  (90% CI [0.615, 0.855]) -> SUBLINEAR at 10.7M.
- Scale trend of the floor exponent: 0.73 (10.7M) -> 1.06 (25M) -> 1.49 (100M)
  -> 1.25 (400M): superlinearity (the edge population) emerges with scale,
  consistent with tau ~ 1/eta also being absent at 10M (ladder remeasures
  tau ~ eta^-0.18 with r2 >= 0.98 per rung).
- Consequence: delta>0 transfer gains at 10M are attributable to the
  concentration-dependent visible amplitude, NOT floor curvature; universality
  of delta across scales is excluded (use the measured per-scale route).
