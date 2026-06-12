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

## Structural-overhaul round (post-PR adversarial extension)

Direction A -- nonlinear-relaxation ODE law
  r' = -lam0(1+r/r*) r (in S-time) + deposits; linear limit r*->inf = paper law.
  - Joint family fits (probes+sharp pooled, one parameter set): nonlinear wins
    at every scale; sharp R2 0.002->0.266 (25M), 0.083->0.349 (100M),
    0.520->0.652 (400M) at modest probe cost -> the mechanism (faster
    relaxation at large excursion) is REAL and reconciles the family mismatch.
  - BUT r* is unidentifiable within a single family (probes-only fit sends
    r*->1e12, reduces to linear): deployment via probes-only fails (-15.3%).
  - LOO cross-scale shape transfer: -23.1% (6/6) -- beats linear (-17.1%) but
    LOSES to the simpler shipped delta=0.5 closure (-28.6%) while needing two
    extra transferred constants.
  VERDICT: closed as a prediction formula (dominated); retained as the
  mechanistic explanation of concentration-dependent c.

Direction B -- annexation (joint 9-param fit of MPL+lag on official split
  [cosine_24000, constant_24000, wsdcon_9]; held-out below):
  - sharp held-out (wsd+wsdld mean over scales): A0 frozen 0.00368,
    A1 frozen+patch 0.00346 (-6.1%), A2 joint 0.00172 (-53.3%).
  - BUT cosine_72000 collapses under A2: 0.00745->0.01379 / 0.00727->0.01242 /
    0.00561->0.01665 (+85% to +197%); wsdcon_3 degrades at 25/100M.
  - Fit pathology: gamma warped 0.64->1.47 (25M) / 0.66->1.36 (100M),
    B inflated 364->1630, beta -> 0.03; at 400M kappa -> 0 (lag absorbed
    entirely into warped backbone params).
  VERDICT: annexation trades the smooth family for the sharp family -- no
  uniform win.  The lag term and gamma/B are partially interchangeable
  in-fit, which is direct evidence for the paper's architecture: freeze the
  backbone on its balanced fit, add the identified lag term post hoc.
  Closed as a formula direction; kept as an architecture-justification result
  (repro/formula_lab/annexation.py).

## Multi-seed paired A2/A1 (3 seeds: 1337/1338/1339, raw losses)

Best-fit spec family (tau-free): A2/A1 = 0.862 / 0.924 / 0.881 (w=1000/1200/
1500), r2 0.19-0.38; tau=850-fixed specs fit poorly (r2 0.08-0.29) and give
1.19-1.71.  Full spec range [0.86, 1.71], median 1.06.

CROSS-CHECK: floor-gap (integral) deposit with the independently measured
ladder exponent p_real=0.731 predicts (0.5^p - 0.15^p)/(1 - 0.5^p) = 0.889 --
inside the best-spec band.  Strong weighted closures (0.30-0.38) excluded at
this scale.  Two independent measurements (equal-S ladder; seed-paired
differences) agree through the floor-gap form: at 10.7M the deposit follows
the (sublinear) floor gap, and the delta>0 OOF transfer gains are carried by
the concentration channel -- fully consistent with the ladder conclusion.

## Adjudication round (convergence workflow output: TESTS REQUIRED x3)

Test 1 -- matched-probe calibration: SHIPPED.
  Rule: if a probe's stage-2 LR equals the target's terminal LR at the
  schedule-spec level (1% rel tol; wsd/wsdld end at 3e-5 = wsdcon_3 exactly),
  calibrate kappa on that probe alone; else pool.  Leakage-clean (schedule
  known a priori); post-hoc origin documented (identified after the 10.7M-bed
  mismatch; on that bed no match exists -> pools -> bed numbers unchanged).
  Results (ratio-of-means, 6/6 everywhere):
    arm                probes-only   dilution    (pooled shipped)
    d=0                -18.2         -20.6       (-17.1 / -19.0)
    d=1/4 default      -27.0         -30.4       (-23.0 / -25.8)
    d=(p-1)+ measured  -28.8         -32.1       (-23.3 / --)
    d=1/2 frontier     -36.8         -39.0       (-28.6 / -31.6; beats the
                                                  pooled grid oracle -34.5)
  LOS / T1 / matrix untouched by construction.

Test 2 -- backlog-saturation D-factor: CLOSED at the gates.
  Gate (i) grading 0/3 scales (model requires probe/sharp kappa ratio
  increasing in eta2; observed non-monotone/decreasing); best D*=0.3 rmse
  0.127 cannot beat the observation-LR null without the grading.
  Gate (ii) A2/A1 = 0.728-0.744 for all D* -- outside the measured
  [0.86, 0.92].  Per adjudication rule: direction closed, ship test 1 alone.

Test 3 -- merged clock (lam_S*eta + 1/tau_0): DESCRIPTIVE FORM ONLY;
  law and deployment procedure unchanged.
  (i) AIC: public wsdcon transients (9 pts, tau 649-5487, r2 0.76-0.98):
      pure-S wins (-29.1 vs affine -27.6; affine tau0->52k = degenerate);
      10.7M ladder: affine wins decisively (-23.9 vs pure-S -0.1) with
      lam_S=5.1, tau0=168 -- one functional form covers both regimes, lam_S
      same order across scales (5-6), tau0 is the scale-emergent part.
  (ii) FALSIFIER FAILED: fixing tau0 leaves the public probe/sharp lam_S gap
      unchanged (16/5 -> 16/4.8) -- the merged clock does not resolve the
      family mismatch.
  (iv) Non-regression: matched probes-only -27.7% (affine) vs -27.0% (pure).
  (iii) RETRODICTION FAILS (added after the re-adjudication audit caught its
      omission): deploying the affine kernel with the ladder constants
      (lam_S=5.11, tau0=168) on the 10.7M bed gives sharp600 -0.6% / wsd
      -0.3% / wsdld -0.3% vs the shipped lam*-grid -18.1/-7.6/-6.2 --
      tau0=168 caps kernel memory far below the deployed effective memory,
      and the m-suite probes show no measurable decaying transient to
      calibrate (lam_S, tau0) from.
  Verdict (re-scoped): the affine clock DESCRIBES the floor-relaxation
  transients (ladder AIC -23.9 vs -0.1) but provides NO working deployment
  route; lam_slow=10 stays on the public curves and the lam-grid step is
  RETAINED on new setups.  (test3_clock.py)

## Final re-adjudication round (verdict: TESTS REQUIRED x3 -> all executed)

T3-RESCOPE: DONE.  (iii) retrodiction implemented in test3_clock.py and
  reproduces the audit failure exactly (sharp600 -0.6% / wsd -0.3% / wsdld
  -0.3% vs shipped lam*-grid -18.1/-7.6/-6.2); Test-3 entry re-scoped to
  descriptive-only above; paper sentence re-scoped; doc nits fixed
  (matched tolerance 0.225% wsdld; ladder r2 >= 0.977).

T-A matrix matched-cells override: SHIPPED.  Promoted to
  repro/formula_lab/matrix_matched_cells.py; reproduces the mandated numbers:
    lr@10  : worst -2.99 (beats -2.72), mean -12.40 (beats -12.08) -- strict
             Pareto win at d=0, zero new DOF;
    d=1/4  : worst -2.85 (tie), median -11.42 (tie), mean -13.97 (beats
             -13.51); matched cells wsdcon_3->wsd -28.14 (baseline -21.33),
             ->wsdld -24.29 (baseline -18.32); 9 scale-cells overridden,
             maxCosK unchanged (override never touches cosine rows).
  Residual objection answered: the override is target-conditioned by the
  already-adjudicated schedule-level rule (no fitting, no labels).

T-C bed validation: RULE SCOPE-GATED (the pre-set 'cannot regress' framing
  assumed the rule would not fire; with m_wsdcon_15 trained it fires and
  HURTS at 10.7M):
    d=0    matched -1.1/-0.5/-0.4% vs pooled -18.1/-7.6/-6.2%
    d=0.75 matched -3.8/-1.8/-1.5% vs pooled -71.2/-36.1/-32.7%
    (kappa_matched = 0.0002-0.0009 vs pooled 0.0028-0.0223: the deepest-drop
    probe has the most concentration-suppressed visible amplitude.)
  Resolution (honest post-hoc scope refinement, same epistemic status as the
  rule's own post-hoc origin): the matched rule presumes floor-curvature-
  dominated amplitudes; gate it on the probe-measured floor exponent p > 1
  (public scales 1.06/1.49/1.25 -> fires, gains stand; 10.7M p_real=0.73 ->
  pools, shipped bed numbers stand).  The gate uses only quantities the
  shipped law already measures, never target data.
  (train_wsdcon15.py, analyze_matched_bed.py)

  PRE-REGISTERED PREDICTION (gate hardening, zero cost): on any future
  scale/bed, measure p from settled probe floors FIRST and commit
  fire (p > 1) / pool (p <= 1) BEFORE inspecting target residuals.  Note
  25M's p = 1.06 sits within qualitative three-point-fit uncertainty of the
  threshold; its boundary coverage is carried by the matched rule's existing
  6/6 per-scale held-out wins.  Failure asymmetry: wrongly pooling only
  forfeits upside (never regresses below shipped pooled numbers); the
  falsifiable direction is firing at p > 1 where the mechanism fails.

## Round "post-CONVERGED reopen" -- Attempt 1 verdicts (5090 bed)

1F derby (14 arms, 3 seeds, prereg=optsched_predictions_m.json): FINAL.
  Tail means [5800,6000); paired gaps vs ds=3000:
    g(1300)=+3.23e-3 +/-1.37e-3 | g(5000)=+7.14e-3 +/-1.05e-3 |
    g(5700)=+28.31e-3 +/-1.37e-3
  V1 FIRES (+7.14e-3 vs adiabatic +0.06e-3 + spread 0.62e-3; 2SE=2.09e-3)
  V2 FIRES (+28.31e-3 vs adiabatic +12.53e-3 + spread 1.26e-3; 2SE=2.73e-3)
  -> the lag law prices FINAL LOSS, not just curve MAE.  Closure chi2
  (1300/5000/5700, GLS-lite): MPL 205 > d=0 179 > d=0.5 113 > d=0.75 76
  (delta-chi2=37 >= 6: d=0.75 selected; ALL closures underprice late
  cooldowns -- mechanism stronger than strongest shipped closure).
  V4 FLAG: ds=1300 measured +3.2e-3 vs MPL +9.8e-3 (backbone overprices
  early cooldown); per prereg the absolute-pricing claims carry this flag.
  Consequence per plan: Attempt-2 schedule appendix gates may run in full.

1A bladder PRELIMINARY (14/19; all 5 B2=192 arms OOMed under 12-way
  concurrency -- refilling at P=3): b_B=-0.338 [90% CI -0.522,-0.241]
  (excludes 0, contains -0.5); paired floor gaps q(1e-4)=-0.891,
  q(4e-4)=-0.701 (C2 band [-0.65,-0.35]).  FINAL verdict after b192 refill.
