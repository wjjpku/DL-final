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

1A bladder FINAL (19/19, b192 refilled at P=3, prereg=bladder_prereg.json):
  regression log tau = a + b_B log B2 + b_eta log eta2 over 14 used fits
  (r2 gate 0.6 passed by all):
    b_B = -0.199  (90% CI [-0.273, -0.117]),  b_eta = +0.145
  Verdict vs pre-registered buckets: NONE FIRES -- the CI excludes BOTH
  -0.5 (C2 noise clock) AND 0 (C1/C3 B-blind).  Amplitude check confirms:
  q(1e-4) = -0.891, q(4e-4) = -1.796, far outside C2's [-0.65,-0.35]; and
  the gap = G*B^q power law is itself misspecified (paired gaps flip sign
  at B2 >= 96: drop arms END WORSE than no-drop controls at large batch).
  => DICHOTOMY FALSIFIED at 10.7M: the slow-mode clock carries a weak but
  real batch dependence tau ~ B^-0.2, inconsistent with both the pure
  optimizer-noise clock (B^-1/2) and a pure step clock (B^0).
  Note the preliminary 14-arm b_B=-0.338 collapsed to -0.199 once the five
  b192 arms anchored the high-B end -- the OOM'd arms were load-bearing;
  preliminary leaning ("noise clock") would have been the WRONG call.
  Post-hoc interpretation (not verdict-bearing, G5): an intermediate
  exponent is the natural signature of a MIXTURE of relaxation channels
  (one B-clocked, one B-blind); single-mechanism accounts are too simple.
  Sign flip at large B is an independent novel observation: at B2>=96 the
  benefit of dropping eta vanishes/inverts in the tail mean.
  Disposition: B-axis at 10.7M is ANSWERED (not unmeasurable, not either
  prereg branch).  Deployment kernel keeps the shipped B-blind clock as
  scoped (10.7M bed, bs=48); B-corrected clock would need the mixture
  decomposition -- routed to Attempt 3 spectroscopy as a target question
  (does the slow channel's edge population shift with B?).
  (train_bladder.py, analyze_bladder.py, results/BLADDER_REPORT.json)

Attempt 2 Stage-1 FINAL (scale ladder m/l, prereg=scaleladder_prereg.json):
  m (10.7M, 8 rungs x 2 seeds): p = 0.815  (90% CI [0.658, 0.974])
  l (25M,   8 rungs x 2 seeds): p = 0.834  (90% CI [0.611, 1.046])
  dp(m->l) = +0.019 vs pre-registered fire line 0.380 -> PRIMARY emergence
  DOES NOT FIRE; CIs overlap massively; BONUS tier no.
  => The public 25M bed's p=1.06 is NOT a parameter-count effect: at the
  matched single recipe (same depth/width co-scaling, bs=48, equal-S
  ladder) 25M shows the SAME sublinear floor exponent as 10.7M (m's CI
  excludes 1).  The superlinear-floor onset reported on public beds is a
  recipe/batch-size effect, not scale emergence.  Paper sentence on p(N)
  re-scoped accordingly.
  STAGING GATE: CLOSED (p_l=0.834 < 0.9, CIs not disjoint) -- ml/xl NOT
  trained, per prereg (no fishing).
  GATE-HARDENING at l: NOT SCOREABLE (CI straddles 1); per prereg no
  fire/pool claim is made.
  tau ladder: p_tau = 0.10 (m, r2=0.39), -0.01 (l, r2=0.01) -- no 1/eta
  clock at either scale; per-scale AIC prefers affine clock with finite
  tau0 (140 m / 183 l), consistent with the shipped lam_slow account.
  Clock-rescope committed prediction (tau0 -> infinity as p crosses 1)
  untestable here since p never crosses 1 -- recorded, not claimed.
  Attempt-3 scale slot (gate-filled per repin_prereg): m + l.
  Operational note: first chain launch lost 26 arms to dash-vs-bash <<< and
  6+2 jobs to OOM against the b192 refill (zero curves lost for verdicts;
  trunks survived, refill idempotent via skip-if-exists).  schedules.json
  read-modify-write race killed wsdcon_20 (relaunched solo).
  (train_floor2.py, train_suite.py, analyze_floor2.py, FLOOR2_REPORT.json)

Attempt 3 spectroscopy, m bed FINAL (prereg=repin_prereg.json):
  OFF_REGIME fires.  Control arm preconditioned sharpness S_pre(0) = 7158
  vs AdamW momentum-corrected edge 38/eta1 = 25333: ratio 0.28, outside
  the pre-registered [0.5, 1.5] sanity band.  Training at 10.7M/bs=48 is
  NOT edge-pinned (progressive-sharpening regime instead); the edge
  re-pinning account of the post-drop slow mode is FALSIFIED on this bed
  (terminal per prereg: spectral account CLOSED, slow mode remains
  phenomenological).  Decoupling evidence doubles it: tau_loss = 209-273
  across drop arms while R(dS) never reaches 1-1/e on any drop arm
  (tau_spectral unfittable); probes clean (Ritz residual 0.000, only 2
  late probes dropped on the control).
  B-axis secondary NOT SCORED (requires C_REPIN); descriptive note: b192
  shows the only clean monotone S_pre rise (rho=0.94) -- recorded, no
  claim.  l-bed verdict pending (l wave re-running at micro-12 probes
  P=1 after the 3x10.4GB OOM collision; coexists with Phase-B derby).
  (train_spectrum.py, analyze_spectrum.py, SPECTRUM_REPORT.json)

Attempt 2 appendix FINAL-pending-contingency (25M derby, 12 arms,
prereg=optsched_predictions_l.json committed 8af32fa before launch):
  Tail means [5800,6000); paired gaps vs ds=3000:
    g(1300)=+2.14e-3 +/-0.87e-3 | g(5000)=+8.61e-3 +/-0.63e-3 |
    g(5700)=+30.34e-3 +/-0.70e-3
  V1_l FIRES (+8.61e-3 vs adiabatic +0.17e-3 + spread 0.14e-3; 2SE=1.27e-3)
  V2_l FIRES (+30.34e-3 vs adiabatic +13.95e-3 + spread 0.51e-3; 2SE=1.39e-3)
  => lag pricing of FINAL loss REPLICATES at 25M: not a 10.7M artifact.
  V5_transfer: B_identity beats naive on ALL closures (delta-chi2 = +73.4 /
  +206.7 / +226.9 for d=0/0.5/0.75, far over the 6 line) => the
  zero-measurement kappa transfer kappa_l = kappa_m * B_l/B_m is
  decisively selected; the B-identity chain generalizes across scale.
  Best closure at l remains d=0.75 (chi2 120 vs d=0 668) and ALL closures
  still underprice late cooldowns -- same mechanism-stronger-than-formula
  signature as the m bed.
  Contingency clause triggered to the letter: max paired SD 1.51e-3 >
  1.5e-3 (driven by the ds=1300 sanity arm; verdict-bearing gaps are at
  1.10/1.21e-3) -> seeds 1340/1341 on ds {3000,5700} launched; verdicts
  will be restated on 5 seeds when they land (no directional risk: V1/V2
  margins are 6-12x their 2SE).
  (train_optsched.py --scale l, analyze_optsched_l.py,
   OPTSCHED_L_REPORT.json)

## Mid-round adversarial audit (5-agent workflow) -- remediation record

AUDIT-C (Attempt 2 floor-extraction bug, severity major, verdict SURVIVES
AND TIGHTENS): analyze_floor2.py took the last n//4 CSV ROWS, which is
layout-dependent (old m files start at step 0, new at 3000 with dense early
sampling; the no-drop rung's window even started 120 steps BEFORE the
drop).  Fixed to the design window step >= 3000 + 0.75*T2.  Corrected:
  m: p = 0.647 (90% CI [0.610, 0.683]);  l: p = 0.641 ([0.613, 0.673])
  dp(m->l) = -0.006 vs fire line 0.067 (the test is now properly powered:
  prereg expected signal 0.33 >> 0.067, unlike the shipped 0.380)
  -> PRIMARY no; both CIs EXCLUDE 1 -> sublinear THROUGH the top trained
  scale is now established (the prereg sublinear_through_top branch applies
  legitimately); per-seed "chimera" (0.715 vs 0.861) collapses to
  0.626/0.669 -- a uniform seed offset, protocol identity confirmed.
  GATE-HARDENING at l RE-SCORED: was scoreable all along -> committed call
  is "pool (p<=1)" (operationally unchanged; record corrected).
  ATTRIBUTION WEAKENED per audit: "public 25M p=1.06 is not a
  parameter-count effect at the matched recipe; the responsible bed-level
  difference (batch size, recipe, data/tokenization, horizon, schedule
  family, or floor protocol) is NOT localized."  A bs=192 mini-ladder
  (bs192_ladder_prereg.json, committed before launch) is running to give
  the batch-size hypothesis its first direct test; wording upgrades only
  if BS_DRIVES fires.
  Sensitivity per audit: no-150-rung p ~ 0.71 at both scales (stable under
  corrected windows); floors are equal-S endpoint losses, still relaxing
  (slopes -0.002..-0.025/1k steps) -- p is a budget-indexed equal-S
  exponent, not an equilibrium exponent; p_tau covers the fast mode only.

AUDIT-A (Attempt 1A, severity FATAL, VERDICT FLIPS): the shipped
b_B = -0.199 was an artifact of per-arm fit windows confounded with B2
(b12 fit over 8000 steps, b48+ over 4000; single-exp tau grows 2-3x with
window length on the same arm because the model is misspecified).
Window-matched re-analysis (bladder_rewindow.py, common caps, r2 gate 0.6):
  W=1000: b_B = +0.014 [-0.023, +0.056]
  W=2000: b_B = -0.016 [-0.059, +0.031]
  W=4000: b_B = +0.121 [+0.077, +0.190]
Every matched window lands in the pre-registered C1/C3 B-BLIND bucket
(point in [-0.15, 0.15], CI excludes -0.5).  RESTATED VERDICT: C1/C3
B-BLIND fires; "tau ~ B^-0.2" and "dichotomy falsified" are RETRACTED.
What survives: exclusion of the C2 noise clock (-0.5) under every spec;
the e10 sign flip at B2 >= 96 (robust at common horizon [6000,7000):
-0.109/-0.066/-0.029/+0.006/+0.034), qualified to eta2=1e-4 only
(B192/4e-4 gap is zero within noise).  The q amplitude claims are struck:
the gap power law crosses zero so q is unestimable (the shipped fit also
used un-preregistered sign-censoring).  Prereg disposition (B_blind
branch): slow mode at 10.7M is not optimizer-noise relaxation; mechanism
hunt moved to Attempt 3 -- which has since returned OFF_REGIME at m.
Deployment unchanged (shipped clock was already B-blind).
(bladder_rewindow.py)

AUDIT-B (1F + l-appendix, severity FATAL on metric construction, verdicts
RESTATED like-for-like): committed predictions were ENDPOINT (step 5999)
values while the metric is the window mean over [5800,6000) of
still-cooling curves -- the late-cooldown arm descends 15e-3 inside the
window, mechanically inflating measured-vs-predicted excesses.  Both sides
now use the identical 21-step eval grid (derby_likeforlike.py; zero new
freedom).  RESTATED:
  m bed:  V1 fires marginally vs the core ensemble (+7.14e-3 vs 3.35+0.84;
    margin 2.95e-3 vs 2SE 2.09e-3) but goes NULL under a widened 7-split
    ensemble (spread 18.8e-3) and sits below the backbone's own
    demonstrated V4 systematic (-6.3e-3 at ds=1300) => absolute lag
    pricing at m is NOT established.  V2 is NULL-consistent like-for-like
    (+28.31 vs 26.53+2.11).  Closure comparison (POST-HOC at m -- the chi2
    machinery was not in the m prereg; it was prereg'd only at l): d>0
    closures beat d=0/MPL (chi2 36.4/31.0 -> 18.0/15.5), d=0.5 vs d=0.75
    indistinguishable (delta 2.5 < 6).  "Mechanism stronger than strongest
    closure" RETRACTED (it inverts like-for-like).  Headline weakened to:
    late-cooldown final-loss gaps are consistent with the lag-corrected
    backbone; the adiabatic-only excess is marginal and ensemble-dependent.
  l bed (g(5700) restated on 5 seeds per contingency: +29.48e-3 +/-0.70):
    V1_l fires vs core ensemble (+8.61 vs 3.67+0.25; margin 4.7e-3 vs 2SE
    1.27e-3) -- stronger than m -- but NULL under the widened ensemble;
    V2_l NULL like-for-like (+29.48 vs 28.82+0.97).
    V5_transfer like-for-like is MIXED: B_identity wins d=0 (+12.7) and
    d=0.5 (+12.9), naive wins d=0.75 (-43.5).  "B-identity decisively
    selected" is RETRACTED -> "closure-dependent; not separated overall".
    kappa-ratio split instability (1.18 vs 2.42; B_m 581 vs 277) now
    reported as the transfer chain's dominant internal uncertainty.
  m contingency seeds 1340/1341 on ds {3000,5000} are running (the m
  prereg's own 1.2e-3 clause had fired unnoticed); V1 will be restated on
  5 seeds when they land.  Bookkeeping fixed per audit: V4 band and the
  closure chi2 are labeled post-hoc at m; wsdld cross-check is
  cross-driver and not comparable; "14 arms 3 seeds" -> 14 runs.

AUDIT-D (Attempt 3 m-bed, severity major on wording, gate outcome STANDS):
  the OFF_REGIME gate is robust (median control ratio 0.23, all edge
  conventions outside band: 38/eta -> 0.28, 2/eta -> 5.4, 3.8/eta -> 2.8
  -- under the prereg's own definition the band check fails under all).
  BUT the terminal language overreached: the [0.5,1.5]x38/eta yardstick
  assumes deterministic full-batch dynamics with slow preconditioner;
  this bed is bs=48 + bf16 noise + beta2=0.95 (20-step v_hat memory),
  where noise-regulated SUB-edge curvature equilibration predicts exactly
  the observed sub-band reading -- and the b192 arm's clean monotone
  S_pre rise (7158 -> 21341, rho=0.94) positively supports that account.
  RESTATED: "the deterministic-AEoS edge-pinning frame fails its prereg
  sanity precondition on this bed; re-pinning at a stochastic/adaptive
  effective threshold remains UNTESTED."  "Progressive sharpening
  instead" is deleted (control raw lam_max falls; S_pre shows no trend).
  Disclosures added per audit: control S_pre is violently volatile under
  constant eta (kept range 4182-55136, 13x; the two dropped probes are
  the extremes 156717/45573 -- P(v_hat)-state-driven, not curvature);
  the raw-Hessian secondary channel NEVER converged (res_r 0.19-0.30 >>
  0.05 gate on all but 1 probe) and is unusable -- "probes clean" applied
  only to the preconditioned operator; probe values are not bitwise
  reproducible across driver revisions (2ppm FP-order difference).
  (derby_likeforlike.py, SPECTRUM_REPORT.json)

AUDIT-B addendum (m contingency executed): seeds 1340/1341 on ds {3000,
5000} landed; V1 restated on 5 seeds: g(5000) = +7.75e-3 +/- 0.68e-3,
fires vs the core ensemble with margin 3.6e-3 vs 2SE 1.37e-3 (was 3 seeds
/ margin 2.95 vs 2.09).  Wide-ensemble sensitivity unchanged (null).  The
m prereg's 1.2e-3 contingency clause is now discharged.

bs192 mini-ladder FINAL (prereg=bs192_ladder_prereg.json, committed before
launch; G3 trunk replay PASS):
  floors (design window): 0.9525/0.9677/0.9929/1.0254 at eta2 =
  {2,4,8,15}e-4;  p = 0.614 (90% CI [0.576, 0.651]), monotone.
  dp(bs48 -> bs192) = -0.033 vs fire line 0.074 -> BS_NULL.
  Batch size does NOT reproduce the public-bed superlinearity at the m
  recipe; combined with the matched-recipe scale ladder, BOTH parameter
  count AND batch size are now positively excluded as lone drivers; the
  attribution stays "source not localized (data/recipe/horizon/schedule
  family/floor protocol)".  Clean side observation: B shifts the floor
  LEVEL down substantially (consistent with noise-floor amplitude ~ eta/B)
  while leaving the eta-EXPONENT unchanged -- the floor exponent is
  invariant to the gradient-noise scale on this bed.
  (analyze_b192.py, curves_floor_m_b192/)

Attempt 3 spectroscopy, l bed FINAL (gate-filled second scale, post-audit
wording): OFF_REGIME REPLICATES at 25M.  Control S_pre(0) = 9619 vs
deterministic momentum-corrected edge 38/eta1 = 25333: ratio 0.38 (median
over kept constant-eta control probes 0.37; volatility range 21x, 5/12
probes dropped at the 0.05 Ritz gate -- dropped values 19.9k-80.1k
disclosed; same P(v_hat)-state-driven spike pattern as m).  The e10 drop
arm's in-window S_pre trend is NEGATIVE (rho = -0.80) -- the spectral top
moves AWAY from the new edge while the loss relaxes (tau_loss = 230).
Scoped conclusion (per AUDIT-D language): the deterministic-AEoS
edge-pinning frame fails its pre-registered sanity precondition on BOTH
beds (0.28 at 10.7M, 0.38 at 25M); post-drop loss relaxation proceeds
without deterministic-edge re-pinning at either scale; re-pinning at a
stochastic/adaptive effective threshold remains untested.  The slow mode
remains phenomenological.  Attempt 3 is now terminal at both
pre-registered scales.  (analyze_spectrum.py, SPECTRUM_REPORT.json)

## Round-convergence review #1 (6-agent workflow): CONTINUE -- worklist record

Verdict: CONTINUE (blocking bookkeeping + 5 escaping candidates E1-E5 from
the mechanism-lens exhaustion pass; methodology-lens pass returned empty).
Bookkeeping discharged with this commit:
  (a) Deposit cross-check bridging restatement (completes the parked
  deposit-form item): at the AUDIT-C corrected p = 0.647 the floor-gap
  deposit prediction moves 0.889 -> ~0.96, marginally ABOVE the measured
  0.86-0.92 band; the cross-check weakens from "inside band" to ~5%
  agreement.  The pre-audit sentence at the Multi-seed A2/A1 entry is
  superseded accordingly (paper already revised).
  (b) AUDIT-B m closure chi2 superseded by the 5-seed artifact: dMPL 64.6
  / d=0 53.8 / d=0.5 26.1 / d=0.75 17.1 -- delta(d=0.5 vs d=0.75) = 9.0
  >= 6 would select d=0.75 under the post-hoc rule; recorded 3-seed
  figures (36.4/31.0 -> 18.0/15.5, "indistinguishable") are the
  superseded snapshot.  Still post-hoc at m, non-verdict-bearing.
  (c) Pre-audit l-derby descriptive RETRACTED explicitly: "best closure
  at l remains d=0.75 (chi2 120 vs 668); all closures underprice late
  cooldowns" is inverted like-for-like (d=0.5 best, 85.4 vs 136.1; V2_l
  null => no underpricing).
  (d) AUDIT-E count fix: the audit's five labeled remediations are A-D
  plus the ensemble-widening/kappa-split-instability item that was folded
  into AUDIT-B's like-for-like record; there is no separate finding E.
  (e) BLADDER_REPORT.json is superseded by bladder_rewindow.py (AUDIT-A);
  restated b_B lives in the AUDIT-A record.
  (f) Review pass-2's zero-GPU checks entered on the record: bitwise
  trunk-identity for all seed-paired derby arms at m and l confirmed
  (maxdiff 0.0 over 150 shared-prefix eval rows); model-free t_half
  slow-clock estimator on the bladder common window gives b_B ~ +0.23,
  consistent with the AUDIT-A B-blind-to-mildly-positive restatement.
Escaping candidates adopted as the new round's worklist (preregs to
follow per G4): A1-A5 zero-GPU analyses (sub-edge mode tracking; public
three-point protocol extraction; equilibrium-p from F_inf; floor-level
sign-flip pricing; dL_eq/deta-anchored B_m) gating E1 (backbone-null
matched-S paired derby), E4 (horizon-extended ladder), E5 (sign-flip
seed replication), E3 (concentration-graded drop ladder), E2
(stochastic-threshold re-pinning), in that order.

## Round 2 worklist: A-series zero-GPU analyses (verified, scripts committed)

A1 (sub-edge mode tracking): NO trackable sub-edge preconditioned mode
  relaxes with tau within 3x tau_loss in the spike-robust ABSOLUTE
  observable, any drop arm, either scale (0/16 ranks x 2 associations;
  controls 0 FP).  The ratio channel shows in-band relaxations at m but
  false-positives on the l constant-eta control -> not verdict-bearing.
  "Slow mode remains phenomenological" STANDS, strengthened.  E2 must
  bring its own equilibration observable.  (a1_mode_tracking.py)
A2 (public protocol extraction): the settled-wsdcon protocol MANUFACTURES
  superlinearity on our beds.  Fixed-T2 wsdcon floors are U-shaped (the
  documented S-confound); fit over all probes degenerates (p pins at the
  0.2 bound), but sampled in the public 3-POINT WINDOW the same data read
  p ~ 3.5 (pinned at the upper bound) at BOTH scales whose equal-S
  ladders give 0.647/0.641.  The unlocalized-source sentence now has a
  PRIMARY SUSPECT: floor protocol.  Public 1.06/1.49/1.25 three-point
  fits inherit a demonstrated mechanism-level caveat.
  (a2_protocol_extraction.py)
A3 (equilibrium-p): the fitted-asymptote exponent does NOT stay
  sublinear: p_Finf ~ 1.04 [0.80,1.28] (m) / 1.00 [0.61,1.37] (l), while
  removing only the exp transient keeps p at 0.71/0.67 -- the entire gap
  is carried by the secular drift term b*T2 ~ b/eta.  Sublinearity is
  hereby scoped as a property of the equal-S budget-indexed floor; per
  the pre-stated gate E4 (horizon-extended ladder) is MANDATORY.
  (a3_equilibrium_p.py)
A4 (sign-flip pricing): the large-B sign flip is a horizon-indexed LEVEL
  inversion: benefit = ctrl_tail - F_inf crosses zero at B ~ 96 (measured
  flip B* = 86; predicted 80-86); the B-blind transient is real but ~3e-3
  (10x too small to drive it).  Free-s law gap = +224.6e-3 - 0.552*B^-0.20
  fits all five gaps within 0.65e-3; the eta/B and 1/sqrt(B) forms are
  REJECTED.  Note scope: bs192-ladder "amplitude ~ eta/B" must not be
  extrapolated to paired gaps.  Caveat: no s1338 controls exist -- the
  paired gap is unreplicated across seeds -> E5 sharpened to replicating
  CONTROLS at B in {96,192}.  (a4_signflip_pricing.py)
A5 (split-free B anchor): direct ladder-floor dL_eq/deta gives an
  anchored ratio B_l/B_m = 1.099 -- close to the wsdcon_20-split 1.18,
  far from wsdcon_40's 2.42 (confirming that fit as the unstable member)
  and below the committed ensemble mean 1.80.  Under the anchored ratio
  the kappa transfer is NOT SEPARATED from naive at ANY closure (deltas
  +1.6/+3.2/-0.8).  "Closure-dependent; not separated" is now confirmed
  split-free; B-identity's curve-shape support unaffected.
  (a5_B_anchor.py, sensitivity-only per G5)
