# The second-order amplitude structure of the non-adiabatic correction

Companion derivations for the formula-structure upgrade (2026-06).  Everything
here refers to the operational law

    L(t) = L_MPL(t) + kappa * Phi(t),
    Phi(t) = (1/eta_peak) sum_{k<=t} w(eta_k) * exp(-lambda_slow (S_t - S_k)) * (eta_{k-1}-eta_k)_+,

with the paper's original (first-order) closure w == 1.  Empirical support:
`results/formula_lab/DECISION_TABLE.md`.

## 1. Proposition (B-identity): MPL's B is the equilibrium-floor sensitivity

Claim: the MPL parameter `B` equals the MPL-implied fixed-progress LR
sensitivity of the equilibrium loss,

    dL_eq^MPL / deta |_S = B.

Proof sketch.  Compare two schedules that agree up to step k, after which one
holds eta and the other drops to eta - deta and holds.  MPL's annealing term
changes by  -B * deta * G(eta^{-gamma} (S_t - S_k)),  and G -> 1 as
S_t - S_k -> infinity (G(x) = 1 - (1+Cx)^{-beta} saturates).  The S^{-alpha}
backbone difference vanishes at matched S.  Hence the saturated (equilibrated)
loss difference per unit LR drop is exactly B: MPL structurally encodes a
LINEAR equilibrium floor  L_eq(S, eta) = L0 + A S^{-alpha} + B eta + const.

Empirical check (public curves, frozen official params): B = 363.8 / 437.9 /
523.4 at 25/100/400M vs the independently measured probe slope dL_eq/deta =
381.7 / 499.3 / 561.5 — agreement within 5-15% at all three scales, with the
same N-trend (~N^0.26).

Consequences:
(a) Zero-extra-measurement amplitude rule: kappa = c'' * eta_peak * B, with
    c'' transferred leave-one-scale-out.  kappa_fit/(eta_peak B) =
    0.449/0.584/0.646 (CV 14.7%), as stable as the probe chain (13.9%).
    Table-1 protocol with this rule: -43.1% (6/6) vs -44.0% for the probe
    chain — at zero additional training runs and zero fitting at the target.
    Honesty: B inherits MPL's official fit split, which contains one
    two-stage schedule (wsdcon_9); refitting MPL on smooth-only splits tests
    the dependence (see b_stability.py results).
(b) Because MPL's floor is linear in eta, any superlinear floor component is
    structurally invisible to MPL — it must surface in the correction term's
    amplitude.  That is the next section.

## 2. The floor is superlinear; the exact NQM route fails at magnitude

Measured settled floors (wsdcon stage-2 finals minus the A S^-alpha backbone)
fit  F_eq ~ eta^p  with  p = 1.06 / 1.49 / 1.25  at 25/100/400M (3-point
log-log fits; qualitative support for p > 1, not a precise universal value).

Naive explanation from the paper's own AdamW specialization fails.  The exact
noise-dominated mode gives

    V_i*(eta) = eta^2 / (1 - (1 - eta a_i)^2),  a_i = lambda_i / s_i
    F_i*(eta) = (lambda_i / 2) V_i* = eta s_i / (2 (2 - eta a_i)),
    chi_i(eta) = dF_i*/deta = s_i / (2 - eta a_i)^2 = s_i/4 + (s_i a_i / 4) eta + O(eta^2).

(NB the affine coefficient is s_i a_i/4 = lambda_i/4, not /8.)  For the modes
that carry the measured lag, eta_peak * a_eff ~ 2e-3 << 2, so chi varies by
~0.1% across the decay window — three orders of magnitude short of the
observed ~3x (eta_peak/eta_end = 10, delta ~ 0.5 ⇒ chi ratio ~ 3.2).  The
superlinear floor CANNOT come from the same slow noise-dominated modes that
set lambda_slow ~ 10.

## 3. Bulk + edge spectral derivation: F_eq(eta) = c1 eta + c2 eta^zeta

Let the s-weighted density of preconditioned rates a = lambda/s be g(a),
with two qualitative populations:

  * bulk: slow modes, eta a << 2 (deep noise-dominated); their floor
    contribution is sum s_i eta / 4 — LINEAR in eta;
  * edge: a power-law tail g(a) ~ g0 a^{-zeta} truncated at the adaptive
    stability boundary  a_max(t) = 2 theta / eta(t), theta < 1 — adaptive
    optimizers self-organize so the fastest preconditioned modes sit near
    the edge AT THE CURRENT LR (tracking cutoff; cf. adaptive edge of
    stability).

With the tracking cutoff, substitute u = eta a in the edge integral:

    F_edge(eta) = ∫^{2 theta/eta} eta g0 a^{-zeta} / (2 (2 - eta a)) da
                = g0 eta^{zeta} ∫^{2 theta} u^{-zeta} / (2 (2-u)) du
                = C(zeta, theta) * eta^{zeta}.

For zeta in (1,2) the lower end of the tail integral merges into the bulk
(linear) term, so

    F_eq(eta) = c1 eta + c2 eta^{zeta},   chi(eta) = c1 + zeta c2 eta^{zeta-1}.

A pure power-law fit to floors over one decade of eta then yields an
effective exponent p in (1, zeta), matching p ~ 1.2-1.5 without fine-tuning.
The frozen-cutoff alternative (a_max pinned at 2 theta/eta_peak forever)
yields a near-linear floor with only an O(eta^2/(2-u_max)) correction — the
two hypotheses are distinguishable by whether the floor stays superlinear far
below eta_peak.

Honest gap (flagged, not hidden): in a static spectrum the edge population
relaxes in O(1) steps, so by itself it would equilibrate instantly and
contribute no slow lag.  The conjecture consistent with all observations is
edge RE-ADAPTATION: after an LR change the edge population re-pins to the new
stability boundary on the slow preconditioner/curvature-adaptation timescale,
which is the lambda_slow channel.  We state this as mechanism-consistent, not
first-principles-derived.

## 4. Operational second-order closures and their domains

Two one-parameter closures implement chi(eta):

  * point deposit (recommended): w(eta_k) = (eta_k / eta_peak)^delta with the
    POST-decrement LR (the operating point of the subsequent relaxation);
  * integral deposit: replace w*deta by the traversed floor gap
    (eta_{k-1}^p - eta_k^p)/(p eta_peak^{p-1}) — equal to the point form up to
    O(deta^2) on smooth schedules, ~1/p on instantaneous drops.

Public-curve verdicts (DECISION_TABLE.md):
  * delta = 1/4 (pre-registered universal default) Pareto-improves the audited
    6x6 transfer matrix on every aggregate (worst -2.85 vs -2.72, mean -13.51
    vs -12.08, cosine->WSD -8.44 vs -4.27) and lifts cheap-probe calibration
    from -17.1% to -23.0% (probes-only) and -19.0% to -25.8% (mixed), at a
    <= 3pp cost on in-family sharp protocols.
  * per-scale measured delta = (p-1)_+ behaves similarly (matrix worst -1.97,
    probes-only -23.3%).
  * delta ~ 0.5-0.75 maximizes probe-calibrated transfer (-34%) but is
    matrix-marginal and, at matched (oracle) amplitude, the weighted shape
    fits sharp in-family residuals slightly WORSE than w == 1.  Part of the
    cheap-calibration gain is therefore amplitude recalibration (the <= 1
    weight deflates probe features, inflating fitted kappa toward the sharp
    value) rather than pure shape improvement.  We say this explicitly.

## 5. Concentration-dependence of the spectral compression c (open, measured)

After removing the chi(eta) curvature, instantaneous drops still show a
smaller VISIBLE slow-lag amplitude per unit LR drop than gradual decays
(probe/sharp effective-kappa ratio ~ 0.2-0.45; early-window fits exclude tail
contamination as the cause).  Within the spectral picture this is the
mixture-compression factor c (paper: c ~ 0.5 on sharp decays) being smaller
for drops concentrated in S-time (c_probe ~ 0.15-0.25): the faster part of
the response spectrum completes between logging strides for a concentrated
drop, while a gradual decay re-excites it continuously.  Single-exponential,
two-exponential and Lomax kernels all fail to absorb this (kernel-shape
closure: shapes collapse back to one pole with and without the weight), so we
report c's concentration-dependence as a measured limitation refining the
paper's open problem (a), not as a fitted term.

## 5b. Out-of-family real-model evidence (10.7M transformer, never-scanned beds)

Probe-only calibration (kappa + lam from the suite's wsdcon probes; lam* ~ 1
matches the measured flat tau ~ 850 steps), held-out targets:

| target | delta=0 | delta=1/2 | delta=3/4 |
|---|---|---|---|
| sharp600 (600-step decay)  | -18.1% | -61.3% | -71.2% |
| wsd / wsdld                | -7.6 / -6.2% | -26.5 / -22.2% | -36.1 / -32.7% |
| showcase wsd_sharp (peak 2.5e-3) | -13.8% | -38.0% | -52.4% |
| showcase wsd_grad          | -6.3% | -16.1% | -20.2% |
| invsqrt (smooth, adiabatic)| +6.1% | +25.1% | +40.1%  (over-correction) |

Monotone-in-delta wins on every fast-decay target; neither bed was used to
select delta.  Boundaries: raw probe-kappa over-corrects the smooth invsqrt
family (the adiabatic-regime failure the public-curve estimator machinery
gates; its absolute thresholds do NOT transfer across suites -- both the
smooth_id_weight heuristic (w_id=0.90 on invsqrt) and the NextGen retention
floor 0.01 (which would also abstain on sharp600 here) need per-suite
recalibration).  cyclic re-warm breaks the MPL backbone itself (resid +0.07).
twodrop deposit-ratio test A2/A1 (predictions 0.30 p=2 / 0.38 delta=1/2 /
0.70 delta=0): UNMEASURABLE at this scale -- the kernel decomposition gives
<= 0.13 but its kernels are collinear at lam ~ 1 (artifact), and the
seed-paired difference gives specification-dependent 0.86-2.06 at r2 <= 0.3
(eval noise + smoothing window).  Needs seed averaging or a larger model.

## 5c. Equal-S floor ladder (confound-free, 10.7M): the exponent is scale-dependent

Six seed-matched constant-stage runs ending at identical cumulative LR S*
(backbone cancels):
- floors strictly monotone in eta2 (the earlier U-shape was the backbone
  confound);
- floor = 1.0385 + 0.1178 (eta/peak)^0.731, p_real = 0.731 (90% CI
  [0.615, 0.855]) -- SUBLINEAR at 10.7M;
- tau ~ eta^-0.18 with r2 >= 0.977 per rung (clean long-window confirmation of
  the flat-tau anomaly).

Public settled-probe scale trend of p: 1.06 (25M) / 1.49 (100M) / 1.25 (400M).
SUPERSEDED reading (round-2 RETRACTION): this is NOT scale emergence -- matched-
recipe confound-free equal-S ladders read SUBLINEAR (p=0.65 at 10.7M, 0.64 at
25M, horizon-robust to 24k; dp=-0.006 vs the 0.067 fire line), so the public
superlinearity is a settled-probe PROTOCOL artifact, not a parameter-count
(nor batch-size) effect; the bed-level source is not localized (see
DECISION_TABLE round-2 + paper sec:round2).  The delta>0
transfer gains at 10M are carried by the concentration-dependent visible
amplitude (Section 5), not floor curvature, and no universal delta across
scales exists -- the measured per-scale route is the defensible one.

## 6. What this kills

  * Kernel-shape upgrades (two-pole / Lomax / stretched-exp) as predictors:
    closed, twice (with and without the amplitude weight).
  * The "dropped O(eta) NQM term" story for the superlinear floor: falsified
    at magnitude (Section 2).
  * lambda unification across schedule families by the weight alone: fitted
    lambda stays ~15-19 (probes) vs ~2-5 (sharp, weakly identified); the
    compromise lambda = 10 remains the externally measured value.
