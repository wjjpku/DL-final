# Convergence dossier (input for the adversarial adjudication workflow)

State of the formula-optimization program as of the structural-overhaul round.
Read together with DECISION_TABLE.md (same directory) which holds all numbers.

## Shipped (in paper + PR #1)
1. B-identity: dL_eq/deta|_S = B (proved + verified 5-15%); zero-extra-
   measurement amplitude chain (-43.1% vs -44.0% probe chain, 6/6).
2. Second-order amplitude closure chi(eta) ~ eta^delta (Eq. 12): delta=1/4
   Pareto-dominant on the audited 6x6 matrix; probes-only -17.1 -> -23/-29%;
   real-model never-scanned beds -18 -> -71% (sharp600), -14 -> -52%
   (showcase wsd_sharp).
3. Theory: NQM affine route falsified at magnitude; bulk+edge mechanism
   (tracking cutoff) consistent; floor exponent emerges with scale
   (0.73 -> 1.06 -> 1.49 -> 1.25); edge re-adaptation flagged conjecture.

## Closed directions (with evidence type)
- Kernel shapes (2-exp / Lomax / stretched): EXPERIMENT, twice (with and
  without weight) -- collapse to one pole; never beat matched one-pole in any
  tested protocol.
- delta > 0.5 aggressive weights: EXPERIMENT -- matrix worst-case fails
  (+0.02 at 0.5@10, +12 at 0.5@5, +13.6 affine 0.9).
- Naive NQM affine chi: THEORY, quantitative (3 orders of magnitude).
- lambda cross-family unification via weights: EXPERIMENT (15-19 vs 0.5-5).
- Full state-space replacement of MPL (river-valley laws): EXPERIMENT
  (pre-existing in repo: 0/15 vs MPL; RV-EoS, lag-MPL, EoS-kernel all lost).
- Nonlinear-relaxation ODE law (NEW this round): EXPERIMENT -- mechanism real
  (joint-fit family reconciliation) but r* unidentifiable within family;
  LOO transfer -23.1% dominated by shipped delta=0.5 (-28.6%). Kept as
  mechanism only.
- Two-channel C7 (memoryless superlinear floor + unweighted lag): REASONING
  only -- memoryless term cannot express the decaying transient; high risk of
  absorption by MPL gamma refit. (Weakest closure in the list.)

## Direction B annexation -- RESULT (closed)
Joint 9-param fit reaches -53.3% on held-out sharp BUT collapses cosine_72000
(+85 to +197%) by warping gamma (0.64->1.47) and B (364->1630); at 400M the
fit sets kappa->0 (lag absorbed into gamma/B).  Verdict: lag and gamma/B are
partially interchangeable in-fit; freeze-then-correct is the stable
architecture.  No uniform win; closed (annexation.py; also written into the
paper's gamma-relation paragraph).

## Pending at dossier time
- Multi-seed paired A2/A1 (3 seeds): GPU running, to be appended.

## Known open problems (documented in paper, not claimed solved)
- lambda_slow, c from first principles (needs noise spectrum).
- Concentration-dependence of c (mechanism identified: nonlinear relaxation /
  fast-mode absorption; no closed-form predictor).
- Identifiability-gate threshold portability across suites (invsqrt case).
- Edge re-adaptation conjecture (needs Hessian/noise spectrum logging).
- S-time vs step-time memory variable (needs batch-size manipulation runs).

## Resources & constraints
- Local RTX 5080 laptop (16GB), ~14 min per 6k-step 10.7M curve.
- Public data: fixed (9 schedules x 3 scales). No new public scales available.
- Paper philosophy constraint: every new degree of freedom must carry a
  derivation hook and a leakage-clean calibration route.

## Multi-seed A2/A1 -- RESULT (resolved)
3-seed averaged paired differences: best-spec band 0.86-0.92 (tau-free,
r2 0.19-0.38); floor-gap deposit at ladder-measured p_real=0.73 predicts
0.889 (inside band); strong weighted closures (0.30-0.38) excluded at 10.7M.
Two independent measurements agree through the floor-gap form.  Paper twodrop
paragraph upgraded from 'unmeasurable' to this two-way consistency result.
