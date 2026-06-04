export const meta = {
  name: 'nqm-push-bigger',
  description: 'Push the non-adiabatic-lag conclusion beyond the paper: first-principles lambda_slow, cross-shape transfer, validity boundary, true 2-exp, momentum robustness',
  phases: [
    { title: 'Extend', detail: '5 agents: generalizations the paper did not do' },
    { title: 'Verify', detail: 'adversarial check' },
    { title: 'Synthesize', detail: 'what we added beyond the paper' },
  ],
}

const ROOT = 'c:/Users/21100/Desktop/represent'

const BRIEF = `
Paper: "Learning-Rate Schedules Are Not Adiabatic" (Jiaju Wu). On FAST LR decays an adiabatic
loss-curve law (MPL) leaves a positive residual = a non-adiabatic relaxation LAG. From AdamW's
preconditioned 2nd-moment dynamics the lag is a rate-dependent term DropRelaxS:
  K(t)=sum_{t'<=t} exp(-lambda_slow (S_t - S_t')) * drop_{t'},  drop_t=max(eta_{t-1}-eta_t,0), S=cumsum(eta),
  correction = c * eta_peak * (dL_eq/deta) * K(t).
Theory: noise-dominated AdamW -> eta_eff=eta/s, tau ~ 1/eta (p=1), lambda_slow = 2*(lambda/s)_eff
(a preconditioned curvature). The paper's OWN stated open problems (sec 6 Limitations):
  (a) "lambda_slow and c are measured, not computed from first principles (needs the noise spectrum {s_i})";
  (b) only ONE dataset / 3 scales; "broader scales and architectures are next steps";
  (c) assumes locally-quadratic, static spectrum, slow modes (eta*lambda/s << 1, far from edge of stability).
We attack exactly these in the controlled noisy-quadratic model (NQM) where we KNOW {lambda_i, s_i}.
`

const ENGINE = `
ENGINE ${ROOT}/repro/engine.py (numpy+scipy). Header:
    import sys; sys.path.insert(0, r'${ROOT}/repro'); import numpy as np; from engine import *
Key fns: adamw_nqm(lambdas,sigmas,etas,beta1=.9,beta2=.999,eps=1e-8,n_rep=4000,seed=0,theta0=None,weight_decay=0),
 equilibrate(...)->state, adamw_nqm_from_state(lambdas,sigmas,etas,state,...)->loss,
 nqm_linear_Leq(lambdas,sigmas,eta), nqm_linear_tau(...), cumS, drops, droprelaxS(etas,lambda_slow),
 droprelaxS_twoexp(etas,lam1,lam2,w1), measure_tau(loss,t0,fit_len=None,floor=None)->dict(tau,amp,floor,r2),
 fit_powerlaw(x,y)->(p,c,r2), schedules cosine_lrs/const_lrs/wsd_lrs/wsd_sqrt_lrs/two_stage_lrs.
Adiabatic baseline for a schedule = [nqm_linear_Leq(lambdas,sigmas,e) for e in etas]; residual=L_true-baseline.
Keep eta_eff*lam<<1 unless deliberately probing the edge. n_rep<=4000, steps<=5000. SAVE results/<file>.json.
`

const PEXP = {
  type: 'object', additionalProperties: true,
  properties: {
    summary: { type: 'string' },
    result: { type: 'object', additionalProperties: true },
    headline: { type: 'string', description: 'one sentence: the new finding beyond the paper' },
    success: { type: 'boolean' },
    artifact: { type: 'string' },
    caveats: { type: 'string' },
  },
  required: ['summary', 'result', 'headline', 'success'],
}
const PVERDICT = {
  type: 'object', additionalProperties: true,
  properties: {
    supports: { type: 'boolean' }, confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    concerns: { type: 'array', items: { type: 'string' } }, verdict: { type: 'string' },
  },
  required: ['supports', 'confidence', 'verdict', 'concerns'],
}

const TASKS = [
  {
    key: 'G1_lambda_slow_first_principles',
    task: `G1 -- CLOSE the paper's open problem (a): is lambda_slow COMPUTABLE from the spectrum {lambda_i,s_i}?
Pick a non-trivial noise-dominated spectrum (e.g. lambdas=geomspace(0.2,6,12), sigmas=ones*1.0). Run a wsd schedule
(eta_pk~2e-2,end~2e-3,T~4000,n_warm~200), get L_true (adamw_nqm) and residual r = L_true - adiabatic baseline.
Fit lambda_slow by maximizing R2 of r vs droprelaxS(etas,lambda_slow) (grid). Now compare the FITTED lambda_slow to
candidate first-principles predictors built from {lambda_i,s_i}:
   P_arith = 2*mean(lambda_i/s_i);  P_harm = 2*harmonic_mean(lambda_i/s_i);  P_min = 2*min(lambda_i/s_i);
   P_wmean = 2*sum(w_i * lambda_i/s_i)/sum(w_i) with w_i = lambda_i * dV*_i/deta (the kernel weights, since the
   slow tail dominates the end-of-curve lag). dV*_i/deta for noise-dominated ~ s_i/(2 lambda_i), so w_i ~ s_i/2.
Repeat for 3-4 DIFFERENT spectra (vary range/shape). Report which predictor best matches fitted lambda_slow across
spectra (ratio mean & CV). HEADLINE if one predictor (likely P_harm or P_wmean) matches within ~15% across spectra:
then lambda_slow IS computable from the spectrum -> the paper's 'measured not computed' gap is closed in the NQM.
success if some predictor has CV(ratio)<0.2 across spectra. Save results/G1.json.`,
  },
  {
    key: 'G2_cross_shape',
    task: `G2 -- CROSS-SHAPE transfer (paper only did cross-SCALE). Calibrate (lambda_slow, kappa=c*eta_pk*dLeq) on ONE
schedule shape (wsd), then PREDICT the lag on NOVEL shapes WITHOUT refitting. Spectrum: noise-dominated geomspace(0.3,3,10).
Calibration: wsd schedule -> r_wsd, fit lambda_slow & kappa (through-origin). Then for each NOVEL shape build the schedule
and predict lag_hat = kappa*droprelaxS(etas, lambda_slow); compare to measured r = L_true - adiabatic baseline; report R2 and
relative MAE for each:
  - triangular (warmup up then linear down to end),
  - two-step staircase (peak -> mid -> low),
  - exponential decay,
  - cosine (gradual),
  - a 'reverse wsd' (decay then re-warm) if feasible.
HEADLINE: the SAME (lambda_slow,kappa) predicts the lag across unseen schedule SHAPES (mean R2). success if mean R2>0.6 over
>=3 novel shapes. Save results/G2.json.`,
  },
  {
    key: 'G3_validity_boundary',
    task: `G3 -- MAP the validity boundary (paper assumes eta_eff*lambda<<1, far from edge of stability). Fix a spectrum
(noise-dominated, geomspace(0.5,5,8), sigmas=1). Sweep the PEAK lr so that the max preconditioned rate
rho = eta_eff*lambda_max = (eta/s)*lambda_max ranges from ~1e-3 up toward ~1 (edge of stability ~ rho=1, i.e. (1-rho)^2
loses contraction near rho>=1). For each peak: (i) measure tau from a step-down and check tau~1/eta locally; (ii) run a wsd
schedule and measure how well DropRelaxS predicts the residual (R2). Report rho vs {p_local, droprelaxS_R2}. Identify the rho
where the theory degrades (R2 drops below ~0.6 or p departs from 1 by >0.3). HEADLINE: quantitative validity boundary, e.g.
'DropRelaxS holds for rho<~0.2 and breaks as rho->1'. success if you can produce a monotone degradation curve and a threshold.
Save results/G3.json.`,
  },
  {
    key: 'G4_twoexp_bimodal',
    task: `G4 -- PROPER test of prediction (v) 'a 2-exponential kernel beats 1-exponential'. The earlier attempt failed because a
single exponential already fit a smooth spectrum at R2~0.999. Force a BIMODAL spectrum with TWO well-separated rate clusters:
e.g. lambda/s ratios clustered near 0.5 (slow) and near 8 (fast): build lambdas/sigmas so that lambda_i/s_i takes ~half values
~0.5 and ~half ~8 (e.g. sigmas=1, lambdas = concat(full(6,0.5), full(6,8.0))). Use a wsd schedule whose decay is FAST enough to
excite the fast cluster but whose post-decay window is LONG enough to see the slow cluster relax. Residual r = L_true - adiabatic
baseline over the decay+tail window. Fit 1-exp (best lambda_slow) -> R2_1; fit 2-exp droprelaxS_twoexp(lam1,lam2,w1)+amp (use a
small grid over lam1,lam2 and linear amp, or scipy) -> R2_2. Want R2_1 NOT near 1 (so there's room). Report R2_1,R2_2,delta and the
recovered (lam1,lam2) vs the true two clusters (2*0.5=1 and 2*8=16). HEADLINE if delta>0.03 AND recovered rates match the two
clusters: confirms the lag is a SPECTRAL MIXTURE. success if delta>0.03. Save results/G4.json.`,
  },
  {
    key: 'G5_momentum_wd',
    task: `G5 -- robustness of tau~1/eta to MOMENTUM (beta1) and decoupled WEIGHT DECAY (the 'lambda' in tau~1/(eta*lambda)).
(A) beta1 sweep: noise-dominated spectrum; for beta1 in [0.0, 0.5, 0.9, 0.95, 0.99] measure tau at a fixed eta (equilibrate->step
down). Does tau~1/eta still hold (re-fit p at beta1=0.9 vs 0.0)? Does beta1 rescale tau (momentum ~ 1/(1-beta1) effective)? Report.
(B) weight-decay: add decoupled wd (adamw_nqm weight_decay=wd) which adds curvature; the relaxation rate should become
2*eta*(lambda_eff + wd)/s, so tau should DECREASE as wd grows. Sweep wd in [0, 0.01, 0.05, 0.1] at fixed eta and report tau vs wd;
check 1/tau is ~linear in (lambda_eff + wd). HEADLINE: tau~1/eta is robust to momentum, and weight decay enters the relaxation rate
additively with curvature (testing the 'lambda' in tau~1/(eta*lambda)). success if p stays ~1 across beta1 AND tau decreases with wd.
Save results/G5.json.`,
  },
]

phase('Extend')
log(`Pushing beyond the paper: ${TASKS.length} generalizations.`)
const ext = await parallel(TASKS.map(t => () =>
  agent(`${BRIEF}\n${ENGINE}\n\n${t.task}\n\nWrite ${ROOT}/repro/${t.key}.py importing the engine, run it
(cd ${ROOT} && python repro/${t.key}.py), iterate until clean (no NaNs), then return the structured result.`,
    { label: t.key, phase: 'Extend', schema: PEXP }).then(r => ({ ...t, result: r }))
))

phase('Verify')
const ver = await parallel(ext.filter(Boolean).map(er => () =>
  agent(`${BRIEF}\n\nADVERSARIAL CHECK of ${er.key}. Read ${ROOT}/repro/${er.key}.py and its results/*.json. REFUTE, don't rubber-stamp.
Watch for: circularity (ground truth must be the real adamw_nqm Monte-Carlo, not nqm_linear_Leq used on both sides),
overfit kernels (too many free params for few points), R2 inflated by the flat stable phase, cherry-picked spectra,
claims stronger than the numbers. Reported: ${JSON.stringify(er.result).slice(0, 1400)}.
Spot-check by rerunning if useful. Verdict: does the evidence support the headline?`,
    { label: 'verify:' + er.key, phase: 'Verify', schema: PVERDICT }).then(v => ({ key: er.key, headline: er.result?.headline, success: er.result?.success, verdict: v }))
))

phase('Synthesize')
const synth = await agent(
  `${BRIEF}\n\nSynthesis. Results+verdicts: ${JSON.stringify(ver.filter(Boolean), null, 1)}.
Write ${ROOT}/results/EXTENSIONS_REPORT.md: a table (Extension | What the paper left open | Our finding | Verified? | Confidence)
for G1..G5, then 5 sentences on how much further the non-adiabatic-lag picture now reaches beyond the paper, and which
extensions are solid vs suggestive. Return the markdown.`,
  { label: 'synthesize', phase: 'Synthesize' })

return { extensions: ver.filter(Boolean).map(v => ({ key: v.key, success: v.success, supports: v.verdict?.supports, conf: v.verdict?.confidence, headline: v.headline })), report: `${ROOT}/results/EXTENSIONS_REPORT.md`, preview: synth.slice(0, 600) }
