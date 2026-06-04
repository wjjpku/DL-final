export const meta = {
  name: 'nqm-reproduce-verify',
  description: 'Reproduce + extend the non-adiabatic loss-curve paper via the noisy-quadratic engine',
  phases: [
    { title: 'Experiments', detail: '6 agents: run NQM experiments against engine.py' },
    { title: 'Verify', detail: 'adversarial check of each experiment' },
    { title: 'Synthesize', detail: 'compile results vs paper' },
  ],
}

const ROOT = 'c:/Users/21100/Desktop/represent'

const BRIEF = `
You are reproducing/extending the paper "Learning-Rate Schedules Are Not Adiabatic:
A Rate-Dependent Correction for Loss-Curve Prediction" (Jiaju Wu).

CORE CLAIM: Adiabatic loss-curve laws (MPL/FSL) predict the quasi-static loss. On FAST
LR decays the loss LAGS equilibrium -> a positive residual. From AdamW's preconditioned
second-moment dynamics this lag is a rate-dependent term with relaxation time tau ~ 1/(eta*lambda).

THEORY (Hessian eigenbasis, mode i: curvature lambda_i, grad-noise std s_i):
- SGD step eta: V_{t+1}=(1-eta*lam)^2 V_t + eta^2 s^2 ; V* ~ eta s^2/(2 lam); relax mult e^{-2 eta lam}; tau~1/(eta lam).
- AdamW noise-dominated (preconditioner v* ~ s^2): effective step eta_eff=eta/s, so
  V* ~ eta s/(2 lam), L_eq ~ (eta/4) sum_i s_i, per-step rate 2 eta lam/s -> tau ~ 1/eta (p=1).
  S-time (cumulative-LR) rate lambda_slow = 2 lam/s (a preconditioned curvature; scale-invariant).
- AdamW SIGNAL-dominated (v* ~ lam^2 V): self-consistent treatment gives tau ~ eta^{-2/3} (p=2/3).
- Amplitude identity: total kernel weight sum_i w_i = dL_eq/deta.
- DropRelaxS kernel (Eq.4): K(t)=sum_{t'<=t} exp(-lambda_slow (S_t - S_t')) * drop_{t'},
  drop_t=max(eta_{t-1}-eta_t,0), S_t=cumsum(eta). Correction added to baseline = c * eta_peak * dL_eq/deta * K(t).

PAPER'S REPORTED RESULTS to compare against:
- Real wsdcon curves: tau ~ eta^{-p}, pooled p = 1.00 +/- 0.18.
- From-scratch AdamW NQM sim: tau ~ eta^{-1}, log-log slope p = 1.01.
- beta2 sweep [0.9, 0.9999]: tau FLAT, CV 2%.
- residual = DropRelaxS, R2 = 0.40/0.80/0.87 (25/100/400M).
- 2-exponential improves R2 by 0.06-0.07.
- amplitude: kappa_fit/kappa_pred CV 14%, c~0.5 universal.
`

const ENGINE = `
ENGINE API (file ${ROOT}/repro/engine.py, numpy+scipy only, import it; run python from ${ROOT}).
Example header:
    import sys; sys.path.insert(0, r'${ROOT}/repro')
    import numpy as np; from engine import *

Functions:
- adamw_nqm(lambdas, sigmas, etas, beta1=0.9, beta2=0.999, eps=1e-8, n_rep=4000, seed=0,
            theta0=None, weight_decay=0.0) -> loss_trace (T,)  [real AdamW on NQM, ensemble-averaged]
- equilibrate(lambdas, sigmas, eta, n_steps=4000, n_rep=4000, seed=0, beta1, beta2, eps)
       -> (theta,m,v) state at equilibrium for constant eta.
- adamw_nqm_from_state(lambdas, sigmas, etas, state, beta1=0.9, beta2=0.999, eps=1e-8, seed=1)
       -> loss_trace continuing from a state (bias-correction effectively off). USE THIS for
          relaxation: equilibrate at high eta, then step to low eta and record the transient.
- nqm_linear_Leq(lambdas, sigmas, eta) -> closed-form equilibrium loss (noise-dominated linear approx).
- nqm_linear_tau(lambdas, sigmas, eta) -> per-mode tau (steps) of linear approx (cross-check).
- cumS(etas), drops(etas)
- droprelaxS(etas, lambda_slow) -> K(t) (unit amplitude)
- droprelaxS_twoexp(etas, lam1, lam2, w1)
- mpl_loss(etas, L0,A,alpha,B,C,beta,gamma, S_warmup=0.0)
- measure_tau(loss, t0, fit_len=None, floor=None) -> dict(tau, amp, floor, r2)  [fits exp relaxation]
- fit_powerlaw(x, y) -> (p, c, r2)   [fits y = c x^{-p}]
- schedules: cosine_lrs, const_lrs, wsd_lrs, wsd_sqrt_lrs, wsdld? (use wsd_lrs / two_stage_lrs),
  two_stage_lrs(total, peak, lr_b, step, n_warm)

NOTES:
- Spectrum: lambdas = curvatures (>0), sigmas = grad-noise std per mode. d ~ 8-16 modes.
- Noise-dominated regime: make sigmas O(1) (large) so v* ~ s^2 dominates lam^2 V. Use eps small.
- Signal-dominated regime: make sigmas TINY (e.g. 1e-3) and start far from optimum / large curvature
  so v* ~ lam^2 V dominates; then preconditioner tracks signal not noise.
- For speed use n_rep ~ 2000-3000, ~3000-5000 relaxation steps. Keep eta_eff*lam << 1 (slow modes).
- ALWAYS save your structured results to ${ROOT}/results/<your_file>.json and print them.
`

const PEXP = {
  type: 'object', additionalProperties: true,
  properties: {
    summary: { type: 'string' },
    result: { type: 'object', additionalProperties: true },
    matches_paper: { type: 'boolean' },
    paper_value: { type: 'string' },
    our_value: { type: 'string' },
    artifact: { type: 'string', description: 'path to json/py/png saved' },
    caveats: { type: 'string' },
  },
  required: ['summary', 'result', 'matches_paper', 'our_value', 'paper_value'],
}

const PVERDICT = {
  type: 'object', additionalProperties: true,
  properties: {
    supports_claim: { type: 'boolean' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    concerns: { type: 'array', items: { type: 'string' } },
    is_circular_or_buggy: { type: 'boolean' },
    verdict: { type: 'string' },
  },
  required: ['supports_claim', 'confidence', 'verdict', 'concerns'],
}

const EXPERIMENTS = [
  {
    key: 'E1_tau_vs_eta',
    title: 'E1: tau ~ 1/eta (reproduce sec4 i\' / Fig2)',
    task: `EXPERIMENT E1 -- reproduce tau ~ 1/eta for from-scratch AdamW on a noisy quadratic.
Build a NOISE-DOMINATED spectrum: d=10 modes, lambdas = np.geomspace(0.5, 5.0, 10), sigmas = ones(10).
Equilibrate at a high LR eta_hi=3e-2 (n_steps>=3000, n_rep>=3000), capture state.
Then for each eta_lo in np.geomspace(2e-3, 3.2e-2, 8): continue from the SAME state with
adamw_nqm_from_state(..., etas=full(4000, eta_lo), seed=1); measure_tau(loss, t0=0).
Discard any fit with r2<0.9 or where eta_lo*max(lambda)/min(sigma) is not <<1 (edge of stability).
Fit tau vs eta with fit_powerlaw -> report p, c, r2. Cross-check the smallest-eta tau against
nqm_linear_tau (slowest mode) -- they should agree within ~30%.
PAPER: p ~ 1.01 (sim), pooled p=1.00+/-0.18 (real). Save results/E1.json. matches_paper if |p-1|<0.2.`,
  },
  {
    key: 'E2_beta2',
    title: 'E2: beta2 ablation (tau flat)',
    task: `EXPERIMENT E2 -- show tau is INDEPENDENT of beta2 in the noise-dominated regime.
Same noise-dominated spectrum as E1. Fix eta_hi=3e-2 (equilibrate) and eta_lo=6e-3.
For beta2 in [0.9, 0.95, 0.99, 0.999, 0.9999]: equilibrate at eta_hi WITH THAT beta2, then
adamw_nqm_from_state with same beta2; measure_tau. Report taus and CV = std/mean.
PAPER: tau flat, CV ~ 2%. Save results/E2.json. matches_paper if CV < 0.10.
Also briefly note WHY (preconditioner v~s^2 is beta2-independent at equilibrium on noise-dominated modes).`,
  },
  {
    key: 'E3_p_regimes',
    title: 'E3 (GENERALIZE): p=1 vs p=2/3',
    task: `EXPERIMENT E3 -- THE KEY GENERALIZATION the paper PREDICTS but DID NOT TEST:
tau ~ eta^{-1} for NOISE-dominated modes vs tau ~ eta^{-2/3} for SIGNAL-dominated modes.
(A) Noise-dominated: lambdas=geomspace(0.5,5,8), sigmas=ones(8)*1.0. Sweep eta_lo (equilibrate hi, step down),
    measure tau, fit_powerlaw -> expect p ~ 1.
(B) Signal-dominated: make gradient noise tiny so the Adam preconditioner tracks the SIGNAL g^2~(lam*theta)^2,
    not noise. Use sigmas = ones(8)*1e-3 (or smaller), lambdas=geomspace(0.5,5,8). Start the relaxation from a
    LARGE displacement (theta far from 0) so signal dominates v during relaxation: instead of equilibrate(),
    set the start state theta0 to a large constant (e.g. 1.0) with m=0,v=0 and call adamw_nqm with a constant
    eta_lo schedule and theta0=that; measure tau of the loss decay; sweep eta_lo and fit p.
    NOTE: in the noise-free Adam limit the sign-like update gives a characteristic eta-scaling -> expect p in
    [0.5, 0.8], distinct from 1. Report BOTH p values. Try a couple of variants to bracket p_signal robustly.
PAPER predicts p=1 (noise) and p=2/3 (signal). matches_paper if p_noise~1 (|.-1|<0.2) AND p_signal clearly < p_noise
(ideally near 2/3). Save results/E3.json with both regimes and the eta/tau tables. Be candid in caveats about how
clean the 2/3 is -- this is a genuine test of an untested prediction.`,
  },
  {
    key: 'E4_lag_kernel',
    title: 'E4: full-schedule lag = DropRelaxS + amplitude identity',
    task: `EXPERIMENT E4 -- on a full schedule, show (a) the residual signature (small on cosine, large on wsd),
(b) residual = DropRelaxS kernel, (c) amplitude = dL_eq/deta.
Spectrum: noise-dominated, lambdas=geomspace(0.3,3,12), sigmas=ones(12).
Schedules over T steps (use peak eta_pk=2e-2, end=2e-3, warmup small): build with engine schedules but SCALE so
eta_eff*lam stays <<1. e.g. cosine_lrs(T=4000, peak=2e-2, end=2e-3, n_warm=200) and
wsd_lrs(T=4000, decay_start=2800, peak=2e-2, end=2e-3, n_warm=200).
For each schedule: L_true = adamw_nqm(lambdas,sigmas,etas, n_rep>=4000, seed=0).
Adiabatic baseline L_eq(t) = [nqm_linear_Leq(lambdas,sigmas,eta_t) for eta_t in etas] (quasi-static).
Residual r(t) = L_true - L_eq. Compute lambda_slow_pred = 2*mean(lambda/sigma) (preconditioned curvature).
Regress r(t) on K(t)=droprelaxS(etas, lambda_slow) THROUGH ORIGIN (kappa = sum(r*K)/sum(K*K)); R2.
Try a small grid of lambda_slow around lambda_slow_pred to find best R2; report both.
dL_eq/deta: numeric derivative of nqm_linear_Leq at eta_pk. Compare kappa to (c*eta_pk*dLeq_deta) -> back out c.
EXPECT: wsd R2 high (>0.7), cosine residual MUCH smaller (max|r_cosine| << max|r_wsd|), c in (0,1].
matches_paper if wsd_R2>0.6 and cosine_maxabs < 0.5*wsd_maxabs. Save results/E4.json and a figure figs/E4.png if easy.`,
  },
  {
    key: 'E5_twoexp',
    title: 'E5: two-exponential beats one (spectral mixture)',
    task: `EXPERIMENT E5 -- prediction (v): with a SPREAD of curvature rates lambda/sigma, a 2-exponential kernel
beats 1-exponential. Use a spectrum with WIDE spread: lambdas=geomspace(0.1, 10, 16), sigmas=ones(16)
so lambda/sigma spans 100x. Run a wsd schedule (as in E4). Residual r = L_true - L_eq.
Fit 1-exp: best over grid of lambda_slow -> R2_1. Fit 2-exp: droprelaxS_twoexp(etas, lam1, lam2, w1),
optimize (lam1,lam2,w1, amplitude) by least squares (small grid + linear amplitude, or scipy) -> R2_2.
Report R2_1, R2_2, delta=R2_2-R2_1. PAPER: 2-exp improves R2 by 0.06-0.07.
matches_paper if delta > 0.03 and R2_2 >= R2_1. Save results/E5.json.`,
  },
  {
    key: 'E6_scale_inv',
    title: 'E6: scale invariance of lambda_slow',
    task: `EXPERIMENT E6 -- lambda_slow = 2(lambda/sigma)_eff is a preconditioned curvature, so it should be roughly
INVARIANT across "scales" if the (lambda/sigma) distribution is preserved, even when the OVERALL loss
magnitude, number of modes, or noise amplitude changes. Build 4 configs that mimic "different model sizes":
 cfgA: lambdas=geomspace(0.5,5,8), sigmas=1.0
 cfgB: same lambdas, sigmas=2.0, AND lambdas scaled so lambda/sigma ratio distribution is the SAME as A
       (i.e. lambdas_B = lambdas_A * 2 to keep lambda/sigma constant) -- tests amplitude change, same ratio.
 cfgC: more modes d=16 with the same geomspace ratio range, sigmas=1.0.
 cfgD: shift overall curvature by 0.5x with sigma 0.5x (ratio preserved).
For each: run a wsd schedule, residual, fit best lambda_slow (grid) by max R2. Report lambda_slow per config
and CV across configs. EXPECT CV small (<~0.25) -> lambda_slow ~ invariant when lambda/sigma preserved,
and that lambda_slow tracks 2*mean(lambda/sigma). Save results/E6.json. matches_paper if CV<0.3.`,
  },
]

phase('Experiments')
log(`Engine ready. Fanning out ${EXPERIMENTS.length} NQM experiments...`)

const expResults = await parallel(EXPERIMENTS.map(e => () =>
  agent(
    `${BRIEF}\n${ENGINE}\n\n${e.task}\n\nWrite a clean python script ${ROOT}/repro/${e.key}.py that imports the engine,
runs the experiment, prints a clear summary, and saves results/<...>.json. Run it with:
  cd ${ROOT} && python repro/${e.key}.py
Iterate until it runs cleanly and produces sane numbers (no NaNs, fits with r2>~0.9 where expected).
Then return the structured result. Keep runs fast (n_rep<=4000, steps<=5000).`,
    { label: e.key, phase: 'Experiments', schema: PEXP }
  ).then(r => ({ ...e, result: r }))
))

phase('Verify')
const verified = await parallel(expResults.filter(Boolean).map(er => () =>
  agent(
    `${BRIEF}\n\nADVERSARIAL VERIFICATION of experiment ${er.key} (${er.title}).
The experiment script is at ${ROOT}/repro/${er.key}.py and its results JSON is under ${ROOT}/results/.
READ both files. Your job is to REFUTE, not rubber-stamp. Check for:
- circularity (e.g. using the same linear approximation to BOTH generate the ground truth AND fit it,
  which would trivially confirm the theory -- the ground truth MUST come from the real adamw_nqm Monte-Carlo
  simulation, NOT from nqm_linear_Leq);
- power-law fits driven by 1-2 outlier points or by points at the edge of stability;
- p estimated from too few/too-narrow eta range;
- R2 inflated by including the trivial stable phase (lots of near-zero residual points);
- cherry-picked lambda_slow;
- NaNs / failed fits hidden.
Reported result: ${JSON.stringify(er.result).slice(0, 1500)}
Re-run a quick spot check if useful (cd ${ROOT} && python ...). Give a candid verdict: does the evidence
genuinely support the paper's corresponding claim? Default to supports_claim=false if you find a real bug
or circularity.`,
    { label: 'verify:' + er.key, phase: 'Verify', schema: PVERDICT }
  ).then(v => ({ key: er.key, title: er.title, result: er.result, verdict: v }))
))

phase('Synthesize')
const synth = await agent(
  `${BRIEF}\n\nYou are the synthesis lead. Here are the experiment results and adversarial verdicts:
${JSON.stringify(verified.filter(Boolean).map(v => ({ key: v.key, title: v.title, our_value: v.result?.our_value, paper_value: v.result?.paper_value, matches: v.result?.matches_paper, verdict: v.verdict })), null, 1)}

Write a concise markdown reproduction table to ${ROOT}/results/NQM_REPORT.md with columns:
Claim | Paper value | Our value | Reproduced? (Y/N/partial) | Verifier confidence | Notes.
Cover E1..E6. Clearly separate REPRODUCTIONS (E1,E2,E4,E5,E6 mirror the paper) from the GENERALIZATION
(E3: the p=1 vs p=2/3 prediction the paper did not test). End with a 5-sentence bottom line on whether the
noisy-quadratic mechanism behind the paper holds up, and where it is shakiest.
Return the markdown text.`,
  { label: 'synthesize', phase: 'Synthesize' }
)

return { verified: verified.filter(Boolean).map(v => ({ key: v.key, matches: v.result?.matches_paper, our: v.result?.our_value, supports: v.verdict?.supports_claim, conf: v.verdict?.confidence })), report_path: `${ROOT}/results/NQM_REPORT.md`, synth_preview: synth.slice(0, 600) }
