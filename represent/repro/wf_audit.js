export const meta = {
  name: 'audit-real-curve',
  description: 'Independent adversarial audit of the real-transformer (Part C) code + results before finalizing',
  phases: [
    { title: 'Audit', detail: '4 agents audit engine, analysis methodology, results, and the sign story' },
    { title: 'Synthesize', detail: 'compile issues + required caveats' },
  ],
}
const ROOT = 'c:/Users/21100/Desktop/represent'

const CTX = `
We reproduced "Learning-Rate Schedules Are Not Adiabatic" (Jiaju Wu). Part C trains a real ~10M
byte-level transformer (AdamW b1=.9 b2=.95) under LR schedules and tests the paper's claims:
 - non-adiabatic positive residual on FAST decays (loss lags equilibrium above a well-fit MPL),
 - residual = DropRelaxS kernel K(t)=sum exp(-lambda_slow (S_t-S_t')) drop_t',
 - tau ~ 1/eta from wsdcon two-stage probes.
Files (READ them): ${ROOT}/repro/engine.py (MPL law mpl_loss/mpl_loss_at with S-binning, droprelaxS,
 measure_tau, NQM sim), ${ROOT}/repro/analyze_curves.py (load_scale w/ T_MIN warmup-exclusion +
 smooth_by_step, fit_mpl, residual, fit_droprelaxS, fair_baseline_test, tau_vs_eta),
 ${ROOT}/repro/analyze_show.py (showcase analysis), ${ROOT}/repro/train_showcase.py.
Results: ${ROOT}/results/SHOWCASE_REPORT.json, ${ROOT}/results/analyze_show2.log, ${ROOT}/results/curves_show/*.csv.
Key showcase numbers (better MPL fit on constant+cosine+wsdcon_20, obj 0.00238): residual end-lag
 cosine -0.0074 / wsd_grad -0.0003 / wsd_sharp +0.0104 (sharp = only clearly positive & largest);
 DropRelaxS R2(wsd_sharp)=0.749 kappa>0; fair baseline finite-lambda R2=0.439 vs cumulative-drop
 R2=0.153 (advantage +0.285); wsdcon post-drop relaxation amp ~0.05; tau roughly constant ~850 steps
 (window-truncated for small eta in the showcase; a long-window rerun is in progress).
You run python with: cd ${ROOT} && KMP_DUPLICATE_LIB_OK=TRUE python -c "..."  (engine/analyze import numpy+scipy).
`
const V = {
  type: 'object', additionalProperties: true,
  properties: {
    area: { type: 'string' },
    bugs_found: { type: 'array', items: { type: 'string' } },
    artifacts_or_circularity: { type: 'array', items: { type: 'string' } },
    claims_supported: { type: 'boolean' },
    required_caveats: { type: 'array', items: { type: 'string' } },
    severity: { type: 'string', enum: ['none', 'minor', 'major', 'critical'] },
    verdict: { type: 'string' },
  },
  required: ['area', 'claims_supported', 'severity', 'verdict', 'required_caveats'],
}
const TASKS = [
  { key: 'engine', t: `AUDIT engine.py NUMERICS. Verify: (1) droprelaxS recursion correctly computes sum_{t'<=t} exp(-lambda_slow*(S_t-S_t'))*drop_t' (check vs a brute-force double loop on a small schedule); (2) mpl_loss_at S-binning (_binned_decrements) matches the unbinned mpl_loss within tolerance on cosine (does binning to 600 introduce error?); (3) measure_tau with an explicit t-axis fits tau in correct units; (4) the NQM adamw_nqm is a real AdamW (m,v EMAs, bias correction) not the linear approximation. Write small numeric checks and run them. Report any bug.` },
  { key: 'methodology', t: `AUDIT analyze_curves.py + analyze_show.py METHODOLOGY for circularity/leakage/artifacts. Check: (1) is the residual = L_true - L_MPL where MPL is fit on a TRAIN set NOT containing the held-out test curve? (2) does smooth_by_step (80-step window) wash out or fabricate the relaxation signal? test sensitivity to window=40/80/160; (3) does fair_baseline_test correctly compare finite-lambda vs cumulative-drop (lambda->0) with a single shared amplitude through the origin? (4) does tau_vs_eta / backbone-aware tau avoid using the same model to generate and fit? (5) is T_MIN warmup-exclusion hiding anything? Report artifacts.` },
  { key: 'results', t: `AUDIT the showcase RESULTS vs claims. Read SHOWCASE_REPORT.json + analyze_show2.log + curves_show. Independently RE-CHECK: is wsd_sharp's residual really positive & largest (recompute end-lag yourself)? Is DropRelaxS R2=0.749 inflated by the flat stable phase or by few points? Re-fit with different lambda grids. Is the fair-baseline +0.285 advantage robust? Are the wsdcon relaxation amplitudes (~0.05) real? Be adversarial: could these be MPL-misfit rather than the non-adiabatic lag?` },
  { key: 'sign_and_tau', t: `AUDIT the SIGN story and the tau result. We claim: the residual sign flipped from negative (MPL fit on cosine-only) to positive (MPL fit incl. wsdcon_20) because cosine-only MPL over-extrapolates the long-high-LR backbone. Verify this is a real baseline effect, not cherry-picking the fit set to get the desired sign. Independently fit MPL on a few different train sets and see how the wsd_sharp residual sign/magnitude moves. Also assess: is the claim "tau ~ constant in the showcase is a window-truncation artifact (true tau for small eta exceeds the 3500-step window)" justified? Estimate the expected tau=1/(lambda_slow*eta) from the fitted lambda_slow and compare to the window.` },
]
phase('Audit')
const aud = await parallel(TASKS.map(x => () =>
  agent(`${CTX}\n\nYOUR AUDIT (${x.key}):\n${x.t}\nBe a skeptic. Run real checks. Return structured findings.`,
    { label: 'audit:' + x.key, phase: 'Audit', schema: V }).then(r => ({ ...x, r }))))
phase('Synthesize')
const s = await agent(`${CTX}\n\nAudit findings: ${JSON.stringify(aud.filter(Boolean).map(a => ({ area: a.key, ...a.r })), null, 1)}
Write ${ROOT}/results/AUDIT_PARTC.md: a prioritized list of (a) any real BUGS to fix, (b) artifacts/circularity, (c) caveats that MUST appear in the Part C writeup for it to be honest, and (d) a one-line overall verdict on whether the Part C real-transformer evidence genuinely supports "the non-adiabatic lag appears in a real transformer (qualitatively), with tau~1/eta unconfirmed at this scale." Return the markdown.`,
  { label: 'synth', phase: 'Synthesize' })
return { audits: aud.filter(Boolean).map(a => ({ area: a.key, severity: a.r?.severity, supported: a.r?.claims_supported, bugs: (a.r?.bugs_found || []).length })), report: `${ROOT}/results/AUDIT_PARTC.md`, preview: s.slice(0, 800) }
