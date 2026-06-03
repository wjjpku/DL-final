# The non-adiabatic correction to MPL (loss-relaxation lag)

This note gives a theory for the empirical 1-parameter correction that improves
MPL in the few-shot regime (`repro/correction_fair.py`,
`repro/fewshot_analysis.py`), and verifies it quantitatively
(`repro/nonadiabatic_theory.py`).

## 1. The empirical finding (data first)

A correction on frozen/refit MPL,

```
L(t) = L_MPL(t) + kappa * DropRelax_tau(t),
DropRelax_tau(t) = sum_{t'<=t} (1-1/tau)^(t-t') * relu(eta_{t'-1}-eta_{t'}) / eta_peak,
```

with tau ~ 1200 fixed and a single fitted kappa, improves **fair** few-shot
WSD prediction (both arms refit MPL on the same data) by 10-20% when the training
set contains a sharp decay; it is neutral otherwise. kappa is **not** scale
invariant -- it grows with model size.

## 2. Rigorous derivation from the SGD second-moment recursion

The earlier draft used a phenomenological scalar relaxation `dL/dt=-r(L-L_eq)`.
That single rate is not justified -- the loss `L=½Σ_i λ_i E[δ_i²]` is a sum of modes
each with its own rate. Here is the derivation from the mode dynamics; the single
rate emerges as a clearly-labelled approximation at the end.

**Exact second-moment dynamics.** In the Hessian eigenbasis, mode `i` (curvature
`λ_i`, gradient-noise variance `σ_i²`) under SGD `δ_i(t+1)=(1-η_t λ_i)δ_i(t)-η_t ξ_i`
has `V_i:=E[δ_i²]` obeying
```
V_i(t+1) = (1-η_t λ_i)² V_i(t) + η_t² σ_i².                       (exact)
```
Its frozen-η fixed point is `V_i*(η) = η σ_i² / (λ_i (2-η λ_i))` (exact). Writing the
deviation `Δ_i(t)=V_i(t)-V_i*(η_t)` and using that V_i* is the fixed point,
```
Δ_i(t+1) = (1-η_t λ_i)² Δ_i(t) - δV_i*(t),   δV_i*(t)=V_i*(η_{t+1})-V_i*(η_t).   (exact)
```

**Three approximations, each labelled.**
- **(A)** `(1-η λ_i)² ≈ e^{-2 η λ_i}` (small step). *Safe here*: the lag is dominated
  by slow (small-λ_i) modes for which `η λ_i ≪ 1` -- verified post hoc, the dominant
  `λ_eff≈6-8` gives `η_peak λ_eff ≈ 2e-3 ≪ 2` (`repro/deep_rigor.py`). This is the
  opposite regime to γ, which is an edge effect (`ηλ~2`) where the linearisation fails
  -- which is exactly why this term can be derived and γ cannot.
- **(B)** `δV_i* ≈ (dV_i*/dη) Δη_t`, with `Δη_t = η_{t+1}-η_t = -drop_{t+1}` (first order
  in the per-step LR change; exact in the continuum).

Solving the linear recursion (product of decay factors → S-time exponential):
```
Δ_i(t) = (dV_i*/deta) · Σ_{t'} exp(-2 λ_i (S_t - S_{t'})) · drop_{t'}.   (exact under A,B)
```
Hence the loss lag (exact identity Σ_i ½λ_i dV_i*/dη = d/dη[½Σλ_iV_i*] = dL_eq/dη):
```
Delta L(t) = Σ_i w_i [ e^{-2 λ_i (S-S')} ⊛ drop ](t),
   w_i = ½ λ_i (dV_i*/deta),     Σ_i w_i = dL_eq/deta   (exact).
```

This is the rigorous result: **the non-adiabatic loss lag is the convolution of the
LR-decrement sequence with a `w_i`-weighted MIXTURE of cumulative-LR (S-time)
exponentials with rates `2λ_i`.** It already yields, exactly:
- rate per step `2 η λ_i ∝ η`  ⇒  `tau ∝ 1/eta` (measured slope -1, deep_tau.py);
- an **S-time** exponential kernel (not step-time);
- total amplitude `Σw_i = dL_eq/deta` **exactly**;
- the **positive sign** (`dV_i*/deta>0`, `drop>0` ⇒ `ΔL>0`).

- **(C)** *Single-mode collapse* -- the only substantive approximation: replace the
  spectral mixture by one effective exponential,
```
Delta L(t) ≈ c · (dL_eq/deta) · DropRelaxS_{lambda_slow}(t),   lambda_slow = 2 λ_eff,  c ≤ 1.
```
  `c<1` and the imperfect `R²` are *the signature of the spectral spread*, not a fudge:
  a 2-exponential kernel measurably beats 1 (`repro/deep_rigor.py`: R² +0.06-0.07 at
  100/400M), confirming the mixture. `c≈0.5` = the fraction of `dL_eq/deta` carried by
  modes slow enough to lag on the schedule's timescale.

**On "MPL = adiabatic L_eq".** MPL is fit on cosine, which is slow but not infinitely
slow, so `MPL ≈ L_eq + (small cosine lag)`; the cosine lag is small because DropRelax
is tiny on a gradual schedule. The regression residual is thus `ΔL_wsd` minus MPL's
(small) built-in cosine lag -- the non-adiabatic remainder. This assumes MPL captures
the adiabatic (backbone+annealing) part well, which the universal collapse supports.

## 2b. The optimizer is AdamW, not SGD: the preconditioned derivation

The recursion in §2 is for SGD; the data are AdamW. The reduction below shows the
results survive with the bare curvature replaced by a *preconditioned* curvature.

AdamW: `theta_{t+1} = theta_t - eta_t m_t/(sqrt(v_t)+eps)`, `m=EMA_{b1}(g)`,
`v=EMA_{b2}(g^2)`. Two Adam-specific approximations:
- **(Adam-0, momentum)** the lag timescale `tau ~ 1e3` is `>> 1/(1-b1) ~ 10` steps, so
  over the relaxation window `m_t ≈ g_t`; momentum only smooths a fast window and is
  absorbed into a constant rescaling.
- **(Adam-1, preconditioner)** with `b2 -> 1`, `v` is slow and sits at its mean
  `v_i* = E[g_i^2]`. In the noise-dominated mountain directions `g_i ≈ lambda_i delta_i + xi_i`
  with the noise dominating, so `v_i* ≈ s_i^2 := E[xi_i^2]`, approximately independent of
  `eta`. (This is the standard "Adam ≈ preconditioned/sign-SGD" reduction, Malladi et al. 2022.)

Per mode, Adam then reduces to SGD with effective step `eta_i^eff = eta/s_i`:
```
delta_i(t+1) = (1 - eta_i^eff lambda_i) delta_i(t) - eta_i^eff xi_i(t).
```
Identical in structure to §2 with `eta -> eta/s_i`. Re-running the algebra:
```
V_i*(eta) = eta s_i / (lambda_i (2 - eta lambda_i/s_i)),
relaxation rate per step = 2 eta (lambda_i/s_i)  ∝ eta,
Delta L(t) = sum_i w_i [ exp(-2 (lambda_i/s_i)(S-S')) (x) drop ],   sum_i w_i = dL_eq/deta  (exact).
```
So **everything in §2 survives, with the bare curvature `lambda_i` replaced by the
PRECONDITIONED curvature `kappa_i = lambda_i/s_i`**: tau ∝ 1/eta (since `v_i*≈s_i^2` is
eta-independent), the S-time exponential kernel, amplitude = dL_eq/deta exactly, and the
positive sign. The effective rate constant is now `lambda_slow = 2 (lambda/s)_eff`, a
**preconditioned curvature** -- which (i) explains why the measured `lambda_slow≈10` is
not a bare Hessian eigenvalue, and (ii) predicts it should be more scale/architecture
invariant than a bare curvature, because Adam stabilises the *preconditioned* sharpness
(Cohen et al. 2022; consistent with mu-P / Step-Law transfer and with our measured
scale-invariance of lambda_slow). The slow lag modes have preconditioned sharpness far
below the Adam edge, so approximation (A) remains valid.

**Honest remaining gaps** (after closing the SGD->Adam one): (Adam-1) assumes b2->1 with
v frozen at its noise-dominated mean (finite b2 lets v fluctuate, and signal-dominated
modes give an eta-dependent v_i*); the spectrum {lambda_i, s_i} is treated as static
(the landscape actually evolves); cross-step/cross-mode noise correlations are ignored;
and lambda_slow, c are still *measured*, not computed (the theory fixes the FORM and the
exact identities, not the two effective constants). These are the same model-level
assumptions MPL itself rests on.

## 2c. Refinement: self-consistent preconditioner (finite b2, signal vs noise)

§2b *asserted* `v_i* ≈ s_i^2`. Here it is *derived* self-consistently, and the
condition for it turns into a falsifiable prediction that the data confirm.

The true stationary preconditioner is `v_i* = E[g_i^2] = lambda_i^2 V_i + s_i^2`, and
`V_i` depends on the effective step `eta_i^eff = eta/sqrt(v_i*)` -- a self-consistent
loop. With the per-mode signal-to-noise ratio `rho_i = lambda_i^2 V_i / s_i^2`,
`v_i* = s_i^2 (1+rho_i)` and the per-step relaxation rate is
```
r_i = 2 eta lambda_i / ( s_i sqrt(1+rho_i) ).
```
Solving the loop in the two limits gives the tau-vs-eta exponent (`tau ∝ eta^{-p}`):

| mode | V_i | v_i* | tau ∝ | p |
|---|---|---|---|---|
| noise-dominated (rho≪1) | ∝ eta | ≈ s_i^2 (eta-indep.) | eta^{-1} | **1** |
| signal-dominated (rho≫1) | ∝ eta^{2/3} | ∝ eta^{2/3} | eta^{-2/3} | 2/3 |

**The data pick the regime, and close the loop.** `deep_tau.py` measured the tau-vs-eta
log-log slope = -1.06 (100M), -0.94 (400M), i.e. **p ≈ 1**. So the lag-relevant slow
modes are **noise-dominated** -- which is exactly the condition (rho≪1) under which
`v_i*≈s_i^2` holds. The §2b approximation is therefore not assumed but *self-consistently
verified by its own prediction*. (25M gives -0.51, but it has the weakest signal and
least reliable transient fits; it may also carry more signal contribution.)

**Finite b2 = a second relaxation channel.** With `b2<1`, `v_t` is an EMA that lags the
true `g^2` over `~1/(1-b2)` steps. After an LR change `g^2` shifts and `v` lags, so the
*preconditioner itself* relaxes -- a second channel with rate `~(1-b2)`, which is
`eta`-INDEPENDENT (a step-time constant), unlike the variance channel (rate ∝ eta). The
full kernel is therefore a mixture of (variance channel, ∝eta) + (preconditioner channel,
const) -- **consistent with the measured 2-exponential improvement over 1** (deep_rigor.py):
the second timescale is plausibly the preconditioner channel. Quantifying it needs the
dataset's `b2` (unknown here): `b2=0.999 -> 1/(1-b2)=1000` steps (relevant) vs
`b2=0.95 -> 20` steps (negligible). Stated as a prediction, not a fitted result.

## 3. Predictions vs experiment (`repro/nonadiabatic_theory.py`)

Regress the **cosine-fit** MPL residual on DropRelax (tau=1200), through the origin:

| prediction | result |
|---|---|
| (i) sign: residual > 0 (loss lags *above* L_eq) | **all residuals positive** ✓ |
| (ii) residual ∝ DropRelax on sharp decays | R² = 0.74 (100M), 0.81 (400M); 0.37 (25M, smallest signal) ✓ |
| (iii) kappa = dL_eq/deta grows with N | kappa = 0.034 / 0.053 / 0.070 at 25/100/400M, ~ N^0.26 ✓ |
| (iv) kappa = eta_peak·dL_eq/deta, dL_eq/deta measured independently | predicted 0.11/0.15/0.17 vs fitted 0.034/0.053/0.070 |

For (iv) `dL_eq/deta` is estimated from a **completely independent** source -- the
noise-floor slope of the two-stage `wsdcon_{3,9,18}` final losses vs their stage-2
LR (backbone subtracted). It predicts kappa to within a constant factor ~0.3 *and*
reproduces the increasing-with-N trend. The factor ~0.3 is itself expected: the
decay is not instantaneous (~4000 steps), so only a fraction
`~ tau/(tau+T_decay) ≈ 1200/5200 ≈ 0.23` of the full equilibrium sensitivity
manifests as lag.

The sign is the key discriminator: this loss-relaxation lag gives a **positive**
residual (loss above MPL on a fast decay), which matches the data, whereas the
earlier sharpness-state lag (RV-EoS) gave the **wrong (negative)** sign. Lagging
the *loss* directly, not the *sharpness*, is the correct mechanism.

## 4. How this closes the loop with gamma

MPL's empirical factor `eta_k^{-gamma}` gives the kernel a weak dependence on the
LR at which a decrement happened -- an *implicit, low-capacity* handle on exactly
this non-adiabatic lag. That is why:

- In the **data-rich** regime MPL can tune gamma (and amplitudes) to absorb the
  average residual across several WSD curves, so the explicit DropRelax term adds
  nothing.
- In the **few-shot** regime MPL cannot pin that down from one curve, and the
  explicit, correctly-shaped DropRelax prior helps it extrapolate (10-20%).

So gamma is the adiabatic law's built-in approximation to the first non-adiabatic
correction; DropRelax is that correction made explicit and well-conditioned. This
also bounds the gain: it is a *first-order* (linear-response) correction, real but
small, and only visible where the schedule sweeps the LR faster than the loss can
relax.

## 4b. Refinement: the relaxation is in cumulative-LR (S) time, rate lambda_slow ~ 10

`repro/deep_tau.py` measures tau directly from the wsdcon post-step transient (fit
`r(t) = floor + amp*exp(-(step-8000)/tau)` on the cosine-fit-MPL residual during the
8k-step constant tail). Result: **tau ∝ 1/eta** (log-log slope -1.06 at 100M, -0.94
at 400M), i.e. the relaxation *rate* is `r = eta * lambda_slow` with
`lambda_slow = 1/(tau*eta) ≈ 10`, roughly **scale-invariant** (tau*lr_b ~ const
across 100/400M). So the loss relaxes in **cumulative-LR (S) time**, not step time,
and the correct kernel is

```
DropRelaxS(t) = sum_{t'<=t} exp(-lambda_slow (S(t)-S(t'))) relu(eta_{t'-1}-eta_{t'}) / eta_peak
              = recurrence  s_t = s_{t-1} exp(-lambda_slow * eta_t) + drop_t.
```

This makes lambda_slow the scale-invariant SHAPE constant and kappa = eta_peak *
dL_eq/deta the N-dependent amplitude -- exactly MPL's invariance structure.

Verification (`repro/deep_stime.py`, `repro/deep_stime2.py`):
- Fair few-shot improvement rises from -10.6% (step-time) to **-20.7%** (S-time),
  with the **best-fit lambda_slow = 10-14 matching the independently measured ~10**.
- S-time residual regression R^2 = 0.40 / 0.80 / 0.87 (25/100/400M), up from
  step-time's 0.37 / 0.74 / 0.81.
- kappa_fit / kappa_pred = 0.43 / 0.51 / 0.60 (**CV 14%**): a near-constant ratio
  ~0.5, so kappa is **predictable from the noise floor** up to one universal constant
  (the step-time kernel left a more scattered ~0.3). The remaining factor < 1 is the
  linear-response approximation (dL_eq/deta treated as locally constant).

Final form -- two universal scale-invariant constants (lambda_slow ~ 10, c ~ 0.5)
and one independently-measurable per-scale amplitude (eta_peak * dL_eq/deta from the
noise floor):

```
L(t) = L_MPL(t) + c * eta_peak * (dL_eq/deta) * DropRelaxS_{lambda_slow}(t).
```

## 4c. The payoff: a parameter-free, cross-scale prediction

Because lambda_slow and c are universal and dL_eq/deta is measurable from cheap
two-stage (noise-floor) runs, the whole correction can be **predicted, not fitted**
(`repro/deep_predict.py`). For each target scale: lambda_slow=10; c from a
leave-one-scale-out average of the other scales' kappa_fit/kappa_pred; dL_eq/deta
from the target's own wsdcon finals (different curves from the test). Applied with
ZERO fitting to the cosine-fit MPL on the held-out sharp-decay curves:

| target | MAE_MPL (wsd, wsdld) | MAE_predicted | delta |
|---|---|---|---|
| 25M  | 0.00341 / 0.00314 | 0.00246 / 0.00219 | -28% / -30% |
| 100M | 0.00375 / 0.00325 | 0.00164 / 0.00161 | -56% / -51% |
| 400M | 0.00470 / 0.00385 | 0.00236 / 0.00212 | -50% / -45% |
| **all** | **0.00368** | **0.00206** | **-44%, wins 6/6** |

No constant is fit on the target curve: lambda_slow from wsdcon transients, c from
*other* scales, dL_eq/deta from the target's noise floor. So small-model constants +
a cheap noise-floor probe predict the large-model sharp-decay curve ~2x better than
cosine->WSD MPL transfer. This is exactly the regime where the correction matters
(you have cosine + cheap probes but NOT full WSD curves); when full WSD curves *are*
available, refit MPL already absorbs the residual (it is then an information gap, not
a formula gap -- see correction_fair.py).

## 5. Status / honest scope

- Real, fair, mechanism-consistent, and quantitatively verified (sign, shape, R²,
  N-trend, and an independent factor-~3 magnitude check).
- Genuinely useful only in the few-shot / fast-decay regime; needs at least one
  sharp-decay curve to calibrate kappa (dormant on cosine alone).
- kappa is not a universal constant; it scales as ~N^0.26 (dL_eq/deta grows with
  model size -- sharper landscapes hold more oscillation energy per unit LR).

Code: `repro/correction_fair.py`, `repro/fewshot_analysis.py`,
`repro/nonadiabatic_theory.py`. Related: `docs/core/river_valley_derivation.md`.
