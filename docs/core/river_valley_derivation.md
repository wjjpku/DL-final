# River-Valley EoS derivation of the loss law (and the true origin of γ)

This note corrects an earlier mistake (applying the **gradient-descent** edge of
stability, sharpness ≈ 2/η, to **AdamW** pretraining data) and re-derives the
loss law from the mechanism that actually governs this regime.

## 1. What EoS really is here — mechanism and timing

Two facts from the literature, both essential and both ignored in the first
attempt:

1. **Adaptive EoS, not 2/η.** For full-batch / large-batch *adaptive* methods,
   it is the *preconditioned* sharpness (max eigenvalue of `P^{-1}H`) that
   equilibrates at an optimiser-dependent threshold, ≈ 38/η for Adam's default
   β's — **not** the raw sharpness at 2/η (Cohen et al. 2022,
   *Adaptive Gradient Methods at the Edge of Stability*, arXiv:2207.14484).
   For AdamW the raw sharpness keeps drifting while the preconditioned one
   self-stabilises.

2. **Sharpening takes time, and it is *non-monotone in the schedule*.** The
   sharpness is not instantaneously pinned to the edge. It relaxes there by
   *progressive sharpening* over many steps; for Adam/transformers the
   preconditioned sharpness actually starts high and *decreases* early (a
   catapult/“sharpness reduction” regime; Kalra & Barkeshli 2024,
   arXiv:2406.09405), and sharpness *increases again along the iterates while
   the LR is annealed* (Wen et al. 2024; Belloni et al. 2026).

So "ηλ ≈ 2 at η_peak" is wrong twice over: wrong threshold (adaptive, not 2),
and wrong assumption that it holds *instantaneously and everywhere*. The
**timing** of sharpening is the crux.

## 2. The river-valley picture (Wen et al. 2024, arXiv:2410.05192)

The pretraining loss landscape is a **river valley**: a flat low-curvature
**river** direction carrying genuine progress, and sharp high-curvature
**mountain** directions transverse to it. Decompose the observed loss as

```
L_obs(t) = L_river(S(t))  +  P_osc(t)
```

- `L_river` — the river-bottom loss = true accumulated progress (the *bias*
  term). It depends on how far along the river we have travelled, i.e. on the
  cumulative LR `S(t)=Σ η_i`. Keep the MPL/Tissue backbone `L0 + A S^{-α}`.
- `P_osc` — the **oscillation penalty**: how far above the river bottom the
  iterate sits because the high LR makes it bounce across the mountains.

This directly matches the WSD phenomenology: stable phase keeps loss *elevated*
(`P_osc` large), the decay *reveals accumulated progress* (`P_osc → 0`).

## 3. Deriving `P_osc` exactly (SGD on a quadratic mountain)

Model one mountain mode as a quadratic of curvature `H`, optimised by discrete
SGD with step `η` and isotropic gradient noise of variance `σ²`:

```
x_{t+1} = (1 - η H) x_t + η ξ_t,   Var(ξ)=σ².
```

Stationary variance (geometric sum of the linear recurrence):

```
Var(x) = η²σ² / (1 - (1-ηH)²) = η σ² / ( H (2 - η H) ).
```

Excess loss above the bottom = `½ H Var(x)`:

```
        ┌─────────────────────────────────┐
        │  P_osc(η,H) = (σ²/2) · η/(2 - ηH) │
        └─────────────────────────────────┘
```

Two limits fall straight out of the formula — no extra assumptions:

- **Edge of stability** `ηH → 2`  ⇒  `P_osc → ∞`.  The penalty diverges at the
  stability boundary; this *is* why the stable phase loss is elevated, and why
  the system self-stabilises just below the edge.
- **Annealing** `η → 0`  ⇒  `P_osc → 0`.  The decay removes the penalty and the
  loss falls to the river bottom — "revealing accumulated progress".

## 4. The timescale of sharpening = the origin of γ

`H` is a *state variable*, not a constant. Progressive sharpening relaxes it
toward the adaptive-EoS target `u*/η` (the edge value, `u* < 2`) with a
timescale `τ` in steps:

```
H_{t+1} = H_t + (1/τ) ( u*/η_t − H_t ).
```

Now read off the two regimes:

- **Instant tracking** (`τ → 0`): `η_t H_t ≡ u*` always, so
  `P_osc = (σ²/2) η_t /(2−u*) ∝ η_t`. The penalty is a pure **instantaneous
  floor**, a function of the *current* LR only — *no history, no γ*. This is
  exactly the `γ=0` limit of MPL, and exactly the floor term `D·f(η)` that the
  earlier experiment falsified (`D` came out negative / confounded). Good: the
  theory explains *why* that simple floor had to fail.

- **Finite lag** (`τ > 0`): a *sharp* LR drop outruns `H`. During a fast WSD
  cooldown `η` collapses while `H` is still near its stable-phase value, so the
  edge ratio `η_t H_t` swings away from `u*` and `P_osc` is **no longer ∝ η_t`.
  The penalty now depends on the *history* of the schedule through the lagged
  `H_t`. A *gradual* cosine decay keeps `H` on the edge (`η_t H_t ≈ u*`
  throughout) → penalty stays ∝ η → small residual. A sharp WSD decay does not
  → residual. This history dependence — same current η, different residual
  depending on how you got there — is precisely the behaviour MPL absorbs into
  its empirical `η_k^{-γ}` kernel-speed factor.

**Claim.** MPL's mysterious exponent γ is the *effective signature of the
progressive-sharpening lag* `τ`. It is not a free spectral exponent; it is what
you get when you compress the (η_t, H_t) two-variable dynamics into a single
history kernel on cumulative progress.

## 5. The resulting law (RV-EoS), 7 parameters, all physical

```
L(t) = L0 + A S(t)^{-α}  +  (B/2) · η_t / (2 − η_t H_t),
H_{t+1} = H_t + (1/τ)(u*/η_t − H_t),     H_0 given.
```

| param | meaning | physical sanity check |
|-------|---------|------------------------|
| L0,A,α | river/bias backbone | same as MPL/Tissue |
| B = σ² | gradient-noise power × 2 | > 0 |
| u*     | adaptive-EoS edge ratio | **must fit < 2** |
| τ      | progressive-sharpening timescale (steps) | **must fit > 0**, O(10²–10³) |
| H0     | sharpness at end of warmup | ≈ u*/η_peak if warmup reached edge |

Same parameter count as MPL (7), but every parameter is a named dynamical
quantity with a falsifiable range, and the edge `ηH=2` is built in rather than
fitted around.

## 6. Falsifiable predictions (decided *before* fitting)

1. The cosine→WSD transfer fit must give **`u* < 2` and `τ > 0`**; otherwise the
   mechanism is wrong (honest kill criterion).
2. `u*`, `τ` (dynamical) should be **scale-invariant** across 25/100/400M, like
   MPL's exponents, while `L0, A, B` (amplitudes) may drift — same invariance
   structure we already established for MPL.
3. Setting `τ→0` (instant tracking) must *degrade* to the failed floor model;
   the improvement over that floor is the empirical content of the lag.

## 7. Experimental result — an honest negative

`repro/river_valley.py`, cosine→WSD transfer, 25/100/400M, vs the published MPL
fit (`results/river_valley.log`):

| | test mean MAE | vs MPL |
|---|---|---|
| RV-EoS (this law) | 0.00968 | — |
| MPL | 0.00409 | RV is **2.4× worse, wins 0/15 curves** |

The falsifiable predictions of §6 **fail**:

- `u*` rails to its upper bound **1.99 at all three scales** — the fit wants to
  sit *on* the edge, but doing so does not yield a good law. A single static
  edge ratio is too crude (this is exactly the static "ηλ≈2" assumption being
  rejected by the data).
- `τ` is **not scale-invariant** (CV 76%) and **collapses to the floor limit
  τ=1 at 400M**. The progressive-sharpening lag, modelled as relaxation toward
  `u*/η`, is not a transferable closure.
- RV fits cosine *better* than MPL in-sample (train huber 0.00034 vs 0.0011) but
  extrapolates to WSD *worse* — the classic signature of overfitting the
  training schedule with extra flexibility.

**Conclusion.** The river-valley mechanism is qualitatively correct and fixes
the EoS misconception, and it *explains* γ (the sharpening lag) and *why the
naive γ=0 floor must fail*. But unrolling the (η,H) dynamics explicitly is a
**worse** quantitative description than MPL's compressed γ-kernel. MPL's γ is
therefore not redundant: it compresses the sharpness-lag dynamics into a single
transferable history exponent more robustly than the explicit ODE does. This is
evidence *for* MPL, and a mechanistic account of γ — not a replacement that
beats it.

## 8. A constrained follow-up: sharpness-lag correction on frozen MPL

To avoid the added-flexibility trap of §7, attempt #2 *freezes* the published
MPL and adds only a 2-parameter correction `ΔL = κ (tilde_eta − eta)_+`, with
`tilde_eta = EMA_{tau_s}(eta)` a lagged-LR proxy for the lagging sharpness state.
By construction it is ~dormant on smooth cosine (so it cannot fit cosine noise);
it is identified on a `cosine×2 + wsdcon_3` train split and tested on held-out
`wsd / wsdld / wsdcon_9 / wsdcon_18`. Code: `repro/river_floor_lag.py`.

Result (`results/river_floor_lag.log`):

- **Large, consistent gains on true end-of-training sharp decays**: wsd and
  wsdld improve by **20–52%** MAE across all three scales.
- **But it over-corrects the mid-training step-to-constant curves**: wsdcon_9
  worsens by 33–56% (the lag should relax during the 8k-step constant second
  phase; a single τ_s cannot both capture the deep end-decay and decay away in
  the constant phase).
- Net held-out MAE 0.00368 vs frozen-MPL 0.00359 — **2.5% worse overall**.
- `κ` (CV 91%) and `τ_s` (CV 86%) are **not scale-invariant**; τ_s degenerates
  to ≫ run-length at 25/100M.

By the pre-registered kill criteria this **fails** (not scale-invariant, not a
net improvement). The signal is real and mechanism-consistent (sharp end-decays
carry a genuine positive residual MPL misses), but no low-parameter,
scale-invariant closed form captures it without harming other schedules.

**Overall finding across 7 theory-driven attempts** (SC-MPL, Q-MPL, EoS-kernel,
lag, floor, RV-EoS, sharpness-lag): on these 27 curves MPL is empirically
near-optimal for schedule transfer. The decay-phase residual is real and
physically interpretable (river-valley sharpness lag) but appears to require a
schedule-specific *state* (the lagging sharpness) that cannot be pinned from the
cumulative-LR statistic alone — which is exactly why MPL compresses it into the
effective exponent γ rather than resolving it. This is an honest negative that
*supports* MPL and explains γ, not a law that beats it.

## References
- Wen, Li, Liu, et al. *Understanding WSD: a River-Valley Loss-Landscape
  Perspective.* arXiv:2410.05192 (2024).
- Cohen et al. *Adaptive Gradient Methods at the Edge of Stability.*
  arXiv:2207.14484 (2022).
- Cohen et al. *Gradient Descent on Neural Networks Typically Occurs at the Edge
  of Stability.* ICLR 2021.
- Kalra & Barkeshli. *Why Warmup the Learning Rate?* arXiv:2406.09405 (2024).
- Belloni et al. *Universal Dynamics of Warmup-Stable-Decay.* arXiv:2601.09000 (2026).
- Luo et al. *Multi-Power Law for Loss-Curve Prediction.* (2025).
