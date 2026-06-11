# Next-Generation Kappa Formula Candidate

This note consolidates the strongest current research candidate into one auditable formula. It is not yet the paper-facing main estimator because the fixed transfer prior needs external validation, but it is the most promising general `kappa` method found in the current worktree.

## Formula

For each calibration curve `c`, define the MPL residual and response feature

```text
r_c = observed_loss_c - MPL_c
phi_c = response_feature(schedule_c)
```

Use a soft spectral nuisance residualizer. Let `Q` be the normalized DCT basis up to mode 12 and define

```text
A_lambda = (Q^T Q + lambda D)^(-1) Q^T,
M_lambda y = y - Q A_lambda y,
D_jj = j^4, D_00 = 0.
```

Select `lambda` by train-only inner-CV restricted to the identifiable band

```text
lambda in [0.01, 0.03].
```

For a train set `S` of `n` calibration curves, pool projected evidence:

```text
dot_S = sum_{c in S} <M_lambda phi_c, M_lambda r_c>
l2_S = sum_{c in S} ||M_lambda phi_c||^2
full_l2_S = sum_{c in S} ||phi_c||^2
R_S = l2_S / full_l2_S
```

With empirical-Bayes `tau = sigma/k0`, estimate the identifiable amplitude

```text
kappa_pool = sqrt(R_S) * max(0, dot_S / (l2_S + tau^2)).
```

Finally apply posterior-predictive transfer shrinkage for finite calibration coverage:

```text
c_n = n / (n + 0.5)
kappa_transfer = c_n * kappa_pool.
```

## Interpretation

- `M_lambda` removes low-frequency MPL residual drift without hard schedule-family labels.
- The band `lambda in [0.01, 0.03]` is an identifiable soft-prior region: weaker smoothing leaves WSD-con over-transfer, while stronger smoothing starts to remove response signal.
- `sqrt(R_S)` converts the amplitude identified outside the nuisance subspace into a full-feature effective amplitude.
- `c_n = n/(n+0.5)` is a posterior-predictive shrinkage factor from a scalar random-effects view of schedule transfer: finite calibration coverage should not be trusted as a fully population-level amplitude.

## Proposition-Style Derivation

Assume the following weak model for calibration curves `c in S`:

```text
r_c = kappa_c phi_c + g_c + eps_c,
g_c in span(Q) approximately low-frequency,
eps_c is zero-mean with approximately isotropic scale sigma,
kappa_c = theta + u_c,  u_c is schedule-specific transfer variation.
```

The soft residualizer `M_lambda` is the MAP residualizer for a nuisance coefficient vector with a Sobolev-type Gaussian prior penalizing high DCT modes by `j^4`. Thus `M_lambda r_c` removes the low-frequency MPL drift component while preserving response directions not explained by the nuisance prior. Conditional on `lambda`, the pooled projected likelihood for a common amplitude is quadratic in `kappa`, with sufficient statistics `dot_S` and `l2_S`. Combining this likelihood with the nonnegative empirical-Bayes prior gives

```text
kappa_MAP,S = max(0, dot_S / (l2_S + tau^2)).
```

Only `l2_S` of the full response energy `full_l2_S` is identifiable after nuisance residualization, so the transferable full-feature amplitude is

```text
kappa_pool = sqrt(l2_S / full_l2_S) * kappa_MAP,S.
```

Finally, under the scalar random-effects layer `kappa_c = theta + u_c`, applying an amplitude learned from `n` calibration schedules to a new schedule introduces transfer variance. A conjugate Gaussian posterior-predictive mean has the shrinkage form `n/(n+rho)`; the current audit uses the conservative weak prior `rho=0.5`. Therefore the next-generation transfer amplitude is

```text
kappa_transfer = [n / (n + 0.5)] * sqrt(l2_S / full_l2_S) * max(0, dot_S / (l2_S + tau^2)).
```

The derivation uses no schedule-family label. Schedule information enters only through `phi_c`, the observed residual `r_c`, and the train-set size `n`.

## Evidence

Primary audit: `../current_law_predictive_shrinkage_audit/REPORT.md`.

- The selected soft residualizer strengths stay inside the identifiable band: lambda stability audit reports `186/186` `rho=0.5` kappa rows with `lambda in [0.01, 0.03]`, median `0.030`.
- Without predictive shrinkage, the band-limited soft spectral estimator has positive worst held-out failures for small train sets: `+13.2%`, `+5.6%`, and `+3.3%` for one-, two-, and three-curve calibration.
- With `rho=0.5`, the same settings become non-failing: `-1.0%`, `-1.1%`, and `-1.2%`.
- With the target-identifiability gate fixed, rho-margin audit reports `rho=0.40` as the first fully non-harming grid value; `rho=0.40` through `rho=2.00` preserve all `558/558` main-matrix wins, and selected `rho=0.50` has mean `-5.9%`, worst `+0.0%`, and `1116/1116` non-harming cells.
- Core transfer remains useful: `rho=0.5` gives Cosine -> WSD sharp `-20.5%` with `3/3` wins, and WSD-con 9e-5 -> WSD sharp `-8.7%` with `3/3` wins.
- The complete single-curve off-diagonal matrix has `30/30` improving cells, worst mean cell `-1.5%`, and mean off-diagonal `-12.0%`.
- The scale-specific single-curve matrix has `90/90` improving cells across 25M, 100M, and 400M, with worst cell `-1.0%`.
- Rho sensitivity supports the fixed prior: `rho=0.25` is still unsafe, `rho=0.35` is near the boundary, and `rho in {0.5, 0.75, 1.0}` is non-failing but larger values are more conservative.
- A fully automatic train-only rho selector is currently unreliable on this small calibration matrix; it often chooses weak or zero shrinkage and reintroduces held-out failures.

## Target Identifiability And External Holdout Limitation

Additional repo curves not included in the main six-schedule matrix expose an important boundary condition. Raw next-gen transfer is unsafe on `cosine_24000` (mean `+7.2%`, worst `+21.8%`), while `constant_24000` and `constant_72000` are unaffected because their response feature is zero. `cosine_24000` is also one of the MPL baseline fitting curves, so this is not a clean independent benchmark, but it is a useful warning about diffuse target schedules.

The more model-native target safety rule uses the same soft residualizer as the estimator. Define

```text
R_target(lambda) = ||M_lambda phi_target||^2 / ||phi_target||^2.
```

Then use an identifiability gate

```text
a_target = 0 if R_target(lambda) < 0.01,
a_target = 1 otherwise,
kappa_transfer_safe = a_target * kappa_transfer.
```

This gate abstains on target response directions whose energy is almost entirely removed by the nuisance residualizer. The theoretical interpretation is direct: if `M_lambda phi_target` has negligible energy, transferring a positive amplitude into `phi_target` is not identifiable apart from low-frequency MPL drift unless target residual evidence is available.

The target-identifiability audit compares this rule to the older peak/mean schedule-localization gate across all calibration train sizes. Raw next-gen has `930/1116` non-harming cells and worst `+22.5%` because of `cosine_24000`. A retention gate with `R_target(lambda) >= 0.01` has `1116/1116` non-harming cells, worst `+0.0%`, and mean `-5.9%`; this is slightly stronger than the peak/mean gate (`-5.7%`) and is tied to the estimator's nuisance model. The train-size breakdown is non-harming for every calibration size: `144/144`, `315/315`, `360/360`, `225/225`, and `72/72` for train sizes one through five. Thresholds from `0.0075` through `0.015` give non-harming results on the current audit, while `0.005` is unsafe because it lets `cosine_24000` through. The threshold also has a margin interpretation: the lowest positive main-matrix target retention is `0.014797`, the highest positive diffuse extra-holdout retention is `0.005721`, and their geometric midpoint is `0.009201`, so `0.01` separates the two regimes on a log scale without using held-out loss values.

The target-retention margin audit makes this separation explicit: the chosen floor is `1.75x` above the maximum raw-harmful retention and the nearest main-matrix target is `1.48x` above the floor. A threshold of `0.005` restores the `+22.5%` diffuse-cosine failure, while thresholds above the main cosine retention remain non-harming but begin dropping useful main-matrix transfers. This supports treating `0.01` as a margin-based identifiability floor rather than a loss-tuned optimum.

The stress-slice audit checks that this conclusion is not hiding an aggregate-only failure. The safe formula has `1116/1116` non-harming rows overall and `0` slice failures across scale, train-size, target-curve, train-group, and scale-by-train-size summaries. Each scale has `372/372` non-harming rows, each train-size slice is non-harming (`144/144`, `315/315`, `360/360`, `225/225`, `72/72`), and all main-matrix targets improve while non-identifiable extra targets abstain.

A purely train-relative target threshold is not as good on the current evidence. Weak relative gates such as `train_relative_gate_0p05` preserve transfer but still let the diffuse external cosine target through (`+22.5%` worst), while the first safe pure relative gate, `train_relative_gate_0p5`, is more conservative (mean `-5.4%`, `438/1116` wins). Adding the absolute floor back, for example `max(0.01, beta * R_train)`, restores safety, which indicates that the essential rule is the absolute target-identifiability floor rather than a threshold defined only relative to calibration curves.

For comparison, the older peak/mean gate

```text
a_target = 0 if peak(phi_target) / mean(phi_target) < 2
```

also gives `1116/1116` non-harming cells and worst `+0.0%`, but it is less tightly connected to the residualized likelihood and has slightly weaker mean improvement. The retention gate should therefore be treated as the stronger current next-generation deployment rule, while both gates remain limitation-aware safety rules rather than evidence that raw next-gen transfer is universally safe.

## Current Status

This is the best current general-purpose `kappa` candidate, but it should be described as a next-generation extension rather than a final paper claim until validated on additional schedule families or independent runs. The main unresolved issue is not the response direction; it is estimating the population-transfer amplitude from limited calibration coverage.
