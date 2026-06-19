# Theory Note for the Final Kappa Estimator

This note records the theoretical interpretation of the current paper-facing
`kappa` estimator. The goal is not to claim a universal optimal estimator, but
to make explicit the assumptions under which the estimator is derived and the
extra conservative corrections used in the experiments.

## 1. Observation Model

For one calibration curve, let `t_i in [0, 1]` be the normalized training step
and define the residual after the main MPL prediction as

```text
r_i = observed_loss_i - MPL_i.
```

The schedule response law provides a deterministic feature vector `phi`, whose
shape is fixed once the learning-rate schedule, token index, and MPL state are
given. We model the residual as

```text
r = kappa_* phi + g_* + eps,
g_* in G.
```

Here:

- `kappa_* >= 0` is the nonnegative amplitude of the schedule-specific response.
- `G` is a small low-frequency nuisance subspace that absorbs smooth MPL
  residual drift not explained by the response law.
- `eps` is the remaining zero-mean error.

The theoretical object is the nuisance subspace `G`, not a polynomial fit. In
the current implementation, `G` is approximated by the span of `[1, t, t^2]`
over normalized training step. That basis is only a lightweight way to remove
very slow MPL drift; it is not part of the schedule-response law and should not
be presented as a fitted loss model.

The Spectral nuisance-subspace audit replaces this lightweight smooth basis
with a discrete-cosine low-frequency subspace. The four-mode spectral `G` keeps
the transfer matrix non-failing, with worst off-diagonal `-1.8%` and
cosine-to-WSD `-3.6%`. The full sweep is also informative: one or two modes
under-cover MPL drift and can fail badly, while eight or more modes over-cover
the response and become nearly conservative. This is useful theoretically: the
result is not tied to a polynomial-shaped nuisance basis, but it does require a
sufficient-but-not-excessive nuisance bandwidth.

An automatic spectral variant makes this bandwidth condition explicit. First
enforce a minimum low-frequency control bandwidth `K_min=3`; then choose the DCT
bandwidth whose identifiable feature-energy fraction `R = ||M_G phi||^2 /
||phi||^2` is closest to a target. With target `R=0.35`, the constrained
retention-target rule gives worst off-diagonal `-1.7%`, mean off-diagonal
`-11.2%`, and cosine-to-WSD `-10.1%`. The unconstrained retention-target rule
fails because diffuse cosine features can have acceptable `R` at `K=1` before
MPL drift is adequately removed. Thus `R` is useful for amplitude conversion
and for choosing among sufficiently rich nuisance spaces, but it cannot replace
the low-frequency drift-control assumption.

## 2. Partial Regression Derivation

Let

```text
P_G = orthogonal projection onto G
M_G = I - P_G
phi_perp = M_G phi
r_perp = M_G r.
```

`M_G` projects out the low-frequency nuisance subspace. By the
Frisch-Waugh-Lovell theorem, the least-squares coefficient of `phi` in the full
regression on `phi` and `G` equals the least-squares coefficient obtained after
residualizing both `r` and `phi` against `G`:

```text
kappa_OLS = <phi_perp, r_perp> / ||phi_perp||^2.
```

With the nonnegativity constraint implied by the interpretation of `kappa` as a
response amplitude, this becomes

```text
kappa_NNLS = max(0, <phi_perp, r_perp> / ||phi_perp||^2).
```

This step is the core reason the estimator avoids simply matching the raw MPL
residue. It estimates only the component of the residue that remains aligned
with the response feature after smooth trend confounding has been removed.

## 3. Empirical-Bayes Shrinkage

The unregularized coefficient can be unstable when `phi_perp` has small energy.
Assume

```text
eps ~ N(0, sigma^2 I),
kappa ~ N_+(0, k0^2),
```

where `N_+` is a Gaussian prior truncated to `kappa >= 0`. The posterior mode is

```text
kappa_MAP = max(0, <phi_perp, r_perp> / (||phi_perp||^2 + sigma^2 / k0^2)).
```

The implementation writes

```text
tau = sigma / k0
```

and therefore uses

```text
kappa_MAP = max(0, <phi_perp, r_perp> / (||phi_perp||^2 + tau^2)).
```

`tau` is estimated by leave-curve-out empirical Bayes: when estimating the
`kappa` for one calibration curve, the prior/noise ratio is inferred from the
other available curves. This avoids using the target curve as both calibration
data and prior evidence.

## 4. Identifiable-Amplitude Conversion

The projected regression identifies only the component `kappa * phi_perp`.
Its identifiable response norm is

```text
||kappa_MAP * phi_perp|| = kappa_MAP ||phi_perp||.
```

For transfer, however, `kappa` is used as an amplitude on the full response
feature `phi`. If we do not want to extrapolate amplitude from the unobserved
`G` component of `phi`, the identified response norm should be normalized by
the full feature norm:

```text
kappa_effective
  = ||kappa_MAP * phi_perp|| / ||phi||
  = (||phi_perp|| / ||phi||) kappa_MAP.
```

Define

```text
R = ||phi_perp||^2 / ||phi||^2.
```

Then

```text
kappa_effective = sqrt(R) * kappa_MAP.
```

This is the current estimator. The interpretation is not that the FWL
coefficient mathematically requires `sqrt(R)` under the exact structural model.
Rather, `sqrt(R)` converts the amplitude estimated from the identifiable
projected response into a full-feature effective amplitude. It encodes the weak
assumption that response energy hidden inside the nuisance subspace should not
be freely extrapolated to transfer curves.

This is also the observed failure mode in the experiments: the projected trend
can have the right direction while the amplitude is too large if we transfer
the full `kappa_MAP` without accounting for the fraction of `phi` that was
actually identified.

The optional cap

```text
kappa_final = min(kappa_final, 0.03)
```

corresponds to a truncated susceptibility prior `0 <= kappa <= kappa_max`.
Experimentally, the cap is not the main stabilizer: the cap-free estimator
retains the same worst off-diagonal result as the capped estimator.

## 5. Proposition-Style Summary

The estimator can be stated as the following paper-ready proposition.

Assumptions:

- The MPL residual can be decomposed as `r = kappa_* phi + g_* + eps`, where
  `g_* in G` is low-frequency nuisance drift.
- The response amplitude is nonnegative: `kappa_* >= 0`.
- Conditional on `G` and `phi`, the residual noise is approximately isotropic
  with scale `sigma`.
- Across related calibration curves, the response amplitude has prior scale
  `k0`, estimated without using the held-out target curve.
- For transfer, amplitude should be normalized by the full response-feature
  norm `||phi||`, while only `||M_G phi||` is identifiable from the projected
  regression.

Proposition:

Let `M_G` be the orthogonal residualizer against `G`, and define

```text
phi_perp = M_G phi,
r_perp = M_G r,
tau = sigma / k0,
R = ||phi_perp||^2 / ||phi||^2.
```

Then the nonnegative empirical-Bayes MAP coefficient in the identifiable
projected problem is

```text
kappa_MAP = max(0, <phi_perp, r_perp> / (||phi_perp||^2 + tau^2)).
```

The full-feature effective amplitude implied by the identifiable response norm
is

```text
kappa_eff = sqrt(R) * kappa_MAP.
```

Therefore the final cap-free estimator is

```text
kappa_hat = sqrt(R) * max(0, <phi_perp, r_perp> / (||phi_perp||^2 + tau^2)).
```

The optional capped variant corresponds to imposing the additional prior
constraint `kappa_hat <= kappa_max`.

## 6. Complete Estimator

The current estimator is therefore

```text
r = observed_loss - MPL
choose a low-frequency nuisance subspace G
phi_perp = M_G phi
r_perp = M_G r
tau = sigma / k0
R = ||phi_perp||^2 / ||phi||^2
kappa = sqrt(R) * max(0, <phi_perp, r_perp> / (||phi_perp||^2 + tau^2))
optional: kappa = min(kappa, 0.03)
```

This formula is schedule-agnostic: it does not use schedule-family labels such
as cosine, WSD, or WSD-con. Different schedules enter only through their
generated response feature `phi`.

## 7. Empirical Evidence in the Current Worktree

Main audit: `results/current_law_final_kappa/REPORT.md`.

- `final_cap_0p03`: worst off-diagonal change `-2.7%`, mean off-diagonal
  `-12.4%`, cosine to WSD `-4.3%`, WSD-con 9e-5 to WSD `-15.9%`.
- `final_no_cap`: worst off-diagonal change `-2.7%`, mean off-diagonal
  `-12.1%`, cosine to WSD `-4.3%`.
- `no_retention_cap_0p03`: worst off-diagonal change `+1.8%`, cosine to WSD
  `-17.4%`, max cosine `kappa = 0.03`, cap saturation `61.1%`.
- `smooth_cap`: worst off-diagonal change approximately `0.0%`, but cosine to
  WSD is also approximately `0.0%`, meaning it is too conservative to capture
  the desired transfer.

Robustness audit: `results/current_law_final_kappa_robustness/REPORT.md`.

- Uniform half subsampling remains close to full calibration:
  worst off-diagonal `-2.4%`, cosine to WSD `-4.0%`.
- Uniform quarter subsampling remains close as well:
  worst off-diagonal `-2.8%`, cosine to WSD `-3.5%`.
- Contiguous half-curve subsets become conservative, which is expected because
  they do not cover the full response excitation and relaxation pattern.

Bootstrap audit: `results/current_law_final_kappa_bootstrap/REPORT.md`.

- Cosine has a wide interval including zero: mean full `kappa = 0.0050`, boot
  mean `0.0061`, p05 `0.0000`, p95 `0.0188`.
- WSD sharp and WSD linear have tighter positive intervals:
  WSD sharp p05 `0.0185`, p95 `0.0285`; WSD linear p05 `0.0201`, p95 `0.0299`.
- This matches the identifiability interpretation: diffuse schedules weakly
  identify the response amplitude, while sharper schedules identify it more
  strongly.

Retention-power audit:
`results/current_law_retention_power_audit/REPORT.md`.

- The audit sweeps `kappa = R^alpha * MAP` while keeping the same lightweight
  nuisance projection and leave-curve-out EB `tau`.
- `alpha = 0.00` without a cap has worst off-diagonal `+100.1%`, cosine to WSD
  `-24.3%`, and max cosine `kappa = 0.0508`, confirming the raw projected MAP
  amplitude can explode.
- `alpha = 0.00` with cap improves worst off-diagonal to `+1.8%`, but saturates
  the cap on `61.1%` of calibrations, so the hard cap is carrying much of that
  setting.
- `alpha = 0.50` has worst off-diagonal `-2.7%` both with and without cap, with
  cosine to WSD `-4.3%`, confirming that the square-root correction is stable
  without relying on the hard cap.
- Larger exponents such as `alpha = 1.00` or above are safer but too
  conservative: they drive cosine to WSD close to `0%`, erasing useful transfer.

Tau-sensitivity audit:
`results/current_law_tau_sensitivity_audit/REPORT.md`.

- The audit multiplies the leave-curve-out EB `tau` by constants while keeping
  the lightweight nuisance projection and `sqrt(R)` correction fixed.
- From `0.00x` to `1.00x` EB tau, the capped estimator keeps worst
  off-diagonal at `-2.7%`; cosine to WSD moves from `-5.7%` to `-4.3%`.
- `2.00x` EB tau remains stable with worst off-diagonal `-2.0%` and cosine to
  WSD `-2.5%`.
- Larger `tau` values are increasingly conservative, as expected from the MAP
  denominator. At `8.00x`, cosine to WSD is only `-0.3%`.
- Therefore the method is not relying on a finely tuned `tau`. The EB estimate
  selects a useful regularization scale, while most amplitude stabilization
  comes from nuisance projection and identifiable-amplitude conversion.

Train-only tau audit:
`results/current_law_trainonly_tau_audit/REPORT.md`.

- The stricter train-only `tau` audit estimates EB `tau` from the calibration
  curve only, rather than from other curves that may include the held-out test
  curve.
- The result remains stable: worst off-diagonal stays `-2.7%`, mean
  off-diagonal stays `-12.1%`, and cosine to WSD changes from `-4.3%` to
  `-5.6%`.
- This shows that the single-curve transfer conclusion does not rely on
  test-side information in the EB regularization scale.

Multi-curve calibration audit:
`results/current_law_multicurve_kappa_audit/REPORT.md`.

- The same estimator extends to a train set `S` by summing projected inner
  products and norms:

```text
dot_S = sum_c <M_G phi_c, M_G r_c>
l2_S = sum_c ||M_G phi_c||^2
full_l2_S = sum_c ||phi_c||^2
kappa_S = sqrt(l2_S / full_l2_S) * max(0, dot_S / (l2_S + tau^2))
```

- For each train subset, `tau` is estimated from training curves only, and
  evaluation is performed only on held-out curves. Median worst held-out change
  improves from `-1.0%` with one calibration curve to `-8.4%` with five
  calibration curves.
- This supports the interpretation that the estimator can use additional
  calibration coverage without introducing schedule-family labels.

Predictive shrinkage audit:
`results/current_law_predictive_shrinkage_audit/REPORT.md`.

- The soft spectral band-limited estimator identifies a useful response
  direction but can over-transfer amplitude to unseen WSD-con schedules. This
  failure mode is corrected by multiplying the pooled amplitude by a
  train-size factor

```text
c_n = n / (n + rho).
```

- This factor has a posterior-predictive interpretation. Suppose each
  calibration schedule has a latent transferable amplitude
  `kappa_c = theta + u_c`, where `theta` is the population transfer amplitude
  and `u_c` is schedule-specific transfer variation. Estimating `theta` from
  `n` schedules and applying it to a new schedule incurs an additional
  transfer-uncertainty term. Under a scalar Gaussian random-effects model, this
  yields a shrinkage of the form `n/(n+rho)`, where `rho` is the ratio of
  transfer variation to the effective population-amplitude prior precision.
- The audit tests the conservative half-degree-of-freedom prior `rho=0.5`.
  Equivalently, the deployed audit factor is `c_n = n/(n+0.5)`.
  Starting from the band-limited soft spectral estimator, this gives worst
  worst-heldout changes of `-1.0%`, `-1.1%`, and `-1.2%` for one-, two-, and
  three-curve calibration sets, respectively. It also preserves useful
  cosine-to-WSD transfer. Thus the remaining over-correction is better
  explained as finite-calibration amplitude uncertainty than as a failure of
  the response direction.
- Because `rho=0.5` has only been validated on the current schedule matrix, it
  should be presented as a promising extension rather than as the primary
  paper-facing estimator.
- The rho-sensitivity sweep supports this interpretation: `rho=0.25` still
  leaves positive WSD-con failures, `rho=0.35` is nearly on the boundary, and
  `rho in {0.5, 0.75, 1.0}` is non-failing in the predictive-shrinkage audit.
  After adding the target-identifiability gate, the rho-margin audit finds a
  wider plateau: the first fully non-harming grid value is `rho=0.40`, and
  `rho=0.40` through `rho=2.00` preserve all `558/558` main-matrix wins. Thus
  `rho=0.5` is not a knife-edge; it is a simple half-degree prior inside the
  stable safe range.
- A fully automatic inner-CV rho selector was also tested. It is not reliable
  on the current small calibration matrix: the inner problem often selects
  weak or zero shrinkage, which reintroduces held-out WSD-con failures. This is
  why the preferred extension uses a fixed weak posterior-predictive prior
  rather than claiming that `rho` can already be estimated robustly from the
  available train curves.

## 8. Predictive-Transfer Extension

The multi-curve estimator above pools projected evidence across calibration
curves and returns `kappa_S`. To transfer that scalar to an unseen schedule, we
can add a weak random-effects layer:

```text
kappa_c = theta + u_c,
u_c ~ N(0, sigma_transfer^2),
theta ~ N_+(0, k0^2).
```

Here `theta` is the schedule-agnostic component of the response amplitude, and
`u_c` captures idiosyncratic schedule-to-schedule transfer mismatch. When only
`n` calibration schedules are available, the posterior-predictive mean for an
unseen schedule is smaller than the in-sample pooled amplitude by

```text
c_n = n / (n + rho),
rho = sigma_transfer^2 / k0^2_eff.
```

This gives the extension

```text
kappa_transfer = c_n * kappa_S.
```

The current best fixed value is `rho=0.5`, which can be read as a conservative
half-degree-of-freedom prior: a single calibration curve is informative but not
fully trusted for transfer to a new schedule. The shrinkage weakens as
calibration coverage grows, so it does not erase the benefit of multi-curve
calibration.

For deployment on a target schedule, the target response direction must also be
identifiable after applying the same nuisance residualizer. Define

```text
R_target(lambda) = ||M_lambda phi_target||^2 / ||phi_target||^2.
```

If `R_target(lambda)` is too small, the target response feature is mostly
absorbed by the nuisance model; transferring a positive amplitude would then be
indistinguishable from injecting low-frequency MPL drift. The next-generation
safety extension therefore uses

```text
a_target = 0 if R_target(lambda) < 0.01,
a_target = 1 otherwise,
kappa_safe = a_target * kappa_transfer.
```

This is a target-side identifiability condition, not a schedule-family label.
The weak assumption behind the gate is that a scalar amplitude learned from
calibration curves should only be deployed when the target has enough
observable response energy outside the nuisance subspace. If `R_target` falls
below a fixed floor, the target's identifiable component is too small relative
to its full response feature, so the transferred amplitude is dominated by the
unidentified nuisance-confounded component. The principled action is to abstain
rather than extrapolate that component.

The current all-train-size audit gives `1116/1116` non-harming cells, worst
`+0.0%`, and mean `-5.9%` for `R_target >= 0.01`, while raw next-gen transfer
has `930/1116` non-harming cells and worst `+22.5%`. The threshold has a margin
interpretation in the current artifacts: the lowest positive main-matrix
target retention is `0.014797`, the highest positive diffuse extra-holdout
retention is `0.005721`, and their geometric midpoint is `0.009201`.
The target-retention margin audit makes this non-knife-edge: the chosen
threshold is `1.75x` above the maximum raw-harmful retention and the closest
main-matrix target is `1.48x` above the threshold. Lowering the floor to
`0.005` admits the diffuse cosine target and restores the `+22.5%` failure,
while raising it beyond the main cosine retention remains non-harming but
drops useful main-matrix transfers.

The stress-slice audit checks that the gate is not hiding a subgroup failure
inside the aggregate. It reports `0` slice failures across scale, train-size,
target-curve, train-group, and scale-by-train-size summaries. Each scale has
`372/372` non-harming rows, and each train-size slice is non-harming
(`144/144`, `315/315`, `360/360`, `225/225`, and `72/72`). This is still an
in-worktree robustness claim, not an external universality proof, but it
removes the main aggregate-masking concern.

The deployment estimator audit checks implementation coherence. A reusable
`NextGenKappaEstimator` computes the formula end to end from train curves,
target features, train-only lambda selection, empirical-Bayes `tau`, the
posterior-predictive shrinkage, and the target-retention gate. Across all
`1116` audited rows, it matches the rho-margin reference exactly: maximum
absolute differences for delta, kappa, target retention, selected lambda, and
target factor are all `0.000e+00`. This makes the next-generation rule a
single auditable estimator rather than a report-specific construction.

The target-loss blindness audit checks the deployment information boundary.
After freezing train-side `kappa_transfer`, every target loss curve is replaced
by a deterministic fake loss and the target gate is recomputed from the target
schedule feature. Across all `1116` rows, max differences in `R_target`, gate
factor, and `kappa_safe` are all `0.000e+00`. Thus target loss is used only for evaluation. Deployment uses training residuals plus target schedule
features.

The Scale-holdout constant audit checks whether the constants are only tuned to
the pooled three-scale matrix. Holding out one model scale at a time, `0.01`
remains inside the target-retention margin inferred from the other two scales
in `3/3` splits. For rho, the first safe value inferred from the two training
scales is at most the selected `rho=0.50` in `3/3` splits. Each held-out scale
has `372/372` non-harming rows and `186/186` main-matrix wins. This supports
scale stability within the current matrix, while still not replacing external
validation on new schedules or scales.

This evidence supports the following unified next-generation rule:

```text
kappa_safe
  = 1{R_target(lambda) >= 0.01}
    * [n / (n + 0.5)]
    * sqrt(l2_S / full_l2_S)
    * max(0, dot_S / (l2_S + tau^2)).
```

The rule remains schedule-agnostic. Calibration schedules enter through
`dot_S`, `l2_S`, `full_l2_S`, and `n`; the target schedule enters only through
the response feature `phi_target` and its residualized retention.

## 9. Paper-Ready Claim

A careful paper statement would be:

> We estimate the response amplitude by a partial-regression empirical-Bayes
> estimator. The estimator first projects out a small low-frequency nuisance
> component of the MPL residual, then estimates the nonnegative response
> amplitude with a leave-curve-out prior/noise ratio, and finally applies a
> full-feature amplitude conversion based on the fraction of response-feature
> energy that remains identifiable after nuisance projection.

This is stronger than an ad hoc fitted scalar because the main coefficient is
derived from partial regression and the denominator follows from a Gaussian
empirical-Bayes model. The `sqrt(R)` factor should be presented as an
identifiable-amplitude conversion: it normalizes the response norm observed
outside the nuisance subspace by the full response-feature norm, avoiding
unjustified extrapolation of the nuisance-confounded component.

The next-generation extension adds a posterior-predictive transfer shrinkage
and a target identifiability condition. In words: learn only the response
amplitude that is identifiable outside low-frequency MPL drift, shrink that
amplitude for finite calibration coverage, and apply it only when the target
response direction is itself identifiable after the same nuisance projection.

## 10. Current Limitations

- The derivation assumes the response feature `phi` is correctly specified up
  to amplitude. If the feature shape is wrong, the estimator can only find the
  best projected amplitude.
- The nuisance subspace `G` is a weak smooth-drift model, not a complete model
  of all MPL errors. The current implementation uses `[1, t, t^2]` only as a
  simple low-frequency basis.
- The empirical-Bayes `tau` is estimated from the current family of public
  curves. External validation on additional schedules is still needed before
  claiming universal validity.
- The retention factor is an identifiable-amplitude conversion, not a standalone
  optimality theorem. Its strength comes from the combination of geometric
  reasoning and ablation evidence.
- The next-generation target gate is a conservative identifiability condition.
  It prevents known diffuse-target failures in the current matrix, but it should
  still be validated on additional independent schedule families before being
  claimed as a universal deployment guarantee.
