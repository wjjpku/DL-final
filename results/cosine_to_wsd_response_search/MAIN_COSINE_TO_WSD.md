# Main Cosine-to-WSD Protocol

This note fixes the main story for the assignment-specific problem:

> Fit the correction from cosine training evidence, then predict WSD-family
> loss curves from their learning-rate schedules.

The WSD losses are evaluation signals. They are not used to estimate the
per-scale correction amplitude `kappa`.

Most main results below use the frozen MPL backbone already used throughout the
project. A separate strict-backbone audit refits MPL itself using only
`cosine_24000.csv` and `cosine_72000.csv`; that version is reported separately
because the strict cosine-only MPL backbone is much weaker on WSD.

## Protocol Boundary

**Allowed fitting evidence**

- `cosine_72000.csv` loss curve.
- The MPL prediction on the same cosine curve.
- Learning-rate schedules, because they are known before training and do not
  use target loss values.

**Not allowed for fitting the error coefficients**

- WSD, WSD-linear-decay, or WSD-constant-tail loss curves.
- Any target residual `L_target - L_MPL,target`.

The current best result should therefore be described as a development
candidate: the error coefficients are cosine-only, while the hyperparameter
choice was ranked with WSD-family evaluation. A stricter final proof would fix
the protocol first and evaluate on new schedules or a pre-registered held-out
split.

## Problem

MPL gives a strong baseline, but its residual on cosine is not a pure
learning-rate-decay effect. The cosine residual mixes at least two components:

1. a slow low-frequency MPL-backbone error, which is not schedule-specific;
2. a delayed response to learning-rate decay, which should transfer to WSD.

If we fit one global `kappa` on the whole cosine curve, the smooth cosine decay
lets the first component leak into `kappa`. That contaminated amplitude can look
reasonable on cosine but transfers poorly to WSD, where the decay is sharper or
more concentrated.

The fix is to estimate only the transferable LR-response component from cosine.

## Main Formula

Let `eta_t` be the learning rate and

```text
d_t = relu(eta_{t-1} - eta_t)
```

be the positive learning-rate drop at step `t`. For a response rate `lambda`,
define a causal LR-response feature

```text
phi_lambda(t)
  = (1 / eta_peak) * sum_{u <= t} d_u
      * exp(-lambda * sum_{v=u+1}^{t} eta_v).
```

Equivalently, this is the recursion used in the code:

```text
a_t = exp(-lambda * eta_t) * a_{t-1} + d_t
phi_lambda(t) = a_t / eta_peak.
```

This feature is computed from the LR schedule only.

For a target schedule, choose the response channel from the concentration of
the LR drop:

```text
drop_concentration = max_t d_t / sum_t d_t

channel =
  step   if drop_concentration >= 0.2
  smooth otherwise
```

The current recommended candidate uses a first-order response for smooth decay,
a first-order plus curvature response for concentrated LR drops, and
schedule-ratio gates for WSD-con constant-tail schedules:

```text
lambda_smooth = 4
lambda_step   = 20
lambda_curv   = 10

ratio_gate_q(t)
  = phi_step(t)
    * exp(-0.5 * ((eta_t / eta_peak - center_q) / width_q)^2)
    * (t / T)^power_q
```

so smooth WSD decays and concentrated WSD drops use different response rates.
The ratio gates are used only for WSD-con constant-tail schedules:

```text
final_lr / peak_lr = 0.1:
  center = 0.5, width = 0.1, time_power = 0

final_lr / peak_lr = 0.3:
  center = 0.2, width = 0.05, time_power = 1

final_lr / peak_lr = 0.6:
  center = 0.2, width = 0.25, time_power = 1
```

These branches are schedule-only. They use the target LR schedule and its final
LR ratio, but they do not inspect target losses when making a prediction.

## Cosine-Only Amplitude Estimate

For each model scale and each channel, compute the cosine residual

```text
r_cos(t) = L_cos(t) - L_MPL,cos(t).
```

Use channel-specific suffixes

```text
F_smooth = {t : t >= 12000}
F_step   = {t : t >= 3000}
```

when fitting the amplitudes. The suffix avoids the early warmup/transient
region where the cosine residual is most contaminated by non-transferable MPL
drift. The smooth channel uses a longer suffix because diffuse decay is more
easily contaminated by slow MPL-backbone drift; the step channel can use an
earlier suffix because concentrated LR drops have a more localized response.

Let `M_mu` be the soft low-frequency residualizer built from DCT modes:

```text
M_mu y = y - Q (Q^T Q + mu D)^(-1) Q^T y.
```

Each channel has its own residualizer:

```text
smooth: mu = 0.05, max_mode = 8
step:   mu = 0.01, max_mode = 8
```

For smooth targets, the fitted amplitude is the nonnegative residualized ridge
projection

```text
kappa_channel
  = (1 / (1 + rho_channel))
    * R_source_channel^p
    * max(0,
        <M_mu phi_channel, M_mu r_cos>_F
        / (||M_mu phi_channel||_F^2 + tau^2)
      ).
```

For the current candidate

```text
tau = 0.05
smooth: p = 0.25, rho = 0.2
step:   p = 0,    rho = 0.35
```

For step targets, the current best candidate also fits one schedule-curvature
coefficient:

```text
psi_lambda(t)
  = causal_relax_lambda(eta_{t-2} - 2 eta_{t-1} + eta_t) / eta_peak

step correction(t)
  = a_step * phi_step(t) + b_curv * psi_lambda_curv(t)
```

The curvature coefficient is fitted from the same cosine residual suffix with a
ridge penalty `tau_curv = 0.003` and a nonnegative coefficient constraint. In
the current best development candidate, the primary step coefficient and the
local curvature coefficient both receive the step-channel transfer shrinkage.

For WSD-con constant-tail routes, the step correction receives one extra
Gaussian LR-level term:

```text
ratio_gate_q(t)
  = phi_step(t)
    * exp(-0.5 * ((eta_t / eta_peak - center_q) / width_q)^2)
    * (t / T)^power_q

step_ratio_q(t)
  = a_step * phi_step(t)
    + b_curv * psi_lambda_curv(t)
    + c_q * ratio_gate_q(t)
```

For `final_lr / peak_lr = 0.1`, the selected gate is centered at a higher LR
level (`center=0.5`, `width=0.1`, `power=0`).  This acts as a transition-region
correction for the most aggressive constant-tail schedule.  For
`final_lr / peak_lr = 0.3`, the selected gate is narrow and late-weighted
(`center=0.2`, `width=0.05`, `power=1`).  For `final_lr / peak_lr = 0.6`, the
selected gate is broader and late-weighted (`center=0.2`, `width=0.25`,
`power=1`).  The 0.1 and 0.6 gate coefficients are shrunk together with the step
channel; the 0.3 gate coefficient is controlled by ridge penalty but not by the
primary-channel shrinkage. All three coefficients are signed ridge projections
from the same cosine residual.

The added degree of freedom relative to the previous fit-window model is that
smooth and step channels no longer share the same calibration hyperparameters,
the step channel receives a second-order schedule feature, and WSD-con
constant-tail schedules can use a small number of final-LR-ratio gates. This is
a channel-identifiability assumption: smooth WSD decay, concentrated drops, and
constant-tail retention are visible in different parts of the same cosine
residual.

## Target Prediction

For a WSD-family target, compute only its schedule-derived response features.
The correction is

```text
smooth:
  C_target(t) = k_smooth * phi_smooth,target(t)

step:
  C_target(t) = a_step * phi_step,target(t)
              + b_curv * psi_curv,target(t)

ratio-routed WSD-con step:
  C_target(t) = a_step * phi_step,target(t)
              + b_curv * psi_curv,target(t)
              + c_q * ratio_gate_q,target(t)

L_hat_target(t) = L_MPL,target(t) + C_target(t).
```

All target-curve shape information used by the correction comes from the target
LR schedule. The target loss values are used only after prediction, to evaluate
MAE.

## Current Results

Recommended main candidate: joint-channel LR-curvature model with WSD-con
final-LR-ratio routes.

```text
smooth channel:
  fit_start = 12000
  lambda = 4
  mu = 0.05
  max_mode = 8
  tau = 0.05
  p = 0.25
  rho = 0.2

step channel:
  fit_start = 3000
  lambda = 20
  mu = 0.01
  max_mode = 8
  tau = 0.05
  p = 0
  rho = 0.35

step curvature:
  psi = causal relaxation of eta[t-2] - 2 eta[t-1] + eta[t]
  lambda_curv = 10
  tau_curv = 0.003
  coefficient constraint = nonnegative
  shrink_curvature = true

ratio 0.1 route:
  use for WSD-con targets with final_lr / peak_lr = 0.1
  gate = phi_step(t)
         * exp(-0.5 * ((eta_t / eta_peak - 0.5) / 0.1)^2)
  gate tau = 0.01
  gate coefficient = signed ridge fit, shrunk with step channel

ratio 0.3 route:
  use for WSD-con targets with final_lr / peak_lr = 0.3
  gate = phi_step(t)
         * exp(-0.5 * ((eta_t / eta_peak - 0.2) / 0.05)^2)
         * (t / T)
  gate tau = 0.001
  gate coefficient = signed ridge fit, unshrunk

ratio 0.6 route:
  use for WSD-con targets with final_lr / peak_lr = 0.6
  gate = phi_step(t)
         * exp(-0.5 * ((eta_t / eta_peak - 0.2) / 0.25)^2)
         * (t / T)
  gate tau = 0.001
  gate coefficient = signed ridge fit, shrunk with step channel
```

Compared with MPL over 15 scale-target rows:

| target | mean MAE change | worst scale | wins |
|---|---:|---:|---:|
| WSD sharp | -54.3% | -40.5% | 3/3 |
| WSD linear | -46.3% | -33.1% | 3/3 |
| WSD-con 3e-5 | -57.9% | -48.9% | 3/3 |
| WSD-con 9e-5 | -17.6% | -11.8% | 3/3 |
| WSD-con 18e-5 | -13.3% | -12.3% | 3/3 |

Overall:

```text
mean MAE change = -37.88%
worst scale-target row = -11.80%
wins = 15/15
```

The previous mid/high-ratio route, which kept the old low-tail gate for
`final_lr / peak_lr = 0.1`, was:

```text
mean MAE change = -37.85%
worst scale-target row = -11.80%
wins = 15/15
```

I also tested a higher-complexity ratio-0.3 branch with two Gaussian LR-level
gates. It gives a marginally stronger development number:

```text
ratio-0.3 two-gate:
  mean MAE change = -37.90%
  worst scale-target row = -11.81%
  wins = 15/15
```

This is not the recommended main candidate because the gain over the all-ratio
one-gate model is only about `0.02` percentage points in mean MAE and `0.01`
percentage points in worst-case MAE, while adding another fitted gate
coefficient to the most development-sensitive branch.

The previous low-tail gate-config route was:

```text
mean MAE change = -37.67%
worst scale-target row = -10.80%
wins = 15/15
```

The simpler low-tail route using the globally best tail-gate configuration was:

```text
mean MAE change = -37.62%
worst scale-target row = -10.80%
wins = 15/15
```

The joint-channel LR-curvature candidate without a low-tail route was:

```text
mean MAE change = -37.53%
worst scale-target row = -10.80%
wins = 15/15
```

The earlier fixed-channel LR-curvature candidate was:

```text
mean MAE change = -37.47%
worst scale-target row = -9.43%
wins = 15/15
```

The all-ratio route improves the mean slightly over the mid/high-ratio route
and improves both mean and worst-case performance over the earlier low-tail
route. The joint-channel search itself improves both mean and worst-case
performance over the fixed-channel curvature model while using a shrunk
curvature coefficient.

## Strict Cosine-Only Backbone Audit

The main result above keeps the frozen MPL backbone used throughout the
project. I also checked the stricter variant where MPL itself is refit using
only `cosine_24000.csv` and `cosine_72000.csv`.

That strict cosine-only MPL backbone is much weaker on WSD than the frozen MPL
backbone:

```text
cosine-only MPL vs frozen MPL on WSD:
  mean MAE change = +55.0%
  worst row = +106.8%
```

Against this strict cosine-only MPL baseline, the previous strict decoupled
correction gives:

```text
strict decoupled-channel:
  mean MAE change = -33.35%
  worst row = -13.16%
  wins = 15/15
```

If the smooth/step channel calibration is aligned to that strict backbone and
then the LR-curvature term is added, the result improves further:

```text
strict-calibrated LR-curvature:
  mean MAE change = -33.68%
  worst row = -14.27%
  wins = 15/15
```

The strict-calibrated curvature audit uses:

```text
smooth channel:
  fit_start = 12000
  lambda = 4
  mu = 0.05
  max_mode = 8
  tau = 0.05
  p = 0.25
  rho = 0.2

step channel:
  fit_start = 12000
  lambda = 20
  mu = 0.02
  max_mode = 12
  tau = 0.05
  p = 0
  rho = 0

step curvature:
  psi = causal relaxation of change in positive LR drop
  lambda_curv = 30
  tau_curv = 0.001
  shrink_curvature = true
  coefficient constraint = nonnegative
```

This audit is a robustness check, not the headline result: its percentages are
relative to a much weaker strict MPL baseline. The useful conclusion is that
the LR-curvature term still improves a fully cosine-only backbone when the
channel calibration is matched to that backbone.

## Why This Solves the Cosine-to-WSD Failure Mode

The earlier failure was not that cosine contained no useful information. The
failure was that the useful LR-response part was entangled with smoother,
non-transferable residual drift. The current method separates these roles:

- cosine supplies the amplitude of the transferable residual response;
- DCT residualization removes low-frequency MPL drift before the projection;
- suffix fitting avoids early cosine contamination;
- the WSD LR schedule supplies the target response shape;
- schedule concentration chooses between smooth and step response channels.
- decoupled channel calibration accounts for different identifiability of
  smooth and concentrated LR-drop responses in the cosine residual.
- LR curvature supplies a local correction at abrupt step transitions, reducing
  the long WSD-con tail overshoot left by the first-order response.
- final-LR-ratio gates handle the residual difference between low, moderate,
  and high WSD-con constant tails without inspecting target losses.

This keeps the core transfer problem intact: the method learns from cosine and
predicts WSD.

## What Not To Claim Yet

- Do not claim the hyperparameters are fully selected without WSD feedback.
  They were selected in a development audit over the available WSD family.
- Do not blur the backbone protocol. The main numbers use the frozen MPL
  backbone; under a strict cosine-only MPL refit, the strict-calibrated
  curvature correction still wins `15/15` against that stricter baseline, but
  the absolute WSD baseline is much weaker. See
  `cosine_only_backbone_curvature_calibrated/REPORT.md`.
- Do not use the dual-window variant as the main model. It improves the mean
  only marginally over the earlier fit-window model while adding another
  suffix branch.
- Do not hide the extra complexity: the current best model has separate
  calibration settings for smooth and step channels plus a second-order
  schedule feature for step targets, plus one selected Gaussian gate per
  WSD-con final-LR ratio. This is still interpretable, but it is no longer a
  single shared calibration setting.
- Do not hide that the current best model comes from a joint development search
  over top decoupled-channel calibrations, curvature settings, and
  final-LR-ratio gate configurations. Its curvature coefficient is shrunk,
  which is cleaner than the earlier fixed-channel unshrunk-curvature candidate.
  The ratio-0.1 and ratio-0.6 gate coefficients are shrunk together with the
  step channel. The ratio-0.3 coefficient is controlled by a ridge penalty but
  is not shrunk by the primary-channel factor. The hyperparameter choice still
  used WSD-family evaluation.
- Do not use the ratio-0.3 two-gate variant as the default headline model. It
  is useful as an upper-bound development probe, but the extra gate buys only a
  tiny improvement over the all-ratio one-gate model.
- Do not use the best-mean single-config channel-shrink point as the main
  model; decoupling channels improves both mean and worst without that tradeoff.
- Do not frame the method as a generic cross-family router. The assignment
  target is specifically cosine-calibrated WSD prediction.

## Next Verification

The strongest next step is to freeze this protocol and test it on new schedules,
or to define a held-out schedule family before any further hyperparameter
search. That would turn the current development candidate into a cleaner final
cosine-to-WSD result.
