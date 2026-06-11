# Next-Gen Kappa Formula Card

This is the compact, implementation-facing statement of the strongest current
research `kappa` candidate. It is not the paper-facing main estimator; use
`PAPER_METHOD.md` for the conservative paper claim.

## Estimator

For calibration curves `c in S`, define

```text
r_c = observed_loss_c - MPL_c
phi_c = response_feature(schedule_c)
```

Use the soft DCT/Sobolev nuisance residualizer

```text
M_lambda y = y - Q (Q^T Q + lambda D)^(-1) Q^T y,
D_jj = j^4, D_00 = 0,
lambda in [0.01, 0.03].
```

Pool calibration evidence:

```text
dot_S = sum_c <M_lambda phi_c, M_lambda r_c>
l2_S = sum_c ||M_lambda phi_c||^2
full_l2_S = sum_c ||phi_c||^2
n = |S|
```

The transferable amplitude is

```text
kappa_transfer
  = [n / (n + 0.5)]
    * sqrt(l2_S / full_l2_S)
    * max(0, dot_S / (l2_S + tau^2)).
```

Apply the target-identifiability gate:

```text
R_target(lambda) = ||M_lambda phi_target||^2 / ||phi_target||^2
a_target = 1{R_target(lambda) >= 0.01}
kappa_safe = a_target * kappa_transfer
```

## Interpretation

- `M_lambda` removes low-frequency MPL residual drift.
- `sqrt(l2_S / full_l2_S)` converts the identifiable projected response norm
  into a full-feature effective amplitude.
- `n/(n+0.5)` is a weak posterior-predictive shrinkage for finite calibration
  coverage.
- `R_target(lambda) >= 0.01` requires the target response direction to remain
  identifiable after the same nuisance residualization.
- No schedule-family labels are used.

## Current Evidence

- Lambda stability audit: all `rho=0.5` next-generation kappa rows stay inside
  the identifiable band `lambda in [0.01, 0.03]` (`186/186`), with median
  selected lambda `0.030`.
- Predictive shrinkage with `rho=0.5` is non-failing across train sizes in the
  current matrix and preserves useful cosine-to-WSD transfer.
- Rho margin audit with the target gate fixed finds a stable safe plateau:
  `rho=0.40` is the first fully non-harming grid value, and `rho=0.40`
  through `rho=2.00` preserve all `558/558` main-matrix wins.
- Target-identifiability gating gives `1116/1116` non-harming cells across all
  calibration train sizes, with worst `+0.0%` and mean `-5.9%`.
- Target-retention margin audit places the chosen threshold inside the interval
  `0.005721 < 0.01 < 0.014797`, with `1.75x` lower-side and `1.48x`
  upper-side margins; `0.005` restores the `+22.5%` diffuse-cosine failure.
- Component ablation isolates the two stabilizers: no predictive shrinkage has
  worst `+32.6%`, `rho=0.5` shrinkage improves worst to `+22.5%`, and adding
  the `R_target(lambda) >= 0.01` gate gives `1116/1116` non-harming cells.
- Stress-slice audit finds `0` safe-formula slice failures across scale,
  train-size, target-curve, train-group, and scale-by-train-size checks;
  every audited slice remains non-harming.
- Deployment estimator audit verifies the reusable `NextGenKappaEstimator`:
  it reproduces the rho-margin reference exactly across `1116` rows, with
  max absolute delta and kappa differences `0.000e+00`.
- Target-loss blindness audit replaces every target loss curve with fake
  losses and leaves `R_target`, the target gate, and `kappa_safe` unchanged
  across `1116` rows; target loss is used only for evaluation.
- Raw next-gen transfer fails the same all-train-size audit with worst
  `+22.5%`, driven by diffuse `cosine_24000` targets.

## Limitation

This is the best current general-purpose `kappa` candidate in the worktree, but
it remains a next-generation extension until validated on additional independent
schedule families or runs.
