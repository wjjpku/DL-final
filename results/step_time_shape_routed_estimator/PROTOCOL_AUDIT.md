# Shape-Routed Protocol Audit

This audit checks the deployment protocol behind the shape-routed step-time estimator.  The target LR schedule is allowed to choose a route, but the target loss residual is not allowed to choose the calibration source, tau, nuisance basis, kappa, or predicted correction.

## Checks

- Core route-table lock: `6/6` committed routes match the LR-schedule-only recomputation.
- Extended route-table lock: `9/9` committed routes match the LR-schedule-only recomputation.
- Target exclusion: `27/27` target-scale predictions exclude the target curve from the calibration set.
- Nonzero correction routes audited: `18/27`.
- Target residual scramble: max `|delta kappa| = 0.000e+00`, max `|delta correction| = 0.000e+00`.
- Overall protocol status: `27/27` rows pass.

## Interpretation

The audit does not claim that the route thresholds were chosen prospectively.  It verifies the narrower but essential deployment property: after the rule is fixed, a target's own loss residual cannot affect its assigned correction.  Any measured target-holdout gain therefore comes from LR-shape routing plus source-curve calibration, not from fitting the target residual.
