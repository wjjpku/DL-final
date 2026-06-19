# Core Formula Card: q2 Half-Life MPL-LD Response

> 2026-06-19 update: current recommended formula is the theory-refined
> observation-bracket MPL-LD response.  The core architecture is unchanged:
> MPL baseline plus one causal LR-drop response term.  The refinement replaces
> the less interpretable max-drop concentration with an effective-drop-count
> concentration, and explains the locality factor as a projection, not a gate.

## Formula

For target schedule \(s\),

\[
\widehat L_s(t)
=
L_{\mathrm{MPL},s}(t)
+
a_s\widehat\kappa_s\phi_{\lambda_s,s}(t).
\]

Only \(\widehat\kappa_s\) is fitted from loss residuals, and it is fitted only
from the source cosine residual.  Target losses are used only for evaluation.

## Schedule Variables

Positive LR drop:

\[
d_{s,t}=[\eta_{s,t-1}-\eta_{s,t}]_+,\qquad
D_s=\sum_t d_{s,t}.
\]

If \(D_s>0\), define the normalized drop mass

\[
p_{s,t}=\frac{d_{s,t}}{D_s}.
\]

The response-rate concentration is the Herfindahl / effective-count statistic

\[
q_s=\sum_t p_{s,t}^2=\frac{1}{n_{\mathrm{eff},s}}.
\]

This is more interpretable than the older \(q_\infty=\max_t p_t\):

- single-step drop: \(q_s=1\);
- uniformly spread drop over \(n\) steps: \(q_s=1/n\);
- diffuse cosine-like decay: \(q_s\approx 0\).

Let \(\Delta_{\mathrm{obs}}\) be the modal logging interval and

\[
\lambda_{\mathrm{obs}}
=
\frac{\log 2}{\eta_{\max}\Delta_{\mathrm{obs}}}.
\]

Use a half-life bracket:

\[
H_s=(2-q_s)\Delta_{\mathrm{obs}},
\qquad
\lambda_s
=
\frac{\log 2}{\eta_{\max}H_s}
=
\frac{\lambda_{\mathrm{obs}}}{2-q_s}.
\]

Interpretation: diffuse LR decay receives a two-observation half-life, while a
single sharp drop receives a one-observation half-life.  This is a schedule-only
rule, not a target-loss-fitted parameter.

## Causal Response Feature

The response is computed by the LR-time recursion

\[
z_t=\exp(-\lambda_s\eta_{s,t})z_{t-1}+d_{s,t},
\qquad
\phi_{\lambda_s,s}(t)=z_t/\eta_{\max}.
\]

Equivalently,

\[
\phi_{\lambda_s,s}(t)
=
\frac{1}{\eta_{\max}}
\sum_{u\le t}
d_{s,u}
\exp\left(
-\lambda_s\sum_{v=u+1}^{t}\eta_{s,v}
\right).
\]

The feature is causal, drop-only, and schedule-only.

## MPL-LD Tangent Projection

Cosine residual contains both transferable LR-drop response and MPL parameter
drift.  We remove the local tangent space of the MPL LR-dependent term:

\[
J_{\mathrm{LD}}(t)=
\left[
\frac{\partial L_{\mathrm{MPL}}}{\partial\log B},
\frac{\partial L_{\mathrm{MPL}}}{\partial\log C},
\frac{\partial L_{\mathrm{MPL}}}{\partial\log\beta},
\frac{\partial L_{\mathrm{MPL}}}{\partial\log\gamma}
\right].
\]

On the cosine calibration suffix \(t\ge8000\), let \(P_{\mathrm{LD}}\) be the
orthogonal projector onto this tangent space.  For each target response rate,

\[
x_s=(I-P_{\mathrm{LD}})\phi_{\lambda_s,\cos},
\qquad
y=(I-P_{\mathrm{LD}})r_{\cos}.
\]

The fitted residual amplitude is one nonnegative scalar:

\[
\widehat\kappa_s
=
\frac{\langle x_s,y\rangle_+}
{\|x_s\|_2^2+1/N_{\mathrm{cal}}}.
\]

The \(1/N_{\mathrm{cal}}\) term is a finite-sample identifiability floor.  It
replaces the older fixed \(\tau=0.05\) ridge.

## Support-Projection Locality

The locality factor is not a learned gate.  It is the energy retained after
removing the full-horizon diffuse mode from a local drop-support forcing.

Let \(H_{\mathrm{post}}=T_s-W\).  Let \(\ell_s\) be the support span of positive
LR drops after warmup.  Let \(m_s\) be the uniform density on that support and
let \(u_s\) be the uniform density on the whole post-warmup horizon.  Then

\[
a_s
=
\mathbf{1}\{D_s>0\}
\frac{\|(I-P_{u_s})m_s\|_2^2}{\|m_s\|_2^2}
=
\mathbf{1}\{D_s>0\}
\left[1-\frac{\ell_s}{H_{\mathrm{post}}}\right]_+.
\]

This explains why WSD-con single drops keep almost all correction, WSD
cooldowns keep a local-cooldown fraction, and full-horizon cosine controls are
suppressed.

## Results

Same protocol as the restored observation-bracket audit: cosine-only
calibration, MPL-LD tangent projection, sample-size ridge, no target-loss
fitting.

| variant | WSD same-scale | WSD cross-scale | controls same-scale |
|---|---:|---:|---:|
| current q_inf + support projection | -29.87%, 15/15 | -24.95%, 30/30 | 9/9 non-harm |
| q2 + support projection | -29.88%, 15/15 | -24.95%, 30/30 | 9/9 non-harm |
| q2 + half-life bracket + support projection | -29.88%, 15/15 | -24.95%, 30/30 | 9/9 non-harm |
| q2 + density projection | -30.22%, 15/15 | -24.93%, 30/30 | worst +8.25% |
| q2 without locality | -30.88%, 15/15 | -24.60%, 30/30 | worst +56.99% |
| no nuisance raw projection | +602.17%, 0/15 | not used | failure mode |

Decision: use \(q_2\), half-life bracket, support-projection locality, MPL-LD
tangent projection, and sample-size ridge as the current interpretable formula.

## Parameter Ledger

| quantity | source | residual-fitted? | target loss? |
|---|---|---:|---:|
| MPL parameters | frozen MPL baseline | 0 | outside error model |
| \(d_t,D_s,p_t,q_s\) | target LR schedule | 0 | 0 |
| \(\lambda_{\mathrm{obs}}\) | logging interval and peak LR | 0 | 0 |
| \(\lambda_s\) | \(q_s\) half-life bracket | 0 | 0 |
| \(\phi_{\lambda_s,s}\) | target LR schedule | 0 | 0 |
| \(P_{\mathrm{LD}}\) | MPL LR-dependent tangent finite differences | 0 | 0 |
| \(1/N_{\mathrm{cal}}\) | cosine suffix sample count | 0 | 0 |
| \(a_s\) | support-projection geometry | 0 | 0 |
| \(\widehat\kappa_s\) | one projection from cosine residual | 1 | 0 |

## Limits

- The half-life bracket is still a modeling assumption tied to logging
  resolution, not a theorem from first-principles optimizer dynamics.
- The calibration suffix rule remains a protocol component and must be frozen
  before external validation.
- Current evidence is internal to the available loss-curve repository.  New
  schedules or new training runs are still required before making a broad claim.
