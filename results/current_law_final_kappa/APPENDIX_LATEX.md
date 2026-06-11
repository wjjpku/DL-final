# LaTeX-Ready Appendix Derivation

This file contains a compact appendix-style derivation that can be copied into
the paper with minor notation adjustments.

```latex
\paragraph{Nuisance-projected response amplitude estimator.}
For a calibration run, let
\[
    r = \ell_{\mathrm{obs}} - \ell_{\mathrm{MPL}}
\]
denote the residual after subtracting the main power-law prediction, and let
\(\phi\) denote the schedule-response feature produced by our response law. We
assume the residual admits the decomposition
\[
    r = \kappa_\star \phi + g_\star + \varepsilon,
    \qquad g_\star \in \mathcal{G}, \qquad \kappa_\star \ge 0,
\]
where \(\mathcal{G}\) is a low-frequency nuisance subspace that captures smooth
MPL residual drift. Let \(P_{\mathcal{G}}\) be the orthogonal projection onto
\(\mathcal{G}\), and define \(M_{\mathcal{G}} = I - P_{\mathcal{G}}\). We write
\[
    \phi_\perp = M_{\mathcal{G}}\phi,
    \qquad
    r_\perp = M_{\mathcal{G}}r.
\]

By the Frisch--Waugh--Lovell theorem, the least-squares coefficient of \(\phi\)
after controlling for \(\mathcal{G}\) is the coefficient obtained by regressing
\(r_\perp\) on \(\phi_\perp\). With the nonnegativity constraint on the response
amplitude, the unregularized estimate is
\[
    \widehat{\kappa}_{\mathrm{NNLS}}
    =
    \left(
    \frac{\langle \phi_\perp, r_\perp\rangle}
         {\|\phi_\perp\|_2^2}
    \right)_+ .
\]

We regularize this coefficient with an empirical-Bayes prior. Assume
\[
    \varepsilon \sim \mathcal{N}(0,\sigma^2 I),
    \qquad
    \kappa \sim \mathcal{N}_+(0,k_0^2),
\]
where \(\mathcal{N}_+\) denotes truncation to \(\kappa\ge 0\). The posterior mode
in the projected problem is
\[
    \widehat{\kappa}_{\mathrm{MAP}}
    =
    \left(
    \frac{\langle \phi_\perp, r_\perp\rangle}
         {\|\phi_\perp\|_2^2 + \tau^2}
    \right)_+,
    \qquad
    \tau = \sigma / k_0 .
\]
In our experiments, \(\tau\) is estimated by empirical Bayes from calibration
curves only.

The projected regression identifies only the response component
\(\widehat{\kappa}_{\mathrm{MAP}}\phi_\perp\). For transfer, however, the scalar
amplitude multiplies the full response feature \(\phi\). To avoid extrapolating
amplitude from the nuisance-confounded component of \(\phi\), we normalize the
identified response norm by the full feature norm:
\[
    \widehat{\kappa}
    =
    \frac{\|\widehat{\kappa}_{\mathrm{MAP}}\phi_\perp\|_2}
         {\|\phi\|_2}
    =
    \sqrt{
      \frac{\|\phi_\perp\|_2^2}{\|\phi\|_2^2}
    }
    \widehat{\kappa}_{\mathrm{MAP}} .
\]
Therefore our cap-free estimator is
\[
    \boxed{
    \widehat{\kappa}
    =
    \sqrt{
      \frac{\|M_{\mathcal{G}}\phi\|_2^2}{\|\phi\|_2^2}
    }
    \left(
    \frac{
      \langle M_{\mathcal{G}}\phi, M_{\mathcal{G}}r\rangle
    }{
      \|M_{\mathcal{G}}\phi\|_2^2 + \tau^2
    }
    \right)_+
    } .
\]
An optional capped variant imposes \(\widehat{\kappa}\le \kappa_{\max}\), which
corresponds to an additional truncated susceptibility prior. We use the
cap-free estimator as the main method.

\paragraph{A spectral definition of the nuisance subspace.}
The derivation above only requires \(\mathcal{G}\) to be a low-frequency
nuisance subspace. A schedule-agnostic implementation is obtained from a
discrete-cosine basis over the observed training points. Let
\[
    q_0(i)=1,
    \qquad
    q_j(i)=\cos\!\left(\pi j(i+\tfrac12)/n\right),\quad j\ge 1,
\]
and let
\[
    \mathcal{G}_K = \operatorname{span}\{q_0,q_1,\ldots,q_K\}.
\]
For each \(K\), define
\[
    R_K =
    \frac{\|M_{\mathcal{G}_K}\phi\|_2^2}{\|\phi\|_2^2}.
\]
The experiments show that \(R_K\) alone is not sufficient for choosing
\(\mathcal{G}\): a very small \(K\) can preserve response energy while failing
to remove MPL drift. We therefore use a two-stage bandwidth rule:
\[
    K^\star
    =
    \arg\min_{K\ge K_{\min}}
    |R_K-\rho|,
\]
where \(K_{\min}\) enforces minimum low-frequency drift control and \(\rho\)
sets the desired identifiable feature-energy fraction. In the spectral audit,
\(K_{\min}=3\) and \(\rho=0.35\) give a non-failing transfer matrix, while the
same retention-target rule without \(K_{\min}\) selects under-covered
subspaces and fails. Thus the role of \(R_K\) is to choose among sufficiently
rich nuisance spaces, not to replace the low-frequency drift-control
assumption.

\paragraph{Posterior-predictive shrinkage for finite calibration coverage.}
When several calibration schedules are available, the projected evidence can be
pooled before applying the same identifiable-amplitude conversion:
\[
    d_S = \sum_{c\in S}
    \langle M_{\mathcal{G}}\phi_c, M_{\mathcal{G}}r_c\rangle,
    \qquad
    v_S = \sum_{c\in S}\|M_{\mathcal{G}}\phi_c\|_2^2,
    \qquad
    V_S = \sum_{c\in S}\|\phi_c\|_2^2 .
\]
The pooled cap-free estimate is
\[
    \widehat{\kappa}_S
    =
    \sqrt{\frac{v_S}{V_S}}
    \left(
    \frac{d_S}{v_S+\tau^2}
    \right)_+ .
\]
To transfer this scalar amplitude to an unseen schedule, we may add a weak
random-effects layer
\[
    \kappa_c = \theta + u_c,
    \qquad
    u_c \sim \mathcal{N}(0,\sigma_{\mathrm{tr}}^2),
    \qquad
    \theta \sim \mathcal{N}_+(0,k_0^2).
\]
The posterior-predictive mean for a new schedule then has the scalar shrinkage
form
\[
    c_n = \frac{n}{n+\rho},
    \qquad
    \rho = \sigma_{\mathrm{tr}}^2/k_{0,\mathrm{eff}}^2,
\]
so the transferable amplitude is
\[
    \widehat{\kappa}_{\mathrm{transfer}}
    =
    c_n \widehat{\kappa}_S .
\]
In the current predictive-shrinkage audit, the conservative choice
\(\rho=0.5\) removes the observed WSD-con over-correction failures while
preserving useful cosine-to-WSD transfer. We therefore treat this factor as a
promising posterior-predictive extension rather than as a required component of
the main estimator.

\paragraph{Target-side identifiability for deployment.}
The transfer amplitude above should only be applied to a target schedule whose
response direction is identifiable after the same nuisance residualization.
For a target feature \(\phi_{\mathrm{tar}}\), define
\[
    R_{\mathrm{tar}}(\lambda)
    =
    \frac{\|M_{\lambda}\phi_{\mathrm{tar}}\|_2^2}
         {\|\phi_{\mathrm{tar}}\|_2^2},
\]
where \(M_{\lambda}\) is the soft DCT/Sobolev residualizer used in the
next-generation estimator. If \(R_{\mathrm{tar}}(\lambda)\) is too small, the
target response direction is nearly indistinguishable from low-frequency MPL
drift, so transferring a positive scalar amplitude is not identifiable without
target residual evidence. The deployment rule is therefore
\[
    a_{\mathrm{tar}}
    =
    \mathbf{1}\{R_{\mathrm{tar}}(\lambda)\ge 0.01\},
    \qquad
    \widehat{\kappa}_{\mathrm{safe}}
    =
    a_{\mathrm{tar}}
    \widehat{\kappa}_{\mathrm{transfer}} .
\]
Combining the finite-calibration shrinkage and the target-identifiability gate
gives the next-generation transfer rule
\[
    \widehat{\kappa}_{\mathrm{safe}}
    =
    \mathbf{1}\{R_{\mathrm{tar}}(\lambda)\ge 0.01\}
    \frac{n}{n+0.5}
    \sqrt{\frac{v_S}{V_S}}
    \left(
    \frac{d_S}{v_S+\tau^2}
    \right)_+ .
\]
The threshold \(0.01\) is an identifiability floor: in the current audit it lies
between the lowest retained main-matrix target retention and the highest
diffuse extra-holdout retention, and the resulting gate is non-harming across
all calibration train sizes. More concretely, the maximum retention among
raw-harmful target transfers is \(0.005721\), the minimum retention among
main-matrix target transfers is \(0.014797\), and the chosen floor satisfies
\[
    0.005721 < 0.01 < 0.014797 .
\]
Thus the floor is not a loss-tuned knife-edge: it is \(1.75\times\) above the
harmful diffuse-target boundary and the closest main target remains
\(1.48\times\) above the floor. Lowering the floor to \(0.005\) admits the
diffuse cosine target and restores the observed failure, while raising it above
the main cosine retention stays non-harming but drops useful main-matrix
transfers.
```

## Notes For Integration

- Replace `MPL` with the paper's exact name for the main power-law baseline.
- Define \(\mathcal{G}\) as a small low-frequency nuisance subspace; do not make
  the polynomial implementation detail the theoretical object.
- State that \(\tau\) is estimated from calibration curves only. The train-only
  tau audit confirms the single-curve conclusion does not rely on held-out test
  curves.
