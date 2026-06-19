# MPL-LD Finite-Response Audit

This audit tests the most directly interpretable contraction of the residual model: replace MPL's own LR-dependent term \(D(t)\) by a causal lagged term \(D_\tau(t)\), rather than adding a new residual basis.

## Formula

Original MPL:

\[
L_{\mathrm{MPL}}(t)=L_0+A S(t)^{-\alpha}+B D(t).
\]

Finite-response variant:

\[
D_\tau(t_i)=\rho_iD_\tau(t_{i-1})+(1-\rho_i)D(t_i),\quad \rho_i=\exp[-(t_i-t_{i-1})/\tau].
\]

\[
\hat L_\tau(t)=L_{\mathrm{MPL}}(t)+B[D_\tau(t)-D(t)].
\]

The fixed-tau rows introduce no fitted residual coefficient.  The cosine-fit-amplitude rows are included as a contamination check, not as a recommended method.

## Fixed-Tau Direct Results

| tau steps | WSD mean / worst / wins | controls mean / worst / nonharm |
|---:|---:|---:|
| 32 | -0.42% / -0.28% / 15/15 | +0.01% / +0.19% / 5/9 |
| 64 | -3.11% / -2.38% / 15/15 | +0.04% / +1.62% / 5/9 |
| 128 | -9.52% / -5.95% / 15/15 | +0.20% / +6.01% / 6/9 |
| 256 | -15.09% / +18.57% / 14/15 | +1.31% / +15.90% / 6/9 |
| 512 | -6.59% / +92.63% / 10/15 | +7.74% / +36.20% / 4/9 |
| 1024 | +45.32% / +241.12% / 4/15 | +26.02% / +103.07% / 4/9 |
| 2048 | +164.24% / +506.69% / 0/15 | +63.18% / +250.27% / 3/9 |
| 4096 | +326.76% / +904.24% / 0/15 | +127.01% / +489.68% / 2/9 |
| 8192 | +479.51% / +1289.00% / 0/15 | +216.99% / +788.19% / 1/9 |
| 16384 | +589.29% / +1561.60% / 0/15 | +314.54% / +1058.50% / 0/9 |

## Cooldown-Only Direct Results

This variant decomposes MPL's \(D(t)\) by LR-change sign and lags only the LR-decrease contribution \(D_{\downarrow}(t)\):

\[
\hat L_\tau(t)=L_{\mathrm{MPL}}(t)+B[D_{\downarrow,\tau}(t)-D_{\downarrow}(t)].
\]

| tau steps | WSD mean / worst / wins | controls mean / worst / nonharm |
|---:|---:|---:|
| 32 | -0.42% / -0.28% / 15/15 | +0.01% / +0.21% / 8/9 |
| 64 | -3.10% / -2.30% / 15/15 | +0.05% / +1.76% / 8/9 |
| 128 | -9.44% / -6.22% / 15/15 | +0.26% / +6.53% / 8/9 |
| 256 | -14.90% / +14.96% / 14/15 | +1.52% / +17.28% / 8/9 |
| 512 | -6.22% / +81.91% / 9/15 | +8.25% / +39.36% / 6/9 |
| 1024 | +45.23% / +215.65% / 5/15 | +26.70% / +101.29% / 6/9 |
| 2048 | +160.75% / +462.20% / 0/15 | +62.96% / +251.69% / 6/9 |
| 4096 | +318.58% / +858.34% / 0/15 | +123.97% / +505.62% / 6/9 |
| 8192 | +470.81% / +1274.87% / 0/15 | +203.25% / +836.34% / 6/9 |
| 16384 | +582.14% / +1593.52% / 0/15 | +278.09% / +1148.71% / 6/9 |

## Cooldown + Adiabatic Boundary Results

This variant keeps the same cooldown-only MPL term, then applies a schedule-support attenuation
\(a_s=[1-\ell_\downarrow/(T-W)]_+\).  The factor is not fitted; it encodes the boundary that a full-horizon diffuse LR decay should be treated as quasi-adiabatic, not as a local cooldown transient.

\[
\hat L_\tau(t)=L_{\mathrm{MPL}}(t)+a_sB[D_{\downarrow,\tau}(t)-D_{\downarrow}(t)].
\]

| tau steps | WSD mean / worst / wins | controls mean / worst / nonharm |
|---:|---:|---:|
| 32 | -0.40% / -0.23% / 15/15 | +0.00% / +0.00% / 9/9 |
| 64 | -2.91% / -1.96% / 15/15 | +0.00% / +0.00% / 9/9 |
| 128 | -8.73% / -6.22% / 15/15 | +0.00% / +0.00% / 9/9 |
| 256 | -13.52% / +14.96% / 14/15 | +0.00% / +0.00% / 9/9 |
| 512 | -6.79% / +81.89% / 9/15 | +0.00% / +0.00% / 9/9 |
| 1024 | +38.14% / +215.61% / 7/15 | +0.00% / +0.00% / 9/9 |
| 2048 | +148.57% / +462.13% / 0/15 | +0.00% / +0.00% / 9/9 |
| 4096 | +302.33% / +858.21% / 0/15 | +0.00% / +0.00% / 9/9 |
| 8192 | +451.70% / +1274.68% / 0/15 | +0.00% / +0.00% / 9/9 |
| 16384 | +561.31% / +1593.28% / 0/15 | +0.00% / +0.00% / 9/9 |

## Cooldown + Support-Bracket Tau Results

This is the cleanest current candidate.  It keeps the cooldown-only MPL term and adiabatic boundary, but replaces fixed \(\tau\) by a schedule-only observation bracket:

\[
\tau_s=\Delta_{\mathrm{obs}}\left(1+\min\left(1,\frac{\ell_\downarrow}{\Delta_{\mathrm{obs}}}\right)\right).
\]

A single-step drop receives nearly one observed interval; a cooldown that lasts at least one observed interval receives two observed intervals.  No loss values are used.

| tau rule | WSD mean / worst / wins | controls mean / worst / nonharm |
|---|---:|---:|
| support bracket | -13.77% / -6.29% / 15/15 | +0.00% / +0.00% / 9/9 |

## Cosine-Fit Amplitude Check

| variant | tau steps | same-scale WSD | cross-scale WSD | same-scale controls |
|---|---:|---:|---:|---:|
| cosine_fit_amplitude_mpl_ld_lag | 64 | +586.87% / +1391.47% / 0/15 | +569.04% / +1588.46% / 0/30 | +161.82% / +656.49% / 2/9 |
| cosine_fit_amplitude_mpl_ld_lag | 128 | +565.16% / +1314.46% / 0/15 | +548.27% / +1500.39% / 0/30 | +161.33% / +656.40% / 2/9 |
| cosine_fit_amplitude_mpl_ld_lag | 256 | +552.71% / +1287.72% / 0/15 | +535.75% / +1468.79% / 0/30 | +160.36% / +656.15% / 2/9 |
| cosine_fit_amplitude_mpl_ld_lag | 1024 | +509.26% / +1260.93% / 0/15 | +492.19% / +1430.84% / 0/30 | +155.00% / +644.15% / 3/9 |
| cosine_fit_amplitude_mpl_ld_lag | 4096 | +333.61% / +1014.74% / 0/15 | +322.55% / +1123.83% / 0/30 | +132.34% / +551.34% / 2/9 |
| cosine_fit_amplitude_mpl_ld_lag | 16384 | +96.11% / +443.03% / 2/15 | +87.15% / +475.39% / 8/30 | +73.13% / +298.57% / 2/9 |
| cosine_fit_amplitude_cooldown_mpl_ld_lag | 64 | +546.90% / +1242.72% / 0/15 | +529.95% / +1420.46% / 0/30 | +145.00% / +641.45% / 0/9 |
| cosine_fit_amplitude_cooldown_mpl_ld_lag | 128 | +525.54% / +1166.45% / 0/15 | +509.60% / +1333.22% / 0/30 | +144.99% / +641.52% / 0/9 |
| cosine_fit_amplitude_cooldown_mpl_ld_lag | 256 | +513.70% / +1140.61% / 0/15 | +497.85% / +1302.86% / 0/30 | +144.94% / +641.48% / 0/9 |
| cosine_fit_amplitude_cooldown_mpl_ld_lag | 1024 | +473.78% / +1122.59% / 0/15 | +458.54% / +1276.86% / 0/30 | +143.52% / +636.10% / 0/9 |
| cosine_fit_amplitude_cooldown_mpl_ld_lag | 4096 | +319.45% / +953.14% / 0/15 | +309.81% / +1067.60% / 0/30 | +126.40% / +563.33% / 0/9 |
| cosine_fit_amplitude_cooldown_mpl_ld_lag | 16384 | +97.28% / +444.54% / 4/15 | +90.40% / +486.61% / 11/30 | +70.14% / +315.83% / 0/9 |

## Reading

- The direct finite-response modification has a real WSD signal: around one observation interval, it improves all WSD-family rows.
- It is not yet a final method.  Larger tau values create severe WSD failures, and even `tau=128` harms extra controls.
- Fitting an amplitude from cosine residuals is a negative control: it strongly over-transfers, which confirms that cosine residual contamination remains the central difficulty.
- The cooldown-only decomposition tests whether the control harm comes from lagging the wrong part of MPL's LD term.  It remains inside MPL's own formula because it only splits \(D(t)\) by the sign of \(\Delta\eta\).
- The adiabatic boundary is the only extra schedule-level assumption in the strongest safe row.  It restores constant and full-horizon cosine controls, but should be presented as a boundary condition rather than a learned mechanism.
- This audit should replace broad residual-basis search as the next interpretable baseline.  If a final method is built, it should modify or constrain \(D_\tau\), not add gates, channels, sinusoids, or generic DCT bases as the main story.
