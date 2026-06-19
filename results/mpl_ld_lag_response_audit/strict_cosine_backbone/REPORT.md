# Strict Cosine-Only Backbone Audit for MPL-LD Finite Response

This audit keeps the finite-response correction fixed and changes only the MPL backbone source.  It separates mechanism evidence from protocol evidence.

Recommended correction under audit:

\[
\hat L_s(t)=L_{\mathrm{MPL},s}(t)+a_sB_s[D_{\downarrow,\tau_s,s}(t)-D_{\downarrow,s}(t)].
\]

No residual amplitude, gate, channel selector, sinusoid, DCT basis, or target-loss-fitted parameter is used here.

## Summary

| backbone | group | correction vs own MPL | wins / non-harm | own MPL vs official MPL | corrected vs official MPL |
|---|---|---:|---:|---:|---:|
| official frozen MPL | WSD-family | -13.77% mean / -6.29% worst | 15/15 / 15/15 | +0.00% mean | -13.77% mean |
| official frozen MPL | controls | +0.00% mean / +0.00% worst | 0/9 / 9/9 | +0.00% mean | +0.00% mean |
| cosine-only MPL | WSD-family | -11.44% mean / -6.40% worst | 15/15 / 15/15 | +55.05% mean | +37.34% mean |
| cosine-only MPL | controls | +0.00% mean / +0.00% worst | 0/9 / 9/9 | +79.78% mean | +79.78% mean |

## Strict Backbone Per-Target WSD Rows

| scale | target | own MPL MAE | corrected MAE | correction delta | corrected vs official MPL |
|---:|---|---:|---:|---:|---:|
| 25 | WSD sharp | 0.004303 | 0.003868 | -10.12% | +13.48% |
| 25 | WSD linear | 0.003861 | 0.003550 | -8.04% | +13.21% |
| 25 | WSD-con 3e-5 | 0.008591 | 0.007582 | -11.75% | +68.97% |
| 25 | WSD-con 9e-5 | 0.006052 | 0.005267 | -12.97% | +79.99% |
| 25 | WSD-con 18e-5 | 0.004134 | 0.003686 | -10.85% | +47.15% |
| 100 | WSD sharp | 0.004345 | 0.003710 | -14.62% | -1.03% |
| 100 | WSD linear | 0.003618 | 0.003112 | -13.98% | -4.19% |
| 100 | WSD-con 3e-5 | 0.007928 | 0.007014 | -11.53% | +23.72% |
| 100 | WSD-con 9e-5 | 0.006257 | 0.005546 | -11.37% | +12.38% |
| 100 | WSD-con 18e-5 | 0.002671 | 0.002265 | -15.22% | +63.92% |
| 400 | WSD sharp | 0.007276 | 0.006324 | -13.09% | +34.49% |
| 400 | WSD linear | 0.006066 | 0.005219 | -13.95% | +35.44% |
| 400 | WSD-con 3e-5 | 0.015378 | 0.014395 | -6.40% | +76.11% |
| 400 | WSD-con 9e-5 | 0.009033 | 0.008260 | -8.55% | +23.52% |
| 400 | WSD-con 18e-5 | 0.004869 | 0.004424 | -9.13% | +72.96% |

## Reading

- The correction formula itself is still parameter-free on top of MPL: all new quantities come from the LR schedule, the logging interval, and MPL's own \(D_\downarrow\) term.
- The frozen-official result should be treated as a mechanism diagnostic, because the MPL backbone was not trained under a strict cosine-only split.
- The strict cosine-only rows are the fairer protocol for the assignment question.  If they are weaker than the frozen-official rows, the honest conclusion is that the current formula helps but is not yet a complete cosine-to-WSD solution.
- This audit intentionally does not add extra fitted residual parameters to recover performance; doing so would reintroduce the interpretability problem this audit is meant to expose.
