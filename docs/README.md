# Documentation Index

This directory is an index for readers who want more context than the slides.
The current public-facing story is the MPL residual-identification line:

```text
frozen MPL baseline
-> cosine residual is confounded
-> MPL-LD tangent projection removes nuisance drift
-> source-only kappa transfers LR-drop response to WSD-family schedules
```

For the shortest path, read these files in order:

1. `../README.md`
2. `../slides/main_zh.pdf`
3. `../results/schedule_response_robustness/REPORT.md`
4. `../results/schedule_response_robustness/LEAKAGE_AUDIT.md`
5. `../REPRODUCIBILITY.md`

## Current Main Artifacts

| File or directory | Role |
|---|---|
| `../slides/main_zh.pdf` | Chinese standalone presentation of the current work. |
| `../slides/main.pdf` | English version of the same presentation. |
| `../paper/main.pdf` | Technical report draft; slides are the most up-to-date narrative. |
| `../repro/schedule_response_robustness_audit.py` | Main audit script for projected kappa response. |
| `../repro/reproduce_cosine_to_wsd.py` | MPL/Tissue baseline reproduction script. |
| `../results/schedule_response_robustness/REPORT.md` | Main robustness tables and interpretation. |
| `../results/schedule_response_robustness/LEAKAGE_AUDIT.md` | Target-loss usage audit. |
| `../external/MultiPowerLaw/loss_curve_repo/` | Public loss curves used by the main audit. |

## Method Notes

| Topic | Where to look |
|---|---|
| Residual decomposition | `slides/main_zh.tex`, method slides. |
| MPL-LD projection | `repro/interpretable_observation_bracket_audit.py`, `repro/schedule_response_robustness_audit.py`. |
| LR-drop response feature | `repro/schedule_response_robustness_audit.py`. |
| Kappa / oracle diagnostics | `results/schedule_response_robustness/REPORT.md`. |
| Leakage boundary | `results/schedule_response_robustness/LEAKAGE_AUDIT.md`. |
| WSD-con failure mode | `results/schedule_response_robustness/wsdcon_failure_slice.csv`. |

## Reproduction And Data

| File | Content |
|---|---|
| `../REPRODUCIBILITY.md` | End-to-end commands and expected outputs. |
| `../repro/README.md` | Script entry points and historical-script policy. |
| `../results/README.md` | Result directory index. |
| `../requirements.txt` | Python dependency list. |

## Historical Documents

The repository contains many exploration reports from earlier modeling routes:
`current_law_*`, `step_time_*`, `cosine_to_wsd_response_search`, and related
directories under `results/`.  They are retained for provenance and auditability,
but they are not the current public-facing claim.

Older derivation documents under `docs/core/` and `docs/explorations/` should
be read as historical background unless explicitly referenced by the current
slides or `results/schedule_response_robustness/REPORT.md`.
