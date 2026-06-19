# DL-final

**From Cosine to WSD: Identifying Transferable LR-Drop Response in MPL Residuals**

This repository contains the minimal code, public CSV data, slides, and
release-facing audit artifacts needed to reproduce the slides' core experiments
for a deep-learning course project on loss-curve prediction under learning-rate
schedule changes.

## Project Thesis

The main question is:

> Given a source cosine training loss curve and a target WSD-family learning-rate
> schedule, can we predict the target WSD loss curve without using any target
> WSD loss for calibration?

The key observation is that a strong MPL baseline still leaves structured
residuals near WSD transition / tail regions.  However, the source cosine
residual is not a pure schedule response.  It mixes two components:

```text
transferable LR-drop response
+ non-transferable MPL-LD parameter drift
```

The current method treats cosine-to-WSD transfer as a residual identification
problem.  It projects out the MPL learning-rate-dependent tangent nuisance and
only transfers the remaining LR-drop response component:

```text
L_hat_s(t) = L_MPL,s(t) + kappa_hat_s * phi_{lambda_s,s}(t)
```

where:

- `L_MPL,s(t)` is the frozen MPL baseline.
- `phi_{lambda_s,s}(t)` is a causal LR-drop response shape computed only from
  the target LR schedule.
- `lambda_s` is fixed by a schedule-only `q2` half-life rule.
- `kappa_hat_s` is the only residual-fitted scalar, estimated from source
  cosine residual after MPL-LD projection.
- Target WSD loss is used only for evaluation and oracle diagnostics.

## Headline Evidence

Main deployable setting: same-scale cosine source to WSD-family targets across
`25M`, `100M`, and `400M`.

| Evidence | Result |
|---|---:|
| Same-scale WSD-family mean MAE change vs MPL | `-30.88%` |
| Worst WSD-family row | `-4.67%` |
| WSD-family wins | `15/15` |
| Projected `kappa_hat` vs target oracle `kappa_star` Pearson | `+0.910` |
| No-projection negative control | `+625.92%`, `0/15` wins |
| Leave-one-scale-out mean-kappa transfer | `-25.62%`, `15/15` wins |

The negative control is central: directly fitting cosine residual without
MPL-LD projection catastrophically over-transfers low-frequency MPL drift.  The
project is therefore not just adding a correction term; it identifies which part
of the residual is transferable.

## What To Read First

1. `slides/main_zh.pdf`  
   Chinese standalone presentation of the current story.
2. `results/schedule_response_robustness/REPORT.md`  
   Main tables for lambda sensitivity, kernel ablation, cross-scale transfer,
   calibration window audit, and WSD-con failure mode.
3. `results/schedule_response_robustness/LEAKAGE_AUDIT.md`  
   Checklist of which quantities use target WSD loss.
4. `REPRODUCIBILITY.md`  
   Exact commands, data boundary, generated outputs, and expected checks.
5. `DATA_MANIFEST.md`  
   Included public-curve data and generated-result boundary.
6. `RELEASE_CHECKLIST.md`  
   Final commit / push gate for the minimal release package.
7. `FINAL_DELIVERABLES.md`  
   Current package checklist and verification commands.

## Quick Reproduction

Install dependencies:

```bash
pip install -r requirements.txt
```

Regenerate the main robustness audit, CSVs, reports, and figures:

```bash
python3 repro/schedule_response_robustness_audit.py
```

Reproduce the MPL and Tissue et al. / Momentum Law baseline comparison:

```bash
python3 repro/reproduce_cosine_to_wsd.py --scales 25 100 400
```

Compile the slides:

```bash
cd slides
xelatex -interaction=nonstopmode main_zh.tex
xelatex -interaction=nonstopmode main_zh.tex
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

These commands do not rerun expensive transformer training.  They reproduce the
main public-curve analysis from committed data and regenerate the presentation
figures used by the current slides.

## Data Included

The core public-curve data needed for the main results is vendored in:

```text
external/MultiPowerLaw/loss_curve_repo/csv_25/
external/MultiPowerLaw/loss_curve_repo/csv_100/
external/MultiPowerLaw/loss_curve_repo/csv_400/
```

Each scale contains the cosine source and WSD-family target curves used in the
main audit:

- `cosine_72000.csv`
- `wsd_20000_24000.csv`
- `wsdld_20000_24000.csv`
- `wsdcon_3.csv`
- `wsdcon_9.csv`
- `wsdcon_18.csv`

Only the public CSV curves needed for the slides' core experiments are part of
the release package.  Raw transformer-training bytes and historical search
outputs are intentionally not committed.

## Repository Layout

| Path | Purpose |
|---|---|
| `slides/` | Chinese and English Beamer slide decks; `main_zh.pdf` is the current presentation. |
| `repro/` | Minimal scripts needed to reproduce the baseline and projected-kappa audit. |
| `results/schedule_response_robustness/` | Current main audit report, leakage audit, and summary CSVs. |
| `results/tables/`, `results/figures/` | Baseline reproduction metrics and small summary plots. |
| `external/MultiPowerLaw/loss_curve_repo/` | Vendored public CSV loss curves used by the scripts. |
| `DATA_MANIFEST.md` | Data boundary for GitHub reproduction. |
| `RELEASE_CHECKLIST.md` | Final push checklist and required file set. |

## Main Generated Figures

The current slides use these regenerated figures:

- `slides/figs/fig_mpl_residual_anomaly_100M.png`
- `slides/figs/fig_projection_decomposition_cosine_100M.png`
- `slides/figs/fig_projection_ablation_time_errors_100M.png`
- `slides/figs/fig_schedule_response_mae_heatmap.png`
- `slides/figs/fig_schedule_response_time_errors_100M.png`
- `slides/figs/fig_kappa_clean_scatter.png`

## Scope And Limitations

Supported claim:

- On the committed WSD-family public curves, source-only projected cosine
  calibration identifies a transferable LR-drop response component on top of a
  frozen MPL baseline.
- MPL-LD projection is necessary; raw cosine residual transfer fails.
- The deployable rule is low capacity: one residual-fitted scalar `kappa_hat`
  per source scale / response shape.

Not claimed:

- A universal training-loss law.
- A universal constant calibration window.
- Fully solved WSD-con final-LR ranking.
- Prospective validation on newly trained held-out WSD schedules.

The most important next experiment is not another ablation on existing curves,
but new WSD-family training runs after freezing the protocol:

```text
t >= 8000, q2 half-life, MPL-LD projection, 1/N_cal ridge floor, kappa >= 0
```
