# DL-final

**From Cosine to WSD: Identifying Transferable LR-Drop Response in MPL Residuals**

This repository is a compact reproduction package for the slides' core
experiments.  It contains the public CSV loss curves, the minimal reproduction
scripts, the slide decks, and the small set of reports / figures needed to
verify the result.

## What This Project Studies

Learning-rate schedules change the shape of a training loss curve.  The
question here is deliberately concrete:

> Given a source cosine loss curve and a target WSD-family learning-rate
> schedule, can we predict the target WSD loss curve without using target WSD
> loss for calibration?

The baseline is MPL.  MPL is already strong, but it leaves systematic residuals
near WSD transition and tail regions.  The key difficulty is that the cosine
residual is not a pure schedule-response signal.  It mixes:

```text
transferable LR-drop response
+ non-transferable MPL-LD parameter drift
```

The project treats cosine-to-WSD transfer as an **identification problem**:
remove the MPL-LD nuisance component first, then transfer only the identified
LR-drop response.

<p align="center">
  <img src="slides/figs/fig_mpl_residual_anomaly_100M.png" width="47%" alt="MPL residual anomaly near WSD transition and tail">
  <img src="slides/figs/fig_projection_decomposition_cosine_100M.png" width="47%" alt="Projection decomposition of cosine residual">
</p>

The prediction rule is intentionally low capacity:

```text
L_hat_s(t) = L_MPL,s(t) + kappa_hat_s * phi_{lambda_s,s}(t)
```

- `L_MPL,s(t)` is the frozen MPL baseline.
- `phi_{lambda_s,s}(t)` is a causal LR-drop response feature computed from the
  target LR schedule.
- `kappa_hat_s` is the only residual-fitted scalar, estimated from the source
  cosine residual after MPL-LD projection.
- Target WSD loss is used only for evaluation and oracle diagnostics.

## Main Evidence

The main deployable setting is same-scale cosine source to WSD-family targets
across `25M`, `100M`, and `400M`.

<p align="center">
  <img src="slides/figs/fig_schedule_response_mae_heatmap.png" width="47%" alt="MAE improvement heatmap">
  <img src="slides/figs/fig_kappa_clean_scatter.png" width="47%" alt="Source kappa versus target oracle kappa">
</p>

| Evidence | Result |
|---|---:|
| Same-scale WSD-family mean MAE change vs MPL | `-30.88%` |
| Worst WSD-family row | `-4.67%` |
| WSD-family wins | `15/15` |
| Projected `kappa_hat` vs target oracle `kappa_star` Pearson | `+0.910` |
| No-projection negative control | `+625.92%`, `0/15` wins |
| Leave-one-scale-out mean-kappa transfer | `-25.62%`, `15/15` wins |

The negative control is important: directly fitting the raw cosine residual
without MPL-LD projection over-transfers low-frequency MPL drift.  The result is
not just "add one correction term"; the useful part is identifying which
residual component can transfer.

## How To Use This Repository

Start with the slides:

1. `slides/main_zh.pdf` is the Chinese standalone presentation.
2. `slides/main.pdf` is the English version with the explicit Tissue/Momentum
   baseline reproduction section.
3. `results/schedule_response_robustness/REPORT.md` contains the main tables.
4. `results/schedule_response_robustness/LEAKAGE_AUDIT.md` states what does and
   does not read target WSD loss.

Install dependencies:

```bash
pip install -r requirements.txt
```

Regenerate the main projected-kappa audit:

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

Verify that the repository is still the minimal release package:

```bash
python3 repro/verify_release.py --require-index
```

## Data Included

The public-curve CSV data used by the slides is vendored under:

```text
external/MultiPowerLaw/loss_curve_repo/csv_25/
external/MultiPowerLaw/loss_curve_repo/csv_100/
external/MultiPowerLaw/loss_curve_repo/csv_400/
```

Each scale contains:

- `cosine_72000.csv`: source curve for projected residual calibration;
- `wsd_20000_24000.csv`: WSD sharp cooldown target;
- `wsdld_20000_24000.csv`: WSD linear decay target;
- `wsdcon_3.csv`, `wsdcon_9.csv`, `wsdcon_18.csv`: WSD-con targets;
- `cosine_24000.csv`, `constant_24000.csv`, `constant_72000.csv`: auxiliary
  public curves used by baseline scripts.

No expensive transformer training is required for the public reproduction path.

## Minimal Release Layout

| Path | Purpose |
|---|---|
| `slides/` | Chinese and English Beamer slide decks plus the figures used in them. |
| `repro/` | Minimal scripts needed to reproduce the baseline and projected-kappa audit. |
| `results/schedule_response_robustness/` | Main report, leakage audit, and summary CSVs. |
| `results/tables/`, `results/figures/` | Baseline reproduction metrics and small summary plots. |
| `external/MultiPowerLaw/loss_curve_repo/` | Public CSV loss curves used by the scripts. |
| `REPRODUCIBILITY.md` | Command-level reproduction guide. |
| `DATA_MANIFEST.md` | Exact data boundary. |
| `RELEASE_CHECKLIST.md` | Minimal push checklist and release allowlist. |

This repository intentionally excludes exploratory result dumps, paper drafts,
independent training branches, and old search scripts.  The release verifier
fails if those files are still tracked.

## Scope

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
