# Reproducibility Guide

This guide describes how to reproduce the current public-facing results from
the committed data and code.  It does not require GPU training.

## Environment

Use Python 3.10+ when possible.

```bash
pip install -r requirements.txt
```

The main audit uses NumPy, SciPy, Matplotlib, and standard-library CSV/Path
utilities.  Pandas is included for lightweight table checks.

For slides:

- Chinese deck: XeLaTeX.
- English deck: pdfLaTeX or XeLaTeX.

## Data

The main loss curves are vendored here:

```text
external/MultiPowerLaw/loss_curve_repo/csv_25/
external/MultiPowerLaw/loss_curve_repo/csv_100/
external/MultiPowerLaw/loss_curve_repo/csv_400/
```

Each scale contains:

- `cosine_72000.csv`: source curve for projected residual calibration.
- `wsd_20000_24000.csv`: WSD sharp cooldown target.
- `wsdld_20000_24000.csv`: WSD linear decay target.
- `wsdcon_3.csv`: WSD-con final LR `3e-5`.
- `wsdcon_9.csv`: WSD-con final LR `9e-5`.
- `wsdcon_18.csv`: WSD-con final LR `18e-5`.

`constant_24000.csv`, `constant_72000.csv`, and `cosine_24000.csv` are also
included because some reproduction and baseline scripts use them.

See `DATA_MANIFEST.md` for the complete data boundary.

## Main Result Reproduction

Run:

```bash
python3 repro/schedule_response_robustness_audit.py
```

This regenerates committed summary files and local intermediate files:

```text
results/schedule_response_robustness/REPORT.md
results/schedule_response_robustness/LEAKAGE_AUDIT.md
results/schedule_response_robustness/*_summary.csv
results/schedule_response_robustness/window_rule.csv
results/schedule_response_robustness/wsdcon_failure_slice.csv
slides/figs/fig_mpl_residual_anomaly_100M.png
slides/figs/fig_projection_decomposition_cosine_100M.png
slides/figs/fig_projection_ablation_time_errors_100M.png
slides/figs/fig_schedule_response_mae_heatmap.png
slides/figs/fig_schedule_response_time_errors_100M.png
slides/figs/fig_kappa_clean_scatter.png
```

The script may also write detailed local CSVs and intermediate figures under
`results/schedule_response_robustness/`; those are ignored by the release
allowlist because the summary files above are enough to reproduce the slides.

Expected headline numbers in `REPORT.md`:

| Metric | Expected value |
|---|---:|
| q2 half-life mean MAE change | `-30.88%` |
| q2 half-life worst row | `-4.67%` |
| q2 half-life wins | `15/15` |
| projected kappa vs oracle Pearson | `+0.910` |
| no-projection negative control | `+625.92%`, `0/15` wins |

## Baseline Reproduction: MPL and Tissue/Momentum

Run:

```bash
python3 repro/reproduce_cosine_to_wsd.py
```

The script fits CPU-friendly MPL and Tissue et al. / Momentum Law baselines on
public cosine curves and evaluates WSD-family targets.  It is useful for
checking baseline behavior, but the current slides use the frozen MPL
residual-identification audit as the main story.

To restrict scales:

```bash
python3 repro/reproduce_cosine_to_wsd.py --scales 25 100 400
```

Committed baseline summary outputs:

```text
results/tables/cosine_to_wsd_metrics.csv
results/tables/fitted_params.json
results/figures/avg_test_mae.png
results/figures/avg_test_rmse.png
```

The script also writes per-curve prediction CSVs locally under
`results/predictions/`; those are regenerable and are not part of the release
allowlist.

Expected aggregate values over the 15 WSD-family test targets:

| Baseline | Cosine train MAE | WSD test MAE | WSD test RMSE | WSD test R2 |
|---|---:|---:|---:|---:|
| Tissue/Momentum | `0.002242` | `0.007493` | `0.010415` | `0.995223` |
| MPL | `0.003517` | `0.006292` | `0.009848` | `0.995760` |

## Compile Slides

Chinese:

```bash
cd slides
xelatex -interaction=nonstopmode -halt-on-error main_zh.tex
xelatex -interaction=nonstopmode -halt-on-error main_zh.tex
```

English:

```bash
cd slides
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Expected:

- `slides/main_zh.pdf`: 36 pages.
- `slides/main.pdf`: 38 pages.

## Target-Loss Leakage Boundary

Deployable prediction uses:

```text
source cosine residual -> projected kappa
target LR schedule -> response feature phi
frozen MPL + kappa * phi -> target prediction
target loss -> evaluation only
```

Target WSD loss is not used for calibration.  It is used only for evaluation and
for oracle diagnostics such as `kappa_star`.

## Release Verification

The release package is intentionally minimal.  Historical exploration scripts,
large local result dumps, paper drafts, and independent training branches are
not part of the GitHub-facing submission.

The lightweight verification path is:

```bash
python3 repro/verify_release.py --require-index
python3 repro/schedule_response_robustness_audit.py
cd slides && xelatex -interaction=nonstopmode -halt-on-error main_zh.tex
```
