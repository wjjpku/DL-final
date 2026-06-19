# GitHub Release Checklist

Use this checklist before the final commit / push.

## Must Pass

```bash
python3 repro/verify_release.py
python3 repro/schedule_response_robustness_audit.py
cd slides
xelatex -interaction=nonstopmode -halt-on-error main_zh.tex
xelatex -interaction=nonstopmode -halt-on-error main_zh.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

After staging the final release files, run the strict git-index gate:

```bash
python3 repro/verify_release.py --require-index
```

To avoid accidentally committing historical scratch files, generate the exact
release staging command instead of using `git add .`:

```bash
python3 repro/verify_release.py --print-git-add
```

If `results/` already contains tracked exploratory dumps, clean the result index
first and then re-add only the release allowlist:

```bash
git rm -r --cached results
python3 repro/verify_release.py --print-git-add | sh
python3 repro/verify_release.py --require-index
```

Expected slide counts:

- `slides/main_zh.pdf`: 36 pages.
- `slides/main.pdf`: 38 pages.

## Files That Must Be Committed

Current public-facing entry files:

- `README.md`
- `.gitignore`
- `FINAL_DELIVERABLES.md`
- `REPRODUCIBILITY.md`
- `DATA_MANIFEST.md`
- `RELEASE_CHECKLIST.md`
- `requirements.txt`

Main code:

- `repro/schedule_response_robustness_audit.py`
- `repro/reproduce_cosine_to_wsd.py`
- `repro/interpretable_error_model.py`
- `repro/interpretable_nuisance_origin_audit.py`
- `repro/interpretable_observation_bracket_audit.py`
- `repro/verify_release.py`
- `repro/README.md`

Main outputs:

- `results/schedule_response_robustness/`
- `results/tables/cosine_to_wsd_metrics.csv`
- `results/tables/fitted_params.json`
- `results/figures/avg_test_mae.png`
- `results/figures/avg_test_rmse.png`
- `results/README.md`
- `slides/main_zh.tex`
- `slides/main_zh.pdf`
- `slides/main.tex`
- `slides/main.pdf`
- `slides/figs/fig_mpl_residual_anomaly_100M.png`
- `slides/figs/fig_projection_decomposition_cosine_100M.png`
- `slides/figs/fig_projection_ablation_time_errors_100M.png`
- `slides/figs/fig_schedule_response_mae_heatmap.png`
- `slides/figs/fig_schedule_response_time_errors_100M.png`
- `slides/figs/fig_kappa_clean_scatter.png`

Data:

- `external/MultiPowerLaw/loss_curve_repo/`

## Do Not Commit

- `represent/data/`
- `represent/MultiPowerLaw/`
- LaTeX intermediates: `*.aux`, `*.log`, `*.nav`, `*.snm`, `*.toc`, `*.out`
- Python caches: `__pycache__/`, `*.pyc`
- Machine-specific helper files under `tools/`

## Current Scope Statement

The GitHub-facing claim is the residual-identification story:

```text
frozen MPL baseline
-> cosine residual is confounded
-> MPL-LD projection removes nuisance drift
-> source-only kappa transfers LR-drop response to WSD-family schedules
```

Historical directories such as `current_law_*`, `step_time_*`, and
`cosine_to_wsd_response_search/` may be retained for provenance, but they should
not be committed under `results/` in the GitHub-facing release.
