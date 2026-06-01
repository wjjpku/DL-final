# Curated Curve Package

## Purpose

This folder is a compact, non-duplicated handoff of the curve figures.
It keeps only:

- 4 summary figures
- 9 representative prediction figures

The 9 representative prediction figures correspond one-to-one to the 9 curve types in the public dataset, using the 100M scale as the representative example.

## Data Source

All public curves come from the official `MultiPowerLaw` repository data:

- `external/MultiPowerLaw/loss_curve_repo/csv_25`
- `external/MultiPowerLaw/loss_curve_repo/csv_100`
- `external/MultiPowerLaw/loss_curve_repo/csv_400`

Each CSV contains:

- `step`
- `lr`
- `loss`

## Model Scales

- `25M`
- `100M`
- `400M`

## Schedule Families And Curves

### Training curves used for fitting

- `constant_24000`: warmup + constant LR to 24k steps
- `cosine_24000`: warmup + cosine decay to 24k steps
- `wsdcon_9`: warmup + two-stage constant schedule, switch at 8k steps from `3e-4` to `9e-5`

### Test curves used for prediction

- `constant_72000`: warmup + constant LR to 72k steps
- `cosine_72000`: warmup + cosine decay to 72k steps
- `wsd_20000_24000`: warmup + plateau + WSD decay from 20k to 24k
- `wsdld_20000_24000`: warmup + plateau + linear decay from 20k to 24k
- `wsdcon_3`: warmup + two-stage constant schedule, switch at 8k from `3e-4` to `3e-5`
- `wsdcon_18`: warmup + two-stage constant schedule, switch at 8k from `3e-4` to `1.8e-4`

## What Each Figure Is Used For

### Summary figures

- `01_train_set_overview.png`: shows the official training set curves and LR schedules across all 3 scales
- `02_dataset_structure.png`: shows how many sampled points each public curve contains
- `03_avg_train_mae_compare.png`: compares train-set fit quality of `MPL` and `Tissue`
- `04_avg_test_mae_compare.png`: compares test-set generalization of `MPL` and `Tissue`

### Representative curve figures

- `05_constant_24000_example.png`: training curve type `constant_24000`
- `06_cosine_24000_example.png`: training curve type `cosine_24000`
- `07_wsdcon_9_example.png`: training curve type `wsdcon_9`
- `08_constant_72000_example.png`: test curve type `constant_72000`
- `09_cosine_72000_example.png`: test curve type `cosine_72000`
- `10_wsd_20000_24000_example.png`: test curve type `wsd_20000_24000`
- `11_wsdld_20000_24000_example.png`: test curve type `wsdld_20000_24000`
- `12_wsdcon_3_example.png`: test curve type `wsdcon_3`
- `13_wsdcon_18_example.png`: test curve type `wsdcon_18`

## Figure Convention

For representative prediction figures:

- black line: ground truth
- orange dashed line: MPL
- blue dash-dot line: Tissue

## Why 100M For Representative Curves

The 100M scale is the middle scale and is used here as the single representative scale to avoid duplication while still showing every curve type exactly once.
