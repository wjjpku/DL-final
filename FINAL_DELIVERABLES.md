# Final Deliverables Checklist

Date: 2026-06-19

This file is the course-facing checklist for the repository.  The current
deliverable is the residual-identification version of the project:

```text
cosine source residual
-> remove MPL-LD tangent nuisance
-> estimate one source-only kappa
-> predict WSD-family loss with schedule-only LR-drop response
```

## 1. Submitted Artifacts

| Category | Files | Status |
|---|---|---|
| Chinese slides | `slides/main_zh.pdf`, `slides/main_zh.tex` | Current main presentation, 36 pages. |
| English slides | `slides/main.pdf`, `slides/main.tex` | English version of the residual-identification story, with an explicit Tissue/Momentum baseline reproduction section, 38 pages. |
| Main audit report | `results/schedule_response_robustness/REPORT.md` | Main tables for robustness, ablations, cross-scale transfer, and limitations. |
| Leakage audit | `results/schedule_response_robustness/LEAKAGE_AUDIT.md` | Explicit statement of which stages read target loss. |
| Reproduction guide | `REPRODUCIBILITY.md` | Commands, data boundary, generated outputs, and expected verification. |
| Code index | `repro/README.md` | Minimal scripts needed to reproduce the slides' core experiments. |
| Results index | `results/README.md` | Release-facing result files only. |
| Top-level README | `README.md` | GitHub entry point and project summary. |

## 2. Core Claim

Given a source cosine loss curve and a target WSD-family schedule, this project
predicts the WSD loss curve without using target WSD loss for calibration.

The key technical point is that the cosine residual is confounded:

```text
transferable LR-drop response
+ non-transferable MPL-LD parameter drift
```

The method uses MPL-LD tangent projection to remove the nuisance component, then
fits one nonnegative scalar `kappa_hat` from the projected source cosine
residual.  The target correction shape `phi` is computed only from the target
learning-rate schedule.

## 3. Main Evidence

Main setting: same-scale cosine source to WSD-family targets over `25M`, `100M`,
and `400M`.

| Evidence | Result |
|---|---:|
| Same-scale WSD-family mean MAE change vs MPL | `-30.88%` |
| Worst WSD-family row | `-4.67%` |
| WSD-family wins | `15/15` |
| Projected `kappa_hat` vs target oracle `kappa_star` Pearson | `+0.910` |
| No-projection negative control | `+625.92%`, `0/15` wins |
| Leave-one-scale-out mean-kappa transfer | `-25.62%`, `15/15` wins |

These results support the identification story: direct residual transfer fails,
while projected residual transfer is stable on the committed WSD-family curves.

## 4. Required Course Items

| Requirement | Covered by |
|---|---|
| Problem background and goal | `slides/main_zh.pdf` slides 1--3; `README.md`. |
| Data processing and experimental setup | `slides/main_zh.pdf` slides 4--6; `REPRODUCIBILITY.md`. |
| Reproduction of existing method | `slides/main_zh.pdf` slide 6; `repro/reproduce_cosine_to_wsd.py`. |
| Proposed method and result comparison | `slides/main_zh.pdf` slides 8--20; `results/schedule_response_robustness/REPORT.md`. |
| Analysis and discussion | `slides/main_zh.pdf` slides 21--24; limitations in `README.md`. |
| Code and division | final slide of both decks. |

## 5. Quick Verification Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Regenerate the main audit:

```bash
python3 repro/schedule_response_robustness_audit.py
```

Compile slides:

```bash
cd slides
xelatex -interaction=nonstopmode -halt-on-error main_zh.tex
xelatex -interaction=nonstopmode -halt-on-error main_zh.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## 6. Data Boundary

The main public-curve data is included under:

```text
external/MultiPowerLaw/loss_curve_repo/csv_25/
external/MultiPowerLaw/loss_curve_repo/csv_100/
external/MultiPowerLaw/loss_curve_repo/csv_400/
```

Each scale contains the cosine source curve and the WSD-family target curves
used in the main audit:

- `cosine_72000.csv`
- `wsd_20000_24000.csv`
- `wsdld_20000_24000.csv`
- `wsdcon_3.csv`
- `wsdcon_9.csv`
- `wsdcon_18.csv`

The repository does not contain newly trained unseen WSD schedules beyond these
public curves.  Therefore the remaining validation gap is prospective: new
held-out WSD-family training runs after freezing the protocol.

## 7. Scope

Supported claim:

- Source-only projected cosine calibration identifies a transferable LR-drop
  response component on top of frozen MPL.
- MPL-LD projection is necessary; raw cosine residual transfer fails.
- The deployable correction has one residual-fitted scalar per source scale.

Not claimed:

- A universal training-loss law.
- A universal calibration-window constant.
- Fully solved WSD-con final-LR fine ranking.
- Prospective validation on all learning-rate schedule families.

## 8. Division

This project is completed individually by Jiaju Wu.  Method design, code
implementation, experiment execution, figure/table generation, slides, and
report writing are all completed by Jiaju Wu.

The MPL baseline reproduction is based on public implementation / public data.
Relevant citations and modifications are documented in the repository.
