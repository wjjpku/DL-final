# Paper Build Notes

主文件：`main.tex`

编译器：`pdfLaTeX`

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
```

本目录包含主论文源文件、已编译 PDF 和论文图。当前 `main.tex` 围绕
MPL residual identification 组织：MPL 作为冻结 baseline，cosine residual
先经过 MPL-LD projection 去除 nuisance drift，再估计一个非负 response
amplitude；WSD-family loss 只用于评价和 oracle diagnostic。参考文献以内联
`thebibliography` 形式写在 `main.tex` 中，不需要 BibTeX。

## Files

| 文件或目录 | 内容 |
|---|---|
| `main.tex` | 主论文正文，当前主线为 cosine-to-WSD residual identification。 |
| `main.pdf` | 已编译论文 PDF。 |
| `figs/` | 论文中使用的图，包含新公式 schedule feature、per-target、ablation 和 residual-curve 图。 |
| `theory.tex` / `theory.pdf` | 当前公式的补充理论说明：linear response、`q2` half-life、support projection、MPL-LD projection。 |

## Related Artifacts

- Slides: `../slides/main_zh.pdf` and `../slides/main.pdf`.
- Final checklist: `../FINAL_DELIVERABLES.md`.
- Core documentation index: `../docs/README.md`.
- Main robustness audit: `../results/schedule_response_robustness/REPORT.md`.
- Target-loss leakage audit: `../results/schedule_response_robustness/LEAKAGE_AUDIT.md`.
- Reproducibility guide: `../REPRODUCIBILITY.md`.
