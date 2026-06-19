# 演示稿（Beamer）

本目录包含中文书面报告版和英文书面报告版。两个版本均围绕
MPL residual identification 组织：先说明 cosine residual 的混淆来源，
再给出 MPL-LD projection 与 source-only response amplitude 的识别方法。

| 文件 | 语言 | 编译器 | 成品 | 页数 |
|---|---|---|---|---|
| `main_zh.tex` | 中文（主版本） | XeLaTeX | `main_zh.pdf` | 36 |
| `main.tex` | English（英文版） | pdfLaTeX 或 XeLaTeX | `main.pdf` | 38 |

## 编译

中文版使用 TeX Live 自带 Fandol 字体，Overleaf 或本地均可直接用 XeLaTeX 编译：

```bash
cd slides
xelatex main_zh.tex && xelatex main_zh.tex
```

英文版使用同一条 residual-identification 主线：

```bash
cd slides
latexmk -pdf main.tex
```

## 内容结构

中文主版本采用“问题与障碍 -> 识别方法 -> 证据链 -> 边界与结论”的结构：

1. 封面
2. 问题与障碍
3. 识别方法
4. 证据链
5. 边界与结论
6. Appendix

同时显式覆盖课程要求中的主要内容：

1. 问题背景与目标
2. 数据处理与实验设置
3. 复现实验结果
4. 方法与结果对比
5. 分析与思考
6. 代码入口与分工说明

中文主线是：

```text
frozen MPL baseline
-> WSD residual anomaly
-> cosine residual confounding
-> no-projection negative control
-> MPL-LD tangent projection
-> projection-before/after visualization
-> schedule-only LR-drop response
-> source-only one-dimensional kappa
-> leakage audit
-> WSD-family prediction and robustness checks
```

重点结果包括：预测公式
`L_hat_s(t) = L_MPL,s(t) + kappa_hat_s phi_{lambda_s,s}(t)`、
因果 LR-drop response、MPL-LD tangent projection、`q2` half-life rule、
source cosine residual 的 projection 前后可视化、
每个 source scale 只新增一个 residual-fitted scalar `kappa_hat_s`、
same-scale WSD-family `mean MAE change = -30.88%`、`worst row = -4.67%`、
`15/15 wins`，`kappa_hat` 与 oracle `kappa_star` 的 Pearson 相关为 `+0.910`，
leave-one-scale-out mean-kappa cross-scale `-25.62% / 15/15 wins`，
no-projection 负控 `+625.92% / 0/15 wins`，
projection/no-projection 时间误差对比，
以及清晰的适用边界和后续 held-out 验证计划。

新增稳健性审计入口：

```bash
python3 repro/schedule_response_robustness_audit.py
```
