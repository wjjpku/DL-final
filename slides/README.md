# 演示稿(Beamer)

两个版本,内容完全一致(21 帧,自包含、可独立阅读):

| 文件 | 语言 | 编译器 | 成品 |
|---|---|---|---|
| `main_zh.tex` | **中文(交作业用)** | **XeLaTeX** | `main_zh.pdf` |
| `main.tex` | English(备份) | pdfLaTeX | `main.pdf` |

## 编译

- **中文版(推荐提交):** 用 **XeLaTeX**,跑两遍出目录:
  ```bash
  xelatex main_zh.tex && xelatex main_zh.tex
  ```
  字体用 TeX Live 自带的 **Fandol**(`fontset=fandol`),无需系统字体;
  **Overleaf 上把编译器选 XeLaTeX 即可**直接编译。
- **英文版:** Overleaf 选 pdfLaTeX,或本地 `latexmk -pdf main.tex`。

## 内容结构(21 帧)

目标 → 背景/复现 → 从 SGD 动力学推导 MPL → 定位(对接 Li 2025 FSL / Zhang 2026 NCPL)
→ 尺度不变性 + 27 曲线塌缩(R²=0.997)→ 方法开发(SC-MPL / Q-MPL)→
γ 的稳定边缘来源(river-valley 锐化滞后)+ 两个显式律的诚实负结果(近乎不可约)→
结论 → 可复现性 → 分工与致谢。

> 封面与末页已含组员信息(姓名、学号、本科生)、GitHub 链接与分工说明,满足 Task 2 要求。
