# Source Calibration Data Drop Ablation

目标：检查从 cosine residual 拟合 \(\kappa\) 时，丢弃 source 训练点是否能改善 cosine-to-WSD 迁移，并定位哪些 source 时间段最关键。目标 WSD/controls 的 loss 只用于评价，不参与拟合。

固定公式：

\[
\widehat L_s(t)=L_{\mathrm{MPL},s}(t)+a_s\widehat\kappa_s\phi_{\lambda_s,s}(t),
\quad q_s=\sum_t(d_t/D)^2,\quad \lambda_s=\lambda_{obs}/(2-q_s).
\]

固定投影：

\[
\widehat\kappa_s=\frac{[((I-P_{LD})\phi)^\top ((I-P_{LD})r)]_+}{\|(I-P_{LD})\phi\|_2^2+1/N_{cal}}.
\]

唯一变化是 source calibration mask，即哪些 cosine 点参与上式中的内积、范数和 MPL-LD tangent projection。

## Main Reading

- `fit_start=5000`: +131.99% / +602.88% / 2/45。
- 当前 `fit_start=8000`: -26.59% / -3.15% / 45/45。
- prefix sweep 按 mean 最好的是 `suffix_ge_8000`: -26.59% / -3.15% / 45/45。
- prefix sweep 按 worst 最好的是 `suffix_ge_8000`: -26.59% / -3.15% / 45/45。
- 在当前 suffix>=8k 里，drop 后最伤性能的是 `8k-12k`，mean change +25.04 pp。
- 没有发现丢掉后能改善的 current-suffix block；最不敏感的是 `48k-60k`，但 mean change 仍为 +2.01 pp。

## Prefix Start Sweep

表中每格为 `mean delta / worst delta / wins`。delta 越负越好。

| fit_start | n_cal | same-scale | cross-scale | all WSD |
| ---: | ---: | ---: | ---: | ---: |
| 2160 | 546 | +375.49% / +1149.09% / 0/15 | +393.33% / +1666.62% / 0/30 | +387.39% / +1666.62% / 0/45 |
| 5000 | 523 | +123.34% / +399.52% / 0/15 | +136.31% / +602.88% / 2/30 | +131.99% / +602.88% / 2/45 |
| 6500 | 512 | +11.90% / +90.39% / 5/15 | +15.85% / +164.19% / 11/30 | +14.53% / +164.19% / 16/45 |
| 8000 | 500 | -29.88% / -4.67% / 15/15 | -24.95% / -3.15% / 30/30 | -26.59% / -3.15% / 45/45 |
| 10000 | 484 | -8.54% / -1.37% / 15/15 | -9.25% / -1.14% / 30/30 | -9.02% / -1.14% / 45/45 |
| 12000 | 469 | -1.49% / +0.00% / 6/15 | -1.58% / +0.00% / 12/30 | -1.55% / +0.00% / 18/45 |
| 16000 | 437 | -0.05% / +0.00% / 2/15 | -0.06% / +0.00% / 4/30 | -0.06% / +0.00% / 6/45 |
| 20000 | 406 | -0.07% / +0.00% / 5/15 | -0.09% / +0.00% / 10/30 | -0.08% / +0.00% / 15/45 |
| 24000 | 375 | -0.44% / +0.00% / 10/15 | -0.62% / +0.00% / 20/30 | -0.56% / +0.00% / 30/45 |
| 32000 | 312 | -0.54% / +0.00% / 12/15 | -0.74% / +0.00% / 24/30 | -0.68% / +0.00% / 36/45 |
| 48000 | 187 | -0.04% / -0.02% / 15/15 | -0.04% / -0.01% / 30/30 | -0.04% / -0.01% / 45/45 |
| 60000 | 94 | -0.00% / +0.00% / 3/15 | -0.00% / +0.00% / 6/30 | -0.00% / +0.00% / 9/45 |

## Leave-One-Block-Out From Current Suffix >= 8k

这里以当前 `suffix_ge_8000` 为 reference。`mean change` 为丢掉该 block 后 all-WSD mean delta 的变化：正数表示丢掉会变差，所以该 block 有用；负数表示丢掉反而变好，所以该 block 可能更 noisy。

| dropped block | n_cal | all WSD | mean change vs suffix>=8k | interpretation |
| --- | ---: | ---: | ---: | --- |
| 8k-12k | 469 | -1.55% / +0.00% / 18/45 | +25.04 pp | dropping hurts; useful block |
| 12k-20k | 437 | -19.13% / -2.12% / 45/45 | +7.46 pp | dropping hurts; useful block |
| 20k-32k | 406 | -23.41% / -2.64% / 45/45 | +3.18 pp | dropping hurts; useful block |
| 32k-48k | 375 | -16.21% / -1.94% / 45/45 | +10.38 pp | dropping hurts; useful block |
| 48k-60k | 407 | -24.59% / -2.77% / 45/45 | +2.01 pp | dropping hurts; useful block |
| 60k-72k | 406 | -19.59% / -2.45% / 45/45 | +7.00 pp | dropping hurts; useful block |

## Only-Window Calibration

只用单个 source window 拟合 \(\kappa\)。这个实验检查每段数据单独是否足以支撑迁移。

| only window | n_cal | all WSD | source retention |
| --- | ---: | ---: | ---: |
| 2.16k-5k | 23 | -0.01% / -0.00% / 45/45 | 0.006436 |
| 5k-8k | 23 | -0.00% / +0.00% / 15/45 | 8.689e-10 |
| 8k-12k | 31 | -0.00% / +0.00% / 15/45 | 9.897e-10 |
| 12k-20k | 63 | -0.00% / -0.00% / 45/45 | 1.305e-07 |
| 20k-32k | 94 | -0.00% / +0.00% / 15/45 | 1.547e-08 |
| 32k-48k | 125 | +0.00% / +0.00% / 0/45 | 2.304e-09 |
| 48k-60k | 93 | -0.00% / -0.00% / 45/45 | 8.968e-09 |
| 60k-72k | 94 | -0.00% / +0.00% / 9/45 | 2.569e-07 |

## Figures

- `figs/prefix_fit_start_sweep.png`
- `figs/prefix_fit_start_zoom.png`
- `figs/leave_one_block_out_suffix8k.png`
- `figs/only_window_calibration.png`
