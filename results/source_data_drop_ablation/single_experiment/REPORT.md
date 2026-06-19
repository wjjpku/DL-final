# Single-Experiment Prefix Drop Error Estimate

固定 target：`100M WSD-con 3e-5`。只改变 cosine source 中参与 \(\kappa\) 拟合的 prefix drop 边界。目标 loss 只用于画真实 residual 和计算 MAE。

## Formula

\[
\widehat e_s(t)=a_s\widehat\kappa_s\phi_{\lambda_s,s}(t),\qquad \widehat L_s(t)=L_{MPL,s}(t)+\widehat e_s(t).
\]

每个 `fit_start` 都重新在 source mask 上构造 MPL-LD tangent projection，并重新估计 \(\widehat\kappa_s\)。

## Metrics

| fit_start | n_cal | kappa | source retention | correction / true residual L1 | MAE delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2160 | 546 | 0.593713 | 0.06614 | 12.75x | +1149.09% |
| 5000 | 523 | 0.244566 | 0.0114 | 5.25x | +399.52% |
| 6500 | 512 | 0.100577 | 0.003472 | 2.16x | +90.39% |
| 8000 | 500 | 0.0327189 | 0.001181 | 0.70x | -42.76% |
| 10000 | 484 | 0.00521752 | 0.0004699 | 0.11x | -9.08% |
| 12000 | 469 | 0 | 0.0003533 | 0.00x | +0.00% |

## Figures

- `figs/estimated_residual_by_fit_start_full.png`
- `figs/estimated_residual_by_fit_start_zoom.png`
- `figs/remaining_residual_by_fit_start_full.png`
- `figs/remaining_residual_by_fit_start_zoom.png`

## Reading

`fit_start=2160/5000/6500` 会把误差估计放大成过大的 positive spike；`fit_start=8000` 的估计幅度与真实 MPL residual 最接近；`fit_start=10000/12000` 则开始欠估计，说明 8k-10k 附近是把幅度定准的关键 source 区间。
