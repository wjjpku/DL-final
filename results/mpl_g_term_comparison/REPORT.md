# MPL Final G-Term Comparison

这里的 MPL 最后一项指代码中的

\[
L_G(t)=B\,D_{LD}(t),\qquad D_{LD}(t)=\sum_{k\le t}\Delta\eta_k\,G\!\left(\eta_k^{-\gamma}(S(t)-S(k))\right),
\]

其中 \(G(x)=1-(1+Cx)^{-\beta}\)。这项已经包含在 MPL baseline 里，不是我们新增的 residual correction。

为了和正 residual 比较，图中还画了

\[
-\Delta L_G(t)=-\bigl(L_G(t)-L_G(t_{drop})\bigr),
\]

它表示 MPL 最后一项在 LR drop 之后预测的 quasi-static loss decrease。我们的误差项则是

\[
\widehat e(t)=a_s\widehat\kappa_s\phi_{\lambda_s}(t).
\]

## Key Single Example

固定 `100M WSD-con 3e-5`：

| quantity | value |
| --- | ---: |
| our MAE delta | -42.76% |
| raw `-Delta L_G` / true residual L1 | 11.04x |
| our correction / true residual L1 | 0.70x |
| corr(`-Delta L_G`, residual) after drop | -0.326 |
| corr(our correction, residual) after drop | +0.926 |

## All Core Targets

| scale | target | `-Delta L_G` / residual L1 | our / residual L1 | corr G-res | corr ours-res | our delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 25M | WSD sharp | 3.39x | 0.38x | +0.95 | +0.96 | -37.12% |
| 25M | WSD linear | 2.76x | 0.30x | +0.98 | +0.96 | -29.81% |
| 25M | WSD-con 3e-5 | 10.16x | 0.57x | -0.38 | +0.94 | -44.64% |
| 25M | WSD-con 9e-5 | 12.24x | 0.25x | -0.53 | +0.85 | -21.19% |
| 25M | WSD-con 18e-5 | 8.17x | 0.09x | -0.73 | +0.87 | -9.19% |
| 100M | WSD sharp | 3.71x | 0.60x | +0.96 | +0.95 | -55.90% |
| 100M | WSD linear | 3.19x | 0.50x | +1.00 | +0.98 | -49.01% |
| 100M | WSD-con 3e-5 | 11.04x | 0.70x | -0.33 | +0.93 | -42.76% |
| 100M | WSD-con 9e-5 | 9.90x | 0.23x | -0.46 | +0.85 | -9.84% |
| 100M | WSD-con 18e-5 | 20.10x | 0.26x | -0.50 | +0.77 | -12.80% |
| 400M | WSD sharp | 3.56x | 0.47x | +0.98 | +0.95 | -45.79% |
| 400M | WSD linear | 3.26x | 0.41x | +1.00 | +0.98 | -40.38% |
| 400M | WSD-con 3e-5 | 9.08x | 0.47x | -0.29 | +0.89 | -35.97% |
| 400M | WSD-con 9e-5 | 8.69x | 0.16x | -0.39 | +0.79 | -9.17% |
| 400M | WSD-con 18e-5 | 12.95x | 0.13x | -0.35 | +0.69 | -4.67% |

## Figures

- `figs/single_100M_wsdcon_3.png`
- `figs/core_shape_grid_25M.png`
- `figs/core_shape_grid_100M.png`
- `figs/core_shape_grid_400M.png`

## Reading

MPL 最后一项主要解释的是 LR schedule 变化导致的 quasi-static equilibrium movement：LR 降低后，它给出一个较大的、平滑的 loss decrease，并且这个 decrease 是 MPL baseline 的一部分。我们现在拟合的 residual correction 不是重复这件事，而是在 MPL 已经下降以后补一个正的 lag residual，表示真实 loss 没有立即跟上 MPL 的新 equilibrium。

因此二者形状方向不同：`-Delta L_G` 更像累计的平滑 equilibrium shift；我们的误差项更像 cooldown 后的正向 transient / relaxation lag。