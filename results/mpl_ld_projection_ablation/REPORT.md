# MPL-LD Tangent Projection Ablation

本消融只检查一个问题：在当前 q2 half-life 公式里，是否必须先取 MPL-LD 梯度矩阵并做正交投影。

除这一项外，所有设定保持一致：cosine 作为 source，`fit_start=8000`，`q_rule=q2`，`lambda_rule=halflife`，`locality=support_projection`，ridge 为 `1/N_cal`，目标曲线的 loss 只用于最终评价。

## 被消融的操作

当前带投影版本使用

\[
x=(I-P_{LD})\phi_{\lambda_s,\mathrm{cos}},\qquad y=(I-P_{LD})r_{\mathrm{cos}},
\]

其中 \(P_{LD}=Q_{LD}Q_{LD}^{\top}\)，\(Q_{LD}\) 来自 cosine 后缀区间上 MPL 参数 \((\log B,\log C,\log\beta,\log\gamma)\) 的 finite-difference tangent matrix 的 QR 正交化。然后

\[
\widehat\kappa_s=\frac{[x^{\top}y]_+}{\|x\|_2^2+1/N_{cal}}.
\]

无投影消融把上式替换为

\[
x=\phi_{\lambda_s,\mathrm{cos}},\qquad y=r_{\mathrm{cos}},
\]

其余 \(\widehat\kappa_s\) 的估计式完全不变。

最终预测仍然是

\[
\widehat L_s(t)=L_{\mathrm{MPL},s}(t)+a_s\widehat\kappa_s\phi_{\lambda_s,s}(t),
\]

其中 \(q_s=\sum_t(d_t/D)^2\)，\(\lambda_s=\lambda_{obs}/(2-q_s)\)，\(a_s=1-\mathrm{support\_span}/\mathrm{post\_warmup\_horizon}\)。

## 结果

| variant | split | group | mean / worst / wins |
| --- | --- | --- | --- |
| current_with_mplld_projection | same_scale | core_wsd | -29.88% / -4.67% / 15/15 |
| current_with_mplld_projection | cross_scale | core_wsd | -24.95% / -3.15% / 30/30 |
| current_with_mplld_projection | all | core_wsd | -26.59% / -3.15% / 45/45 |
| current_with_mplld_projection | same_scale | extra_control | +0.00% / +0.00% / 0/9 |
| current_with_mplld_projection | cross_scale | extra_control | +0.00% / +0.00% / 0/18 |
| current_with_mplld_projection | all | extra_control | +0.00% / +0.00% / 0/27 |
| current_without_mplld_projection | same_scale | core_wsd | +602.16% / +2366.35% / 0/15 |
| current_without_mplld_projection | cross_scale | core_wsd | +572.51% / +2657.86% / 0/30 |
| current_without_mplld_projection | all | core_wsd | +582.40% / +2657.86% / 0/45 |
| current_without_mplld_projection | same_scale | extra_control | +0.00% / +0.00% / 0/9 |
| current_without_mplld_projection | cross_scale | extra_control | +0.00% / +0.00% / 0/18 |
| current_without_mplld_projection | all | extra_control | +0.00% / +0.00% / 0/27 |

## 系数诊断

| variant | mean coef | median coef | max coef | mean source retention | median source retention |
| --- | ---: | ---: | ---: | ---: | ---: |
| current_with_mplld_projection | 0.0537575 | 0.0473923 | 0.0819648 | 0.00133324 | 0.00135934 |
| current_without_mplld_projection | 0.543056 | 0.473406 | 0.919067 | 1 | 1 |

## 结论

这个投影是必要的。不开投影时，cosine residual 中可被 MPL-LD 参数微调解释的平滑结构会直接进入 \(\widehat\kappa_s\)，导致校正项把 MPL 本身的低维拟合误差也当成 schedule-response 误差。因此跨到 WSD 时不是小幅退化，而是系统性失败。

带投影版本虽然只保留 source response feature 中很小的一部分正交能量，但它保留的是 MPL-LD 低维重拟合不能解释的残差方向。这正是我们希望 \(\kappa\) 学到的部分：与 learning-rate drop 的滞后响应有关，而不是与 MPL 参数偏移有关。
