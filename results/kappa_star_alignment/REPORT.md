# Kappa-Star Alignment Audit

目的：把问题从最终 MAE 拆开，直接检查 \(\widehat\kappa\) 是否接近每条 WSD 曲线自己的 oracle \(\kappa^\star\)。这里暂时不把 \(a_s\) 当作 kappa 的一部分。

## Oracle Definition

固定当前 response shape：

\[
\phi_s(t)=\mathrm{causal\_drop\_response}(\lambda_s),\qquad \lambda_s=\lambda_{obs}/(2-q_2).
\]

对每条目标 WSD 曲线，用目标 residual 只做诊断性 oracle fit：

\[
\kappa_s^\star=\frac{[\langle \phi_s, L_s-L_{MPL,s}\rangle]_+}{\|\phi_s\|_2^2}.
\]

这个 \(\kappa_s^\star\) 不用于部署，只用于回答：我们从 cosine 算出的 \(\widehat\kappa\) 是否方向正确。

## Deployable Kappa Estimators

推荐的 kappa 本体是 source-only 的 projected cosine estimator：

\[
\widehat\kappa_{\cos}(s)=
\frac{[((I-P_{LD})\phi_{\lambda_s,\cos})^\top((I-P_{LD})r_{\cos})]_+}
{\|(I-P_{LD})\phi_{\lambda_s,\cos}\|_2^2+1/N_{cal}}.
\]

注意这里没有乘 \(a_s\)。如果后续需要 safety abstention，可以另外讨论；但它不应该混进 kappa 的定义。

## Main Result

- `projected_cosine_kappa`: Pearson +0.910, Spearman +0.715, RMSE 0.0103, MAE delta -30.88% / worst -4.67%。
- `current_effective_a_times_kappa`: Pearson +0.898, Spearman +0.715, RMSE 0.0135, MAE delta -29.88% / worst -4.67%。
- `sqrt_drop_projected_kappa`: Pearson +0.921, Spearman +0.804, RMSE 0.0115, MAE delta -30.06% / worst -3.33%。
- `raw_cosine_no_projection`: Pearson -0.849, Spearman -0.852, RMSE 0.6294, MAE delta +625.92% / worst +2366.71%。

直接结论：`projected_cosine_kappa` 已经和 oracle \(\kappa^\star\) 强相关，Pearson 为 0.91；把 \(a_s\) 乘进去反而略降相关性并增加 kappa RMSE。不做 MPL-LD projection 的 raw cosine kappa 则完全失败，说明 projection 仍然是 kappa 估计的关键。

## Summary Table

| estimator | split | Pearson | Spearman | kappa RMSE | mean delta | worst delta | wins |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| projected_cosine_kappa | all_core | +0.910 | +0.715 | 0.0103 | -30.88% | -4.67% | 15/15 |
| projected_cosine_kappa | wsd_final | +0.840 | +0.543 | 0.0108 | -45.50% | -33.43% | 6/6 |
| projected_cosine_kappa | wsdcon | +0.263 | -0.053 | 0.0099 | -21.14% | -4.67% | 9/9 |
| current_effective_a_times_kappa | all_core | +0.898 | +0.715 | 0.0135 | -29.88% | -4.67% | 15/15 |
| current_effective_a_times_kappa | wsd_final | +0.840 | +0.543 | 0.0176 | -43.00% | -29.81% | 6/6 |
| current_effective_a_times_kappa | wsdcon | +0.263 | -0.053 | 0.0099 | -21.14% | -4.67% | 9/9 |
| sqrt_drop_projected_kappa | all_core | +0.921 | +0.804 | 0.0115 | -30.06% | -3.33% | 15/15 |
| sqrt_drop_projected_kappa | wsd_final | +0.840 | +0.371 | 0.0117 | -45.76% | -32.92% | 6/6 |
| sqrt_drop_projected_kappa | wsdcon | +0.459 | +0.350 | 0.0114 | -19.59% | -3.33% | 9/9 |
| drop_projected_kappa | all_core | +0.920 | +0.793 | 0.0139 | -29.03% | -2.48% | 15/15 |
| drop_projected_kappa | wsd_final | +0.840 | +0.371 | 0.0134 | -45.49% | -32.05% | 6/6 |
| drop_projected_kappa | wsdcon | +0.506 | +0.300 | 0.0142 | -18.06% | -2.48% | 9/9 |
| raw_cosine_no_projection | all_core | -0.849 | -0.852 | 0.6294 | +625.92% | +2366.71% | 0/15 |
| raw_cosine_no_projection | wsd_final | -0.661 | -0.543 | 0.3415 | +215.30% | +336.03% | 0/6 |
| raw_cosine_no_projection | wsdcon | -0.646 | -0.580 | 0.7633 | +899.67% | +2366.71% | 0/9 |

## Schedule-Information Check

我也扫了一个纯 schedule multiplier：

\[
\widehat\kappa(p)=\widehat\kappa_{\cos}(D_s/\eta_{max})^p.
\]

在当前 15 条 WSD 目标上，按相关性最好的 in-sample `p=0.65`，Pearson `+0.921`；按 RMSE 最好的 `p=0.05`，RMSE `0.0102`。这说明 total drop 信息可能有用，但这个指数目前是 development diagnostic，不能直接当最终定理。

一个保守、可解释的候选是 `sqrt_drop_projected_kappa`，它用 \(p=1/2\)：Pearson 略高于 raw projected kappa，Spearman 提升明显，但最终 MAE 均值没有 raw projected kappa 好。所以我现在不建议替换主 kappa，只建议把它作为下一轮候选。

## Figures

- `figs/kappa_hat_vs_star_scatter.png`
- `figs/kappa_by_target.png`
- `figs/drop_power_scan.png`

## Current Recommendation

把 kappa 的定义收缩为：`projected_cosine_kappa`。也就是说，\(\kappa\) 只来自 cosine residual + MPL-LD nuisance projection + sample-size ridge。\(a_s\) 不作为 kappa 的一部分；如果还需要保护 controls，应单独作为 safety/abstention 条件讨论，不要把它写成 kappa 学习机制。
