# 解释性重置：只保留可讲清楚的残差模型

> 2026-06-19 回滚修正：零参数 finite-response 版本太保守，当前主线恢复为 observation-bracket MPL-LD。这个版本仍然不能写成“唯一物理定律”，但它比零参数版本更符合 cosine-to-WSD 的性能目标。finite-response 版本保留为机制下界和负控。

这份记录是对前面复杂模型的收缩。当前不再推荐把 gate、channel routing、正弦展开、curvature patch 或 `sqrt-localized` 作为主公式。这些变体可以留作探索和消融。更新后的主线必须更严格：优先从 MPL 自身的 learning-rate dependent decay term 出发，而不是在 MPL 外部叠加 residual basis。

## 1. 历史诊断公式

下面的公式现在仅作为历史诊断公式保留：

\[
\hat L_s(t)=L_{\mathrm{MPL},s}(t)+\hat\kappa_s\phi_{\lambda_s,s}(t).
\]

其中

\[
\phi_{\lambda,s}(t)=\sum_{u\le t}\exp[-\lambda\eta_u]\frac{[\eta_{u-1}-\eta_u]_+}{\eta_{\max}}.
\]

解释很简单：MPL 已经给出主趋势；当 LR 下降时，真实 loss 对新 LR 的响应不是瞬时完成的，因此残差中可能出现一个因果、只由过去 LR drop 激发的 relaxation response。唯一从 loss residual 拟合的量是非负幅度 \(\hat\kappa_s\)，而且只从 `cosine_72000.csv` residual 中估计。

## 2. 训练与测试协议

训练 / 校准：

1. 对 `cosine_72000.csv` 计算 MPL prediction 与 residual \(r_{\cos}=L_{\cos}-L_{\mathrm{MPL},\cos}\)。
2. 给定目标 schedule 的 \(\lambda_s\)，在 cosine schedule 上构造同一个响应核 \(\phi_{\lambda_s,\cos}\)。
3. 只在 \(t\ge8000\) 的点上做一维 partial regression：

\[
\hat\kappa_s=
\frac{\langle M_\mu\phi_{\lambda_s,\cos},M_\mu r_{\cos}\rangle_+}
{\|M_\mu\phi_{\lambda_s,\cos}\|_2^2+\tau^2}.
\]

这里 \(M_\mu\) 只是去掉 cosine residual 中的低频 MPL drift；当前固定 \(\mu=0.01, \tau=0.05\)。

测试 / 转移：

1. 不使用目标 loss 拟合任何参数。
2. 由目标 LR schedule 构造 \(\phi_{\lambda_s,s}\)。
3. 输出 \(L_{\mathrm{MPL},s}+\hat\kappa_s\phi_{\lambda_s,s}\)。

## 3. 哪个版本可以作为主线

本节保留早期 DCT-projected core decision 的结果，用作历史对照。当前不再把这里的 DCT tau-free 或 `strict_exact` 写作最终主线；最新主线判断见 `INTERPRETABILITY_AUDIT_2026_06_19.md`。

| status | variant | extra structure | WSD mean | WSD worst | WSD wins | control note |
|---|---|---|---:|---:|---:|---|
| minimal sanity | fixed_lambda_obs | one observed half-life, no schedule geometry | -20.55% | -1.09% | 15/15 | shows the mechanism already helps |
| minimal rounded | fixed_lambda_20 | round observed half-life to 20 | -22.06% | -5.30% | 15/15 | still one response kernel |
| historical reference | strict_exact | \(\lambda_s\) from drop concentration and exact observed half-life endpoints | -31.97% | -1.09% | 15/15 | formerly considered explainable, now only a DCT-based reference |
| performance variant | rounded_fast20 | same, but round fast endpoint to 20 | -34.56% | -5.30% | 15/15 | stronger WSD, slightly less pure |
| optional safety | rounded_fast20_localized | linear locality factor only for control-safety discussion | -32.83% | -5.30% | 15/15 | controls non-harm 9/9 |
| not main | rounded_fast20_sqrtlocalized | square-root locality amplitude | -34.15% | -5.30% | 15/15 | controls non-harm 9/9, but weaker explanation |

早期结论曾认为 `strict_exact` 或 `rounded_fast20` 可以作为严谨主线。这个判断现在下调为 DCT-based ablation：它们数值强，但 nuisance projection 解释性不足。`sqrt-localized` 仍不作为主公式，因为它的 square-root amplitude 解释不够硬。

## 4. 为什么不再用复杂模型

- gate / channel routing：分类规则很容易被理解成针对当前几条曲线调出来的经验开关，泛化解释弱。
- 正弦展开：能贴合 cosine residual 的形状，但没有明确 schedule 机制，最容易过拟合。
- curvature patch：可能有帮助，但和 MPL backbone 的误差边界纠缠，难以说明新增项到底是在修正 schedule lag 还是在重拟合 MPL。
- `sqrt-localized`：虽然没有新增拟合参数，也能保护 short-cosine control，但 square-root 从 energy 到 amplitude 的论证偏软，不适合当核心贡献。

这些结果可以作为负证据：它们说明单纯追求指标会不断诱导我们加入解释不稳的项。因此当前主线应宁愿少一点性能，也要保证公式每一步都能讲清楚。

## 5. 需要诚实承认的局限

- 不加 locality 时，`rounded_fast20` 对 extra controls 的 mean/worst 为 +14.02% / +56.43%。这说明该公式不是 universal schedule predictor，而是针对 cosine-to-WSD transfer 的机制修正。
- \(\lambda_s\) 的 slow endpoint 仍含有 2.5-observation 这个 protocol choice；虽然 sensitivity 是宽的，但还不是严格定理。
- \(\mu=0.01,\tau=0.05\) 是 identifiability protocol，不是从第一性原理唯一推出。
- 当前证据仍只来自已有曲线，最终需要新 schedule 或新训练 run 做外部验证。

## 6. 当前写作建议

主文只讲 `MPL + causal LR-drop response`，但 nuisance removal 应优先使用 MPL-LD tangent 解释，而不是 DCT。如果老师追问 short-cosine control，再把 linear locality 作为 boundary condition，而不是把它写成核心理论项。
