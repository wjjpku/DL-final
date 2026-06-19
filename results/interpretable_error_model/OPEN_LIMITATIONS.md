# 当前仍未完全解决的限制

这份记录用于防止把 observation-bracket MPL-LD 讲过头。2026-06-19 的解释性审计后，当前结论进一步收缩：observation-bracket MPL-LD 只能作为数值强的诊断/消融参考，不能写成最终主模型。

## 1. 外部泛化仍缺失

当前 loss-curve repository 只有三组 scale 上的 cosine、constant 和 WSD-family 曲线。已有验证包括 same-scale、cross-scale、controls 和 no-nuisance failure，但还没有新的 LR schedule 或新的 training run。

因此当前可以说：

- cosine calibration 到 WSD-family evaluation 有稳定正优化；
- 跨 scale calibration 仍保持 30/30 WSD 正优化；
- short-cosine 和 constant controls 在 locality boundary 下 non-harm。

但还不能说：

- 对任意 LR schedule 都成立；
- 对新模型架构、数据集或优化器都成立；
- 已经达到真正外部验证标准。

## 2. Observation Bracket 是强 prior，不是定理

当前 response rate 为

\[
\lambda_s=\lambda_{\mathrm{obs}}\frac{1+q_s}{2}.
\]

它的解释是：diffuse LR decay 使用 two-observation half-life，sharp LR drop 使用 one-observation half-life。这个比旧的 `2.5` 和 `20` 干净，因为只用 logging interval 和 schedule drop concentration。

但它仍然是 observation-resolution prior，不是从 loss dynamics 第一定理推出。写作中应强调它是一个可解释、可审计、无需 target loss 的建模假设。

## 2.5. 更干净的 MPL-LD finite-response 还没有过关

更可解释的方向是直接修改 MPL 最后一项：

\[
\hat L_\tau(t)=L_{\mathrm{MPL}}(t)+B[D_\tau(t)-D(t)].
\]

这个版本不新增 residual basis，也不需要从 cosine residual 拟合幅度。初步审计显示：

| variant | WSD | controls |
|---|---:|---:|
| \(\tau=64\) fixed | -3.11%, 15/15 | worst +1.62% |
| \(\tau=128\) fixed | -9.52%, 15/15 | worst +6.01% |
| \(\tau=256\) fixed | -15.09%, 14/15 | worst +15.90% |
| \(\tau=128\), cosine-fitted amplitude | +565.16%, 0/15 | worst +656.40% |

进一步的 signed-decomposition 审计显示，若只 lag MPL 中 LR 下降贡献 \(D_\downarrow(t)\)，再乘以 schedule-only adiabatic boundary

\[
a_s=\left[1-\frac{\ell_\downarrow}{T-W}\right]_+,
\]

则 \(\tau=128\) 时得到 WSD `-8.73%`、15/15 wins，并且 controls 9/9 non-harm。进一步使用 support-bracket

\[
\tau_s=\Delta_{\mathrm{obs}}\left(1+\min(1,\ell_\downarrow/\Delta_{\mathrm{obs}})\right)
\]

后，WSD 改善到 `-13.77%`、worst `-6.29%`、15/15 wins，controls 仍然 9/9 non-harm。这个结果更干净，因为没有 residual-fitted coefficient，\(\tau_s\) 只由 logging interval 和 cooldown support span 决定。

但它仍未完全过关：

- 收益仍比 observation-bracket MPL-LD 小；
- \(a_s\) 仍是 schedule-level boundary prior，不是从 MPL 内部唯一推出；
- \(\tau_s\) 的 support-bracket 需要写成 observation-resolution 假设，并做更多稳定性说明；
- cosine-fitted amplitude 的失败说明 cosine residual contamination 仍然是核心问题。

## 3. Locality 是边界条件，不是核心机制

当前 locality factor 为

\[
a_s=\mathbf 1\{\sum_t d_t>0\}
\left[1-\frac{\ell_s}{T_s-W}\right]_+.
\]

它解决的是 control boundary：full-horizon cosine decay 不应被当作 local cooldown transient。审计显示：

| variant | WSD | controls |
|---|---:|---:|
| no locality | -30.89%, 15/15 | +13.39% mean, +56.99% worst |
| linear locality | -29.87%, 15/15 | 9/9 non-harm |

因此 locality 不应被写成核心理论贡献。核心机制是 MPL-LD projected LR-drop response；locality 只是 schedule-support boundary condition。

## 4. MPL Baseline 误差仍是前提

模型使用已有 MPL prediction 作为 backbone。MPL 参数不是 residual correction 中新增的拟合参数，但 residual correction 的意义依赖 MPL baseline 足够合理。

当前 MPL-LD tangent projection 解释了为什么要去掉 LR-dependent MPL parameter-error directions，但没有重新证明 MPL 本身是最优 backbone。更保守的下一步是先验证 MPL 自身 \(D(t)\) 的 finite-response 版本，而不是继续添加 MPL 外部 residual basis。

## 5. 当前最强结论

可以稳妥表述为：

> 在现有 loss-curve repository 中，MPL 对 LR drop 后的 response 存在系统残差。直接把 MPL 的 LR-dependent decay term 中的 cooldown 子项改成有限响应时间版本，并用 support-bracket \(\tau_s\) 与 schedule-only adiabatic boundary，可以在不拟合 residual 参数的情况下稳定改善 WSD-family 并保持 controls non-harm。带 MPL-LD tangent nuisance 的 observation-bracket correction 数值更强，但目前只能作为诊断/参考，不应宣称为最终可解释模型。

不应表述为：

> 已经发现了通用训练 loss 物理定律。

## 6. 下一步优先级

1. 找到或生成新的 LR schedule 作为外部验证。
2. 若无法新增训练 run，就在 slides/paper 中明确声明当前验证边界。
3. 在写作中把 DCT、sqrt-locality、gate、channel、sine 都放到 rejected/ablation，而不是主方法。
4. 最终同步 slides 前，必须先决定 direct MPL-LD finite-response 是否能成为主线；否则 slides 只能如实写 exploratory diagnostics。
