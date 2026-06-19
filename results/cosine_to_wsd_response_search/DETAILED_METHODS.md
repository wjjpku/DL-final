# 详细方法说明：无局部路由项的 Cosine-to-WSD Loss Curve 修正

本文档说明当前保守版本的方法。相较上一版，已经删除 WSD-con 的 target-specific 局部修正项。主模型只保留：

```text
MPL baseline
+ causal LR-drop response
+ DCT residualization
+ smooth/step channel split
+ LR curvature for step-like schedules
```

最终预测形式为：

```text
L_hat(t) = L_MPL(t) + C_schedule(t).
```

其中 `C_schedule(t)` 的形状完全由目标 learning-rate schedule 计算；每个 scale 的 correction coefficients 只从 cosine residual 拟合。目标 WSD-family loss 只用于最后评估 MAE。

---

## 1. 问题背景

MPL 是当前强 baseline。对任意目标 schedule，它给出：

```text
L_hat_MPL(t) = L_MPL(t).
```

但在 cosine-to-WSD transfer 中，MPL residual 仍有结构：

```text
r(t) = L_true(t) - L_MPL(t).
```

关键问题是 cosine residual 不是纯 learning-rate response：

```text
r_cos(t)
  = transferable LR-response error
  + non-transferable MPL slow drift
  + early transient
  + noise.
```

如果直接在整条 cosine residual 上拟合一个全局系数，很容易把 MPL 的低频慢漂移也学进 correction。这样在 cosine 上可能拟合得更好，但迁移到 WSD-family schedules 时并不可靠。

当前方法的目标是：

```text
只提取 cosine residual 中较可迁移的 LR-response 部分，
避免使用 target-specific local correction。
```

---

## 2. 数据与协议

### 2.1 Model scales

实验包含三个 scale：

```text
25M
100M
400M
```

### 2.2 Source curve

用于拟合 correction coefficients 的曲线是：

```text
cosine_72000.csv
```

对每个 scale：

```text
r_cos(t) = L_cos(t) - L_MPL,cos(t).
```

### 2.3 Target curves

评估使用五条 WSD-family target：

```text
wsd_20000_24000.csv    -> WSD sharp
wsdld_20000_24000.csv  -> WSD linear
wsdcon_3.csv           -> WSD-con 3e-5
wsdcon_9.csv           -> WSD-con 9e-5
wsdcon_18.csv          -> WSD-con 18e-5
```

总评估行数：

```text
3 scales * 5 targets = 15 scale-target rows.
```

### 2.4 Allowed information during prediction

预测目标 schedule 时允许使用：

```text
1. L_MPL,target(t)
2. target LR schedule eta_t
3. schedule-derived features computed from eta_t
4. coefficients fitted from cosine residual
```

不使用：

```text
1. target loss values
2. target residuals
3. per-target loss fitting
```

---

## 3. 主公式

定义 positive LR drop：

```text
d_t = relu(eta_{t-1} - eta_t).
```

定义因果 LR-response：

```text
a_t = exp(-lambda * eta_t) * a_{t-1} + d_t
phi_lambda(t) = a_t / eta_peak.
```

展开为：

```text
phi_lambda(t)
  = (1 / eta_peak) * sum_{u <= t} d_u
      * exp(-lambda * sum_{v=u+1}^{t} eta_v).
```

最终 correction 为：

```text
smooth target:
  C_schedule(t) = k_smooth * phi_4(t)

step-like target:
  C_schedule(t) = a_step * phi_20(t)
                + b_curv * psi_10(t)
```

WSD sharp、WSD linear 和 WSD-con 都按 schedule geometry 进入 smooth 或 step channel。当前版本不再为 WSD-con 额外引入 target-specific 局部项。

---

## 4. 因果 LR-response 的动机

当 learning rate 下降时，训练状态不会在同一步瞬间达到新的有效平衡。可以把每次 LR drop 看成一次误差注入：

```text
d_t = relu(eta_{t-1} - eta_t).
```

之后训练会逐步消化这份误差。消化速度和后续 LR 有关：

```text
exp(-lambda * eta_t)
```

因此历史上每一次 LR drop 对当前 residual 的贡献是：

```text
d_u * exp(-lambda * sum_{v=u+1}^{t} eta_v).
```

这给出一个 schedule-only feature。它不需要目标 loss，因此可以用于训练前预测。

---

## 5. Smooth / Step Channel Split

不同 target schedule 的 LR drop 集中程度不同。定义：

```text
drop_concentration = max_t d_t / sum_t d_t.
```

路由规则：

```text
smooth channel:
  drop_concentration < 0.2

step channel:
  drop_concentration >= 0.2
```

当前 response rates：

```text
smooth:
  lambda = 4

step:
  lambda = 20
```

解释：

```text
smooth channel
  用于平滑、分散的 LR decay。

step channel
  用于集中、突变的 LR drop。
```

这个拆分的目的是减少 cosine 平滑 residual 对 step-like target 的污染。

---

## 6. DCT Residualization

cosine residual 中包含 MPL slow drift。为了避免 correction coefficient 吸收这部分低频漂移，在拟合前对 residual 和 feature 都做 DCT residualization。

构造 DCT basis：

```text
Q = [q_0, q_1, ..., q_m]
q_0 = constant mode
q_k(i) = cos(pi * (i + 0.5) * k / n)
```

soft residualizer：

```text
M_mu y
  = y - Q (Q^T Q + diag(mu * k^4))^{-1} Q^T y.
```

拟合时使用：

```text
x_o = M_mu x
y_o = M_mu y
```

当前设置：

```text
smooth:
  mu = 0.05
  max_mode = 8

step:
  mu = 0.01
  max_mode = 8
```

DCT residualization 的作用是解决 identifiability：让 coefficient 尽量解释 schedule-response，而不是解释 MPL 的低频慢漂移。

---

## 7. Fit Windows

训练早期 residual 更容易受到 warmup、optimization transient 和 loss scale 快速变化影响，因此只在 suffix window 上拟合：

```text
smooth:
  F_smooth = {t : t >= 12000}

step:
  F_step = {t : t >= 3000}
```

smooth channel 使用更晚窗口，因为平滑 decay 更容易和 low-frequency drift 混在一起。step channel 的 LR drop 更局部，因此可以使用更早窗口。

---

## 8. Smooth Coefficient

smooth correction：

```text
C_smooth(t) = k_smooth * phi_4(t).
```

在 `F_smooth` 上：

```text
x = phi_4
y = r_cos
x_o = M_mu x
y_o = M_mu y
```

非负 ridge projection：

```text
raw = max(0, <x_o, y_o> / (||x_o||^2 + tau^2)).
```

source-retention shrinkage：

```text
R_source = ||x_o||^2 / ||x||^2

k_smooth
  = (1 / (1 + rho))
    * R_source^p
    * raw.
```

当前参数：

```text
fit_start = 12000
lambda = 4
mu = 0.05
max_mode = 8
tau = 0.05
p = 0.25
rho = 0.2
coefficient constraint = nonnegative
```

---

## 9. Step Correction With Curvature

step primary feature：

```text
phi_step(t) = phi_20(t).
```

curvature feature 使用二阶 LR 差分：

```text
delta2_eta_t = eta_{t-2} - 2 eta_{t-1} + eta_t.
```

再做 causal relaxation：

```text
psi_10(t)
  = causal_relax_lambda=10(delta2_eta_t) / eta_peak.
```

step correction：

```text
C_step(t)
  = a_step * phi_20(t)
    + b_curv * psi_10(t).
```

在 `F_step` 上 joint constrained ridge：

```text
X = [phi_20, psi_10]
y = r_cos

X_o = M_mu X
y_o = M_mu y

min_{a_step >= 0, b_curv >= 0}
  ||X_o [a_step, b_curv]^T - y_o||^2
  + tau_step^2 * a_step^2
  + tau_curv^2 * b_curv^2.
```

当前参数：

```text
fit_start = 3000
lambda_step = 20
lambda_curv = 10
mu = 0.01
max_mode = 8
tau_step = 0.05
tau_curv = 0.003
rho = 0.35
coefficient constraint = nonnegative
```

这里必须 joint fitting。若先拟合 primary response，再把剩余 residual 交给 curvature，会把第一阶段的解释误差错误地塞进 curvature。

---

## 10. 完整预测流程

### 10.1 Source fitting

对每个 scale：

```text
1. Load cosine_72000.csv.
2. Compute L_MPL,cos(t).
3. Compute r_cos(t) = L_cos(t) - L_MPL,cos(t).
4. Compute phi_4, phi_20, psi_10 from the cosine LR schedule.
5. Fit k_smooth on t >= 12000.
6. Jointly fit a_step and b_curv on t >= 3000.
```

### 10.2 Target prediction

对每条 target schedule：

```text
1. Compute L_MPL,target(t).
2. Compute d_t and drop_concentration from target LR schedule.
3. If smooth:
     C(t) = k_smooth * phi_4,target(t).
4. If step-like:
     C(t) = a_step * phi_20,target(t)
            + b_curv * psi_10,target(t).
5. Predict:
     L_hat_target(t) = L_MPL,target(t) + C(t).
6. Evaluate MAE using target loss only after prediction.
```

---

## 11. Target Retention Guard

对 target feature 计算：

```text
target_retention = ||M_mu feature||^2 / ||feature||^2.
```

如果：

```text
target_retention < 0.01
```

则禁用该 correction。

作用：如果目标 feature 几乎完全落在 low-frequency nuisance subspace 中，就不强行施加 correction。

---

## 12. 参数数量

当前 core model 每个 scale 新增 fitted correction coefficients：

```text
k_smooth
a_step
b_curv
```

总数：

```text
3 coefficients per scale.
```

这些是从 cosine residual 拟合的系数。

下面是 method-level hyperparameters，不是每个 scale 重新拟合的自由参数：

```text
lambda_smooth
lambda_step
lambda_curv
fit_start
mu
max_mode
tau
rho
p
retention floor
```

严谨表述：

```text
per-scale correction coefficients are learned from cosine residuals;
method-level hyperparameters are selected using WSD-family development evaluation.
```

---

## 13. 当前主结果

当前推荐主模型：

```text
MPL
+ cosine-fitted LR response
+ DCT residualization
+ smooth/step channel split
+ LR curvature for step-like schedules
```

总体结果：

```text
mean MAE change = -37.53%
worst scale-target row = -10.80%
wins = 15/15
```

分 target：

| target | mean MAE change | worst scale | wins |
|---|---:|---:|---:|
| WSD sharp | -54.3% | -40.5% | 3/3 |
| WSD linear | -46.3% | -33.1% | 3/3 |
| WSD-con 3e-5 | -57.1% | -46.7% | 3/3 |
| WSD-con 9e-5 | -17.0% | -10.8% | 3/3 |
| WSD-con 18e-5 | -13.0% | -12.1% | 3/3 |

WSD-con 详细结果：

| target | scale | MAE change | corrected MAE | MPL MAE |
|---|---|---:|---:|---:|
| WSD-con 3e-5 | 25M | -58.9% | 0.001845 | 0.004487 |
| WSD-con 3e-5 | 100M | -46.7% | 0.003023 | 0.005669 |
| WSD-con 3e-5 | 400M | -65.7% | 0.002807 | 0.008174 |
| WSD-con 9e-5 | 25M | -21.4% | 0.002299 | 0.002926 |
| WSD-con 9e-5 | 100M | -10.8% | 0.004402 | 0.004935 |
| WSD-con 9e-5 | 400M | -18.8% | 0.005432 | 0.006687 |
| WSD-con 18e-5 | 25M | -12.1% | 0.002202 | 0.002505 |
| WSD-con 18e-5 | 100M | -13.2% | 0.001199 | 0.001382 |
| WSD-con 18e-5 | 400M | -13.6% | 0.002209 | 0.002558 |

---

## 14. Ablation Path

| variant | mean MAE change | worst row | wins | 说明 |
|---|---:|---:|---:|---|
| adaptive fit-window | -34.53% | -6.08% | 15/15 | suffix fitting 减少 early contamination |
| channel shrink | -35.07% | -6.12% | 15/15 | shrinkage 提高稳定性 |
| decoupled channel | -36.18% | -6.29% | 15/15 | smooth 和 step 不共享全部 calibration |
| fixed-channel LR curvature | -37.47% | -9.43% | 15/15 | curvature 改善 abrupt transitions |
| joint channel + curvature | -37.53% | -10.80% | 15/15 | joint fitting 改善 mean 和 worst |

报告在 joint curvature 处停止。继续增加 target-specific local correction 虽可能带来小幅 development-set 收益，但选择痕迹太强，不适合作为主方法。

---

## 15. 局限性

必须诚实承认：

```text
This is still a development candidate.
```

原因：

```text
1. correction coefficients are cosine-only;
2. features are schedule-only;
3. but method-level hyperparameters were selected with WSD-family development evaluation.
```

因此可以说：

```text
当前方法在 WSD-family development suite 上稳定优于 MPL。
```

不能说：

```text
已经证明对所有 schedule families 都泛化。
```

最关键的下一步：

```text
Freeze the current core protocol.
Evaluate on new held-out schedules or a pre-registered held-out family.
```

---

## 16. 文件位置

主结果：

```text
results/cosine_to_wsd_response_search/joint_curvature/REPORT.md
results/cosine_to_wsd_response_search/joint_curvature/best_target_summary.csv
results/cosine_to_wsd_response_search/joint_curvature/top_safe_details.csv
```

主脚本：

```text
repro/cosine_to_wsd_response_search.py
repro/cosine_to_wsd_curvature_correction.py
repro/cosine_to_wsd_joint_curvature_search.py
repro/plot_new_formula_slides.py
```

---

## 17. 一句话总结

当前方法不再使用 target-specific local correction。核心公式是：

```text
L_hat(t) = L_MPL(t) + C_schedule(t)
```

其中：

```text
smooth:
  C_schedule(t) = k_smooth * phi_4(t)

step-like:
  C_schedule(t) = a_step * phi_20(t)
                + b_curv * psi_10(t)
```

它用 DCT residualization 尽量压掉 cosine residual 中的 MPL slow drift，再从剩余信号中估计可迁移 LR-response error。

当前结果：

```text
mean MAE change = -37.53%
worst scale-target row = -10.80%
wins = 15/15
```
