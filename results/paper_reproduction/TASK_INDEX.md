# 论文口径完整预测任务索引

## 训练任务

- `cosine_24000`
- `constant_24000`
- `wsdcon_9`

## 测试任务

- `constant_72000`
- `cosine_72000`
- `wsd_20000_24000`
- `wsdld_20000_24000`
- `wsdcon_3`
- `wsdcon_18`

## 结果目录

- `MPL` 官方原始预测图：`results/paper_reproduction/mpl_only`
- `MPL + Tissue + Ground Truth` 对比图：`results/paper_reproduction/mpl_vs_tissue_compare`

## 覆盖范围

- 尺度：`25M / 100M / 400M`
- 每个尺度 9 个任务：3 个训练任务 + 6 个测试任务
- `MPL` 原始图：共 27 张任务图，另含 3 张 `loss_monitor`
- `MPL vs Tissue` 对比图：共 27 张任务图，另含 1 张平均测试 MAE 汇总图

## 文件命名

- `mpl_only/25M/constant_72000_mplfit.png`：官方 `MPL` 在 `25M` 上对 `constant_72000` 的预测图
- `mpl_vs_tissue_compare/25_compare_constant_72000.png`：`25M` 上 `Ground Truth + MPL + Tissue` 对比图