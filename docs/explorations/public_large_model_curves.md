# 公开可下载的大参数模型曲线可视化

本次只保留当前可以直接结构化下载并重建的公开曲线，统一画成同图双轴：左轴为 loss，右轴为 learning rate。

| Source | Task | Params | Parsed Points | LR Source | Output |
| --- | --- | --- | ---: | --- | --- |
| OLMoE SFT | SFT | 1.3B active / 6.9B total | 9500 | official log | `olmoe_sft.png` |
| OLMoE DPO | DPO | 1.3B active / 6.9B total | 5712 | official log | `olmoe_dpo.png` |
| Mistral-7B SFT beta | SFT | 7B | 1 | reconstructed cosine | `mistral_7b_sft_beta.png` |
| Zephyr-7B alpha | DPO | 7B | 19 | reconstructed linear | `zephyr_7b_alpha.png` |
| Zephyr-7B beta | DPO | 7B | 58 | reconstructed linear | `zephyr_7b_beta.png` |

## 说明

- `OLMoE SFT` 与 `OLMoE DPO` 直接来自官方 GitHub 原始日志，`LR` 和 `Loss` 都是日志中逐步记录的真实值。
- `Mistral-7B SFT beta`、`Zephyr-7B alpha`、`Zephyr-7B beta` 来自 Hugging Face 原始 `README.md` 训练结果表；`Loss` 取训练结果表中的 `Training Loss`，`LR` 根据模型卡中公开的 `learning_rate`、`lr_scheduler_type`、`lr_scheduler_warmup_ratio` 和估计总步数重建。
- 这里的 `total_steps_estimate` 是用 `last_step / last_epoch * num_epochs` 从模型卡表格反推得到，因此 Hugging Face 这几条的 `LR` 是公开超参数下的可重建 schedule，不是站点直接提供的逐步学习率历史。
