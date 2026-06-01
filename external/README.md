# external/

本目录存放第三方仓库的 vendored 副本（已作为普通文件纳入本仓库，便于离线复现与数据依赖）。

## MultiPowerLaw

- 上游：https://github.com/thu-yao-01-luo/MultiPowerLaw
- vendored 自 commit：`8da9f19`（Update CITATION.cff）
- 用途：提供官方公开训练/测试 loss 曲线（`loss_curve_repo/csv_{25,100,400}`）与官方拟合入口 `main.py`，供 `repro/` 下脚本读取与复现。
- 许可：见 `MultiPowerLaw/LICENSE`。

> 该副本已移除其原始嵌套 `.git`，以普通文件形式纳入本仓库。如需同步上游更新，请到上述地址重新拉取。
