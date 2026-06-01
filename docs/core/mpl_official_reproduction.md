# MPL 官方仓库公开实验严格复现

## 复现目标

本次复现严格采用 `MultiPowerLaw` 官方仓库公开代码、公开数据目录与公开训练/测试划分，不使用此前为课程任务改写的 `cosine-only -> WSD` 自定义脚本设定。

官方仓库位置：

- `external/MultiPowerLaw`

官方训练集划分见 [config.py](file:///Users/jiaju/Documents/github/DL-final/external/MultiPowerLaw/src/config.py#L4-L19)：

- 训练集：`cosine_24000.csv`、`constant_24000.csv`、`wsdcon_9.csv`
- 测试集：`constant_72000.csv`、`cosine_72000.csv`、`wsd_20000_24000.csv`、`wsdld_20000_24000.csv`、`wsdcon_3.csv`、`wsdcon_18.csv`

公开数据目录：

- `external/MultiPowerLaw/loss_curve_repo/csv_25`
- `external/MultiPowerLaw/loss_curve_repo/csv_100`
- `external/MultiPowerLaw/loss_curve_repo/csv_400`

## 运行方式

在官方仓库目录下分别执行：

```bash
python3 main.py --folder_path 25
python3 main.py --folder_path 100
python3 main.py --folder_path 400
```

本次实际运行环境：

- 平台：macOS
- Python：`python3`
- 依赖：`numpy`、`torch`、`scipy`、`matplotlib`、`tqdm`、`scikit-learn`

官方入口脚本：

- [main.py](file:///Users/jiaju/Documents/github/DL-final/external/MultiPowerLaw/main.py)

官方拟合实现：

- [fitting.py](file:///Users/jiaju/Documents/github/DL-final/external/MultiPowerLaw/src/fitting.py)

## 复现结果

下表中的“复现值”来自官方日志文件：

- [25.log](file:///Users/jiaju/Documents/github/DL-final/external/MultiPowerLaw/logs/25.log)
- [100.log](file:///Users/jiaju/Documents/github/DL-final/external/MultiPowerLaw/logs/100.log)
- [400.log](file:///Users/jiaju/Documents/github/DL-final/external/MultiPowerLaw/logs/400.log)

README 对照值来自：

- [README.md](file:///Users/jiaju/Documents/github/DL-final/external/MultiPowerLaw/README.md#L25-L47)

### 平均测试指标对照

| Scale | R2 README | R2 复现 | MAE README | MAE 复现 | RMSE README | RMSE 复现 | PredE README | PredE 复现 | WorstE README | WorstE 复现 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 25M  | 0.9988 | 0.9988023029 | 0.00376 | 0.0037597759 | 0.00465 | 0.0046515038 | 0.00110 | 0.0011020977 | 0.00409 | 0.0040946433 |
| 100M | 0.9983 | 0.9983007794 | 0.00435 | 0.0043482103 | 0.00592 | 0.0059189573 | 0.00142 | 0.0014248686 | 0.00583 | 0.0058293259 |
| 400M | 0.9978 | 0.9977621117 | 0.00484 | 0.0048350118 | 0.00730 | 0.0073054895 | 0.00168 | 0.0016793872 | 0.00995 | 0.0099483663 |

结论：

- 三个尺度的平均测试指标与官方 README 基本一致。
- 差异只出现在小数点后更深位，属于正常数值波动。
- 从公开材料角度，可以认为本次已经成功复现 `MPL` 官方仓库公开实验。

### 最优参数对照

参数顺序为：

`[L0, A, alpha, B, C, beta, gamma]`

| Scale | README 参数 | 复现参数 |
| --- | --- | --- |
| 25M | `[3.040, 0.525, 0.508, 363.788, 2.066, 0.583, 0.641]` | `[3.0404543615, 0.5246868260, 0.5078684456, 363.7889266658, 2.0656086774, 0.5827906057, 0.6414228857]` |
| 100M | `[2.651, 0.601, 0.453, 437.946, 2.132, 0.598, 0.655]` | `[2.6514472945, 0.6011519698, 0.4529576017, 437.9457946886, 2.1324646322, 0.5978527324, 0.6552377531]` |
| 400M | `[2.375, 0.654, 0.429, 523.425, 2.025, 0.594, 0.635]` | `[2.3747391968, 0.6542104812, 0.4287856924, 523.4249698734, 2.0246288978, 0.5935047023, 0.6347246957]` |

结论：

- 三组参数与官方 README 一致到三位小数。
- 说明官方代码、公开数据和当前环境下的优化过程是稳定可复现的。

## 输出文件

官方脚本本次实际生成的主要输出包括：

- 拟合图目录：
  - `external/MultiPowerLaw/25M/fit`
  - `external/MultiPowerLaw/100M/fit`
  - `external/MultiPowerLaw/400M/fit`
- 日志：
  - `external/MultiPowerLaw/logs/25.log`
  - `external/MultiPowerLaw/logs/100.log`
  - `external/MultiPowerLaw/logs/400.log`
- 优化 schedule：
  - `external/MultiPowerLaw/optimized_schedules/25.npy`
  - `external/MultiPowerLaw/optimized_schedules/100.npy`
  - `external/MultiPowerLaw/optimized_schedules/400.npy`

## 与此前自定义复现的区别

此前课程任务中的自定义复现设置是：

- 训练集：`cosine_24000.csv`、`cosine_72000.csv`
- 测试集：`WSD/WSDLD/WSDCon`

那一版是为了满足“`fit on cosine, evaluate on WSD`”的课程要求而改写的，不等同于官方 MPL 仓库公开实验。

本报告对应的是官方公开实验：

- 训练集：`cosine_24000 + constant_24000 + wsdcon_9`
- 测试集：其余 6 条官方公开曲线

因此，后续讨论实验现象时需要区分：

- `docs/reproduction_report.md`：课程要求下的自定义设定
- `docs/mpl_official_reproduction.md`：官方仓库公开实验的严格复现

## 当前判断

- 公开仓库范围内的 `MPL` 实验已经复现成功。
- 如果目标是“严格复现论文全部结果”，当前公开仓库仍只覆盖 `25M / 100M / 400M` 及其公开曲线，不包含论文后续补充提到的更大规模全部实验。
