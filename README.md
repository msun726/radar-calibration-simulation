# Sensor Calibration Simulation

面向传感器数据岗位的 Python 学习实验。**全部为合成液位数据，不使用 Agora 数据，不代表真实雷达精度、生产性能或云成本。** 代码由 AI 辅助编写，适合逐步阅读、修改和复现实验后用于面试讨论。

## 快速运行

需要 Python 3.10+，在本文件所在目录打开终端：

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m unittest -v
.venv\Scripts\python calibration.py
```

输出到 `results/`。默认随机种子 42；可用 `python calibration.py --seed 123 --output results-seed123` 检查结果对其他随机样本是否稳定。已附的结果来自本次实际运行，计时依赖机器和环境。

## 实验一：校准误差

假设每台传感器存在固定增益误差、固定偏移、随机噪声及少量异常值：

`raw = gain * reference + offset + noise`

拟合 `reference ≈ slope * raw + intercept`，用 NumPy 最小二乘求解。比较原始读数、普通 OLS、训练残差 MAD 筛选后重新拟合的 OLS。

- 24 台合成传感器，每台 180 次参考测量；一行是一次参考测量事件，不是每十分钟的遥测。
- 0–15 号用于建立历史通用模型；16–23 号为新传感器，按传感器隔离。
- 对新传感器，用前 60 次测量训练，最后 60 次测试；中间 60 次不用。所有方法使用相同的未来测试集。
- 异常标签只用于保留模拟生成记录，**不参与模型训练或筛选**；测试集中的异常读数保留。
- MAE 是平均绝对误差，RMSE 对大误差更敏感；CSV 和 JSON 的单位是米，图表误差单位是毫米。

这是输入有噪声时的近似线性预测，并不是严格的误差变量模型，也不复刻雷达信号处理。

## 实验二：Day-one / 冷启动

新传感器分别只有 3、5、10、20、40、60 次参考读数时，比较：

1. 用其他 16 台传感器建立的历史通用模型。
2. 仅用该新传感器已有读数建立的局部 OLS 模型。

所有点都在固定的后续测试集上计算误差。曲线是 8 台新传感器的平均值，阴影是跨传感器四分位范围，**不是置信区间**。少量读数覆盖范围窄或含异常时，局部拟合可能很差；数据增加也不保证每一步都改善。历史通用模型的表现取决于设备间差异，不能据此断言现场一定更好。

## 实验三：全量重算与增量更新

生成另外 100,000 条静态线性数据，分 100 批到达：

- 全量：每批到达后对截至当前的全部历史重新拟合。
- 增量：维护样本数、均值、中心化平方和与交叉乘积，仅处理新增批次。

每次更新都核对两种方式的系数。预热后交替执行 5 次，报告中位耗时。测量包含更新和拟合，不包含生成数据、图表、网络或云服务。该结果体现避免重复扫描历史的收益，**不等于生产加速倍数或 AWS 费用下降**。

此增量方法对应普通、无权重、全历史 OLS；不适用于直接替代前面的异常筛选拟合。历史数据修改、滑动窗口、漂移和重新筛选都需要另外设计。这里没有实现并行计算、云部署、内存 profiling 或事件基础设施。

## 数据质量观察

`training_quality_indicators.csv` 输出参考数量、原始读数跨度、筛除比例和保留训练点的残差 RMSE。这些是可调查的信号，**不是经过验证的健康评分或校准可信度**。较低训练误差不能替代未来误差；读数跨度也可能被异常值放大。

## 文件

| 文件 | 内容 |
|---|---|
| `calibration.py` | 数据生成、校准、评估、增量基准和绘图 |
| `test_calibration.py` | 已知映射恢复、每个增量前缀一致性、退化数据、异常筛选、可复现性测试 |
| `results/comparison.png` | 三项实验图 |
| `results/results.json` | 汇总结果和每次计时 |
| `results/accuracy_by_sensor.csv` | 每台传感器的未来误差 |
| `results/day_one_by_sensor.csv` | 冷启动逐传感器结果 |
| `results/holdout_predictions.csv` | 测试预测供复核 |
| `results/synthetic_readings.csv` | 合成输入与生成时的异常标签 |

## 局限与下一步

固定线性偏差、无时间漂移、参考值无误差，都是简化假设；当前数据不是物理时间序列，不模拟充放液动态、自相关、泡沫、搅拌器、罐形和液位到体积的几何转换。因此适合演示校准评估思路，不能验证真实传感器系统。

下一步可以逐个加入参考测量误差、时间漂移和填充范围不足；跑多个随机种子，逐设备检查失败案例，再讨论正则化、部分池化或稳健回归。先解释现有结果，再增加复杂度。

## 面试如何讲

先能自己解释这四点：为什么按时间留出测试数据；为什么三条数据拟合可能不稳定；MAD 筛选的局限；增量统计为何与全量 OLS 一致。主动说明模拟假设、AI 辅助范围，以及尚未做过的现场验证和云部署。

可用的英文描述（阅读并复现后）：

> Built an AI-assisted Python simulation to compare sensor calibration accuracy, cold-start behaviour and incremental least-squares updates using synthetic liquid-level data. Evaluated models on later reference readings and verified incremental coefficients against full refits.

不要写成在 Agora 工作过、真实雷达校准系统、已部署的云平台，或独立完成尚未理解的算法。
