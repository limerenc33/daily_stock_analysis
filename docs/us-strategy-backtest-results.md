# 美股策略真实历史回测结果

> 综合验证结论（2026-08-10）：三套策略均未通过多股票池、多基准、持有期、成本压力和统计置信度组成的综合门槛。此前 `us_quality_momentum` 仅相对 SPY 通过单项门槛，不应被描述为策略整体有效，也不应进入自动购买建议。

> 复审状态（2026-08-11）：旧下载器没有为回测起始日预取 140 日特征 warm-up，且缓存没有严格验证日期范围。代码现已修复，综合矩阵已用扩展历史窗口重跑；以下第一阶段表格与综合验证结果均以修正后的结果为准。

## 运行记录

- 数据窗口：2015-01-01 至 2025-12-31
- 数据源：Yahoo Finance（`yfinance 1.5.2`），76/76 个标的下载成功，无合成数据或 Stooq 降级；请求历史窗口从 2014-03-27 开始，用于覆盖 140 日特征 warm-up
- 股票池：22 只内置美股 + `SPY`、`QQQ` 基准
- 信号与执行：每日重算生产特征，下一交易日开盘入场，持有 10 个交易日后收盘退出，Top 5 等权
- 成本假设：单边手续费 5 bps + 滑点 5 bps；无候选周期保留为现金
- 验证门槛：至少 50 个周期和交易、总超额收益为正、Sharpe >= 0.5、至少两个样本外区间且多数样本外区间超额为正

当前可审计结果：[`data/us_validation/results.json`](../data/us_validation/results.json)。第一阶段数字取其 `large_cap_22` 单股票池结果；行情缓存位于 [`data/us_backtest/`](../data/us_backtest/)。旧的 [`data/us_backtest/results.json`](../data/us_backtest/results.json) 仅保留作历史复现，不作为当前数值依据。

## 第一阶段：单股票池结果

| 策略 | 基准 | 周期/交易数 | CAGR | Sharpe | 最大回撤 | 胜率 | 基准收益 | 超额收益 | 门槛 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `us_quality_momentum` | SPY | 276 / 1220 | 11.5731% | 0.6228 | -40.5571% | 57.9710% | 115.9318% | **+115.8792pp** | 研究项（综合未通过） |
| `us_quality_momentum` | QQQ | 276 / 1220 | 11.5731% | 0.6228 | -40.5571% | 57.9710% | 268.2157% | -36.4047pp | 不通过 |
| `us_breakout_continuation` | SPY | 276 / 111 | -7.4575% | -0.3953 | -66.3990% | 14.1304% | 115.9318% | -173.1404pp | 不通过 |
| `us_low_volatility_quality` | SPY | 276 / 1248 | 5.0250% | 0.3607 | -41.6717% | 58.6957% | 115.9318% | -44.8494pp | 不通过 |

`QQQ` 对另外两套策略的结论同样是不通过；完整字段和每个基准的结果见 JSON。

## 第二阶段：综合验证设计

综合验证使用 76 个唯一 Yahoo Finance 标的、209,152 行真实复权日线：

- 股票池 A：22 只原始大盘股；股票池 B：60 只跨科技、金融、工业、医疗、消费、能源、公用事业等行业的股票。
- 探索池：11 只 SPDR 行业 ETF，用于检查信号跨行业资产的泛化，不参与正式通过门槛。
- 基准：`SPY`、`QQQ`、`IWM`、`DIA`、`RSP`，避免只挑一个有利基准。
- 参数压力：5、10、20 个交易日持有期；20 bps 和 50 bps 往返成本。
- 时间检验：2015-2019 历史参考，2020-2022 与 2023-2025 两个冻结参数样本外区间。
- 阶段检验：另按 2015-2019、2020、2021、2022、2023-2025 五个固定日历阶段输出指标，不在阶段之间重新拟合参数。
- 统计检验：对非重叠持有周期的平均超额收益做固定随机种子的 2,000 次 bootstrap，要求 95% 置信区间下界为正。
- 数据质量：76/76 个文件读取成功，日期有序且无重复，OHLC 无缺失和非正值，全部更新到 2025-12-31。XLC、XLRE 成立较晚，因此 ETF 池只有 189 个共同周期。

综合通过必须同时满足：两个股票池各跑赢至少 3/5 基准且策略自身盈利、中位 Sharpe 不低于 0.5、两个股票池各有至少 3/5 基准的样本外多数区间为正、两个股票池各有至少 3/5 固定阶段同时盈利并跑赢 SPY、两个股票池各有至少 2/3 持有期盈利并跑赢 SPY、50 bps 成本下两个股票池均盈利并跑赢 SPY，以及 10 个股票池×基准组合中至少 6 个 bootstrap 置信下界为正。门槛在查看综合结果前固定。

原始综合结果：[`data/us_validation/results.json`](../data/us_validation/results.json)。可复现入口：[`scripts/run_us_strategy_validation_suite.py`](../scripts/run_us_strategy_validation_suite.py)。

## 综合验证结果

| 策略 | 两股票池正超额基准数 | 中位 Sharpe | 样本外多数为正 | 固定阶段稳健性 | 稳健持有期 | 50 bps 成本 | bootstrap 下界为正 | 结论 |
| --- | --- | ---: | --- | --- | --- | --- | ---: | --- |
| `us_quality_momentum` | 4/5、3/5 | 0.4962 | 5/5、3/5 | 3/5、1/5 | 3/3、1/3 | 通过、失败 | 0/10 | **未通过** |
| `us_breakout_continuation` | 0/5、0/5 | -0.2352 | 0/5、0/5 | 0/5、1/5 | 0/3、0/3 | 失败、失败 | 0/10 | **未通过** |
| `us_low_volatility_quality` | 3/5、3/5 | 0.4125 | 1/5、1/5 | 3/5、1/5 | 1/3、0/3 | 失败、失败 | 0/10 | **未通过** |

`us_quality_momentum` 是三者中相对最好的一套，但扩展到 60 只股票后，10 日策略收益为 83.0645%，低于 SPY 的 115.9318%，Sharpe 从 0.6228 降至 0.3695；两股票池的中位 Sharpe 为 0.4962，低于 0.5 门槛。5/10/20 日中，60 只股票池只有 1 个持有期同时盈利并跑赢 SPY，50 bps 成本下该股票池失败。固定阶段检验中，大盘股池为 3/5，跨行业池为 1/5。其 bootstrap 最强组合的正均值概率也没有达到 95%，所有 10 个正式组合的 95% 下界均跨过零。

行业 ETF 探索池进一步显示信号缺乏跨资产泛化：三套策略全样本收益分别为 -6.8225%、-13.6543% 和 +5.3270%，均明显落后 SPY。这个结果不用于正式淘汰门槛，但与“策略尚未稳健”的结论一致。

## 训练冻结候选 v2

为避免在 2020-2025 样本外区间反复调参，研究脚本 [`scripts/research_us_momentum_v2.py`](../scripts/research_us_momentum_v2.py) 只读取 2015-2019 收益，预先固定五组趋势候选和选择顺序。`trend_balanced` 在训练段两个股票池的 10 个多基准组合全部自身盈利且正超额，中位 Sharpe 为 0.8771，中位超额收益为 42.53 个百分点，因此被选中并冻结。

冻结后的 `us_quality_momentum_v2` 只进行一次 2020-2025 综合样本外验证，结果未通过：

| 股票池 | 策略收益 | CAGR | Sharpe | 最大回撤 | 对 SPY 超额 | 正超额基准数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 大盘股 22 | 30.68% | 4.60% | 0.3170 | -31.00% | -31.09pp | 2/5 |
| 跨行业 60 | 13.40% | 2.14% | 0.2084 | -30.28% | -48.36pp | 0/5 |

两个股票池的 5/10/20 日持有期都只有 1/3 同时满足自身盈利和跑赢 SPY，50 bps 往返成本均失败，10 个正式基准组合的 bootstrap 95% 下界仍全部跨零。训练段优势没有延续到样本外，v2 被判定为过拟合或存在明显市场阶段依赖，已从生产策略目录移除并归档为 [`research/us_strategy_candidates/us_quality_momentum_v2_rejected.yaml`](../research/us_strategy_candidates/us_quality_momentum_v2_rejected.yaml)。

训练选择原始结果：[`data/us_research/momentum_v2_training.json`](../data/us_research/momentum_v2_training.json)。样本外原始结果：[`data/us_validation/us_quality_momentum_v2_oos_2020_2025.json`](../data/us_validation/us_quality_momentum_v2_oos_2020_2025.json)。这次失败后不得再用 2020-2025 调整同一候选并声称该区间仍是样本外。

## 样本外分段

`us_quality_momentum` 对 SPY 的 walk-forward 超额收益如下（`large_cap_22`，warm-up 修正后）：

| 区间 | CAGR | Sharpe | 超额收益 |
| --- | ---: | ---: | ---: |
| 2015-2020（训练/历史参考） | 8.4129% | 0.5874 | +9.8959pp |
| 2020-2023（样本外） | 5.15% | 0.3285 | +13.45pp |
| 2023-2025（样本外） | 24.53% | 0.9884 | +40.39pp |

该表只说明原始 22 只股票池对 SPY 的分段表现，不能覆盖扩展股票池、其他基准、参数稳定性和统计显著性。综合矩阵已经证明它不足以支持“策略整体有效”的结论。

固定阶段诊断（策略参数未重拟合）：

| 股票池 | 2015-2019 | 2020 | 2021 | 2022 | 2023-2025 |
| --- | --- | --- | --- | --- | --- |
| 大盘股 22 | +49.76% / +9.90pp | +2.72% / -9.67pp | +24.18% / +10.25pp | -8.80% / +10.86pp | +90.44% / +40.39pp |
| 跨行业 60 | +50.57% / +10.70pp | -8.00% / -20.40pp | +9.98% / -3.96pp | -16.67% / +2.99pp | +43.36% / -6.69pp |

表中每格为“策略累计收益 / 相对 SPY 的组合超额收益”。跨行业池只有 2015-2019 阶段同时满足两项，说明策略收益对股票池和市场阶段敏感。

## 使用边界

这次回测没有可靠的 point-in-time 市值、PE、PB，因此这些字段在历史回放中被明确排除，结果不是完整基本面回测。股票池也不是历史成分股快照，存在幸存者偏差；Yahoo 免费数据的复权、退市和公司行动口径仍需在上线前增加第二数据源交叉校验。

因此当前建议是：三套策略都不得进入自动购买建议或真实下单。可以把 `us_quality_momentum` 保留为研究型 paper-trading 对照组，但界面必须明确标记“综合验证未通过”，不得把模拟盈利当成有效性证明。后续若开发新版本，应只在训练区间完成规则和参数选择，冻结后进行新的、未被调参观察过的样本外检验，并补充历史成分股、退市股票、point-in-time 基本面、财报事件窗口和第二行情源。

## Point-in-time 数据复审

已验证公开仓库 [`fja05680/sp500`](https://github.com/fja05680/sp500) 提供 1996 年以来按日期记录的 S&P 500 历史成分，可用于消除“用今天的成分股回测过去”的前视偏差。仓库维护者同时明确说明 Yahoo Finance 不保留全部退市和改名股票，完整股票回测需要 Norgate Data、EODData 或其他含退市证券的授权数据源。

已对 2015-07-01 的真实 PIT 截面做 Yahoo 覆盖审计：500 个历史成员中只有 380 个能返回该周日线，覆盖率 `76.00%`，120 个返回“possibly delisted / no price data”。因此即使成分股时间点正确，免费 Yahoo 行情仍不能满足 95% 动态回测覆盖门槛。审计原始结果：[`data/us_validation/yahoo_pit_coverage_2015.json`](../data/us_validation/yahoo_pit_coverage_2015.json)。

项目已增加 [`src/services/screening/us_point_in_time.py`](../src/services/screening/us_point_in_time.py)：只允许读取信号日或此前的最近成分快照；动态回测在可用特征覆盖率低于 95% 时直接失败，不会静默忽略缺失退市股票。覆盖率审计入口为 [`scripts/audit_us_pit_coverage.py`](../scripts/audit_us_pit_coverage.py)。

```bash
PYTHONPATH=/tmp/dsa-test-deps \
python3.12 scripts/audit_us_pit_coverage.py \
  --universe-file '/path/to/S&P 500 Historical Components & Changes (Updated).csv' \
  --dates 2015-07-01 2020-07-01 2025-07-01 \
  --required-coverage 0.95
```

warm-up 修复后的固定股票池矩阵已经完成，但 PIT 动态股票池仍因退市行情覆盖率不足 95% 而无法运行。策略有效性目标仍未完成。`results.json` 中每个 baseline 单元的 `regime_metrics` 保存了五阶段原始诊断。

运行命令：

```bash
PYTHONPATH=.:/tmp/dsa-test-deps \
python3.12 scripts/run_us_strategy_backtest.py \
  --start 2015-01-01 --end 2025-12-31 \
  --source auto --top-k 5 --holding-days 10
```

综合验证命令：

```bash
PYTHONPATH=/tmp/dsa-test-deps \
python3.12 scripts/run_us_strategy_validation_suite.py \
  --start 2015-01-01 --end 2025-12-31 --source auto
```

复现已拒绝 v2 的样本外验证：

```bash
PYTHONPATH=/tmp/dsa-test-deps \
python3.12 scripts/run_us_strategy_validation_suite.py \
  --start 2020-01-01 --end 2025-12-31 --source auto \
  --strategies-dir research/us_strategy_candidates \
  --strategies us_quality_momentum_v2 \
  --output data/us_validation/us_quality_momentum_v2_oos_2020_2025.json
```
