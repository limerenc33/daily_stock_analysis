# 美股选股策略与链路说明

本文记录当前项目对美股选股的实现边界、外部开源项目调研结论，以及三套可以直接运行的内置策略。策略输出是研究和模拟交易候选，不构成投资建议，也不会自动向券商下单。

## 外部项目调研

仓库热度和 GitHub star 会随时间变化，因此这里记录项目和可复用的设计，不把某个时间点的 star 数当成稳定指标。

| 项目 | 资源 | 对本项目的借鉴 |
| --- | --- | --- |
| Microsoft Qlib | [文档](https://qlib.readthedocs.io/en/latest/) / [GitHub](https://github.com/microsoft/qlib) | 将预测信号、组合策略、交易执行解耦；`TopkDropoutStrategy` 通过 Top-K 与有限换手管理持仓；强调 point-in-time 数据、交易成本和组合级指标。 |
| FinRL | [文档](https://finrl.readthedocs.io/en/latest/) / [GitHub](https://github.com/AI4Finance-Foundation/FinRL) | 数据层、环境层、训练层分离；显式建模交易成本、保证金/做空限制、余额约束和风险状态；缺失数据不能静默当成有效数据。 |
| vectorbt | [文档](https://vectorbt.dev/) / [GitHub](https://github.com/polakowo/vectorbt) | 使用 pandas/NumPy 批量测试股票、参数、时间窗口和策略组合，适合补充参数网格与 walk-forward 验证。 |
| QuantConnect LEAN | [GitHub](https://github.com/QuantConnect/Lean) | 可作为未来接入美股回测、手续费、滑点和券商模拟执行的独立引擎参考；当前项目暂不引入其运行时。 |
| yfinance | [GitHub](https://github.com/ranaroussi/yfinance) | 当前美股快照和日线适配器的数据源；适合研究用途，生产交易仍应增加供应商冗余、数据质量和授权检查。 |

这些项目共同指向同一个落地顺序：先把股票池、特征、信号、组合约束和交易成本拆开，再做回测和实盘适配；不要把单一技术指标直接包装成自动买入指令。

## 当前美股链路

1. `snapshot_us.py` 使用 yfinance 拉取快照。股票池由 `SCREENING_US_UNIVERSE_SOURCE` 控制：`env` 读取 `SCREENING_US_TICKERS`，`sp500` 抓取 Wikipedia 的 S&P 500 成分，`default` 使用内置大盘股列表，`auto` 按 `env -> sp500 -> default` 降级。
2. `pipeline.screen(..., market="us")` 校验策略的 `market_scope`，先执行快照硬过滤，再对缩小后的候选补日线特征。
3. 美股且 `DAILY_SOURCE=auto` 时，管线自动选择 `yfinance` 日线；中国市场仍沿用原有 `auto` 源顺序。
4. L1 只输出候选和分数。已有 L2 LLM 重排、风险 overlay、scorecard 和可选 DSA 深度分析，但它们不会替代人工确认或券商风控。

## 内置策略

### `us_quality_momentum`

适用于流动性较好的美股大盘股趋势观察。硬条件包含成交额、市值、价格、MA20 上方、60 日表现、MACD、波动和回撤约束；排序以 momentum、liquidity、stability、size 为主。

适合风险偏好中等、持仓周期偏 swing 的场景。趋势延续不等于未来收益，财报、监管、盘前盘后流动性和跳空风险需要额外检查。

### `us_breakout_continuation`

适用于短中线突破候选。硬条件要求突破前有收敛、当日有量能确认、收盘站上 MA20、信号分数较高，并限制 ATR 和区间波动。

该策略最容易受到追高、假突破、财报跳空和成交量失真影响。执行层必须另行定义入场滑点、止损、单股仓位上限和财报 blackout window。

### `us_low_volatility_quality`

适用于防守型观察池。硬条件限制 20 日年化波动、ATR、最大回撤和区间幅度，同时要求较高流动性和市值；排序以 stability、liquidity、size 为主。

低波动不是低风险，利率变化、行业集中和基准风格切换仍可能造成持续回撤。

## 为什么不把 PE/PB 做成美股硬过滤

yfinance 的 `fast_info` 和 `info` 对估值字段的覆盖并不稳定，ADR、亏损公司、数据延迟和供应商字段差异都会产生空值。当前过滤器对配置的数值条件采用“缺失即拒绝”语义，因此在美股策略中把 PE/PB 配成 hard filter 会造成系统性误杀。三套策略只将 value 设为低权重排序因子；需要估值约束时，应增加有明确更新时间和口径的基本面数据源，并单独记录数据质量。

## 回测和收益监控要求

在把候选接到购买建议或自动监控前，至少补齐以下验证：

- 基准：`SPY`（S&P 500）和 `QQQ`（Nasdaq 100），同时报告超额收益。
- 指标：CAGR、Sharpe、最大回撤、Calmar、胜率、盈亏比、换手率、手续费和滑点后的净收益。
- 方法：按时间切分的 walk-forward；所有财报、成分股和估值字段必须 point-in-time，禁止使用未来修订值。
- 组合：Top-K、单票权重上限、行业/主题集中度、现金比例、再平衡频率和停牌/退市处理。
- 风险：VIX 或其他市场状态信号、财报事件窗口、隔夜跳空、美元现金和交易时段限制。
- 记录：每次筛选保存股票池版本、数据时间、策略版本、候选分数、最终人工决策、成交价、费用、持仓和退出原因。

当前仓库已提供筛选结果和 portfolio 相关 API，但尚未把上述回测、券商订单确认、交易 ledger 和个性化止盈/移动止损全部串成默认自动任务。下一阶段应先实现 paper-trading ledger 与定时收益快照，再考虑券商 API 和真实下单。

当前版本已提供独立历史回放器 `src/services/screening/us_backtest.py` 和命令行入口 `scripts/run_us_strategy_backtest.py`。回放器在每个历史信号日重算 `compute_daily_features()`，按硬过滤和 screen score 选 Top-K，下一交易日开盘入场，固定交易日数后收盘退出，并将交易成本/滑点计入净收益。由于免费行情通常没有可靠的 point-in-time 市值、PE/PB，回放结果会显式列出被排除的字段；这是一项可审计的技术信号验证，不应被描述为完整的基本面回测。

验证门槛默认要求至少 50 个非重叠持有周期、全样本超额收益为正、Sharpe 不低于 0.5，以及至少两个样本外 walk-forward 区间且多数区间跑赢基准。门槛通过才会输出 `validation_gate.effective=true`；门槛不通过应保留为研究结果，不进入自动购买建议。

## 使用示例

```bash
export SCREENING_US_UNIVERSE_SOURCE=env
export SCREENING_US_TICKERS=AAPL,MSFT,NVDA,AMZN,GOOGL
export DAILY_SOURCE=auto  # market=us 时管线自动改用 yfinance

python scripts/run_us_strategy_backtest.py \
  --start 2015-01-01 \
  --end 2025-12-31 \
  --source auto \
  --top-k 5 \
  --holding-days 10
```

`--source auto` 先尝试 yfinance，失败后尝试 Stooq，并将每只股票实际使用的源写入结果 JSON。运行前请确认网络和数据授权可用；下载失败时不要用合成数据替代真实回测。结果位于 `data/us_backtest/results.json`，缓存位于 `data/us_backtest/`。
