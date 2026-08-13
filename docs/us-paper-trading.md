# 美股策略每日模拟交易

该任务使用 `us_quality_momentum` 2.1 作为研究对照，但不把它标记为已验证策略。2.1 参考 UZI-Skill 的“结论、证据、风险、数据缺口”组织方式，将公开新闻、SEC 财报/披露与分析师评级变化加入可审计证据层。定时任务不调用 LLM，不生成不可复现的投资人观点。它同时运行两个正式股票池，并比较 `SPY`、`QQQ`、`IWM`、`DIA`、`RSP`，避免把“只跑赢 SPY”当成有效。

## 选股要素与分数

候选先通过价格、成交额、60 日涨跌幅、MA20、趋势信号、MACD、20 日波动率、20 日最大回撤和 ATR 等硬筛选，再按以下六维规则分排序：

| 因子 | 权重 | 证据 |
| --- | ---: | --- |
| 趋势确认 | 25% | 收盘价与 MA20、MA5/20/60 结构、MACD、趋势信号分 |
| 动量质量 | 20% | 当日与 60 日涨跌幅、趋势信号、MACD，包含追高和过热惩罚 |
| 风险控制 | 20% | 20 日波动率、最大回撤、ATR 相对目标区间和硬上限 |
| 流动性容量 | 15% | 当日成交额在同日候选截面中的相对排名 |
| 相对强度 | 10% | 60 日涨跌幅在同日候选截面中的相对排名 |
| 数据质量 | 10% | OHLCV 完整性、价格/成交量合法性、缓存和降级标记 |

总分范围为 0-100，只用于同一交易日、同一股票池内排序，不是上涨概率、目标收益或买入置信度。流动性和相对强度使用截面排名，因此同一只股票在不同股票池中的分数可以不同。

当前 Yahoo 日线回放没有 point-in-time PE、PB、市值和财务质量数据。2.0 将这些字段明确标记为“未评分”，不再用缺失值默认分参与排名。每个候选同时保存因子分与权重、入选证据、观察项、风险标记、失效条件、数据来源和数据缺口；这些字段会随开仓和关闭交易一起保留。

## 每日资讯、财报与研报层

每日收盘任务通过 yfinance 的 Yahoo Finance 搜索、SEC filing 索引和公开分析师评级变更抓取资料，按公司新闻、财报、监管披露、分析师研究分类，并把标题、来源、发布时间和 URL 原样写入账本。资讯源不可用时记录为 `unavailable`，不把缺失数据解释为中性或利好。

2.1 对选股的影响采用保守、可复现规则：原技术 Top 15 进入资讯核验，最终选择 Top 5；明确的业绩超预期、上调指引、评级上调和股东回报事件最多增加 4 分，业绩不及预期、下调指引和评级下调扣分，监管、财务困境或会计风险最多扣 8 分；同一标的出现至少两类严重风险证据时排除。研报观点权重低于 SEC 披露，且没有 URL 的评级变化仍会注明来源。

这项影响在交易语义上有严格版本边界：部署时已经 `open` 或 `awaiting_settlement` 的周期只更新资讯展示，不改变候选、持仓、网格价或卖出规则；该周期结束后创建的下一 `pending` 周期才固化 2.1 的资讯调整。周期自身保存 `strategy_version` 与 `scorecard_version`，因此后续可以将 2.0 和 2.1 的模拟结果分开评估。

本次调研参考了以下公开实现，但没有直接复制其交易逻辑：

- [OpenBB](https://github.com/OpenBB-finance/OpenBB)：成熟的多提供方金融数据与 SEC 接口架构，适合作为未来数据源扩展参考。
- [FinBERT](https://github.com/ProsusAI/finBERT)：金融文本情绪模型；本版不直接采用，因为定时 Actions 的模型体积、推理可重复性和点时回测数据尚未验证。
- [FinGPT](https://github.com/AI4Finance-Foundation/FinGPT)：金融 LLM 与数据管线参考；本版只采用“证据与判断分层”的思路，不引入生成式交易决策。
- [StockSentimentTrading](https://github.com/jasonyip184/StockSentimentTrading) 与 [LSTMppo-DRL-StockTrader](https://github.com/MahanVeisi8/LSTMppo-DRL-StockTrader)：证明新闻情绪可以作为实验特征，但前者只有单次 Notebook 提交，后者属于 DRL 研究原型，都不足以直接进入当前可审计模拟账本。

## 交易规则

- 股票池：`large_cap_22` 与 `diversified_60`。
- 每个股票池初始模拟资金 100,000 美元，Top 5 等权，可使用碎股。
- 收盘后生成信号，下一交易日开盘模拟成交。
- 网格默认每格 3%：价格上涨一格（+3%）卖出一半，上涨两格（+6%）卖出剩余仓位；价格下跌两格（-6%）卖出全部剩余仓位。
- 10 个交易日是最长持有期；尚未被网格卖出的仓位在第 10 个交易日收盘模拟退出，周期不重叠。
- 盘中观察价穿越网格时按观察价模拟成交；开盘跳空穿越网格时按开盘或首次观察价成交，不假设可以成交在更优的触发价。
- 收盘任务使用真实日线 OHLC 回补盘中任务可能漏掉的网格触发。同一根日线同时触及止损和止盈时按止损先发生处理，避免利用未知的盘中路径高估收益。
- 单边综合成本 10 bps，包括手续费与滑点假设。
- 行情或特征覆盖率低于 95% 时任务直接失败，不会静默忽略缺失标的。
- 每个股票池至少完成 20 个周期，且组合盈利并跑赢至少 3/5 基准，实时观察才会标记通过；历史综合回测门槛仍独立生效。

该账本不连接券商，不发送订单，不构成投资建议。

## 本地运行

```bash
python scripts/run_us_paper_trading.py
```

启用已有通知渠道：

```bash
python scripts/run_us_paper_trading.py --notify
```

默认状态写入 `data/us_paper_trading/state.json`，日报写入 `data/us_paper_trading/latest.md`。相同市场日期重复执行是幂等的，不会重复成交或累计收益。

可通过环境变量或 GitHub Repository Variables 调整后续新开仓使用的网格参数：

```bash
US_GRID_STEP_PCT=3.0
US_GRID_TAKE_PROFIT_LEVELS=2
US_GRID_STOP_LOSS_LEVELS=2
US_NEWS_INTELLIGENCE_ENABLED=true
US_NEWS_MAX_ITEMS=8
US_NEWS_MAX_WORKERS=4
```

已开仓持仓会把实际网格价格固化到账本中，之后修改配置不会改写其历史成交。

## GitHub Actions 部署

`.github/workflows/us-paper-trading.yml` 在每个美股交易日收盘后运行，即北京时间周二至周六约 06:30。任务会：

1. 从 `paper-trading-state` 分支恢复上次账本。
2. 下载真实 Yahoo 复权日线并推进两个模拟组合。
3. 将组合收益、回撤、多基准比较和每只候选的选股理由写入 Actions Summary，并保存 90 天 artifact。
4. 若仓库启用了 Issues，创建或更新“美股模拟交易日报”Issue，后续每天追加评论；若 Issues 已关闭，该步骤降级跳过，不影响 artifact 与账本持久化。
5. 将最新状态提交到独立的 `paper-trading-state` 分支。
6. 若仓库已配置通知 Secret，同时发送到已有企业微信、飞书、Telegram、邮件、ntfy、Gotify、PushPlus、Server酱、自定义 Webhook、Discord 或 Slack 渠道。

`.github/workflows/us-grid-monitor.yml` 在覆盖美股常规交易时段的 UTC 窗口内每 15 分钟启动一次，并在脚本中再次校验 `America/New_York` 的 09:30-16:00。它只读取仍在持仓的代码，复用统一实时行情链路：配置 Longbridge 时优先使用 Longbridge，否则使用 Yahoo，并可由 Finnhub、Alpha Vantage 补充。陈旧或不可用报价不会触发成交。

只有网格实际触发时，盘中任务才会更新 `paper-trading-state`、重新部署 Pages 并发送通知。通知会包含触发价、模拟成交价、数量、剩余仓位和报价来源。GitHub Actions 调度可能延迟，15 分钟轮询也可能错过短暂穿价，因此这是可审计的模拟验证，不是执行级止损；真实资金应使用券商托管的条件单。

网格退出属于新的成交模型，旧版固定持有周期不会计入网格模型所需的 20 个完成周期。加入网格不代表策略已经有效；仍需积累真实前向模拟周期，并单独补做覆盖网格参数和盘中路径假设的历史回测。

手动验证可以在 Actions 页面运行 `US Paper Trading Daily` workflow。首次执行只生成候选信号；下一交易日收盘后的执行才会记录下一交易日开盘成交。

## 公开只读看板

`.github/workflows/deploy-paper-trading-pages.yml` 将 Web 应用构建为不依赖后端 API 的只读模拟交易看板，并发布到 GitHub Pages。它在前端代码进入 `main` 后部署一次，也会在 `US Paper Trading Daily` 每次成功完成后，从 `paper-trading-state` 分支读取最新账本并重新部署。

看板展示组合净值、收益、回撤、候选、五基准对比、验证进度，以及每只候选的核心理由、六维因子、观察项、风险和失效条件。开仓后还会展示入场价、止损线、下一止盈格、剩余仓位、最新记录价和最近网格成交；不开放设置、交易录入或真实下单。GitHub Pages 必须在仓库设置中启用并选择 GitHub Actions 作为发布来源。

## 版本与验证边界

2026-08-10 及以前的原综合历史回测使用 `us_quality_momentum` 1.0 权重，不能证明 2.0 或 2.1 有效。2.0 已使用相同的 2015-2025 Yahoo 数据重新完成多股票池、多基准、成本压力、样本外和统计置信度验证，但综合门槛仍未通过：中位 Sharpe 0.4587、跨行业池持有期与 50 bps 成本压力失败、bootstrap 正下界 0/10。2.1 的资讯特征缺少无幸存者偏差、带准确发布时间的历史资料集，因此不能用当前价格历史回测证明有效；它必须从下一周期开始与 2.0 分版本积累前向模拟证据，继续保持 `not_validated`。不得产生真实购买建议或连接券商下单。完整价格策略结果见 [`us-strategy-backtest-results.md`](./us-strategy-backtest-results.md)。
