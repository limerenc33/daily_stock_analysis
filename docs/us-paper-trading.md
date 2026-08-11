# 美股策略每日模拟交易

该任务使用当前历史综合验证中表现相对最好的 `us_quality_momentum` 作为研究对照，但不把它标记为已验证策略。它同时运行两个正式股票池，并比较 `SPY`、`QQQ`、`IWM`、`DIA`、`RSP`，避免把“只跑赢 SPY”当成有效。

## 交易规则

- 股票池：`large_cap_22` 与 `diversified_60`。
- 每个股票池初始模拟资金 100,000 美元，Top 5 等权，可使用碎股。
- 收盘后生成信号，下一交易日开盘模拟成交。
- 持有 10 个交易日，在第 10 个交易日收盘模拟退出；周期不重叠。
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

## GitHub Actions 部署

`.github/workflows/us-paper-trading.yml` 在每个美股交易日收盘后运行，即北京时间周二至周六约 06:30。任务会：

1. 从 `paper-trading-state` 分支恢复上次账本。
2. 下载真实 Yahoo 复权日线并推进两个模拟组合。
3. 将日报写入 Actions Summary 并保存 90 天 artifact。
4. 创建或更新仓库中的“美股模拟交易日报”Issue，后续每天追加评论。
5. 将最新状态提交到独立的 `paper-trading-state` 分支。
6. 若仓库已配置通知 Secret，同时发送到已有企业微信、飞书、Telegram、邮件、ntfy、Gotify、PushPlus、Server酱、自定义 Webhook、Discord 或 Slack 渠道。

手动验证可以在 Actions 页面运行 `US Paper Trading Daily` workflow。首次执行只生成候选信号；下一交易日收盘后的执行才会记录下一交易日开盘成交。
