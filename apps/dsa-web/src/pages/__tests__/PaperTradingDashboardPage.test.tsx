import { render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import PaperTradingDashboardPage from '../PaperTradingDashboardPage';

const state = {
  strategy: 'us_quality_momentum',
  scorecard_version: 'us_evidence_v2',
  research_status: 'not_validated',
  updated_at: '2026-08-11T03:12:48Z',
  latest_market_date: '2026-08-10',
  benchmarks: ['SPY', 'QQQ', 'IWM', 'DIA', 'RSP'],
  config: {
    initial_capital: 100000,
    holding_days: 10,
    per_side_cost_bps: 10,
    minimum_universe_coverage: 0.95,
    minimum_completed_cycles: 20,
    top_k: 5,
  },
  portfolios: {
    large_cap_22: {
      universe: ['LLY'],
      closed_cycles: [],
      active_cycle: {
        status: 'pending',
        signal_date: '2026-08-10',
        selected: [{
          code: 'LLY',
          screen_score: 71.3253,
          signal_close: 1231.94,
          scorecard_version: 'us_evidence_v2',
          score_explanation: '0-100 规则排序分，不是上涨概率或预期收益。',
          factor_scores: {
            trend_confirmation: 82.4,
            momentum: 76.1,
            risk_control: 68.7,
            liquidity: 74.2,
            relative_strength: 90,
            data_quality: 100,
          },
          factor_weights: {
            trend_confirmation: 0.25,
            momentum: 0.2,
            risk_control: 0.2,
            liquidity: 0.15,
            relative_strength: 0.1,
            data_quality: 0.1,
          },
          reasons_pass: ['收盘价站上 MA20', 'MA5 >= MA20 >= MA60，均线结构偏多'],
          reasons_watch: ['估值、市值与财务质量未接入本次日线回放，未参与评分'],
          risk_flags: [],
          selection_thesis: '趋势确认、动量质量贡献居前并通过全部硬筛选；未触发额外技术风险标记。',
          invalidation_conditions: ['收盘价跌破 MA20 或 MACD 转为 bearish'],
          data_sources: ['Yahoo Finance 复权 OHLCV 日线（收盘后信号）'],
          data_gaps: ['估值、市值与财务质量未接入本次日线回放，未参与评分'],
        }],
        coverage: { expected_count: 22, covered_count: 22, ratio: 1 },
      },
      snapshots: [{
        date: '2026-08-10',
        strategy_equity: 100000,
        strategy_total_return_pct: 0,
        strategy_daily_return_pct: 0,
        strategy_drawdown_pct: 0,
        benchmark_total_return_pct: { SPY: 0, QQQ: 0, IWM: 0, DIA: 0, RSP: 0 },
      }],
    },
  },
  live_validation: {
    effective: false,
    status: 'insufficient_or_failed',
    warning: 'Paper trading is observational evidence.',
    diagnostics: {
      large_cap_22: {
        completed_cycles: 0,
        required_completed_cycles: 20,
        positive_excess_benchmarks: 0,
        required_positive_excess_benchmarks: 3,
        effective: false,
      },
    },
  },
};

describe('PaperTradingDashboardPage', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: async () => state,
    }));
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('renders the auditable portfolio state without a backend API', async () => {
    render(<PaperTradingDashboardPage />);

    expect(await screen.findByRole('heading', { name: '模拟组合收益与候选追踪' })).toBeInTheDocument();
    expect(screen.getByText('证据不足或未通过')).toBeInTheDocument();
    expect(screen.getAllByText('LLY')).toHaveLength(2);
    expect(screen.getAllByText('71.3')).toHaveLength(2);
    expect(screen.getByRole('heading', { name: '候选证据卡' })).toBeInTheDocument();
    expect(screen.getAllByText(/趋势确认、动量质量贡献居前/)).toHaveLength(2);
    expect(screen.getByText('趋势确认')).toBeInTheDocument();
    expect(screen.getByText('收盘价站上 MA20')).toBeInTheDocument();
    expect(screen.getAllByText(/估值、市值与财务质量未接入/).length).toBeGreaterThan(0);
    expect(screen.getByText('未触发额外技术风险标记')).toBeInTheDocument();
    expect(screen.getByText('等待次日开盘')).toBeInTheDocument();
    expect(screen.getByText('0 / 20')).toBeInTheDocument();
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(1));
  });
});
