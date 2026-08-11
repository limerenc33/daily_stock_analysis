import { render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import PaperTradingDashboardPage from '../PaperTradingDashboardPage';

const state = {
  strategy: 'us_quality_momentum',
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
        selected: [{ code: 'LLY', screen_score: 71.3253, signal_close: 1231.94 }],
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
    expect(screen.getByText('LLY')).toBeInTheDocument();
    expect(screen.getByText('71.3')).toBeInTheDocument();
    expect(screen.getByText('等待次日开盘')).toBeInTheDocument();
    expect(screen.getByText('0 / 20')).toBeInTheDocument();
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(1));
  });
});
