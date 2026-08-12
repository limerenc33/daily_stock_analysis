import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  CalendarDays,
  CheckCircle2,
  Clock3,
  Database,
  Eye,
  ExternalLink,
  Github,
  Grid2X2,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  TrendingDown,
  TrendingUp,
} from 'lucide-react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

type Candidate = {
  code: string;
  screen_score: number;
  signal_close: number;
  scorecard_version?: string;
  score_explanation?: string;
  factor_scores?: Record<string, number | null>;
  factor_weights?: Record<string, number>;
  evidence_metrics?: Record<string, string | number | boolean | null>;
  reasons_pass?: string[];
  reasons_watch?: string[];
  risk_flags?: string[];
  selection_thesis?: string;
  invalidation_conditions?: string[];
  data_sources?: string[];
  data_gaps?: string[];
};

type Snapshot = {
  date: string;
  strategy_equity: number;
  strategy_total_return_pct: number;
  strategy_daily_return_pct: number;
  strategy_drawdown_pct: number;
  benchmark_total_return_pct: Record<string, number>;
};

type GridFill = {
  date: string;
  observed_at?: string | null;
  reason: string;
  grid_level?: number | null;
  trigger_price: number;
  fill_price: number;
  quantity: number;
  source: string;
};

type GridPosition = Candidate & {
  status: string;
  entry_date: string;
  entry_open: number;
  quantity: number;
  remaining_quantity: number;
  last_price?: number;
  fills: GridFill[];
  grid: {
    step_pct: number;
    take_profit_prices: number[];
    stop_loss_price: number;
    completed_take_profit_levels: number[];
  };
};

type GridEvent = GridFill & {
  type: string;
  code: string;
  remaining_quantity: number;
};

type ActiveCycle = {
  status: string;
  signal_date?: string;
  entry_date?: string;
  exit_date?: string;
  selected?: Candidate[];
  positions?: GridPosition[];
  coverage?: {
    expected_count: number;
    covered_count: number;
    ratio: number;
  };
};

type Portfolio = {
  universe: string[];
  active_cycle?: ActiveCycle | null;
  closed_cycles: unknown[];
  snapshots: Snapshot[];
  event_log?: GridEvent[];
};

type ValidationDiagnostic = {
  completed_cycles: number;
  required_completed_cycles: number;
  positive_excess_benchmarks: number;
  required_positive_excess_benchmarks: number;
  effective: boolean;
};

type PaperTradingState = {
  strategy: string;
  scorecard_version?: string;
  research_status: string;
  updated_at: string;
  latest_market_date: string;
  benchmarks: string[];
  config: {
    initial_capital: number;
    holding_days: number;
    per_side_cost_bps: number;
    minimum_universe_coverage: number;
    minimum_completed_cycles: number;
    top_k: number;
    grid_step_pct?: number;
    grid_take_profit_levels?: number;
    grid_stop_loss_levels?: number;
  };
  portfolios: Record<string, Portfolio>;
  live_validation: {
    effective: boolean;
    status: string;
    diagnostics: Record<string, ValidationDiagnostic>;
    warning: string;
  };
};

const DEFAULT_DATA_URL = 'https://limerenc33.github.io/daily_stock_analysis/paper-trading-state.json';
const DATA_URL = import.meta.env.VITE_PAPER_TRADING_DATA_URL?.trim() || DEFAULT_DATA_URL;
const AUTO_REFRESH_MS = 60_000;
const ACTIONS_URL = 'https://github.com/limerenc33/daily_stock_analysis/actions/workflows/us-paper-trading.yml';
const REPOSITORY_URL = 'https://github.com/limerenc33/daily_stock_analysis';
const PORTFOLIO_LABELS: Record<string, string> = {
  large_cap_22: '大盘股 22',
  diversified_60: '跨行业 60',
};
const FACTOR_LABELS: Record<string, string> = {
  trend_confirmation: '趋势确认',
  momentum: '动量质量',
  risk_control: '风险控制',
  liquidity: '流动性容量',
  relative_strength: '相对强度',
  data_quality: '数据质量',
};
const FACTOR_ORDER = Object.keys(FACTOR_LABELS);
const CHART_COLORS: Record<string, string> = {
  strategy: '#22d3ee',
  SPY: '#f59e0b',
  QQQ: '#a78bfa',
  IWM: '#34d399',
  DIA: '#fb7185',
  RSP: '#60a5fa',
};

const formatMoney = (value: number) => new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
}).format(value);

const formatPercent = (value: number, digits = 2) => `${value >= 0 ? '+' : ''}${value.toFixed(digits)}%`;

const formatUpdatedAt = (value: string) => {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value || '未知';
  }
  return new Intl.DateTimeFormat('zh-CN', {
    timeZone: 'Asia/Shanghai',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  }).format(parsed);
};

const cycleLabel = (status?: string) => ({
  active: '持仓中',
  open: '持仓中',
  awaiting_settlement: '等待收盘结算',
  closed: '已退出',
  idle: '等待信号',
  pending: '等待次日开盘',
}[status || ''] || status || '暂无周期');

const exitReasonLabel = (reason?: string) => ({
  grid_take_profit: '网格止盈',
  grid_stop_loss: '网格止损',
  time_exit: '到期退出',
}[reason || ''] || reason || '退出');

const Metric = ({
  label,
  value,
  detail,
  tone = 'neutral',
}: {
  label: string;
  value: string;
  detail: string;
  tone?: 'neutral' | 'positive' | 'negative';
}) => (
  <div className="min-w-0 px-4 py-4 sm:px-5">
    <div className="text-xs font-medium text-muted-text">{label}</div>
    <div className={`mt-1 truncate text-xl font-semibold tabular-nums ${
      tone === 'positive' ? 'text-emerald-400' : tone === 'negative' ? 'text-rose-400' : 'text-foreground'
    }`}>
      {value}
    </div>
    <div className="mt-1 truncate text-xs text-muted-text">{detail}</div>
  </div>
);

const EvidenceList = ({
  title,
  items,
  emptyText,
  tone,
}: {
  title: string;
  items?: string[];
  emptyText: string;
  tone: 'pass' | 'watch' | 'risk';
}) => {
  const Icon = tone === 'pass' ? CheckCircle2 : tone === 'watch' ? Eye : ShieldAlert;
  const color = tone === 'pass' ? 'text-emerald-400' : tone === 'watch' ? 'text-amber-300' : 'text-rose-400';
  const visibleItems = items?.length ? items : [emptyText];
  return (
    <div className="min-w-0">
      <div className={`flex items-center gap-1.5 text-xs font-medium ${color}`}>
        <Icon className="h-3.5 w-3.5 shrink-0" />
        {title}
      </div>
      <ul className="mt-2 space-y-1.5 text-xs leading-5 text-secondary-text">
        {visibleItems.map((item) => <li key={item}>{item}</li>)}
      </ul>
    </div>
  );
};

const PaperTradingDashboardPage = () => {
  const [state, setState] = useState<PaperTradingState | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [activePortfolio, setActivePortfolio] = useState('large_cap_22');

  const loadState = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const response = await fetch(`${DATA_URL}?t=${Date.now()}`, { cache: 'no-store' });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      setState(await response.json() as PaperTradingState);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : '未知错误');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    document.title = '美股模拟交易看板 - DSA';
    void loadState();
    const refreshWhenVisible = () => {
      if (document.visibilityState === 'visible') {
        void loadState();
      }
    };
    const intervalId = window.setInterval(refreshWhenVisible, AUTO_REFRESH_MS);
    window.addEventListener('focus', refreshWhenVisible);
    document.addEventListener('visibilitychange', refreshWhenVisible);
    return () => {
      window.clearInterval(intervalId);
      window.removeEventListener('focus', refreshWhenVisible);
      document.removeEventListener('visibilitychange', refreshWhenVisible);
    };
  }, [loadState]);

  const portfolioEntries = useMemo(() => Object.entries(state?.portfolios || {}), [state]);
  const portfolio = state?.portfolios[activePortfolio] || portfolioEntries[0]?.[1];
  const portfolioName = state?.portfolios[activePortfolio] ? activePortfolio : portfolioEntries[0]?.[0] || '';
  const latest = portfolio?.snapshots.at(-1);
  const cycle = portfolio?.active_cycle || null;
  const candidates = cycle?.selected || [];
  const positions = cycle?.positions || [];
  const recentGridEvents = (portfolio?.event_log || []).slice(-6).reverse();
  const diagnostic = state?.live_validation.diagnostics[portfolioName];
  const chartData = useMemo(() => (portfolio?.snapshots || []).map((snapshot) => ({
    date: snapshot.date.slice(5),
    strategy: snapshot.strategy_total_return_pct,
    ...snapshot.benchmark_total_return_pct,
  })), [portfolio]);
  const completionRatio = diagnostic
    ? Math.min(diagnostic.completed_cycles / diagnostic.required_completed_cycles * 100, 100)
    : 0;

  if (loading && !state) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-base text-foreground">
        <div className="flex items-center gap-3 text-sm text-muted-text">
          <RefreshCw className="h-4 w-4 animate-spin text-cyan" />
          正在读取模拟交易账本
        </div>
      </div>
    );
  }

  if (!state || !portfolio || !latest) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-base px-4 text-foreground">
        <div className="w-full max-w-md border border-rose-400/30 bg-card p-6 text-center shadow-soft-card">
          <AlertTriangle className="mx-auto h-7 w-7 text-rose-400" />
          <h1 className="mt-3 text-lg font-semibold">模拟交易数据暂不可用</h1>
          <p className="mt-2 text-sm text-muted-text">{error || '账本中还没有可展示的收益快照。'}</p>
          <button
            type="button"
            className="mt-5 inline-flex h-10 w-10 items-center justify-center border border-border bg-hover text-foreground hover:border-cyan/50"
            onClick={() => void loadState()}
            aria-label="重新加载"
          >
            <RefreshCw className="h-4 w-4" />
          </button>
        </div>
      </div>
    );
  }

  const returnTone = latest.strategy_total_return_pct > 0
    ? 'positive'
    : latest.strategy_total_return_pct < 0 ? 'negative' : 'neutral';

  return (
    <div className="min-h-screen bg-base text-foreground">
      <header className="border-b border-border/70 bg-card/80 backdrop-blur">
        <div className="mx-auto flex min-h-16 max-w-7xl items-center justify-between gap-4 px-4 sm:px-6">
          <div className="flex min-w-0 items-center gap-3">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center border border-cyan/35 bg-cyan/10 text-cyan">
              <Activity className="h-5 w-5" />
            </div>
            <div className="min-w-0">
              <div className="truncate text-sm font-semibold">DSA 美股模拟验证</div>
              <div className="truncate text-xs text-muted-text">{state.strategy}</div>
            </div>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <a
              href={REPOSITORY_URL}
              target="_blank"
              rel="noreferrer"
              className="flex h-9 w-9 items-center justify-center border border-border bg-card text-muted-text hover:border-cyan/40 hover:text-cyan"
              aria-label="打开 GitHub 仓库"
            >
              <Github className="h-4 w-4" />
            </a>
            <button
              type="button"
              className="flex h-9 w-9 items-center justify-center border border-border bg-card text-muted-text hover:border-cyan/40 hover:text-cyan disabled:opacity-50"
              onClick={() => void loadState()}
              disabled={loading}
              aria-label="刷新账本"
            >
              <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6 sm:py-8">
        <section className="flex flex-col justify-between gap-5 border-b border-border/70 pb-6 lg:flex-row lg:items-end">
          <div>
            <div className="flex flex-wrap items-center gap-2 text-xs text-muted-text">
              <span className={`inline-flex items-center gap-1.5 border px-2 py-1 font-medium ${
                state.live_validation.effective
                  ? 'border-emerald-400/35 bg-emerald-400/10 text-emerald-400'
                  : 'border-amber-400/35 bg-amber-400/10 text-amber-300'
              }`}>
                {state.live_validation.effective
                  ? <ShieldCheck className="h-3.5 w-3.5" />
                  : <AlertTriangle className="h-3.5 w-3.5" />}
                {state.live_validation.effective ? '验证通过' : '证据不足或未通过'}
              </span>
              <span className="inline-flex items-center gap-1.5">
                <CalendarDays className="h-3.5 w-3.5" /> 数据日 {state.latest_market_date}
              </span>
              <span className="inline-flex items-center gap-1.5">
                <Clock3 className="h-3.5 w-3.5" /> 北京时间 {formatUpdatedAt(state.updated_at)} 更新
              </span>
            </div>
            <h1 className="mt-4 text-2xl font-semibold sm:text-3xl">模拟组合收益与候选追踪</h1>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-secondary-text">
              仅记录规则驱动的虚拟成交与基准对比，不连接券商，不构成投资建议。策略分用于候选相对排序，不代表上涨概率或预期收益。
            </p>
          </div>
          <a
            href={ACTIONS_URL}
            target="_blank"
            rel="noreferrer"
            className="inline-flex h-10 items-center justify-center gap-2 self-start border border-border bg-card px-3 text-sm font-medium text-foreground hover:border-cyan/40 hover:text-cyan lg:self-auto"
          >
            查看每日运行
            <ExternalLink className="h-4 w-4" />
          </a>
        </section>

        <section className="py-5">
          <div className="inline-flex max-w-full border border-border bg-card p-1" role="tablist" aria-label="股票池">
            {portfolioEntries.map(([name]) => (
              <button
                key={name}
                type="button"
                role="tab"
                aria-selected={portfolioName === name}
                onClick={() => setActivePortfolio(name)}
                className={`min-h-9 px-3 text-sm font-medium transition-colors sm:px-4 ${
                  portfolioName === name
                    ? 'bg-cyan/15 text-cyan'
                    : 'text-muted-text hover:bg-hover hover:text-foreground'
                }`}
              >
                {PORTFOLIO_LABELS[name] || name}
              </button>
            ))}
          </div>
        </section>

        <section className="grid divide-y divide-border border-y border-border bg-card sm:grid-cols-2 sm:divide-x sm:divide-y-0 lg:grid-cols-4">
          <Metric label="组合净值" value={formatMoney(latest.strategy_equity)} detail={`初始 ${formatMoney(state.config.initial_capital)}`} />
          <Metric label="累计收益" value={formatPercent(latest.strategy_total_return_pct)} detail={`当日 ${formatPercent(latest.strategy_daily_return_pct)}`} tone={returnTone} />
          <Metric label="最大回撤" value={formatPercent(latest.strategy_drawdown_pct)} detail="按每日组合净值计算" tone={latest.strategy_drawdown_pct < 0 ? 'negative' : 'neutral'} />
          <Metric label="当前周期" value={cycleLabel(cycle?.status)} detail={`${portfolio.closed_cycles.length} 个周期已完成`} />
        </section>

        <section className="grid border-b border-border lg:grid-cols-[1.7fr_1fr] lg:divide-x lg:divide-border">
          <div className="min-w-0 py-7 lg:pr-7">
            <div className="flex flex-wrap items-end justify-between gap-3">
              <div>
                <div className="flex items-center gap-2">
                  <Grid2X2 className="h-4 w-4 text-cyan" />
                  <h2 className="text-[16px] font-semibold text-foreground">网格止损止盈</h2>
                </div>
                <p className="mt-1 text-xs text-muted-text">
                  每格 {(state.config.grid_step_pct || 3).toFixed(1)}% · 上涨 {state.config.grid_take_profit_levels || 2} 格分批止盈 · 下跌 {state.config.grid_stop_loss_levels || 2} 格止损
                </p>
              </div>
              <span className="text-xs text-muted-text">最长持有 {state.config.holding_days} 个交易日</span>
            </div>
            <div className="mt-4 overflow-x-auto">
              <table className="w-full min-w-[44rem] text-left text-sm">
                <thead className="border-y border-border/70 text-xs text-muted-text">
                  <tr>
                    <th className="py-2 font-medium">代码</th>
                    <th className="py-2 text-right font-medium">入场价</th>
                    <th className="py-2 text-right font-medium">止损线</th>
                    <th className="py-2 text-right font-medium">下一止盈格</th>
                    <th className="py-2 text-right font-medium">剩余仓位</th>
                    <th className="py-2 text-right font-medium">最新记录价</th>
                    <th className="py-2 text-right font-medium">状态</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/50">
                  {positions.map((position) => {
                    const completed = new Set(position.grid.completed_take_profit_levels || []);
                    const nextTarget = position.grid.take_profit_prices.find((_, index) => !completed.has(index + 1));
                    const remainingRatio = position.quantity > 0 ? position.remaining_quantity / position.quantity * 100 : 0;
                    return (
                      <tr key={position.code}>
                        <td className="py-3 font-semibold">{position.code}</td>
                        <td className="py-3 text-right tabular-nums">{formatMoney(position.entry_open)}</td>
                        <td className="py-3 text-right tabular-nums text-rose-400">{formatMoney(position.grid.stop_loss_price)}</td>
                        <td className="py-3 text-right tabular-nums text-emerald-400">
                          {nextTarget == null ? '已完成' : formatMoney(nextTarget)}
                        </td>
                        <td className="py-3 text-right tabular-nums">{remainingRatio.toFixed(0)}%</td>
                        <td className="py-3 text-right tabular-nums">{formatMoney(position.last_price || position.entry_open)}</td>
                        <td className="py-3 text-right text-xs text-secondary-text">{position.status === 'closed' ? '已退出' : '监控中'}</td>
                      </tr>
                    );
                  })}
                  {!positions.length ? (
                    <tr><td className="py-5 text-muted-text" colSpan={7}>下一交易日开盘后生成持仓网格价格</td></tr>
                  ) : null}
                </tbody>
              </table>
            </div>
          </div>

          <div className="min-w-0 py-7 lg:pl-7">
            <h2 className="text-[16px] font-semibold text-foreground">最近网格成交</h2>
            <p className="mt-1 text-xs text-muted-text">触发价与模拟成交价分开记录</p>
            <div className="mt-4 divide-y divide-border/60">
              {recentGridEvents.map((event) => (
                <div key={`${event.code}-${event.date}-${event.reason}-${event.grid_level || 0}`} className="py-3 first:pt-0">
                  <div className="flex items-center justify-between gap-3 text-sm">
                    <span className="font-medium">{event.code} · {exitReasonLabel(event.type || event.reason)}</span>
                    <span className="text-xs tabular-nums text-muted-text">{event.date}</span>
                  </div>
                  <div className="mt-1 text-xs leading-5 text-secondary-text">
                    触发 {formatMoney(event.trigger_price)} · 成交 {formatMoney(event.fill_price)} · {event.quantity.toFixed(4)} 股
                  </div>
                  <div className="mt-1 truncate text-xs text-muted-text">{event.source}</div>
                </div>
              ))}
              {!recentGridEvents.length ? <p className="py-4 text-sm text-muted-text">当前还没有网格成交记录</p> : null}
            </div>
          </div>
        </section>

        <section className="border-b border-border py-7">
          <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
            <div>
              <h2 className="text-[16px] font-semibold text-foreground">累计收益轨迹</h2>
              <p className="mt-1 text-xs text-muted-text">策略与五个基准，单位 %</p>
            </div>
            <div className="flex flex-wrap gap-x-4 gap-y-2 text-xs text-muted-text">
              {['strategy', ...state.benchmarks].map((key) => (
                <span key={key} className="inline-flex items-center gap-1.5">
                  <span className="h-2 w-2" style={{ backgroundColor: CHART_COLORS[key] }} />
                  {key === 'strategy' ? '策略' : key}
                </span>
              ))}
            </div>
          </div>
          <div className="h-64 w-full sm:h-72">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
                <CartesianGrid stroke="rgba(148, 163, 184, 0.12)" vertical={false} />
                <XAxis dataKey="date" stroke="rgba(148, 163, 184, 0.55)" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} />
                <YAxis
                  domain={chartData.length < 2 ? [-1, 1] : ['auto', 'auto']}
                  stroke="rgba(148, 163, 184, 0.55)"
                  tick={{ fontSize: 11 }}
                  tickFormatter={(value) => `${value}%`}
                  tickLine={false}
                  axisLine={false}
                />
                <Tooltip
                  contentStyle={{ background: '#15181d', border: '1px solid rgba(148,163,184,.28)', borderRadius: 4 }}
                  labelStyle={{ color: '#e5e7eb' }}
                  formatter={(value) => [`${Number(value).toFixed(2)}%`]}
                />
                {['strategy', ...state.benchmarks].map((key) => (
                  <Line
                    key={key}
                    type="monotone"
                    dataKey={key}
                    name={key === 'strategy' ? '策略' : key}
                    stroke={CHART_COLORS[key]}
                    strokeWidth={key === 'strategy' ? 2.5 : 1.5}
                    dot={chartData.length < 3}
                    activeDot={{ r: 4 }}
                    connectNulls
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="grid border-b border-border lg:grid-cols-2 lg:divide-x lg:divide-border">
          <div className="min-w-0 py-7 lg:pr-7">
            <div className="mb-4 flex items-end justify-between gap-3">
              <div>
                <h2 className="text-[16px] font-semibold text-foreground">本周期候选</h2>
                <p className="mt-1 text-xs text-muted-text">信号日 {cycle?.signal_date || 'N/A'} · {cycleLabel(cycle?.status)}</p>
              </div>
              <span className="text-xs text-muted-text">Top {state.config.top_k}</span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[46rem] text-left text-sm">
                <thead className="border-y border-border/70 text-xs text-muted-text">
                  <tr>
                    <th className="w-12 py-2 font-medium">#</th>
                    <th className="py-2 font-medium">代码</th>
                    <th className="py-2 font-medium">策略分</th>
                    <th className="py-2 font-medium">核心理由</th>
                    <th className="py-2 text-right font-medium">信号收盘价</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/50">
                  {candidates.map((candidate, index) => (
                    <tr key={candidate.code}>
                      <td className="py-3 text-muted-text">{index + 1}</td>
                      <td className="py-3 font-semibold">{candidate.code}</td>
                      <td className="py-3">
                        <div className="flex items-center gap-3">
                          <span className="w-10 tabular-nums">{candidate.screen_score.toFixed(1)}</span>
                          <span className="h-1.5 w-24 overflow-hidden bg-hover">
                            <span className="block h-full bg-cyan" style={{ width: `${Math.min(candidate.screen_score, 100)}%` }} />
                          </span>
                        </div>
                      </td>
                      <td className="max-w-[20rem] py-3 pr-4 text-xs leading-5 text-secondary-text">
                        {candidate.selection_thesis || '旧周期暂无结构化理由'}
                      </td>
                      <td className="py-3 text-right tabular-nums">{formatMoney(candidate.signal_close)}</td>
                    </tr>
                  ))}
                  {!candidates.length ? (
                    <tr><td className="py-5 text-muted-text" colSpan={5}>本周期保持现金</td></tr>
                  ) : null}
                </tbody>
              </table>
            </div>
          </div>

          <div className="min-w-0 py-7 lg:pl-7">
            <div className="mb-4">
              <h2 className="text-[16px] font-semibold text-foreground">多基准比较</h2>
              <p className="mt-1 text-xs text-muted-text">策略累计收益相对同周期基准</p>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[28rem] text-left text-sm">
                <thead className="border-y border-border/70 text-xs text-muted-text">
                  <tr>
                    <th className="py-2 font-medium">基准</th>
                    <th className="py-2 text-right font-medium">基准收益</th>
                    <th className="py-2 text-right font-medium">策略超额</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/50">
                  {state.benchmarks.map((benchmark) => {
                    const benchmarkReturn = latest.benchmark_total_return_pct[benchmark] || 0;
                    const excess = latest.strategy_total_return_pct - benchmarkReturn;
                    return (
                      <tr key={benchmark}>
                        <td className="py-3 font-medium">{benchmark}</td>
                        <td className="py-3 text-right tabular-nums">{formatPercent(benchmarkReturn)}</td>
                        <td className={`py-3 text-right tabular-nums ${excess > 0 ? 'text-emerald-400' : excess < 0 ? 'text-rose-400' : ''}`}>
                          {`${excess >= 0 ? '+' : ''}${excess.toFixed(2)}pp`}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </section>

        <section className="border-b border-border py-7">
          <div className="flex flex-wrap items-end justify-between gap-3">
            <div>
              <h2 className="text-[16px] font-semibold text-foreground">候选证据卡</h2>
              <p className="mt-1 text-xs text-muted-text">规则证据、风险与数据缺口随交易账本保存</p>
            </div>
            <span className="text-xs text-muted-text">{state.scorecard_version || candidates[0]?.scorecard_version || '旧版账本'}</span>
          </div>
          <div className="mt-3 divide-y divide-border/70">
            {candidates.map((candidate, index) => {
              const factors = FACTOR_ORDER
                .filter((key) => candidate.factor_scores?.[key] != null)
                .map((key) => ({
                  key,
                  score: Number(candidate.factor_scores?.[key]),
                  weight: Number(candidate.factor_weights?.[key] || 0),
                }));
              return (
                <article key={candidate.code} className="grid gap-6 py-6 lg:grid-cols-[13rem_1fr_1.15fr]">
                  <div className="min-w-0">
                    <div className="flex items-baseline gap-3">
                      <span className="text-xs text-muted-text">#{index + 1}</span>
                      <h3 className="text-xl font-semibold">{candidate.code}</h3>
                      <span className="text-sm tabular-nums text-cyan">{candidate.screen_score.toFixed(1)}</span>
                    </div>
                    <p className="mt-3 text-sm leading-6 text-secondary-text">
                      {candidate.selection_thesis || '该候选来自旧版账本，尚无结构化证据。'}
                    </p>
                    <p className="mt-3 text-xs leading-5 text-muted-text">
                      {candidate.score_explanation || '规则排序分，不代表上涨概率或预期收益。'}
                    </p>
                  </div>

                  <div className="min-w-0">
                    <div className="text-xs font-medium text-muted-text">六维因子</div>
                    <div className="mt-3 grid gap-x-5 gap-y-3 sm:grid-cols-2">
                      {factors.map(({ key, score, weight }) => (
                        <div key={key} className="min-w-0">
                          <div className="flex items-center justify-between gap-3 text-xs">
                            <span className="truncate text-secondary-text">{FACTOR_LABELS[key] || key}</span>
                            <span className="shrink-0 tabular-nums">{score.toFixed(1)} · {(weight * 100).toFixed(0)}%</span>
                          </div>
                          <div className="mt-1.5 h-1.5 overflow-hidden bg-hover">
                            <div className="h-full bg-cyan" style={{ width: `${Math.max(0, Math.min(score, 100))}%` }} />
                          </div>
                        </div>
                      ))}
                      {!factors.length ? <p className="text-xs text-muted-text">旧周期未保存因子明细</p> : null}
                    </div>
                    <div className="mt-4 border-t border-border/60 pt-3 text-xs leading-5 text-muted-text">
                      <div>数据：{candidate.data_sources?.join('；') || '未记录'}</div>
                      {candidate.data_gaps?.length ? <div className="mt-1 text-amber-300">缺口：{candidate.data_gaps.join('；')}</div> : null}
                    </div>
                  </div>

                  <div className="grid min-w-0 gap-5 sm:grid-cols-3 lg:grid-cols-1 xl:grid-cols-3">
                    <EvidenceList title="入选证据" items={candidate.reasons_pass} emptyText="未记录" tone="pass" />
                    <EvidenceList title="观察项" items={candidate.reasons_watch} emptyText="暂无额外观察项" tone="watch" />
                    <EvidenceList title="风险提示" items={candidate.risk_flags} emptyText="未触发额外技术风险标记" tone="risk" />
                    <div className="sm:col-span-3 lg:col-span-1 xl:col-span-3">
                      <div className="text-xs font-medium text-muted-text">失效条件</div>
                      <p className="mt-1.5 text-xs leading-5 text-secondary-text">
                        {candidate.invalidation_conditions?.join('；') || '按统一硬风控执行'}
                      </p>
                    </div>
                  </div>
                </article>
              );
            })}
            {!candidates.length ? <p className="py-6 text-sm text-muted-text">本周期没有候选证据，组合保持现金。</p> : null}
          </div>
        </section>

        <section className="grid gap-0 border-b border-border lg:grid-cols-[1.4fr_1fr] lg:divide-x lg:divide-border">
          <div className="py-7 lg:pr-7">
            <div className="flex items-center gap-2">
              {diagnostic?.effective ? <ShieldCheck className="h-5 w-5 text-emerald-400" /> : <Database className="h-5 w-5 text-amber-300" />}
              <h2 className="text-[16px] font-semibold text-foreground">有效性证据进度</h2>
            </div>
            <div className="mt-5 grid gap-5 sm:grid-cols-2">
              <div>
                <div className="flex justify-between text-xs">
                  <span className="text-muted-text">完成周期</span>
                  <span className="tabular-nums">{diagnostic?.completed_cycles || 0} / {diagnostic?.required_completed_cycles || state.config.minimum_completed_cycles}</span>
                </div>
                <div className="mt-2 h-2 overflow-hidden bg-hover">
                  <div className="h-full bg-amber-400" style={{ width: `${completionRatio}%` }} />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-xs">
                  <span className="text-muted-text">跑赢基准</span>
                  <span className="tabular-nums">{diagnostic?.positive_excess_benchmarks || 0} / {diagnostic?.required_positive_excess_benchmarks || 3}</span>
                </div>
                <div className="mt-2 h-2 overflow-hidden bg-hover">
                  <div
                    className="h-full bg-cyan"
                    style={{ width: `${Math.min((diagnostic?.positive_excess_benchmarks || 0) / (diagnostic?.required_positive_excess_benchmarks || 3) * 100, 100)}%` }}
                  />
                </div>
              </div>
            </div>
            <p className="mt-4 text-sm leading-6 text-secondary-text">
              每个正式股票池需要至少 {state.config.minimum_completed_cycles} 个周期、组合盈利且跑赢至少 3/5 基准。历史综合验证仍独立生效。
            </p>
          </div>

          <div className="grid grid-cols-2 gap-x-5 gap-y-6 py-7 lg:pl-7">
            {[
              ['初始资金', formatMoney(state.config.initial_capital)],
              ['持有周期', `${state.config.holding_days} 个交易日`],
              ['网格步长', `${(state.config.grid_step_pct || 3).toFixed(1)}%`],
              ['单边成本', `${state.config.per_side_cost_bps} bps`],
              ['行情覆盖', `${((cycle?.coverage?.ratio || 0) * 100).toFixed(0)}%`],
            ].map(([label, value]) => (
              <div key={label} className="border-l-2 border-border px-3 py-1">
                <div className="text-xs text-muted-text">{label}</div>
                <div className="mt-1 text-sm font-medium tabular-nums">{value}</div>
              </div>
            ))}
          </div>
        </section>

        <footer className="flex flex-col gap-2 py-5 text-xs text-muted-text sm:flex-row sm:items-center sm:justify-between">
          <span className="inline-flex items-center gap-1.5">
            {latest.strategy_total_return_pct >= 0
              ? <TrendingUp className="h-3.5 w-3.5" />
              : <TrendingDown className="h-3.5 w-3.5" />}
            数据来自可审计的 paper-trading-state 分支
          </span>
          <span>虚拟成交 · 无真实订单 · 非投资建议</span>
        </footer>
      </main>
    </div>
  );
};

export default PaperTradingDashboardPage;
