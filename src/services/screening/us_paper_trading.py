"""Deterministic paper-trading ledger for the US screening strategies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.services.screening.filter import apply_hard_filters
from src.services.screening.scorer import compute_screen_scores
from src.services.screening.strategy import load_all_strategies
from src.services.screening.us_candidate_evidence import FACTOR_LABELS, build_us_candidate_evidence
from src.services.screening.us_backtest import (
    _backtest_filters,
    _feature_row,
    normalize_price_history,
)


@dataclass(frozen=True)
class USPaperTradingConfig:
    strategy_name: str = "us_quality_momentum"
    top_k: int = 5
    holding_days: int = 10
    lookback_days: int = 140
    initial_capital: float = 100_000.0
    per_side_cost_bps: float = 10.0
    minimum_universe_coverage: float = 0.95
    minimum_completed_cycles: int = 20


def create_paper_trading_state(
    universes: Mapping[str, Sequence[str]],
    *,
    benchmarks: Sequence[str],
    config: USPaperTradingConfig | None = None,
) -> dict[str, object]:
    """Create a JSON-serializable multi-universe ledger."""
    config = config or USPaperTradingConfig()
    if config.top_k <= 0 or config.holding_days <= 0 or config.lookback_days < 60:
        raise ValueError("invalid paper-trading configuration")
    if not 0.0 <= config.minimum_universe_coverage <= 1.0:
        raise ValueError("minimum_universe_coverage must be between 0 and 1")
    if not benchmarks:
        raise ValueError("at least one benchmark is required")
    return {
        "schema_version": 1,
        "strategy": config.strategy_name,
        "scorecard_version": None,
        "research_status": "not_validated",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": None,
        "config": asdict(config),
        "benchmarks": list(dict.fromkeys(str(item) for item in benchmarks)),
        "portfolios": {
            name: {
                "universe": list(dict.fromkeys(str(item) for item in tickers)),
                "realized_equity": config.initial_capital,
                "benchmark_realized_equity": {
                    benchmark: config.initial_capital for benchmark in benchmarks
                },
                "active_cycle": None,
                "closed_cycles": [],
                "snapshots": [],
                "last_processed_date": None,
            }
            for name, tickers in universes.items()
        },
    }


def rank_candidates_on_date(
    strategy_name: str,
    histories: Mapping[str, pd.DataFrame],
    signal_date: date,
    *,
    top_k: int,
    lookback_days: int,
    minimum_universe_coverage: float,
    strategies_dir: Path | None = None,
) -> dict[str, object]:
    """Rank one point-in-time cross section without reading future bars."""
    strategies_dir = strategies_dir or Path(__file__).with_name("strategies")
    strategy = load_all_strategies(strategies_dir)[strategy_name]
    if strategy.screening.market_scope != ["us"]:
        raise ValueError(f"{strategy_name} is not a US strategy")
    normalized = {code: normalize_price_history(frame) for code, frame in histories.items()}
    feature_rows: list[dict[str, object]] = []
    for code, frame in normalized.items():
        matches = frame.index[frame["date"] == signal_date].tolist()
        if not matches:
            continue
        row = _feature_row(code, frame, int(matches[-1]), lookback_days)
        if row is not None:
            feature_rows.append(row)
    expected_count = len(normalized)
    covered_count = len(feature_rows)
    coverage_ratio = covered_count / expected_count if expected_count else 0.0
    if coverage_ratio < minimum_universe_coverage:
        raise ValueError(
            f"paper-trading universe coverage {coverage_ratio:.2%} below required "
            f"{minimum_universe_coverage:.2%} on {signal_date} "
            f"({covered_count}/{expected_count})"
        )
    if not feature_rows:
        selected = pd.DataFrame()
    else:
        candidates = apply_hard_filters(
            pd.DataFrame(feature_rows),
            _backtest_filters(strategy.screening.hard_filters),
        )
        selected = compute_screen_scores(candidates, strategy.screening)
        selected = selected.sort_values(
            ["screen_score", "code"], ascending=[False, True]
        ).head(top_k)
    selected_payload = []
    for _, row in selected.iterrows():
        payload = {
            "code": str(row["code"]),
            "screen_score": round(float(row["screen_score"]), 4),
            "signal_close": round(float(row["price"]), 6),
        }
        payload.update(build_us_candidate_evidence(row, strategy.screening))
        selected_payload.append(payload)
    return {
        "signal_date": signal_date.isoformat(),
        "universe_expected_count": expected_count,
        "universe_feature_covered_count": covered_count,
        "universe_coverage_ratio": round(coverage_ratio, 6),
        "selected": selected_payload,
    }


def advance_paper_trading_state(
    state: dict[str, object],
    histories: Mapping[str, pd.DataFrame],
    *,
    as_of: date | None = None,
    strategies_dir: Path | None = None,
) -> dict[str, object]:
    """Advance every portfolio through the latest completed market session."""
    config = USPaperTradingConfig(**dict(state["config"]))
    benchmarks = [str(item) for item in state["benchmarks"]]
    normalized = {code: normalize_price_history(frame) for code, frame in histories.items()}
    missing_benchmarks = [code for code in benchmarks if code not in normalized or normalized[code].empty]
    if missing_benchmarks:
        raise ValueError(f"missing benchmark histories: {', '.join(missing_benchmarks)}")
    common_dates = set(normalized[benchmarks[0]]["date"])
    for benchmark in benchmarks[1:]:
        common_dates &= set(normalized[benchmark]["date"])
    eligible_dates = sorted(value for value in common_dates if as_of is None or value <= as_of)
    if not eligible_dates:
        raise ValueError("no common benchmark date is available")
    effective_date = eligible_dates[-1]
    events: dict[str, list[dict[str, object]]] = {}
    strategies_dir = strategies_dir or Path(__file__).with_name("strategies")
    strategy = load_all_strategies(strategies_dir)[config.strategy_name]
    state["scorecard_version"] = str(
        strategy.screening.scorecard_profile.get("version") or "us_evidence_v2"
    )
    for name, portfolio in dict(state["portfolios"]).items():
        migration_event = _ensure_active_cycle_evidence(
            portfolio,
            normalized,
            config,
            strategies_dir=strategies_dir,
        )
        events[name] = _advance_portfolio(
            portfolio,
            normalized,
            benchmarks,
            effective_date,
            config,
            strategies_dir=strategies_dir,
        )
        if migration_event is not None:
            events[name].insert(0, migration_event)
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    state["latest_market_date"] = effective_date.isoformat()
    state["live_validation"] = evaluate_live_validation(state)
    state["last_events"] = events
    return state


def _ensure_active_cycle_evidence(
    portfolio: dict[str, object],
    histories: Mapping[str, pd.DataFrame],
    config: USPaperTradingConfig,
    *,
    strategies_dir: Path,
) -> dict[str, object] | None:
    """Upgrade pre-evidence cycles without rewriting already executed trades."""
    cycle = portfolio.get("active_cycle")
    if not isinstance(cycle, dict):
        return None
    selected = list(cycle.get("selected", []))
    if not selected or all(item.get("selection_thesis") for item in selected):
        return None
    signal_date_text = str(cycle.get("signal_date") or "")
    try:
        signal_date = date.fromisoformat(signal_date_text)
    except ValueError:
        return None
    universe = [str(item) for item in portfolio.get("universe", [])]
    requested_top_k = (
        config.top_k
        if cycle.get("status") == "pending"
        else max(len(universe), 1)
    )
    ranked = rank_candidates_on_date(
        config.strategy_name,
        {code: histories.get(code, pd.DataFrame()) for code in universe},
        signal_date,
        top_k=requested_top_k,
        lookback_days=config.lookback_days,
        minimum_universe_coverage=config.minimum_universe_coverage,
        strategies_dir=strategies_dir,
    )
    rebuilt = list(ranked["selected"])
    if cycle.get("status") == "pending":
        cycle["selected"] = rebuilt
        return {
            "type": "scorecard_upgraded",
            "signal_date": signal_date_text,
            "mode": "pending_signal_recomputed",
        }

    evidence_by_code = {str(item["code"]): item for item in rebuilt}
    migrated = 0
    for item in selected:
        evidence = evidence_by_code.get(str(item.get("code") or ""))
        if evidence is None:
            continue
        original_score = item.get("screen_score")
        for key, value in evidence.items():
            if key not in {"code", "screen_score", "signal_close"}:
                item[key] = value
        item["scorecard_version"] = "legacy_v1_evidence_backfill"
        item["score_explanation"] = (
            "保留信号生成时的旧版排序分；证据卡按同一信号日重建，不改写已执行交易。"
        )
        item["screen_score"] = original_score
        migrated += 1
    if migrated:
        return {
            "type": "scorecard_upgraded",
            "signal_date": signal_date_text,
            "mode": "open_cycle_evidence_backfilled",
            "candidate_count": migrated,
        }
    return None


def _advance_portfolio(
    portfolio: dict[str, object],
    histories: Mapping[str, pd.DataFrame],
    benchmarks: Sequence[str],
    as_of: date,
    config: USPaperTradingConfig,
    *,
    strategies_dir: Path | None,
) -> list[dict[str, object]]:
    if portfolio.get("last_processed_date") == as_of.isoformat():
        return [{"type": "no_change", "market_date": as_of.isoformat()}]
    benchmark_calendar = histories[benchmarks[0]]
    events: list[dict[str, object]] = []
    cycle = portfolio.get("active_cycle")
    if isinstance(cycle, dict) and cycle.get("status") == "pending":
        entry_dates = [value for value in benchmark_calendar["date"] if value > date.fromisoformat(cycle["signal_date"]) and value <= as_of]
        if entry_dates:
            entry_date = entry_dates[0]
            _open_cycle(cycle, portfolio, histories, benchmarks, entry_date, config)
            events.append({"type": "opened", "entry_date": entry_date.isoformat(), "codes": [item["code"] for item in cycle["positions"]]})
    cycle = portfolio.get("active_cycle")
    if isinstance(cycle, dict) and cycle.get("status") == "open":
        holding_dates = [value for value in benchmark_calendar["date"] if value >= date.fromisoformat(cycle["entry_date"]) and value <= as_of]
        if len(holding_dates) >= config.holding_days:
            exit_date = holding_dates[config.holding_days - 1]
            closed = _close_cycle(cycle, portfolio, histories, benchmarks, exit_date, config)
            events.append({"type": "closed", "exit_date": exit_date.isoformat(), "strategy_return_pct": closed["strategy_return_pct"]})
            portfolio["active_cycle"] = None
    if portfolio.get("active_cycle") is None:
        universe = [str(item) for item in portfolio["universe"]]
        ranked = rank_candidates_on_date(
            config.strategy_name,
            {code: histories.get(code, pd.DataFrame()) for code in universe},
            as_of,
            top_k=config.top_k,
            lookback_days=config.lookback_days,
            minimum_universe_coverage=config.minimum_universe_coverage,
            strategies_dir=strategies_dir,
        )
        portfolio["active_cycle"] = {
            "status": "pending",
            "signal_date": as_of.isoformat(),
            "selected": ranked["selected"],
            "coverage": {
                "expected_count": ranked["universe_expected_count"],
                "covered_count": ranked["universe_feature_covered_count"],
                "ratio": ranked["universe_coverage_ratio"],
            },
        }
        events.append({"type": "signal", "signal_date": as_of.isoformat(), "codes": [item["code"] for item in ranked["selected"]]})
    snapshot = _portfolio_snapshot(portfolio, histories, benchmarks, as_of, config)
    snapshots = list(portfolio.get("snapshots", []))
    if snapshots and snapshots[-1].get("date") == as_of.isoformat():
        snapshots[-1] = snapshot
    else:
        snapshots.append(snapshot)
    portfolio["snapshots"] = snapshots
    portfolio["last_processed_date"] = as_of.isoformat()
    return events


def _open_cycle(
    cycle: dict[str, object],
    portfolio: dict[str, object],
    histories: Mapping[str, pd.DataFrame],
    benchmarks: Sequence[str],
    entry_date: date,
    config: USPaperTradingConfig,
) -> None:
    selected = list(cycle.get("selected", []))
    positions: list[dict[str, object]] = []
    starting_equity = float(portfolio["realized_equity"])
    allocation = starting_equity / len(selected) if selected else 0.0
    for item in selected:
        code = str(item["code"])
        bar = _bar_on_date(histories.get(code), entry_date)
        if bar is None:
            raise ValueError(f"missing entry bar for {code} on {entry_date}")
        entry_open = float(bar["open"])
        positions.append({
            **item,
            "entry_date": entry_date.isoformat(),
            "entry_open": round(entry_open, 6),
            "allocated_capital": round(allocation, 6),
            "quantity": round(allocation / entry_open, 8),
        })
    benchmark_entries: dict[str, float] = {}
    for benchmark in benchmarks:
        bar = _bar_on_date(histories[benchmark], entry_date)
        if bar is None:
            raise ValueError(f"missing benchmark entry bar for {benchmark} on {entry_date}")
        benchmark_entries[benchmark] = round(float(bar["open"]), 6)
    cycle.update({
        "status": "open",
        "entry_date": entry_date.isoformat(),
        "starting_equity": starting_equity,
        "benchmark_starting_equity": dict(portfolio["benchmark_realized_equity"]),
        "benchmark_entries": benchmark_entries,
        "positions": positions,
    })


def _close_cycle(
    cycle: dict[str, object],
    portfolio: dict[str, object],
    histories: Mapping[str, pd.DataFrame],
    benchmarks: Sequence[str],
    exit_date: date,
    config: USPaperTradingConfig,
) -> dict[str, object]:
    trade_returns: list[float] = []
    closed_trades: list[dict[str, object]] = []
    for position in cycle.get("positions", []):
        code = str(position["code"])
        bar = _bar_on_date(histories.get(code), exit_date)
        if bar is None:
            raise ValueError(f"missing exit bar for {code} on {exit_date}")
        exit_close = float(bar["close"])
        net_return = _net_return(exit_close / float(position["entry_open"]) - 1.0, config.per_side_cost_bps)
        trade_returns.append(net_return)
        closed_trades.append({
            **position,
            "exit_date": exit_date.isoformat(),
            "exit_close": round(exit_close, 6),
            "net_return_pct": round(net_return * 100.0, 6),
            "pnl": round(float(position["allocated_capital"]) * net_return, 6),
        })
    strategy_return = float(np.mean(trade_returns)) if trade_returns else 0.0
    portfolio["realized_equity"] = round(float(cycle["starting_equity"]) * (1.0 + strategy_return), 6)
    benchmark_returns: dict[str, float] = {}
    for benchmark in benchmarks:
        bar = _bar_on_date(histories[benchmark], exit_date)
        if bar is None:
            raise ValueError(f"missing benchmark exit bar for {benchmark} on {exit_date}")
        benchmark_return = _net_return(
            float(bar["close"]) / float(cycle["benchmark_entries"][benchmark]) - 1.0,
            config.per_side_cost_bps,
        )
        benchmark_returns[benchmark] = benchmark_return
        portfolio["benchmark_realized_equity"][benchmark] = round(
            float(cycle["benchmark_starting_equity"][benchmark]) * (1.0 + benchmark_return), 6
        )
    closed = {
        "signal_date": cycle["signal_date"],
        "entry_date": cycle["entry_date"],
        "exit_date": exit_date.isoformat(),
        "strategy_return_pct": round(strategy_return * 100.0, 6),
        "benchmark_return_pct": {
            key: round(value * 100.0, 6) for key, value in benchmark_returns.items()
        },
        "trades": closed_trades,
        "coverage": cycle.get("coverage", {}),
    }
    portfolio.setdefault("closed_cycles", []).append(closed)
    return closed


def _portfolio_snapshot(
    portfolio: dict[str, object],
    histories: Mapping[str, pd.DataFrame],
    benchmarks: Sequence[str],
    as_of: date,
    config: USPaperTradingConfig,
) -> dict[str, object]:
    cycle = portfolio.get("active_cycle")
    strategy_equity = float(portfolio["realized_equity"])
    benchmark_equity = dict(portfolio["benchmark_realized_equity"])
    if isinstance(cycle, dict) and cycle.get("status") == "open":
        returns = []
        for position in cycle.get("positions", []):
            bar = _bar_on_or_before(histories.get(str(position["code"])), as_of)
            if bar is not None:
                returns.append(_net_return(float(bar["close"]) / float(position["entry_open"]) - 1.0, config.per_side_cost_bps))
        if returns:
            strategy_equity = float(cycle["starting_equity"]) * (1.0 + float(np.mean(returns)))
        for benchmark in benchmarks:
            bar = _bar_on_or_before(histories[benchmark], as_of)
            if bar is not None:
                value = _net_return(float(bar["close"]) / float(cycle["benchmark_entries"][benchmark]) - 1.0, config.per_side_cost_bps)
                benchmark_equity[benchmark] = float(cycle["benchmark_starting_equity"][benchmark]) * (1.0 + value)
    previous = list(portfolio.get("snapshots", []))
    previous_equity = float(previous[-1]["strategy_equity"]) if previous else config.initial_capital
    historical_equity = [float(item["strategy_equity"]) for item in previous] + [strategy_equity]
    peak = max([config.initial_capital, *historical_equity])
    return {
        "date": as_of.isoformat(),
        "strategy_equity": round(strategy_equity, 6),
        "strategy_total_return_pct": round((strategy_equity / config.initial_capital - 1.0) * 100.0, 6),
        "strategy_daily_return_pct": round((strategy_equity / previous_equity - 1.0) * 100.0, 6),
        "strategy_drawdown_pct": round((strategy_equity / peak - 1.0) * 100.0, 6),
        "benchmark_equity": {key: round(float(value), 6) for key, value in benchmark_equity.items()},
        "benchmark_total_return_pct": {
            key: round((float(value) / config.initial_capital - 1.0) * 100.0, 6)
            for key, value in benchmark_equity.items()
        },
    }


def evaluate_live_validation(state: Mapping[str, object]) -> dict[str, object]:
    """Require enough completed cycles and multi-benchmark success in every universe."""
    config = USPaperTradingConfig(**dict(state["config"]))
    benchmarks = [str(item) for item in state["benchmarks"]]
    diagnostics: dict[str, object] = {}
    checks: list[bool] = []
    for name, portfolio in dict(state["portfolios"]).items():
        snapshots = list(portfolio.get("snapshots", []))
        latest = snapshots[-1] if snapshots else {}
        strategy_return = float(latest.get("strategy_total_return_pct") or 0.0)
        benchmark_returns = dict(latest.get("benchmark_total_return_pct") or {})
        positive_excess = sum(
            strategy_return > 0.0 and strategy_return > float(benchmark_returns.get(item, 0.0))
            for item in benchmarks
        )
        completed = len(portfolio.get("closed_cycles", []))
        universe_pass = completed >= config.minimum_completed_cycles and positive_excess >= 3
        checks.append(universe_pass)
        diagnostics[name] = {
            "completed_cycles": completed,
            "required_completed_cycles": config.minimum_completed_cycles,
            "positive_excess_benchmarks": positive_excess,
            "required_positive_excess_benchmarks": 3,
            "effective": universe_pass,
        }
    return {
        "effective": bool(checks) and all(checks),
        "status": "validated" if checks and all(checks) else "insufficient_or_failed",
        "diagnostics": diagnostics,
        "warning": "Paper trading is observational evidence and does not authorize live orders.",
    }


def render_paper_trading_report(state: Mapping[str, object]) -> str:
    """Render a compact Markdown report suitable for notifications and GitHub issues."""
    latest_date = str(state.get("latest_market_date") or "N/A")
    strategy = str(state.get("strategy") or "N/A")
    gate = dict(state.get("live_validation") or {})
    lines = [
        f"# 美股模拟交易日报 - {latest_date}",
        "",
        f"> 策略：`{strategy}`。仅用于研究验证，不构成投资建议，不连接券商或真实下单。",
        f"> 实时验证状态：**{'通过' if gate.get('effective') else '证据不足或未通过'}**。",
        "",
        "## 组合概览",
        "",
        "| 股票池 | 组合净值 | 累计收益 | 当日收益 | 最大回撤 | 完成周期 | 当前状态 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, portfolio in dict(state["portfolios"]).items():
        snapshots = list(portfolio.get("snapshots", []))
        latest = snapshots[-1] if snapshots else {}
        active = portfolio.get("active_cycle") or {}
        lines.append(
            f"| `{name}` | {float(latest.get('strategy_equity') or 0.0):,.2f} | "
            f"{float(latest.get('strategy_total_return_pct') or 0.0):+.2f}% | "
            f"{float(latest.get('strategy_daily_return_pct') or 0.0):+.2f}% | "
            f"{float(latest.get('strategy_drawdown_pct') or 0.0):.2f}% | "
            f"{len(portfolio.get('closed_cycles', []))} | {active.get('status', 'idle')} |"
        )
    lines.extend(["", "## 多基准比较", ""])
    for name, portfolio in dict(state["portfolios"]).items():
        latest = list(portfolio.get("snapshots", []))[-1]
        strategy_return = float(latest.get("strategy_total_return_pct") or 0.0)
        lines.extend([
            f"### {name}",
            "",
            "| 基准 | 基准累计收益 | 策略超额 |",
            "| --- | ---: | ---: |",
        ])
        for benchmark, value in dict(latest.get("benchmark_total_return_pct") or {}).items():
            benchmark_return = float(value)
            lines.append(f"| {benchmark} | {benchmark_return:+.2f}% | {strategy_return - benchmark_return:+.2f}pp |")
        active = portfolio.get("active_cycle") or {}
        selected = list(active.get("selected", []))
        lines.extend(["", f"当前周期：`{active.get('status', 'idle')}`，信号日 `{active.get('signal_date', 'N/A')}`。"])
        if selected:
            lines.append("候选：" + "、".join(f"{item['code']} ({float(item['screen_score']):.1f})" for item in selected))
        else:
            lines.append("候选：无，当前周期保持现金。")
        lines.append("")
    lines.extend(["## 候选与选股理由", ""])
    for name, portfolio in dict(state["portfolios"]).items():
        active = portfolio.get("active_cycle") or {}
        selected = list(active.get("selected", []))
        lines.extend([
            f"### {name}",
            "",
            f"信号日 `{active.get('signal_date', 'N/A')}`，当前周期 `{active.get('status', 'idle')}`。",
            "",
        ])
        if not selected:
            lines.extend(["当前无候选，组合保持现金。", ""])
            continue
        for index, item in enumerate(selected, start=1):
            lines.extend([
                f"#### {index}. {item['code']} - {float(item['screen_score']):.1f} 分",
                "",
                f"> {item.get('selection_thesis') or '旧周期未保存结构化选股理由。'}",
                "",
            ])
            factor_scores = dict(item.get("factor_scores") or {})
            factor_weights = dict(item.get("factor_weights") or {})
            if factor_scores:
                factors = []
                for key, value in factor_scores.items():
                    if value is None:
                        continue
                    label = FACTOR_LABELS.get(str(key), str(key))
                    weight = float(factor_weights.get(key) or 0.0) * 100.0
                    factors.append(f"{label} {float(value):.1f}（{weight:.0f}%）")
                if factors:
                    lines.append("- 因子：" + "；".join(factors))
            lines.append("- 入选证据：" + _join_report_items(item.get("reasons_pass"), "未记录"))
            lines.append("- 观察项：" + _join_report_items(item.get("reasons_watch"), "暂无"))
            lines.append(
                "- 风险提示："
                + _join_report_items(item.get("risk_flags"), "未触发额外技术风险标记")
            )
            lines.append(
                "- 失效条件："
                + _join_report_items(
                    item.get("invalidation_conditions"),
                    "按统一硬风控执行",
                )
            )
            lines.append("- 数据来源：" + _join_report_items(item.get("data_sources"), "未记录"))
            lines.extend([
                "- 分数说明："
                + str(
                    item.get("score_explanation")
                    or "规则排序分，不是上涨概率或预期收益。"
                ),
                "",
            ])
    lines.extend([
        "## 验证门槛",
        "",
        "每个正式股票池至少完成 20 个不重叠的 10 交易日周期，并同时盈利且跑赢至少 3/5 基准，才会把实时观察状态标记为通过。历史综合回测门槛仍独立生效。",
        "",
    ])
    return "\n".join(lines)


def _join_report_items(value: object, fallback: str) -> str:
    if not isinstance(value, (list, tuple)):
        return fallback
    items = [str(item).strip() for item in value if str(item).strip()]
    return "；".join(items) if items else fallback


def _bar_on_date(frame: pd.DataFrame | None, target: date):
    if frame is None or frame.empty:
        return None
    matches = frame.loc[frame["date"] == target]
    return None if matches.empty else matches.iloc[-1]


def _bar_on_or_before(frame: pd.DataFrame | None, target: date):
    if frame is None or frame.empty:
        return None
    matches = frame.loc[frame["date"] <= target]
    return None if matches.empty else matches.iloc[-1]


def _net_return(gross_return: float, per_side_cost_bps: float) -> float:
    round_trip_cost = 2.0 * per_side_cost_bps / 10_000.0
    return (1.0 + gross_return) * (1.0 - round_trip_cost) - 1.0
