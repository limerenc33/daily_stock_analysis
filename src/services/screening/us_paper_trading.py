"""Deterministic paper-trading ledger for the US screening strategies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from src.services.screening.filter import apply_hard_filters
from src.services.screening.scorer import compute_screen_scores
from src.services.screening.strategy import load_all_strategies
from src.services.screening.us_candidate_evidence import FACTOR_LABELS, build_us_candidate_evidence
from src.services.screening.us_news_intelligence import (
    INTELLIGENCE_SCORECARD_VERSION,
    INTELLIGENCE_STRATEGY_VERSION,
    YahooUSNewsIntelligenceProvider,
    apply_intelligence_adjustment,
)
from src.services.screening.us_backtest import (
    _backtest_filters,
    _feature_row,
    normalize_price_history,
)

MARKET_TIMEZONE = "America/New_York"
DISPLAY_TIMEZONE = "Asia/Shanghai"
UTC_TIMEZONE = "UTC"


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
    grid_step_pct: float = 3.0
    grid_take_profit_levels: int = 2
    grid_stop_loss_levels: int = 2
    news_intelligence_enabled: bool = True
    news_max_items: int = 8
    news_max_workers: int = 4


def _validate_config(config: USPaperTradingConfig) -> None:
    if config.top_k <= 0 or config.holding_days <= 0 or config.lookback_days < 60:
        raise ValueError("invalid paper-trading configuration")
    if (
        config.grid_step_pct <= 0
        or config.grid_take_profit_levels <= 0
        or config.grid_stop_loss_levels <= 0
    ):
        raise ValueError("invalid grid configuration")
    if not 0.0 <= config.minimum_universe_coverage <= 1.0:
        raise ValueError("minimum_universe_coverage must be between 0 and 1")
    if config.news_max_items <= 0 or config.news_max_workers <= 0:
        raise ValueError("invalid news intelligence configuration")


def create_paper_trading_state(
    universes: Mapping[str, Sequence[str]],
    *,
    benchmarks: Sequence[str],
    config: USPaperTradingConfig | None = None,
) -> dict[str, object]:
    """Create a JSON-serializable multi-universe ledger."""
    config = config or USPaperTradingConfig()
    _validate_config(config)
    if not benchmarks:
        raise ValueError("at least one benchmark is required")
    return {
        "schema_version": 2,
        "strategy": config.strategy_name,
        "strategy_version": "2.0",
        "scorecard_version": None,
        "research_status": "not_validated",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": None,
        "time_semantics": {
            "market_timezone": MARKET_TIMEZONE,
            "display_timezone": DISPLAY_TIMEZONE,
            "timestamp_timezone": UTC_TIMEZONE,
            "market_date_definition": "US trading session date in America/New_York",
            "data_cutoff_definition": "latest completed common US market date; daily bars use session OHLC",
        },
        "news_intelligence": {
            "status": "not_run",
            "scorecard_version": INTELLIGENCE_SCORECARD_VERSION,
            "effective_from": "next_cycle",
            "market_digest": None,
        },
        "strategy_activation": {
            "deployed_version": INTELLIGENCE_STRATEGY_VERSION,
            "effective_from": "next_created_cycle",
            "preserve_existing_cycle": True,
        },
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
                "event_log": [],
                "trade_log": [],
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
    news_provider: YahooUSNewsIntelligenceProvider | None = None,
) -> dict[str, object]:
    """Advance every portfolio through the latest completed market session."""
    upgrade_paper_trading_state(state)
    config = USPaperTradingConfig(**dict(state["config"]))
    _validate_config(config)
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
    state["scorecard_version"] = (
        INTELLIGENCE_SCORECARD_VERSION
        if config.news_intelligence_enabled and news_provider is not None
        else str(strategy.screening.scorecard_profile.get("version") or "us_evidence_v2")
    )
    if news_provider is not None and config.news_intelligence_enabled:
        market_digest = news_provider.market_digest(as_of=effective_date)
        state.setdefault("news_intelligence", {})["market_digest"] = market_digest
        state["news_intelligence"]["status"] = str(market_digest.get("status") or "unavailable")
    for name, portfolio in dict(state["portfolios"]).items():
        migration_event = _ensure_active_cycle_evidence(
            portfolio,
            normalized,
            config,
            strategies_dir=strategies_dir,
        )
        if news_provider is not None and config.news_intelligence_enabled:
            _refresh_active_cycle_news(portfolio, effective_date, news_provider)
        events[name] = _advance_portfolio(
            portfolio,
            normalized,
            benchmarks,
            effective_date,
            config,
            strategies_dir=strategies_dir,
            news_provider=news_provider if config.news_intelligence_enabled else None,
        )
        if migration_event is not None:
            events[name].insert(0, migration_event)
        _ensure_trade_log(portfolio)
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    state["latest_market_date"] = effective_date.isoformat()
    if config.news_intelligence_enabled and news_provider is not None:
        state["strategy_version"] = INTELLIGENCE_STRATEGY_VERSION
        state.setdefault("news_intelligence", {})["scorecard_version"] = INTELLIGENCE_SCORECARD_VERSION
        state["news_intelligence"]["effective_from"] = "next_cycle"
        state.setdefault("strategy_activation", {})["deployed_version"] = INTELLIGENCE_STRATEGY_VERSION
        state["strategy_activation"]["effective_from"] = "next_created_cycle"
    state["live_validation"] = evaluate_live_validation(state)
    state["last_events"] = events
    return state


def upgrade_paper_trading_state(state: dict[str, object]) -> bool:
    """Upgrade an existing ledger in place without replaying old market bars."""
    changed = int(state.get("schema_version") or 0) < 2
    defaults = asdict(USPaperTradingConfig())
    config_payload = dict(state.get("config") or {})
    for key, value in defaults.items():
        if key not in config_payload:
            config_payload[key] = value
            changed = True
    state["config"] = config_payload
    config = USPaperTradingConfig(**config_payload)
    _validate_config(config)
    for portfolio in dict(state.get("portfolios") or {}).values():
        portfolio.setdefault("event_log", [])
        portfolio.setdefault("trade_log", [])
        _ensure_trade_log(portfolio)
        cycle = portfolio.get("active_cycle")
        if not isinstance(cycle, dict) or cycle.get("status") not in {"open", "awaiting_settlement"}:
            continue
        for position in cycle.get("positions", []):
            if "grid" not in position:
                _initialize_position_grid(position, config)
                position["last_evaluated_date"] = portfolio.get("last_processed_date")
                changed = True
    state.setdefault("news_intelligence", {
        "status": "not_run",
        "scorecard_version": INTELLIGENCE_SCORECARD_VERSION,
        "effective_from": "next_cycle",
        "market_digest": None,
    })
    state.setdefault("time_semantics", {
        "market_timezone": MARKET_TIMEZONE,
        "display_timezone": DISPLAY_TIMEZONE,
        "timestamp_timezone": UTC_TIMEZONE,
        "market_date_definition": "US trading session date in America/New_York",
        "data_cutoff_definition": "latest completed common US market date; daily bars use session OHLC",
    })
    state.setdefault("strategy_version", "2.0")
    state.setdefault("strategy_activation", {
        "deployed_version": INTELLIGENCE_STRATEGY_VERSION,
        "effective_from": "next_created_cycle",
        "preserve_existing_cycle": True,
    })
    if state.get("schema_version") != 2:
        state["schema_version"] = 2
        changed = True
    return changed


def _ensure_trade_log(portfolio: dict[str, object]) -> None:
    """Backfill a complete, idempotent buy/sell ledger from existing state."""
    trade_log = list(portfolio.get("trade_log") or [])
    known = {str(item.get("trade_id")) for item in trade_log}

    def add(item: dict[str, object]) -> None:
        trade_id = str(item["trade_id"])
        if trade_id not in known:
            trade_log.append(item)
            known.add(trade_id)

    active = portfolio.get("active_cycle")
    if isinstance(active, dict):
        cycle_key = str(active.get("signal_date") or "unknown")
        for position in list(active.get("positions") or []):
            code = str(position.get("code") or "")
            if not code or position.get("entry_open") is None:
                continue
            add({
                "trade_id": f"{cycle_key}:{code}:entry",
                "side": "buy",
                "reason": "cycle_entry",
                "code": code,
                "date": str(position.get("entry_date") or cycle_key),
                "price": float(position.get("entry_open") or 0.0),
                "quantity": float(position.get("quantity") or 0.0),
                "gross_notional": float(position.get("allocated_capital") or 0.0),
                "source": "US daily open simulation",
                "market_timezone": MARKET_TIMEZONE,
            })
            for fill in list(position.get("fills") or []):
                add(_trade_from_fill(fill, code, cycle_key))
    for cycle in list(portfolio.get("closed_cycles") or []):
        cycle_key = str(cycle.get("signal_date") or cycle.get("entry_date") or "unknown")
        for trade in list(cycle.get("trades") or []):
            code = str(trade.get("code") or "")
            if code and trade.get("entry_open") is not None:
                add({
                    "trade_id": f"{cycle_key}:{code}:entry",
                    "side": "buy",
                    "reason": "cycle_entry",
                    "code": code,
                    "date": str(trade.get("entry_date") or cycle.get("entry_date") or cycle_key),
                    "price": float(trade.get("entry_open") or 0.0),
                    "quantity": float(trade.get("quantity") or 0.0),
                    "gross_notional": float(trade.get("allocated_capital") or 0.0),
                    "source": "US daily open simulation",
                    "market_timezone": MARKET_TIMEZONE,
                })
            for fill in list(trade.get("fills") or []):
                add(_trade_from_fill(fill, code, cycle_key))
    trade_log.sort(key=lambda item: (str(item.get("date") or ""), str(item.get("trade_id") or "")))
    portfolio["trade_log"] = trade_log


def _trade_from_fill(fill: Mapping[str, object], code: str, cycle_key: str) -> dict[str, object]:
    return {
        "trade_id": f"{cycle_key}:{code}:{fill.get('date')}:{fill.get('reason')}:{fill.get('grid_level') or 0}",
        "side": "sell",
        "reason": str(fill.get("reason") or "exit"),
        "code": code,
        "date": str(fill.get("date") or "unknown"),
        "observed_at": fill.get("observed_at"),
        "price": float(fill.get("fill_price") or 0.0),
        "trigger_price": float(fill.get("trigger_price") or 0.0),
        "quantity": float(fill.get("quantity") or 0.0),
        "net_proceeds": float(fill.get("net_proceeds") or 0.0),
        "source": str(fill.get("source") or "unknown"),
        "market_timezone": MARKET_TIMEZONE,
    }


def apply_realtime_grid_quotes(
    state: dict[str, object],
    quotes: Mapping[str, Mapping[str, object]],
    *,
    observed_at: datetime,
    market_date: date | None = None,
) -> dict[str, list[dict[str, object]]]:
    """Apply fresh observed prices to open positions and persist only exits."""
    upgrade_paper_trading_state(state)
    config = USPaperTradingConfig(**dict(state["config"]))
    _validate_config(config)
    effective_date = market_date or observed_at.date()
    events_by_portfolio: dict[str, list[dict[str, object]]] = {}
    for name, portfolio in dict(state.get("portfolios") or {}).items():
        cycle = portfolio.get("active_cycle")
        events: list[dict[str, object]] = []
        if isinstance(cycle, dict) and cycle.get("status") in {"open", "awaiting_settlement"}:
            for position in cycle.get("positions", []):
                if position.get("status") == "closed":
                    continue
                quote = quotes.get(str(position["code"]))
                if not quote:
                    continue
                price = float(quote.get("price") or 0.0)
                if price <= 0:
                    continue
                source = str(quote.get("source") or "realtime quote")
                grid = dict(position["grid"])
                stop_price = float(grid["stop_loss_price"])
                if price <= stop_price:
                    events.append(_execute_position_exit(
                        position,
                        float(position["remaining_quantity"]),
                        price,
                        effective_date,
                        "grid_stop_loss",
                        stop_price,
                        config,
                        source=source,
                        observed_at=observed_at.isoformat(),
                    ))
                else:
                    completed = {
                        int(level)
                        for level in grid.get("completed_take_profit_levels", [])
                    }
                    targets = [float(value) for value in grid["take_profit_prices"]]
                    for level, target in enumerate(targets, start=1):
                        if level in completed or price < target:
                            continue
                        events.append(_execute_position_exit(
                            position,
                            _take_profit_quantity(position, level, len(targets)),
                            price,
                            effective_date,
                            "grid_take_profit",
                            target,
                            config,
                            source=source,
                            grid_level=level,
                            observed_at=observed_at.isoformat(),
                        ))
                        if position.get("status") == "closed":
                            break
                if any(event.get("code") == position.get("code") for event in events):
                    position["last_evaluated_date"] = effective_date.isoformat()
            if _all_positions_exit_date(cycle) is not None:
                cycle["status"] = "awaiting_settlement"
        if events:
            portfolio["event_log"] = [*list(portfolio.get("event_log", [])), *events][-200:]
            events_by_portfolio[str(name)] = events
    if events_by_portfolio:
        for portfolio in dict(state.get("portfolios") or {}).values():
            _ensure_trade_log(portfolio)
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        state["last_realtime_check"] = observed_at.isoformat()
        state["last_events"] = events_by_portfolio
    return events_by_portfolio


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
    news_provider: YahooUSNewsIntelligenceProvider | None = None,
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
    if isinstance(cycle, dict) and cycle.get("status") in {"open", "awaiting_settlement"}:
        holding_dates = [
            value
            for value in benchmark_calendar["date"]
            if value >= date.fromisoformat(cycle["entry_date"]) and value <= as_of
        ]
        exit_date = _all_positions_exit_date(cycle)
        if exit_date is None:
            for session_number, market_date in enumerate(holding_dates[: config.holding_days], start=1):
                for position in cycle.get("positions", []):
                    last_evaluated = position.get("last_evaluated_date")
                    if last_evaluated and date.fromisoformat(str(last_evaluated)) >= market_date:
                        continue
                    bar = _bar_on_date(histories.get(str(position["code"])), market_date)
                    if bar is None:
                        raise ValueError(f"missing grid bar for {position['code']} on {market_date}")
                    events.extend(
                        _process_position_daily(
                            position,
                            bar,
                            market_date,
                            config,
                            time_exit=session_number == config.holding_days,
                        )
                    )
                exit_date = _all_positions_exit_date(cycle)
                if exit_date is not None:
                    break
        if exit_date is not None and exit_date <= as_of:
            closed = _finalize_cycle(cycle, portfolio, histories, benchmarks, exit_date, config)
            events.append({
                "type": "closed",
                "exit_date": exit_date.isoformat(),
                "strategy_return_pct": closed["strategy_return_pct"],
            })
            portfolio["active_cycle"] = None
    if portfolio.get("active_cycle") is None:
        universe = [str(item) for item in portfolio["universe"]]
        ranked = rank_candidates_on_date(
            config.strategy_name,
            {code: histories.get(code, pd.DataFrame()) for code in universe},
            as_of,
            top_k=max(config.top_k * 3, config.top_k),
            lookback_days=config.lookback_days,
            minimum_universe_coverage=config.minimum_universe_coverage,
            strategies_dir=strategies_dir,
        )
        if news_provider is not None and ranked["selected"]:
            selected, news_status = _apply_news_to_candidates(
                list(ranked["selected"]),
                as_of=as_of,
                top_k=config.top_k,
                provider=news_provider,
            )
            ranked["selected"] = selected
            portfolio["latest_news_intelligence"] = news_status
        else:
            ranked["selected"] = list(ranked["selected"])[: config.top_k]
        portfolio["active_cycle"] = {
            "status": "pending",
            "signal_date": as_of.isoformat(),
            "strategy_version": (
                INTELLIGENCE_STRATEGY_VERSION if news_provider is not None else "2.0"
            ),
            "scorecard_version": (
                INTELLIGENCE_SCORECARD_VERSION if news_provider is not None else "us_evidence_v2"
            ),
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
    exit_events = [event for event in events if event.get("type") in {
        "grid_take_profit", "grid_stop_loss", "time_exit",
    }]
    if exit_events:
        portfolio["event_log"] = [*list(portfolio.get("event_log", [])), *exit_events][-200:]
    return events


def _apply_news_to_candidates(
    candidates: list[dict[str, object]],
    *,
    as_of: date,
    top_k: int,
    provider: YahooUSNewsIntelligenceProvider,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    codes = [str(item.get("code") or "") for item in candidates]
    intelligence = provider.collect(codes, as_of=as_of)
    adjusted = []
    excluded = []
    audit_rows = []
    for candidate in candidates:
        item = dict(candidate)
        item["signal_date"] = as_of.isoformat()
        item["scorecard_version"] = INTELLIGENCE_SCORECARD_VERSION
        apply_intelligence_adjustment(item, intelligence.get(str(item.get("code") or "")))
        audit_rows.append({
            "code": str(item.get("code") or ""),
            "technical_screen_score": item.get("technical_screen_score"),
            "intelligence_adjustment": item.get("intelligence_adjustment"),
            "screen_score": item.get("screen_score"),
            "hard_exclusion": bool(item.get("hard_exclusion")),
            "risk_flags": list(item.get("risk_flags") or []),
        })
        if not item.get("hard_exclusion"):
            adjusted.append(item)
        else:
            excluded.append(item)
    adjusted.sort(key=lambda item: (-float(item.get("screen_score") or 0.0), str(item.get("code") or "")))
    available = sum(
        str(item.get("news_intelligence", {}).get("status")) in {"available", "partial"}
        for item in adjusted
    )
    return adjusted[:top_k], {
        "as_of": as_of.isoformat(),
        "requested": len(codes),
        "available": available,
        "status": "available" if available else "unavailable",
        "scorecard_version": INTELLIGENCE_SCORECARD_VERSION,
        "items": {str(item.get("code")): item.get("news_intelligence") for item in adjusted[:top_k]},
        "technical_candidates": audit_rows,
        "excluded_candidates": [
            {
                "code": str(item.get("code") or ""),
                "technical_screen_score": item.get("technical_screen_score"),
                "intelligence_adjustment": item.get("intelligence_adjustment"),
                "risk_flags": list(item.get("risk_flags") or []),
                "news_intelligence": item.get("news_intelligence"),
            }
            for item in excluded
        ],
    }


def _refresh_active_cycle_news(
    portfolio: dict[str, object],
    as_of: date,
    provider: YahooUSNewsIntelligenceProvider,
) -> None:
    """Refresh daily evidence for display without changing an existing cycle."""
    cycle = portfolio.get("active_cycle")
    if not isinstance(cycle, dict):
        return
    selected = list(cycle.get("selected") or [])
    positions = list(cycle.get("positions") or [])
    codes = list(dict.fromkeys(
        str(item.get("code") or "")
        for item in [*selected, *positions]
        if str(item.get("code") or "")
    ))
    if not codes:
        return
    intelligence = provider.collect(codes, as_of=as_of)
    portfolio["latest_news_intelligence"] = {
        "as_of": as_of.isoformat(),
        "requested": len(codes),
        "available": sum(
            str(item.get("status")) in {"available", "partial"}
            for item in intelligence.values()
        ),
        "status": "available" if any(
            str(item.get("status")) in {"available", "partial"}
            for item in intelligence.values()
        ) else "unavailable",
        "scorecard_version": INTELLIGENCE_SCORECARD_VERSION,
        "items": intelligence,
        "affects_current_cycle": False,
    }
    cycle["latest_news_intelligence"] = portfolio["latest_news_intelligence"]


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
        position = {
            **item,
            "entry_date": entry_date.isoformat(),
            "entry_open": round(entry_open, 6),
            "allocated_capital": round(allocation, 6),
            "quantity": round(allocation / entry_open, 8),
        }
        _initialize_position_grid(position, config)
        positions.append(position)
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


def _initialize_position_grid(
    position: dict[str, object],
    config: USPaperTradingConfig,
) -> None:
    entry_open = float(position["entry_open"])
    quantity = float(position["quantity"])
    step = config.grid_step_pct / 100.0
    position.update({
        "status": "open",
        "remaining_quantity": round(quantity, 8),
        "realized_value": 0.0,
        "fills": [],
        "last_price": round(entry_open, 6),
        "last_evaluated_date": None,
        "grid": {
            "step_pct": config.grid_step_pct,
            "take_profit_prices": [
                round(entry_open * (1.0 + step * level), 6)
                for level in range(1, config.grid_take_profit_levels + 1)
            ],
            "stop_loss_price": round(
                entry_open * (1.0 - step * config.grid_stop_loss_levels), 6
            ),
            "completed_take_profit_levels": [],
        },
    })


def _process_position_daily(
    position: dict[str, object],
    bar: Mapping[str, object],
    market_date: date,
    config: USPaperTradingConfig,
    *,
    time_exit: bool,
) -> list[dict[str, object]]:
    if position.get("status") == "closed":
        return []
    open_price = float(bar["open"])
    high = float(bar["high"])
    low = float(bar["low"])
    close = float(bar["close"])
    position["last_price"] = round(close, 6)
    position["last_evaluated_date"] = market_date.isoformat()
    grid = dict(position["grid"])
    stop_price = float(grid["stop_loss_price"])
    events: list[dict[str, object]] = []

    # Daily bars do not reveal whether the high or low happened first. Use the
    # adverse ordering so validation cannot benefit from look-ahead bias.
    if open_price <= stop_price:
        events.append(_execute_position_exit(
            position, float(position["remaining_quantity"]), open_price,
            market_date, "grid_stop_loss", stop_price, config,
            source="Yahoo Finance daily open gap",
        ))
        return events
    if low <= stop_price:
        events.append(_execute_position_exit(
            position, float(position["remaining_quantity"]), stop_price,
            market_date, "grid_stop_loss", stop_price, config,
            source="Yahoo Finance daily OHLC",
        ))
        return events

    completed = {int(level) for level in grid.get("completed_take_profit_levels", [])}
    targets = [float(value) for value in grid["take_profit_prices"]]
    for level, target in enumerate(targets, start=1):
        if level in completed or float(position["remaining_quantity"]) <= 0:
            continue
        if open_price >= target:
            fill_price = open_price
            source = "Yahoo Finance daily open gap"
        elif high >= target:
            fill_price = target
            source = "Yahoo Finance daily OHLC"
        else:
            break
        quantity = _take_profit_quantity(position, level, len(targets))
        events.append(_execute_position_exit(
            position, quantity, fill_price, market_date, "grid_take_profit",
            target, config, source=source, grid_level=level,
        ))

    if time_exit and float(position["remaining_quantity"]) > 0:
        events.append(_execute_position_exit(
            position, float(position["remaining_quantity"]), close,
            market_date, "time_exit", close, config,
            source="Yahoo Finance daily close",
        ))
    return events


def _take_profit_quantity(
    position: Mapping[str, object],
    level: int,
    total_levels: int,
) -> float:
    remaining = float(position["remaining_quantity"])
    if level >= total_levels:
        return remaining
    return min(round(float(position["quantity"]) / total_levels, 8), remaining)


def _execute_position_exit(
    position: dict[str, object],
    quantity: float,
    fill_price: float,
    exit_date: date,
    reason: str,
    trigger_price: float,
    config: USPaperTradingConfig,
    *,
    source: str,
    grid_level: int | None = None,
    observed_at: str | None = None,
) -> dict[str, object]:
    remaining = float(position["remaining_quantity"])
    sold_quantity = min(max(float(quantity), 0.0), remaining)
    if sold_quantity <= 0:
        raise ValueError("grid exit quantity must be positive")
    cost_multiplier = 1.0 - 2.0 * config.per_side_cost_bps / 10_000.0
    net_proceeds = sold_quantity * fill_price * cost_multiplier
    fill = {
        "date": exit_date.isoformat(),
        "observed_at": observed_at,
        "reason": reason,
        "grid_level": grid_level,
        "trigger_price": round(trigger_price, 6),
        "fill_price": round(fill_price, 6),
        "quantity": round(sold_quantity, 8),
        "net_proceeds": round(net_proceeds, 6),
        "source": source,
    }
    position.setdefault("fills", []).append(fill)
    position["realized_value"] = round(float(position.get("realized_value") or 0.0) + net_proceeds, 6)
    position["remaining_quantity"] = round(max(remaining - sold_quantity, 0.0), 8)
    position["last_price"] = round(fill_price, 6)
    if grid_level is not None:
        completed = list(dict.fromkeys([
            *list(position["grid"].get("completed_take_profit_levels", [])),
            grid_level,
        ]))
        position["grid"]["completed_take_profit_levels"] = completed
    if float(position["remaining_quantity"]) <= 1e-7:
        position["remaining_quantity"] = 0.0
        position["status"] = "closed"
        position["exit_date"] = exit_date.isoformat()
        position["exit_reason"] = reason
        position["exit_price"] = round(
            sum(float(item["fill_price"]) * float(item["quantity"]) for item in position["fills"])
            / float(position["quantity"]),
            6,
        )
        position["net_return_pct"] = round(
            (float(position["realized_value"]) / float(position["allocated_capital"]) - 1.0) * 100.0,
            6,
        )
        position["pnl"] = round(
            float(position["realized_value"]) - float(position["allocated_capital"]),
            6,
        )
    return {
        "type": reason,
        "code": str(position["code"]),
        **fill,
        "remaining_quantity": position["remaining_quantity"],
    }


def _all_positions_exit_date(cycle: Mapping[str, object]) -> date | None:
    positions = list(cycle.get("positions", []))
    if not positions or not all(item.get("status") == "closed" for item in positions):
        return None
    return max(date.fromisoformat(str(item["exit_date"])) for item in positions)


def _finalize_cycle(
    cycle: dict[str, object],
    portfolio: dict[str, object],
    histories: Mapping[str, pd.DataFrame],
    benchmarks: Sequence[str],
    exit_date: date,
    config: USPaperTradingConfig,
) -> dict[str, object]:
    closed_trades = []
    for position in cycle.get("positions", []):
        if position.get("status") != "closed":
            raise ValueError(f"cannot finalize cycle with open position {position['code']}")
        closed_trades.append({
            **position,
            "exit_close": position.get("exit_price"),
        })
    final_equity = sum(float(position["realized_value"]) for position in closed_trades)
    strategy_return = final_equity / float(cycle["starting_equity"]) - 1.0
    portfolio["realized_equity"] = round(final_equity, 6)
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
        "exit_model": "grid_v1",
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
    if isinstance(cycle, dict) and cycle.get("status") in {"open", "awaiting_settlement"}:
        position_values = []
        for position in cycle.get("positions", []):
            bar = _bar_on_or_before(histories.get(str(position["code"])), as_of)
            mark_price = (
                float(bar["close"])
                if bar is not None
                else float(position.get("last_price") or position["entry_open"])
            )
            position_values.append(_position_marked_value(position, mark_price, config))
        if position_values:
            strategy_equity = sum(position_values)
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


def _position_marked_value(
    position: Mapping[str, object],
    mark_price: float,
    config: USPaperTradingConfig,
) -> float:
    cost_multiplier = 1.0 - 2.0 * config.per_side_cost_bps / 10_000.0
    return (
        float(position.get("realized_value") or 0.0)
        + float(position.get("remaining_quantity") or 0.0) * mark_price * cost_multiplier
    )


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
        completed = sum(
            cycle.get("exit_model") == "grid_v1"
            for cycle in portfolio.get("closed_cycles", [])
        )
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
        f"> 策略：`{strategy}` {state.get('strategy_version', '2.0')}。仅用于研究验证，不构成投资建议，不连接券商或真实下单。",
        f"> 实时验证状态：**{'通过' if gate.get('effective') else '证据不足或未通过'}**。",
        "> 新闻、财报和研报证据每日更新；新资讯策略只对部署后创建的下一周期生效，不改写当前持仓。",
        "> 时间口径：美股交易日按 `America/New_York`；行情截止为最近完成的共同美股交易日；时间戳以 UTC 存储，页面同时换算为北京时间。",
        (
            f"> 网格风控：每格 {float(dict(state['config']).get('grid_step_pct') or 3.0):.1f}%，"
            f"上涨 {int(dict(state['config']).get('grid_take_profit_levels') or 2)} 格分批止盈，"
            f"下跌 {int(dict(state['config']).get('grid_stop_loss_levels') or 2)} 格全部止损。"
        ),
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
    lines.extend(["", "## 网格风控与成交", ""])
    for name, portfolio in dict(state["portfolios"]).items():
        active = portfolio.get("active_cycle") or {}
        positions = list(active.get("positions", []))
        lines.extend([f"### {name}", ""])
        if positions:
            lines.extend([
                "| 代码 | 入场价 | 止损价 | 下一止盈格 | 剩余仓位 | 状态 |",
                "| --- | ---: | ---: | ---: | ---: | --- |",
            ])
            for position in positions:
                grid = dict(position.get("grid") or {})
                completed = {int(value) for value in grid.get("completed_take_profit_levels", [])}
                targets = [float(value) for value in grid.get("take_profit_prices", [])]
                next_target = next(
                    (value for level, value in enumerate(targets, start=1) if level not in completed),
                    None,
                )
                original_quantity = float(position.get("quantity") or 0.0)
                remaining_quantity = float(position.get("remaining_quantity") or 0.0)
                remaining_pct = remaining_quantity / original_quantity * 100.0 if original_quantity else 0.0
                lines.append(
                    f"| {position['code']} | {float(position['entry_open']):.2f} | "
                    f"{float(grid.get('stop_loss_price') or 0.0):.2f} | "
                    f"{next_target:.2f} | {remaining_pct:.0f}% | {position.get('status', 'open')} |"
                    if next_target is not None
                    else
                    f"| {position['code']} | {float(position['entry_open']):.2f} | "
                    f"{float(grid.get('stop_loss_price') or 0.0):.2f} | 已完成 | "
                    f"{remaining_pct:.0f}% | {position.get('status', 'open')} |"
                )
        else:
            lines.append("当前尚未开仓，网格价将在下一交易日开盘成交后生成。")
        recent_events = list(portfolio.get("event_log", []))[-5:]
        if recent_events:
            lines.extend(["", "最近成交："])
            for event in reversed(recent_events):
                reason = _exit_reason_label(str(event.get("type") or event.get("reason") or ""))
                lines.append(
                    f"- `{event.get('date', 'N/A')}` {event.get('code', 'N/A')} {reason}，"
                    f"触发 {float(event.get('trigger_price') or 0.0):.2f}，"
                    f"成交 {float(event.get('fill_price') or 0.0):.2f}，"
                    f"数量 {float(event.get('quantity') or 0.0):.4f}。"
                )
        lines.append("")
    lines.extend(["## 每日重点资讯、财报与研报", ""])
    market_digest = dict(dict(state.get("news_intelligence") or {}).get("market_digest") or {})
    if market_digest:
        lines.append(f"- 市场摘要：{market_digest.get('summary', '无可用摘要')}")
        lines.append("")
    for name, portfolio in dict(state["portfolios"]).items():
        news = dict(portfolio.get("latest_news_intelligence") or {})
        lines.extend([f"### {name}", ""])
        if not news:
            lines.extend(["尚未生成资讯快照；来源不可用时会明确记录原因，不会作为中性分。", ""])
            continue
        lines.append(
            f"资讯日期 `{news.get('as_of', 'N/A')}`，可用标的 "
            f"{int(news.get('available') or 0)}/{int(news.get('requested') or 0)}。"
        )
        if news.get("affects_current_cycle") is False:
            lines.append("当前持仓周期仅展示资讯证据，不据此改仓；调整从下一周期选股开始。")
        lines.append("")
        for code, intelligence_payload in dict(news.get("items") or {}).items():
            intelligence = dict(intelligence_payload or {})
            lines.extend([
                f"#### {code}",
                "",
                f"- 汇总：{intelligence.get('summary', '无')}",
                f"- 分析：{intelligence.get('analysis', '无')}",
                f"- 预期影响：{intelligence.get('expected_impact', '无')}",
                f"- 影响面：{'、'.join(intelligence.get('impact_channels') or []) or '待核验'}；期限：{intelligence.get('impact_horizon', 'unknown')}",
            ])
            items = list(intelligence.get("items") or [])
            if not items:
                lines.append("- 资料：没有可核验的近期资料。")
            for item in items[:5]:
                title = str(item.get("title") or "无标题")
                url = str(item.get("url") or "").strip()
                source = str(item.get("source") or "来源未知")
                published = str(item.get("published_at") or "日期未知")
                lines.append(f"- 资料：[{title}]({url})（{source}，{published}）" if url else f"- 资料：{title}（{source}，{published}）")
                if item.get("summary"):
                    lines.append(f"  - 原文摘要：{item['summary']}")
                if item.get("analysis"):
                    lines.append(f"  - 条目分析：{item['analysis']}")
                if item.get("expected_impact"):
                    lines.append(f"  - 条目预期影响：{item['expected_impact']}")
            lines.append("")
        excluded = list(news.get("excluded_candidates") or [])
        if excluded:
            lines.append(
                "资讯风险排除：" + "、".join(str(item.get("code") or "N/A") for item in excluded)
                + "。候选保留在审计记录中，但不会进入下一周期持仓。"
            )
            lines.append("")
    lines.extend(["## 完整模拟交易流水", ""])
    for name, portfolio in dict(state["portfolios"]).items():
        lines.extend([f"### {name}", "", "| 日期 | 方向 | 代码 | 价格 | 数量 | 原因 | 来源 |", "| --- | --- | --- | ---: | ---: | --- | --- |"])
        trade_log = list(portfolio.get("trade_log") or [])
        if not trade_log:
            lines.append("| N/A | - | - | - | - | 尚无流水 | - |")
        for trade in trade_log:
            lines.append(
                f"| {trade.get('date', 'N/A')} | {'买入' if trade.get('side') == 'buy' else '卖出'} | "
                f"{trade.get('code', 'N/A')} | {float(trade.get('price') or 0.0):.2f} | "
                f"{float(trade.get('quantity') or 0.0):.4f} | {trade.get('reason', 'N/A')} | {trade.get('source', 'N/A')} |"
            )
        lines.append("")
    lines.extend(["## 多基准比较", ""])
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
        lines.append(
            f"周期策略版本：`{active.get('strategy_version', '2.0')}`；"
            f"评分卡：`{active.get('scorecard_version', state.get('scorecard_version', 'N/A'))}`。"
        )
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
            if item.get("technical_screen_score") is not None:
                lines.append(
                    f"- 分数拆分：技术 {float(item.get('technical_screen_score') or 0.0):.1f}，"
                    f"资讯调整 {float(item.get('intelligence_adjustment') or 0.0):+.1f}，"
                    f"最终 {float(item.get('screen_score') or 0.0):.1f}"
                )
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
        "每个正式股票池至少完成 20 个不重叠周期（最长 10 个交易日，可能因网格止损或止盈提前退出），并同时盈利且跑赢至少 3/5 基准，才会把实时观察状态标记为通过。历史综合回测门槛仍独立生效。",
        "",
    ])
    return "\n".join(lines)


def render_grid_event_notification(
    state: Mapping[str, object],
    events_by_portfolio: Mapping[str, Sequence[Mapping[str, object]]],
) -> str:
    """Render the concise message sent only when an intraday grid fills."""
    lines = [
        "# 美股模拟交易网格触发",
        "",
        f"> 观察时间：`{state.get('last_realtime_check', 'N/A')}`。虚拟成交，不连接券商。",
        "",
    ]
    for name, events in events_by_portfolio.items():
        lines.extend([f"## {name}", ""])
        for event in events:
            lines.append(
                f"- **{event.get('code', 'N/A')}** {_exit_reason_label(str(event.get('type') or ''))}："
                f"触发价 {float(event.get('trigger_price') or 0.0):.2f}，"
                f"模拟成交价 {float(event.get('fill_price') or 0.0):.2f}，"
                f"数量 {float(event.get('quantity') or 0.0):.4f}，"
                f"剩余 {float(event.get('remaining_quantity') or 0.0):.4f}，"
                f"来源 {event.get('source', '未记录')}。"
            )
        lines.append("")
    lines.extend([
        "报价源、观察时间和成交明细已写入 paper-trading-state 账本；实际成交可能受延迟、跳空和滑点影响。",
        "",
    ])
    return "\n".join(lines)


def _exit_reason_label(reason: str) -> str:
    return {
        "grid_take_profit": "网格止盈",
        "grid_stop_loss": "网格止损",
        "time_exit": "到期退出",
    }.get(reason, reason or "退出")


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
