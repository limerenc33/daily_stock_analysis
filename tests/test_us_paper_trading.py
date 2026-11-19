from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
import yaml

import scripts.run_us_paper_trading as paper_trading_runner
from scripts.run_us_paper_trading import split_yfinance_download
from scripts.run_us_paper_trading import _write_archive_snapshot
from src.services.screening.us_paper_trading import (
    USPaperTradingConfig,
    advance_paper_trading_state,
    apply_realtime_grid_quotes,
    create_paper_trading_state,
    evaluate_live_validation,
    rank_candidates_on_date,
    render_grid_event_notification,
    render_paper_trading_report,
    upgrade_paper_trading_state,
)
from src.services.screening.strategy import load_all_strategies
from src.services.screening.us_candidate_evidence import build_us_candidate_evidence
from src.services.screening.us_news_intelligence import (
    INTELLIGENCE_SCORECARD_VERSION,
    YahooUSNewsIntelligenceProvider,
    _summarize_items,
    _normalize_analyst_actions,
    apply_intelligence_adjustment,
    unavailable_intelligence,
)


def _history(*, start: float = 100.0, step: float = 0.5, points: int = 180) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-01", periods=points)
    close = pd.Series([start + step * index for index in range(points)], dtype=float)
    return pd.DataFrame({
        "date": dates,
        "open": close - 0.1,
        "high": close + 0.4,
        "low": close - 0.4,
        "close": close,
        "volume": 1_000_000,
    })


def test_paper_trading_enters_next_open_and_exits_after_ten_sessions():
    histories = {
        "AAA": _history(step=0.7),
        "BBB": _history(step=0.5),
        "SPY": _history(step=0.3),
        "QQQ": _history(step=0.4),
        "IWM": _history(step=0.2),
        "DIA": _history(step=0.25),
        "RSP": _history(step=0.28),
    }
    dates = [value.date() for value in histories["SPY"]["date"]]
    config = USPaperTradingConfig(top_k=2, holding_days=10, minimum_universe_coverage=1.0)
    state = create_paper_trading_state(
        {"test_universe": ["AAA", "BBB"]},
        benchmarks=("SPY", "QQQ", "IWM", "DIA", "RSP"),
        config=config,
    )

    advance_paper_trading_state(state, histories, as_of=dates[100])
    portfolio = state["portfolios"]["test_universe"]
    assert portfolio["active_cycle"]["status"] == "pending"
    assert portfolio["active_cycle"]["signal_date"] == dates[100].isoformat()
    assert portfolio["active_cycle"]["selected"][0]["selection_thesis"]
    assert portfolio["active_cycle"]["selected"][0]["reasons_pass"]

    advance_paper_trading_state(state, histories, as_of=dates[101])
    active = portfolio["active_cycle"]
    assert active["status"] == "open"
    assert active["entry_date"] == dates[101].isoformat()
    assert active["positions"][0]["entry_open"] == pytest.approx(
        float(histories[active["positions"][0]["code"]].iloc[101]["open"])
    )

    advance_paper_trading_state(state, histories, as_of=dates[110])
    assert len(portfolio["closed_cycles"]) == 1
    closed = portfolio["closed_cycles"][0]
    assert closed["exit_date"] == dates[110].isoformat()
    assert closed["strategy_return_pct"] > 0.0
    assert set(closed["benchmark_return_pct"]) == {"SPY", "QQQ", "IWM", "DIA", "RSP"}
    assert closed["trades"][0]["selection_thesis"]
    assert closed["trades"][0]["factor_scores"]
    assert portfolio["active_cycle"]["status"] == "pending"
    assert portfolio["active_cycle"]["signal_date"] == dates[110].isoformat()


def test_paper_trading_is_idempotent_for_same_market_date():
    histories = {"AAA": _history(), "SPY": _history(step=0.3)}
    state = create_paper_trading_state(
        {"test_universe": ["AAA"]},
        benchmarks=("SPY",),
        config=USPaperTradingConfig(top_k=1, minimum_universe_coverage=1.0),
    )
    as_of = histories["SPY"].iloc[100]["date"].date()

    advance_paper_trading_state(state, histories, as_of=as_of)
    first = state["portfolios"]["test_universe"]
    snapshot_count = len(first["snapshots"])
    signal_date = first["active_cycle"]["signal_date"]
    advance_paper_trading_state(state, histories, as_of=as_of)

    assert len(first["snapshots"]) == snapshot_count
    assert first["active_cycle"]["signal_date"] == signal_date
    assert state["last_events"]["test_universe"][0]["type"] == "no_change"


def test_daily_grid_takes_partial_profit_then_exits_remaining_at_time_limit():
    histories = {"AAA": _history(step=0.1), "SPY": _history(step=0.1)}
    dates = [value.date() for value in histories["SPY"]["date"]]
    state = create_paper_trading_state(
        {"test_universe": ["AAA"]},
        benchmarks=("SPY",),
        config=USPaperTradingConfig(top_k=1, minimum_universe_coverage=1.0),
    )
    advance_paper_trading_state(state, histories, as_of=dates[100])
    advance_paper_trading_state(state, histories, as_of=dates[101])
    position = state["portfolios"]["test_universe"]["active_cycle"]["positions"][0]
    first_target = position["grid"]["take_profit_prices"][0]
    histories["AAA"].loc[102, "high"] = first_target + 0.01
    histories["AAA"].loc[102, "low"] = position["grid"]["stop_loss_price"] + 0.01

    advance_paper_trading_state(state, histories, as_of=dates[102])

    position = state["portfolios"]["test_universe"]["active_cycle"]["positions"][0]
    assert position["status"] == "open"
    assert position["remaining_quantity"] == pytest.approx(position["quantity"] / 2)
    assert position["fills"][0]["reason"] == "grid_take_profit"
    assert position["fills"][0]["fill_price"] == pytest.approx(first_target)
    assert state["portfolios"]["test_universe"]["event_log"][-1]["type"] == "grid_take_profit"

    advance_paper_trading_state(state, histories, as_of=dates[110])
    closed = state["portfolios"]["test_universe"]["closed_cycles"][0]
    assert closed["exit_model"] == "grid_v1"
    assert [fill["reason"] for fill in closed["trades"][0]["fills"]] == [
        "grid_take_profit",
        "time_exit",
    ]


def test_daily_grid_uses_stop_first_when_bar_touches_both_sides():
    histories = {"AAA": _history(step=0.1), "SPY": _history(step=0.1)}
    dates = [value.date() for value in histories["SPY"]["date"]]
    state = create_paper_trading_state(
        {"test_universe": ["AAA"]},
        benchmarks=("SPY",),
        config=USPaperTradingConfig(top_k=1, minimum_universe_coverage=1.0),
    )
    advance_paper_trading_state(state, histories, as_of=dates[100])
    advance_paper_trading_state(state, histories, as_of=dates[101])
    position = state["portfolios"]["test_universe"]["active_cycle"]["positions"][0]
    histories["AAA"].loc[102, "low"] = position["grid"]["stop_loss_price"] - 0.01
    histories["AAA"].loc[102, "high"] = position["grid"]["take_profit_prices"][-1] + 0.01

    advance_paper_trading_state(state, histories, as_of=dates[102])

    trade = state["portfolios"]["test_universe"]["closed_cycles"][0]["trades"][0]
    assert trade["exit_reason"] == "grid_stop_loss"
    assert [fill["reason"] for fill in trade["fills"]] == ["grid_stop_loss"]


def test_realtime_grid_quotes_are_incremental_and_idempotent():
    histories = {"AAA": _history(step=0.1), "SPY": _history(step=0.1)}
    dates = [value.date() for value in histories["SPY"]["date"]]
    state = create_paper_trading_state(
        {"test_universe": ["AAA"]},
        benchmarks=("SPY",),
        config=USPaperTradingConfig(top_k=1, minimum_universe_coverage=1.0),
    )
    advance_paper_trading_state(state, histories, as_of=dates[100])
    advance_paper_trading_state(state, histories, as_of=dates[101])
    position = state["portfolios"]["test_universe"]["active_cycle"]["positions"][0]
    observed_at = datetime(2025, 5, 22, 15, 0, tzinfo=timezone.utc)
    first_target = position["grid"]["take_profit_prices"][0]

    first = apply_realtime_grid_quotes(
        state,
        {"AAA": {"price": first_target + 0.25, "source": "Longbridge"}},
        observed_at=observed_at,
        market_date=dates[102],
    )
    repeated = apply_realtime_grid_quotes(
        state,
        {"AAA": {"price": first_target + 0.25, "source": "Longbridge"}},
        observed_at=observed_at,
        market_date=dates[102],
    )

    assert first["test_universe"][0]["type"] == "grid_take_profit"
    assert first["test_universe"][0]["source"] == "Longbridge"
    assert repeated == {}
    assert len(position["fills"]) == 1

    stop = apply_realtime_grid_quotes(
        state,
        {"AAA": {"price": position["grid"]["stop_loss_price"] - 0.5, "source": "Finnhub"}},
        observed_at=observed_at,
        market_date=dates[103],
    )
    assert stop["test_universe"][0]["type"] == "grid_stop_loss"
    assert state["portfolios"]["test_universe"]["active_cycle"]["status"] == "awaiting_settlement"
    notification = render_grid_event_notification(state, stop)
    assert "网格止损" in notification
    assert "Finnhub" in notification


def test_schema_one_state_upgrades_without_replaying_existing_open_days():
    histories = {"AAA": _history(), "SPY": _history(step=0.3)}
    dates = [value.date() for value in histories["SPY"]["date"]]
    state = create_paper_trading_state(
        {"test_universe": ["AAA"]}, benchmarks=("SPY",),
        config=USPaperTradingConfig(top_k=1, minimum_universe_coverage=1.0),
    )
    advance_paper_trading_state(state, histories, as_of=dates[100])
    advance_paper_trading_state(state, histories, as_of=dates[101])
    position = state["portfolios"]["test_universe"]["active_cycle"]["positions"][0]
    for key in ("grid", "remaining_quantity", "realized_value", "fills", "status", "last_evaluated_date"):
        position.pop(key, None)
    state["schema_version"] = 1
    for key in ("grid_step_pct", "grid_take_profit_levels", "grid_stop_loss_levels"):
        state["config"].pop(key, None)

    changed = upgrade_paper_trading_state(state)

    assert changed is True
    assert state["schema_version"] == 2
    assert position["grid"]["take_profit_prices"]
    assert position["last_evaluated_date"] == dates[101].isoformat()


def test_live_validation_does_not_count_legacy_fixed_hold_cycles():
    state = create_paper_trading_state(
        {"test_universe": ["AAA"]},
        benchmarks=("SPY",),
        config=USPaperTradingConfig(
            top_k=1,
            minimum_universe_coverage=1.0,
            minimum_completed_cycles=1,
        ),
    )
    portfolio = state["portfolios"]["test_universe"]
    portfolio["closed_cycles"] = [{"exit_model": "fixed_hold_v1"}]
    portfolio["snapshots"] = [{
        "strategy_total_return_pct": 10.0,
        "benchmark_total_return_pct": {"SPY": 1.0},
    }]

    validation = evaluate_live_validation(state)

    assert validation["diagnostics"]["test_universe"]["completed_cycles"] == 0
    assert validation["effective"] is False


def test_live_validation_reports_outperformance_progress_before_profitability():
    state = create_paper_trading_state(
        {"test_universe": ["AAA"]},
        benchmarks=("SPY", "QQQ"),
        config=USPaperTradingConfig(
            top_k=1,
            minimum_universe_coverage=1.0,
            minimum_completed_cycles=1,
        ),
    )
    portfolio = state["portfolios"]["test_universe"]
    portfolio["closed_cycles"] = [{"exit_model": "grid_v1"}]
    portfolio["snapshots"] = [{
        "strategy_total_return_pct": -0.5,
        "benchmark_total_return_pct": {"SPY": -1.0, "QQQ": 0.5},
    }]

    validation = evaluate_live_validation(state)

    diagnostic = validation["diagnostics"]["test_universe"]
    assert diagnostic["positive_excess_benchmarks"] == 1
    assert diagnostic["effective"] is False
    assert validation["effective"] is False


def test_legacy_pending_signal_is_upgraded_on_idempotent_run():
    histories = {
        "AAA": _history(step=0.7),
        "BBB": _history(step=0.5),
        "SPY": _history(step=0.3),
    }
    state = create_paper_trading_state(
        {"test_universe": ["AAA", "BBB"]},
        benchmarks=("SPY",),
        config=USPaperTradingConfig(top_k=2, minimum_universe_coverage=1.0),
    )
    as_of = histories["SPY"].iloc[100]["date"].date()
    advance_paper_trading_state(state, histories, as_of=as_of)
    cycle = state["portfolios"]["test_universe"]["active_cycle"]
    cycle["selected"] = [
        {
            "code": item["code"],
            "screen_score": item["screen_score"],
            "signal_close": item["signal_close"],
        }
        for item in cycle["selected"]
    ]

    advance_paper_trading_state(state, histories, as_of=as_of)

    selected = state["portfolios"]["test_universe"]["active_cycle"]["selected"]
    assert all(item["scorecard_version"] == "us_evidence_v2" for item in selected)
    assert all(item["selection_thesis"] for item in selected)
    assert [event["type"] for event in state["last_events"]["test_universe"]] == [
        "scorecard_upgraded",
        "no_change",
    ]


def test_candidate_ranking_fails_closed_on_incomplete_universe():
    history = _history()
    signal_date = history.iloc[100]["date"].date()

    with pytest.raises(ValueError, match="paper-trading universe coverage"):
        rank_candidates_on_date(
            "us_quality_momentum",
            {"AAA": history, "MISSING": pd.DataFrame()},
            signal_date,
            top_k=1,
            lookback_days=140,
            minimum_universe_coverage=0.95,
        )


def test_candidate_evidence_uses_only_observed_scorecard_factors():
    histories = {
        "AAA": _history(step=0.7),
        "BBB": _history(step=0.5),
    }
    signal_date = histories["AAA"].iloc[100]["date"].date()

    ranked = rank_candidates_on_date(
        "us_quality_momentum",
        histories,
        signal_date,
        top_k=2,
        lookback_days=140,
        minimum_universe_coverage=1.0,
    )

    candidate = ranked["selected"][0]
    assert candidate["scorecard_version"] == "us_evidence_v2"
    assert set(candidate["factor_scores"]) == {
        "trend_confirmation",
        "momentum",
        "risk_control",
        "liquidity",
        "relative_strength",
        "data_quality",
    }
    assert sum(candidate["factor_weights"].values()) == pytest.approx(1.0)
    reconstructed_score = sum(
        float(candidate["factor_scores"][factor]) * weight
        for factor, weight in candidate["factor_weights"].items()
    )
    assert candidate["screen_score"] == pytest.approx(reconstructed_score, abs=0.001)
    assert "value" not in candidate["factor_scores"]
    assert "size" not in candidate["factor_scores"]
    assert any("未参与评分" in item for item in candidate["data_gaps"])
    assert any("MA20" in item for item in candidate["reasons_pass"])
    assert "不是上涨概率" in candidate["score_explanation"]
    assert candidate["data_sources"] == ["Yahoo Finance 复权 OHLCV 日线（收盘后信号）"]


def test_candidate_evidence_surfaces_near_limit_risks():
    strategy = load_all_strategies(Path("src/services/screening/strategies"))["us_quality_momentum"]
    row = {
        "price": 110.0,
        "ma20": 100.0,
        "price_above_ma20": True,
        "ma_bullish": True,
        "macd_status": "bullish",
        "rsi_status": "overbought",
        "signal_score": 75.0,
        "change_pct": 6.5,
        "change_60d": 38.0,
        "volatility_20d_pct": 40.0,
        "max_drawdown_20d_pct": -13.0,
        "atr_20_pct": 5.5,
        "daily_quality_score": 100.0,
        "daily_quality_flags": "",
        "pe_ratio": float("nan"),
        "pb_ratio": float("nan"),
        "total_mv": float("nan"),
        "factor_trend_confirmation_score": 90.0,
        "factor_momentum_score": 80.0,
        "factor_risk_control_score": 35.0,
        "factor_liquidity_score": 80.0,
        "factor_relative_strength_score": 90.0,
        "factor_data_quality_score": 100.0,
    }

    evidence = build_us_candidate_evidence(row, strategy.screening)

    assert any("追高警戒线" in item for item in evidence["risk_flags"])
    assert any("overbought" in item for item in evidence["risk_flags"])
    assert any("接近上限" in item for item in evidence["risk_flags"])
    assert any("动量过热" in item for item in evidence["reasons_watch"])


def test_news_adjustment_is_bounded_and_auditable():
    candidate = {
        "code": "AAA",
        "screen_score": 80.0,
        "data_sources": [],
        "data_gaps": [],
        "risk_flags": [],
    }

    apply_intelligence_adjustment(candidate, {
        "status": "available",
        "score_adjustment": 3.5,
        "summary": "财报与评级证据可用。",
        "risk_flags": [],
        "hard_exclusion": False,
        "items": [{"title": "AAA raises guidance", "url": "https://example.com/aaa"}],
    })

    assert candidate["technical_screen_score"] == 80.0
    assert candidate["intelligence_adjustment"] == 3.5
    assert candidate["screen_score"] == 83.5
    assert candidate["scorecard_version"] == INTELLIGENCE_SCORECARD_VERSION
    assert candidate["news_intelligence"]["items"][0]["url"] == "https://example.com/aaa"


def test_unavailable_news_does_not_create_a_neutral_score():
    candidate = {
        "code": "AAA",
        "screen_score": 80.0,
        "data_sources": [],
        "data_gaps": [],
        "risk_flags": [],
    }

    apply_intelligence_adjustment(
        candidate,
        unavailable_intelligence("AAA", date(2026, 8, 13), "offline"),
    )

    assert candidate["screen_score"] == 80.0
    assert candidate["intelligence_adjustment"] == 0.0
    assert any("未参与" in item for item in candidate["data_gaps"])


def test_severe_risk_exclusion_requires_distinct_risk_types():
    repeated_regulatory = _summarize_items(
        "AAA",
        date(2026, 8, 13),
        [
            {"category": "company_news", "title": "AAA faces SEC charges"},
            {"category": "company_news", "title": "AAA faces regulatory probe"},
        ],
    )
    distinct_risks = _summarize_items(
        "AAA",
        date(2026, 8, 13),
        [
            {"category": "company_news", "title": "AAA faces SEC charges"},
            {"category": "earnings", "title": "AAA reports accounting restatement"},
        ],
    )

    assert repeated_regulatory["hard_exclusion"] is False
    assert distinct_risks["hard_exclusion"] is True


def test_news_source_weights_are_applied_from_strategy_configuration():
    provider = YahooUSNewsIntelligenceProvider(source_weights={"company_news": 0.0})
    result = _summarize_items(
        "AAA",
        date(2026, 8, 13),
        [{"category": "company_news", "title": "AAA raises guidance"}],
        source_weights=provider.source_weights,
    )

    assert result["score_adjustment"] == 0.0


def test_news_items_keep_auditable_summary_and_impact_annotations():
    from src.services.screening.us_news_intelligence import _normalize_yahoo_news

    items = _normalize_yahoo_news([{
        "content": {
            "title": "AAA raises guidance",
            "description": "Management raised full-year outlook.",
            "pubDate": "2026-08-12T12:00:00Z",
            "provider": {"displayName": "Example News"},
            "canonicalUrl": {"url": "https://example.com/aaa"},
        },
    }], as_of=date(2026, 8, 13))

    assert items[0]["summary"] == "Management raised full-year outlook."
    assert items[0]["sentiment"] == "偏正面"
    assert "guidance_raise" in items[0]["tags"]
    assert items[0]["impact_channels"] == ["盈利预期"]


def test_news_normalization_rejects_explicitly_unrelated_tickers():
    from src.services.screening.us_news_intelligence import _normalize_yahoo_news

    items = _normalize_yahoo_news([
        {"title": "ABBV stock update", "relatedTickers": ["ABBV"]},
        {"title": "LLY raises guidance", "relatedTickers": ["LLY"]},
        {"title": "LLY market story"},
    ], as_of=date(2026, 8, 13), code="LLY")

    assert [item["title"] for item in items] == ["LLY raises guidance", "LLY market story"]


def test_archive_snapshot_does_not_overwrite_existing_market_date(tmp_path):
    state_path = tmp_path / "state.json"
    report_path = tmp_path / "latest.md"
    state = {"latest_market_date": "2026-08-13", "value": 1}
    first = _write_archive_snapshot(state_path, report_path, state, "first")
    _write_archive_snapshot(state_path, report_path, {**state, "value": 2}, "second")

    assert first.joinpath("state.json").read_text() == '{\n  "latest_market_date": "2026-08-13",\n  "value": 1\n}'
    assert first.joinpath("latest.md").read_text() == "first"


def test_analyst_action_codes_are_normalized_for_scoring():
    class _Frame:
        def reset_index(self):
            return self

        def to_dict(self, mode):
            assert mode == "records"
            return [{
                "GradeDate": date(2026, 8, 12),
                "Firm": "Example Research",
                "Action": "up",
                "FromGrade": "Hold",
                "ToGrade": "Buy",
            }]

    items = _normalize_analyst_actions(_Frame(), as_of=date(2026, 8, 13))
    assert "analyst upgrade" in items[0]["title"]


class _FakeNewsProvider:
    def collect(self, tickers, *, as_of):
        return {
            ticker: {
                "code": ticker,
                "status": "available",
                "summary": "可审计资讯 1 条，资讯调整 +1.0 分。",
                "items": [{
                    "category": "earnings",
                    "title": f"{ticker} raises guidance",
                    "url": f"https://example.com/{ticker}",
                    "source": "test",
                }],
                "score_adjustment": 1.0,
                "risk_flags": [],
                "hard_exclusion": False,
            }
            for ticker in tickers
        }

    def market_digest(self, *, as_of):
        return {
            "code": "US_MARKET",
            "status": "available",
            "summary": "市场摘要可用。",
            "items": [],
            "score_adjustment": 0.0,
            "risk_flags": [],
            "hard_exclusion": False,
        }


def test_news_strategy_preserves_open_cycle_and_starts_with_next_cycle():
    histories = {"AAA": _history(step=0.1), "SPY": _history(step=0.1)}
    dates = [value.date() for value in histories["SPY"]["date"]]
    state = create_paper_trading_state(
        {"test_universe": ["AAA"]},
        benchmarks=("SPY",),
        config=USPaperTradingConfig(top_k=1, minimum_universe_coverage=1.0),
    )
    advance_paper_trading_state(state, histories, as_of=dates[100])
    advance_paper_trading_state(state, histories, as_of=dates[101])
    cycle = state["portfolios"]["test_universe"]["active_cycle"]
    original_score = cycle["selected"][0]["screen_score"]
    original_entry = cycle["positions"][0]["entry_open"]

    advance_paper_trading_state(
        state,
        histories,
        as_of=dates[102],
        news_provider=_FakeNewsProvider(),
    )

    cycle = state["portfolios"]["test_universe"]["active_cycle"]
    assert cycle["selected"][0]["screen_score"] == original_score
    assert cycle["positions"][0]["entry_open"] == original_entry
    assert cycle["latest_news_intelligence"]["affects_current_cycle"] is False

    advance_paper_trading_state(
        state,
        histories,
        as_of=dates[110],
        news_provider=_FakeNewsProvider(),
    )
    next_cycle = state["portfolios"]["test_universe"]["active_cycle"]
    assert next_cycle["status"] == "pending"
    assert next_cycle["strategy_version"] == "2.1"
    assert next_cycle["scorecard_version"] == INTELLIGENCE_SCORECARD_VERSION
    assert next_cycle["selected"][0]["intelligence_adjustment"] == 1.0


def test_report_keeps_live_validation_and_all_benchmarks_visible():
    histories = {
        "AAA": _history(),
        "SPY": _history(step=0.3),
        "QQQ": _history(step=0.4),
        "IWM": _history(step=0.2),
        "DIA": _history(step=0.25),
        "RSP": _history(step=0.28),
    }
    state = create_paper_trading_state(
        {"test_universe": ["AAA"]},
        benchmarks=("SPY", "QQQ", "IWM", "DIA", "RSP"),
        config=USPaperTradingConfig(top_k=1, minimum_universe_coverage=1.0),
    )
    advance_paper_trading_state(
        state,
        histories,
        as_of=histories["SPY"].iloc[100]["date"].date(),
    )

    report = render_paper_trading_report(state)

    assert "证据不足或未通过" in report
    assert "不连接券商或真实下单" in report
    assert "候选与选股理由" in report
    assert "入选证据" in report
    assert "分数说明" in report
    assert "网格风控与成交" in report
    assert "每日重点资讯、财报与研报" in report
    assert "每格 3.0%" in report
    for benchmark in ("SPY", "QQQ", "IWM", "DIA", "RSP"):
        assert benchmark in report


@pytest.mark.parametrize("ticker_level", [0, 1])
def test_split_yfinance_download_supports_both_multiindex_layouts(ticker_level):
    dates = pd.bdate_range("2025-01-01", periods=2, name="Date")
    tuples = []
    values = []
    for ticker, base in (("AAA", 100.0), ("BBB", 200.0)):
        row_values = {"Open": base, "High": base + 2, "Low": base - 2, "Close": base + 1, "Volume": 1_000_000}
        for field, value in row_values.items():
            tuples.append((ticker, field) if ticker_level == 0 else (field, ticker))
            values.append(value)
    raw = pd.DataFrame([values, values], index=dates, columns=pd.MultiIndex.from_tuples(tuples))

    result = split_yfinance_download(raw, ["AAA", "BBB"])

    assert set(result) == {"AAA", "BBB"}
    assert result["AAA"].iloc[-1]["close"] == 101.0
    assert result["BBB"].iloc[-1]["close"] == 201.0


def test_scheduled_runner_skips_write_and_notification_without_new_session(monkeypatch):
    state = {
        "latest_market_date": "2026-08-26",
        "config": {"news_intelligence_enabled": False},
    }
    writes = []
    notifications = []

    monkeypatch.setattr(paper_trading_runner, "_load_or_create_state", lambda *args, **kwargs: state)
    monkeypatch.setattr(paper_trading_runner, "download_yahoo_histories", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        paper_trading_runner,
        "advance_paper_trading_state",
        lambda *args, **kwargs: state,
    )
    monkeypatch.setattr(
        paper_trading_runner,
        "_write_atomic",
        lambda *args, **kwargs: writes.append(args),
    )
    monkeypatch.setattr(
        paper_trading_runner,
        "_send_notification",
        lambda *args, **kwargs: notifications.append(args),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["run_us_paper_trading.py", "--notify", "--skip-unchanged-session"],
    )

    assert paper_trading_runner.main() == 0
    assert writes == []
    assert notifications == []


def test_paper_trading_workflow_persists_state_and_publishes_daily_report():
    workflow_path = Path(".github/workflows/us-paper-trading.yml")
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    triggers = workflow.get(True, workflow.get("on"))
    rendered = workflow_path.read_text(encoding="utf-8")

    assert triggers["schedule"] == [
        {"cron": "17 22 * * 1-5"},
        {"cron": "17 0 * * 2-6"},
        {"cron": "17 3 * * 2-6"},
    ]
    assert workflow["permissions"] == {"contents": "write", "issues": "write"}
    assert "python scripts/run_us_paper_trading.py --notify --skip-unchanged-session" in rendered
    assert "steps.advance.outputs.changed == 'true'" in rendered
    assert "sha256sum data/us_paper_trading/state.json" in rendered
    assert "US_GRID_STEP_PCT" in rendered
    assert "paper-trading-state" in rendered
    assert "美股模拟交易日报" in rendered
    assert "continue-on-error: true" in rendered
    assert "actions/upload-artifact@v6" in rendered
    assert "actions/github-script@v8" in rendered


def test_paper_trading_dashboard_deploys_after_successful_daily_run():
    workflow_path = Path(".github/workflows/deploy-paper-trading-pages.yml")
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    triggers = workflow.get(True, workflow.get("on"))
    rendered = workflow_path.read_text(encoding="utf-8")

    assert triggers["workflow_run"]["workflows"] == ["US Paper Trading Daily"]
    assert workflow["permissions"] == {"contents": "read", "pages": "write", "id-token": "write"}
    assert "paper-trading-state" in rendered
    assert "VITE_PAPER_TRADING_DASHBOARD: 'true'" in rendered
    assert "enablement: true" in rendered
    assert "actions/deploy-pages@v4" in rendered


def test_realtime_grid_monitor_is_serialized_with_daily_state_updates():
    workflow_path = Path(".github/workflows/us-grid-monitor.yml")
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    triggers = workflow.get(True, workflow.get("on"))
    rendered = workflow_path.read_text(encoding="utf-8")

    assert triggers["schedule"] == [{"cron": "*/15 13-21 * * 1-5"}]
    assert workflow["permissions"] == {"contents": "write", "actions": "write"}
    assert workflow["concurrency"]["group"] == "us-paper-trading-state"
    assert "python scripts/run_us_grid_monitor.py --notify" in rendered
    assert "LONGBRIDGE_OAUTH_TOKEN_CACHE_B64" in rendered
    assert "FINNHUB_API_KEY" in rendered
    assert "git diff --cached --quiet" in rendered
    assert "gh workflow run deploy-paper-trading-pages.yml --ref main" in rendered


def test_full_market_waits_for_next_cycle_and_retries_without_duplicate_fills(monkeypatch):
    from scripts.us_listed_stocks import ordinary_symbols

    header = "Nasdaq Traded|Symbol|Security Name|ETF|Test Issue"
    directory = "\n".join([header, "Y|NEW.A|New Class A Common Stock|N|N",
                           "Y|ETF|Fund Common Shares|Y|N", "Y|WAR|Warrants on Common Stock|N|N",
                           "Y|TEST|Test Common Shares|N|Y", "File Creation Time: test"])
    assert ordinary_symbols(directory) == ["NEW-A"]
    with pytest.raises(ValueError, match="不完整"):
        ordinary_symbols(header)

    full = paper_trading_runner.FULL_MARKET
    config = USPaperTradingConfig(top_k=1, holding_days=2)
    old_bars = {"AAA": _history(), "SPY": _history(step=0.3)}
    added_bars = {"NEW-A": _history(step=0.2)}
    dates = [t.date() for t in old_bars["SPY"]["date"]]
    state = create_paper_trading_state({"large_cap_22": ["AAA"]}, benchmarks=["SPY"], config=config)
    advance_paper_trading_state(state, old_bars, as_of=dates[100])

    def rank(session):
        return paper_trading_runner.rank_full_market(session, old_bars, config, 420, 25)

    monkeypatch.setattr(paper_trading_runner, "current_symbols", lambda session: ordinary_symbols(directory))
    monkeypatch.setattr(paper_trading_runner, "download_yahoo_histories", lambda *args, **kwargs: {})
    advance_paper_trading_state(state, old_bars, as_of=dates[101], market_ranker=rank)
    assert full not in state["portfolios"]
    advance_paper_trading_state(state, old_bars, as_of=dates[102], market_ranker=rank)
    market = state["portfolios"][full]
    assert market["scan_error"] and market["active_cycle"] is None
    assert len(state["portfolios"]["large_cap_22"]["closed_cycles"]) == 1

    monkeypatch.setattr(paper_trading_runner, "download_yahoo_histories", lambda *args, **kwargs: added_bars)
    advance_paper_trading_state(state, old_bars, as_of=dates[102], market_ranker=rank)
    assert "scan_error" not in market
    assert market["active_cycle"]["selected"][0]["code"] == "NEW-A"
    assert len(market["snapshots"]) == 1
    assert market["snapshots"][0]["strategy_equity"] == config.initial_capital
    advance_paper_trading_state(state, old_bars, as_of=dates[103], market_ranker=rank)
    assert market["pricing_error"] and market["trade_log"] == []
    assert market["last_processed_date"] == dates[102].isoformat()
    for index in (103, 103, 104):
        advance_paper_trading_state(state, {**old_bars, **added_bars}, as_of=dates[index], market_ranker=rank)
    assert len(market["closed_cycles"]) == 1
    assert "pricing_error" not in market
    assert market["closed_cycles"][0]["universe"] == ["NEW-A"]
    assert [t["side"] for t in market["trade_log"]] == ["buy", "sell"]
    assert market["active_cycle"]["signal_date"] == dates[104].isoformat()


def test_full_market_historical_directory_and_empty_selection(monkeypatch):
    from scripts.us_listed_stocks import current_symbols

    with pytest.raises(ValueError, match="不适用于该信号日"):
        current_symbols(date(2025, 1, 6))

    histories = {"AAA": _history(), "SPY": _history()}
    dates = [t.date() for t in histories["SPY"]["date"]]
    state = create_paper_trading_state({"large_cap_22": ["AAA"]}, benchmarks=["SPY"])
    illiquid = _history()
    illiquid["volume"] = 1
    monkeypatch.setattr(paper_trading_runner, "current_symbols", lambda _: ["NEW"])
    monkeypatch.setattr(paper_trading_runner, "download_yahoo_histories", lambda *a, **kw: {"NEW": illiquid})

    def rank(session):
        return paper_trading_runner.rank_full_market(session, {}, USPaperTradingConfig(), 420, 25)

    advance_paper_trading_state(state, histories, as_of=dates[100], market_ranker=rank)
    market = state["portfolios"][paper_trading_runner.FULL_MARKET]
    assert market["active_cycle"] is None
    assert state["last_events"][paper_trading_runner.FULL_MARKET][0]["type"] == "no_candidates"
    monkeypatch.setattr(paper_trading_runner, "download_yahoo_histories", lambda *a, **kw: {"NEW": _history()})
    advance_paper_trading_state(state, histories, as_of=dates[101], market_ranker=rank)
    assert market["active_cycle"]["selected"][0]["code"] == "NEW"
    assert market["snapshots"][-1]["strategy_equity"] == 100000
