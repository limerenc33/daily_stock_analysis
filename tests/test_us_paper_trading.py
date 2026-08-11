from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.run_us_paper_trading import split_yfinance_download
from src.services.screening.us_paper_trading import (
    USPaperTradingConfig,
    advance_paper_trading_state,
    create_paper_trading_state,
    rank_candidates_on_date,
    render_paper_trading_report,
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


def test_paper_trading_workflow_persists_state_and_publishes_daily_report():
    workflow_path = Path(".github/workflows/us-paper-trading.yml")
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    triggers = workflow.get(True, workflow.get("on"))
    rendered = workflow_path.read_text(encoding="utf-8")

    assert triggers["schedule"] == [{"cron": "30 22 * * 1-5"}]
    assert workflow["permissions"] == {"contents": "write", "issues": "write"}
    assert "python scripts/run_us_paper_trading.py --notify" in rendered
    assert "paper-trading-state" in rendered
    assert "美股模拟交易日报" in rendered
