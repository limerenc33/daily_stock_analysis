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
from src.services.screening.strategy import load_all_strategies
from src.services.screening.us_candidate_evidence import build_us_candidate_evidence


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
