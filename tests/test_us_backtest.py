from datetime import date
from io import BytesIO
import json

import pandas as pd
import pytest

from src.services.screening.us_backtest import (
    USBacktestConfig,
    backtest_strategy,
    bootstrap_excess_ci,
    build_feature_cache,
    normalize_price_history,
    simulate_trade,
)
from scripts import run_us_strategy_backtest as backtest_cli
from scripts.run_us_strategy_backtest import cache_covers_window, required_history_start


def _trend_history(start: str = "2020-01-01", points: int = 180) -> pd.DataFrame:
    dates = pd.bdate_range(start=start, periods=points)
    close = pd.Series(range(100, 100 + points), dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1_000_000,
        }
    )


def test_history_download_window_and_cache_require_warmup_coverage():
    requested_start = date(2015, 1, 1)
    requested_end = date(2025, 12, 31)

    assert required_history_start(requested_start, 140) == date(2014, 3, 27)
    complete = pd.DataFrame({"date": [date(2014, 3, 27), date(2025, 12, 29)]})
    missing_warmup = pd.DataFrame({"date": [date(2015, 1, 2), date(2025, 12, 29)]})
    stale_end = pd.DataFrame({"date": [date(2014, 3, 27), date(2025, 12, 1)]})
    late_inception = pd.DataFrame({"date": [date(2018, 6, 19), date(2025, 12, 29)]})

    assert cache_covers_window(complete, start=required_history_start(requested_start, 140), end=requested_end)
    assert not cache_covers_window(missing_warmup, start=required_history_start(requested_start, 140), end=requested_end)
    assert not cache_covers_window(stale_end, start=required_history_start(requested_start, 140), end=requested_end)
    assert cache_covers_window(
        late_inception,
        start=required_history_start(requested_start, 140),
        end=requested_end,
        earliest_available=date(2018, 6, 18),
    )
    assert not cache_covers_window(
        late_inception,
        start=required_history_start(requested_start, 140),
        end=requested_end,
    )


def test_cached_history_preserves_original_provider(tmp_path):
    path = tmp_path / "TEST.csv"
    history = _trend_history(points=5)
    backtest_cli._write_history_cache(
        path,
        history,
        ticker="TEST",
        provider="yfinance",
        start=date(2020, 1, 1),
        end=date(2020, 1, 7),
    )
    metadata = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))

    cached, source = backtest_cli._download_or_cache(
        "TEST",
        start=date(2020, 1, 1),
        end=date(2020, 1, 7),
        cache_dir=tmp_path,
        source="auto",
    )

    assert len(cached) == 5
    assert source == "cache:yfinance"
    assert metadata["requested_start"] == "2020-01-01"
    assert metadata["requested_end"] == "2020-01-07"
    assert metadata["row_count"] == 5

    metadata["row_count"] = 99
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    _, untrusted_source = backtest_cli._download_or_cache(
        "TEST",
        start=date(2020, 1, 1),
        end=date(2020, 1, 7),
        cache_dir=tmp_path,
        source="auto",
    )
    assert untrusted_source == "cache:unknown"


def test_stooq_html_challenge_is_not_parsed_as_price_data(monkeypatch):
    monkeypatch.setattr(
        backtest_cli,
        "urlopen",
        lambda *_args, **_kwargs: BytesIO(b"<html><body>verification required</body></html>"),
    )

    with pytest.raises(RuntimeError, match="HTML instead of daily-bar CSV"):
        backtest_cli._download_stooq(
            "TEST",
            start=date(2020, 1, 1),
            end=date(2020, 1, 7),
        )


def test_normalize_price_history_supports_yfinance_price_ticker_columns():
    dates = pd.bdate_range("2025-01-01", periods=2, name="Date")
    columns = pd.MultiIndex.from_tuples(
        [(field, "SPY") for field in ("Close", "High", "Low", "Open", "Volume")],
        names=("Price", "Ticker"),
    )
    raw = pd.DataFrame(
        [[101.0, 102.0, 99.0, 100.0, 1_000_000], [102.0, 103.0, 100.0, 101.0, 1_100_000]],
        index=dates,
        columns=columns,
    )

    normalized = normalize_price_history(raw)

    assert list(normalized.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert len(normalized) == 2
    assert normalized.iloc[-1]["close"] == 102.0


def test_simulate_trade_enters_next_session_and_applies_costs():
    history = normalize_price_history(
        pd.DataFrame(
            {
                "date": pd.bdate_range("2025-01-01", periods=4),
                "open": [100.0, 110.0, 120.0, 130.0],
                "high": [101.0, 111.0, 121.0, 131.0],
                "low": [99.0, 109.0, 119.0, 129.0],
                "close": [100.0, 112.0, 125.0, 135.0],
                "volume": [1_000_000] * 4,
            }
        )
    )
    trade = simulate_trade(
        history,
        signal_index=0,
        holding_days=2,
        config=USBacktestConfig(transaction_cost_bps=0.0, slippage_bps=0.0),
    )

    assert trade is not None
    assert trade["entry_date"] == str(history.iloc[1]["date"])
    assert trade["exit_date"] == str(history.iloc[2]["date"])
    assert trade["entry_price"] == 110.0
    assert trade["exit_price"] == 125.0
    assert round(float(trade["net_return_pct"]), 6) == round((125.0 / 110.0 - 1.0) * 100.0, 6)


def test_backtest_strategy_replays_historical_features_and_reports_omitted_fields():
    history = _trend_history()
    result = backtest_strategy(
        "us_quality_momentum",
        {"TEST": history},
        history,
        config=USBacktestConfig(
            start=date(2020, 5, 1),
            end=date(2020, 9, 1),
            top_k=1,
            holding_days=5,
            transaction_cost_bps=0.0,
            slippage_bps=0.0,
        ),
    )

    assert result["strategy"] == "us_quality_momentum"
    assert result["periods"]
    assert 0 < result["metrics"]["trade_count"] <= result["metrics"]["period_count"]
    assert "market_cap_min" in result["omitted_point_in_time_filters"]


def test_feature_cache_preserves_backtest_results():
    history = _trend_history()
    config = USBacktestConfig(
        start=date(2020, 5, 1),
        end=date(2020, 9, 1),
        top_k=1,
        holding_days=5,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )
    uncached = backtest_strategy("us_quality_momentum", {"TEST": history}, history, config=config)
    cached = backtest_strategy(
        "us_quality_momentum",
        {"TEST": history},
        history,
        config=config,
        feature_cache=build_feature_cache({"TEST": history}, lookback_days=config.lookback_days),
    )

    assert cached["metrics"] == uncached["metrics"]
    assert cached["periods"] == uncached["periods"]


def test_feature_cache_does_not_use_future_bars():
    history = _trend_history(points=180)
    target_date = history.iloc[100]["date"].date()
    baseline = build_feature_cache({"TEST": history})["TEST"][target_date]

    future_changed = history.copy()
    future_changed.loc[101:, ["open", "high", "low", "close"]] = 10_000.0
    changed = build_feature_cache({"TEST": future_changed})["TEST"][target_date]

    for key in (
        "price", "change_60d", "ma5", "ma20", "ma60", "prev_high_20d",
        "breakout_20d_pct", "volatility_20d_pct", "daily_quality_score",
    ):
        assert changed[key] == baseline[key]


def test_bootstrap_excess_ci_is_deterministic_and_directional():
    result = {
        "periods": [
            {"strategy_return_pct": value, "benchmark_return_pct": 0.0}
            for value in (1.0, 2.0, 3.0, 4.0)
        ]
    }

    first = bootstrap_excess_ci(result, samples=500, seed=7)
    second = bootstrap_excess_ci(result, samples=500, seed=7)

    assert first == second
    assert first["ci95_low_pct"] > 0.0
    assert first["probability_mean_excess_positive"] == 1.0


def test_backtest_keeps_cash_periods_when_strategy_has_no_candidates():
    history = _trend_history()
    result = backtest_strategy(
        "us_breakout_continuation",
        {"TEST": history},
        history,
        config=USBacktestConfig(
            start=date(2020, 5, 1),
            end=date(2020, 9, 1),
            top_k=1,
            holding_days=5,
            transaction_cost_bps=0.0,
            slippage_bps=0.0,
        ),
    )

    assert result["periods"]
    assert result["metrics"]["trade_count"] == 0
    assert result["metrics"]["total_return_pct"] == 0.0
    assert result["metrics"]["benchmark_return_pct"] > 0.0


def test_point_in_time_backtest_fails_closed_on_incomplete_universe():
    history = _trend_history()

    with pytest.raises(ValueError, match="point-in-time universe coverage"):
        backtest_strategy(
            "us_quality_momentum",
            {"AAA": history},
            history,
            config=USBacktestConfig(
                start=date(2020, 5, 1),
                end=date(2020, 9, 1),
                top_k=1,
                holding_days=5,
                minimum_universe_coverage=0.75,
            ),
            universe_by_date=lambda _: ("AAA", "MISSING"),
        )
