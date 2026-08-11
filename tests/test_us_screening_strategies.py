from pathlib import Path

import pandas as pd
import pytest

from src.services.screening.config import Config
from src.services.screening.filter import apply_hard_filters, without_daily_filters
from src.services.screening.snapshot_us import fetch_us_universe
from src.services.screening.strategy import load_all_strategies


REPO_ROOT = Path(__file__).resolve().parents[1]
STRATEGIES_DIR = REPO_ROOT / "src" / "services" / "screening" / "strategies"
US_STRATEGIES = {
    "us_quality_momentum",
    "us_breakout_continuation",
    "us_low_volatility_quality",
}


def test_builtin_us_strategies_are_market_scoped_and_loadable():
    strategies = load_all_strategies(STRATEGIES_DIR)

    assert US_STRATEGIES.issubset(strategies)
    for name in US_STRATEGIES:
        strategy = strategies[name]
        assert strategy.screening.market_scope == ["us"]
        assert strategy.screening.hard_filters.pe_ttm_min is None
        assert strategy.screening.hard_filters.pe_ttm_max is None
        assert strategy.screening.hard_filters.pb_min is None
        assert strategy.screening.hard_filters.pb_max is None


def test_us_strategy_does_not_reject_missing_valuation_fields():
    strategy = load_all_strategies(STRATEGIES_DIR)["us_quality_momentum"]
    snapshot = pd.DataFrame(
        [
            {
                "code": "MSFT",
                "name": "Microsoft",
                "price": 400.0,
                "amount": 2_000_000_000.0,
                "total_mv": 3_000_000_000_000.0,
                "change_pct": 1.0,
                "pe_ratio": None,
                "pb_ratio": None,
            }
        ]
    )

    filtered = apply_hard_filters(
        snapshot,
        without_daily_filters(strategy.screening.hard_filters),
    )

    assert filtered["code"].tolist() == ["MSFT"]


def test_us_universe_auto_prefers_explicit_environment(monkeypatch):
    monkeypatch.setenv("SCREENING_US_TICKERS", "AAPL, MSFT, BRK.B")

    def fail_sp500():
        raise AssertionError("auto must not scrape S&P 500 when env is configured")

    monkeypatch.setattr(
        "src.services.screening.snapshot_us._fetch_sp500_tickers",
        fail_sp500,
    )

    assert fetch_us_universe("auto") == ["AAPL", "MSFT", "BRK.B"]


def test_us_strategy_rejects_cn_market_before_fetching_data():
    from src.services.screening import pipeline

    config = Config(strategies_dir=STRATEGIES_DIR)
    with pytest.raises(ValueError, match="does not support market 'cn'"):
        pipeline.screen(
            "us_quality_momentum",
            market="cn",
            use_llm=False,
            config=config,
        )


def test_us_pipeline_routes_auto_daily_source_to_yfinance(monkeypatch):
    from src.services.screening import pipeline

    snapshot = pd.DataFrame(
        [
            {
                "code": "MSFT",
                "name": "Microsoft",
                "price": 400.0,
                "amount": 2_000_000_000.0,
                "total_mv": 3_000_000_000_000.0,
                "change_pct": 1.0,
                "pe_ratio": None,
                "pb_ratio": None,
            }
        ]
    )
    captured = {}

    def fake_enrich(df, **kwargs):
        captured["source"] = kwargs["source"]
        result = df.copy()
        for key, value in {
            "change_60d": 12.0,
            "price_above_ma20": True,
            "signal_score": 70.0,
            "macd_status": "bullish",
            "volatility_20d_pct": 20.0,
            "max_drawdown_20d_pct": -5.0,
            "atr_20_pct": 2.0,
        }.items():
            result[key] = value
        result.attrs.update(
            {
                "daily_errors": [],
                "daily_success_count": 1,
                "daily_source_counts": {"yfinance": 1},
                "daily_quality_flag_counts": {},
                "daily_source_order_notes": [],
                "daily_source_health": {},
            }
        )
        return result

    monkeypatch.setattr(pipeline, "fetch_snapshot_with_fallback", lambda *a, **k: snapshot.copy())
    monkeypatch.setattr(pipeline, "enrich_daily_features", fake_enrich)

    config = Config(
        strategies_dir=STRATEGIES_DIR,
        daily_enrich_enabled=True,
        daily_enrich_max_candidates=1,
        post_analyzers=[],
        risk_enabled=False,
        portfolio_diversity_enabled=False,
    )
    result = pipeline.screen(
        "us_quality_momentum",
        market="us",
        use_llm=False,
        config=config,
    )

    assert captured["source"] == "yfinance"
    assert result.market == "us"
