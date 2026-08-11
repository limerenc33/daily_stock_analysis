#!/usr/bin/env python3
"""Select a US momentum v2 candidate using only the 2015-2019 training window."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import statistics
import sys
import tempfile

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services.screening.us_backtest import USBacktestConfig, backtest_strategy, build_feature_cache
from scripts.run_us_strategy_validation_suite import BENCHMARKS, STOCK_UNIVERSES, UNIVERSES


CANDIDATES = {
    "trend_balanced": {
        "hard_filters": {
            "amount_min": 50_000_000, "price_min": 5, "change_pct_min": -5, "change_pct_max": 6,
            "change_60d_min": 0, "change_60d_max": 45, "require_price_above_ma20": True,
            "signal_score_min": 55, "macd_status_whitelist": ["bullish", "neutral"],
            "volatility_20d_pct_max": 40, "max_drawdown_20d_pct_min": -15, "atr_20_pct_max": 6,
        },
        "factor_weights": {"momentum": 0.50, "stability": 0.25, "liquidity": 0.20, "activity": 0.05},
        "scoring_profile": {"momentum_60d_slope": 1.2, "momentum_60d_overheat_pct": 45},
    },
    "trend_strict": {
        "hard_filters": {
            "amount_min": 50_000_000, "price_min": 5, "change_pct_min": -4, "change_pct_max": 5,
            "change_60d_min": 5, "change_60d_max": 40, "require_ma_bullish": True,
            "signal_score_min": 60, "macd_status_whitelist": ["bullish"],
            "volatility_20d_pct_max": 35, "max_drawdown_20d_pct_min": -12, "atr_20_pct_max": 5.5,
        },
        "factor_weights": {"momentum": 0.55, "stability": 0.25, "liquidity": 0.15, "activity": 0.05},
        "scoring_profile": {"momentum_60d_slope": 1.4, "momentum_60d_overheat_pct": 40},
    },
    "trend_broad": {
        "hard_filters": {
            "amount_min": 50_000_000, "price_min": 5, "change_pct_min": -6, "change_pct_max": 8,
            "change_60d_min": 0, "change_60d_max": 70, "require_price_above_ma20": True,
            "signal_score_min": 50, "macd_status_whitelist": ["bullish", "neutral"],
            "volatility_20d_pct_max": 50, "max_drawdown_20d_pct_min": -20, "atr_20_pct_max": 8,
        },
        "factor_weights": {"momentum": 0.70, "stability": 0.15, "liquidity": 0.10, "activity": 0.05},
        "scoring_profile": {"momentum_60d_slope": 1.5, "momentum_60d_overheat_pct": 65},
    },
    "quality_trend": {
        "hard_filters": {
            "amount_min": 50_000_000, "price_min": 5, "change_pct_min": -4, "change_pct_max": 5,
            "change_60d_min": 0, "change_60d_max": 35, "require_ma_bullish": True,
            "signal_score_min": 55, "macd_status_whitelist": ["bullish", "neutral"],
            "volatility_20d_pct_max": 30, "max_drawdown_20d_pct_min": -10, "atr_20_pct_max": 4.5,
        },
        "factor_weights": {"momentum": 0.40, "stability": 0.40, "liquidity": 0.15, "activity": 0.05},
        "scoring_profile": {"momentum_60d_slope": 1.0, "momentum_60d_overheat_pct": 35},
    },
    "persistent_momentum": {
        "hard_filters": {
            "amount_min": 50_000_000, "price_min": 5, "change_pct_min": -5, "change_pct_max": 6,
            "change_60d_min": 10, "change_60d_max": 60, "require_ma_bullish": True,
            "signal_score_min": 55, "macd_status_whitelist": ["bullish", "neutral"],
            "volatility_20d_pct_max": 45, "max_drawdown_20d_pct_min": -15, "atr_20_pct_max": 6.5,
        },
        "factor_weights": {"momentum": 0.65, "stability": 0.20, "liquidity": 0.10, "activity": 0.05},
        "scoring_profile": {"momentum_60d_slope": 1.5, "momentum_60d_overheat_pct": 60},
    },
}


def _strategy_document(name: str, candidate: dict[str, object]) -> dict[str, object]:
    return {
        "name": name,
        "display_name": name,
        "description": "Training-only US momentum v2 research candidate",
        "version": "research-1",
        "category": "trend",
        "screening": {
            "enabled": True,
            "market_scope": ["us"],
            "hard_filters": candidate["hard_filters"],
            "factor_weights": candidate["factor_weights"],
            "scoring_profile": candidate["scoring_profile"],
            "max_output": 5,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train-only selection of a frozen US momentum v2 candidate")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/us_backtest"))
    parser.add_argument("--output", type=Path, default=Path("data/us_research/momentum_v2_training.json"))
    args = parser.parse_args()
    training_start, training_end = date(2015, 1, 1), date(2019, 12, 31)

    import pandas as pd

    required = sorted(set(BENCHMARKS).union(*(set(UNIVERSES[name]) for name in STOCK_UNIVERSES)))
    histories = {
        ticker: pd.read_csv(args.cache_dir / f"{ticker.replace('-', '_')}.csv")
        for ticker in required
    }
    feature_caches = {
        universe: build_feature_cache(
            {ticker: histories[ticker] for ticker in UNIVERSES[universe]}, lookback_days=140
        )
        for universe in STOCK_UNIVERSES
    }
    candidate_results: dict[str, object] = {}
    with tempfile.TemporaryDirectory(prefix="us-momentum-v2-") as temp_dir:
        strategies_dir = Path(temp_dir)
        for name, candidate in CANDIDATES.items():
            (strategies_dir / f"{name}.yaml").write_text(
                yaml.safe_dump(_strategy_document(name, candidate), sort_keys=False), encoding="utf-8"
            )
        for name in CANDIDATES:
            print(f"training {name}", flush=True)
            cells = []
            by_universe: dict[str, object] = {}
            for universe in STOCK_UNIVERSES:
                by_universe[universe] = {}
                universe_histories = {ticker: histories[ticker] for ticker in UNIVERSES[universe]}
                for benchmark in BENCHMARKS:
                    result = backtest_strategy(
                        name,
                        universe_histories,
                        histories[benchmark],
                        config=USBacktestConfig(
                            start=training_start, end=training_end, top_k=5, holding_days=10
                        ),
                        strategies_dir=strategies_dir,
                        feature_cache=feature_caches[universe],
                    )
                    metrics = result["metrics"]
                    by_universe[universe][benchmark] = metrics
                    cells.append(metrics)
            positive_by_universe = {
                universe: sum(
                    float(by_universe[universe][benchmark]["total_return_pct"] or 0.0) > 0
                    and float(by_universe[universe][benchmark]["excess_return_pct"] or 0.0) > 0
                    for benchmark in BENCHMARKS
                )
                for universe in STOCK_UNIVERSES
            }
            candidate_results[name] = {
                "parameters": CANDIDATES[name],
                "metrics": by_universe,
                "selection_diagnostics": {
                    "minimum_positive_benchmark_cells": min(positive_by_universe.values()),
                    "positive_benchmark_cells": positive_by_universe,
                    "median_sharpe": statistics.median(float(cell["sharpe"] or -999.0) for cell in cells),
                    "median_excess_return_pct": statistics.median(float(cell["excess_return_pct"] or 0.0) for cell in cells),
                    "minimum_trade_count": min(int(cell["trade_count"] or 0) for cell in cells),
                },
            }

    eligible = {
        name: result for name, result in candidate_results.items()
        if result["selection_diagnostics"]["minimum_trade_count"] >= 200
    }
    selected = max(
        eligible,
        key=lambda name: (
            eligible[name]["selection_diagnostics"]["minimum_positive_benchmark_cells"],
            eligible[name]["selection_diagnostics"]["median_sharpe"],
            eligible[name]["selection_diagnostics"]["median_excess_return_pct"],
        ),
    ) if eligible else None
    output = {
        "training_window": {"start": training_start.isoformat(), "end": training_end.isoformat()},
        "selection_rule": [
            "minimum 200 trades in every cell",
            "maximize worst-universe count of profitable positive-excess benchmark cells",
            "then maximize median Sharpe",
            "then maximize median excess return",
        ],
        "selected_candidate": selected,
        "candidates": candidate_results,
        "oos_not_read_by_this_script": "2020-01-01 through 2025-12-31",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"selected={selected}")
    for name, result in candidate_results.items():
        print(name, result["selection_diagnostics"])
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
