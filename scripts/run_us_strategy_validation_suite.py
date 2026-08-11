#!/usr/bin/env python3
"""Run a reproducible multi-universe robustness suite for US strategies."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import statistics
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_us_strategy_backtest import _download_or_cache, required_history_start
from src.services.screening.us_backtest import (
    DEFAULT_US_BACKTEST_UNIVERSE,
    USBacktestConfig,
    backtest_strategy,
    bootstrap_excess_ci,
    build_feature_cache,
    validation_gate,
    walk_forward_metrics,
)
from src.services.screening.strategy import load_all_strategies


STRATEGIES = (
    "us_quality_momentum",
    "us_breakout_continuation",
    "us_low_volatility_quality",
)
BENCHMARKS = ("SPY", "QQQ", "IWM", "DIA", "RSP")
STOCK_UNIVERSES = ("large_cap_22", "diversified_60")
UNIVERSES = {
    "large_cap_22": DEFAULT_US_BACKTEST_UNIVERSE,
    "diversified_60": tuple(dict.fromkeys((*DEFAULT_US_BACKTEST_UNIVERSE, *(
        "ADBE", "AMAT", "BAC", "CAT", "CSCO", "CVX", "DE", "DIS", "GE", "GILD",
        "GM", "GS", "HON", "IBM", "INTC", "LIN", "LOW", "MCD", "MRK", "NKE",
        "ORCL", "PFE", "PLD", "RTX", "SBUX", "T", "TMO", "TXN", "UPS", "VZ",
        "COP", "F", "HCA", "LMT", "MDT", "MU", "QCOM", "SO",
    )))),
    "sector_etfs": (
        "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
    ),
}
SPLIT_DATES = (date(2015, 1, 1), date(2020, 1, 1), date(2023, 1, 1), date(2026, 1, 1))
# Fixed calendar epochs expose regime dependence without refitting parameters.
REGIME_SPLIT_DATES = (
    date(2015, 1, 1), date(2020, 1, 1), date(2021, 1, 1),
    date(2022, 1, 1), date(2023, 1, 1), date(2026, 1, 1),
)
KNOWN_INCEPTION_DATES = {
    "XLC": date(2018, 6, 18),
    "XLRE": date(2015, 10, 8),
}


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _summary(result: dict[str, object]) -> dict[str, object]:
    walk_forward = walk_forward_metrics(result, SPLIT_DATES)
    regime_metrics = walk_forward_metrics(result, REGIME_SPLIT_DATES)
    gate = validation_gate(result, split_dates=SPLIT_DATES)
    return {
        "metrics": result["metrics"],
        "walk_forward": walk_forward,
        "regime_metrics": regime_metrics,
        "validation_gate": gate,
        "bootstrap_excess": bootstrap_excess_ci(result),
        "omitted_point_in_time_filters": result["omitted_point_in_time_filters"],
    }


def _generalization_gate(
    baseline: dict[str, dict[str, dict[str, dict[str, object]]]],
    stress: dict[str, dict[str, dict[str, dict[str, dict[str, object]]]]],
    strategy: str,
) -> dict[str, object]:
    stock_cells = [baseline[universe][strategy][benchmark] for universe in STOCK_UNIVERSES for benchmark in BENCHMARKS]
    positive_by_universe = {
        universe: sum(
            float(baseline[universe][strategy][benchmark]["metrics"]["total_return_pct"] or 0.0) > 0.0
            and float(baseline[universe][strategy][benchmark]["metrics"]["excess_return_pct"] or 0.0) > 0.0
            for benchmark in BENCHMARKS
        )
        for universe in STOCK_UNIVERSES
    }
    oos_by_universe = {
        universe: sum(
            bool(baseline[universe][strategy][benchmark]["validation_gate"]["checks"]["oos_excess_positive_in_majority"])
            for benchmark in BENCHMARKS
        )
        for universe in STOCK_UNIVERSES
    }
    regime_positive_by_universe = {
        universe: sum(
            float(item["metrics"].get("total_return_pct") or 0.0) > 0.0
            and float(item["metrics"].get("excess_return_pct") or 0.0) > 0.0
            for item in baseline[universe][strategy]["SPY"].get("regime_metrics", [])
        )
        for universe in STOCK_UNIVERSES
    }
    sharpes = [float(cell["metrics"]["sharpe"] or -999.0) for cell in stock_cells]
    bootstrap_supported = sum(float(cell["bootstrap_excess"]["ci95_low_pct"] or -999.0) > 0.0 for cell in stock_cells)
    holding_positive: dict[str, int] = {}
    cost_stress_positive: dict[str, bool] = {}
    for universe in STOCK_UNIVERSES:
        holding_positive[universe] = sum(
            float(stress[universe][strategy][f"holding_{holding}"]["round_trip_20bps"]["metrics"]["total_return_pct"] or 0.0) > 0.0
            and float(stress[universe][strategy][f"holding_{holding}"]["round_trip_20bps"]["metrics"]["excess_return_pct"] or 0.0) > 0.0
            for holding in (5, 10, 20)
        )
        cost_stress_positive[universe] = (
            float(stress[universe][strategy]["holding_10"]["round_trip_50bps"]["metrics"]["total_return_pct"] or 0.0)
            > 0.0
            and
            float(stress[universe][strategy]["holding_10"]["round_trip_50bps"]["metrics"]["excess_return_pct"] or 0.0)
            > 0.0
        )
    checks = {
        "beats_majority_of_benchmarks_in_each_stock_universe": all(value >= 3 for value in positive_by_universe.values()),
        "median_sharpe_at_least_0_5": statistics.median(sharpes) >= 0.5,
        "oos_positive_for_majority_of_benchmarks_in_each_universe": all(value >= 3 for value in oos_by_universe.values()),
        "positive_in_at_least_3_of_5_fixed_regime_epochs_in_each_universe": all(
            value >= 3 for value in regime_positive_by_universe.values()
        ),
        "holding_period_robust_in_each_universe": all(value >= 2 for value in holding_positive.values()),
        "survives_50bps_round_trip_cost_in_each_universe": all(cost_stress_positive.values()),
        "bootstrap_ci_positive_in_at_least_6_of_10_cells": bootstrap_supported >= 6,
    }
    return {
        "effective": all(checks.values()),
        "checks": checks,
        "diagnostics": {
            "positive_benchmark_cells_by_universe": positive_by_universe,
            "oos_positive_cells_by_universe": oos_by_universe,
            "regime_positive_cells_by_universe": regime_positive_by_universe,
            "median_sharpe": round(statistics.median(sharpes), 6),
            "bootstrap_positive_ci_cells": bootstrap_supported,
            "holding_period_positive_cells_by_universe": holding_positive,
            "cost_stress_positive_by_universe": cost_stress_positive,
        },
        "thresholds": {
            "benchmarks_per_universe": "at least 3 of 5 with positive strategy return and excess",
            "median_sharpe": 0.5,
            "oos_benchmarks_per_universe": "at least 3 of 5",
            "positive_regime_epochs_per_universe": "at least 3 of 5 fixed epochs",
            "holding_periods_per_universe": "at least 2 of 3 with positive strategy return and excess",
            "round_trip_cost_bps": "50 with positive strategy return and excess",
            "bootstrap_positive_ci_cells": "at least 6 of 10",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate US strategies across universes, benchmarks and stresses")
    parser.add_argument("--start", type=_parse_date, default=date(2015, 1, 1))
    parser.add_argument("--end", type=_parse_date, default=date(2025, 12, 31))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/us_backtest"))
    parser.add_argument("--output", type=Path, default=Path("data/us_validation/results.json"))
    parser.add_argument("--source", choices=("auto", "yfinance", "stooq"), default="auto")
    parser.add_argument("--lookback-days", type=int, default=140)
    parser.add_argument("--strategies", nargs="*", default=list(STRATEGIES))
    parser.add_argument(
        "--strategies-dir",
        type=Path,
        default=Path("src/services/screening/strategies"),
        help="strategy directory; use research/us_strategy_candidates only for audit reproduction",
    )
    args = parser.parse_args()
    if args.lookback_days < 60:
        parser.error("--lookback-days must be at least 60")
    strategies = tuple(dict.fromkeys(args.strategies))
    unknown_strategies = sorted(set(strategies) - set(load_all_strategies(args.strategies_dir)))
    if unknown_strategies:
        parser.error(f"unknown strategies: {', '.join(unknown_strategies)}")

    tickers = list(dict.fromkeys(ticker for universe in UNIVERSES.values() for ticker in universe))
    tickers = list(dict.fromkeys((*tickers, *BENCHMARKS)))
    downloaded: dict[str, tuple[object, str]] = {}
    history_start = required_history_start(args.start, args.lookback_days)
    for index, ticker in enumerate(tickers, start=1):
        print(f"download {index}/{len(tickers)} {ticker}", flush=True)
        downloaded[ticker] = _download_or_cache(
            ticker,
            start=history_start,
            end=args.end,
            cache_dir=args.cache_dir,
            source=args.source,
            earliest_available=KNOWN_INCEPTION_DATES.get(ticker),
        )
    histories = {ticker: item[0] for ticker, item in downloaded.items()}
    baseline: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
    stress: dict[str, dict[str, dict[str, dict[str, dict[str, object]]]]] = {}
    feature_caches: dict[str, object] = {}

    for universe_name, universe in UNIVERSES.items():
        print(f"features {universe_name} ({len(universe)} instruments)", flush=True)
        universe_histories = {ticker: histories[ticker] for ticker in universe}
        feature_cache = build_feature_cache(universe_histories, lookback_days=args.lookback_days)
        feature_caches[universe_name] = feature_cache
        baseline[universe_name] = {}
        for strategy in strategies:
            baseline[universe_name][strategy] = {}
            for benchmark in BENCHMARKS:
                print(f"baseline {universe_name} {strategy} vs {benchmark}", flush=True)
                result = backtest_strategy(
                    strategy,
                    universe_histories,
                    histories[benchmark],
                    config=USBacktestConfig(
                        start=args.start,
                        end=args.end,
                        top_k=5,
                        holding_days=10,
                        lookback_days=args.lookback_days,
                    ),
                    strategies_dir=args.strategies_dir,
                    feature_cache=feature_cache,
                )
                baseline[universe_name][strategy][benchmark] = _summary(result)

    for universe_name in STOCK_UNIVERSES:
        universe = UNIVERSES[universe_name]
        universe_histories = {ticker: histories[ticker] for ticker in universe}
        stress[universe_name] = {}
        for strategy in strategies:
            stress[universe_name][strategy] = {}
            for holding_days in (5, 10, 20):
                holding_key = f"holding_{holding_days}"
                stress[universe_name][strategy][holding_key] = {}
                for cost_name, per_side_component_bps in (("round_trip_20bps", 5.0), ("round_trip_50bps", 12.5)):
                    print(f"stress {universe_name} {strategy} {holding_key} {cost_name}", flush=True)
                    result = backtest_strategy(
                        strategy,
                        universe_histories,
                        histories["SPY"],
                        config=USBacktestConfig(
                            start=args.start,
                            end=args.end,
                            top_k=5,
                            holding_days=holding_days,
                            lookback_days=args.lookback_days,
                            transaction_cost_bps=per_side_component_bps,
                            slippage_bps=per_side_component_bps,
                        ),
                        strategies_dir=args.strategies_dir,
                        feature_cache=feature_caches[universe_name],
                    )
                    stress[universe_name][strategy][holding_key][cost_name] = _summary(result)

    gates = {strategy: _generalization_gate(baseline, stress, strategy) for strategy in strategies}
    output = {
        "data_window": {"start": args.start.isoformat(), "end": args.end.isoformat()},
        "history_window": {"start": history_start.isoformat(), "end": args.end.isoformat()},
        "data_sources": {ticker: source for ticker, (_, source) in downloaded.items()},
        "universes": {name: list(tickers) for name, tickers in UNIVERSES.items()},
        "benchmarks": list(BENCHMARKS),
        "split_dates": [value.isoformat() for value in SPLIT_DATES],
        "baseline": baseline,
        "stress": stress,
        "generalization_gates": gates,
        "limitations": [
            "Stock universes use current members and therefore retain survivorship bias.",
            "Market-cap and valuation filters are omitted without point-in-time fundamentals.",
            "Yahoo Finance adjusted bars require licensed-source parity checks before live use.",
            "Bootstrap intervals do not eliminate regime dependence or multiple-testing risk.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    for strategy, gate in gates.items():
        print(f"{strategy}: effective={gate['effective']} diagnostics={gate['diagnostics']}")
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
