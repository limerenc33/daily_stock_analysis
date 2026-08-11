"""Historical replay for the built-in US screening strategies.

The production screening pipeline evaluates a current snapshot. This module
replays the same daily feature calculations against historical bars and turns
each signal into a fixed-horizon, next-session paper trade. It deliberately
does not fabricate point-in-time market-cap or valuation data: those filters
are excluded and reported in the result when the source does not provide them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import math
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from src.services.screening.daily import compute_daily_features
from src.services.screening.filter import apply_hard_filters
from src.services.screening.models import HardFilterConfig
from src.services.screening.scorer import compute_screen_scores
from src.services.screening.strategy import load_all_strategies


DEFAULT_US_BACKTEST_UNIVERSE = (
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO",
    "JPM", "LLY", "V", "MA", "UNH", "XOM", "COST", "HD", "PG", "JNJ",
    "WMT", "NFLX", "CRM", "AMD",
)


@dataclass(frozen=True)
class USBacktestConfig:
    start: date = date(2015, 1, 1)
    end: date | None = None
    top_k: int = 5
    holding_days: int = 10
    signal_step_days: int | None = None
    lookback_days: int = 140
    initial_capital: float = 100_000.0
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 5.0
    minimum_universe_coverage: float = 0.95

    @property
    def step_days(self) -> int:
        return int(self.signal_step_days or self.holding_days)


def normalize_price_history(history: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance or local bars to date/open/high/low/close/volume."""
    if history is None or history.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    frame = history.copy()
    rename = {
        "Date": "date", "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume", "日期": "date", "开盘": "open",
        "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume",
    }
    if isinstance(frame.columns, pd.MultiIndex):
        # yfinance has returned both (Price, Ticker) and (Ticker, Price)
        # layouts across releases. Select the level containing OHLCV labels.
        label_scores = [
            sum(str(value) in rename for value in frame.columns.get_level_values(level))
            for level in range(frame.columns.nlevels)
        ]
        frame.columns = frame.columns.get_level_values(int(np.argmax(label_scores)))
    frame = frame.rename(columns=rename)
    if "date" not in frame.columns:
        frame = frame.rename_axis("date").reset_index()
    for column in ("open", "high", "low", "close", "volume"):
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True).dt.tz_localize(None).dt.date
    frame = frame.dropna(subset=["date", "close"]).sort_values("date")
    frame = frame.drop_duplicates(subset=["date"], keep="last")
    for column in ("open", "high", "low"):
        frame[column] = frame[column].fillna(frame["close"])
    return frame[["date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def _backtest_filters(filters: HardFilterConfig) -> HardFilterConfig:
    """Remove fields unavailable point-in-time in the lightweight data feed."""
    values = {field: getattr(filters, field) for field in filters.__dataclass_fields__}
    for field in ("market_cap_min", "market_cap_max", "pe_ttm_min", "pe_ttm_max", "pb_min", "pb_max"):
        values[field] = None
    return HardFilterConfig(**values)


def _feature_row(code: str, history: pd.DataFrame, end_index: int, lookback_days: int) -> dict[str, object] | None:
    if end_index < 1:
        return None
    window = history.iloc[max(0, end_index - lookback_days + 1): end_index + 1].copy()
    if len(window) < 60:
        return None
    features = compute_daily_features(window)
    current = history.iloc[end_index]
    previous = history.iloc[end_index - 1]
    close = float(current["close"])
    previous_close = float(previous["close"])
    if close <= 0 or previous_close <= 0:
        return None
    volume = float(current["volume"]) if pd.notna(current["volume"]) else 0.0
    return {
        "code": code,
        "name": code,
        "price": close,
        "change_pct": (close / previous_close - 1.0) * 100.0,
        "amount": close * volume,
        "total_mv": np.nan,
        "circ_mv": np.nan,
        "pe_ratio": np.nan,
        "pb_ratio": np.nan,
        "volume_ratio": features.get("volume_ratio_20d"),
        "turnover_rate": np.nan,
        "industry": "",
        **features,
    }


def build_feature_cache(
    histories: Mapping[str, pd.DataFrame],
    *,
    lookback_days: int = 140,
) -> dict[str, dict[date, dict[str, object]]]:
    """Precompute daily production features once for a validation suite.

    A strategy universe and benchmark matrix reuses the same historical rows;
    caching here avoids recalculating rolling indicators for every cell while
    keeping the feature calculation identical to the normal replay path.
    """
    normalized = {code: normalize_price_history(frame) for code, frame in histories.items()}
    output: dict[str, dict[date, dict[str, object]]] = {}
    for code, frame in normalized.items():
        rows: dict[date, dict[str, object]] = {}
        for end_index, bar in frame.iterrows():
            row = _feature_row(code, frame, int(end_index), lookback_days)
            if row is not None:
                rows[bar["date"]] = row
        output[code] = rows
    return output


def _net_return(gross_return: float, config: USBacktestConfig) -> float:
    round_trip_cost = 2.0 * (config.transaction_cost_bps + config.slippage_bps) / 10_000.0
    return (1.0 + gross_return) * (1.0 - round_trip_cost) - 1.0


def simulate_trade(
    history: pd.DataFrame,
    *,
    signal_index: int,
    holding_days: int,
    config: USBacktestConfig,
) -> dict[str, object] | None:
    """Enter on the next session open and exit after ``holding_days`` closes."""
    entry_index = signal_index + 1
    exit_index = entry_index + holding_days - 1
    if entry_index >= len(history) or exit_index >= len(history):
        return None
    entry = history.iloc[entry_index]
    exit_bar = history.iloc[exit_index]
    entry_price = float(entry["open"])
    exit_price = float(exit_bar["close"])
    if not math.isfinite(entry_price) or not math.isfinite(exit_price) or entry_price <= 0:
        return None
    gross = exit_price / entry_price - 1.0
    return {
        "signal_date": str(history.iloc[signal_index]["date"]),
        "entry_date": str(entry["date"]),
        "exit_date": str(exit_bar["date"]),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "gross_return_pct": gross * 100.0,
        "net_return_pct": _net_return(gross, config) * 100.0,
    }


def _period_metrics(periods: list[dict[str, object]], config: USBacktestConfig) -> dict[str, object]:
    if not periods:
        return {"period_count": 0, "total_return_pct": None, "cagr_pct": None, "sharpe": None,
                "max_drawdown_pct": None, "win_rate_pct": None, "avg_return_pct": None,
                "benchmark_return_pct": None, "excess_return_pct": None}
    strategy_returns = np.array([float(row["strategy_return_pct"]) / 100.0 for row in periods])
    benchmark_returns = np.array([float(row["benchmark_return_pct"]) / 100.0 for row in periods])
    equity = np.cumprod(1.0 + strategy_returns)
    benchmark_equity = np.cumprod(1.0 + benchmark_returns)
    peaks = np.maximum.accumulate(np.concatenate(([1.0], equity)))
    drawdowns = np.concatenate(([1.0], equity)) / peaks - 1.0
    benchmark_peaks = np.maximum.accumulate(np.concatenate(([1.0], benchmark_equity)))
    benchmark_drawdowns = np.concatenate(([1.0], benchmark_equity)) / benchmark_peaks - 1.0
    periods_per_year = 252.0 / max(config.step_days, 1)
    total_return = float(equity[-1] - 1.0)
    years = len(periods) / periods_per_year
    cagr = (float(equity[-1]) ** (1.0 / years) - 1.0) if years > 0 and equity[-1] > 0 else None
    volatility = float(strategy_returns.std(ddof=1) * math.sqrt(periods_per_year)) if len(periods) > 1 else None
    sharpe = float(strategy_returns.mean() / strategy_returns.std(ddof=1) * math.sqrt(periods_per_year)) if len(periods) > 1 and strategy_returns.std(ddof=1) > 0 else None
    return {
        "period_count": len(periods),
        "trade_count": int(sum(int(row["selected_count"]) for row in periods)),
        "total_return_pct": round(total_return * 100.0, 4),
        "cagr_pct": None if cagr is None else round(cagr * 100.0, 4),
        "annualized_volatility_pct": None if volatility is None else round(volatility * 100.0, 4),
        "sharpe": None if sharpe is None else round(sharpe, 4),
        "max_drawdown_pct": round(float(drawdowns.min()) * 100.0, 4),
        "benchmark_max_drawdown_pct": round(float(benchmark_drawdowns.min()) * 100.0, 4),
        "win_rate_pct": round(float((strategy_returns > 0).mean()) * 100.0, 4),
        "avg_return_pct": round(float(strategy_returns.mean()) * 100.0, 4),
        "benchmark_return_pct": round(float(benchmark_equity[-1] - 1.0) * 100.0, 4),
        "excess_return_pct": round(float(equity[-1] - benchmark_equity[-1]) * 100.0, 4),
        "mean_excess_per_period_pct": round(float((strategy_returns - benchmark_returns).mean()) * 100.0, 4),
    }


def validation_gate(
    result: Mapping[str, object],
    *,
    split_dates: Iterable[date],
    min_periods: int = 50,
    min_sharpe: float = 0.5,
) -> dict[str, object]:
    """Apply a conservative research gate; this is not an investment guarantee."""
    metrics = dict(result.get("metrics", {}))
    walk_forward = walk_forward_metrics(result, split_dates)
    out_of_sample = [
        dict(item["metrics"])
        for item in walk_forward[1:]
        if int(item["metrics"].get("period_count") or 0) > 0
    ]
    checks = {
        "enough_periods": (
            int(metrics.get("period_count") or 0) >= min_periods
            and int(metrics.get("trade_count") or 0) >= min_periods
        ),
        "positive_excess_return": float(metrics.get("excess_return_pct") or 0.0) > 0.0,
        "minimum_sharpe": float(metrics.get("sharpe") or -math.inf) >= min_sharpe,
        "out_of_sample_segments": len(out_of_sample) >= 2,
        "oos_excess_positive_in_majority": (
            bool(out_of_sample)
            and sum(float(item.get("excess_return_pct") or 0.0) > 0.0 for item in out_of_sample)
            >= math.ceil(len(out_of_sample) / 2)
        ),
    }
    return {
        "effective": all(checks.values()),
        "checks": checks,
        "walk_forward": walk_forward,
        "minimums": {"periods": min_periods, "sharpe": min_sharpe},
    }


def bootstrap_excess_ci(
    result: Mapping[str, object],
    *,
    samples: int = 2_000,
    seed: int = 42,
) -> dict[str, object]:
    """Bootstrap the mean per-period excess return with a fixed seed.

    The default replay uses non-overlapping holding periods, but this interval
    still does not prove independence or model market-regime autocorrelation.
    It estimates uncertainty in the mean simple excess per holding period.
    """
    periods = list(result.get("periods", []))
    excess = np.asarray(
        [
            (float(row["strategy_return_pct"]) - float(row["benchmark_return_pct"])) / 100.0
            for row in periods
        ],
        dtype=float,
    )
    if excess.size == 0:
        return {"period_count": 0, "mean_excess_pct": None, "ci95_low_pct": None, "ci95_high_pct": None,
                "probability_mean_excess_positive": None}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, excess.size, size=(max(1, samples), excess.size))
    means = excess[draws].mean(axis=1)
    return {
        "period_count": int(excess.size),
        "mean_excess_pct": round(float(excess.mean()) * 100.0, 6),
        "ci95_low_pct": round(float(np.quantile(means, 0.025)) * 100.0, 6),
        "ci95_high_pct": round(float(np.quantile(means, 0.975)) * 100.0, 6),
        "probability_mean_excess_positive": round(float((means > 0).mean()), 6),
        "samples": max(1, samples),
        "seed": seed,
    }


def backtest_strategy(
    strategy_name: str,
    histories: Mapping[str, pd.DataFrame],
    benchmark: pd.DataFrame,
    *,
    config: USBacktestConfig | None = None,
    strategies_dir: Path | None = None,
    feature_cache: Mapping[str, Mapping[date, Mapping[str, object]]] | None = None,
    universe_by_date: Callable[[date], Iterable[str]] | None = None,
) -> dict[str, object]:
    """Replay one built-in US strategy and return auditable period records."""
    config = config or USBacktestConfig()
    strategies_dir = strategies_dir or Path(__file__).with_name("strategies")
    strategy = load_all_strategies(strategies_dir)[strategy_name]
    if strategy.screening.market_scope != ["us"]:
        raise ValueError(f"{strategy_name} is not a US strategy")
    if config.top_k <= 0 or config.holding_days <= 0:
        raise ValueError("top_k and holding_days must be positive")
    if not 0.0 <= config.minimum_universe_coverage <= 1.0:
        raise ValueError("minimum_universe_coverage must be between 0 and 1")

    normalized = {code: normalize_price_history(frame) for code, frame in histories.items()}
    benchmark_frame = normalize_price_history(benchmark)
    if benchmark_frame.empty:
        raise ValueError("benchmark history is empty")
    date_sets = [set(frame["date"]) for frame in normalized.values() if not frame.empty]
    if universe_by_date is not None:
        common_dates = sorted(set(benchmark_frame["date"]))
    else:
        common_dates = sorted(set(benchmark_frame["date"]).intersection(*date_sets)) if date_sets else []
    end_date = config.end or (common_dates[-1] if common_dates else None)
    signal_dates = [d for d in common_dates if d >= config.start and (end_date is None or d <= end_date)]
    if not signal_dates:
        raise ValueError("no overlapping dates in requested backtest window")
    index_by_code = {
        code: {bar_date: idx for idx, bar_date in enumerate(frame["date"])}
        for code, frame in normalized.items()
    }
    benchmark_index = {bar_date: idx for idx, bar_date in enumerate(benchmark_frame["date"])}
    filters = _backtest_filters(strategy.screening.hard_filters)
    periods: list[dict[str, object]] = []
    omitted_fields = ["market_cap_min", "market_cap_max", "pe_ttm_min", "pe_ttm_max", "pb_min", "pb_max"]
    for date_position, signal_date in enumerate(signal_dates):
        if date_position % config.step_days != 0:
            continue
        candidates: list[pd.DataFrame] = []
        active_codes = (
            sorted({str(code) for code in universe_by_date(signal_date)})
            if universe_by_date is not None
            else sorted(normalized)
        )
        feature_covered_count = 0
        for code in active_codes:
            frame = normalized.get(code)
            if frame is None or frame.empty:
                continue
            current_index = index_by_code[code].get(signal_date)
            if current_index is None:
                continue
            if feature_cache is not None:
                cached_row = feature_cache.get(code, {}).get(signal_date)
                row = dict(cached_row) if cached_row is not None else None
            else:
                row = _feature_row(code, frame, current_index, config.lookback_days)
            if row is None:
                continue
            feature_covered_count += 1
            candidate = pd.DataFrame([row])
            try:
                candidate = apply_hard_filters(candidate, filters)
            except ValueError:
                continue
            if not candidate.empty:
                candidates.append(candidate)
        expected_count = len(active_codes)
        coverage_ratio = feature_covered_count / expected_count if expected_count else 0.0
        if universe_by_date is not None and coverage_ratio < config.minimum_universe_coverage:
            raise ValueError(
                f"point-in-time universe coverage {coverage_ratio:.2%} below required "
                f"{config.minimum_universe_coverage:.2%} on {signal_date} "
                f"({feature_covered_count}/{expected_count})"
            )
        benchmark_signal_index = benchmark_index.get(signal_date)
        if benchmark_signal_index is None:
            continue
        benchmark_trade = simulate_trade(
            benchmark_frame,
            signal_index=benchmark_signal_index,
            holding_days=config.holding_days,
            config=USBacktestConfig(
                start=config.start, end=config.end, top_k=1, holding_days=config.holding_days,
                signal_step_days=config.signal_step_days, lookback_days=config.lookback_days,
                transaction_cost_bps=config.transaction_cost_bps, slippage_bps=config.slippage_bps,
            ),
        )
        if benchmark_trade is None:
            continue
        selected_trades: list[dict[str, object]] = []
        if candidates:
            ranked = compute_screen_scores(pd.concat(candidates, ignore_index=True), strategy.screening)
            ranked = ranked.sort_values(["screen_score", "code"], ascending=[False, True]).head(config.top_k)
            for _, selected in ranked.iterrows():
                code = str(selected["code"])
                frame = normalized[code]
                current_index = index_by_code[code][signal_date]
                trade = simulate_trade(frame, signal_index=current_index, holding_days=config.holding_days, config=config)
                if trade is not None:
                    trade["code"] = code
                    trade["screen_score"] = round(float(selected["screen_score"]), 4)
                    selected_trades.append(trade)
        periods.append({
            "signal_date": str(signal_date),
            "selected_count": len(selected_trades),
            "universe_expected_count": expected_count,
            "universe_feature_covered_count": feature_covered_count,
            "universe_coverage_ratio": round(coverage_ratio, 6),
            "codes": [trade["code"] for trade in selected_trades],
            "strategy_return_pct": round(
                float(np.mean([trade["net_return_pct"] for trade in selected_trades]))
                if selected_trades else 0.0,
                4,
            ),
            "benchmark_return_pct": round(float(benchmark_trade["net_return_pct"]), 4),
            "trades": selected_trades,
        })
    return {
        "strategy": strategy_name,
        "config": asdict(config),
        "universe": sorted(normalized),
        "omitted_point_in_time_filters": omitted_fields,
        "periods": periods,
        "metrics": _period_metrics(periods, config),
    }


def walk_forward_metrics(result: Mapping[str, object], split_dates: Iterable[date]) -> list[dict[str, object]]:
    """Summarize fixed historical periods without refitting parameters."""
    periods = list(result.get("periods", []))
    boundaries = sorted(str(value) for value in split_dates)
    output: list[dict[str, object]] = []
    for start, end in zip(boundaries, boundaries[1:]):
        subset = [row for row in periods if start <= str(row["signal_date"]) < end]
        config_data = dict(result.get("config", {}))
        config = USBacktestConfig(**{key: value for key, value in config_data.items() if key in USBacktestConfig.__dataclass_fields__})
        output.append({"start": start, "end": end, "metrics": _period_metrics(subset, config)})
    return output
