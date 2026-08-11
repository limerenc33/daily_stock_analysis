#!/usr/bin/env python3
"""Download US daily bars and replay the built-in screening strategies.

Example:
    python scripts/run_us_strategy_backtest.py --start 2015-01-01 --end 2025-12-31
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
from pathlib import Path
import sys
from io import StringIO
from urllib.parse import urlencode
from urllib.request import Request, urlopen

# Allow `python scripts/run_us_strategy_backtest.py` from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services.screening.us_backtest import (
    DEFAULT_US_BACKTEST_UNIVERSE,
    USBacktestConfig,
    backtest_strategy,
    normalize_price_history,
    validation_gate,
    walk_forward_metrics,
)


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def required_history_start(start: date, lookback_days: int) -> date:
    """Return the earliest date needed to warm up rolling features."""
    return start - timedelta(days=max(lookback_days * 2, 180))


def cache_covers_window(
    cached,
    *,
    start: date,
    end: date,
    earliest_available: date | None = None,
) -> bool:
    """Require warm-up coverage and tolerate a short end-date market gap."""
    if cached is None or cached.empty:
        return False
    first_date = cached.iloc[0]["date"]
    last_date = cached.iloc[-1]["date"]
    required_start = max(start, earliest_available) if earliest_available is not None else start
    start_tolerance = timedelta(days=7) if earliest_available is not None and earliest_available > start else timedelta(0)
    return first_date <= required_start + start_tolerance and last_date >= end - timedelta(days=7)


def _download_stooq(ticker: str, *, start: date, end: date):
    import pandas as pd

    query = urlencode({
        "s": f"{ticker.lower()}.us", "i": "d",
        "d1": start.strftime("%Y%m%d"), "d2": end.strftime("%Y%m%d"),
    })
    request = Request(
        f"https://stooq.com/q/d/l/?{query}",
        headers={"User-Agent": "Mozilla/5.0 (compatible; DSA strategy research)"},
    )
    with urlopen(request, timeout=30) as response:
        payload = response.read().decode("utf-8", "ignore")
    if payload.lstrip().startswith("<"):
        raise RuntimeError("Stooq returned HTML instead of daily-bar CSV")
    if not payload.lstrip().lower().startswith("date,"):
        raise RuntimeError("Stooq returned an invalid daily-bar CSV header")
    return normalize_price_history(pd.read_csv(StringIO(payload)))


def _cache_metadata_path(path: Path) -> Path:
    return path.with_suffix(".metadata.json")


def _cached_provider(path: Path, *, ticker: str, history) -> str:
    metadata_path = _cache_metadata_path(path)
    if not metadata_path.is_file():
        return "unknown"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return "unknown"
    if not isinstance(metadata, dict):
        return "unknown"
    if (
        metadata.get("schema_version") != 1
        or metadata.get("ticker") != ticker
        or metadata.get("row_count") != len(history)
        or metadata.get("first_date") != str(history.iloc[0]["date"])[:10]
        or metadata.get("last_date") != str(history.iloc[-1]["date"])[:10]
    ):
        return "unknown"
    provider = str(metadata.get("provider") or "").strip().lower()
    return provider or "unknown"


def _write_history_cache(
    path: Path,
    history,
    *,
    ticker: str,
    provider: str,
    start: date,
    end: date,
) -> None:
    history.to_csv(path, index=False)
    metadata = {
        "schema_version": 1,
        "ticker": ticker,
        "provider": provider,
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "first_date": str(history.iloc[0]["date"])[:10],
        "last_date": str(history.iloc[-1]["date"])[:10],
        "row_count": len(history),
    }
    _cache_metadata_path(path).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _download_or_cache(
    ticker: str,
    *,
    start: date,
    end: date,
    cache_dir: Path,
    source: str,
    earliest_available: date | None = None,
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{ticker.replace('-', '_')}.csv"
    if path.is_file():
        cached = normalize_price_history(__import__("pandas").read_csv(path))
        if cache_covers_window(
            cached,
            start=start,
            end=end,
            earliest_available=earliest_available,
        ):
            return cached, f"cache:{_cached_provider(path, ticker=ticker, history=cached)}"

    errors = []
    if source in {"auto", "yfinance"}:
        try:
            import yfinance as yf

            raw = yf.download(
                ticker,
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                auto_adjust=True,
                actions=False,
                progress=False,
                threads=False,
            )
            history = normalize_price_history(raw)
            if not history.empty:
                _write_history_cache(
                    path,
                    history,
                    ticker=ticker,
                    provider="yfinance",
                    start=start,
                    end=end,
                )
                return history, "yfinance"
        except Exception as exc:
            errors.append(f"yfinance: {exc}")
    if source in {"auto", "stooq"}:
        try:
            history = _download_stooq(ticker, start=start, end=end)
            if not history.empty:
                _write_history_cache(
                    path,
                    history,
                    ticker=ticker,
                    provider="stooq",
                    start=start,
                    end=end,
                )
                return history, "stooq"
        except Exception as exc:
            errors.append(f"stooq: {exc}")
    raise RuntimeError(f"No historical bars returned for {ticker}: {'; '.join(errors)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay built-in US screening strategies")
    parser.add_argument("--start", type=_parse_date, default=date(2015, 1, 1))
    parser.add_argument("--end", type=_parse_date, default=date.today())
    parser.add_argument("--cache-dir", type=Path, default=Path("data/us_backtest"))
    parser.add_argument("--output", type=Path, default=Path("data/us_backtest/results.json"))
    parser.add_argument("--source", choices=("auto", "yfinance", "stooq"), default="auto")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--holding-days", type=int, default=10)
    parser.add_argument("--lookback-days", type=int, default=140)
    parser.add_argument("--universe", nargs="*", default=list(DEFAULT_US_BACKTEST_UNIVERSE))
    args = parser.parse_args()

    if args.lookback_days < 60:
        parser.error("--lookback-days must be at least 60")
    history_start = required_history_start(args.start, args.lookback_days)
    tickers = list(dict.fromkeys([*args.universe, "SPY", "QQQ"]))
    downloaded = {
        ticker: _download_or_cache(
            ticker, start=history_start, end=args.end, cache_dir=args.cache_dir, source=args.source
        )
        for ticker in tickers
    }
    histories = {ticker: item[0] for ticker, item in downloaded.items()}
    config = USBacktestConfig(
        start=args.start,
        end=args.end,
        top_k=args.top_k,
        holding_days=args.holding_days,
        lookback_days=args.lookback_days,
    )
    output: dict[str, object] = {
        "data_window": {"start": args.start.isoformat(), "end": args.end.isoformat()},
        "history_window": {"start": history_start.isoformat(), "end": args.end.isoformat()},
        "universe": list(args.universe),
        "benchmarks": ["SPY", "QQQ"],
        "data_sources": {ticker: item[1] for ticker, item in downloaded.items()},
        "strategies": {},
    }
    for strategy_name in ("us_quality_momentum", "us_breakout_continuation", "us_low_volatility_quality"):
        strategy_output: dict[str, object] = {}
        for benchmark in ("SPY", "QQQ"):
            result = backtest_strategy(
                strategy_name,
                {ticker: histories[ticker] for ticker in args.universe},
                histories[benchmark],
                config=config,
            )
            result["walk_forward"] = walk_forward_metrics(
                result,
                (args.start, date(2020, 1, 1), date(2023, 1, 1), args.end + timedelta(days=1)),
            )
            result["validation_gate"] = validation_gate(
                result,
                split_dates=(args.start, date(2020, 1, 1), date(2023, 1, 1), args.end + timedelta(days=1)),
            )
            strategy_output[benchmark] = result
        output["strategies"][strategy_name] = strategy_output

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    for strategy_name, benchmark_results in output["strategies"].items():
        print(strategy_name)
        for benchmark, result in benchmark_results.items():
            print(f"  vs {benchmark}: {json.dumps(result['metrics'], ensure_ascii=False)}")
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
