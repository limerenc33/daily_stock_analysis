#!/usr/bin/env python3
"""Advance the US strategy paper-trading ledger with real Yahoo bars."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_us_strategy_validation_suite import BENCHMARKS, STOCK_UNIVERSES, UNIVERSES
from src.services.screening.us_backtest import normalize_price_history
from src.services.screening.us_paper_trading import (
    USPaperTradingConfig,
    advance_paper_trading_state,
    create_paper_trading_state,
    render_paper_trading_report,
)


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def split_yfinance_download(raw: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Split either yfinance MultiIndex layout into normalized ticker frames."""
    if raw is None or raw.empty:
        return {}
    if len(tickers) == 1 and not isinstance(raw.columns, pd.MultiIndex):
        normalized = normalize_price_history(raw)
        return {tickers[0]: normalized} if not normalized.empty else {}
    if not isinstance(raw.columns, pd.MultiIndex):
        return {}
    upper_tickers = {item.upper() for item in tickers}
    scores = [
        sum(str(value).upper() in upper_tickers for value in raw.columns.get_level_values(level))
        for level in range(raw.columns.nlevels)
    ]
    ticker_level = int(max(range(len(scores)), key=scores.__getitem__))
    available = {str(value).upper(): value for value in raw.columns.get_level_values(ticker_level)}
    output: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        label = available.get(ticker.upper())
        if label is None:
            continue
        frame = raw.xs(label, axis=1, level=ticker_level, drop_level=True)
        normalized = normalize_price_history(frame)
        if not normalized.empty:
            output[ticker] = normalized
    return output


def download_yahoo_histories(
    tickers: list[str],
    *,
    start: date,
    end: date,
    batch_size: int = 25,
) -> dict[str, pd.DataFrame]:
    """Download adjusted daily bars in bounded batches with individual retries."""
    import yfinance as yf

    histories: dict[str, pd.DataFrame] = {}
    for offset in range(0, len(tickers), batch_size):
        batch = tickers[offset: offset + batch_size]
        print(f"Yahoo batch {offset // batch_size + 1}: {len(batch)} tickers", flush=True)
        raw = yf.download(
            batch,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            auto_adjust=True,
            actions=False,
            progress=False,
            threads=False,
            group_by="column",
        )
        histories.update(split_yfinance_download(raw, batch))
    missing = [ticker for ticker in tickers if ticker not in histories]
    for ticker in missing:
        print(f"Yahoo individual retry: {ticker}", flush=True)
        raw = yf.download(
            ticker,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            auto_adjust=True,
            actions=False,
            progress=False,
            threads=False,
        )
        normalized = normalize_price_history(raw)
        if not normalized.empty:
            histories[ticker] = normalized
    return histories


def _load_or_create_state(path: Path) -> dict[str, object]:
    if path.is_file():
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"paper-trading state is unreadable: {path}") from exc
        if int(state.get("schema_version") or 0) != 1:
            raise ValueError("unsupported paper-trading state schema")
        return state
    return create_paper_trading_state(
        {name: UNIVERSES[name] for name in STOCK_UNIVERSES},
        benchmarks=BENCHMARKS,
        config=USPaperTradingConfig(),
    )


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _send_notification(report: str, market_date: str) -> dict[str, object]:
    from src.notification import NotificationService

    result = NotificationService().send_with_results(
        report,
        route_type="report",
        severity="info",
        dedup_key=f"us-paper-trading:{market_date}",
        cooldown_key="us-paper-trading-daily",
    )
    return {
        "dispatched": result.dispatched,
        "success": result.success,
        "status": result.status,
        "message": result.message,
        "channels": [
            {"channel": item.channel, "success": item.success, "error_code": item.error_code}
            for item in result.channel_results
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the US strategy paper-trading ledger")
    parser.add_argument("--state", type=Path, default=Path("data/us_paper_trading/state.json"))
    parser.add_argument("--report", type=Path, default=Path("data/us_paper_trading/latest.md"))
    parser.add_argument("--as-of", type=_parse_date, default=None)
    parser.add_argument("--history-days", type=int, default=420)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--notify", action="store_true")
    args = parser.parse_args()
    if args.history_days < 240 or args.batch_size <= 0:
        parser.error("--history-days must be at least 240 and --batch-size must be positive")

    state = _load_or_create_state(args.state)
    requested_end = args.as_of or date.today()
    tickers = list(dict.fromkeys(
        ticker
        for name in STOCK_UNIVERSES
        for ticker in UNIVERSES[name]
    ))
    tickers = list(dict.fromkeys([*tickers, *BENCHMARKS]))
    histories = download_yahoo_histories(
        tickers,
        start=requested_end - timedelta(days=args.history_days),
        end=requested_end,
        batch_size=args.batch_size,
    )
    missing = sorted(set(tickers) - set(histories))
    if missing:
        print(f"warning: missing histories: {', '.join(missing)}", file=sys.stderr)
    advance_paper_trading_state(state, histories, as_of=args.as_of)
    report = render_paper_trading_report(state)
    _write_atomic(args.state, json.dumps(state, ensure_ascii=False, indent=2))
    _write_atomic(args.report, report)
    notification = None
    if args.notify:
        notification = _send_notification(report, str(state["latest_market_date"]))
        print("notification=" + json.dumps(notification, ensure_ascii=False))
    print(report)
    print(f"state={args.state}")
    print(f"report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
