#!/usr/bin/env python3
"""Monitor open US paper positions and persist triggered grid exits."""

from __future__ import annotations

import argparse
from datetime import datetime, time, timezone
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_provider import DataFetcherManager
from src.services.screening.us_paper_trading import (
    apply_realtime_grid_quotes,
    render_grid_event_notification,
    render_paper_trading_report,
    upgrade_paper_trading_state,
)


NEW_YORK = ZoneInfo("America/New_York")


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _is_us_market_open(observed_at: datetime) -> bool:
    market_time = observed_at.astimezone(NEW_YORK)
    return (
        market_time.weekday() < 5
        and time(9, 30) <= market_time.time().replace(tzinfo=None) <= time(16, 0)
    )


def _active_codes(state: dict[str, object]) -> list[str]:
    codes: list[str] = []
    for portfolio in dict(state.get("portfolios") or {}).values():
        cycle = portfolio.get("active_cycle")
        if not isinstance(cycle, dict) or cycle.get("status") not in {"open", "awaiting_settlement"}:
            continue
        for position in cycle.get("positions", []):
            if position.get("status") != "closed":
                codes.append(str(position["code"]))
    return list(dict.fromkeys(codes))


def _source_name(value: object) -> str:
    raw = getattr(value, "value", value)
    return str(raw or "realtime quote")


def _fetch_fresh_quotes(codes: list[str]) -> dict[str, dict[str, object]]:
    manager = DataFetcherManager()
    quotes: dict[str, dict[str, object]] = {}
    for code in codes:
        try:
            quote = manager.get_realtime_quote(code)
        except Exception as exc:
            print(f"warning: realtime quote failed for {code}: {exc}", file=sys.stderr)
            continue
        if quote is None or getattr(quote, "is_stale", False) is True:
            continue
        if str(getattr(quote, "data_quality", "") or "").lower() == "unavailable":
            continue
        price = float(getattr(quote, "price", 0.0) or 0.0)
        if price <= 0:
            continue
        quotes[code] = {
            "price": price,
            "source": _source_name(getattr(quote, "source", None)),
            "provider_timestamp": getattr(quote, "provider_timestamp", None),
            "data_quality": getattr(quote, "data_quality", None),
        }
    return quotes


def _send_notification(
    report: str,
    events: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    from src.notification import NotificationService

    fingerprint = hashlib.sha256(
        json.dumps(events, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    result = NotificationService().send_with_results(
        report,
        route_type="report",
        severity="warning",
        dedup_key=f"us-paper-grid:{fingerprint}",
        cooldown_key=f"us-paper-grid:{fingerprint}",
    )
    return {
        "dispatched": result.dispatched,
        "success": result.success,
        "status": result.status,
        "message": result.message,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor US paper-trading grid exits")
    parser.add_argument("--state", type=Path, default=Path("data/us_paper_trading/state.json"))
    parser.add_argument("--report", type=Path, default=Path("data/us_paper_trading/latest.md"))
    parser.add_argument("--notify", action="store_true")
    parser.add_argument("--force", action="store_true", help="run outside regular US market hours")
    args = parser.parse_args()

    if not args.state.is_file():
        print(f"state not found: {args.state}")
        return 0
    state = json.loads(args.state.read_text(encoding="utf-8"))
    upgrade_paper_trading_state(state)
    observed_at = datetime.now(timezone.utc)
    if not args.force and not _is_us_market_open(observed_at):
        print("US regular market is closed; no realtime grid check")
        return 0
    codes = _active_codes(state)
    if not codes:
        print("no open US paper positions")
        return 0
    quotes = _fetch_fresh_quotes(codes)
    market_date = observed_at.astimezone(NEW_YORK).date()
    events = apply_realtime_grid_quotes(
        state,
        quotes,
        observed_at=observed_at,
        market_date=market_date,
    )
    if not events:
        print(f"no grid trigger; fresh_quotes={len(quotes)}/{len(codes)}")
        return 0

    report = render_paper_trading_report(state)
    event_report = render_grid_event_notification(state, events)
    _write_atomic(args.state, json.dumps(state, ensure_ascii=False, indent=2))
    _write_atomic(args.report, report)
    print(event_report)
    if args.notify:
        print("notification=" + json.dumps(
            _send_notification(event_report, events),
            ensure_ascii=False,
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
