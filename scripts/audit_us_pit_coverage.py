#!/usr/bin/env python3
"""Audit local price coverage for dated S&P 500 constituent snapshots."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services.screening.us_point_in_time import PointInTimeUniverse


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit point-in-time universe price coverage")
    parser.add_argument("--universe-file", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/us_backtest"))
    parser.add_argument("--dates", nargs="+", type=_parse_date, required=True)
    parser.add_argument("--minimum-history-points", type=int, default=60)
    parser.add_argument("--required-coverage", type=float, default=0.95)
    parser.add_argument("--output", type=Path, default=Path("data/us_validation/pit_coverage.json"))
    args = parser.parse_args()
    if not 0.0 <= args.required_coverage <= 1.0:
        parser.error("--required-coverage must be between 0 and 1")

    universe = PointInTimeUniverse.from_csv(args.universe_file)
    required_tickers = sorted(set().union(*(universe.constituents_on(value) for value in args.dates)))
    histories = {}
    for ticker in required_tickers:
        path = args.cache_dir / f"{ticker.replace('-', '_')}.csv"
        if path.is_file():
            histories[ticker] = pd.read_csv(path)
    snapshots = [
        universe.coverage_on(
            value,
            histories,
            minimum_history_points=args.minimum_history_points,
        )
        for value in args.dates
    ]
    output = {
        "universe_source": str(args.universe_file),
        "cache_dir": str(args.cache_dir),
        "required_coverage": args.required_coverage,
        "effective": all(float(item["coverage_ratio"]) >= args.required_coverage for item in snapshots),
        "snapshots": snapshots,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    for item in snapshots:
        print(
            f"{item['as_of']}: {item['available_count']}/{item['expected_count']} "
            f"({float(item['coverage_ratio']):.2%})"
        )
    print(f"effective={output['effective']}")
    print(f"saved: {args.output}")
    return 0 if output["effective"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
