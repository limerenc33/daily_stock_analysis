#!/usr/bin/env python3
"""Measure Yahoo Finance coverage for one historical constituent snapshot."""

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

from src.services.screening.us_point_in_time import PointInTimeUniverse


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Yahoo coverage for a point-in-time universe")
    parser.add_argument("--universe-file", type=Path, required=True)
    parser.add_argument("--as-of", type=_parse_date, required=True)
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--required-coverage", type=float, default=0.95)
    parser.add_argument("--output", type=Path, default=Path("data/us_validation/yahoo_pit_coverage.json"))
    args = parser.parse_args()
    if args.window_days <= 0 or args.batch_size <= 0:
        parser.error("--window-days and --batch-size must be positive")
    if not 0.0 <= args.required_coverage <= 1.0:
        parser.error("--required-coverage must be between 0 and 1")

    import yfinance as yf

    universe = PointInTimeUniverse.from_csv(args.universe_file)
    expected = sorted(universe.constituents_on(args.as_of))
    available: list[str] = []
    errors: list[str] = []
    for offset in range(0, len(expected), args.batch_size):
        batch = expected[offset: offset + args.batch_size]
        print(f"batch {offset // args.batch_size + 1}/{(len(expected) - 1) // args.batch_size + 1}", flush=True)
        try:
            raw = yf.download(
                batch,
                start=args.as_of.isoformat(),
                end=(args.as_of + timedelta(days=args.window_days)).isoformat(),
                auto_adjust=True,
                actions=False,
                progress=False,
                threads=False,
            )
            if isinstance(raw.columns, pd.MultiIndex) and "Close" in raw.columns.get_level_values(0):
                close = raw.xs("Close", axis=1, level=0)
                available.extend(
                    ticker for ticker in batch if ticker in close.columns and close[ticker].notna().any()
                )
            elif len(batch) == 1 and "Close" in raw.columns and raw["Close"].notna().any():
                available.extend(batch)
        except Exception as exc:
            errors.append(f"batch {offset // args.batch_size + 1}: {exc}")
    available_set = set(available)
    missing = [ticker for ticker in expected if ticker not in available_set]
    coverage = len(available_set) / len(expected) if expected else 0.0
    output = {
        "universe_source": str(args.universe_file),
        "as_of": args.as_of.isoformat(),
        "window_days": args.window_days,
        "provider": "Yahoo Finance",
        "yfinance_version": getattr(yf, "__version__", "unknown"),
        "expected_count": len(expected),
        "available_count": len(available_set),
        "coverage_ratio": round(coverage, 6),
        "required_coverage": args.required_coverage,
        "effective": coverage >= args.required_coverage,
        "missing": missing,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"coverage={len(available_set)}/{len(expected)} ({coverage:.2%})")
    print(f"effective={output['effective']}")
    print(f"saved: {args.output}")
    return 0 if output["effective"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
