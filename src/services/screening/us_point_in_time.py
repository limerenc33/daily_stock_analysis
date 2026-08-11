"""Point-in-time US equity universe parsing and coverage checks."""

from __future__ import annotations

from bisect import bisect_right
import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping

import pandas as pd


def normalize_yahoo_ticker(value: object) -> str:
    """Normalize class-share tickers to Yahoo's dash convention."""
    return str(value or "").strip().upper().replace(".", "-")


@dataclass(frozen=True)
class PointInTimeUniverse:
    """Immutable dated constituent snapshots with no future lookup."""

    dates: tuple[date, ...]
    members: tuple[frozenset[str], ...]
    source: str = ""

    @classmethod
    def from_csv(cls, path: str | Path) -> "PointInTimeUniverse":
        source_path = Path(path)
        snapshots: list[tuple[date, frozenset[str]]] = []
        with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not {"date", "tickers"}.issubset(reader.fieldnames):
                raise ValueError("point-in-time universe CSV must contain date and tickers columns")
            for line_number, row in enumerate(reader, start=2):
                try:
                    snapshot_date = date.fromisoformat(str(row.get("date") or "").strip())
                except ValueError as exc:
                    raise ValueError(f"invalid universe date on line {line_number}") from exc
                tickers = frozenset(
                    normalized
                    for raw in str(row.get("tickers") or "").split(",")
                    if (normalized := normalize_yahoo_ticker(raw))
                )
                if not tickers:
                    raise ValueError(f"empty universe on line {line_number}")
                snapshots.append((snapshot_date, tickers))
        if not snapshots:
            raise ValueError("point-in-time universe CSV is empty")
        if any(current[0] <= previous[0] for previous, current in zip(snapshots, snapshots[1:])):
            raise ValueError("point-in-time universe dates must be strictly increasing")
        return cls(
            dates=tuple(item[0] for item in snapshots),
            members=tuple(item[1] for item in snapshots),
            source=str(source_path),
        )

    def constituents_on(self, as_of: date) -> frozenset[str]:
        """Return the latest membership published on or before ``as_of``."""
        position = bisect_right(self.dates, as_of) - 1
        if position < 0:
            raise ValueError(f"no point-in-time universe snapshot on or before {as_of}")
        return self.members[position]

    def coverage_on(
        self,
        as_of: date,
        histories: Mapping[str, pd.DataFrame],
        *,
        minimum_history_points: int = 60,
    ) -> dict[str, object]:
        """Measure usable price-history coverage for an as-of membership."""
        if minimum_history_points <= 0:
            raise ValueError("minimum_history_points must be positive")
        expected = self.constituents_on(as_of)
        available: list[str] = []
        missing: list[str] = []
        for ticker in sorted(expected):
            frame = histories.get(ticker)
            if frame is None or frame.empty:
                missing.append(ticker)
                continue
            if "date" in frame.columns:
                dates = pd.to_datetime(frame["date"], errors="coerce").dt.date
            else:
                dates = pd.to_datetime(frame.index, errors="coerce").date
            observations = sum(value <= as_of for value in dates if pd.notna(value))
            (available if observations >= minimum_history_points else missing).append(ticker)
        expected_count = len(expected)
        return {
            "as_of": as_of.isoformat(),
            "expected_count": expected_count,
            "available_count": len(available),
            "coverage_ratio": round(len(available) / expected_count, 6) if expected_count else 0.0,
            "minimum_history_points": minimum_history_points,
            "available": available,
            "missing": missing,
        }
