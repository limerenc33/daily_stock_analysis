from datetime import date

import pandas as pd
import pytest

from src.services.screening.us_point_in_time import PointInTimeUniverse, normalize_yahoo_ticker


def test_point_in_time_universe_never_looks_ahead(tmp_path):
    source = tmp_path / "members.csv"
    source.write_text(
        "date,tickers\n"
        '2019-01-01,"AAA,BRK.B"\n'
        '2020-01-01,"AAA,CCC"\n',
        encoding="utf-8",
    )
    universe = PointInTimeUniverse.from_csv(source)

    assert universe.constituents_on(date(2019, 12, 31)) == frozenset({"AAA", "BRK-B"})
    assert universe.constituents_on(date(2020, 1, 1)) == frozenset({"AAA", "CCC"})
    with pytest.raises(ValueError, match="no point-in-time universe snapshot"):
        universe.constituents_on(date(2018, 12, 31))


def test_point_in_time_coverage_requires_history_as_of_date(tmp_path):
    source = tmp_path / "members.csv"
    source.write_text('date,tickers\n2019-01-01,"AAA,BBB"\n', encoding="utf-8")
    universe = PointInTimeUniverse.from_csv(source)
    histories = {
        "AAA": pd.DataFrame({"date": pd.bdate_range("2019-01-01", periods=70)}),
        "BBB": pd.DataFrame({"date": pd.bdate_range("2019-03-01", periods=70)}),
    }

    coverage = universe.coverage_on(
        date(2019, 4, 1), histories, minimum_history_points=60
    )

    assert coverage["expected_count"] == 2
    assert coverage["available_count"] == 1
    assert coverage["coverage_ratio"] == 0.5
    assert coverage["missing"] == ["BBB"]


def test_normalize_yahoo_ticker_handles_class_shares():
    assert normalize_yahoo_ticker(" brk.b ") == "BRK-B"
