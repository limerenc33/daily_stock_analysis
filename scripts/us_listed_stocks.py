"""Current US exchange-listed ordinary shares for forward-only paper signals."""

import csv
from datetime import date, datetime, timezone
from io import StringIO
import re

import exchange_calendars as xcals
import requests


DIRECTORY_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqtraded.txt"
COMMON = re.compile(
    r"\b(common (stock|shares?)|ordinary shares?|capital stock|subordinate voting shares?)\b", re.I
)
OTHER = re.compile(
    r"\b(preferred|preference|warrants?|rights?|units?|depositary|depository|fund|trust|portfolio|beneficial interest|notes?|"
    r"debentures?|bonds?|ETF|ETN)\b", re.I
)


def current_symbols(session: date) -> list[str]:
    # A live directory cannot reconstruct historical membership after the next open.
    calendar = xcals.get_calendar("XNYS")
    now = datetime.now(timezone.utc)
    if not (calendar.session_close(session.isoformat()) <= now <
            calendar.session_open(calendar.next_session(session.isoformat()))):
        raise ValueError("当前上市目录不适用于该信号日")
    response = requests.get(DIRECTORY_URL, timeout=20)
    response.raise_for_status()
    directory = response.text
    footer = directory.rstrip().rsplit("\n", 1)[-1]
    stamp = re.match(r"File Creation Time:\s*(\d{8})", footer)
    if not stamp or datetime.strptime(stamp[1], "%m%d%Y").date() < session:
        raise ValueError("Nasdaq 上市目录日期早于信号日或缺少日期")
    return ordinary_symbols(directory)


def ordinary_symbols(directory: str) -> list[str]:
    footer = directory.rstrip().rsplit("\n", 1)[-1]
    if not footer.startswith("File Creation Time:"):
        raise ValueError("Nasdaq 上市目录不完整")
    symbols = set()
    for row in csv.DictReader(StringIO(directory), delimiter="|"):
        symbol = row["Symbol"]
        name = row["Security Name"]
        if (row["Nasdaq Traded"] == "Y" and row["ETF"] == "N" and row["Test Issue"] == "N"
                and COMMON.search(name) and not OTHER.search(name)
                and re.fullmatch(r"[A-Z]+(?:\.[A-Z]+)?", symbol)):
            symbols.add(symbol.replace(".", "-"))
    if not symbols:
        raise ValueError("Nasdaq 上市目录没有可识别的普通股")
    return sorted(symbols)
