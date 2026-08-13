"""Auditable public-news intelligence for US paper-trading candidates.

The module deliberately keeps collection and scoring deterministic.  It does
not ask an LLM to guess sentiment, and unavailable sources never become a
neutral score.  Each adjustment remains traceable to the stored headline and
URL that produced it.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
import math
import re
from typing import Mapping, Sequence


INTELLIGENCE_SCORECARD_VERSION = "us_evidence_news_v3"
INTELLIGENCE_STRATEGY_VERSION = "2.1"

_POSITIVE_TERMS = {
    "earnings_beat": ("earnings beat", "beats estimates", "beat estimates", "profit beat"),
    "guidance_raise": ("raises guidance", "raised guidance", "guidance raised", "raises outlook"),
    "analyst_upgrade": ("upgraded to", "analyst upgrade", "price target raised", "initiated at buy"),
    "shareholder_return": ("share buyback", "repurchase authorization", "dividend increase"),
}
_NEGATIVE_TERMS = {
    "earnings_miss": ("earnings miss", "misses estimates", "missed estimates", "profit miss"),
    "guidance_cut": ("cuts guidance", "cut guidance", "lowers outlook", "guidance lowered"),
    "analyst_downgrade": ("downgraded to", "analyst downgrade", "price target cut"),
    "regulatory_risk": ("sec charges", "doj investigation", "antitrust lawsuit", "regulatory probe"),
    "financial_distress": ("bankruptcy", "chapter 11", "going concern", "debt default"),
    "accounting_risk": ("accounting restatement", "restates financial", "material weakness"),
}
_SEVERE_NEGATIVE_TAGS = {"regulatory_risk", "financial_distress", "accounting_risk"}
_EARNINGS_TERMS = ("earnings", "revenue", "profit", "guidance", "quarter", "10-q", "10-k")
_RESEARCH_TERMS = (
    "analyst", "upgrade", "downgrade", "price target", "initiated", "rating", "outperform",
    "overweight", "underweight",
)
_FILING_FORMS = {"8-K", "10-Q", "10-K", "6-K", "20-F", "DEF 14A"}
_CATEGORY_WEIGHTS = {
    "earnings": 1.0,
    "filing": 1.0,
    "analyst_research": 0.6,
    "company_news": 0.4,
}


class YahooUSNewsIntelligenceProvider:
    """Best-effort Yahoo/yfinance collector with an in-process per-day cache."""

    def __init__(
        self,
        *,
        news_count: int = 8,
        max_workers: int = 4,
        source_weights: Mapping[str, float] | None = None,
    ) -> None:
        self.news_count = max(1, int(news_count))
        self.max_workers = max(1, int(max_workers))
        self.source_weights = {
            **_CATEGORY_WEIGHTS,
            **{
                str(key): max(0.0, float(value))
                for key, value in dict(source_weights or {}).items()
            },
        }
        self._cache: dict[tuple[str, str], dict[str, object]] = {}

    def collect(
        self,
        tickers: Sequence[str],
        *,
        as_of: date,
    ) -> dict[str, dict[str, object]]:
        codes = list(dict.fromkeys(str(code).strip().upper() for code in tickers if str(code).strip()))
        output: dict[str, dict[str, object]] = {}
        missing = [code for code in codes if (code, as_of.isoformat()) not in self._cache]
        if missing:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(missing))) as executor:
                futures = {
                    executor.submit(self._collect_one, code, as_of): code
                    for code in missing
                }
                for future in as_completed(futures):
                    code = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = unavailable_intelligence(code, as_of, str(exc))
                    self._cache[(code, as_of.isoformat())] = result
        for code in codes:
            output[code] = self._cache[(code, as_of.isoformat())]
        return output

    def market_digest(self, *, as_of: date) -> dict[str, object]:
        """Collect a small US-market/Fed digest through the same public source."""
        try:
            import yfinance as yf

            search = yf.Search(
                "US stock market Federal Reserve earnings",
                news_count=self.news_count,
                lists_count=0,
                recommended=0,
            )
            items = _normalize_yahoo_news(getattr(search, "news", []), as_of=as_of)
            return _summarize_items("US_MARKET", as_of, items, scope="market", source_weights=self.source_weights)
        except Exception as exc:
            return unavailable_intelligence("US_MARKET", as_of, str(exc), scope="market")

    def _collect_one(self, code: str, as_of: date) -> dict[str, object]:
        import yfinance as yf

        errors: list[str] = []
        items: list[dict[str, object]] = []
        try:
            search = yf.Search(
                code,
                news_count=self.news_count,
                lists_count=0,
                recommended=0,
            )
            items.extend(_normalize_yahoo_news(getattr(search, "news", []), as_of=as_of, code=code))
        except Exception as exc:
            errors.append(f"news: {exc}")

        ticker = yf.Ticker(code)
        try:
            raw_filings = getattr(ticker, "sec_filings", None)
            if callable(raw_filings):
                raw_filings = raw_filings()
            items.extend(_normalize_sec_filings(raw_filings, as_of=as_of))
        except Exception as exc:
            errors.append(f"filings: {exc}")

        try:
            upgrades = getattr(ticker, "upgrades_downgrades", None)
            if callable(upgrades):
                upgrades = upgrades()
            items.extend(_normalize_analyst_actions(upgrades, as_of=as_of))
        except Exception as exc:
            errors.append(f"research: {exc}")

        result = _summarize_items(
            code,
            as_of,
            _dedupe_items(items),
            source_weights=self.source_weights,
        )
        if errors:
            result["errors"] = errors
            if result["status"] == "available":
                result["status"] = "partial"
        return result


def unavailable_intelligence(
    code: str,
    as_of: date,
    reason: str,
    *,
    scope: str = "company",
) -> dict[str, object]:
    return {
        "code": code,
        "scope": scope,
        "as_of": as_of.isoformat(),
        "status": "unavailable",
        "summary": "公开资讯源不可用，本次不做资讯加减分。",
        "analysis": "由于公开来源未返回可核验资料，无法判断该标的近期事件方向；技术排序保持原值。",
        "expected_impact": "暂无可验证的事件影响，下一周期不因资讯缺失改变候选。",
        "impact_horizon": "unknown",
        "impact_channels": [],
        "items": [],
        "category_counts": {},
        "positive_tags": [],
        "risk_flags": [],
        "score_adjustment": 0.0,
        "hard_exclusion": False,
        "errors": [reason[:240]],
    }


def apply_intelligence_adjustment(
    candidate: dict[str, object],
    intelligence: Mapping[str, object] | None,
) -> dict[str, object]:
    """Attach evidence and adjust only the ranking score for a future signal."""
    base_score = float(candidate.get("screen_score") or 0.0)
    intel = dict(intelligence or {})
    status = str(intel.get("status") or "unavailable")
    adjustment = float(intel.get("score_adjustment") or 0.0) if status in {"available", "partial"} else 0.0
    adjusted = min(100.0, max(0.0, base_score + adjustment))
    candidate["technical_screen_score"] = round(base_score, 4)
    candidate["intelligence_adjustment"] = round(adjustment, 4)
    candidate["screen_score"] = round(adjusted, 4)
    candidate["news_intelligence"] = intel or unavailable_intelligence(
        str(candidate.get("code") or ""),
        date.fromisoformat(str(candidate.get("signal_date") or date.today().isoformat())),
        "collector not configured",
    )
    candidate["hard_exclusion"] = bool(intel.get("hard_exclusion"))
    candidate["scorecard_version"] = INTELLIGENCE_SCORECARD_VERSION
    candidate["score_explanation"] = (
        "0-100 规则排序分：技术分叠加可审计资讯调整（-8 至 +4 分）；"
        "只用于同日同股票池比较，不是上涨概率或预期收益。"
    )
    sources = list(candidate.get("data_sources") or [])
    if status in {"available", "partial"}:
        sources.append("Yahoo Finance 新闻、SEC 披露索引与公开分析师评级变更")
    candidate["data_sources"] = list(dict.fromkeys(sources))
    data_gaps = list(candidate.get("data_gaps") or [])
    if status == "unavailable":
        data_gaps.append("新闻、财报与研报证据不可用，资讯因子未参与本次排序")
    candidate["data_gaps"] = list(dict.fromkeys(data_gaps))
    summary = str(intel.get("summary") or "").strip()
    if summary:
        candidate["selection_thesis"] = (
            str(candidate.get("selection_thesis") or "").rstrip() + " 资讯层：" + summary
        ).strip()
    risk_flags = [*list(candidate.get("risk_flags") or []), *list(intel.get("risk_flags") or [])]
    candidate["risk_flags"] = list(dict.fromkeys(str(item) for item in risk_flags if str(item).strip()))
    return candidate


def _summarize_items(
    code: str,
    as_of: date,
    items: Sequence[Mapping[str, object]],
    *,
    scope: str = "company",
    source_weights: Mapping[str, float] | None = None,
) -> dict[str, object]:
    normalized = [dict(item) for item in items]
    positive_tags: list[str] = []
    risk_flags: list[str] = []
    points = 0.0
    severe_tags: set[str] = set()
    weights = {**_CATEGORY_WEIGHTS, **dict(source_weights or {})}
    category_counts: dict[str, int] = {}
    for item in normalized:
        category = str(item.get("category") or "company_news")
        category_counts[category] = category_counts.get(category, 0) + 1
        category_weight = max(0.0, float(weights.get(category, weights["company_news"])))
        title = str(item.get("title") or "").lower()
        for tag, terms in _POSITIVE_TERMS.items():
            if any(term in title for term in terms):
                positive_tags.append(tag)
                base_points = 1.5 if tag != "analyst_upgrade" else 1.0
                points += base_points * category_weight
        for tag, terms in _NEGATIVE_TERMS.items():
            if any(term in title for term in terms):
                risk_flags.append(tag)
                base_points = 2.0 if tag not in _SEVERE_NEGATIVE_TAGS else 4.0
                points -= base_points * category_weight
                if tag in _SEVERE_NEGATIVE_TAGS:
                    severe_tags.add(tag)
    adjustment = round(min(4.0, max(-8.0, points)), 2)
    status = "available" if normalized else "unavailable"
    if status == "unavailable":
        summary = "未检索到可核验的近期公开资讯，本次不做资讯加减分。"
        analysis = "没有可核验的近期新闻、披露或评级资料，不能据此推断利好或利空。"
        expected_impact = "对下一周期选股无资讯加减分影响；仍需依赖技术证据。"
        impact_horizon = "unknown"
        impact_channels: list[str] = []
    else:
        summary = (
            f"近期待核验资料 {len(normalized)} 条，"
            f"财报/披露 {category_counts.get('earnings', 0) + category_counts.get('filing', 0)} 条，"
            f"研报/评级 {category_counts.get('analyst_research', 0)} 条，"
            f"资讯调整 {adjustment:+.1f} 分。"
        )
        risk_text = "、".join(dict.fromkeys(risk_flags)) or "未识别规则化负面标签"
        positive_text = "、".join(dict.fromkeys(positive_tags)) or "未识别规则化正面标签"
        analysis = (
            f"规则化分析：正面标签为{positive_text}；风险标签为{risk_text}；"
            f"资料覆盖 {', '.join(sorted(category_counts)) or '无'}。"
        )
        if adjustment > 0:
            expected_impact = "资讯层偏正面，下一新周期最多提高排序分至 +4 分；不直接保证价格上涨。"
        elif adjustment < 0:
            expected_impact = "资讯层偏负面，下一新周期降低排序分；若出现多类严重风险则排除候选。"
        else:
            expected_impact = "资讯层未形成净调整，候选仍主要由技术证据排序。"
        impact_horizon = "short_term" if any(
            tag in positive_tags or tag in risk_flags
            for tag in ["earnings_beat", "earnings_miss", "guidance_raise", "guidance_cut", "regulatory_risk"]
        ) else "medium_term"
        combined_tags = positive_tags + risk_flags
        impact_channels = [
            label for label, enabled in (
                ("盈利预期", any(tag in combined_tags for tag in ["earnings_beat", "earnings_miss", "guidance_raise", "guidance_cut"])),
                ("估值与资金偏好", any(tag in combined_tags for tag in ["analyst_upgrade", "analyst_downgrade"])),
                ("监管与治理风险", any(tag in risk_flags for tag in _SEVERE_NEGATIVE_TAGS)),
                ("股东回报", "shareholder_return" in positive_tags),
            ) if enabled
        ]
    return {
        "code": code,
        "scope": scope,
        "as_of": as_of.isoformat(),
        "status": status,
        "summary": summary,
        "analysis": analysis,
        "expected_impact": expected_impact,
        "impact_horizon": impact_horizon,
        "impact_channels": impact_channels,
        "items": normalized[:12],
        "category_counts": category_counts,
        "positive_tags": list(dict.fromkeys(positive_tags)),
        "risk_flags": list(dict.fromkeys(risk_flags)),
        "score_adjustment": adjustment if status == "available" else 0.0,
        "hard_exclusion": len(severe_tags) >= 2,
        "errors": [],
    }


def _normalize_yahoo_news(
    raw_items: object,
    *,
    as_of: date,
    code: str | None = None,
) -> list[dict[str, object]]:
    if not isinstance(raw_items, list):
        return []
    output = []
    for raw in raw_items:
        if not isinstance(raw, Mapping):
            continue
        content = raw.get("content") if isinstance(raw.get("content"), Mapping) else raw
        if code and not _news_item_matches_code(raw, content, code):
            continue
        title = str(content.get("title") or "").strip()
        if not title:
            continue
        published_at = _iso_datetime(content.get("pubDate") or content.get("providerPublishTime"))
        if published_at and published_at.date() > as_of:
            continue
        provider = content.get("provider")
        if isinstance(provider, Mapping):
            source = str(provider.get("displayName") or provider.get("name") or "Yahoo Finance")
        else:
            source = str(content.get("publisher") or provider or "Yahoo Finance")
        url = _first_url(
            content.get("canonicalUrl"), content.get("clickThroughUrl"), content.get("link")
        )
        lower = title.lower()
        category = "company_news"
        if any(term in lower for term in _EARNINGS_TERMS):
            category = "earnings"
        if any(term in lower for term in _RESEARCH_TERMS):
            category = "analyst_research"
        annotations = _item_annotations(title, category)
        output.append({
            "category": category,
            "title": title[:300],
            "summary": _first_text(content.get("description"), content.get("summary"), content.get("text")),
            "url": url,
            "source": source[:120],
            "published_at": published_at.isoformat() if published_at else None,
            **annotations,
        })
    return output


def _news_item_matches_code(
    raw: Mapping[str, object],
    content: Mapping[str, object],
    code: str,
) -> bool:
    """Reject explicitly unrelated Yahoo stories before scoring them."""
    normalized_code = code.upper().strip()
    metadata_values: list[object] = []
    for payload in (raw, content):
        metadata_values.extend([
            payload.get("relatedTickers"),
            payload.get("related_tickers"),
            payload.get("tickerSymbols"),
            payload.get("ticker_symbols"),
            payload.get("symbol"),
            payload.get("ticker"),
        ])
        finance = payload.get("finance")
        if isinstance(finance, Mapping):
            metadata_values.extend([
                finance.get("tickerSymbols"),
                finance.get("ticker_symbols"),
            ])
    explicit_symbols: set[str] = set()
    for value in metadata_values:
        values = value if isinstance(value, (list, tuple, set)) else [value]
        for item in values:
            text = str(item or "").strip().upper()
            if text:
                explicit_symbols.add(text)
    return not explicit_symbols or normalized_code in explicit_symbols


def _normalize_sec_filings(raw: object, *, as_of: date) -> list[dict[str, object]]:
    records: list[Mapping[str, object]] = []
    if isinstance(raw, list):
        records = [item for item in raw if isinstance(item, Mapping)]
    elif hasattr(raw, "to_dict"):
        try:
            records = list(raw.reset_index().to_dict("records"))
        except Exception:
            records = []
    output = []
    for item in records[:20]:
        form = str(item.get("type") or item.get("form") or item.get("Form") or "").upper()
        if form not in _FILING_FORMS:
            continue
        filed = _date_value(item.get("date") or item.get("filingDate") or item.get("Date"))
        if filed and (filed > as_of or filed < as_of - timedelta(days=45)):
            continue
        title = str(item.get("title") or item.get("description") or f"SEC {form} filing")
        annotations = _item_annotations(title, "earnings" if form in {"10-Q", "10-K", "20-F"} else "filing")
        output.append({
            "category": "earnings" if form in {"10-Q", "10-K", "20-F"} else "filing",
            "title": f"{form}: {title}"[:300],
            "summary": str(item.get("description") or "").strip()[:800],
            "url": _first_url(item.get("edgarUrl"), item.get("url"), item.get("link")),
            "source": "SEC EDGAR",
            "published_at": filed.isoformat() if filed else None,
            **annotations,
        })
    return output


def _normalize_analyst_actions(raw: object, *, as_of: date) -> list[dict[str, object]]:
    if raw is None or not hasattr(raw, "reset_index"):
        return []
    try:
        records = raw.reset_index().to_dict("records")
    except Exception:
        return []
    output = []
    for item in records[:30]:
        action_date = _date_value(item.get("GradeDate") or item.get("Date") or item.get("index"))
        if action_date and (action_date > as_of or action_date < as_of - timedelta(days=30)):
            continue
        firm = str(item.get("Firm") or item.get("firm") or "分析师")
        raw_action = str(item.get("Action") or item.get("action") or "更新评级").strip()
        action_key = raw_action.lower()
        action = {
            "up": "analyst upgrade",
            "down": "analyst downgrade",
            "init": "initiated",
        }.get(action_key, raw_action)
        from_grade = str(item.get("FromGrade") or item.get("fromGrade") or "").strip()
        to_grade = str(item.get("ToGrade") or item.get("toGrade") or "").strip()
        title = " ".join(value for value in [firm, action, from_grade, "to" if from_grade and to_grade else "", to_grade] if value)
        annotations = _item_annotations(title, "analyst_research")
        output.append({
            "category": "analyst_research",
            "title": title[:300],
            "summary": "公开分析师评级变更记录；具体观点以来源页面为准。",
            "url": "",
            "source": "Yahoo Finance analyst actions",
            "published_at": action_date.isoformat() if action_date else None,
            **annotations,
        })
    return output


def _first_text(*values: object) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text[:800]
    return ""


def _item_annotations(title: str, category: str) -> dict[str, object]:
    """Attach deterministic, item-level interpretation without inventing facts."""
    lower = title.lower()
    positive = [tag for tag, terms in _POSITIVE_TERMS.items() if any(term in lower for term in terms)]
    negative = [tag for tag, terms in _NEGATIVE_TERMS.items() if any(term in lower for term in terms)]
    tags = positive + negative
    channels = [label for label, enabled in (
        ("盈利预期", any(tag in tags for tag in ("earnings_beat", "earnings_miss", "guidance_raise", "guidance_cut"))),
        ("估值与资金偏好", any(tag in tags for tag in ("analyst_upgrade", "analyst_downgrade"))),
        ("监管与治理风险", any(tag in negative for tag in _SEVERE_NEGATIVE_TAGS)),
        ("股东回报", "shareholder_return" in positive),
        ("披露透明度", category in {"filing", "earnings"} and not tags),
    ) if enabled]
    if negative:
        sentiment = "偏负面"
        analysis = f"规则化识别到负面标签：{'、'.join(negative)}；仅作为事件证据，不代表价格必然下跌。"
        expected = "短期可能压制风险偏好或盈利预期，需结合后续价格和公司原文复核。"
    elif positive:
        sentiment = "偏正面"
        analysis = f"规则化识别到正面标签：{'、'.join(positive)}；仅作为事件证据，不代表价格必然上涨。"
        expected = "短期可能改善盈利预期、估值或资金偏好，实际影响取决于市场定价。"
    else:
        sentiment = "中性/待核验"
        analysis = "标题未命中预设正负面词表，未对事实作额外推断。"
        expected = "暂不改变排序分，等待原文、后续披露和价格反应。"
    return {
        "sentiment": sentiment,
        "tags": tags,
        "analysis": analysis,
        "expected_impact": expected,
        "impact_channels": channels,
        "impact_horizon": "short_term" if any(tag in tags for tag in ("earnings_beat", "earnings_miss", "guidance_raise", "guidance_cut", "regulatory_risk")) else "medium_term",
    }


def _iso_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _date_value(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except ValueError:
        return None


def _first_url(*values: object) -> str:
    for value in values:
        if isinstance(value, Mapping):
            value = value.get("url")
        text = str(value or "").strip()
        if text.startswith(("http://", "https://")):
            return text
    return ""


def _dedupe_items(items: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    seen: set[str] = set()
    output = []
    for item in items:
        key = re.sub(r"\W+", " ", str(item.get("title") or "").lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(dict(item))
    output.sort(key=lambda item: str(item.get("published_at") or ""), reverse=True)
    return output
