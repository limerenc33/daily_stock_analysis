"""Auditable evidence cards for deterministic US paper-trading candidates."""

from __future__ import annotations

import math
from typing import Mapping

from src.services.screening.models import ScreeningConfig
from src.services.screening.scorer import factor_score_columns


FACTOR_LABELS = {
    "trend_confirmation": "趋势确认",
    "momentum": "动量质量",
    "risk_control": "风险控制",
    "liquidity": "流动性容量",
    "relative_strength": "相对强度",
    "data_quality": "数据质量",
}


def build_us_candidate_evidence(
    row: Mapping[str, object],
    config: ScreeningConfig,
) -> dict[str, object]:
    """Build a JSON-safe, rule-grounded explanation for one ranked row."""
    weights = _normalized_weights(config.factor_weights)
    columns = factor_score_columns()
    factor_scores = {
        factor: _rounded_number(row.get(columns.get(factor, "")))
        for factor in weights
    }
    metrics = {
        "change_pct": _rounded_number(row.get("change_pct")),
        "change_60d_pct": _rounded_number(row.get("change_60d")),
        "signal_score": _rounded_number(row.get("signal_score")),
        "volume_ratio_20d": _rounded_number(row.get("volume_ratio_20d")),
        "volatility_20d_pct": _rounded_number(row.get("volatility_20d_pct")),
        "max_drawdown_20d_pct": _rounded_number(row.get("max_drawdown_20d_pct")),
        "atr_20_pct": _rounded_number(row.get("atr_20_pct")),
        "daily_quality_score": _rounded_number(row.get("daily_quality_score")),
        "price_above_ma20": _safe_bool(row.get("price_above_ma20")),
        "ma_bullish": _safe_bool(row.get("ma_bullish")),
        "macd_status": _safe_text(row.get("macd_status")),
        "rsi_status": _safe_text(row.get("rsi_status")),
    }
    reasons_pass = _pass_reasons(row, config)
    reasons_watch = _watch_reasons(row)
    risk_flags = _risk_flags(row, config)
    data_gaps = _data_gaps(row)
    for gap in data_gaps:
        if gap not in reasons_watch:
            reasons_watch.append(gap)

    top_factors = sorted(
        (
            (factor, float(score) * weights.get(factor, 0.0))
            for factor, score in factor_scores.items()
            if score is not None
        ),
        key=lambda item: item[1],
        reverse=True,
    )[:2]
    strengths = "、".join(FACTOR_LABELS.get(name, name) for name, _ in top_factors)
    if not strengths:
        strengths = "规则证据"
    caution = risk_flags[0] if risk_flags else "未触发额外技术风险标记"
    thesis = f"{strengths}贡献居前并通过全部硬筛选；{caution}。"
    if data_gaps:
        thesis += " 基本面未验证，因此只作为模拟候选。"

    hard_filters = config.hard_filters
    invalidation = [
        "收盘价跌破 MA20 或 MACD 转为 bearish",
    ]
    risk_limits = []
    if hard_filters.volatility_20d_pct_max is not None:
        risk_limits.append(f"20 日波动率高于 {hard_filters.volatility_20d_pct_max:.1f}%")
    if hard_filters.max_drawdown_20d_pct_min is not None:
        risk_limits.append(f"20 日回撤低于 {hard_filters.max_drawdown_20d_pct_min:.1f}%")
    if hard_filters.atr_20_pct_max is not None:
        risk_limits.append(f"ATR 高于 {hard_filters.atr_20_pct_max:.1f}%")
    if risk_limits:
        invalidation.append("、".join(risk_limits))
    invalidation.append("行情数据质量低于可审计要求")

    scorecard = config.scorecard_profile or {}
    return {
        "scorecard_version": str(scorecard.get("version") or "us_evidence_v2"),
        "score_method": "weighted_rule_score_0_100",
        "score_explanation": (
            "0-100 规则排序分，只用于同日同股票池比较，不是上涨概率或预期收益。"
        ),
        "factor_scores": factor_scores,
        "factor_weights": {key: round(value, 4) for key, value in weights.items()},
        "evidence_metrics": metrics,
        "reasons_pass": reasons_pass,
        "reasons_watch": reasons_watch,
        "risk_flags": risk_flags,
        "selection_thesis": thesis,
        "invalidation_conditions": invalidation,
        "data_sources": ["Yahoo Finance 复权 OHLCV 日线（收盘后信号）"],
        "data_gaps": data_gaps,
    }


def _pass_reasons(
    row: Mapping[str, object],
    config: ScreeningConfig,
) -> list[str]:
    reasons: list[str] = []
    price = _number(row.get("price"))
    ma20 = _number(row.get("ma20"))
    if _safe_bool(row.get("price_above_ma20")):
        if price is not None and ma20 is not None:
            reasons.append(f"收盘价 {price:.2f} 站上 MA20 {ma20:.2f}")
        else:
            reasons.append("收盘价站上 MA20")
    if _safe_bool(row.get("ma_bullish")):
        reasons.append("MA5 >= MA20 >= MA60，均线结构偏多")

    macd = _safe_text(row.get("macd_status"))
    if macd in {"bullish", "neutral"}:
        reasons.append(f"MACD 为 {macd}，满足趋势白名单")

    signal = _number(row.get("signal_score"))
    minimum_signal = config.hard_filters.signal_score_min
    if signal is not None and minimum_signal is not None:
        reasons.append(f"趋势信号 {signal:.1f}，高于门槛 {minimum_signal}")

    change_60d = _number(row.get("change_60d"))
    if change_60d is not None:
        reasons.append(f"60 日涨跌幅 {change_60d:+.1f}%，处于规则允许区间")

    volatility = _number(row.get("volatility_20d_pct"))
    drawdown = _number(row.get("max_drawdown_20d_pct"))
    atr = _number(row.get("atr_20_pct"))
    if volatility is not None and drawdown is not None and atr is not None:
        reasons.append(
            f"20 日波动率 {volatility:.1f}%、回撤 {drawdown:.1f}%、ATR {atr:.1f}% 均未触发硬风控"
        )
    return reasons[:6]


def _watch_reasons(row: Mapping[str, object]) -> list[str]:
    reasons: list[str] = []
    if not _safe_bool(row.get("ma_bullish")):
        reasons.append("尚未形成 MA5 >= MA20 >= MA60 的完整多头排列")
    if _safe_text(row.get("macd_status")) == "neutral":
        reasons.append("MACD 仅为 neutral，需要观察是否转强")
    change_60d = _number(row.get("change_60d"))
    if change_60d is not None and change_60d >= 35.0:
        reasons.append(f"60 日涨幅已达 {change_60d:.1f}%，存在动量过热风险")
    quality = _number(row.get("daily_quality_score"))
    if quality is not None and quality < 100.0:
        reasons.append(f"日线数据质量分为 {quality:.1f}，需核对降级标记")
    return reasons


def _risk_flags(
    row: Mapping[str, object],
    config: ScreeningConfig,
) -> list[str]:
    flags: list[str] = []
    risk = config.risk_profile or {}
    change = _number(row.get("change_pct"))
    chase_level = _number(risk.get("chase_change_pct"))
    if change is not None and chase_level is not None and change >= chase_level * 0.75:
        flags.append(f"当日涨幅 {change:+.1f}%，接近追高警戒线 {chase_level:.1f}%")
    if _safe_text(row.get("rsi_status")) == "overbought":
        flags.append("RSI 处于 overbought，短线回撤概率可能上升")

    hard_filters = config.hard_filters
    volatility = _number(row.get("volatility_20d_pct"))
    if (
        volatility is not None
        and hard_filters.volatility_20d_pct_max is not None
        and volatility >= hard_filters.volatility_20d_pct_max * 0.8
    ):
        flags.append(
            f"20 日波动率 {volatility:.1f}%，接近上限 {hard_filters.volatility_20d_pct_max:.1f}%"
        )
    drawdown = _number(row.get("max_drawdown_20d_pct"))
    if (
        drawdown is not None
        and hard_filters.max_drawdown_20d_pct_min is not None
        and drawdown <= hard_filters.max_drawdown_20d_pct_min * 0.8
    ):
        flags.append(
            f"20 日回撤 {drawdown:.1f}%，接近下限 {hard_filters.max_drawdown_20d_pct_min:.1f}%"
        )
    atr = _number(row.get("atr_20_pct"))
    if (
        atr is not None
        and hard_filters.atr_20_pct_max is not None
        and atr >= hard_filters.atr_20_pct_max * 0.8
    ):
        flags.append(f"ATR {atr:.1f}%，接近上限 {hard_filters.atr_20_pct_max:.1f}%")
    quality_flags = _safe_text(row.get("daily_quality_flags"))
    if quality_flags:
        flags.append(f"行情质量标记：{quality_flags}")
    return flags


def _data_gaps(row: Mapping[str, object]) -> list[str]:
    fields = ("pe_ratio", "pb_ratio", "total_mv")
    if any(_number(row.get(field)) is None for field in fields):
        return ["估值、市值与财务质量未接入本次日线回放，未参与评分"]
    return []


def _normalized_weights(raw_weights: Mapping[str, object]) -> dict[str, float]:
    weights = {
        str(name): max(float(value), 0.0)
        for name, value in raw_weights.items()
        if str(name) in FACTOR_LABELS
    }
    total = sum(weights.values())
    if total <= 0:
        return {}
    return {name: value / total for name, value in weights.items()}


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _rounded_number(value: object) -> float | None:
    number = _number(value)
    return None if number is None else round(number, 4)


def _safe_bool(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    try:
        number = float(value)
    except (TypeError, ValueError):
        try:
            return bool(value)
        except TypeError:
            return False
    return math.isfinite(number) and bool(number)


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "<na>"} else text
