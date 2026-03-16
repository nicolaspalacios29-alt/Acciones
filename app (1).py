from __future__ import annotations

import asyncio
import math
import os
import time
from statistics import median
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# ============================================================
# CONFIG
# ============================================================

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "300"))

DEFAULT_SYMBOLS = [
    "AAPL", "GOOGL", "AMZN", "NVDA", "MSFT", "TSM", "AMD", "CAT", "DE", "KO", "V", "MA",
    "AXP", "BRK.B", "TSLA", "META", "UNH", "MELI", "AVGO", "JPM", "PG", "JNJ", "GS", "CVX",
    "HD", "WMT", "MCD", "F", "C", "BAC", "WFC", "NFLX", "T",
]

INDEX_SYMBOLS = {
    "SPY": "S&P 500",
    "QQQ": "NASDAQ-100",
    "DIA": "Dow Jones",
}

SECTOR_MAP = {
    "AAPL": "Technology",
    "GOOGL": "Communication",
    "AMZN": "Consumer",
    "NVDA": "Technology",
    "MSFT": "Technology",
    "TSM": "Technology",
    "AMD": "Technology",
    "CAT": "Industrials",
    "DE": "Industrials",
    "KO": "Consumer",
    "V": "Financials",
    "MA": "Financials",
    "AXP": "Financials",
    "BRK.B": "Financials",
    "TSLA": "Consumer",
    "META": "Communication",
    "UNH": "Healthcare",
    "MELI": "Consumer",
    "AVGO": "Technology",
    "JPM": "Financials",
    "PG": "Consumer",
    "JNJ": "Healthcare",
    "GS": "Financials",
    "CVX": "Energy",
    "HD": "Consumer",
    "WMT": "Consumer",
    "MCD": "Consumer",
    "F": "Consumer",
    "C": "Financials",
    "BAC": "Financials",
    "WFC": "Financials",
    "NFLX": "Communication",
    "T": "Communication",
}

STYLE_MAP = {
    "AAPL": "quality",
    "GOOGL": "quality",
    "AMZN": "growth",
    "NVDA": "growth",
    "MSFT": "quality",
    "TSM": "growth",
    "AMD": "growth",
    "CAT": "value",
    "DE": "quality",
    "KO": "defensive",
    "V": "quality",
    "MA": "quality",
    "AXP": "value",
    "BRK.B": "quality",
    "TSLA": "growth",
    "META": "quality",
    "UNH": "defensive",
    "MELI": "growth",
    "AVGO": "quality",
    "JPM": "value",
    "PG": "defensive",
    "JNJ": "defensive",
    "GS": "value",
    "CVX": "value",
    "HD": "quality",
    "WMT": "defensive",
    "MCD": "defensive",
    "F": "value",
    "C": "value",
    "BAC": "value",
    "WFC": "value",
    "NFLX": "growth",
    "T": "defensive",
}

app = FastAPI(title="Equity Dashboard Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_CACHE: dict[str, tuple[float, Any]] = {}


# ============================================================
# HELPERS
# ============================================================

def safe_num(value: Any, fallback: float = 0.0) -> float:
    try:
        n = float(value)
        if math.isfinite(n):
            return n
    except (TypeError, ValueError):
        pass
    return fallback


def safe_div(a: float, b: float, fallback: float = 0.0) -> float:
    if not math.isfinite(a) or not math.isfinite(b) or b == 0:
        return fallback
    return a / b


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def scale(value: float, min_value: float, max_value: float) -> float:
    if max_value == min_value:
        return 0.0
    return (value - min_value) / (max_value - min_value)


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def cache_get(key: str) -> Any | None:
    item = _CACHE.get(key)
    if not item:
        return None
    ts, data = item
    if time.time() - ts > CACHE_TTL_SECONDS:
        _CACHE.pop(key, None)
        return None
    return data


def cache_set(key: str, value: Any) -> None:
    _CACHE[key] = (time.time(), value)


def ema(values: list[float], period: int) -> list[float]:
    if not values:
        return []
    k = 2 / (period + 1)
    out: list[float] = []
    prev = values[0]
    for i, v in enumerate(values):
        current = v if i == 0 else v * k + prev * (1 - k)
        out.append(current)
        prev = current
    return out


def sma(values: list[float], period: int) -> list[float | None]:
    out: list[float | None] = []
    for i in range(len(values)):
        if i < period - 1:
            out.append(None)
        else:
            window = values[i - period + 1 : i + 1]
            out.append(sum(window) / period)
    return out


def rsi(values: list[float], period: int = 14) -> list[float]:
    if len(values) < period + 1:
        return [50.0 for _ in values]

    output = [50.0 for _ in values]
    gains = 0.0
    losses = 0.0

    for i in range(1, period + 1):
        diff = values[i] - values[i - 1]
        if diff >= 0:
            gains += diff
        else:
            losses -= diff

    avg_gain = gains / period
    avg_loss = losses / period
    output[period] = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1 + avg_gain / avg_loss)

    for i in range(period + 1, len(values)):
        diff = values[i] - values[i - 1]
        gain = max(diff, 0.0)
        loss = max(-diff, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        output[i] = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1 + avg_gain / avg_loss)

    return output


def macd(values: list[float]) -> tuple[list[float], list[float], list[float]]:
    ema12 = ema(values, 12)
    ema26 = ema(values, 26)
    line = [a - b for a, b in zip(ema12, ema26)]
    signal = ema(line, 9)
    hist = [a - b for a, b in zip(line, signal)]
    return line, signal, hist


def calc_vwap(candles: list[dict[str, Any]]) -> list[float]:
    cumulative_pv = 0.0
    cumulative_v = 0.0
    result: list[float] = []
    for c in candles:
        typical = (c["high"] + c["low"] + c["close"]) / 3
        cumulative_pv += typical * c["volume"]
        cumulative_v += c["volume"]
        result.append(safe_div(cumulative_pv, cumulative_v, c["close"]))
    return result


def calc_relative_volume(volumes: list[float], lookback: int = 30) -> float:
    if len(volumes) < 2:
        return 1.0
    last = volumes[-1]
    base = avg(volumes[max(0, len(volumes) - lookback - 1) : len(volumes) - 1])
    return safe_div(last, base, 1.0)


def calc_momentum(prices: list[float], lookback: int) -> float:
    if not prices:
        return 0.0
    last = prices[-1]
    prev = prices[max(0, len(prices) - 1 - lookback)]
    return safe_div(last, prev, 1.0) - 1.0


def trend_label(price: float, ema20_value: float, sma50_value: float, sma200_value: float) -> str:
    if price > ema20_value and ema20_value > sma50_value and sma50_value > sma200_value:
        return "Alcista"
    if price < ema20_value and ema20_value < sma50_value and sma50_value < sma200_value:
        return "Bajista"
    return "Lateral"


def decision_label(score: int, margin_of_safety: float, forecast_quality: int) -> str:
    if score >= 78 and margin_of_safety >= 0.12 and forecast_quality >= 70:
        return "Comprar"
    if score >= 66 and margin_of_safety >= 0.05:
        return "Acumular"
    if score >= 55:
        return "Vigilar"
    return "Esperar"


# ============================================================
# FINNHUB
# ============================================================

async def finnhub_get(client: httpx.AsyncClient, endpoint: str, **params: Any) -> Any:
    if not FINNHUB_API_KEY:
        raise HTTPException(status_code=500, detail="FINNHUB_API_KEY no configurada en variables de entorno")

    url = f"https://finnhub.io/api/v1{endpoint}"
    response = await client.get(url, params={**params, "token": FINNHUB_API_KEY}, timeout=30.0)
    response.raise_for_status()
    return response.json()


async def get_quote(client: httpx.AsyncClient, symbol: str) -> dict[str, Any]:
    return await finnhub_get(client, "/quote", symbol=symbol)


async def get_profile(client: httpx.AsyncClient, symbol: str) -> dict[str, Any]:
    return await finnhub_get(client, "/stock/profile2", symbol=symbol)


async def get_metrics(client: httpx.AsyncClient, symbol: str) -> dict[str, Any]:
    data = await finnhub_get(client, "/stock/metric", symbol=symbol, metric="all")
    return data.get("metric", {}) if isinstance(data, dict) else {}


async def get_candles(client: httpx.AsyncClient, symbol: str, count_days: int = 320) -> dict[str, Any]:
    now_ts = int(time.time())
    from_ts = now_ts - count_days * 86400
    return await finnhub_get(
        client,
        "/stock/candle",
        symbol=symbol,
        resolution="D",
        **{"from": from_ts, "to": now_ts},
    )


# ============================================================
# MODEL
# ============================================================

def calc_dcf(row: dict[str, Any]) -> float:
    revenue0 = safe_num(row.get("revenue"), 1.0)
    margin0 = safe_num(row.get("ebitMargin"), 0.12)
    tax = 0.18
    capex_ratio = safe_num(row.get("capexToSales"), 0.08)
    wc_ratio = safe_num(row.get("workingCapitalToSales"), 0.03)
    growth = safe_num(row.get("revenueGrowthYoy"), 0.05)
    margin_step = safe_num(row.get("marginExpansion"), 0.003)
    wacc = max(0.06, safe_num(row.get("wacc"), 0.09))
    terminal_growth = min(0.035, max(0.02, safe_num(row.get("longTermGrowth"), 0.025)))

    revenue = revenue0
    pv = 0.0
    for y in range(1, 6):
        revenue *= 1 + max(growth - (y - 1) * 0.015, terminal_growth)
        margin = min(0.42, margin0 + y * margin_step)
        ebit = revenue * margin
        nopat = ebit * (1 - tax)
        capex = revenue * capex_ratio
        wc = revenue * wc_ratio * 0.25
        fcf = nopat - capex - wc
        pv += fcf / ((1 + wacc) ** y)

    terminal_fcf = (
        revenue * min(0.42, margin0 + 5 * margin_step) * (1 - tax)
        - revenue * capex_ratio
        - revenue * wc_ratio * 0.25
    ) * (1 + terminal_growth)
    terminal_value = terminal_fcf / max(0.01, wacc - terminal_growth)
    terminal_pv = terminal_value / ((1 + wacc) ** 5)
    equity_value = pv + terminal_pv - safe_num(row.get("netDebt"), 0.0)

    return round(safe_div(equity_value, safe_num(row.get("sharesOutstanding"), 1.0), 0.0), 2)


def calc_multiple_fair_value(row: dict[str, Any], peer_median: dict[str, float]) -> float:
    target_pe = min(
        safe_num(row.get("peForward"), 18.0) * 1.08,
        safe_num(peer_median.get("pe"), safe_num(row.get("peForward"), 18.0)),
    )
    target_ev = min(
        safe_num(row.get("evToEbitda"), 12.0) * 1.08,
        safe_num(peer_median.get("evEbitda"), safe_num(row.get("evToEbitda"), 12.0)),
    )
    pe_price = safe_num(row.get("epsForward"), 1.0) * target_pe
    ev_price = safe_div(
        (safe_num(row.get("ebitda"), 0.0) * target_ev) - safe_num(row.get("netDebt"), 0.0),
        safe_num(row.get("sharesOutstanding"), 1.0),
        0.0,
    )
    return round(0.5 * pe_price + 0.5 * ev_price, 2)


def calc_forecast_quality(row: dict[str, Any]) -> int:
    size = clamp(scale(safe_num(row.get("marketCap"), 0.0), 50e9, 3e12), 0.0, 1.0)
    stability = 1 - clamp(scale(safe_num(row.get("revenueVolatility"), 0.10), 0.02, 0.30), 0.0, 1.0)
    leverage = 1 - clamp(scale(safe_num(row.get("netDebtToEbitda"), 0.0), 0.0, 4.0), 0.0, 1.0)
    margin = clamp(scale(safe_num(row.get("ebitdaMargin"), 0.10), 0.08, 0.45), 0.0, 1.0)
    return round((size * 0.30 + stability * 0.25 + leverage * 0.20 + margin * 0.25) * 100)


def calc_composite_score(row: dict[str, Any]) -> int:
    score = 0.0
    score += clamp(scale(safe_num(row.get("revenueGrowthYoy"), 0.0), -0.03, 0.30), 0.0, 1.0) * 10
    score += clamp(scale(safe_num(row.get("ebitdaMargin"), 0.0), 0.05, 0.50), 0.0, 1.0) * 8
    score += clamp(scale(safe_num(row.get("fcfYield"), 0.0), 0.00, 0.08), 0.0, 1.0) * 9
    score += (1 - clamp(scale(safe_num(row.get("netDebtToEbitda"), 0.0), 0.0, 4.0), 0.0, 1.0)) * 7
    score += (1 - clamp(scale(safe_num(row.get("peForward"), 18.0), 10.0, 45.0), 0.0, 1.0)) * 6
    score += (1 - clamp(scale(safe_num(row.get("evToEbitda"), 12.0), 8.0, 30.0), 0.0, 1.0)) * 5
    score += (1 - clamp(scale(safe_num(row.get("priceToSales"), 2.0), 1.0, 15.0), 0.0, 1.0)) * 4
    score += clamp(scale(safe_num(row.get("priceVs200Sma"), 0.0), -0.20, 0.25), 0.0, 1.0) * 5
    score += 4 if safe_num(row.get("ema20"), 0.0) > safe_num(row.get("sma50"), 0.0) else 1.5
    score += clamp(scale(safe_num(row.get("momentum6m"), 0.0), -0.25, 0.35), 0.0, 1.0) * 5
    score += clamp(scale(safe_num(row.get("macdSignalSpread"), 0.0), -4.0, 4.0), 0.0, 1.0) * 4
    score += (1 - abs(safe_num(row.get("rsi14"), 50.0) - 55.0) / 45.0) * 4
    score += clamp(scale(safe_num(row.get("relVolume"), 1.0), 0.6, 2.0), 0.0, 1.0) * 3
    score += clamp(scale(safe_num(row.get("buybacksToMktCap"), 0.0), 0.0, 0.04), 0.0, 1.0) * 4
    score += clamp(scale(safe_num(row.get("etfFlowSupport"), 0.0), -1.0, 1.0), 0.0, 1.0) * 4
    score += clamp(scale(safe_num(row.get("institutionalNetFlow"), 0.0), -1.0, 1.0), 0.0, 1.0) * 4
    score += clamp(scale(safe_num(row.get("hedgeFundSentiment"), 0.0), -1.0, 1.0), 0.0, 1.0) * 3
    score += clamp(scale(safe_num(row.get("roic"), 0.0), 0.05, 0.30), 0.0, 1.0) * 7
    score += clamp(scale(safe_num(row.get("upsideToFairValue"), 0.0), -0.20, 0.40), 0.0, 1.0) * 8
    return round(score)


def project_price(row: dict[str, Any], months: int) -> dict[str, Any]:
    fair_gap = safe_num(row.get("upsideToFairValue"), 0.0)
    growth = safe_num(row.get("revenueGrowthYoy"), 0.0) * (months / 12)
    margin = safe_num(row.get("marginExpansion"), 0.0) * (months / 12) * 1.2
    technical = (
        safe_num(row.get("momentum6m"), 0.0) * 0.18
        + safe_num(row.get("priceVs200Sma"), 0.0) * 0.10
        + safe_num(row.get("macdSignalSpread"), 0.0) * 0.01
        + (0.02 if safe_num(row.get("ema20"), 0.0) > safe_num(row.get("sma50"), 0.0) else -0.02)
    )
    flows = (
        safe_num(row.get("etfFlowSupport"), 0.0) * 0.03
        + safe_num(row.get("institutionalNetFlow"), 0.0) * 0.03
        + safe_num(row.get("buybacksToMktCap"), 0.0) * 0.5
    )
    penalty = (safe_num(row.get("wacc"), 0.09) - 0.09) * 0.5 + safe_num(row.get("netDebtToEbitda"), 0.0) * 0.006
    expected_return = 0.26 * fair_gap + 0.20 * growth + 0.08 * margin + 0.16 * technical + 0.08 * flows - 0.10 * penalty

    caps = {6: 0.25, 12: 0.40, 18: 0.55, 24: 0.70}
    floors = {6: -0.20, 12: -0.28, 18: -0.35, 24: -0.40}
    expected_return = clamp(expected_return, floors[months], caps[months])
    price = safe_num(row.get("price"), 0.0)

    return {
        "months": months,
        "low": round(price * (1 + expected_return * 0.6), 2),
        "base": round(price * (1 + expected_return), 2),
        "high": round(price * (1 + expected_return * 1.35), 2),
        "expectedReturn": round(expected_return * 100, 1),
    }


# ============================================================
# DATA TRANSFORMATION
# ============================================================

def normalize_candles(raw: dict[str, Any]) -> list[dict[str, Any]]:
    if raw.get("s") != "ok":
        return []

    closes = raw.get("c", [])
    highs = raw.get("h", [])
    lows = raw.get("l", [])
    opens = raw.get("o", [])
    volumes = raw.get("v", [])
    times = raw.get("t", [])

    out: list[dict[str, Any]] = []
    size = min(len(closes), len(highs), len(lows), len(opens), len(volumes), len(times))
    for i in range(size):
        out.append(
            {
                "day": i + 1,
                "timestamp": times[i],
                "open": safe_num(opens[i]),
                "high": safe_num(highs[i]),
                "low": safe_num(lows[i]),
                "close": safe_num(closes[i]),
                "volume": safe_num(volumes[i]),
            }
        )
    return out


async def fetch_symbol_bundle(client: httpx.AsyncClient, symbol: str) -> dict[str, Any]:
    cache_key = f"bundle:{symbol}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    quote_data, profile_data, metrics_data, candles_data = await asyncio.gather(
        get_quote(client, symbol),
        get_profile(client, symbol),
        get_metrics(client, symbol),
        get_candles(client, symbol),
    )

    result = {
        "symbol": symbol,
        "quote": quote_data,
        "profile": profile_data,
        "metrics": metrics_data,
        "candles": candles_data,
    }
    cache_set(cache_key, result)
    return result


def build_pre_row(bundle: dict[str, Any]) -> dict[str, Any]:
    symbol = bundle["symbol"]
    profile = bundle.get("profile") or {}
    metrics = bundle.get("metrics") or {}

    market_cap = safe_num(profile.get("marketCapitalization"), 0.0) * 1_000_000
    shares_outstanding = safe_num(profile.get("shareOutstanding"), 0.0) * 1_000_000
    revenue_ttm = safe_num(metrics.get("revenuePerShareTTM"), 0.0) * shares_outstanding
    ebitda_margin = safe_num(metrics.get("netMargin"), 0.10)
    ebitda = revenue_ttm * ebitda_margin if revenue_ttm > 0 else market_cap * 0.08

    sector = profile.get("finnhubIndustry") or SECTOR_MAP.get(symbol, "Other")

    return {
        "ticker": symbol,
        "sector": sector,
        "peForward": safe_num(metrics.get("peTTM"), 18.0),
        "evToEbitda": safe_div(market_cap, max(ebitda, 1.0), 12.0),
    }


def build_row(bundle: dict[str, Any], peer_median: dict[str, float]) -> dict[str, Any]:
    symbol = bundle["symbol"]
    profile = bundle.get("profile") or {}
    metrics = bundle.get("metrics") or {}
    candles = normalize_candles(bundle.get("candles") or {})

    if len(candles) < 210:
        raise HTTPException(status_code=502, detail=f"No hay suficientes candles para {symbol}")

    closes = [c["close"] for c in candles]
    volumes = [c["volume"] for c in candles]
    ema20_values = ema(closes, 20)
    sma50_values = sma(closes, 50)
    sma200_values = sma(closes, 200)
    rsi_values = rsi(closes, 14)
    macd_line, macd_signal, _ = macd(closes)
    vwap_values = calc_vwap(candles)

    price = safe_num(closes[-1], 0.0)
    quote_current = safe_num((bundle.get("quote") or {}).get("c"), price)
    if quote_current > 0:
        price = quote_current

    ema20_value = safe_num(ema20_values[-1], price)
    sma50_value = safe_num(sma50_values[-1], price)
    sma200_value = safe_num(sma200_values[-1], price)
    rsi14 = safe_num(rsi_values[-1], 50.0)
    macd_line_last = safe_num(macd_line[-1], 0.0)
    macd_signal_last = safe_num(macd_signal[-1], 0.0)
    macd_spread = macd_line_last - macd_signal_last
    volume = safe_num(volumes[-1], 0.0)
    rel_volume = calc_relative_volume(volumes, 30)
    momentum3m = calc_momentum(closes, 63)
    momentum6m = calc_momentum(closes, 126)
    vwap = safe_num(vwap_values[-1], price)

    market_cap = safe_num(profile.get("marketCapitalization"), 0.0) * 1_000_000
    shares_outstanding = safe_num(profile.get("shareOutstanding"), 0.0) * 1_000_000
    if shares_outstanding == 0 and market_cap > 0 and price > 0:
        shares_outstanding = market_cap / price

    ebit_margin = safe_num(metrics.get("operatingMarginTTM"), 0.12)
    ebitda_margin = safe_num(metrics.get("netMargin"), ebit_margin + 0.04)
    revenue_ttm = safe_num(metrics.get("revenuePerShareTTM"), 0.0) * shares_outstanding
    revenue_growth_yoy = safe_num(metrics.get("3YAnnualRevenueGrowth"), 0.05)
    pe_forward = safe_num(metrics.get("peTTM"), 18.0)
    ps = safe_num(metrics.get("psTTM"), 2.0)
    eps_ttm = safe_num(metrics.get("epsTTM"), safe_div(price, max(pe_forward, 1.0), 1.0))
    fcf_per_share = safe_num(metrics.get("freeCashFlowPerShareTTM"), 0.0)
    fcf_yield = safe_div(fcf_per_share, price, 0.0)
    roe = safe_num(metrics.get("roeTTM"), 0.12)
    roic = safe_num(metrics.get("roiTTM"), roe * 0.7)
    current_ratio = safe_num(metrics.get("currentRatioAnnual"), 1.5)

    ebitda = revenue_ttm * ebitda_margin if revenue_ttm > 0 else market_cap * 0.08
    net_debt = max(0.0, market_cap * 0.08)
    net_debt_to_ebitda = safe_div(net_debt, max(ebitda, 1.0), 0.0)
    ev_to_ebitda = safe_div(market_cap + net_debt, max(ebitda, 1.0), 12.0)

    sector = profile.get("finnhubIndustry") or SECTOR_MAP.get(symbol, "Other")

    row = {
        "ticker": symbol,
        "name": profile.get("name") or symbol,
        "sector": sector,
        "style": STYLE_MAP.get(symbol, "quality"),
        "price": price,
        "marketCap": market_cap,
        "sharesOutstanding": shares_outstanding,
        "revenue": revenue_ttm,
        "ebitMargin": ebit_margin,
        "ebitdaMargin": ebitda_margin,
        "ebitda": ebitda,
        "epsForward": eps_ttm,
        "revenueGrowthYoy": revenue_growth_yoy,
        "revenueVolatility": 0.10,
        "capexToSales": 0.08,
        "workingCapitalToSales": 0.03 if current_ratio >= 1 else 0.05,
        "fcfYield": fcf_yield,
        "netDebt": net_debt,
        "netDebtToEbitda": net_debt_to_ebitda,
        "peForward": pe_forward,
        "priceToSales": ps,
        "evToEbitda": ev_to_ebitda,
        "roic": roic,
        "buybacksToMktCap": 0.004,
        "etfFlowSupport": 0.0,
        "institutionalNetFlow": 0.0,
        "hedgeFundSentiment": 0.0,
        "wacc": 0.09 if sector == "Technology" else 0.095,
        "longTermGrowth": 0.032 if sector == "Technology" else 0.025,
        "marginExpansion": 0.008 if sector == "Technology" else 0.003,
        "ema20": ema20_value,
        "sma50": sma50_value,
        "sma200": sma200_value,
        "rsi14": rsi14,
        "macdLine": macd_line_last,
        "macdSignal": macd_signal_last,
        "macdSignalSpread": macd_spread,
        "volume": volume,
        "relVolume": rel_volume,
        "momentum3m": momentum3m,
        "momentum6m": momentum6m,
        "vwap": vwap,
        "priceVs200Sma": safe_div(price, sma200_value, 1.0) - 1.0,
        "trend": trend_label(price, ema20_value, sma50_value, sma200_value),
        "country": profile.get("country"),
        "currency": profile.get("currency"),
        "exchange": profile.get("exchange"),
        "priceSeries": [
            {
                "day": c["day"],
                "close": c["close"],
                "ema20": safe_num(ema20_values[i], c["close"]),
                "sma50": safe_num(sma50_values[i], c["close"]),
                "sma200": safe_num(sma200_values[i], c["close"]),
                "volume": c["volume"],
            }
            for i, c in enumerate(candles)
        ],
    }

    fair_value_dcf = calc_dcf(row)
    fair_value_multiples = calc_multiple_fair_value(row, peer_median)
    fair_value = round(fair_value_dcf * 0.6 + fair_value_multiples * 0.4, 2)
    upside_to_fair_value = safe_div(fair_value, price, 1.0) - 1.0
    forecast_quality = calc_forecast_quality(row)
    margin_of_safety = max(0.0, safe_div(fair_value - price, fair_value, 0.0))

    row.update(
        {
            "fairValueDcf": fair_value_dcf,
            "fairValueMultiples": fair_value_multiples,
            "fairValue": fair_value,
            "upsideToFairValue": upside_to_fair_value,
            "forecastQuality": forecast_quality,
            "marginOfSafety": margin_of_safety,
        }
    )

    row["score"] = calc_composite_score(row)
    row["decision"] = decision_label(row["score"], row["marginOfSafety"], row["forecastQuality"])
    row["projections"] = [project_price(row, m) for m in (6, 12, 18, 24)]
    row["factorRadar"] = [
        {"factor": "Growth", "value": round(clamp(scale(row["revenueGrowthYoy"], 0.0, 0.30), 0.0, 1.0) * 100)},
        {"factor": "Quality", "value": round(clamp(scale(row["roic"], 0.05, 0.30), 0.0, 1.0) * 100)},
        {"factor": "Value", "value": round((1 - clamp(scale(row["peForward"], 10.0, 40.0), 0.0, 1.0)) * 100)},
        {"factor": "Momentum", "value": round(clamp(scale(row["momentum6m"], -0.20, 0.30), 0.0, 1.0) * 100)},
        {"factor": "Risk", "value": round((1 - clamp(scale(row["netDebtToEbitda"], 0.0, 4.0), 0.0, 1.0)) * 100)},
        {"factor": "Flows", "value": round(clamp(scale(row["etfFlowSupport"] + row["institutionalNetFlow"], -2.0, 2.0), 0.0, 1.0) * 100)},
    ]
    row["technicalSignal"] = sum(
        [
            row["price"] > row["sma200"],
            row["ema20"] > row["sma50"],
            45 < row["rsi14"] < 70,
            row["macdSignalSpread"] > 0,
            row["relVolume"] > 1,
            row["momentum6m"] > 0,
        ]
    )

    return row


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "message": "Equity Dashboard Backend running",
        "health": "/health",
        "quote": "/api/quote?symbol=AAPL",
        "dashboard": "/api/equity-dashboard",
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "hasFinnhubKey": bool(FINNHUB_API_KEY)}


@app.get("/api/quote")
async def quote(symbol: str = Query(..., description="Ticker, por ejemplo AAPL")) -> Any:
    try:
        async with httpx.AsyncClient() as client:
            return await get_quote(client, symbol)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Error consultando Finnhub: {exc}") from exc


@app.get("/api/equity-dashboard")
async def equity_dashboard(symbols: str | None = Query(None, description="Lista de tickers separada por comas")) -> dict[str, Any]:
    symbol_list = [s.strip().upper() for s in symbols.split(",")] if symbols else DEFAULT_SYMBOLS
    symbol_list = [s for s in symbol_list if s]
    if not symbol_list:
        raise HTTPException(status_code=400, detail="No se recibieron símbolos")

    cache_key = f"dashboard:{','.join(symbol_list)}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        async with httpx.AsyncClient() as client:
            bundles = await asyncio.gather(*[fetch_symbol_bundle(client, symbol) for symbol in symbol_list])
            index_quotes = await asyncio.gather(*[get_quote(client, symbol) for symbol in INDEX_SYMBOLS.keys()])
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Error consultando Finnhub: {exc}") from exc

    pre_rows = [build_pre_row(bundle) for bundle in bundles]
    sector_groups: dict[str, list[dict[str, Any]]] = {}
    for row in pre_rows:
        sector_groups.setdefault(row["sector"], []).append(row)

    peer_medians = {
        sector: {
            "pe": median([r["peForward"] for r in rows]) if rows else 18.0,
            "evEbitda": median([r["evToEbitda"] for r in rows]) if rows else 12.0,
        }
        for sector, rows in sector_groups.items()
    }

    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        sector = (bundle.get("profile") or {}).get("finnhubIndustry") or SECTOR_MAP.get(bundle["symbol"], "Other")
        rows.append(build_row(bundle, peer_medians.get(sector, {"pe": 18.0, "evEbitda": 12.0})))

    rows.sort(key=lambda r: r["score"], reverse=True)

    indices: list[dict[str, Any]] = []
    for symbol, quote_data in zip(INDEX_SYMBOLS.keys(), index_quotes):
        current = safe_num(quote_data.get("c"), 0.0)
        previous = safe_num(quote_data.get("pc"), current)
        change_pct = safe_div(current - previous, previous, 0.0)
        indices.append(
            {
                "ticker": symbol,
                "name": INDEX_SYMBOLS[symbol],
                "price": round(current, 2),
                "changePct": change_pct,
                "trend": "Alcista" if change_pct > 0.002 else "Bajista" if change_pct < -0.002 else "Lateral",
                "peForward": 0,
                "momentum6m": 0,
            }
        )

    payload = {
        "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "indices": indices,
        "rows": rows,
    }
    cache_set(cache_key, payload)
    return payload


# ============================================================
# RUN LOCAL
# ============================================================
# 1) crear archivo .env:
#    FINNHUB_API_KEY=tu_api_key
#
# 2) instalar dependencias:
#    pip install -r requirements.txt
#
# 3) correr local:
#    uvicorn app:app --reload --port 8000
