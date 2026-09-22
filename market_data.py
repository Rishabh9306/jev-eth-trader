"""Market data fetcher and quantitative indicator engine for BTC trading.

Integrates:
- Multi-Timeframe Structure (1h Macro + 5m Execution candles)
- Technical Indicators (RSI, EMA 9/21/50/200, MACD, Bollinger Bands, ATR)
- Institutional Volume & VWAP (24h Volume-Weighted Average Price)
- Order Flow Aggression (Taker Buy vs Taker Sell Volume Ratio)
- Market Sentiment (Crypto Fear & Greed Index)
- Swing Support & Resistance Levels
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import requests
import time

logger = logging.getLogger(__name__)


@dataclass
class TechnicalIndicators:
    rsi_14: float
    rsi_1h: float
    ema_9: float
    ema_21: float
    ema_50: float
    ema_200: Optional[float]
    macd: float
    macd_signal: float
    macd_hist: float
    macd_cross: str  # "bullish_cross", "bearish_cross", "neutral"
    bollinger_upper: float
    bollinger_middle: float
    bollinger_lower: float
    bollinger_percent_b: float
    atr_14: float
    trend_alignment_5m: str
    trend_alignment_1h: str
    support_level: float
    resistance_level: float


@dataclass
class OrderFlowContext:
    vwap_24h: float
    price_to_vwap_pct: float
    taker_buy_volume_btc: float
    taker_sell_volume_btc: float
    taker_buy_sell_ratio: float
    flow_bias: str  # "AGGRESSIVE_BUYING", "AGGRESSIVE_SELLING", "NEUTRAL"


@dataclass
class MarketSentiment:
    fear_and_greed_score: int
    fear_and_greed_sentiment: str


@dataclass
class DerivativesContext:
    mark_price: float
    index_price: float
    futures_basis_usd: float
    futures_basis_bps: float
    funding_rate_8h: float
    funding_rate_annualized_pct: float
    funding_bias: str  # "LONGS_PAY_SHORTS", "SHORTS_PAY_LONGS", "NEUTRAL"
    open_interest_usd: float
    stats_24h_volume_usd: float


@dataclass
class MarketState:
    symbol: str
    timestamp: str
    current_price: float
    bid_price: float
    ask_price: float
    spread: float
    spread_bps: float
    change_24h_pct: float
    high_24h: float
    low_24h: float
    volume_24h: float
    indicators: TechnicalIndicators
    order_flow: OrderFlowContext
    sentiment: MarketSentiment
    derivatives: DerivativesContext
    recent_candles_summary: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MarketDataFetcher:
    """Fetches real-time multi-timeframe market data, order flow, sentiment, and perpetual derivatives from Binance Futures."""

    _cached_markets: List[Dict[str, Any]] = []
    _cached_markets_time: float = 0.0

    @classmethod
    def get_available_markets(cls) -> List[Dict[str, Any]]:
        """Fetches all active USDT-M Perpetual Futures markets from Binance, sorted by 24h volume."""
        now = time.time()
        if cls._cached_markets and (now - cls._cached_markets_time < 45.0):
            return cls._cached_markets

        try:
            url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
            resp = requests.get(url, timeout=5, headers={"User-Agent": "TypeSafe-Jev-Trader/2.0"})
            if resp.status_code == 200:
                raw = resp.json()
                markets = []
                for item in raw:
                    sym = item.get("symbol", "")
                    if sym.endswith("USDT") and sym.isascii() and sym.isalnum():
                        vol_quote = float(item.get("quoteVolume", 0.0))
                        last_p = float(item.get("lastPrice", 0.0))
                        chg = float(item.get("priceChangePercent", 0.0))
                        base = sym.replace("USDT", "")
                        markets.append({
                            "symbol": sym,
                            "base_asset": base,
                            "price": last_p,
                            "change_24h_pct": round(chg, 2),
                            "quote_volume_usd": round(vol_quote, 2),
                            "volume_24h": float(item.get("volume", 0.0)),
                            "high_24h": float(item.get("highPrice", 0.0)),
                            "low_24h": float(item.get("lowPrice", 0.0)),
                        })
                markets.sort(key=lambda m: m["quote_volume_usd"], reverse=True)
                cls._cached_markets = markets
                cls._cached_markets_time = now
                return markets
        except Exception as e:
            logger.warning(f"Failed to fetch available markets from Binance: {e}")

        if cls._cached_markets:
            return cls._cached_markets

        # Fallback default top markets if network hiccup
        return [
            {"symbol": "BTCUSDT", "base_asset": "BTC", "price": 85200.0, "change_24h_pct": 0.5, "quote_volume_usd": 15000000000.0},
            {"symbol": "ETHUSDT", "base_asset": "ETH", "price": 2650.0, "change_24h_pct": 1.2, "quote_volume_usd": 7000000000.0},
            {"symbol": "SOLUSDT", "base_asset": "SOL", "price": 185.0, "change_24h_pct": -0.8, "quote_volume_usd": 4000000000.0},
            {"symbol": "BNBUSDT", "base_asset": "BNB", "price": 640.0, "change_24h_pct": 0.2, "quote_volume_usd": 1200000000.0},
            {"symbol": "XRPUSDT", "base_asset": "XRP", "price": 2.45, "change_24h_pct": 3.4, "quote_volume_usd": 2500000000.0},
            {"symbol": "DOGEUSDT", "base_asset": "DOGE", "price": 0.22, "change_24h_pct": -1.5, "quote_volume_usd": 1800000000.0},
            {"symbol": "SUIUSDT", "base_asset": "SUI", "price": 3.10, "change_24h_pct": 4.1, "quote_volume_usd": 1500000000.0},
        ]

    def __init__(self, symbol: str = "BTCUSDT", timeframe: str = "5m"):
        cleaned = symbol.replace("/", "").replace("-", "").upper()
        if "BTC" in cleaned and not ("USDT" in cleaned):
            self.binance_symbol = "BTCUSDT"
        elif not cleaned.endswith("USDT"):
            self.binance_symbol = cleaned + "USDT"
        else:
            self.binance_symbol = cleaned
        self.symbol = self.binance_symbol
        self.timeframe = timeframe
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TypeSafe-Jev-Trader/2.0"})

    def fetch_current_ticker(self) -> Dict[str, Any]:
        """Fetches 24h ticker info, price, and VWAP exclusively from official Binance Futures REST API."""
        url = f"https://fapi.binance.com/fapi/v1/ticker/24hr?symbol={self.binance_symbol}"
        for attempt in range(1, 4):
            try:
                resp = self.session.get(url, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    last = float(data["lastPrice"])
                    vwap = float(data.get("weightedAvgPrice", last))
                    spread_bps = 0.5  # Standard tight perpetual spread on Binance Futures
                    spread = last * 0.00005
                    price_to_vwap_pct = ((last - vwap) / vwap) * 100 if vwap > 0 else 0.0

                    return {
                        "price": last,
                        "bid": round(last - (spread / 2), 2),
                        "ask": round(last + (spread / 2), 2),
                        "spread": round(spread, 2),
                        "spread_bps": round(spread_bps, 2),
                        "change_24h_pct": round(float(data["priceChangePercent"]), 2),
                        "high_24h": float(data["highPrice"]),
                        "low_24h": float(data["lowPrice"]),
                        "volume_24h": float(data["volume"]),
                        "vwap_24h": round(vwap, 2),
                        "price_to_vwap_pct": round(price_to_vwap_pct, 2),
                    }
            except Exception as e:
                logger.warning(f"Binance Futures ticker attempt {attempt} failed: {e}")
                time.sleep(0.5)

        raise RuntimeError(f"Failed to fetch ticker from Binance Futures REST API ({url})")

    def fetch_klines(self, interval: str = "5m", limit: int = 100) -> pd.DataFrame:
        """Fetches OHLCV candlestick and taker flow data exclusively from Binance Futures REST API."""
        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={self.binance_symbol}&interval={interval}&limit={limit}"
        for attempt in range(1, 4):
            try:
                resp = self.session.get(url, timeout=6)
                if resp.status_code == 200:
                    raw = resp.json()
                    df = pd.DataFrame(
                        raw,
                        columns=[
                            "open_time",
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume",
                            "close_time",
                            "quote_volume",
                            "trades",
                            "taker_buy_base",
                            "taker_buy_quote",
                            "ignore",
                        ],
                    )
                    for col in ["open", "high", "low", "close", "volume", "taker_buy_base", "taker_buy_quote"]:
                        df[col] = df[col].astype(float)
                    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
                    return df[["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"]]
            except Exception as e:
                logger.warning(f"Binance Futures klines attempt {attempt} failed: {e}")
                time.sleep(0.5)

        raise RuntimeError(f"Failed to fetch OHLCV bars from Binance Futures REST for interval {interval}")

    def fetch_sentiment(self) -> MarketSentiment:
        """Fetches Crypto Fear and Greed Index."""
        try:
            url = "https://api.alternative.me/fng/?limit=1"
            resp = self.session.get(url, timeout=4)
            if resp.status_code == 200:
                data = resp.json().get("data", [{}])[0]
                score = int(data.get("value", 50))
                sentiment = str(data.get("value_classification", "Neutral"))
                return MarketSentiment(
                    fear_and_greed_score=score,
                    fear_and_greed_sentiment=sentiment,
                )
        except Exception as e:
            logger.warning(f"Failed to fetch Fear & Greed Index: {e}. Using default.")
        return MarketSentiment(fear_and_greed_score=50, fear_and_greed_sentiment="Neutral")

    def calculate_order_flow(self, df_1h: pd.DataFrame, ticker: Dict[str, Any]) -> OrderFlowContext:
        """Calculates 24h taker buy vs sell volume and flow aggression."""
        vwap = ticker.get("vwap_24h", ticker["price"])
        price_to_vwap_pct = ticker.get("price_to_vwap_pct", 0.0)

        # Sum taker buy vs taker sell from recent 24 1h bars
        recent_24h = df_1h.tail(24)
        total_vol = recent_24h["volume"].sum()
        taker_buy = recent_24h["taker_buy_base"].sum()
        taker_sell = total_vol - taker_buy
        ratio = round(taker_buy / taker_sell, 3) if taker_sell > 0 else 1.0

        if ratio >= 1.15:
            bias = "AGGRESSIVE_BUYING"
        elif ratio <= 0.85:
            bias = "AGGRESSIVE_SELLING"
        else:
            bias = "NEUTRAL"

        return OrderFlowContext(
            vwap_24h=vwap,
            price_to_vwap_pct=price_to_vwap_pct,
            taker_buy_volume_btc=round(taker_buy, 2),
            taker_sell_volume_btc=round(taker_sell, 2),
            taker_buy_sell_ratio=ratio,
            flow_bias=bias,
        )

    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> float:
        """Computes Wilder's smoothed RSI."""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        for i in range(period, len(delta)):
            avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
            avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0

    def calculate_indicators(self, df_5m: pd.DataFrame, df_1h: pd.DataFrame) -> TechnicalIndicators:
        """Computes multi-timeframe indicators across 5m and 1h intervals."""
        # 5m indicators
        close_5m = df_5m["close"]
        high_5m = df_5m["high"]
        low_5m = df_5m["low"]
        curr_price = close_5m.iloc[-1]

        # 1. 5m EMAs
        ema_9 = close_5m.ewm(span=9, adjust=False).mean()
        ema_21 = close_5m.ewm(span=21, adjust=False).mean()
        ema_50 = close_5m.ewm(span=50, adjust=False).mean()
        ema_200 = close_5m.ewm(span=200, adjust=False).mean() if len(close_5m) >= 100 else None

        e9 = float(ema_9.iloc[-1])
        e21 = float(ema_21.iloc[-1])
        e50 = float(ema_50.iloc[-1])
        e200 = float(ema_200.iloc[-1]) if ema_200 is not None else None

        trend_5m = (
            "strong_bullish" if e9 > e21 > e50
            else "bullish" if e9 > e21
            else "strong_bearish" if e9 < e21 < e50
            else "bearish" if e9 < e21
            else "mixed"
        )

        # 2. 1h Macro Trend & RSI
        close_1h = df_1h["close"]
        ema_20_1h = float(close_1h.ewm(span=20, adjust=False).mean().iloc[-1])
        ema_50_1h = float(close_1h.ewm(span=50, adjust=False).mean().iloc[-1])
        trend_1h = "BULLISH_EXPANSION" if ema_20_1h > ema_50_1h else "BEARISH_CONTRACTION"
        rsi_1h = round(self.calculate_rsi(close_1h, 14), 1)

        # 3. Swing Support & Resistance (from recent 1h candles)
        recent_1h = df_1h.tail(24)
        support_level = round(float(recent_1h["low"].min()), 2)
        resistance_level = round(float(recent_1h["high"].max()), 2)

        # 4. 5m RSI (14)
        rsi_5m = round(self.calculate_rsi(close_5m, 14), 1)

        # 5. MACD (12, 26, 9)
        ema_12 = close_5m.ewm(span=12, adjust=False).mean()
        ema_26 = close_5m.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        hist = macd_line - signal_line

        curr_macd = float(macd_line.iloc[-1])
        curr_sig = float(signal_line.iloc[-1])
        curr_hist = float(hist.iloc[-1])
        prev_macd = float(macd_line.iloc[-2])
        prev_sig = float(signal_line.iloc[-2])

        if prev_macd <= prev_sig and curr_macd > curr_sig:
            macd_cross = "bullish_cross"
        elif prev_macd >= prev_sig and curr_macd < curr_sig:
            macd_cross = "bearish_cross"
        else:
            macd_cross = "neutral"

        # 6. Bollinger Bands (20, 2)
        sma_20 = close_5m.rolling(window=20).mean()
        std_20 = close_5m.rolling(window=20).std()
        upper_bb = sma_20 + (std_20 * 2)
        lower_bb = sma_20 - (std_20 * 2)
        u_bb = float(upper_bb.iloc[-1])
        l_bb = float(lower_bb.iloc[-1])
        m_bb = float(sma_20.iloc[-1])
        pct_b = (curr_price - l_bb) / (u_bb - l_bb) if (u_bb - l_bb) > 0 else 0.5

        # 7. ATR (14) for dynamic volatility stops
        tr1 = high_5m - low_5m
        tr2 = (high_5m - close_5m.shift(1)).abs()
        tr3 = (low_5m - close_5m.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_14 = float(tr.rolling(window=14).mean().iloc[-1])

        return TechnicalIndicators(
            rsi_14=rsi_5m,
            rsi_1h=rsi_1h,
            ema_9=round(e9, 2),
            ema_21=round(e21, 2),
            ema_50=round(e50, 2),
            ema_200=round(e200, 2) if e200 is not None else None,
            macd=round(curr_macd, 2),
            macd_signal=round(curr_sig, 2),
            macd_hist=round(curr_hist, 2),
            macd_cross=macd_cross,
            bollinger_upper=round(u_bb, 2),
            bollinger_middle=round(m_bb, 2),
            bollinger_lower=round(l_bb, 2),
            bollinger_percent_b=round(pct_b, 3),
            atr_14=round(atr_14, 2),
            trend_alignment_5m=trend_5m,
            trend_alignment_1h=trend_1h,
            support_level=support_level,
            resistance_level=resistance_level,
        )

    def fetch_derivatives_data(self) -> DerivativesContext:
        """Fetches live funding rate, open interest, and mark price exclusively from Binance Futures REST API."""
        for attempt in range(1, 4):
            try:
                prem_url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={self.binance_symbol}"
                resp = self.session.get(prem_url, timeout=5)
                if resp.status_code == 200:
                    p_data = resp.json()
                    mark = float(p_data.get("markPrice", 0.0))
                    index = float(p_data.get("indexPrice", mark))
                    basis_usd = mark - index
                    basis_bps = (basis_usd / index) * 10000 if index > 0 else 0.0
                    funding_8h = float(p_data.get("lastFundingRate", 0.0001))
                    funding_annualized = funding_8h * 3 * 365 * 100

                    if funding_8h > 0.0001:
                        bias = "LONGS_PAY_SHORTS (BULLISH_CROWDING)"
                    elif funding_8h < -0.0001:
                        bias = "SHORTS_PAY_LONGS (BEARISH_CROWDING)"
                    else:
                        bias = "NEUTRAL"

                    # Open Interest in USD from Binance Futures
                    oi_usd = 0.0
                    try:
                        oi_url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={self.binance_symbol}"
                        oi_resp = self.session.get(oi_url, timeout=4)
                        if oi_resp.status_code == 200:
                            oi_btc = float(oi_resp.json().get("openInterest", 0.0))
                            oi_usd = oi_btc * mark
                    except Exception as e:
                        logger.debug(f"Binance Open Interest fetch: {e}")

                    # 24h Futures Quote Volume (USD) from Binance Futures
                    vol_usd = 0.0
                    try:
                        t_url = f"https://fapi.binance.com/fapi/v1/ticker/24hr?symbol={self.binance_symbol}"
                        t_resp = self.session.get(t_url, timeout=4)
                        if t_resp.status_code == 200:
                            vol_usd = float(t_resp.json().get("quoteVolume", 0.0))
                    except Exception as e:
                        logger.debug(f"Binance 24h Volume fetch: {e}")

                    return DerivativesContext(
                        mark_price=round(mark, 2),
                        index_price=round(index, 2),
                        futures_basis_usd=round(basis_usd, 2),
                        futures_basis_bps=round(basis_bps, 2),
                        funding_rate_8h=round(funding_8h, 6),
                        funding_rate_annualized_pct=round(funding_annualized, 2),
                        funding_bias=bias,
                        open_interest_usd=round(oi_usd, 2),
                        stats_24h_volume_usd=round(vol_usd, 2),
                    )
            except Exception as e:
                logger.warning(f"Binance Futures derivatives attempt {attempt} failed: {e}")
                time.sleep(0.5)

        logger.error("Failed to fetch derivatives data from Binance Futures REST.")
        return DerivativesContext(
            mark_price=0.0,
            index_price=0.0,
            futures_basis_usd=0.0,
            futures_basis_bps=0.0,
            funding_rate_8h=0.0001,
            funding_rate_annualized_pct=10.95,
            funding_bias="NEUTRAL",
            open_interest_usd=0.0,
            stats_24h_volume_usd=0.0,
        )

    def get_market_state(self) -> MarketState:
        """Fetches complete multi-timeframe BTC state, order flow, sentiment, and perpetual derivatives."""
        ticker = self.fetch_current_ticker()
        derivatives = self.fetch_derivatives_data()
        df_5m = self.fetch_klines(interval=self.timeframe, limit=60)
        df_1h = self.fetch_klines(interval="1h", limit=48)

        indicators = self.calculate_indicators(df_5m, df_1h)
        order_flow = self.calculate_order_flow(df_1h, ticker)
        sentiment = self.fetch_sentiment()

        # Pure Binance Futures mark price or ticker last price
        current_price = derivatives.mark_price if derivatives.mark_price > 0 else ticker["price"]

        # Recent 5 candles summary for state context
        recent = df_5m.tail(5)
        recent_summary = []
        for _, row in recent.iterrows():
            c_return = round(((row["close"] - row["open"]) / row["open"]) * 100, 3)
            recent_summary.append({
                "time": str(row["timestamp"]),
                "open": round(row["open"], 2),
                "high": round(row["high"], 2),
                "low": round(row["low"], 2),
                "close": round(row["close"], 2),
                "volume": round(row["volume"], 2),
                "return_pct": c_return,
            })

        return MarketState(
            symbol=self.symbol,
            timestamp=str(pd.Timestamp.utcnow()),
            current_price=current_price,
            bid_price=ticker["bid"],
            ask_price=ticker["ask"],
            spread=ticker["spread"],
            spread_bps=ticker["spread_bps"],
            change_24h_pct=ticker["change_24h_pct"],
            high_24h=ticker["high_24h"],
            low_24h=ticker["low_24h"],
            volume_24h=ticker["volume_24h"],
            indicators=indicators,
            order_flow=order_flow,
            sentiment=sentiment,
            derivatives=derivatives,
            recent_candles_summary=recent_summary,
        )
