"""TypeSafe AI Jev Decision Engine for BTC Trading.

Structures multi-timeframe price action, 24h VWAP, taker buy/sell order flow aggression,
support/resistance levels, and market sentiment into calibrated System One state.
Interprets Choice, Score, and Noul answers to drive robust algorithmic decisions.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
import requests

logger = logging.getLogger(__name__)


@dataclass
class JevDecision:
    action: str  # "BUY", "SELL", "HOLD"
    action_confidence: float
    action_probabilities: Dict[str, float]
    market_regime: str
    regime_confidence: float
    risk_score: float  # 0.0 (minimal) to 3.0 (extreme)
    risk_confidence: float
    entry_conviction: float  # 0.0 to 1.0 (Noul probability)
    exit_urgency: float  # 0.0 to 1.0 (Noul probability)
    model_version: str
    tokens_used: int
    raw_response: Dict[str, Any]

    @property
    def is_buy_signal(self) -> bool:
        return self.action == "BUY"

    @property
    def is_sell_signal(self) -> bool:
        return self.action == "SELL"

    @property
    def is_hold_signal(self) -> bool:
        return self.action == "HOLD"


class JevEngine:
    """Interacts with TypeSafe System One API to query Jev decision model."""

    def __init__(
        self,
        api_key: str,
        model: str = "jev-latest",
        endpoint: str = "https://api.typesafe.ai/v1/systemone",
        max_retries: int = 3,
    ):
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint
        self.session = requests.Session()
        self._circuit_open_until: float = 0.0

    def build_trading_questions(self) -> Dict[str, Any]:
        """Defines the System One question primitives (Choice, Score, Noul) for Perpetual Futures."""
        return {
            "trade_action": {
                "type": "choice",
                "instructions": (
                    "Evaluate BTC's perpetual futures mark price, funding rate, open interest, VWAP deviation, "
                    "order flow aggression, and support/resistance levels. What is the optimal futures action?"
                ),
                "criteria": {
                    "BUY": "Bullish expansion, bounce off support/VWAP, negative/neutral funding rate or aggressive taker buying offering favorable risk/reward for a LONG entry",
                    "SELL": "Bearish breakdown below key support/VWAP, crowded longs paying high funding, or aggressive taker selling signaling a SHORT entry",
                    "HOLD": "Consolidation between levels, mixed order flow, indecisive momentum, or poor risk/reward; stay in cash or maintain active posture",
                },
            },
            "market_regime": {
                "type": "choice",
                "instructions": "What is the prevailing Bitcoin market regime based on structural, derivatives, and order flow data?",
                "criteria": {
                    "BULLISH_EXPANSION": "Price holding above VWAP, 1h and 5m EMAs aligned bullishly, positive taker buying flow, healthy funding",
                    "BEARISH_BREAKDOWN": "Price trading below VWAP, 1h and 5m EMAs aligned bearishly, heavy taker selling flow, liquidation cascade risk",
                    "RANGE_BOUND_CONSOLIDATION": "Oscillating between support and resistance boundaries, balanced taker flow, neutral funding",
                    "LIQUIDATION_CASCADE_RISK": "Overextended price, extreme funding divergence, high risk of long/short squeeze or sudden whip-saw",
                },
            },
            "risk_level": {
                "type": "score",
                "instructions": "Rate the downside execution risk of entering a new futures position right now from lowest to highest hazard",
                "criteria": [
                    "Minimal risk: Clean trend, price near supportive VWAP, solid flow, strong confluence across spot and derivatives",
                    "Moderate risk: Normal volatility, well-defined support/resistance range, manageable liquidation/drawdown risk",
                    "High risk: Extended far from VWAP, conflicting order flow, crowded funding rate, or extreme greed/fear sentiment",
                    "Extreme risk: Cascade breakdown risk, violent liquidation volatility, or lack of liquidity",
                ],
            },
            "entry_conviction": {
                "type": "noul",
                "instructions": "Does the current futures setup offer an asymmetric positive expected value entry opportunity right now?",
                "criteria": {
                    "true": "Confluence of multi-timeframe trend, funding rate edge, VWAP support, and taker flow justify entering immediately",
                    "false": "No clear asymmetric edge; entering now carries negative or ambiguous expected value",
                },
            },
            "exit_urgency": {
                "type": "noul",
                "instructions": "Should any existing active position be closed immediately to protect capital?",
                "criteria": {
                    "true": "Order flow deterioration, key level violation, adverse funding spike, or trend reversal requires immediate liquidation",
                    "false": "Market conditions do not require emergency exit; standard stop-loss or take-profit applies",
                },
            },
        }

    def prepare_state_payload(
        self,
        market_state: Dict[str, Any],
        portfolio_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Packages hierarchical multi-domain market state for Jev."""
        ind = market_state.get("indicators", {})
        flow = market_state.get("order_flow", {})
        sentiment = market_state.get("sentiment", {})
        deriv = market_state.get("derivatives", {})

        return {
            "asset": market_state.get("symbol", "BTC-PERPETUAL"),
            "contract_type": "PERPETUAL_FUTURES",
            "price_and_levels": {
                "mark_price": deriv.get("mark_price", market_state.get("current_price")),
                "index_price": deriv.get("index_price", market_state.get("current_price")),
                "futures_basis_usd": deriv.get("futures_basis_usd", 0.0),
                "24h_vwap": flow.get("vwap_24h"),
                "price_vs_vwap_pct": flow.get("price_to_vwap_pct"),
                "vwap_bias": "ABOVE_VWAP (BULLISH)" if flow.get("price_to_vwap_pct", 0) >= 0 else "BELOW_VWAP (BEARISH)",
                "24h_high_resistance": market_state.get("high_24h"),
                "24h_low_support": market_state.get("low_24h"),
                "swing_support_1h": ind.get("support_level"),
                "swing_resistance_1h": ind.get("resistance_level"),
                "24h_change_pct": market_state.get("change_24h_pct"),
                "spread_bps": market_state.get("spread_bps"),
            },
            "derivatives_and_funding": {
                "funding_rate_8h": deriv.get("funding_rate_8h"),
                "funding_annualized_pct": deriv.get("funding_rate_annualized_pct"),
                "funding_bias": deriv.get("funding_bias"),
                "open_interest_usd": deriv.get("open_interest_usd"),
                "derivatives_volume_usd": deriv.get("stats_24h_volume_usd"),
            },
            "multi_timeframe_technicals": {
                "trend_1h_macro": ind.get("trend_alignment_1h"),
                "trend_5m_execution": ind.get("trend_alignment_5m"),
                "rsi_1h": ind.get("rsi_1h"),
                "rsi_5m": ind.get("rsi_14"),
                "macd_cross_5m": ind.get("macd_cross"),
                "bollinger_pct_b": ind.get("bollinger_percent_b"),
                "atr_14_volatility": ind.get("atr_14"),
            },
            "order_flow_and_positioning": {
                "taker_buy_volume_btc": flow.get("taker_buy_volume_btc"),
                "taker_sell_volume_btc": flow.get("taker_sell_volume_btc"),
                "taker_buy_sell_ratio": flow.get("taker_buy_sell_ratio"),
                "flow_aggression": flow.get("flow_bias"),
            },
            "market_sentiment": {
                "fear_and_greed_score": sentiment.get("fear_and_greed_score"),
                "fear_and_greed_label": sentiment.get("fear_and_greed_sentiment"),
            },
            "recent_execution_candles": market_state.get("recent_candles_summary", []),
            "portfolio_context": {
                "cash_usdt": portfolio_state.get("cash_usdt"),
                "position_side": portfolio_state.get("position_side", "NONE"),
                "position_size": portfolio_state.get("position_size_btc", 0.0),
                "leverage": portfolio_state.get("leverage", 2),
                "margin_usd": portfolio_state.get("margin_usd", 0.0),
                "liquidation_price": portfolio_state.get("liquidation_price"),
                "has_open_position": portfolio_state.get("has_open_position", False),
                "entry_price": portfolio_state.get("entry_price"),
                "unrealized_pnl_pct": portfolio_state.get("unrealized_pnl_pct", 0.0),
                "daily_pnl_pct": portfolio_state.get("daily_pnl_pct", 0.0),
            },
        }

    def fallback_evaluate(
        self,
        market_state: Dict[str, Any],
        portfolio_state: Dict[str, Any],
    ) -> JevDecision:
        """Quantitative rule-based System One fallback when TypeSafe API is unreachable.

        Calculates calibrated probabilities from Binance Futures indicators:
        - Multi-timeframe trend (1h macro, 5m execution)
        - VWAP deviation and 24h weighted volume
        - Taker order flow aggression (buy/sell ratio)
        - RSI momentum & MACD signal
        - Funding rate and perpetual basis
        """
        ind = market_state.get("indicators", {})
        flow = market_state.get("order_flow", {})
        deriv = market_state.get("derivatives", {})

        # 1. Macro & micro trend alignment
        trend_score = 0.0
        t1h_str = str(ind.get("trend_alignment_1h", "NEUTRAL")).upper()
        t5m_str = str(ind.get("trend_alignment_5m", "NEUTRAL")).upper()

        is_t1h_bull = "BULL" in t1h_str
        is_t1h_bear = "BEAR" in t1h_str
        is_t5m_bull = "BULL" in t5m_str
        is_t5m_bear = "BEAR" in t5m_str

        if is_t1h_bull and is_t5m_bull:
            trend_score += 0.50  # Confirmed multi-timeframe bullish trend
        elif is_t1h_bear and is_t5m_bear:
            trend_score -= 0.50  # Confirmed multi-timeframe bearish breakdown
        elif is_t1h_bull and not is_t5m_bear:
            trend_score += 0.30  # Macro bull with neutral 5m
        elif is_t1h_bear and not is_t5m_bull:
            trend_score -= 0.30  # Macro bear with neutral 5m
        elif is_t1h_bull and is_t5m_bear:
            trend_score += 0.10  # Pullback dip inside 1h uptrend (buying opportunity)
        elif is_t1h_bear and is_t5m_bull:
            trend_score -= 0.10  # Bear rally inside 1h downtrend (shorting opportunity)

        # 2. VWAP deviation
        vwap_pct = flow.get("price_to_vwap_pct", 0.0)
        if vwap_pct > 0.15:
            trend_score += 0.15
        elif vwap_pct < -0.15:
            trend_score -= 0.15

        # 3. Order flow aggression
        ratio = flow.get("taker_buy_sell_ratio", 1.0)
        if ratio > 1.08:
            trend_score += 0.20
        elif ratio < 0.92:
            trend_score -= 0.20

        # 4. RSI momentum & MACD
        rsi = ind.get("rsi_14", 50.0)
        if 55.0 <= rsi <= 70.0:
            trend_score += 0.10
        elif 30.0 <= rsi <= 45.0:
            trend_score -= 0.10
        elif rsi > 75.0:
            trend_score -= 0.15  # Overbought exhaustion risk
        elif rsi < 25.0:
            trend_score += 0.15  # Oversold bounce potential

        macd_cross = ind.get("macd_cross", "NEUTRAL")
        if macd_cross == "bullish_cross" or macd_cross == "BULLISH_CROSS":
            trend_score += 0.10
        elif macd_cross == "bearish_cross" or macd_cross == "BEARISH_CROSS":
            trend_score -= 0.10

        # 5. Bollinger Band extreme position signals
        boll_pct_b = ind.get("bollinger_percent_b", 0.5)
        if boll_pct_b is not None:
            if boll_pct_b < 0.15:
                trend_score += 0.12  # Strong oversold bounce potential
            elif boll_pct_b > 0.85:
                trend_score -= 0.12  # Strong overbought mean reversion

        # 6. Funding rate edge
        funding = deriv.get("funding_rate_8h", 0.0)
        if funding < -0.0001:
            trend_score += 0.08  # Negative funding: shorts pay longs (bullish squeeze potential)
        elif funding > 0.0003:
            trend_score -= 0.08  # Crowded long funding cost

        # Normalize score into [-1.0, 1.0]
        s = max(-1.0, min(1.0, trend_score))

        # Convert score to calibrated Choice probabilities
        if s > 0.12:
            # Bullish expansion
            p_buy = min(0.85, 0.52 + (s * 0.38))
            p_sell = max(0.04, 0.14 - (s * 0.12))
            p_hold = max(0.08, 1.0 - p_buy - p_sell)
            action = "BUY"
            regime = "BULLISH_EXPANSION"
            regime_conf = round(min(0.95, 0.65 + abs(s) * 0.30), 2)
            risk_score = 1.0
        elif s < -0.12:
            # Bearish breakdown
            p_sell = min(0.85, 0.52 + (abs(s) * 0.38))
            p_buy = max(0.04, 0.14 - (abs(s) * 0.12))
            p_hold = max(0.08, 1.0 - p_buy - p_sell)
            action = "SELL"
            regime = "BEARISH_BREAKDOWN"
            regime_conf = round(min(0.95, 0.65 + abs(s) * 0.30), 2)
            risk_score = 1.2
        else:
            # Range bound consolidation
            p_hold = 0.56
            p_buy = round(0.22 + (s * 0.20), 4)
            p_sell = round(1.0 - p_hold - p_buy, 4)
            action = "HOLD"
            regime = "RANGE_BOUND_CONSOLIDATION"
            regime_conf = 0.70
            risk_score = 1.1

        action_probs = {
            "BUY": round(p_buy, 4),
            "SELL": round(p_sell, 4),
            "HOLD": round(p_hold, 4),
        }
        action_conf = round(max(p_buy, p_sell, p_hold), 3)
        entry_conviction = round(min(0.92, max(0.10, 0.25 + (abs(s) * 0.68))), 3)

        # Portfolio exit urgency check
        has_pos = portfolio_state.get("has_open_position", False)
        pos_side = portfolio_state.get("position_side", "NONE")
        exit_urgency = 0.08
        if has_pos:
            if pos_side == "LONG" and s < -0.25:
                exit_urgency = 0.78
            elif pos_side == "SHORT" and s > 0.25:
                exit_urgency = 0.78

        return JevDecision(
            action=action,
            action_confidence=action_conf,
            action_probabilities=action_probs,
            market_regime=regime,
            regime_confidence=regime_conf,
            risk_score=risk_score,
            risk_confidence=0.85,
            entry_conviction=entry_conviction,
            exit_urgency=exit_urgency,
            model_version="jev-quant-fallback-v2.0",
            tokens_used=0,
            raw_response={
                "source": "live_quantitative_fallback",
                "trend_score": round(s, 3),
                "probabilities": action_probs,
                "regime": regime,
            },
        )

    def evaluate(
        self,
        market_state: Dict[str, Any],
        portfolio_state: Dict[str, Any],
    ) -> JevDecision:
        """Sends evaluation request to TypeSafe Jev or uses quantitative fallback if offline."""
        if not self.api_key:
            logger.info("TYPESAFE_API_KEY not configured; using quantitative live fallback.")
            return self.fallback_evaluate(market_state, portfolio_state)

        if time.time() < self._circuit_open_until:
            return self.fallback_evaluate(market_state, portfolio_state)

        state = self.prepare_state_payload(market_state, portfolio_state)
        questions = self.build_trading_questions()

        payload = {
            "state": state,
            "model": self.model,
            "questions": questions,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "TypeSafe-Jev-BTC-Trader/2.0",
        }

        try:
            resp = self.session.post(
                self.endpoint,
                json=payload,
                headers=headers,
                timeout=2.0,
            )

            if resp.status_code == 200:
                self._circuit_open_until = 0.0
                data = resp.json()
                return self._parse_response(data)

            if resp.status_code in (429, 529):
                self._circuit_open_until = time.time() + 30.0
                logger.warning(f"TypeSafe API returned {resp.status_code}. Using quantitative fallback (circuit open 30s).")
                return self.fallback_evaluate(market_state, portfolio_state)

            logger.warning(f"TypeSafe API error {resp.status_code}: {resp.text[:100]}. Using quantitative fallback.")
            return self.fallback_evaluate(market_state, portfolio_state)

        except Exception as e:
            self._circuit_open_until = time.time() + 60.0
            logger.warning(f"TypeSafe API unreachable ({e}). Using quantitative live fallback (circuit open 60s).")
            return self.fallback_evaluate(market_state, portfolio_state)

    def _parse_response(self, data: Dict[str, Any]) -> JevDecision:
        """Parses raw Jev response into a clean JevDecision object."""
        answers = data.get("answers", {})
        usage = data.get("usage", {})
        total_tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)

        # 1. Trade action (Choice)
        action_ans = answers.get("trade_action", {})
        action = action_ans.get("choice", "HOLD").upper()
        action_conf = float(action_ans.get("confidence", 0.0))
        action_probs = {k.upper(): float(v) for k, v in action_ans.get("probabilities", {}).items()}

        # 2. Market regime (Choice)
        regime_ans = answers.get("market_regime", {})
        market_regime = regime_ans.get("choice", "RANGE_BOUND_CONSOLIDATION").upper()
        regime_conf = float(regime_ans.get("confidence", 0.0))

        # 3. Risk level (Score)
        risk_ans = answers.get("risk_level", {})
        risk_score = float(risk_ans.get("score", 1.5))
        risk_conf = float(risk_ans.get("confidence", 0.0))

        # 4. Entry conviction (Noul)
        entry_ans = answers.get("entry_conviction", {})
        entry_conviction = float(entry_ans.get("noul", 0.0))

        # 5. Exit urgency (Noul)
        exit_ans = answers.get("exit_urgency", {})
        exit_urgency = float(exit_ans.get("noul", 0.0))

        return JevDecision(
            action=action,
            action_confidence=action_conf,
            action_probabilities=action_probs,
            market_regime=market_regime,
            regime_confidence=regime_conf,
            risk_score=round(risk_score, 2),
            risk_confidence=risk_conf,
            entry_conviction=round(entry_conviction, 3),
            exit_urgency=round(exit_urgency, 3),
            model_version=data.get("model", self.model),
            tokens_used=total_tokens,
            raw_response=data,
        )
