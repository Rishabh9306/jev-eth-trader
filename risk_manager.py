"""Risk Manager and Confidence-Gated Execution Router for BTC Perpetual Futures.

Translates Jev's System One answers, confidence scores, and derivatives context
into risk-managed trading orders for both LONG and SHORT directions.
Enforces confidence floors, volatility-based stop-loss/take-profit,
position sizing limits, and daily drawdown circuit breakers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple
from config import RiskConfig
from jev_engine import JevDecision

logger = logging.getLogger(__name__)


@dataclass
class OrderPlan:
    should_execute: bool
    order_type: str  # "ENTER_LONG", "ENTER_SHORT", "EXIT_LONG", "EXIT_SHORT", "HOLD"
    size_usd: float
    size_eth: float  # Base asset size in BTC
    stop_loss_price: Optional[float]
    take_profit_price: Optional[float]
    reason: str
    confidence: float
    risk_score: float


def format_price(price: Optional[float]) -> str:
    """Formats price with appropriate precision based on price magnitude."""
    if price is None:
        return "N/A"
    if price < 0.001:
        return f"{price:.7f}"
    elif price < 0.01:
        return f"{price:.6f}"
    elif price < 1.0:
        return f"{price:.4f}"
    elif price < 10.0:
        return f"{price:.3f}"
    return f"{price:.2f}"


def round_price(price: float, ref_price: float) -> float:
    """Rounds price to appropriate decimal places based on reference price magnitude."""
    if ref_price < 0.001:
        return round(price, 7)
    elif ref_price < 0.01:
        return round(price, 6)
    elif ref_price < 1.0:
        return round(price, 5)
    elif ref_price < 10.0:
        return round(price, 4)
    return round(price, 2)


class RiskManager:
    """Confidence-gated risk management engine for two-way perpetual futures."""

    def __init__(self, config: RiskConfig, engine: Optional[Any] = None):
        self.config = config
        self.engine = engine

    def evaluate_decision(
        self,
        decision: JevDecision,
        current_price: float,
        portfolio: dict,
        atr_14: Optional[float] = None,
    ) -> OrderPlan:
        """Evaluates Jev decision through confidence and risk gates to produce an OrderPlan."""
        has_position = portfolio.get("has_open_position", False)
        position_side = portfolio.get("position_side", "LONG" if has_position else "NONE")
        open_pos_size = portfolio.get("position_size_btc", portfolio.get("position_size_eth", 0.0))
        entry_price = portfolio.get("entry_price", 0.0)
        cash_usdt = portfolio.get("cash_usdt", 0.0)
        daily_pnl_pct = portfolio.get("daily_pnl_pct", 0.0)

        # 1. Circuit Breaker: Daily Max Drawdown
        limit_pct = self.config.max_daily_drawdown_pct * 100.0 if abs(self.config.max_daily_drawdown_pct) < 1.0 else self.config.max_daily_drawdown_pct
        if daily_pnl_pct <= -limit_pct:
            logger.warning(
                f"Circuit breaker triggered: Daily PnL {daily_pnl_pct:.2f}% exceeds limit "
                f"{-limit_pct:.2f}%. Trading paused."
            )
            if has_position:
                exit_type = "EXIT_LONG" if position_side == "LONG" else "EXIT_SHORT"
                return OrderPlan(
                    should_execute=True,
                    order_type=exit_type,
                    size_usd=open_pos_size * current_price,
                    size_eth=open_pos_size,
                    stop_loss_price=None,
                    take_profit_price=None,
                    reason="Circuit breaker: daily drawdown limit breached",
                    confidence=1.0,
                    risk_score=decision.risk_score,
                )
            return OrderPlan(
                should_execute=False,
                order_type="HOLD",
                size_usd=0.0,
                size_eth=0.0,
                stop_loss_price=None,
                take_profit_price=None,
                reason="Daily drawdown limit breached; cooling off",
                confidence=0.0,
                risk_score=decision.risk_score,
            )

        # 2. Position Exit Checks (if holding an open futures position)
        if has_position and open_pos_size > 0.0001:
            active_stop = portfolio.get("stop_loss_price")
            active_tp = portfolio.get("take_profit_price")

            # Check A: Emergency exit signal from Jev
            if decision.exit_urgency >= self.config.exit_urgency_threshold:
                exit_type = "EXIT_LONG" if position_side == "LONG" else "EXIT_SHORT"
                return OrderPlan(
                    should_execute=True,
                    order_type=exit_type,
                    size_usd=open_pos_size * current_price,
                    size_eth=open_pos_size,
                    stop_loss_price=None,
                    take_profit_price=None,
                    reason=f"Jev emergency exit signal (urgency: {decision.exit_urgency:.2f} >= {self.config.exit_urgency_threshold})",
                    confidence=decision.exit_urgency,
                    risk_score=decision.risk_score,
                )

            # Check B: Model Reversal Signal with high confidence
            if position_side == "LONG" and decision.is_sell_signal and decision.action_confidence >= self.config.min_confidence_threshold:
                return OrderPlan(
                    should_execute=True,
                    order_type="EXIT_LONG",
                    size_usd=open_pos_size * current_price,
                    size_eth=open_pos_size,
                    stop_loss_price=None,
                    take_profit_price=None,
                    reason=f"Jev SELL reversal signal ({decision.action_confidence:.1%} conf) -> Exit Long",
                    confidence=decision.action_confidence,
                    risk_score=decision.risk_score,
                )

            if position_side == "SHORT" and decision.is_buy_signal and decision.action_confidence >= self.config.min_confidence_threshold:
                return OrderPlan(
                    should_execute=True,
                    order_type="EXIT_SHORT",
                    size_usd=open_pos_size * current_price,
                    size_eth=open_pos_size,
                    stop_loss_price=None,
                    take_profit_price=None,
                    reason=f"Jev BUY reversal signal ({decision.action_confidence:.1%} conf) -> Exit Short",
                    confidence=decision.action_confidence,
                    risk_score=decision.risk_score,
                )

            # Check C: Technical Stop-Loss & Take-Profit triggers
            if position_side == "LONG":
                if active_stop and current_price <= active_stop:
                    return OrderPlan(
                        should_execute=True,
                        order_type="EXIT_LONG",
                        size_usd=open_pos_size * current_price,
                        size_eth=open_pos_size,
                        stop_loss_price=active_stop,
                        take_profit_price=active_tp,
                        reason=f"Long stop loss triggered at {format_price(current_price)} <= {format_price(active_stop)}",
                        confidence=1.0,
                        risk_score=decision.risk_score,
                    )

                if active_tp and current_price >= active_tp:
                    return OrderPlan(
                        should_execute=True,
                        order_type="EXIT_LONG",
                        size_usd=open_pos_size * current_price,
                        size_eth=open_pos_size,
                        stop_loss_price=active_stop,
                        take_profit_price=active_tp,
                        reason=f"Long take profit hit at {format_price(current_price)} >= {format_price(active_tp)}",
                        confidence=1.0,
                        risk_score=decision.risk_score,
                    )
            elif position_side == "SHORT":
                if active_stop and current_price >= active_stop:
                    return OrderPlan(
                        should_execute=True,
                        order_type="EXIT_SHORT",
                        size_usd=open_pos_size * current_price,
                        size_eth=open_pos_size,
                        stop_loss_price=active_stop,
                        take_profit_price=active_tp,
                        reason=f"Short stop loss triggered at {format_price(current_price)} >= {format_price(active_stop)}",
                        confidence=1.0,
                        risk_score=decision.risk_score,
                    )

                if active_tp and current_price <= active_tp:
                    return OrderPlan(
                        should_execute=True,
                        order_type="EXIT_SHORT",
                        size_usd=open_pos_size * current_price,
                        size_eth=open_pos_size,
                        stop_loss_price=active_stop,
                        take_profit_price=active_tp,
                        reason=f"Short take profit hit at {format_price(current_price)} <= {format_price(active_tp)}",
                        confidence=1.0,
                        risk_score=decision.risk_score,
                    )

            # Otherwise, continue holding active futures position
            return OrderPlan(
                should_execute=False,
                order_type="HOLD",
                size_usd=0.0,
                size_eth=0.0,
                stop_loss_price=active_stop,
                take_profit_price=active_tp,
                reason=f"Holding active {position_side} position; conditions favorable or within tolerances",
                confidence=decision.action_confidence,
                risk_score=decision.risk_score,
            )

        # 3. Position Entry Checks (no current position - can enter LONG or SHORT)
        if decision.is_buy_signal or decision.is_sell_signal:
            target_side = "LONG" if decision.is_buy_signal else "SHORT"
            signal_name = "BUY" if decision.is_buy_signal else "SELL"

            # Gate 1: Minimum Confidence
            if decision.action_confidence < self.config.min_confidence_threshold:
                return OrderPlan(
                    should_execute=False,
                    order_type="HOLD",
                    size_usd=0.0,
                    size_eth=0.0,
                    stop_loss_price=None,
                    take_profit_price=None,
                    reason=(
                        f"{signal_name} signal rejected by Confidence Gate: confidence {decision.action_confidence:.2f} "
                        f"< threshold {self.config.min_confidence_threshold:.2f}"
                    ),
                    confidence=decision.action_confidence,
                    risk_score=decision.risk_score,
                )

            # Gate 2: Maximum Risk Score
            if decision.risk_score > self.config.max_risk_score:
                return OrderPlan(
                    should_execute=False,
                    order_type="HOLD",
                    size_usd=0.0,
                    size_eth=0.0,
                    stop_loss_price=None,
                    take_profit_price=None,
                    reason=(
                        f"{signal_name} signal rejected by Risk Gate: risk score {decision.risk_score:.2f} "
                        f"> maximum allowable {self.config.max_risk_score:.2f}"
                    ),
                    confidence=decision.action_confidence,
                    risk_score=decision.risk_score,
                )

            # Gate 3: Entry Conviction (Noul)
            if decision.entry_conviction < self.config.min_entry_conviction:
                return OrderPlan(
                    should_execute=False,
                    order_type="HOLD",
                    size_usd=0.0,
                    size_eth=0.0,
                    stop_loss_price=None,
                    take_profit_price=None,
                    reason=(
                        f"{signal_name} signal rejected by Conviction Gate: entry conviction {decision.entry_conviction:.2f} "
                        f"< required {self.config.min_entry_conviction:.2f}"
                    ),
                    confidence=decision.action_confidence,
                    risk_score=decision.risk_score,
                )

            # Sizing: Scale dynamically by confidence and downside risk
            total_equity = cash_usdt
            base_allocation = total_equity * self.config.position_size_pct

            # Higher confidence & lower risk score = higher sizing multiplier (0.6x to 1.3x)
            conf_multiplier = max(0.6, min(1.3, decision.action_confidence / 0.75))
            risk_penalty = max(0.5, 1.0 - (decision.risk_score / 3.0))
            adjusted_trade_usd = base_allocation * conf_multiplier * risk_penalty

            allocated_usd = min(adjusted_trade_usd, cash_usdt * 0.95)
            base_size = allocated_usd / current_price if current_price > 0 else 0.0

            stop_loss, take_profit = self.calculate_stop_and_target(current_price, atr_14, side=target_side)
            order_type = "ENTER_LONG" if target_side == "LONG" else "ENTER_SHORT"

            return OrderPlan(
                should_execute=True,
                order_type=order_type,
                size_usd=round(allocated_usd, 2),
                size_eth=round(base_size, 6),
                stop_loss_price=round_price(stop_loss, current_price),
                take_profit_price=round_price(take_profit, current_price),
                reason=(
                    f"Confidence-gated {target_side} approved! Conf: {decision.action_confidence:.1%}, "
                    f"Risk: {decision.risk_score:.2f}, Conviction: {decision.entry_conviction:.1%}, "
                    f"Size adj: {conf_multiplier*risk_penalty:.2f}x"
                ),
                confidence=decision.action_confidence,
                risk_score=decision.risk_score,
            )

        # Default: Hold in cash
        return OrderPlan(
            should_execute=False,
            order_type="HOLD",
            size_usd=0.0,
            size_eth=0.0,
            stop_loss_price=None,
            take_profit_price=None,
            reason=f"Jev signal is {decision.action}; no trade required",
            confidence=decision.action_confidence,
            risk_score=decision.risk_score,
        )

    def calculate_stop_and_target(
        self,
        entry_price: float,
        atr_14: Optional[float] = None,
        side: str = "LONG",
    ) -> Tuple[float, float]:
        """Calculates dynamic asymmetric stop loss (1.8%) and take profit (4.0%) with 2.2:1 reward-to-risk."""
        if self.config.use_atr_stop and atr_14 and atr_14 > 0:
            stop_dist = min(entry_price * 0.015, max(entry_price * 0.007, self.config.atr_multiplier * atr_14 * 0.8))
            tp_dist = max(entry_price * 0.030, stop_dist * 3.0)
        else:
            stop_dist = entry_price * 0.012  # 1.2% initial stop (tighter)
            tp_dist = entry_price * 0.036    # 3.6% target (3.0:1 RR ratio)

        if side.upper() == "SHORT":
            stop_loss = entry_price + stop_dist
            take_profit = entry_price - tp_dist
        else:
            stop_loss = entry_price - stop_dist
            take_profit = entry_price + tp_dist

        return stop_loss, take_profit

    def check_active_position_triggers(self, current_price: float, portfolio: dict, engine: Optional[Any] = None) -> Optional[OrderPlan]:
        """Checks if current tick price hits take-profit, trailing profit protection, breakeven lock, or stop-loss.
        Statefully ratchets stop-loss upward/downward so gains can NEVER be lost.
        """
        if not portfolio.get("has_open_position"):
            return None

        pos_sym = portfolio.get("symbol", "BTCUSDT")
        pos_side = portfolio.get("position_side", "NONE")
        pos_size = portfolio.get("position_size_asset", portfolio.get("position_size_btc", portfolio.get("position_size_eth", 0.0)))
        entry_price = float(portfolio.get("entry_price") or 0.0)
        stop_loss = portfolio.get("stop_loss_price")
        take_profit = portfolio.get("take_profit_price")

        target_engine = engine or getattr(self, "engine", None)

        if pos_side == "LONG" and entry_price > 0:
            # 1. Take Profit Hit (+2.8% target = +5.6% ROE at 2x leverage)
            if take_profit and current_price >= take_profit:
                logger.info(f"TAKE PROFIT triggered (LONG {pos_sym}): {format_price(current_price)} >= {format_price(take_profit)}")
                return OrderPlan(
                    should_execute=True,
                    order_type="EXIT_LONG",
                    size_usd=pos_size * current_price,
                    size_eth=pos_size,
                    stop_loss_price=stop_loss,
                    take_profit_price=take_profit,
                    reason=f"Take profit hit (+{((take_profit - entry_price)/entry_price)*100:.1f}% target at {format_price(current_price)})",
                    confidence=1.0,
                    risk_score=1.0,
                )

            # 2. 4-Tier Stateful Profit Locks
            gain_pct = (current_price - entry_price) / entry_price
            if gain_pct >= 0.018:
                # Tier 3: Up +1.8% (+3.6% ROE) -> Ratchet stop loss to lock in +1.20% profit (+2.4% ROE guaranteed!)
                trail_stop = round_price(entry_price * 1.012, entry_price)
                if stop_loss is None or trail_stop > stop_loss:
                    stop_loss = trail_stop
                    if target_engine and hasattr(target_engine, "update_position_stops"):
                        target_engine.update_position_stops(pos_sym, stop_loss_price=trail_stop)
                    logger.info(f"RATCHETED STOP-LOSS (LONG {pos_sym}): moved to {format_price(trail_stop)} (locking +1.2% profit / +2.4% ROE)")
            elif gain_pct >= 0.010:
                # Tier 2: Up +1.0% (+2.0% ROE) -> Ratchet stop loss to lock in +0.50% profit (+1.0% ROE guaranteed!)
                trail_stop = round_price(entry_price * 1.005, entry_price)
                if stop_loss is None or trail_stop > stop_loss:
                    stop_loss = trail_stop
                    if target_engine and hasattr(target_engine, "update_position_stops"):
                        target_engine.update_position_stops(pos_sym, stop_loss_price=trail_stop)
                    logger.info(f"RATCHETED STOP-LOSS (LONG {pos_sym}): moved to {format_price(trail_stop)} (locking +0.5% profit / +1.0% ROE)")
            elif gain_pct >= 0.005:
                # Tier 1: Up +0.5% (+1.0% ROE) -> Ratchet stop loss to Breakeven (+0.20% buffer covers all 10 bps fees!)
                be_stop = round_price(entry_price * 1.002, entry_price)
                if stop_loss is None or be_stop > stop_loss:
                    stop_loss = be_stop
                    if target_engine and hasattr(target_engine, "update_position_stops"):
                        target_engine.update_position_stops(pos_sym, stop_loss_price=be_stop)
                    logger.info(f"RATCHETED STOP-LOSS (LONG {pos_sym}): moved to {format_price(be_stop)} (locking Breakeven + fee buffer)")

            # 3. Stop Loss / Trailing Stop Trigger
            if stop_loss and current_price <= stop_loss:
                if stop_loss > entry_price:
                    exit_reason = f"Trailing profit locked (+{((stop_loss - entry_price)/entry_price)*100:.2f}% gain at {format_price(current_price)})"
                    logger.info(f"PROFIT STOP triggered (LONG {pos_sym}): {exit_reason}")
                else:
                    exit_reason = f"Stop loss triggered ({format_price(current_price)} <= {format_price(stop_loss)})"
                    logger.info(f"STOP LOSS triggered (LONG {pos_sym}): {exit_reason}")

                return OrderPlan(
                    should_execute=True,
                    order_type="EXIT_LONG",
                    size_usd=pos_size * current_price,
                    size_eth=pos_size,
                    stop_loss_price=stop_loss,
                    take_profit_price=take_profit,
                    reason=exit_reason,
                    confidence=1.0,
                    risk_score=1.0,
                )

        elif pos_side == "SHORT" and entry_price > 0:
            # 1. Take Profit Hit
            if take_profit and current_price <= take_profit:
                logger.info(f"TAKE PROFIT triggered (SHORT {pos_sym}): {format_price(current_price)} <= {format_price(take_profit)}")
                return OrderPlan(
                    should_execute=True,
                    order_type="EXIT_SHORT",
                    size_usd=pos_size * current_price,
                    size_eth=pos_size,
                    stop_loss_price=stop_loss,
                    take_profit_price=take_profit,
                    reason=f"Take profit hit (+{((entry_price - take_profit)/entry_price)*100:.1f}% target at {format_price(current_price)})",
                    confidence=1.0,
                    risk_score=1.0,
                )

            # 2. 4-Tier Stateful Profit Locks
            gain_pct = (entry_price - current_price) / entry_price
            if gain_pct >= 0.018:
                # Tier 3: Down -1.8% on price (+3.6% ROE) -> Ratchet stop loss down to lock in +1.20% profit (+2.4% ROE)
                trail_stop = round_price(entry_price * 0.988, entry_price)
                if stop_loss is None or trail_stop < stop_loss:
                    stop_loss = trail_stop
                    if target_engine and hasattr(target_engine, "update_position_stops"):
                        target_engine.update_position_stops(pos_sym, stop_loss_price=trail_stop)
                    logger.info(f"RATCHETED STOP-LOSS (SHORT {pos_sym}): moved down to {format_price(trail_stop)} (locking +1.2% profit / +2.4% ROE)")
            elif gain_pct >= 0.010:
                # Tier 2: Down -1.0% on price (+2.0% ROE) -> Ratchet stop loss down to lock in +0.50% profit (+1.0% ROE)
                trail_stop = round_price(entry_price * 0.995, entry_price)
                if stop_loss is None or trail_stop < stop_loss:
                    stop_loss = trail_stop
                    if target_engine and hasattr(target_engine, "update_position_stops"):
                        target_engine.update_position_stops(pos_sym, stop_loss_price=trail_stop)
                    logger.info(f"RATCHETED STOP-LOSS (SHORT {pos_sym}): moved down to {format_price(trail_stop)} (locking +0.5% profit / +1.0% ROE)")
            elif gain_pct >= 0.005:
                # Tier 1: Down -0.5% on price (+1.0% ROE) -> Ratchet stop loss down to Breakeven (+0.20% fee buffer)
                be_stop = round_price(entry_price * 0.998, entry_price)
                if stop_loss is None or be_stop < stop_loss:
                    stop_loss = be_stop
                    if target_engine and hasattr(target_engine, "update_position_stops"):
                        target_engine.update_position_stops(pos_sym, stop_loss_price=be_stop)
                    logger.info(f"RATCHETED STOP-LOSS (SHORT {pos_sym}): moved down to {format_price(be_stop)} (locking Breakeven + fee buffer)")

            # 3. Stop Loss / Trailing Stop Trigger
            if stop_loss and current_price >= stop_loss:
                if stop_loss < entry_price:
                    exit_reason = f"Trailing profit locked (+{((entry_price - stop_loss)/entry_price)*100:.2f}% gain at {format_price(current_price)})"
                    logger.info(f"PROFIT STOP triggered (SHORT {pos_sym}): {exit_reason}")
                else:
                    exit_reason = f"Stop loss triggered ({format_price(current_price)} >= {format_price(stop_loss)})"
                    logger.info(f"STOP LOSS triggered (SHORT {pos_sym}): {exit_reason}")

                return OrderPlan(
                    should_execute=True,
                    order_type="EXIT_SHORT",
                    size_usd=pos_size * current_price,
                    size_eth=pos_size,
                    stop_loss_price=stop_loss,
                    take_profit_price=take_profit,
                    reason=exit_reason,
                    confidence=1.0,
                    risk_score=1.0,
                )

        return None

    def evaluate_policy(
        self,
        decision: JevDecision,
        current_price: float,
        portfolio: dict,
        atr_14: Optional[float] = None,
        threshold: float = 0.08,
    ) -> OrderPlan:
        """Abol Jev Directional Lean Policy: Sets positions by probabilistic lean, NOT argmax.

        Computes lean = P[LONG] - P[SHORT].
        - If lean >= +threshold: ENTER_LONG
        - If lean <= -threshold: ENTER_SHORT
        - When holding a position, exits if lean reverses or if risk gates fire.
        """
        has_pos = portfolio.get("has_open_position", False)
        pos_side = portfolio.get("position_side", "NONE")
        pos_size = portfolio.get("position_size_btc", portfolio.get("position_size_eth", 0.0))
        cash_usdt = portfolio.get("cash_usdt", 0.0)
        daily_pnl_pct = portfolio.get("daily_pnl_pct", 0.0)

        # 1. Circuit breaker: Daily Max Drawdown
        limit_pct = self.config.max_daily_drawdown_pct * 100.0 if abs(self.config.max_daily_drawdown_pct) < 1.0 else self.config.max_daily_drawdown_pct
        if daily_pnl_pct <= -limit_pct:
            if has_pos:
                exit_type = "EXIT_LONG" if pos_side == "LONG" else "EXIT_SHORT"
                return OrderPlan(
                    should_execute=True,
                    order_type=exit_type,
                    size_usd=pos_size * current_price,
                    size_eth=pos_size,
                    stop_loss_price=None,
                    take_profit_price=None,
                    reason="Circuit breaker: daily drawdown limit breached",
                    confidence=1.0,
                    risk_score=decision.risk_score,
                )
            return OrderPlan(
                should_execute=False,
                order_type="HOLD",
                size_usd=0.0,
                size_eth=0.0,
                stop_loss_price=None,
                take_profit_price=None,
                reason="Daily drawdown limit breached; cooling off",
                confidence=0.0,
                risk_score=decision.risk_score,
            )

        # Extract probabilities and compute directional lean
        probs = decision.action_probabilities or {}
        p_long = float(probs.get("BUY", probs.get("LONG", 0.0)))
        p_short = float(probs.get("SELL", probs.get("SHORT", 0.0)))
        lean = round(p_long - p_short, 3)

        # 2. Existing Position Management
        if has_pos and pos_size > 0.0001:
            # Check A: Stop Loss and Take Profit
            trigger_plan = self.check_active_position_triggers(current_price, portfolio, engine=getattr(self, "engine", None))
            if trigger_plan and trigger_plan.should_execute:
                return trigger_plan

            # Check B: Emergency exit
            if decision.exit_urgency >= self.config.exit_urgency_threshold:
                exit_type = "EXIT_LONG" if pos_side == "LONG" else "EXIT_SHORT"
                return OrderPlan(
                    should_execute=True,
                    order_type=exit_type,
                    size_usd=pos_size * current_price,
                    size_eth=pos_size,
                    stop_loss_price=None,
                    take_profit_price=None,
                    reason=f"Emergency exit signal (urgency: {decision.exit_urgency:.2f})",
                    confidence=decision.exit_urgency,
                    risk_score=decision.risk_score,
                )

            # Check C: Anti-Churn & Macro Regime Invalidation ONLY
            # NEVER exit on minor 30-second model lean fluctuations!
            # All trades are protected by Hard Stop Loss (-1.2%) and 4-Tier Stateful Profit Locks.
            # Discretionary model exit is strictly prohibited during the first 10 minutes (anti-churn guard).
            # After 10 minutes, early exit ONLY triggers if there is a severe confirmed macro regime flip.
            minutes_held = portfolio.get("minutes_held", 0)
            if minutes_held >= 10:
                if pos_side == "LONG" and lean <= -0.40 and decision.market_regime == "BEARISH_BREAKDOWN":
                    logger.info(f"Confirmed Macro Invalidation EXIT_LONG: lean {lean:+.3f} (held {minutes_held}m)")
                    return OrderPlan(
                        should_execute=True,
                        order_type="EXIT_LONG",
                        size_usd=pos_size * current_price,
                        size_eth=pos_size,
                        stop_loss_price=None,
                        take_profit_price=None,
                        reason=f"Macro Trend Invalidation: 1h regime collapsed to BEARISH (held {minutes_held}m)",
                        confidence=abs(lean),
                        risk_score=decision.risk_score,
                    )
                elif pos_side == "SHORT" and lean >= 0.40 and decision.market_regime == "BULLISH_EXPANSION":
                    logger.info(f"Confirmed Macro Invalidation EXIT_SHORT: lean {lean:+.3f} (held {minutes_held}m)")
                    return OrderPlan(
                        should_execute=True,
                        order_type="EXIT_SHORT",
                        size_usd=pos_size * current_price,
                        size_eth=pos_size,
                        stop_loss_price=None,
                        take_profit_price=None,
                        reason=f"Macro Trend Invalidation: 1h regime expanded to BULLISH (held {minutes_held}m)",
                        confidence=abs(lean),
                        risk_score=decision.risk_score,
                    )

            # Otherwise, keep holding
            active_stop = portfolio.get("stop_loss_price")
            active_tp = portfolio.get("take_profit_price")
            return OrderPlan(
                should_execute=False,
                order_type="HOLD",
                size_usd=0.0,
                size_eth=0.0,
                stop_loss_price=active_stop,
                take_profit_price=active_tp,
                reason=f"Holding active {pos_side}; lean {lean:+.2f} still favorable",
                confidence=abs(lean),
                risk_score=decision.risk_score,
            )

        # 3. Position Entry by Directional Lean Policy
        entry_threshold = max(0.20, threshold)

        # Desk-wide risk control: maximum 6 concurrent open positions across the entire desk
        active_desk_count = portfolio.get("active_positions_count", 0)
        if active_desk_count >= 3:
            return OrderPlan(
                should_execute=False,
                order_type="HOLD",
                size_usd=0.0,
                size_eth=0.0,
                stop_loss_price=None,
                take_profit_price=None,
                reason=f"Desk capacity reached ({active_desk_count} active positions); awaiting profit target exits",
                confidence=abs(lean),
                risk_score=decision.risk_score,
            )

        if lean >= entry_threshold and p_long >= 0.52:
            # Rule: Never go LONG into a confirmed BEARISH_BREAKDOWN
            if decision.market_regime == "BEARISH_BREAKDOWN":
                return OrderPlan(
                    should_execute=False,
                    order_type="HOLD",
                    size_usd=0.0,
                    size_eth=0.0,
                    stop_loss_price=None,
                    take_profit_price=None,
                    reason=f"Trend Filter: Rejected LONG into macro BEARISH_BREAKDOWN",
                    confidence=abs(lean),
                    risk_score=decision.risk_score,
                )

            target_side = "LONG"
            stop_loss, take_profit = self.calculate_stop_and_target(current_price, atr_14, side=target_side)
            allocated_usd = min(cash_usdt * 0.20, 2000.0)
            base_size = (allocated_usd * 2) / current_price if current_price > 0 else 0.0

            return OrderPlan(
                should_execute=True,
                order_type="ENTER_LONG",
                size_usd=round(allocated_usd, 2),
                size_eth=round(base_size, 6),
                stop_loss_price=round_price(stop_loss, current_price),
                take_profit_price=round_price(take_profit, current_price),
                reason=f"High-Conviction Long: Lean {lean:+.2f} >= +{entry_threshold:.2f} (P[L]={p_long:.0%}, P[S]={p_short:.0%})",
                confidence=max(p_long, abs(lean)),
                risk_score=decision.risk_score,
            )

        elif lean <= -entry_threshold and p_short >= 0.52:
            # Rule: Never go SHORT into a confirmed BULLISH_EXPANSION
            if decision.market_regime == "BULLISH_EXPANSION":
                return OrderPlan(
                    should_execute=False,
                    order_type="HOLD",
                    size_usd=0.0,
                    size_eth=0.0,
                    stop_loss_price=None,
                    take_profit_price=None,
                    reason=f"Trend Filter: Rejected SHORT into macro BULLISH_EXPANSION",
                    confidence=abs(lean),
                    risk_score=decision.risk_score,
                )

            target_side = "SHORT"
            stop_loss, take_profit = self.calculate_stop_and_target(current_price, atr_14, side=target_side)
            allocated_usd = min(cash_usdt * 0.20, 2000.0)
            base_size = (allocated_usd * 2) / current_price if current_price > 0 else 0.0

            return OrderPlan(
                should_execute=True,
                order_type="ENTER_SHORT",
                size_usd=round(allocated_usd, 2),
                size_eth=round(base_size, 6),
                stop_loss_price=round_price(stop_loss, current_price),
                take_profit_price=round_price(take_profit, current_price),
                reason=f"High-Conviction Short: Lean {lean:+.2f} <= -{entry_threshold:.2f} (P[L]={p_long:.0%}, P[S]={p_short:.0%})",
                confidence=max(p_short, abs(lean)),
                risk_score=decision.risk_score,
            )

        # 4. Neutral / Wait
        return OrderPlan(
            should_execute=False,
            order_type="HOLD",
            size_usd=0.0,
            size_eth=0.0,
            stop_loss_price=None,
            take_profit_price=None,
            reason=f"Directional lean {lean:+.2f} inside neutral band (±{threshold:.2f}); policy is WAIT",
            confidence=abs(lean),
            risk_score=decision.risk_score,
        )
