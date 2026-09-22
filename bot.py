"""Main orchestrator and trading loop for the TypeSafe Jev ETH Trading Bot.

Polls real-time market data, evaluates indicators, invokes Jev's System One
decision model, enforces confidence and risk gates, and executes orders.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from colorama import Fore, Style, init

from config import BotConfig, load_config
from execution import PaperTradingEngine
from jev_engine import JevDecision, JevEngine
from market_data import MarketDataFetcher, MarketState
from risk_manager import OrderPlan, RiskManager

# Ensure stdout supports UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Initialize terminal colors
init(autoreset=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("JevTrader")


def print_banner(config: BotConfig):
    """Prints startup banner and active configuration."""
    base_asset = config.trading_pair.split("/")[0]
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*70}")
    print(f"  [TYPESAFE JEV {base_asset} TRADING SYSTEM]")
    print(f"  Model: {config.typesafe_model} | Pair: {config.trading_pair} | Mode: {config.trading_mode.upper()}")
    print(f"  Timeframe: {config.timeframe} | Interval: {config.cycle_interval_seconds}s")
    print(f"  Min Confidence Gate: {config.risk.min_confidence_threshold:.0%} | Max Risk Score: {config.risk.max_risk_score}")
    print(f"{'='*70}{Style.RESET_ALL}\n")


def format_cycle_report(
    state: MarketState,
    decision: JevDecision,
    plan: OrderPlan,
    portfolio: dict,
    perf: dict,
):
    """Renders formatted status card to console."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ind = state.indicators
    flow = state.order_flow
    sent = state.sentiment

    # Header
    print(f"\n{Fore.YELLOW}{Style.BRIGHT}--- [ CYCLE TELEMETRY @ {ts} ] ---{Style.RESET_ALL}")

    # 1. Market Data
    p_color = Fore.GREEN if state.change_24h_pct >= 0 else Fore.RED
    vwap_color = Fore.GREEN if flow.price_to_vwap_pct >= 0 else Fore.RED
    print(f"{Fore.WHITE}{Style.BRIGHT}MARKET:{Style.RESET_ALL} {state.symbol}: {p_color}${state.current_price:,.2f} ({state.change_24h_pct:+.2f}% 24h){Style.RESET_ALL} | Spread: {state.spread_bps:.1f} bps | Vol: ${state.volume_24h * state.current_price:,.0f}")
    print(f"  24h VWAP: {vwap_color}${flow.vwap_24h:,.2f} ({flow.price_to_vwap_pct:+.2f}%){Style.RESET_ALL} | 1h Support: ${ind.support_level:,.2f} | 1h Resistance: ${ind.resistance_level:,.2f}")
    print(f"  Trends: 1h [{ind.trend_alignment_1h}] | 5m [{ind.trend_alignment_5m.upper()}] | RSI: 5m={ind.rsi_14:.1f}, 1h={ind.rsi_1h:.1f} | ATR: ${ind.atr_14:,.2f}")

    # 2. Order Flow & Sentiment (Abol's additions)
    flow_color = Fore.GREEN if "BUY" in flow.flow_bias else Fore.RED if "SELL" in flow.flow_bias else Fore.YELLOW
    fng_color = Fore.GREEN if sent.fear_and_greed_score >= 60 else Fore.RED if sent.fear_and_greed_score <= 40 else Fore.YELLOW
    print(f"\n{Fore.WHITE}{Style.BRIGHT}ORDER FLOW & SENTIMENT:{Style.RESET_ALL}")
    print(f"  * Taker Flow Ratio: {flow_color}{flow.taker_buy_sell_ratio:.3f} ({flow.flow_bias}){Style.RESET_ALL} | Buys: {flow.taker_buy_volume_btc:,.1f} vs Sells: {flow.taker_sell_volume_btc:,.1f}")
    print(f"  * Fear & Greed: {fng_color}{sent.fear_and_greed_score}/100 ({sent.fear_and_greed_sentiment}){Style.RESET_ALL}")

    # 3. Jev System One Decisions
    action_color = Fore.GREEN if decision.action == "BUY" else Fore.RED if decision.action == "SELL" else Fore.YELLOW
    print(f"\n{Fore.WHITE}{Style.BRIGHT}JEV SYSTEM ONE EVALUATION ({decision.model_version}):{Style.RESET_ALL}")
    print(f"  * Action: {action_color}{Style.BRIGHT}{decision.action}{Style.RESET_ALL} (Confidence: {decision.action_confidence:.1%}) | Probs: {decision.action_probabilities}")
    print(f"  * Market Regime: {Fore.CYAN}{decision.market_regime}{Style.RESET_ALL} (Confidence: {decision.regime_confidence:.1%})")
    print(f"  * Downside Risk Score: {Fore.MAGENTA}{decision.risk_score:.2f} / 3.00{Style.RESET_ALL} (Confidence: {decision.risk_confidence:.1%})")
    print(f"  * Entry Conviction (Noul): {Fore.LIGHTBLUE_EX}{decision.entry_conviction:.1%}{Style.RESET_ALL} | Exit Urgency (Noul): {Fore.LIGHTMAGENTA_EX}{decision.exit_urgency:.1%}{Style.RESET_ALL}")

    # 4. Risk & Routing Decision
    plan_color = Fore.GREEN if plan.order_type == "ENTER_LONG" else Fore.RED if plan.order_type == "EXIT_LONG" else Fore.WHITE
    print(f"\n{Fore.WHITE}{Style.BRIGHT}ROUTING & RISK GATE:{Style.RESET_ALL}")
    print(f"  * Decision: {plan_color}{Style.BRIGHT}{plan.order_type}{Style.RESET_ALL} (Execute: {plan.should_execute})")
    print(f"  * Rationale: {plan.reason}")
    if plan.stop_loss_price:
        print(f"  * Target Levels: Stop Loss = ${plan.stop_loss_price:,.2f} | Take Profit = ${plan.take_profit_price:,.2f}")

    # 5. Portfolio State
    has_pos = portfolio["has_open_position"]
    pos_color = Fore.GREEN if has_pos else Fore.YELLOW
    pnl_color = Fore.GREEN if portfolio["unrealized_pnl_usd"] >= 0 else Fore.RED
    print(f"\n{Fore.WHITE}{Style.BRIGHT}PORTFOLIO STATUS:{Style.RESET_ALL}")
    print(f"  * Total Equity: ${portfolio['total_equity_usdt']:,.2f} USDT | Cash: ${portfolio['cash_usdt']:,.2f} USDT")
    if has_pos:
        pos_size = portfolio.get('position_size_btc', portfolio.get('position_size_eth', 0.0))
        print(f"  * Active Position: {pos_color}{pos_size:.6f} {state.symbol.split('/')[0]}{Style.RESET_ALL} @ ${portfolio['entry_price']:,.2f} | Value: ${portfolio['position_value_usd']:,.2f}")
        print(f"  * Unrealized PnL: {pnl_color}${portfolio['unrealized_pnl_usd']:+,.2f} ({portfolio['unrealized_pnl_pct']:+.2f}%){Style.RESET_ALL}")
    else:
        print(f"  * Active Position: None (100% Cash / Inactive)")

    print(f"  * Total Closed Trades: {perf['total_trades']} | Win Rate: {perf['win_rate_pct']:.1f}% | Realized PnL: ${perf['total_realized_pnl_usd']:+,.2f}")
    print(f"{Fore.YELLOW}{'-'*70}{Style.RESET_ALL}")


def create_mock_decision(state: MarketState) -> JevDecision:
    """Generates deterministic mock decision for testing without API key."""
    ind = state.indicators
    flow = state.order_flow
    if flow.flow_bias == "AGGRESSIVE_BUYING" and "BULLISH" in ind.trend_alignment_1h:
        action = "BUY"
        conf = 0.81
        probs = {"BUY": 0.80, "HOLD": 0.15, "SELL": 0.05}
        regime = "BULLISH_EXPANSION"
        risk = 0.9
        entry = 0.76
        exit_urg = 0.08
    elif flow.flow_bias == "AGGRESSIVE_SELLING" or "BEARISH" in ind.trend_alignment_1h:
        action = "SELL"
        conf = 0.74
        probs = {"SELL": 0.75, "HOLD": 0.20, "BUY": 0.05}
        regime = "BEARISH_CONTRACTION"
        risk = 2.1
        entry = 0.18
        exit_urg = 0.75
    else:
        action = "HOLD"
        conf = 0.82
        probs = {"HOLD": 0.82, "BUY": 0.10, "SELL": 0.08}
        regime = "RANGE_BOUND_CONSOLIDATION"
        risk = 1.2
        entry = 0.32
        exit_urg = 0.15

    return JevDecision(
        action=action,
        action_confidence=conf,
        action_probabilities=probs,
        market_regime=regime,
        regime_confidence=0.82,
        risk_score=risk,
        risk_confidence=0.88,
        entry_conviction=entry,
        exit_urgency=exit_urg,
        model_version="jev-mock-offline",
        tokens_used=0,
        raw_response={"mock": True},
    )


def run_cycle(
    config: BotConfig,
    data_fetcher: MarketDataFetcher,
    jev: JevEngine,
    risk_manager: RiskManager,
    execution: PaperTradingEngine,
    force_mock: bool = False,
):
    """Executes a single end-to-end trading loop cycle."""
    # 1. Fetch live market state and indicators
    market_state = data_fetcher.get_market_state()
    current_price = market_state.current_price

    # 2. Get current portfolio status
    portfolio = execution.get_portfolio_state(current_price)

    # 3. Query TypeSafe Jev model
    if force_mock or not config.has_valid_api_key:
        if not force_mock and not config.has_valid_api_key:
            logger.info("Notice: No valid TYPESAFE_API_KEY in .env. Running cycle using local mock evaluator.")
        decision = create_mock_decision(market_state)
    else:
        try:
            decision = jev.evaluate(market_state.to_dict(), portfolio)
        except Exception as e:
            logger.error(f"Jev evaluation failed: {e}. Falling back to HOLD.")
            decision = JevDecision(
                action="HOLD",
                action_confidence=0.0,
                action_probabilities={"HOLD": 1.0},
                market_regime="UNKNOWN",
                regime_confidence=0.0,
                risk_score=3.0,
                risk_confidence=0.0,
                entry_conviction=0.0,
                exit_urgency=0.0,
                model_version="error-fallback",
                tokens_used=0,
                raw_response={"error": str(e)},
            )

    # 4. Filter decision through Risk Manager & Confidence Gates
    plan = risk_manager.evaluate_decision(
        decision=decision,
        current_price=current_price,
        portfolio=portfolio,
        atr_14=market_state.indicators.atr_14,
    )

    # 5. Execute Order if approved
    trade = None
    if plan.should_execute:
        trade = execution.execute_order(plan, current_price)
        # Update portfolio after fill
        portfolio = execution.get_portfolio_state(current_price)

    # 6. Display telemetry
    perf = execution.get_performance_summary()
    format_cycle_report(market_state, decision, plan, portfolio, perf)


def main():
    parser = argparse.ArgumentParser(description="TypeSafe Jev ETH Automated Trading Bot")
    parser.add_argument("--once", action="store_true", help="Run a single evaluation cycle and exit")
    parser.add_argument("--mock", action="store_true", help="Run with mock decision model for dry testing")
    parser.add_argument("--duration", type=int, default=None, help="Run for specified duration in seconds then exit")
    args = parser.parse_args()

    config = load_config()
    print_banner(config)

    data_fetcher = MarketDataFetcher(symbol=config.trading_pair, timeframe=config.timeframe)
    jev = JevEngine(
        api_key=config.typesafe_api_key,
        model=config.typesafe_model,
        endpoint=config.typesafe_endpoint,
    )
    risk_manager = RiskManager(config.risk)
    execution = PaperTradingEngine(config.simulation)

    if args.once:
        logger.info("Executing single cycle (--once)...")
        run_cycle(config, data_fetcher, jev, risk_manager, execution, force_mock=args.mock)
        return

    logger.info(f"Starting continuous daemon (interval: {config.cycle_interval_seconds}s). Press Ctrl+C to stop.")
    start_time = time.time()
    try:
        while True:
            run_cycle(config, data_fetcher, jev, risk_manager, execution, force_mock=args.mock)
            if args.duration and (time.time() - start_time) >= args.duration:
                logger.info(f"Duration limit of {args.duration}s reached. Finished run.")
                break
            time.sleep(config.cycle_interval_seconds)
    except KeyboardInterrupt:
        logger.info("\nShutdown signal received. Exiting safely.")
        sys.exit(0)


if __name__ == "__main__":
    main()
