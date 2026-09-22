"""Integration and unit tests for BTC Perpetual Futures trading bot."""

import sys
from config import load_config
from execution import PaperTradingEngine
from jev_engine import JevEngine
from market_data import MarketDataFetcher
from risk_manager import OrderPlan, RiskManager

def test_derivatives_data():
    print("[1/3] Testing Derivatives Data Ingestion (Deribit)...")
    fetcher = MarketDataFetcher(symbol="BTC-PERPETUAL")
    state = fetcher.get_market_state()
    assert state.derivatives is not None, "Derivatives context is None"
    assert state.derivatives.mark_price > 0, "Mark price is not positive"
    print(f"  OK: Mark Price: ${state.derivatives.mark_price:,.2f}")
    print(f"  OK: 8h Funding Rate: {state.derivatives.funding_rate_8h*100:.4f}% ({state.derivatives.funding_bias})")
    print(f"  OK: Open Interest: ${state.derivatives.open_interest_usd/1e6:.1f}M")
    print(f"  OK: Futures Basis: ${state.derivatives.futures_basis_usd:+.2f}")

def test_futures_execution():
    print("\n[2/3] Testing Futures Execution Engine (Long & Short)...")
    cfg = load_config()
    engine = PaperTradingEngine(cfg.simulation, state_file=None)
    engine.reset()

    # Test Short Order
    plan_short = OrderPlan(
        should_execute=True,
        order_type="ENTER_SHORT",
        size_usd=1000.0,
        size_eth=0.012,
        stop_loss_price=83000.0,
        take_profit_price=80000.0,
        reason="Testing short order",
        confidence=0.85,
        risk_score=0.4,
    )
    trade = engine.execute_order(plan_short, current_price=81500.0)
    assert trade is not None, "Short trade failed to execute"
    assert engine.position_side == "SHORT", "Position side is not SHORT"
    assert engine.liquidation_price > 81500.0, "Short liquidation price should be higher than entry"
    print(f"  OK: Opened Short {trade.amount_btc:.4f} BTC, Margin: ${trade.margin_usd:.2f}, Liq: ${engine.liquidation_price:.2f}")

    # Test Short in profit
    state_short = engine.get_portfolio_state(current_price=80500.0)
    assert state_short["unrealized_pnl_usd"] > 0, "Short should be in profit when price drops"
    print(f"  OK: Short in profit PnL: +${state_short['unrealized_pnl_usd']:.2f} (ROE: {state_short['roe_pct']:+.2f}%)")

    # Close Short
    plan_exit = OrderPlan(
        should_execute=True,
        order_type="EXIT_SHORT",
        size_usd=0.0,
        size_eth=0.0,
        stop_loss_price=None,
        take_profit_price=None,
        reason="Take profit",
        confidence=0.9,
        risk_score=0.3,
    )
    exit_t = engine.execute_order(plan_exit, current_price=80500.0)
    assert exit_t is not None, "Exit short failed"
    assert exit_t.realized_pnl_usd > 0, "Realized PnL should be positive"
    print(f"  OK: Closed Short Realized PnL: +${exit_t.realized_pnl_usd:.2f}, Equity: ${engine.cash_usdt:,.2f}")

def test_jev_perpetual_cycle():
    print("\n[3/3] Testing Jev Live Evaluation on BTC Perpetual Futures...")
    cfg = load_config()
    fetcher = MarketDataFetcher(symbol=cfg.trading_pair)
    state = fetcher.get_market_state()
    engine = PaperTradingEngine(cfg.simulation, state_file=None)
    engine.reset()
    portfolio = engine.get_portfolio_state(state.current_price)

    jev = JevEngine(
        api_key=cfg.typesafe_api_key,
        model=cfg.typesafe_model,
        endpoint=cfg.typesafe_endpoint,
    )
    decision = jev.evaluate(state.to_dict(), portfolio)
    print(f"  OK: Jev Action: {decision.action} ({decision.action_confidence:.1%})")
    print(f"  OK: Probabilities: {decision.action_probabilities}")
    print(f"  OK: Market Regime: {decision.market_regime} ({decision.regime_confidence:.1%})")
    print(f"  OK: Downside Risk: {decision.risk_score:.2f} / 3.0")
    print(f"  OK: Entry Conviction: {decision.entry_conviction:.1%}")

if __name__ == "__main__":
    test_derivatives_data()
    test_futures_execution()
    test_jev_perpetual_cycle()
    print("\n[SUCCESS] All BTC Perpetual Futures tests passed perfectly!")
