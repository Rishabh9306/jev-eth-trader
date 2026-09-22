"""Test suite and verification pipeline for the Jev ETH Trading System.

Tests market data ingestion, technical indicators calculation, confidence gating,
paper trading execution, and TypeSafe Jev API integration.
"""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

from config import RiskConfig, SimulationConfig, load_config
from execution import PaperTradingEngine
from jev_engine import JevDecision, JevEngine
from market_data import MarketDataFetcher
from risk_manager import OrderPlan, RiskManager


class TestTradingSystem(unittest.TestCase):
    """Test suite for the core trading system components."""

    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.fetcher = MarketDataFetcher(symbol=cls.config.trading_pair, timeframe=cls.config.timeframe)
        cls.risk_manager = RiskManager(cls.config.risk)
        # Use isolated test file for simulation
        cls.test_state_file = Path("test_trade_history.json")
        if cls.test_state_file.exists():
            cls.test_state_file.unlink()
        cls.engine = PaperTradingEngine(cls.config.simulation, state_file=cls.test_state_file)

    @classmethod
    def tearDownClass(cls):
        if cls.test_state_file.exists():
            cls.test_state_file.unlink()

    def test_01_market_data_and_indicators(self):
        """Verify real-time BTC market data, indicators, order flow, and sentiment."""
        print(f"\n[TEST] 1. Ingesting live {self.config.trading_pair} market data and computing indicators...")
        ticker = self.fetcher.fetch_current_ticker()
        self.assertIn("price", ticker)
        self.assertGreater(ticker["price"], 0.0)
        self.assertIn("bid", ticker)
        self.assertIn("ask", ticker)
        self.assertIn("vwap_24h", ticker)
        print(f"       Current Price: ${ticker['price']:,.2f} | 24h VWAP: ${ticker['vwap_24h']:,.2f}")

        state = self.fetcher.get_market_state()
        ind = state.indicators
        flow = state.order_flow
        sent = state.sentiment

        self.assertGreaterEqual(ind.rsi_14, 0.0)
        self.assertLessEqual(ind.rsi_14, 100.0)
        self.assertGreater(ind.ema_9, 0.0)
        self.assertGreater(ind.ema_21, 0.0)
        self.assertGreater(ind.atr_14, 0.0)
        self.assertGreater(flow.taker_buy_sell_ratio, 0.0)
        self.assertGreaterEqual(sent.fear_and_greed_score, 0)
        print(f"       5m RSI: {ind.rsi_14} | 1h Trend: {ind.trend_alignment_1h} | ATR: ${ind.atr_14:.2f}")
        print(f"       Taker Ratio: {flow.taker_buy_sell_ratio:.3f} ({flow.flow_bias}) | Fear & Greed: {sent.fear_and_greed_score}/100")

    def test_02_confidence_gating(self):
        """Verify that confidence thresholds reject low-confidence signals."""
        print("\n[TEST] 2. Testing Confidence-Gated Routing rules...")
        portfolio = {"cash_usdt": 10000.0, "has_open_position": False, "daily_pnl_pct": 0.0}

        # Case A: Low Confidence BUY signal (e.g. 0.55 < 0.70 threshold) -> MUST BE REJECTED
        low_conf_decision = JevDecision(
            action="BUY",
            action_confidence=0.55,
            action_probabilities={"BUY": 0.55, "HOLD": 0.35, "SELL": 0.10},
            market_regime="BULLISH_TREND",
            regime_confidence=0.60,
            risk_score=1.0,
            risk_confidence=0.70,
            entry_conviction=0.80,
            exit_urgency=0.05,
            model_version="test",
            tokens_used=100,
            raw_response={},
        )
        plan_a = self.risk_manager.evaluate_decision(low_conf_decision, 2500.0, portfolio, atr_14=30.0)
        self.assertFalse(plan_a.should_execute, "Low-confidence BUY should be rejected")
        self.assertIn("Confidence Gate", plan_a.reason)
        print(f"       Case A (Low Confidence 55%): Correctly REJECTED -> {plan_a.reason}")

        # Case B: High Risk Score (e.g. 2.4 > 1.5 max) -> MUST BE REJECTED
        high_risk_decision = JevDecision(
            action="BUY",
            action_confidence=0.85,
            action_probabilities={"BUY": 0.85, "HOLD": 0.10, "SELL": 0.05},
            market_regime="VOLATILE_CHOP",
            regime_confidence=0.80,
            risk_score=2.4,  # High risk
            risk_confidence=0.90,
            entry_conviction=0.85,
            exit_urgency=0.05,
            model_version="test",
            tokens_used=100,
            raw_response={},
        )
        plan_b = self.risk_manager.evaluate_decision(high_risk_decision, 2500.0, portfolio, atr_14=30.0)
        self.assertFalse(plan_b.should_execute, "High-risk BUY should be rejected")
        self.assertIn("Risk Gate", plan_b.reason)
        print(f"       Case B (High Risk 2.4/3.0): Correctly REJECTED -> {plan_b.reason}")

        # Case C: Valid High-Confidence, Low-Risk BUY -> MUST BE APPROVED
        valid_decision = JevDecision(
            action="BUY",
            action_confidence=0.82,
            action_probabilities={"BUY": 0.82, "HOLD": 0.15, "SELL": 0.03},
            market_regime="BULLISH_TREND",
            regime_confidence=0.85,
            risk_score=0.9,
            risk_confidence=0.90,
            entry_conviction=0.78,
            exit_urgency=0.05,
            model_version="test",
            tokens_used=100,
            raw_response={},
        )
        plan_c = self.risk_manager.evaluate_decision(valid_decision, 2500.0, portfolio, atr_14=30.0)
        self.assertTrue(plan_c.should_execute, "High-confidence BUY should be approved")
        self.assertEqual(plan_c.order_type, "ENTER_LONG")
        self.assertIsNotNone(plan_c.stop_loss_price)
        self.assertIsNotNone(plan_c.take_profit_price)
        print(f"       Case C (High Confidence 82%, Risk 0.9): Correctly APPROVED -> {plan_c.reason}")
        print(f"              Calculated SL: ${plan_c.stop_loss_price:,.2f} | TP: ${plan_c.take_profit_price:,.2f}")

    def test_03_paper_trading_execution(self):
        """Verify simulated order execution, fee accounting, and PnL calculation."""
        print("\n[TEST] 3. Testing Paper Trading execution and accounting...")
        current_price = 2500.0
        plan = OrderPlan(
            should_execute=True,
            order_type="ENTER_LONG",
            size_usd=1000.0,
            size_eth=0.4,
            stop_loss_price=2450.0,
            take_profit_price=2600.0,
            reason="Approved BUY test",
            confidence=0.85,
            risk_score=0.8,
        )

        # 1. Execute BUY / LONG
        trade_buy = self.engine.execute_order(plan, current_price)
        self.assertIsNotNone(trade_buy)
        self.assertIn(trade_buy.side, ("BUY", "LONG"))
        self.assertGreater(trade_buy.amount_eth, 0.0)
        self.assertGreater(trade_buy.fee_usd, 0.0)
        print(f"       Executed {trade_buy.side}: {trade_buy.amount_eth:.4f} BTC @ ${trade_buy.effective_price:.2f} (Fee: ${trade_buy.fee_usd:.2f})")

        # Check Portfolio
        port = self.engine.get_portfolio_state(current_price)
        self.assertTrue(port["has_open_position"])
        self.assertAlmostEqual(port["position_size_eth"], trade_buy.amount_eth, places=4)

        # 2. Execute EXIT_LONG at higher price (e.g. $2600 = profit)
        sell_price = 2600.0
        exit_plan = OrderPlan(
            should_execute=True,
            order_type="EXIT_LONG",
            size_usd=port["position_size_eth"] * sell_price,
            size_eth=port["position_size_eth"],
            stop_loss_price=None,
            take_profit_price=None,
            reason="Take profit triggered",
            confidence=1.0,
            risk_score=0.8,
        )
        trade_sell = self.engine.execute_order(exit_plan, sell_price)
        self.assertIsNotNone(trade_sell)
        self.assertIn(trade_sell.side, ("SELL", "EXIT_LONG"))
        self.assertGreater(trade_sell.realized_pnl_usd, 0.0)
        print(f"       Executed {trade_sell.side}: {trade_sell.amount_eth:.4f} BTC @ ${trade_sell.effective_price:.2f}")
        print(f"       Realized PnL: +${trade_sell.realized_pnl_usd:.2f} ({trade_sell.realized_pnl_pct:+.2f}%)")

        # Check Summary
        summary = self.engine.get_performance_summary()
        self.assertEqual(summary["closed_trades"], 1)
        self.assertEqual(summary["win_rate_pct"], 100.0)
        self.assertGreater(summary["total_realized_pnl_usd"], 0.0)

    def test_04_jev_payload_format(self):
        """Verify the JSON state and questions payload conforms to TypeSafe API schema."""
        print("\n[TEST] 4. Validating Jev System One question and state schemas...")
        engine = JevEngine(api_key="test-key")
        questions = engine.build_trading_questions()

        self.assertIn("trade_action", questions)
        self.assertEqual(questions["trade_action"]["type"], "choice")
        self.assertIn("BUY", questions["trade_action"]["criteria"])
        self.assertIn("SELL", questions["trade_action"]["criteria"])
        self.assertIn("HOLD", questions["trade_action"]["criteria"])

        self.assertIn("market_regime", questions)
        self.assertEqual(questions["market_regime"]["type"], "choice")

        self.assertIn("risk_level", questions)
        self.assertEqual(questions["risk_level"]["type"], "score")
        self.assertIsInstance(questions["risk_level"]["criteria"], list)

        self.assertIn("entry_conviction", questions)
        self.assertEqual(questions["entry_conviction"]["type"], "noul")

        self.assertIn("exit_urgency", questions)
        self.assertEqual(questions["exit_urgency"]["type"], "noul")
        print("       All 5 System One primitives (Choice, Score, Noul) correctly formed!")

    def test_05_multi_asset_execution(self):
        """Verify concurrent multi-asset positions under shared margin collateral."""
        print("\n[TEST] 5. Testing multi-asset concurrent positions (BTC + ETH + SOL)...")
        # 1. Open LONG BTC
        btc_plan = OrderPlan(
            should_execute=True,
            order_type="ENTER_LONG",
            size_usd=1000.0,
            size_eth=0.012,
            stop_loss_price=80000.0,
            take_profit_price=90000.0,
            reason="Multi-asset LONG BTC",
            confidence=0.85,
            risk_score=0.8,
        )
        t_btc = self.engine.execute_order(btc_plan, 85000.0, symbol="BTCUSDT")
        self.assertIsNotNone(t_btc)
        self.assertEqual(t_btc.symbol, "BTCUSDT")

        # 2. Open SHORT ETH simultaneously
        eth_plan = OrderPlan(
            should_execute=True,
            order_type="ENTER_SHORT",
            size_usd=1000.0,
            size_eth=0.35,
            stop_loss_price=3000.0,
            take_profit_price=2700.0,
            reason="Multi-asset SHORT ETH",
            confidence=0.82,
            risk_score=0.9,
        )
        t_eth = self.engine.execute_order(eth_plan, 2800.0, symbol="ETHUSDT")
        self.assertIsNotNone(t_eth)
        self.assertEqual(t_eth.symbol, "ETHUSDT")

        # Verify state shows 2 distinct positions
        self.assertEqual(len(self.engine.positions), 2)
        self.assertIn("BTCUSDT", self.engine.positions)
        self.assertIn("ETHUSDT", self.engine.positions)

        # Check symbol-filtered portfolio states
        p_btc = self.engine.get_portfolio_state(86000.0, symbol="BTCUSDT")
        p_eth = self.engine.get_portfolio_state(2750.0, symbol="ETHUSDT")

        self.assertEqual(p_btc["symbol"], "BTCUSDT")
        self.assertEqual(p_btc["position_side"], "LONG")
        self.assertGreater(p_btc["unrealized_pnl_usd"], 0)  # Price rose from 85k to 86k

        self.assertEqual(p_eth["symbol"], "ETHUSDT")
        self.assertEqual(p_eth["position_side"], "SHORT")
        self.assertGreater(p_eth["unrealized_pnl_usd"], 0)  # Short ETH from 2800 to 2750 is profit

        # Check marker filtering
        markers_btc = self.engine.get_chart_markers(symbol="BTCUSDT")
        markers_eth = self.engine.get_chart_markers(symbol="ETHUSDT")
        self.assertTrue(all("LONG" in m["text"] for m in markers_btc if "BUY" in m["text"]))
        self.assertTrue(all("SHORT" in m["text"] for m in markers_eth if "SELL" in m["text"]))

        # 3. Close ETH position independently
        exit_eth_plan = OrderPlan(
            should_execute=True,
            order_type="EXIT_SHORT",
            size_usd=0,
            size_eth=0,
            stop_loss_price=None,
            take_profit_price=None,
            reason="Take profit ETH",
            confidence=1.0,
            risk_score=0.8,
        )
        t_exit_eth = self.engine.execute_order(exit_eth_plan, 2750.0, symbol="ETHUSDT")
        self.assertIsNotNone(t_exit_eth)
        self.assertIn("ETHUSDT", t_exit_eth.symbol)

        # ETH closed, BTC still active!
        self.assertNotIn("ETHUSDT", self.engine.positions)
        self.assertIn("BTCUSDT", self.engine.positions)
        print("       Multi-asset concurrent execution & symbol isolation verified successfully!")

    def test_06_master_portfolio_overview(self):
        """Verify master portfolio overview aggregation, concurrent PnL, and realized accounting."""
        print("\n[TEST] 6. Testing master portfolio overview aggregation and multi-stock PnL...")
        self.engine.reset()

        # 1. Open BTC LONG
        btc_plan = OrderPlan(should_execute=True, order_type="ENTER_LONG", size_usd=2000.0, size_eth=0.023, stop_loss_price=84000.0, take_profit_price=87000.0, reason="BTC Long", confidence=0.85, risk_score=0.9)
        self.engine.execute_order(btc_plan, 85000.0, symbol="BTCUSDT")

        # 2. Open SOL SHORT
        sol_plan = OrderPlan(should_execute=True, order_type="ENTER_SHORT", size_usd=1500.0, size_eth=25.0, stop_loss_price=125.0, take_profit_price=110.0, reason="SOL Short", confidence=0.80, risk_score=1.0)
        self.engine.execute_order(sol_plan, 120.0, symbol="SOLUSDT")

        # 3. Query Master Portfolio Overview with live price movements
        prices = {"BTCUSDT": 86000.0, "SOLUSDT": 115.0}  # Both in profit!
        overview = self.engine.get_master_portfolio_overview(prices)
        summ = overview["portfolio_summary"]

        self.assertEqual(summ["active_positions_count"], 2)
        self.assertGreater(summ["total_unrealized_pnl_usd"], 0)
        self.assertGreater(summ["total_margin_used_usdt"], 3000.0)
        self.assertGreater(summ["margin_utilization_pct"], 15.0)
        self.assertEqual(len(overview["active_positions"]), 2)

        # 4. Close SOL SHORT with profit
        close_sol = OrderPlan(should_execute=True, order_type="EXIT_SHORT", size_usd=0, size_eth=0, stop_loss_price=None, take_profit_price=None, reason="Take profit SOL", confidence=1.0, risk_score=1.0)
        self.engine.execute_order(close_sol, 115.0, symbol="SOLUSDT")

        # Verify get_full_trade_history() contains the closed trade
        history = self.engine.get_full_trade_history()
        closed_sol = [h for h in history if h.get("symbol") == "SOLUSDT" and h.get("status") == "CLOSED"]
        self.assertGreater(len(closed_sol), 0)
        self.assertGreater(closed_sol[0]["realized_pnl_usd"], 0)

        # 5. Query Master Portfolio Overview again: SOL Realized PnL booked!
        overview2 = self.engine.get_master_portfolio_overview(prices)
        summ2 = overview2["portfolio_summary"]

        self.assertEqual(summ2["active_positions_count"], 1)
        self.assertGreater(summ2["total_realized_pnl_usd"], 0)
        self.assertIn("SOLUSDT", overview2["symbol_realized_pnl"])
        self.assertGreater(overview2["symbol_realized_pnl"]["SOLUSDT"], 0)
        self.assertEqual(overview2["symbol_trade_count"]["SOLUSDT"], 2)
        print(f"       Master Overview Verified: {summ2['active_positions_count']} Active, Realized PnL: +${summ2['total_realized_pnl_usd']:.2f}")

    def test_07_isolated_10k_accounts(self):
        """Verify that each market gets its own independent $10,000.00 capital account."""
        print("\n[TEST] 7. Testing independent $10,000.00 accounts per market...")
        self.engine.reset()

        btc_acc = self.engine.get_account("BTCUSDT")
        eth_acc = self.engine.get_account("ETHUSDT")
        sol_acc = self.engine.get_account("SOLUSDT")

        # All start at exactly $10,000.00
        self.assertEqual(btc_acc.cash_usdt, 10000.0)
        self.assertEqual(eth_acc.cash_usdt, 10000.0)
        self.assertEqual(sol_acc.cash_usdt, 10000.0)

        # Open trade on BTC with $2,000 margin
        btc_plan = OrderPlan(should_execute=True, order_type="ENTER_LONG", size_usd=2000.0, size_eth=0.023, stop_loss_price=80000.0, take_profit_price=90000.0, reason="BTC Long", confidence=0.85, risk_score=0.9)
        self.engine.execute_order(btc_plan, 85000.0, symbol="BTCUSDT")

        # BTC cash reduced to $8,000
        self.assertEqual(btc_acc.cash_usdt, 8000.0)

        # ETH and SOL cash remain pristine at $10,000.00!
        self.assertEqual(eth_acc.cash_usdt, 10000.0)
        self.assertEqual(sol_acc.cash_usdt, 10000.0)

        # Open trade on ETH with $3,000 margin
        eth_plan = OrderPlan(should_execute=True, order_type="ENTER_LONG", size_usd=3000.0, size_eth=1.0, stop_loss_price=2600.0, take_profit_price=3000.0, reason="ETH Long", confidence=0.85, risk_score=0.9)
        self.engine.execute_order(eth_plan, 2700.0, symbol="ETHUSDT")

        # ETH cash reduced to $7,000, SOL remains $10,000
        self.assertEqual(eth_acc.cash_usdt, 7000.0)
        self.assertEqual(sol_acc.cash_usdt, 10000.0)

        overview = self.engine.get_master_portfolio_overview({"BTCUSDT": 85000.0, "ETHUSDT": 2700.0, "SOLUSDT": 115.0})
        self.assertEqual(overview["symbol_cash"]["BTCUSDT"], 8000.0)
        self.assertEqual(overview["symbol_cash"]["ETHUSDT"], 7000.0)
        self.assertEqual(overview["symbol_cash"]["SOLUSDT"], 10000.0)
        self.assertEqual(overview["symbol_capital"]["SOLUSDT"], 10000.0)
        print("       Independent $10,000.00 account isolation per market verified successfully!")



def main():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTradingSystem)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
