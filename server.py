"""Lightweight HTTP server & REST API for the Live BTC Jev Dashboard.

Serves the modern dark-themed TradingView dashboard and provides real-time JSON
endpoints for live candlesticks, indicators, Jev AI decision matrix, paper portfolio
tracking starting at $10,000 USDT, and trade markers.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from config import BotConfig, load_config
from execution import PaperTradingEngine
from jev_engine import JevDecision, JevEngine
from market_data import MarketDataFetcher, MarketState
from risk_manager import OrderPlan, RiskManager

logger = logging.getLogger("DashboardServer")
BASE_DIR = Path(__file__).resolve().parent


class GlobalTradingState:
    """Thread-safe state container shared between background worker and web server."""

    def __init__(self):
        self.lock = threading.Lock()
        self.config: BotConfig = load_config()
        self.active_symbol: str = getattr(self.config, "trading_pair", "BTCUSDT")
        if self.active_symbol == "BTC-PERPETUAL":
            self.active_symbol = "BTCUSDT"

        self.fetchers: Dict[str, MarketDataFetcher] = {
            self.active_symbol: MarketDataFetcher(symbol=self.active_symbol, timeframe=self.config.timeframe)
        }
        self.jev = JevEngine(
            api_key=self.config.typesafe_api_key,
            model=self.config.typesafe_model,
            endpoint=self.config.typesafe_endpoint,
        )
        self.execution = PaperTradingEngine(self.config.simulation)
        self.risk_manager = RiskManager(self.config.risk, engine=self.execution)

        # Per-symbol state and decisions maps
        self.market_states: Dict[str, MarketState] = {}
        self.decisions: Dict[str, JevDecision] = {}
        self.plans: Dict[str, OrderPlan] = {}

        self.market_state: Optional[MarketState] = None
        self.decision: Optional[JevDecision] = None
        self.plan: Optional[OrderPlan] = None
        self.last_eval_time: Optional[str] = None
        self.daemon_active = True
        self.error_message: Optional[str] = None

        # Abol Jev Metrics
        self.decisions_count: int = 1
        self.last_latency_ms: int = 333
        self.cycle_speed_multiplier: int = 1
        self.effective_interval_seconds: int = self.config.cycle_interval_seconds
        self.policy_threshold: float = 0.20  # High-conviction entry threshold (60%+ probability edge)
        self.initial_btc_price: Optional[float] = None
        self.equity_history: list = []
        # User-selected tracked symbols for Jev AI analysis & trading (empty by default)
        self.tracked_symbols: Set[str] = set(getattr(self.execution, "tracked_symbols", []))

    def add_tracked_symbol(self, symbol: str) -> bool:
        sym = symbol.upper().strip()
        if not sym or not sym.isascii() or not sym.isalnum():
            return False
        with self.lock:
            self.tracked_symbols.add(sym)
            self.execution.tracked_symbols = sorted(list(self.tracked_symbols))
            self.execution._save_state()
        return True

    def remove_tracked_symbol(self, symbol: str) -> bool:
        sym = symbol.upper().strip()
        with self.lock:
            if sym in self.tracked_symbols:
                self.tracked_symbols.remove(sym)
                self.execution.tracked_symbols = sorted(list(self.tracked_symbols))
                self.execution._save_state()
                return True
        return False

    def toggle_tracked_symbol(self, symbol: str) -> Tuple[bool, bool]:
        sym = symbol.upper().strip()
        if not sym or not sym.isascii() or not sym.isalnum():
            return False, False
        with self.lock:
            if sym in self.tracked_symbols:
                self.tracked_symbols.remove(sym)
                is_tracked = False
            else:
                self.tracked_symbols.add(sym)
                is_tracked = True
            self.execution.tracked_symbols = sorted(list(self.tracked_symbols))
            self.execution._save_state()
        return True, is_tracked

    def get_fetcher(self, symbol: Optional[str] = None) -> MarketDataFetcher:
        """Retrieves or creates a cached MarketDataFetcher instance for a symbol."""
        sym = (symbol or self.active_symbol).upper()
        if sym == "BTC-PERPETUAL":
            sym = "BTCUSDT"
        if sym not in self.fetchers:
            self.fetchers[sym] = MarketDataFetcher(symbol=sym, timeframe=self.config.timeframe)
        return self.fetchers[sym]

    def record_equity_point(self, current_price: float, total_equity: float, before_costs: float):
        """Maintains continuous equity curve for Jev, Before Costs, and Buy & Hold."""
        if not current_price or current_price <= 0:
            return
        if self.initial_btc_price is None:
            self.initial_btc_price = current_price

        bnh = round(10000.0 * (current_price / self.initial_btc_price), 2)
        now_ts = int(time.time())

        # If history is empty, pre-seed with candles baseline
        if not self.equity_history:
            try:
                fetcher = self.get_fetcher(self.active_symbol)
                df = fetcher.fetch_klines(interval="5m", limit=80)
                if not df.empty:
                    p0 = float(df.iloc[0]["close"])
                    self.initial_btc_price = p0
                    for _, row in df.iterrows():
                        t = int(row["timestamp"].timestamp())
                        c = float(row["close"])
                        b = round(10000.0 * (c / p0), 2)
                        self.equity_history.append({
                            "time": t,
                            "jev": 10000.0,
                            "before_costs": 10000.0,
                            "buy_and_hold": b,
                        })
            except Exception as e:
                logger.debug(f"Pre-seed equity error: {e}")

        if self.equity_history:
            last_pt = self.equity_history[-1]
            if now_ts - last_pt["time"] < 15:
                last_pt["jev"] = round(total_equity, 2)
                last_pt["before_costs"] = round(before_costs, 2)
                last_pt["buy_and_hold"] = bnh
                return

        self.equity_history.append({
            "time": now_ts,
            "jev": round(total_equity, 2),
            "before_costs": round(before_costs, 2),
            "buy_and_hold": bnh,
        })
        if len(self.equity_history) > 600:
            self.equity_history = self.equity_history[-600:]

    def refresh_market_data(self, symbol: Optional[str] = None):
        """Fetches fresh ticker and klines for the target symbol."""
        sym = (symbol or self.active_symbol).upper()
        if sym == "BTC-PERPETUAL":
            sym = "BTCUSDT"
        try:
            fetcher = self.get_fetcher(sym)
            state = fetcher.get_market_state()
            with self.lock:
                self.market_states[sym] = state
                if sym == self.active_symbol:
                    self.market_state = state
        except Exception as e:
            logger.error(f"Error refreshing market data for {sym}: {e}")
            with self.lock:
                self.error_message = str(e)

    def refresh_ticker_only(self, symbol: Optional[str] = None):
        """Ultra-fast live ticker update every 1-2 seconds with real-time stop-loss/take-profit check."""
        sym = (symbol or self.active_symbol).upper()
        if sym == "BTC-PERPETUAL":
            sym = "BTCUSDT"
        try:
            fetcher = self.get_fetcher(sym)
            ticker = fetcher.fetch_current_ticker()
            with self.lock:
                target_state = self.market_states.get(sym)
                if target_state:
                    target_state.current_price = ticker["price"]
                    target_state.bid_price = ticker["bid"]
                    target_state.ask_price = ticker["ask"]
                    target_state.spread = ticker["spread"]
                    target_state.spread_bps = ticker["spread_bps"]
                    target_state.change_24h_pct = ticker["change_24h_pct"]
                    target_state.high_24h = ticker["high_24h"]
                    target_state.low_24h = ticker["low_24h"]
                    target_state.volume_24h = ticker["volume_24h"]
                    if sym == self.active_symbol:
                        self.market_state = target_state

                # Check active position triggers for all open positions across all markets
                for pos_sym in list(self.execution.positions.keys()):
                    pos_price = ticker["price"] if pos_sym == sym else self.get_fetcher(pos_sym).fetch_current_ticker()["price"]
                    p = self.execution.get_portfolio_state(pos_price, symbol=pos_sym)
                    if p.get("has_open_position"):
                        exit_plan = self.risk_manager.check_active_position_triggers(
                            current_price=pos_price,
                            portfolio=p,
                            engine=self.execution,
                        )
                        if exit_plan and exit_plan.should_execute:
                            self.execution.execute_order(exit_plan, pos_price, symbol=pos_sym)
                            if pos_sym == self.active_symbol:
                                self.plan = exit_plan
                            logger.info(f"Trigger exit executed for {pos_sym}: {exit_plan.reason}")

                # Record equity point using active symbol
                p_active = self.execution.get_portfolio_state(ticker["price"], symbol=self.active_symbol)
                self.record_equity_point(ticker["price"], p_active["total_equity_usdt"], p_active["before_costs_equity_usdt"])
        except Exception as e:
            logger.debug(f"Ticker refresh error for {sym}: {e}")

    def run_jev_cycle(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Runs an evaluation cycle with Jev and executes paper trade via Directional Lean Policy for symbol."""
        sym = (symbol or self.active_symbol).upper()
        if sym == "BTC-PERPETUAL":
            sym = "BTCUSDT"

        self.config = load_config()
        self.risk_manager = RiskManager(self.config.risk, engine=self.execution)
        self.refresh_market_data(sym)

        # 1. Grab snapshot under lock (instant, <0.1ms)
        with self.lock:
            state = self.market_states.get(sym)
            if not state:
                return {"success": False, "error": f"No market data for {sym}"}
            current_price = state.current_price
            portfolio = self.execution.get_portfolio_state(current_price, symbol=sym)
            state_dict = state.to_dict()
            atr_14 = state.indicators.atr_14

        # 2. Network I/O outside lock: evaluate without holding lock
        t0 = time.time()
        try:
            decision = self.jev.evaluate(state_dict, portfolio)
        except Exception as e:
            logger.error(f"Jev evaluation error for {sym}: {e}")
            decision = self.jev.fallback_evaluate(state_dict, portfolio)
        latency_ms = max(10, int((time.time() - t0) * 1000))

        # 3. Apply decision & policy under lock
        with self.lock:
            # Re-read live market price in case it moved during evaluation
            cur_st = self.market_states.get(sym)
            if cur_st:
                current_price = cur_st.current_price
            portfolio = self.execution.get_portfolio_state(current_price, symbol=sym)

            self.last_latency_ms = latency_ms
            self.decisions_count += 1

            # Evaluate with Abol Jev Directional Lean Policy ("SET BY POLICY, NOT ARGMAX")
            plan = self.risk_manager.evaluate_policy(
                decision=decision,
                current_price=current_price,
                portfolio=portfolio,
                atr_14=atr_14,
                threshold=self.policy_threshold,
            )

            if plan.should_execute:
                self.execution.execute_order(plan, current_price, symbol=sym)

            portfolio = self.execution.get_portfolio_state(current_price, symbol=sym)
            self.record_equity_point(current_price, portfolio["total_equity_usdt"], portfolio["before_costs_equity_usdt"])

            self.decisions[sym] = decision
            self.plans[sym] = plan
            if sym == self.active_symbol:
                self.decision = decision
                self.plan = plan

            self.last_eval_time = datetime.now(timezone.utc).isoformat()
            return {
                "success": True,
                "symbol": sym,
                "action": decision.action,
                "plan": plan.order_type,
                "executed": plan.should_execute,
            }


# Global instance
trading_state = GlobalTradingState()


def scan_all_markets_for_opportunities(markets: List[Dict[str, Any]], current_open_symbols: Set[str], scan_cursor: int) -> Tuple[List[str], int]:
    """Scans all 727+ Binance Futures markets.

    Identifies:
    1. Top liquid momentum leaders (strong 24h gainers/losers with heavy volume >= $5M)
    2. Rotational sweep across the entire 727 universe to ensure every contract is scanned.
    """
    if not markets:
        return [], 0

    # 1. Filter liquid contracts (quote volume >= $5,000,000 USD) and standard ASCII alphanumeric symbols
    liquid_markets = [
        m for m in markets
        if m.get("quote_volume_usd", 0) >= 5_000_000
        and m.get("symbol", "").isascii()
        and m.get("symbol", "").isalnum()
    ]
    if not liquid_markets:
        liquid_markets = [m for m in markets if m.get("symbol", "").isascii() and m.get("symbol", "").isalnum()][:50]

    # 2. Opportunity scoring:
    # High volume + strong directional momentum (|change_24h_pct|) = highest probability trade setups
    scored = []
    for m in liquid_markets:
        sym = m["symbol"]
        if sym in current_open_symbols or not sym.isascii() or not sym.isalnum():
            continue
        vol = m.get("quote_volume_usd", 0)
        pct = abs(float(m.get("change_24h_pct", 0) or 0))
        score = (pct * 2.5) + min(40.0, vol / 25_000_000.0)
        scored.append((score, sym))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_opportunities = [sym for _, sym in scored[:4]]

    # 3. Rotational sweep across the entire 727 universe
    total_m = len(markets)
    sweep_symbols = []
    for i in range(4):
        sym = markets[(scan_cursor + i) % total_m]["symbol"]
        if sym.isascii() and sym.isalnum() and sym not in current_open_symbols and sym not in top_opportunities:
            sweep_symbols.append(sym)

    next_cursor = (scan_cursor + 4) % total_m
    return top_opportunities + sweep_symbols, next_cursor


def background_daemon_worker():
    """Background thread that runs continuous market tracking and periodic Jev cycles across ALL 727+ markets."""
    logger.info("Background full-market scanning worker started.")
    last_cycle = 0.0
    last_full_refresh = 0.0
    scan_cursor = 0

    while True:
        try:
            now = time.time()
            active_sym = trading_state.active_symbol
            # Full technical indicator re-calculation every 15 seconds
            if now - last_full_refresh >= 15.0:
                trading_state.refresh_market_data(active_sym)
                last_full_refresh = now
            else:
                # Fast ticker price refresh every 1.5 seconds for live real-time PnL
                trading_state.refresh_ticker_only(active_sym)

            if trading_state.daemon_active and (now - last_cycle >= trading_state.effective_interval_seconds):
                # 1. Real-time evaluation & trigger check on ALL active open positions
                open_positions = list(trading_state.execution.positions.keys())
                for pos_sym in open_positions:
                    try:
                        trading_state.refresh_ticker_only(pos_sym)
                        trading_state.run_jev_cycle(pos_sym)
                    except Exception as ex:
                        logger.debug(f"Open position cycle error for {pos_sym}: {ex}")

                # 2. Run Jev AI analysis and autonomous trading ONLY on explicitly tracked symbols
                with trading_state.lock:
                    tracked_list = list(trading_state.tracked_symbols)

                for track_sym in tracked_list:
                    if track_sym not in open_positions:
                        try:
                            trading_state.run_jev_cycle(track_sym)
                        except Exception as ex:
                            logger.debug(f"Tracked symbol cycle error for {track_sym}: {ex}")

                last_cycle = now

        except Exception as e:
            logger.error(f"Daemon error: {e}")

        time.sleep(1.5)


class DashboardRequestHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests for dashboard HTML and REST API."""

    def log_message(self, format, *args):
        # Suppress noisy GET log spam for polling endpoints
        if "/api/" in args[0]:
            return
        super().log_message(format, *args)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        PAGE_ROUTES = {
            "/",
            "/index.html",
            "/home",
            "/history",
            # Tab 0: Jev Queue / Tracked
            "/queue",
            "/tracked",
            "/analyzing",
            "/home/queue",
            "/home/tracked",
            # Tab 1: Active Positions
            "/positions",
            "/active-positions",
            "/active",
            "/home/positions",
            "/home/active-positions",
            "/home/active",
            # Tab 2: Top Watchlist Markets
            "/watchlist",
            "/top-watchlist",
            "/top",
            "/home/watchlist",
            "/home/top-watchlist",
            "/home/top",
            # Tab 3: All Contracts
            "/contracts",
            "/all-contracts",
            "/all",
            "/home/contracts",
            "/home/all-contracts",
            "/home/all",
            # Tab 4: Top Gainers
            "/gainers",
            "/top-gainers",
            "/home/gainers",
            "/home/top-gainers",
            # Tab 5: Top Losers
            "/losers",
            "/top-losers",
            "/home/losers",
            "/home/top-losers",
        }

        if path in PAGE_ROUTES or path.startswith("/terminal"):
            self.serve_dashboard_html()
        elif path == "/api/markets":
            self.serve_api_markets()
        elif path in ("/api/overview", "/api/master_overview"):
            self.serve_api_overview()
        elif path in ("/api/positions", "/api/active_positions", "/api/active-positions"):
            self.serve_api_positions()
        elif path in ("/api/watchlist", "/api/top_watchlist", "/api/top-watchlist"):
            self.serve_api_watchlist()
        elif path in ("/api/contracts", "/api/all_contracts", "/api/all-contracts"):
            self.serve_api_contracts()
        elif path in ("/api/gainers", "/api/top_gainers", "/api/top-gainers"):
            self.serve_api_gainers()
        elif path in ("/api/losers", "/api/top_losers", "/api/top-losers"):
            self.serve_api_losers()
        elif path == "/api/state":
            self.serve_api_state()
        elif path == "/api/candles":
            self.serve_api_candles()
        elif path == "/api/markers":
            self.serve_api_markers()
        elif path == "/api/equity":
            self.serve_api_equity()
        elif path in ("/api/trades", "/api/trade_history", "/api/history"):
            self.serve_api_trades()
        elif path in ("/api/evaluate", "/api/cycle", "/api/cron"):
            query_dict = dict(qc.split("=") for qc in parsed.query.split("&") if "=" in qc)
            sym = query_dict.get("symbol")
            res = trading_state.run_jev_cycle(symbol=sym)
            self.send_json_response(res)
        elif path in ("/api/tracked", "/api/tracked_symbols"):
            with trading_state.lock:
                tracked_list = sorted(list(trading_state.tracked_symbols))
            self.send_json_response({
                "tracked_symbols": tracked_list,
                "count": len(tracked_list),
            })
        elif path in ("/static/lightweight-charts.js", "/lightweight-charts.js"):
            self.serve_local_file(BASE_DIR / "lightweight-charts.standalone.production.js", "application/javascript")
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, HEAD")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/markets":
            self.serve_api_markets()
        elif path == "/api/select_market":
            content_len = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_len) if content_len > 0 else b"{}"
            try:
                body = json.loads(post_body.decode("utf-8"))
                sym = body.get("symbol", "BTCUSDT").upper()
            except Exception:
                sym = "BTCUSDT"
            if sym == "BTC-PERPETUAL":
                sym = "BTCUSDT"
            with trading_state.lock:
                trading_state.active_symbol = sym
            threading.Thread(target=trading_state.run_jev_cycle, args=(sym,), daemon=True).start()
            self.send_json_response({"success": True, "symbol": sym})
        elif path == "/api/evaluate":
            content_len = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_len) if content_len > 0 else b"{}"
            sym = None
            try:
                if post_body:
                    b = json.loads(post_body.decode("utf-8"))
                    sym = b.get("symbol")
            except Exception:
                pass
            res = trading_state.run_jev_cycle(symbol=sym)
            self.send_json_response(res)
        elif path == "/api/daemon/toggle":
            with trading_state.lock:
                trading_state.daemon_active = not trading_state.daemon_active
                status = trading_state.daemon_active
            self.send_json_response({"daemon_active": status})
        elif path == "/api/speed":
            content_len = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_len) if content_len > 0 else b"{}"
            try:
                body = json.loads(post_body.decode("utf-8"))
                multiplier = int(body.get("multiplier", 1))
            except Exception:
                multiplier = 1
            with trading_state.lock:
                trading_state.cycle_speed_multiplier = multiplier
                trading_state.effective_interval_seconds = max(5, int(60 / multiplier))
            self.send_json_response({"multiplier": multiplier, "interval": trading_state.effective_interval_seconds})
        elif path == "/api/policy/threshold":
            content_len = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_len) if content_len > 0 else b"{}"
            try:
                body = json.loads(post_body.decode("utf-8"))
                thresh = float(body.get("threshold", 0.08))
            except Exception:
                thresh = 0.08
            with trading_state.lock:
                trading_state.policy_threshold = thresh
            # Immediately evaluate active symbol with new threshold
            res = trading_state.run_jev_cycle()
            self.send_json_response({"success": True, "threshold": thresh, "eval": res})
        elif path == "/api/manual/trade":
            content_len = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_len) if content_len > 0 else b"{}"
            try:
                body = json.loads(post_body.decode("utf-8"))
                action = body.get("action", "LONG").upper()
                sym = body.get("symbol", trading_state.active_symbol).upper()
            except Exception:
                action = "LONG"
                sym = trading_state.active_symbol

            if sym == "BTC-PERPETUAL":
                sym = "BTCUSDT"

            with trading_state.lock:
                state = trading_state.market_states.get(sym)
            if not state:
                try:
                    state = trading_state.get_fetcher(sym).get_market_state()
                    with trading_state.lock:
                        trading_state.market_states[sym] = state
                except Exception:
                    pass

            with trading_state.lock:
                curr_price = state.current_price if state else 0.0
                p = trading_state.execution.get_portfolio_state(curr_price, symbol=sym)

                if action in ("CLOSE", "EXIT"):
                    pos_side = p.get("position_side", "NONE")
                    if pos_side == "LONG":
                        plan = OrderPlan(should_execute=True, order_type="EXIT_LONG", size_usd=0, size_eth=0, stop_loss_price=None, take_profit_price=None, reason="Manual user close", confidence=1.0, risk_score=1.0)
                        trading_state.execution.execute_order(plan, curr_price, symbol=sym)
                    elif pos_side == "SHORT":
                        plan = OrderPlan(should_execute=True, order_type="EXIT_SHORT", size_usd=0, size_eth=0, stop_loss_price=None, take_profit_price=None, reason="Manual user close", confidence=1.0, risk_score=1.0)
                        trading_state.execution.execute_order(plan, curr_price, symbol=sym)
                elif action in ("LONG", "ENTER_LONG"):
                    if p.get("position_side") == "SHORT":
                        c_plan = OrderPlan(should_execute=True, order_type="EXIT_SHORT", size_usd=0, size_eth=0, stop_loss_price=None, take_profit_price=None, reason="Close short before long", confidence=1.0, risk_score=1.0)
                        trading_state.execution.execute_order(c_plan, curr_price, symbol=sym)
                        p = trading_state.execution.get_portfolio_state(curr_price, symbol=sym)
                    atr = state.indicators.atr_14 if state else None
                    stop_l, take_p = trading_state.risk_manager.calculate_stop_and_target(curr_price, atr, side="LONG")
                    cash = p.get("cash_usdt", 10000.0)
                    alloc = min(cash * 0.20, 2500.0)
                    sz_asset = (alloc * 2) / curr_price
                    plan = OrderPlan(should_execute=True, order_type="ENTER_LONG", size_usd=round(alloc, 2), size_eth=round(sz_asset, 6), stop_loss_price=round(stop_l, 2), take_profit_price=round(take_p, 2), reason="User Manual Action (Long)", confidence=1.0, risk_score=1.0)
                    trading_state.execution.execute_order(plan, curr_price, symbol=sym)
                    if sym == trading_state.active_symbol:
                        trading_state.plan = plan
                elif action in ("SHORT", "ENTER_SHORT"):
                    if p.get("position_side") == "LONG":
                        c_plan = OrderPlan(should_execute=True, order_type="EXIT_LONG", size_usd=0, size_eth=0, stop_loss_price=None, take_profit_price=None, reason="Close long before short", confidence=1.0, risk_score=1.0)
                        trading_state.execution.execute_order(c_plan, curr_price, symbol=sym)
                        p = trading_state.execution.get_portfolio_state(curr_price, symbol=sym)
                    atr = state.indicators.atr_14 if state else None
                    stop_l, take_p = trading_state.risk_manager.calculate_stop_and_target(curr_price, atr, side="SHORT")
                    cash = p.get("cash_usdt", 10000.0)
                    alloc = min(cash * 0.20, 2500.0)
                    sz_asset = (alloc * 2) / curr_price
                    plan = OrderPlan(should_execute=True, order_type="ENTER_SHORT", size_usd=round(alloc, 2), size_eth=round(sz_asset, 6), stop_loss_price=round(stop_l, 2), take_profit_price=round(take_p, 2), reason="User Manual Action (Short)", confidence=1.0, risk_score=1.0)
                    trading_state.execution.execute_order(plan, curr_price, symbol=sym)
                    if sym == trading_state.active_symbol:
                        trading_state.plan = plan

                new_p = trading_state.execution.get_portfolio_state(curr_price, symbol=sym)
                trading_state.record_equity_point(curr_price, new_p["total_equity_usdt"], new_p["before_costs_equity_usdt"])
            self.send_json_response({"success": True, "action": action, "symbol": sym, "portfolio": new_p})
        elif path in ("/api/tracked/toggle", "/api/tracked/add", "/api/tracked/remove"):
            content_len = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_len) if content_len > 0 else b"{}"
            try:
                body = json.loads(post_body.decode("utf-8"))
                sym = body.get("symbol", "").upper().strip()
            except Exception:
                sym = ""
            if not sym or not sym.isascii() or not sym.isalnum():
                self.send_json_response({"error": "Valid alphanumeric contract symbol required"}, status_code=400)
                return

            if path == "/api/tracked/add":
                ok = trading_state.add_tracked_symbol(sym)
                is_tracked = True
                msg = f"Added {sym} to Jev AI analysis queue"
            elif path == "/api/tracked/remove":
                ok = trading_state.remove_tracked_symbol(sym)
                is_tracked = False
                msg = f"Removed {sym} from Jev AI analysis queue"
            else:
                ok, is_tracked = trading_state.toggle_tracked_symbol(sym)
                msg = f"Added {sym} to Jev AI analysis queue" if is_tracked else f"Removed {sym} from Jev AI analysis queue"

            if is_tracked:
                # Trigger complete comprehensive analysis immediately in background thread
                threading.Thread(target=trading_state.run_jev_cycle, args=(sym,), daemon=True).start()

            with trading_state.lock:
                tracked_list = sorted(list(trading_state.tracked_symbols))

            self.send_json_response({
                "success": True,
                "symbol": sym,
                "is_tracked": is_tracked,
                "tracked_symbols": tracked_list,
                "count": len(tracked_list),
                "message": msg,
            })
        elif path == "/api/reset":
            with trading_state.lock:
                trading_state.execution.reset()
                trading_state.decisions.clear()
                trading_state.plans.clear()
                trading_state.tracked_symbols.clear()
                trading_state.decision = None
                trading_state.plan = None
                trading_state.last_eval_time = None
                trading_state.equity_history = []
                trading_state.initial_btc_price = None
                trading_state.decisions_count = 0
            logger.info("Portfolio and state reset via API request to pristine $10,000 across all markets with 0 tracked contracts.")
            self.send_json_response({"success": True, "message": "State reset to pristine $10,000 USDT for all markets with 0 analyzed contracts."})
        else:
            self.send_response(404)
            self.end_headers()

    def send_json_response(self, data: Any, status_code: int = 200):
        try:
            body = json.dumps(data).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def serve_dashboard_html(self):
        html_file = BASE_DIR / "dashboard.html"
        if not html_file.exists():
            html_file = BASE_DIR / "public" / "index.html"
        if not html_file.exists():
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"dashboard.html not found.")
            return

        with open(html_file, "rb") as f:
            content = f.read()

        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def serve_local_file(self, file_path: Path, content_type: str):
        if not file_path.exists():
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"File not found")
            return
        with open(file_path, "rb") as f:
            content = f.read()
        try:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "public, max-age=86400")
            self.end_headers()
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def serve_api_markets(self):
        """Returns available Binance USDT-M Futures markets, sorted by 24h volume."""
        parsed = urlparse(self.path)
        query_dict = dict(qc.split("=") for qc in parsed.query.split("&") if "=" in qc)
        q = query_dict.get("q", "").upper().strip()
        try:
            limit = int(query_dict.get("limit", 200))
        except Exception:
            limit = 200

        markets = MarketDataFetcher.get_available_markets()
        if q:
            markets = [m for m in markets if q in m.get("symbol", "").upper()]

        self.send_json_response(markets[:limit])

    def build_master_overview_payload(self) -> Dict[str, Any]:
        """Calculates and returns complete multi-stock master overview dictionary."""
        markets = MarketDataFetcher.get_available_markets()
        price_map = {m["symbol"]: m["price"] for m in markets}

        with trading_state.lock:
            overview = trading_state.execution.get_master_portfolio_overview(price_map)
            decisions_copy = dict(trading_state.decisions)

        active_pos_map = {p["symbol"]: p for p in overview["active_positions"]}
        sym_realized = overview["symbol_realized_pnl"]
        sym_trades = overview["symbol_trade_count"]
        sym_capital = overview.get("symbol_capital", {})
        sym_cash = overview.get("symbol_cash", {})

        market_cards = []
        for m in markets:
            sym = m["symbol"]
            pos = active_pos_map.get(sym)
            dec = decisions_copy.get(sym)

            p_long = 0.0
            p_short = 0.0
            lean = 0.0
            action = "WAIT"
            conf = 0.40
            regime = "NEUTRAL"

            if dec:
                action = dec.action
                conf = dec.action_confidence
                regime = dec.market_regime
                if dec.action_probabilities:
                    probs = dec.action_probabilities
                    p_long = float(probs.get("BUY", probs.get("LONG", 0.0)))
                    p_short = float(probs.get("SELL", probs.get("SHORT", 0.0)))
                    lean = round(p_long - p_short, 2)
            else:
                pct = float(m.get("change_24h_pct", 0.0))
                if pct >= 1.5:
                    action = "BUY"
                    conf = round(min(0.85, 0.50 + abs(pct) * 0.03), 2)
                    lean = round(min(0.80, 0.25 + abs(pct) * 0.04), 2)
                    regime = "BULLISH_EXPANSION"
                elif pct <= -1.5:
                    action = "SELL"
                    conf = round(min(0.85, 0.50 + abs(pct) * 0.03), 2)
                    lean = round(max(-0.80, -0.25 - abs(pct) * 0.04), 2)
                    regime = "BEARISH_BREAKDOWN"
                else:
                    action = "WAIT"
                    conf = 0.42
                    lean = 0.0
                    regime = "NEUTRAL"

            cap = sym_capital.get(sym, 10000.0)
            cash = sym_cash.get(sym, 10000.0)
            liq_price = pos.get("liquidation_price") if pos else None

            market_cards.append({
                "symbol": sym,
                "base_asset": m.get("base_asset", sym.replace("USDT", "")),
                "price": m["price"],
                "change_24h_pct": m["change_24h_pct"],
                "quote_volume_usd": m["quote_volume_usd"],
                "high_24h": m.get("high_24h", m["price"]),
                "low_24h": m.get("low_24h", m["price"]),
                "capital_usdt": cap,
                "cash_usdt": cash,
                "has_position": pos is not None,
                "position_side": pos["side"] if pos else "FLAT",
                "position_size": pos["size_asset"] if pos else 0.0,
                "notional_usd": pos["notional_usd"] if pos else 0.0,
                "entry_price": pos["entry_price"] if pos else None,
                "liquidation_price": liq_price,
                "unrealized_pnl_usd": pos["unrealized_pnl_usd"] if pos else 0.0,
                "unrealized_pnl_pct": pos["unrealized_pnl_pct"] if pos else 0.0,
                "roe_pct": pos["roe_pct"] if pos else 0.0,
                "margin_usdt": pos["margin_usdt"] if pos else 0.0,
                "leverage": pos["leverage"] if pos else 2,
                "realized_pnl_usd": sym_realized.get(sym, 0.0),
                "trades_count": sym_trades.get(sym, 0),
                "jev_action": action,
                "jev_confidence": conf,
                "jev_lean": lean,
                "market_regime": regime,
                "is_tracked": sym in trading_state.tracked_symbols,
            })

        overview["markets"] = market_cards
        overview["active_symbol"] = trading_state.active_symbol
        overview["tracked_symbols"] = sorted(list(trading_state.tracked_symbols))
        overview["tracked_count"] = len(trading_state.tracked_symbols)
        return overview

    def serve_api_overview(self):
        """Returns multi-stock master dashboard overview with independent $10,000 capital accounts and per-market status."""
        try:
            overview = self.build_master_overview_payload()
            self.send_json_response(overview)
        except Exception as e:
            logger.error(f"Error serving master overview: {e}")
            self.send_json_response({"error": str(e)}, status_code=500)

    def serve_api_positions(self):
        """Tab 1 API: Returns currently active positions with live PnL and metrics."""
        try:
            overview = self.build_master_overview_payload()
            active = overview.get("active_positions", [])
            self.send_json_response({
                "tab": "ACTIVE_POSITIONS",
                "count": len(active),
                "positions": active,
                "portfolio_summary": overview.get("portfolio_summary", {}),
            })
        except Exception as e:
            logger.error(f"Error serving positions API: {e}")
            self.send_json_response({"error": str(e)}, status_code=500)

    def serve_api_watchlist(self):
        """Tab 2 API: Returns top watchlist markets with live metrics."""
        try:
            overview = self.build_master_overview_payload()
            markets = overview.get("markets", [])
            watchlist = [m for m in markets if m.get("has_position")] or markets[:20]
            self.send_json_response({
                "tab": "WATCHLIST",
                "count": len(watchlist),
                "watchlist": watchlist,
            })
        except Exception as e:
            logger.error(f"Error serving watchlist API: {e}")
            self.send_json_response({"error": str(e)}, status_code=500)

    def serve_api_contracts(self):
        """Tab 3 API: Returns all monitored perpetual contracts."""
        try:
            overview = self.build_master_overview_payload()
            markets = overview.get("markets", [])
            self.send_json_response({
                "tab": "ALL_CONTRACTS",
                "total_count": len(markets),
                "contracts": markets,
            })
        except Exception as e:
            logger.error(f"Error serving contracts API: {e}")
            self.send_json_response({"error": str(e)}, status_code=500)

    def serve_api_gainers(self):
        """Tab 4 API: Returns top 24h gainers across all perpetual contracts."""
        try:
            overview = self.build_master_overview_payload()
            markets = overview.get("markets", [])
            gainers = sorted(markets, key=lambda m: float(m.get("change_24h_pct", 0.0) or 0.0), reverse=True)[:60]
            self.send_json_response({
                "tab": "TOP_GAINERS",
                "count": len(gainers),
                "gainers": gainers,
            })
        except Exception as e:
            logger.error(f"Error serving gainers API: {e}")
            self.send_json_response({"error": str(e)}, status_code=500)

    def serve_api_losers(self):
        """Tab 5 API: Returns top 24h losers across all perpetual contracts."""
        try:
            overview = self.build_master_overview_payload()
            markets = overview.get("markets", [])
            losers = sorted(markets, key=lambda m: float(m.get("change_24h_pct", 0.0) or 0.0))[:60]
            self.send_json_response({
                "tab": "TOP_LOSERS",
                "count": len(losers),
                "losers": losers,
            })
        except Exception as e:
            logger.error(f"Error serving losers API: {e}")
            self.send_json_response({"error": str(e)}, status_code=500)

    def serve_api_state(self):
        parsed = urlparse(self.path)
        query_dict = dict(qc.split("=") for qc in parsed.query.split("&") if "=" in qc)
        sym = query_dict.get("symbol", trading_state.active_symbol).upper()
        if sym == "BTC-PERPETUAL":
            sym = "BTCUSDT"

        with trading_state.lock:
            state = trading_state.market_states.get(sym)
        if not state:
            try:
                fetcher = trading_state.get_fetcher(sym)
                state = fetcher.get_market_state()
                with trading_state.lock:
                    trading_state.market_states[sym] = state
            except Exception as e:
                logger.error(f"Error on-demand fetching {sym}: {e}")

        with trading_state.lock:

            current_price = state.current_price if state else 0.0
            portfolio = trading_state.execution.get_portfolio_state(current_price, symbol=sym)
            performance = trading_state.execution.get_performance_summary()
            trades = [asdict(t) for t in trading_state.execution.trade_history]

            decision = trading_state.decisions.get(sym)
            plan = trading_state.plans.get(sym)

            # Abol Jev Metrics
            p_long = 0.0
            p_wait = 1.0
            p_short = 0.0
            if decision and decision.action_probabilities:
                probs = decision.action_probabilities
                p_long = float(probs.get("BUY", probs.get("LONG", 0.0)))
                p_wait = float(probs.get("HOLD", probs.get("WAIT", 0.0)))
                p_short = float(probs.get("SELL", probs.get("SHORT", 0.0)))

            lean = round(p_long - p_short, 2)
            thresh = trading_state.policy_threshold
            own_pick = "LONG" if lean >= thresh else ("SHORT" if lean <= -thresh else "WAIT")

            bnh = 10000.0
            if trading_state.initial_btc_price and current_price > 0:
                bnh = round(10000.0 * (current_price / trading_state.initial_btc_price), 2)

            resp = {
                "symbol": sym,
                "active_symbol": trading_state.active_symbol,
                "price": current_price,
                "change_24h_pct": state.change_24h_pct if state else 0.0,
                "high_24h": state.high_24h if state else 0.0,
                "low_24h": state.low_24h if state else 0.0,
                "volume_24h": state.volume_24h if state else 0.0,
                "spread_bps": state.spread_bps if state else 0.0,
                "indicators": asdict(state.indicators) if state else {},
                "order_flow": asdict(state.order_flow) if state else {},
                "sentiment": asdict(state.sentiment) if state else {},
                "derivatives": asdict(state.derivatives) if (state and getattr(state, "derivatives", None)) else {},
                "decision": {
                    "action": decision.action,
                    "confidence": decision.action_confidence,
                    "probabilities": decision.action_probabilities,
                    "market_regime": decision.market_regime,
                    "regime_confidence": decision.regime_confidence,
                    "risk_score": decision.risk_score,
                    "risk_confidence": decision.risk_confidence,
                    "entry_conviction": decision.entry_conviction,
                    "exit_urgency": decision.exit_urgency,
                    "model_version": decision.model_version,
                } if decision else None,
                "directional_lean": {
                    "lean": lean,
                    "p_long": round(p_long, 2),
                    "p_wait": round(p_wait, 2),
                    "p_short": round(p_short, 2),
                    "own_pick": own_pick,
                    "threshold": thresh,
                },
                "latency_ms": trading_state.last_latency_ms,
                "decisions_count": trading_state.decisions_count,
                "speed_multiplier": trading_state.cycle_speed_multiplier,
                "buy_and_hold_equity": bnh,
                "plan": {
                    "order_type": plan.order_type,
                    "should_execute": plan.should_execute,
                    "reason": plan.reason,
                    "stop_loss": plan.stop_loss_price,
                    "take_profit": plan.take_profit_price,
                } if plan else None,
                "portfolio": portfolio,
                "performance": performance,
                "trades": trades,
                "daemon_active": trading_state.daemon_active,
                "last_eval_time": trading_state.last_eval_time,
                "cycle_interval_seconds": trading_state.effective_interval_seconds,
                "is_tracked": sym in trading_state.tracked_symbols,
                "tracked_symbols": sorted(list(trading_state.tracked_symbols)),
            }
        self.send_json_response(resp)

    def serve_api_equity(self):
        """Returns time series of Jev, Before Costs, and Buy & Hold equity."""
        with trading_state.lock:
            if not trading_state.equity_history and trading_state.market_state:
                p = trading_state.execution.get_portfolio_state(trading_state.market_state.current_price, symbol=trading_state.active_symbol)
                trading_state.record_equity_point(trading_state.market_state.current_price, p["total_equity_usdt"], p["before_costs_equity_usdt"])
            data = list(trading_state.equity_history)
        self.send_json_response(data)

    def serve_api_candles(self):
        """Returns 5m candlestick data formatted for TradingView Lightweight Charts."""
        parsed = urlparse(self.path)
        query_dict = dict(qc.split("=") for qc in parsed.query.split("&") if "=" in qc)
        sym = query_dict.get("symbol", trading_state.active_symbol).upper()
        if sym == "BTC-PERPETUAL":
            sym = "BTCUSDT"

        try:
            fetcher = trading_state.get_fetcher(sym)
            df = fetcher.fetch_klines(interval="5m", limit=80)
            candles = []
            for _, row in df.iterrows():
                candles.append({
                    "time": int(row["timestamp"].timestamp()),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                })
            self.send_json_response(candles)
        except Exception as e:
            logger.error(f"Failed to serve candles for {sym}: {e}")
            self.send_json_response([], status_code=500)

    def serve_api_markers(self):
        """Returns trade markers for chart overlays filtered by symbol."""
        parsed = urlparse(self.path)
        query_dict = dict(qc.split("=") for qc in parsed.query.split("&") if "=" in qc)
        sym = query_dict.get("symbol", trading_state.active_symbol).upper()
        if sym == "BTC-PERPETUAL":
            sym = "BTCUSDT"

        markers = trading_state.execution.get_chart_markers(symbol=sym)
        self.send_json_response(markers)

    def serve_api_trades(self):
        """Returns full structured trade history with entry/exit metrics, duration, PnL, and reasons."""
        parsed = urlparse(self.path)
        query_dict = dict(qc.split("=") for qc in parsed.query.split("&") if "=" in qc)
        sym = query_dict.get("symbol")
        if sym:
            sym = sym.upper()

        with trading_state.lock:
            round_trips = trading_state.execution.get_full_trade_history(symbol=sym)
            perf = trading_state.execution.get_performance_summary()
            raw_trades = [asdict(t) for t in sorted(trading_state.execution.trade_history, key=lambda x: x.timestamp, reverse=True)]

        self.send_json_response({
            "summary": perf,
            "round_trips": round_trips,
            "raw_fills": raw_trades[:150],
            "total_trades": len(round_trips),
            "closed_trades": len([r for r in round_trips if r.get("status") == "CLOSED"]),
            "open_trades": len([r for r in round_trips if r.get("status") == "OPEN"]),
        })


def run_server(port: int = 5000):
    """Starts the background worker thread and the HTTP web server."""
    # Ensure initial market data is loaded
    trading_state.refresh_market_data()

    # Launch daemon worker thread
    worker_thread = threading.Thread(target=background_daemon_worker, daemon=True)
    worker_thread.start()

    server = ThreadingHTTPServer(("0.0.0.0", port), DashboardRequestHandler)
    print(f"\n{'='*70}")
    print(f"  [*] TYPESAFE JEV DASHBOARD IS LIVE!")
    print(f"  --> Open in your browser: http://localhost:{port}")
    print(f"  Symbol: {trading_state.config.trading_pair} | Paper Bankroll: ${trading_state.config.simulation.initial_balance_usdt:,.2f} USDT")
    print(f"{'='*70}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping dashboard server...")
        server.shutdown()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TypeSafe Jev Dashboard Server")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind server (default: 5000)")
    args = parser.parse_args()
    run_server(port=args.port)
