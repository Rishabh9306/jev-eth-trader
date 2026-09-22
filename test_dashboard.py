"""Test suite for the Dashboard server and REST API endpoints."""

from __future__ import annotations

import json
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
import requests

from server import DashboardRequestHandler, trading_state


class TestDashboardServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = 5099
        cls.server = ThreadingHTTPServer(("127.0.0.1", cls.port), DashboardRequestHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        # Initialize market data
        trading_state.refresh_market_data()
        time.sleep(1)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_01_serve_html(self):
        """Test GET / serves dashboard.html."""
        res = requests.get(f"http://127.0.0.1:{self.port}/", timeout=5)
        self.assertEqual(res.status_code, 200)
        self.assertIn("TradingView", res.text)
        self.assertIn("Jev", res.text)
        print("\n[DASHBOARD TEST] 1. GET / successfully served dashboard.html (HTTP 200)")

    def test_02_api_state(self):
        """Test GET /api/state returns complete state and $10,000 paper capital."""
        res = requests.get(f"http://127.0.0.1:{self.port}/api/state", timeout=5)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("price", data)
        self.assertGreater(data["price"], 0.0)
        self.assertIn("portfolio", data)
        self.assertIn("total_equity_usdt", data["portfolio"])
        equity = data["portfolio"]["total_equity_usdt"]
        self.assertGreater(equity, 5000.0, "Equity should be above $5000 (not blown up)")
        self.assertLess(equity, 20000.0, "Equity should be below $20000 (not unreasonably large)")
        self.assertIn("order_flow", data)
        self.assertIn("sentiment", data)
        print(f"       Price: ${data['price']:,.2f} | Paper Equity: ${data['portfolio']['total_equity_usdt']:,.2f}")
        print(f"       Order Flow Ratio: {data['order_flow']['taker_buy_sell_ratio']} | Fear & Greed: {data['sentiment']['fear_and_greed_score']}")
        print("[DASHBOARD TEST] 2. GET /api/state successfully validated")

    def test_03_api_candles(self):
        """Test GET /api/candles returns TradingView formatted bars."""
        res = requests.get(f"http://127.0.0.1:{self.port}/api/candles", timeout=8)
        self.assertEqual(res.status_code, 200)
        candles = res.json()
        self.assertIsInstance(candles, list)
        self.assertGreater(len(candles), 0)
        c0 = candles[0]
        self.assertIn("time", c0)
        self.assertIn("open", c0)
        self.assertIn("close", c0)
        self.assertIsInstance(c0["time"], int)
        print(f"       Loaded {len(candles)} 5m candlestick bars for TradingView chart")
        print("[DASHBOARD TEST] 3. GET /api/candles successfully validated")

    def test_04_api_markers(self):
        """Test GET /api/markers returns trade markers list."""
        res = requests.get(f"http://127.0.0.1:{self.port}/api/markers", timeout=5)
        self.assertEqual(res.status_code, 200)
        markers = res.json()
        self.assertIsInstance(markers, list)
        print(f"       Retrieved {len(markers)} trade markers")
        print("[DASHBOARD TEST] 4. GET /api/markers successfully validated")

    def test_05_master_tab_page_routes(self):
        """Test GET page routes for all 5 master tabs return HTTP 200."""
        routes = [
            "/positions",
            "/active-positions",
            "/watchlist",
            "/top-watchlist",
            "/contracts",
            "/all-contracts",
            "/gainers",
            "/top-gainers",
            "/losers",
            "/top-losers",
            "/home",
            "/history",
        ]
        for r in routes:
            res = requests.get(f"http://127.0.0.1:{self.port}{r}", timeout=5)
            self.assertEqual(res.status_code, 200, f"Route {r} should return HTTP 200")
            self.assertIn("Jev", res.text)
        print(f"\n[DASHBOARD TEST] 5. Validated {len(routes)} page routes successfully (HTTP 200)")

    def test_06_master_tab_api_endpoints(self):
        """Test GET JSON API endpoints for all 5 master tabs."""
        # Tab 1: /api/positions
        res = requests.get(f"http://127.0.0.1:{self.port}/api/positions", timeout=5)
        self.assertEqual(res.status_code, 200)
        d = res.json()
        self.assertEqual(d.get("tab"), "ACTIVE_POSITIONS")
        self.assertIn("positions", d)

        # Tab 2: /api/watchlist
        res = requests.get(f"http://127.0.0.1:{self.port}/api/watchlist", timeout=5)
        self.assertEqual(res.status_code, 200)
        d = res.json()
        self.assertIn(d.get("tab"), ("WATCHLIST", "TOP_WATCHLIST"))
        self.assertIn("watchlist", d)

        # Tab 3: /api/contracts
        res = requests.get(f"http://127.0.0.1:{self.port}/api/contracts", timeout=5)
        self.assertEqual(res.status_code, 200)
        d = res.json()
        self.assertEqual(d.get("tab"), "ALL_CONTRACTS")
        self.assertIn("contracts", d)

        # Tab 4: /api/gainers
        res = requests.get(f"http://127.0.0.1:{self.port}/api/gainers", timeout=5)
        self.assertEqual(res.status_code, 200)
        d = res.json()
        self.assertEqual(d.get("tab"), "TOP_GAINERS")
        self.assertIn("gainers", d)

        # Tab 5: /api/losers
        res = requests.get(f"http://127.0.0.1:{self.port}/api/losers", timeout=5)
        self.assertEqual(res.status_code, 200)
        d = res.json()
        self.assertEqual(d.get("tab"), "TOP_LOSERS")
        self.assertIn("losers", d)

        print("[DASHBOARD TEST] 6. Validated all 5 master tab API endpoints successfully (JSON 200)")

    def test_07_vercel_handler(self):
        """Test that api.index.handler executes without errors for Vercel serverless."""
        import io
        from api.index import handler

        class MockVercelHandler(handler):
            def __init__(self, path):
                self.path = path
                self.wfile = io.BytesIO()
                self.headers = {}
                self._headers_buffer = []

            def send_response(self, code, message=None):
                self.status_code = code

            def send_header(self, k, v):
                pass

            def end_headers(self):
                pass

        # Test GET /api/history via handler
        h1 = MockVercelHandler("/api/history")
        h1.do_GET()
        self.assertEqual(h1.status_code, 200)
        self.assertGreater(len(h1.wfile.getvalue()), 0)

        # Test GET /api/evaluate via handler (Vercel Cron)
        h2 = MockVercelHandler("/api/evaluate")
        h2.do_GET()
        self.assertEqual(h2.status_code, 200)
        self.assertGreater(len(h2.wfile.getvalue()), 0)

        # Test OPTIONS /api/overview
        h3 = MockVercelHandler("/api/overview")
        h3.do_OPTIONS()
        self.assertEqual(h3.status_code, 200)

        print("[DASHBOARD TEST] 7. Validated api.index.handler for Vercel serverless successfully")


if __name__ == "__main__":
    unittest.main()
