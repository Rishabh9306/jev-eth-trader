"""Execution engine for BTC Perpetual Futures paper trading simulation.

Supports:
- Two-way directional trading (LONG and SHORT)
- Configurable leverage and isolated margin accounting
- Liquidation price modeling and buffer enforcement
- Realistic slippage (5 bps) and taker exchange fees (5 bps)
- Comprehensive trade history and performance metrics with JSON persistence
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from config import SimulationConfig
from risk_manager import OrderPlan

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    trade_id: str
    symbol: str
    side: str  # "LONG", "SHORT", "EXIT_LONG", "EXIT_SHORT"
    timestamp: str
    price: float
    effective_price: float  # Price after slippage
    amount_btc: float
    amount_eth: float  # Alias for backward compatibility
    notional_usd: float
    margin_usd: float
    leverage: int
    fee_usd: float
    realized_pnl_usd: float
    realized_pnl_pct: float  # Return on Equity (ROE%)
    reason: str
    confidence: float


@dataclass
class AssetPosition:
    symbol: str
    side: str  # "LONG", "SHORT"
    size_asset: float
    entry_price: float
    margin_usdt: float
    leverage: int
    entry_timestamp: str
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    liquidation_price: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AssetPosition:
        return cls(
            symbol=data.get("symbol", "BTCUSDT"),
            side=data.get("side", "NONE"),
            size_asset=float(data.get("size_asset", data.get("position_btc", data.get("amount_btc", 0.0)))),
            entry_price=float(data.get("entry_price", 0.0)),
            margin_usdt=float(data.get("margin_usdt", 0.0)),
            leverage=int(data.get("leverage", 2)),
            entry_timestamp=data.get("entry_timestamp", ""),
            stop_loss_price=data.get("stop_loss_price"),
            take_profit_price=data.get("take_profit_price"),
            liquidation_price=data.get("liquidation_price"),
        )


@dataclass
class MarketAccount:
    """Isolated margin and balance account for a single perpetual futures market."""
    symbol: str
    initial_balance: float = 10000.0
    cash_usdt: float = 10000.0
    position: Optional[AssetPosition] = None
    trade_history: List[TradeRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "initial_balance": self.initial_balance,
            "cash_usdt": round(self.cash_usdt, 2),
            "position": self.position.to_dict() if self.position else None,
            "trade_history": [asdict(t) for t in self.trade_history],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MarketAccount:
        pos_data = data.get("position")
        pos = AssetPosition.from_dict(pos_data) if pos_data else None
        trades = []
        for t in data.get("trade_history", []):
            amt = float(t.get("amount_btc", t.get("amount_eth", 0.0)))
            trades.append(
                TradeRecord(
                    trade_id=t.get("trade_id", ""),
                    symbol=t.get("symbol", data.get("symbol", "BTCUSDT")),
                    side=t.get("side", "BUY"),
                    timestamp=t.get("timestamp", ""),
                    price=float(t.get("price", 0.0)),
                    effective_price=float(t.get("effective_price", 0.0)),
                    amount_btc=amt,
                    amount_eth=amt,
                    notional_usd=float(t.get("notional_usd", t.get("cost_usd", 0.0))),
                    margin_usd=float(t.get("margin_usd", t.get("cost_usd", 0.0))),
                    leverage=int(t.get("leverage", 2)),
                    fee_usd=float(t.get("fee_usd", 0.0)),
                    realized_pnl_usd=float(t.get("realized_pnl_usd", 0.0)),
                    realized_pnl_pct=float(t.get("realized_pnl_pct", 0.0)),
                    reason=t.get("reason", ""),
                    confidence=float(t.get("confidence", 0.0)),
                )
            )
        return cls(
            symbol=data.get("symbol", "BTCUSDT"),
            initial_balance=float(data.get("initial_balance", 10000.0)),
            cash_usdt=float(data.get("cash_usdt", 10000.0)),
            position=pos,
            trade_history=trades,
        )


class BaseExecutionEngine(ABC):
    """Abstract interface for trade execution engines."""

    @abstractmethod
    def get_portfolio_state(self, current_price: float, symbol: Optional[str] = None) -> Dict[str, Any]:
        pass

    @abstractmethod
    def execute_order(self, plan: OrderPlan, current_price: float, symbol: Optional[str] = None) -> Optional[TradeRecord]:
        pass


class PaperTradingEngine(BaseExecutionEngine):
    """Multi-asset Binance Perpetual Futures simulation engine with isolated per-market $10,000 accounts."""

    def __init__(self, config: SimulationConfig, state_file: Optional[Path] = None):
        self.config = config
        if state_file:
            self.state_file = state_file
        elif os.environ.get("VERCEL"):
            self.state_file = Path("/tmp/trade_history.json")
        else:
            self.state_file = Path("trade_history.json")

        self.initial_balance = config.initial_balance_usdt
        self.cash_usdt = config.initial_balance_usdt
        self.leverage: int = getattr(config, "leverage", 2)
        self.maintenance_margin_pct: float = getattr(config, "maintenance_margin_pct", 0.005)
        self.symbol: str = "BTCUSDT"

        # Isolated $10,000 margin accounts per market
        self.market_accounts: Dict[str, MarketAccount] = {}

        # Multi-asset positions map: symbol -> AssetPosition
        self.positions: Dict[str, AssetPosition] = {}

        # Legacy backward-compatibility attributes (mirrors primary active symbol)
        self.position_side: str = "NONE"  # "LONG", "SHORT", "NONE"
        self.position_btc: float = 0.0
        self.position_eth: float = 0.0  # Alias
        self.margin_usdt: float = 0.0
        self.entry_price: float = 0.0
        self.entry_timestamp: Optional[str] = None
        self.stop_loss_price: Optional[float] = None
        self.take_profit_price: Optional[float] = None
        self.liquidation_price: Optional[float] = None
        self.trade_history: List[TradeRecord] = []
        self.tracked_symbols: List[str] = []

        self._load_state()

    def get_account(self, symbol: Optional[str] = None) -> MarketAccount:
        """Gets or lazily initializes an isolated $10,000.00 USDT account for a specific market."""
        sym = (symbol or self.symbol or "BTCUSDT").upper()
        if sym == "BTC-PERPETUAL":
            sym = "BTCUSDT"
        if sym not in self.market_accounts:
            self.market_accounts[sym] = MarketAccount(
                symbol=sym,
                initial_balance=float(self.initial_balance),
                cash_usdt=float(self.initial_balance),
                position=None,
                trade_history=[],
            )
        return self.market_accounts[sym]

    def _sync_legacy_attributes(self, target_sym: Optional[str] = None):
        """Synchronizes legacy single-position attributes with the active symbol's position."""
        sym = target_sym or self.symbol or "BTCUSDT"
        # Check direct or normalized match
        pos = self.positions.get(sym)
        if not pos and sym in ("BTCUSDT", "BTC-PERPETUAL"):
            pos = self.positions.get("BTCUSDT") or self.positions.get("BTC-PERPETUAL")
        if not pos and self.positions:
            # Fallback to first active position if available
            pos = next(iter(self.positions.values()), None)

        if pos and pos.side in ("LONG", "SHORT") and pos.size_asset > 0:
            self.position_side = pos.side
            self.position_btc = pos.size_asset
            self.position_eth = pos.size_asset
            self.margin_usdt = pos.margin_usdt
            self.entry_price = pos.entry_price
            self.entry_timestamp = pos.entry_timestamp
            self.stop_loss_price = pos.stop_loss_price
            self.take_profit_price = pos.take_profit_price
            self.liquidation_price = pos.liquidation_price
            self.leverage = pos.leverage
        else:
            self.position_side = "NONE"
            self.position_btc = 0.0
            self.position_eth = 0.0
            self.margin_usdt = 0.0
            self.entry_price = 0.0
            self.entry_timestamp = None
            self.stop_loss_price = None
            self.take_profit_price = None
            self.liquidation_price = None

    def _apply_state_dict(self, data: Dict[str, Any]):
        """Applies state payload to instance."""
        self.initial_balance = float(data.get("initial_balance", 10000.0))
        self.cash_usdt = float(data.get("cash_usdt", self.initial_balance))
        self.leverage = int(data.get("leverage", self.leverage))
        self.symbol = data.get("symbol", self.symbol)

        raw_accounts = data.get("market_accounts", {})
        self.market_accounts = {}
        if isinstance(raw_accounts, dict) and raw_accounts:
            for sym, acc_data in raw_accounts.items():
                if isinstance(acc_data, dict) and sym.isascii() and sym.isalnum():
                    acc = MarketAccount.from_dict(acc_data)
                    self.market_accounts[sym] = acc
                    if acc.position:
                        self.positions[sym] = acc.position
                    for t in acc.trade_history:
                        if t not in self.trade_history and t.symbol.isascii():
                            self.trade_history.append(t)

        # 1. Load multi-asset positions if present
        raw_positions = data.get("positions", {})
        if isinstance(raw_positions, dict):
            for sym, pos_data in raw_positions.items():
                if isinstance(pos_data, dict) and pos_data.get("side") in ("LONG", "SHORT") and sym.isascii() and sym.isalnum():
                    pos = AssetPosition.from_dict(pos_data)
                    self.positions[sym] = pos
                    acc = self.get_account(sym)
                    acc.position = pos

        # 2. Legacy fallback if positions map was empty but single position was active
        legacy_side = data.get("position_side", "NONE")
        legacy_btc = float(data.get("position_btc", data.get("position_eth", 0.0)))
        if not self.positions and legacy_side in ("LONG", "SHORT") and legacy_btc > 0:
            sym = data.get("symbol", "BTCUSDT")
            if sym == "BTC-PERPETUAL":
                sym = "BTCUSDT"
            pos = AssetPosition(
                symbol=sym,
                side=legacy_side,
                size_asset=legacy_btc,
                entry_price=float(data.get("entry_price", 0.0)),
                margin_usdt=float(data.get("margin_usdt", 0.0)),
                leverage=int(data.get("leverage", self.leverage)),
                entry_timestamp=data.get("entry_timestamp", ""),
                stop_loss_price=data.get("stop_loss_price"),
                take_profit_price=data.get("take_profit_price"),
                liquidation_price=data.get("liquidation_price"),
            )
            self.positions[sym] = pos
            acc = self.get_account(sym)
            acc.position = pos

        # 3. Load trade history
        raw_trades = data.get("trade_history", [])
        self.trade_history = []
        for t in raw_trades:
            amt = float(t.get("amount_btc", t.get("amount_eth", 0.0)))
            t_rec = TradeRecord(
                trade_id=t.get("trade_id", ""),
                symbol=t.get("symbol", self.symbol),
                side=t.get("side", "BUY"),
                timestamp=t.get("timestamp", ""),
                price=float(t.get("price", 0.0)),
                effective_price=float(t.get("effective_price", 0.0)),
                amount_btc=amt,
                amount_eth=amt,
                notional_usd=float(t.get("notional_usd", t.get("cost_usd", 0.0))),
                margin_usd=float(t.get("margin_usd", t.get("cost_usd", 0.0))),
                leverage=int(t.get("leverage", self.leverage)),
                fee_usd=float(t.get("fee_usd", 0.0)),
                realized_pnl_usd=float(t.get("realized_pnl_usd", 0.0)),
                realized_pnl_pct=float(t.get("realized_pnl_pct", 0.0)),
                reason=t.get("reason", ""),
                confidence=float(t.get("confidence", 0.0)),
            )
            self.trade_history.append(t_rec)
            acc = self.get_account(t_rec.symbol)
            if t_rec not in acc.trade_history:
                acc.trade_history.append(t_rec)

        raw_tracked = data.get("tracked_symbols", [])
        if isinstance(raw_tracked, list):
            self.tracked_symbols = [s for s in raw_tracked if isinstance(s, str) and s.isascii() and s.isalnum()]
        else:
            self.tracked_symbols = []

        self._sync_legacy_attributes()

    def _load_state(self):
        """Loads state from cloud KV or local JSON file if exists."""
        import os
        import requests
        redis_url = os.environ.get("UPSTASH_REDIS_REST_URL") or os.environ.get("KV_REST_API_URL")
        redis_token = os.environ.get("UPSTASH_REDIS_REST_TOKEN") or os.environ.get("KV_REST_API_TOKEN")
        if redis_url and redis_token:
            try:
                resp = requests.get(f"{redis_url}/get/jev_trade_history", headers={"Authorization": f"Bearer {redis_token}"}, timeout=2.5)
                if resp.status_code == 200:
                    val = resp.json().get("result")
                    if val:
                        data = json.loads(val) if isinstance(val, str) else val
                        self._apply_state_dict(data)
                        return
            except Exception as e:
                logger.debug(f"KV load skipped: {e}")

        # Fall back to local file
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._apply_state_dict(data)
                logger.info(
                    f"Loaded multi-asset paper state: ${self.cash_usdt:.2f} USDT, "
                    f"Active positions: {list(self.positions.keys())}"
                )
            except Exception as e:
                logger.warning(f"Could not load state file {self.state_file}: {e}. Starting fresh.")

    def _save_state(self):
        """Persists current futures state and trade history to JSON and Cloud KV."""
        try:
            self._sync_legacy_attributes()
            active_acc = self.get_account(self.symbol)
            data = {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "symbol": self.symbol,
                "initial_balance": self.initial_balance,
                "cash_usdt": round(active_acc.cash_usdt, 2),
                "market_accounts": {sym: acc.to_dict() for sym, acc in self.market_accounts.items()},
                "positions": {sym: pos.to_dict() for sym, pos in self.positions.items()},
                "position_side": self.position_side,
                "position_btc": round(self.position_btc, 6),
                "position_eth": round(self.position_btc, 6),
                "margin_usdt": round(self.margin_usdt, 2),
                "leverage": self.leverage,
                "entry_price": round(self.entry_price, 2),
                "entry_timestamp": self.entry_timestamp,
                "stop_loss_price": self.stop_loss_price,
                "take_profit_price": self.take_profit_price,
                "liquidation_price": self.liquidation_price,
                "trade_history": [asdict(t) for t in self.trade_history],
                "performance_summary": self.get_performance_summary(),
                "tracked_symbols": getattr(self, "tracked_symbols", []),
            }
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            # Cloud KV sync if available
            import os
            import requests
            redis_url = os.environ.get("UPSTASH_REDIS_REST_URL") or os.environ.get("KV_REST_API_URL")
            redis_token = os.environ.get("UPSTASH_REDIS_REST_TOKEN") or os.environ.get("KV_REST_API_TOKEN")
            if redis_url and redis_token:
                try:
                    requests.post(
                        f"{redis_url}/set/jev_trade_history",
                        headers={"Authorization": f"Bearer {redis_token}"},
                        json=json.dumps(data),
                        timeout=2.0
                    )
                except Exception as ex:
                    logger.debug(f"Redis save failed: {ex}")
        except Exception as e:
            logger.error(f"Failed to save state to {self.state_file}: {e}")

    def update_position_stops(self, symbol: str, stop_loss_price: Optional[float] = None, take_profit_price: Optional[float] = None):
        """Dynamically updates stop-loss or take-profit price on an active position and persists state."""
        sym = symbol.upper()
        if sym == "BTC-PERPETUAL":
            sym = "BTCUSDT"
        pos = self.positions.get(sym)
        if pos:
            if stop_loss_price is not None:
                pos.stop_loss_price = round(stop_loss_price, 4 if stop_loss_price < 1 else 2)
            if take_profit_price is not None:
                pos.take_profit_price = round(take_profit_price, 4 if take_profit_price < 1 else 2)
            acc = self.get_account(sym)
            if acc and acc.position:
                if stop_loss_price is not None:
                    acc.position.stop_loss_price = pos.stop_loss_price
                if take_profit_price is not None:
                    acc.position.take_profit_price = pos.take_profit_price
            if sym == self.symbol:
                self._sync_legacy_attributes(sym)
            self._save_state()

    def reset(self):
        """Resets all balances, positions, and trades to clean initial $10,000 state for all markets."""
        self.market_accounts.clear()
        self.cash_usdt = float(self.initial_balance)
        self.positions.clear()
        self.position_side = "NONE"
        self.position_btc = 0.0
        self.position_eth = 0.0
        self.margin_usdt = 0.0
        self.entry_price = 0.0
        self.entry_timestamp = None
        self.stop_loss_price = None
        self.take_profit_price = None
        self.liquidation_price = None
        self.trade_history = []
        # Initialize default active symbol with clean $10,000 account
        self.get_account(self.symbol)
        self._save_state()
        logger.info(f"Reset futures paper trading to initial ${self.initial_balance:.2f} USDT across all markets")

    def get_portfolio_state(self, current_price: float, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Calculates current futures equity, positions, margin, and unrealized ROE% for symbol's isolated $10,000 account."""
        target_symbol = (symbol or self.symbol or "BTCUSDT").upper()
        if target_symbol == "BTC-PERPETUAL":
            target_symbol = "BTCUSDT"

        acc = self.get_account(target_symbol)
        pos = acc.position or self.positions.get(target_symbol)
        if not pos and target_symbol == "BTCUSDT":
            pos = self.positions.get("BTC-PERPETUAL")

        has_pos = pos is not None and pos.side in ("LONG", "SHORT") and pos.size_asset > 0.000001
        pos_side = pos.side if has_pos else "NONE"
        pos_size = pos.size_asset if has_pos else 0.0
        pos_margin = pos.margin_usdt if has_pos else 0.0
        pos_entry = pos.entry_price if has_pos else None
        pos_ts = pos.entry_timestamp if has_pos else None
        pos_sl = pos.stop_loss_price if has_pos else None
        pos_tp = pos.take_profit_price if has_pos else None
        pos_liq = pos.liquidation_price if has_pos else None
        pos_lev = pos.leverage if has_pos else self.leverage

        notional = pos_size * current_price if has_pos else 0.0
        unrealized_pnl_usd = 0.0
        unrealized_pnl_pct = 0.0
        roe_pct = 0.0

        if has_pos and pos_entry and pos_entry > 0:
            if pos_side == "LONG":
                unrealized_pnl_usd = (current_price - pos_entry) * pos_size
                unrealized_pnl_pct = ((current_price - pos_entry) / pos_entry) * 100
            elif pos_side == "SHORT":
                unrealized_pnl_usd = (pos_entry - current_price) * pos_size
                unrealized_pnl_pct = ((pos_entry - current_price) / pos_entry) * 100

            if pos_margin > 0:
                roe_pct = (unrealized_pnl_usd / pos_margin) * 100

        # Isolated accounting: this symbol's total equity = symbol cash + symbol margin + symbol unrealized PnL
        market_equity = acc.cash_usdt + pos_margin + unrealized_pnl_usd
        daily_pnl_pct = ((market_equity - acc.initial_balance) / acc.initial_balance)
        margin_utilization = (pos_margin / market_equity) * 100 if market_equity > 0 else 0.0

        # Calculate minutes held if in open position
        minutes_held = 0
        if has_pos and pos_ts:
            try:
                dt_entry = datetime.fromisoformat(pos_ts.replace("Z", "+00:00"))
                minutes_held = max(0, int((datetime.now(timezone.utc) - dt_entry).total_seconds() / 60))
            except Exception:
                minutes_held = 0

        # Cumulative fees paid & before-costs equity for this market
        market_fees = sum(t.fee_usd for t in acc.trade_history)
        before_costs_equity = market_equity + market_fees

        return {
            "symbol": target_symbol,
            "contract_type": "PERPETUAL_FUTURES",
            "initial_balance_usdt": round(acc.initial_balance, 2),
            "cash_usdt": round(acc.cash_usdt, 2),
            "free_margin_usdt": round(acc.cash_usdt, 2),
            "margin_used_usdt": round(pos_margin, 2),
            "total_margin_used_usdt": round(pos_margin, 2),
            "margin_usd": round(pos_margin, 2),
            "margin_utilization_pct": round(margin_utilization, 1),
            "leverage": pos_lev,
            "position_side": pos_side,
            "position_size_asset": round(pos_size, 6),
            "position_size_btc": round(pos_size, 6),
            "position_size_eth": round(pos_size, 6),
            "position_value_usd": round(notional, 2),
            "notional_usd": round(notional, 2),
            "total_equity_usdt": round(market_equity, 2),
            "before_costs_equity_usdt": round(before_costs_equity, 2),
            "cumulative_fees_usdt": round(market_fees, 2),
            "max_drawdown_pct": 0.0,
            "minutes_held": minutes_held,
            "has_open_position": has_pos,
            "entry_price": round(pos_entry, 4 if (pos_entry and pos_entry < 1) else 2) if (has_pos and pos_entry) else None,
            "entry_timestamp": pos_ts,
            "stop_loss_price": pos_sl,
            "take_profit_price": pos_tp,
            "liquidation_price": pos_liq,
            "unrealized_pnl_usd": round(unrealized_pnl_usd, 2),
            "unrealized_pnl_pct": round(unrealized_pnl_pct, 2),
            "roe_pct": round(roe_pct, 2),
            "daily_pnl_pct": round(daily_pnl_pct * 100, 2),
            "active_positions_count": len([a for a in self.market_accounts.values() if a.position is not None]),
            "active_positions": {sym: p.to_dict() for sym, p in self.positions.items()},
        }

    def get_master_portfolio_overview(self, current_prices: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Calculates multi-stock master portfolio overview with independent $10,000 capital accounts across markets."""
        prices = current_prices or {}
        now_dt = datetime.now(timezone.utc)

        total_realized_pnl = 0.0
        total_fees = 0.0
        total_unrealized_pnl = 0.0
        total_margin_used = 0.0
        total_notional_exposure = 0.0
        winning_trades = 0
        losing_trades = 0

        active_positions_list: List[Dict[str, Any]] = []

        symbol_realized_pnl: Dict[str, float] = {}
        symbol_fees: Dict[str, float] = {}
        symbol_trade_count: Dict[str, int] = {}
        symbol_capital: Dict[str, float] = {}
        symbol_cash: Dict[str, float] = {}

        # Collect all trades
        all_trades = []
        for sym, acc in self.market_accounts.items():
            all_trades.extend(acc.trade_history)
        if not all_trades:
            all_trades = self.trade_history

        for t in all_trades:
            sym = "BTCUSDT" if t.symbol == "BTC-PERPETUAL" else t.symbol
            symbol_realized_pnl[sym] = round(symbol_realized_pnl.get(sym, 0.0) + t.realized_pnl_usd, 2)
            symbol_fees[sym] = round(symbol_fees.get(sym, 0.0) + t.fee_usd, 2)
            symbol_trade_count[sym] = symbol_trade_count.get(sym, 0) + 1
            total_realized_pnl += t.realized_pnl_usd
            total_fees += t.fee_usd
            if t.realized_pnl_usd > 0.001:
                winning_trades += 1
            elif t.realized_pnl_usd < -0.001:
                losing_trades += 1

        # Calculate position metrics
        for sym, pos in list(self.positions.items()):
            norm_sym = "BTCUSDT" if sym == "BTC-PERPETUAL" else sym
            cur_price = prices.get(norm_sym, prices.get(sym, pos.entry_price))

            unrealized_usd = 0.0
            unrealized_pct = 0.0
            roe_pct = 0.0

            if pos.entry_price > 0:
                if pos.side == "LONG":
                    unrealized_usd = (cur_price - pos.entry_price) * pos.size_asset
                    unrealized_pct = ((cur_price - pos.entry_price) / pos.entry_price) * 100
                elif pos.side == "SHORT":
                    unrealized_usd = (pos.entry_price - cur_price) * pos.size_asset
                    unrealized_pct = ((pos.entry_price - cur_price) / pos.entry_price) * 100

                if pos.margin_usdt > 0:
                    roe_pct = (unrealized_usd / pos.margin_usdt) * 100

            total_unrealized_pnl += unrealized_usd
            total_margin_used += pos.margin_usdt
            notional = pos.size_asset * cur_price
            total_notional_exposure += notional

            minutes_held = 0
            if pos.entry_timestamp:
                try:
                    dt_entry = datetime.fromisoformat(pos.entry_timestamp.replace("Z", "+00:00"))
                    minutes_held = max(0, int((now_dt - dt_entry).total_seconds() / 60))
                except Exception:
                    minutes_held = 0

            active_positions_list.append({
                "symbol": norm_sym,
                "side": pos.side,
                "size_asset": round(pos.size_asset, 6),
                "notional_usd": round(notional, 2),
                "entry_price": round(pos.entry_price, 4 if pos.entry_price < 1 else 2),
                "current_price": round(cur_price, 4 if cur_price < 1 else 2),
                "margin_usdt": round(pos.margin_usdt, 2),
                "leverage": pos.leverage,
                "unrealized_pnl_usd": round(unrealized_usd, 2),
                "unrealized_pnl_pct": round(unrealized_pct, 2),
                "roe_pct": round(roe_pct, 2),
                "stop_loss_price": round(pos.stop_loss_price, 2) if pos.stop_loss_price else None,
                "take_profit_price": round(pos.take_profit_price, 2) if pos.take_profit_price else None,
                "liquidation_price": round(pos.liquidation_price, 2) if pos.liquidation_price else None,
                "entry_timestamp": pos.entry_timestamp,
                "minutes_held": minutes_held,
            })

        # Calculate per-symbol capital and cash ($10,000 baseline per market)
        for sym, acc in self.market_accounts.items():
            norm_sym = "BTCUSDT" if sym == "BTC-PERPETUAL" else sym
            cur_price = prices.get(norm_sym, 0.0)
            u_pnl = 0.0
            if acc.position and acc.position.entry_price > 0 and cur_price > 0:
                if acc.position.side == "LONG":
                    u_pnl = (cur_price - acc.position.entry_price) * acc.position.size_asset
                elif acc.position.side == "SHORT":
                    u_pnl = (acc.position.entry_price - cur_price) * acc.position.size_asset
            used_m = acc.position.margin_usdt if acc.position else 0.0
            symbol_capital[norm_sym] = round(acc.cash_usdt + used_m + u_pnl, 2)
            symbol_cash[norm_sym] = round(acc.cash_usdt, 2)

        # Multi-market isolated capital calculations:
        active_symbols = [s for s, pos in self.positions.items() if pos and pos.side in ("LONG", "SHORT")]
        active_accounts_count = len(active_symbols) or 1
        
        # Capital pool across markets with active positions (each backed by $10k isolated account)
        active_capital_pool = sum(symbol_capital.get(s, 10000.0) for s in active_symbols) if active_symbols else 10000.0
        active_cash_pool = sum(symbol_cash.get(s, 10000.0) for s in active_symbols) if active_symbols else 10000.0

        # Margin utilization across active accounts:
        # e.g. 6 positions using $2k margin = $12k margin out of 6 x $10k accounts ($60k) = 20.0%
        margin_utilization = (total_margin_used / active_capital_pool) * 100.0 if active_capital_pool > 0 else 0.0

        total_pnl_usd = total_realized_pnl + total_unrealized_pnl
        closed_trades = winning_trades + losing_trades
        win_rate = (winning_trades / closed_trades) * 100.0 if closed_trades > 0 else 0.0

        return {
            "portfolio_summary": {
                "initial_balance_usdt": round(float(self.initial_balance), 2),
                "total_equity_usdt": round(active_capital_pool, 2),
                "active_capital_pool_usdt": round(active_capital_pool, 2),
                "active_cash_usdt": round(active_cash_pool, 2),
                "cash_usdt": round(active_cash_pool, 2),
                "free_margin_usdt": round(active_cash_pool, 2),
                "total_margin_used_usdt": round(total_margin_used, 2),
                "margin_utilization_pct": round(margin_utilization, 1),
                "total_notional_usd": round(total_notional_exposure, 2),
                "total_unrealized_pnl_usd": round(total_unrealized_pnl, 2),
                "total_realized_pnl_usd": round(total_realized_pnl, 2),
                "total_pnl_usd": round(total_pnl_usd, 2),
                "total_pnl_pct": round((total_pnl_usd / active_capital_pool) * 100.0, 2) if active_capital_pool > 0 else 0.0,
                "total_fees_paid_usd": round(total_fees, 2),
                "active_positions_count": len(self.positions),
                "total_trades": len(all_trades),
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate_pct": round(win_rate, 1),
            },
            "active_positions": active_positions_list,
            "symbol_capital": symbol_capital,
            "symbol_cash": symbol_cash,
            "symbol_realized_pnl": symbol_realized_pnl,
            "symbol_fees": symbol_fees,
            "symbol_trade_count": symbol_trade_count,
        }

    def get_full_trade_history(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns structured trade records with paired entry/exit metrics, duration, PnL, fees, and reasons."""
        target_sym = symbol.upper() if symbol else None
        round_trips: List[Dict[str, Any]] = []

        # Gather accounts to process
        if target_sym and target_sym in self.market_accounts:
            accounts = [self.market_accounts[target_sym]]
        else:
            accounts = list(self.market_accounts.values())

        for acc in accounts:
            sym = acc.symbol
            if not sym.isascii():
                continue
            trades = list(acc.trade_history)
            pending_entry: Optional[TradeRecord] = None

            for t in trades:
                if t.side in ("LONG", "SHORT"):
                    pending_entry = t
                elif t.side in ("EXIT_LONG", "EXIT_SHORT") and pending_entry:
                    duration_str = "—"
                    duration_secs = 0
                    try:
                        dt_in = datetime.fromisoformat(pending_entry.timestamp.replace("Z", "+00:00"))
                        dt_out = datetime.fromisoformat(t.timestamp.replace("Z", "+00:00"))
                        duration_secs = max(0, int((dt_out - dt_in).total_seconds()))
                        mins = duration_secs // 60
                        secs = duration_secs % 60
                        if mins >= 60:
                            hrs = mins // 60
                            mins = mins % 60
                            duration_str = f"{hrs}h {mins}m"
                        else:
                            duration_str = f"{mins}m {secs}s"
                    except Exception:
                        pass

                    total_fee = round(pending_entry.fee_usd + t.fee_usd, 2)
                    gross_pnl = round(t.realized_pnl_usd + t.fee_usd, 2)

                    round_trips.append({
                        "trade_id": t.trade_id.replace("EXIT_", "RT_"),
                        "symbol": sym,
                        "side": pending_entry.side,
                        "status": "CLOSED",
                        "entry_time": pending_entry.timestamp,
                        "exit_time": t.timestamp,
                        "entry_price": pending_entry.effective_price,
                        "exit_price": t.effective_price,
                        "size_asset": pending_entry.amount_btc,
                        "notional_usd": pending_entry.notional_usd,
                        "margin_usd": t.margin_usd,
                        "leverage": t.leverage,
                        "entry_fee_usd": pending_entry.fee_usd,
                        "exit_fee_usd": t.fee_usd,
                        "total_fees_usd": total_fee,
                        "gross_pnl_usd": gross_pnl,
                        "realized_pnl_usd": t.realized_pnl_usd,
                        "roe_pct": t.realized_pnl_pct,
                        "exit_reason": t.reason,
                        "entry_reason": pending_entry.reason,
                        "confidence": pending_entry.confidence,
                        "duration_seconds": duration_secs,
                        "duration_str": duration_str,
                    })
                    pending_entry = None

            # Check if there is currently an open position for this account
            if acc.position and acc.position.entry_price > 0:
                pos = acc.position
                duration_str = "—"
                duration_secs = 0
                if pos.entry_timestamp:
                    try:
                        dt_in = datetime.fromisoformat(pos.entry_timestamp.replace("Z", "+00:00"))
                        duration_secs = max(0, int((datetime.now(timezone.utc) - dt_in).total_seconds()))
                        mins = duration_secs // 60
                        secs = duration_secs % 60
                        duration_str = f"{mins}m {secs}s"
                    except Exception:
                        pass

                round_trips.append({
                    "trade_id": f"OPEN_{sym}",
                    "symbol": sym,
                    "side": pos.side,
                    "status": "OPEN",
                    "entry_time": pos.entry_timestamp,
                    "exit_time": None,
                    "entry_price": round(pos.entry_price, 4 if pos.entry_price < 1 else 2),
                    "exit_price": None,
                    "size_asset": round(pos.size_asset, 6),
                    "notional_usd": round(pos.size_asset * pos.entry_price, 2),
                    "margin_usd": round(pos.margin_usdt, 2),
                    "leverage": pos.leverage,
                    "entry_fee_usd": round(pos.margin_usdt * pos.leverage * 0.0005, 2),
                    "exit_fee_usd": 0.0,
                    "total_fees_usd": round(pos.margin_usdt * pos.leverage * 0.0005, 2),
                    "gross_pnl_usd": 0.0,
                    "realized_pnl_usd": 0.0,
                    "roe_pct": 0.0,
                    "exit_reason": "Position Currently Active",
                    "entry_reason": "Autonomous Jev Policy Entry",
                    "confidence": 0.8,
                    "duration_seconds": duration_secs,
                    "duration_str": duration_str,
                })

        # Sort descending by timestamp
        round_trips.sort(key=lambda x: x.get("exit_time") or x.get("entry_time") or "", reverse=True)
        return round_trips

    def execute_order(self, plan: OrderPlan, current_price: float, symbol: Optional[str] = None) -> Optional[TradeRecord]:
        """Executes simulated perpetual futures order against symbol's isolated $10,000 account."""
        if not plan.should_execute or plan.order_type == "HOLD":
            return None

        target_symbol = (symbol or getattr(plan, "symbol", None) or self.symbol or "BTCUSDT").upper()
        if target_symbol == "BTC-PERPETUAL":
            target_symbol = "BTCUSDT"

        if not target_symbol.isascii() or not target_symbol.isalnum():
            logger.warning(f"Rejected non-ASCII symbol {target_symbol}")
            return None

        acc = self.get_account(target_symbol)
        now = datetime.now(timezone.utc).isoformat()
        slippage_factor = self.config.slippage_bps / 10000.0
        fee_factor = self.config.fee_bps / 10000.0

        # 1. ENTER LONG (Buy Perpetuals)
        if plan.order_type == "ENTER_LONG":
            if acc.position is not None or target_symbol in self.positions:
                logger.warning(f"Cannot enter LONG while position for {target_symbol} is already active.")
                return None

            effective_price = current_price * (1.0 + slippage_factor)
            # Allocate margin from this market's isolated $10,000 cash account!
            allocated_margin = min(plan.size_usd if plan.size_usd > 0 else (acc.cash_usdt * 0.20), acc.cash_usdt * 0.95)
            if allocated_margin < 10.0:
                logger.warning(f"Insufficient cash (${acc.cash_usdt:.2f}) in {target_symbol} account for margin allocation.")
                return None

            notional = allocated_margin * self.leverage
            fee = notional * fee_factor
            net_notional = notional - fee
            asset_amount = net_notional / effective_price

            # Liquidation Price for Long: Entry * (1 - 1/Lev + MaintMargin)
            liq_price = effective_price * (1.0 - (1.0 / self.leverage) + self.maintenance_margin_pct)

            acc.cash_usdt -= allocated_margin
            pos = AssetPosition(
                symbol=target_symbol,
                side="LONG",
                size_asset=asset_amount,
                entry_price=effective_price,
                margin_usdt=allocated_margin,
                leverage=self.leverage,
                entry_timestamp=now,
                stop_loss_price=plan.stop_loss_price,
                take_profit_price=plan.take_profit_price,
                liquidation_price=round(liq_price, 4 if liq_price < 1 else 2),
            )
            acc.position = pos
            self.positions[target_symbol] = pos

            trade = TradeRecord(
                trade_id=f"LONG_{target_symbol}_{len(acc.trade_history) + 1}",
                symbol=target_symbol,
                side="LONG",
                timestamp=now,
                price=round(current_price, 4 if current_price < 1 else 2),
                effective_price=round(effective_price, 4 if effective_price < 1 else 2),
                amount_btc=round(asset_amount, 6),
                amount_eth=round(asset_amount, 6),
                notional_usd=round(notional, 2),
                margin_usd=round(allocated_margin, 2),
                leverage=self.leverage,
                fee_usd=round(fee, 2),
                realized_pnl_usd=0.0,
                realized_pnl_pct=0.0,
                reason=plan.reason,
                confidence=plan.confidence,
            )
            acc.trade_history.append(trade)
            self.trade_history.append(trade)
            self._sync_legacy_attributes(target_symbol)
            self._save_state()
            logger.info(
                f"[FUTURES EXECUTION] OPENED LONG {asset_amount:.4f} {target_symbol} @ ${effective_price:.2f} "
                f"({self.leverage}x Lev, Margin: ${allocated_margin:.2f}, Cash Left: ${acc.cash_usdt:.2f}) - {plan.reason}"
            )
            return trade

        # 2. ENTER SHORT (Sell/Short Perpetuals)
        elif plan.order_type == "ENTER_SHORT":
            if acc.position is not None or target_symbol in self.positions:
                logger.warning(f"Cannot enter SHORT while position for {target_symbol} is already active.")
                return None

            effective_price = current_price * (1.0 - slippage_factor)
            allocated_margin = min(plan.size_usd if plan.size_usd > 0 else (acc.cash_usdt * 0.20), acc.cash_usdt * 0.95)
            if allocated_margin < 10.0:
                logger.warning(f"Insufficient cash (${acc.cash_usdt:.2f}) in {target_symbol} account for margin allocation.")
                return None

            notional = allocated_margin * self.leverage
            fee = notional * fee_factor
            net_notional = notional - fee
            asset_amount = net_notional / effective_price

            # Liquidation Price for Short: Entry * (1 + 1/Lev - MaintMargin)
            liq_price = effective_price * (1.0 + (1.0 / self.leverage) - self.maintenance_margin_pct)

            acc.cash_usdt -= allocated_margin
            pos = AssetPosition(
                symbol=target_symbol,
                side="SHORT",
                size_asset=asset_amount,
                entry_price=effective_price,
                margin_usdt=allocated_margin,
                leverage=self.leverage,
                entry_timestamp=now,
                stop_loss_price=plan.stop_loss_price,
                take_profit_price=plan.take_profit_price,
                liquidation_price=round(liq_price, 4 if liq_price < 1 else 2),
            )
            acc.position = pos
            self.positions[target_symbol] = pos

            trade = TradeRecord(
                trade_id=f"SHORT_{target_symbol}_{len(acc.trade_history) + 1}",
                symbol=target_symbol,
                side="SHORT",
                timestamp=now,
                price=round(current_price, 4 if current_price < 1 else 2),
                effective_price=round(effective_price, 4 if effective_price < 1 else 2),
                amount_btc=round(asset_amount, 6),
                amount_eth=round(asset_amount, 6),
                notional_usd=round(notional, 2),
                margin_usd=round(allocated_margin, 2),
                leverage=self.leverage,
                fee_usd=round(fee, 2),
                realized_pnl_usd=0.0,
                realized_pnl_pct=0.0,
                reason=plan.reason,
                confidence=plan.confidence,
            )
            acc.trade_history.append(trade)
            self.trade_history.append(trade)
            self._sync_legacy_attributes(target_symbol)
            self._save_state()
            logger.info(
                f"[FUTURES EXECUTION] OPENED SHORT {asset_amount:.4f} {target_symbol} @ ${effective_price:.2f} "
                f"({self.leverage}x Lev, Margin: ${allocated_margin:.2f}, Cash Left: ${acc.cash_usdt:.2f}) - {plan.reason}"
            )
            return trade

        # 3. EXIT LONG
        elif plan.order_type == "EXIT_LONG":
            pos = acc.position or self.positions.get(target_symbol)
            if not pos and target_symbol == "BTCUSDT":
                pos = self.positions.get("BTC-PERPETUAL")
            if not pos or pos.side != "LONG" or pos.size_asset <= 0.000001:
                return None

            acc.position = None
            self.positions.pop(target_symbol, None)
            if target_symbol == "BTCUSDT":
                self.positions.pop("BTC-PERPETUAL", None)

            effective_price = current_price * (1.0 - slippage_factor)
            notional = pos.size_asset * effective_price
            fee = notional * fee_factor
            gross_pnl = (effective_price - pos.entry_price) * pos.size_asset
            net_realized_pnl = gross_pnl - fee
            roe_pct = (net_realized_pnl / pos.margin_usdt) * 100 if pos.margin_usdt > 0 else 0.0

            returned_cash = max(0.0, pos.margin_usdt + net_realized_pnl)
            acc.cash_usdt += returned_cash

            closed_size = pos.size_asset
            trade = TradeRecord(
                trade_id=f"EXIT_LONG_{target_symbol}_{len(acc.trade_history) + 1}",
                symbol=target_symbol,
                side="EXIT_LONG",
                timestamp=now,
                price=round(current_price, 4 if current_price < 1 else 2),
                effective_price=round(effective_price, 4 if effective_price < 1 else 2),
                amount_btc=round(closed_size, 6),
                amount_eth=round(closed_size, 6),
                notional_usd=round(notional, 2),
                margin_usd=round(pos.margin_usdt, 2),
                leverage=pos.leverage,
                fee_usd=round(fee, 2),
                realized_pnl_usd=round(net_realized_pnl, 2),
                realized_pnl_pct=round(roe_pct, 2),
                reason=plan.reason,
                confidence=plan.confidence,
            )

            acc.trade_history.append(trade)
            self.trade_history.append(trade)
            self._sync_legacy_attributes(target_symbol)
            self._save_state()
            logger.info(
                f"[FUTURES EXECUTION] CLOSED LONG {closed_size:.4f} {target_symbol} @ ${effective_price:.2f} "
                f"(Realized PnL: ${net_realized_pnl:.2f}, ROE: {roe_pct:+.2f}%, Cash: ${acc.cash_usdt:.2f}) - {plan.reason}"
            )
            return trade

        # 4. EXIT SHORT
        elif plan.order_type == "EXIT_SHORT":
            pos = acc.position or self.positions.get(target_symbol)
            if not pos and target_symbol == "BTCUSDT":
                pos = self.positions.get("BTC-PERPETUAL")
            if not pos or pos.side != "SHORT" or pos.size_asset <= 0.000001:
                return None

            acc.position = None
            self.positions.pop(target_symbol, None)
            if target_symbol == "BTCUSDT":
                self.positions.pop("BTC-PERPETUAL", None)

            effective_price = current_price * (1.0 + slippage_factor)
            notional = pos.size_asset * effective_price
            fee = notional * fee_factor
            gross_pnl = (pos.entry_price - effective_price) * pos.size_asset
            net_realized_pnl = gross_pnl - fee
            roe_pct = (net_realized_pnl / pos.margin_usdt) * 100 if pos.margin_usdt > 0 else 0.0

            returned_cash = max(0.0, pos.margin_usdt + net_realized_pnl)
            acc.cash_usdt += returned_cash

            closed_size = pos.size_asset
            trade = TradeRecord(
                trade_id=f"EXIT_SHORT_{target_symbol}_{len(acc.trade_history) + 1}",
                symbol=target_symbol,
                side="EXIT_SHORT",
                timestamp=now,
                price=round(current_price, 4 if current_price < 1 else 2),
                effective_price=round(effective_price, 4 if effective_price < 1 else 2),
                amount_btc=round(closed_size, 6),
                amount_eth=round(closed_size, 6),
                notional_usd=round(notional, 2),
                margin_usd=round(pos.margin_usdt, 2),
                leverage=pos.leverage,
                fee_usd=round(fee, 2),
                realized_pnl_usd=round(net_realized_pnl, 2),
                realized_pnl_pct=round(roe_pct, 2),
                reason=plan.reason,
                confidence=plan.confidence,
            )

            acc.trade_history.append(trade)
            self.trade_history.append(trade)
            self._sync_legacy_attributes(target_symbol)
            self._save_state()
            logger.info(
                f"[FUTURES EXECUTION] CLOSED SHORT {closed_size:.4f} {target_symbol} @ ${effective_price:.2f} "
                f"(Realized PnL: ${net_realized_pnl:.2f}, ROE: {roe_pct:+.2f}%, Cash: ${acc.cash_usdt:.2f}) - {plan.reason}"
            )
            return trade

        return None

    def get_performance_summary(self) -> Dict[str, Any]:
        """Calculates win rate, profit factor, total realized PnL across closed futures trades."""
        closed_trades = [t for t in self.trade_history if t.side.startswith("EXIT")]
        total_fees = sum(t.fee_usd for t in self.trade_history)
        stopped_out = sum(1 for t in closed_trades if "stop" in t.reason.lower())

        if not closed_trades:
            return {
                "total_trades": len(self.trade_history),
                "closed_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "closed_w_l": "0 / 0",
                "stopped_out": 0,
                "fees_paid": round(total_fees, 2),
                "win_rate_pct": 0.0,
                "profit_factor": 0.0,
                "total_realized_pnl_usd": 0.0,
            }

        wins = [t for t in closed_trades if t.realized_pnl_usd > 0]
        losses = [t for t in closed_trades if t.realized_pnl_usd <= 0]
        total_profit = sum(t.realized_pnl_usd for t in wins)
        total_loss = abs(sum(t.realized_pnl_usd for t in losses))
        profit_factor = round(total_profit / total_loss, 2) if total_loss > 0 else (999.0 if total_profit > 0 else 1.0)
        win_rate = round((len(wins) / len(closed_trades)) * 100, 1)

        return {
            "total_trades": len(self.trade_history),
            "closed_trades": len(closed_trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "closed_w_l": f"{len(wins)} / {len(losses)}",
            "stopped_out": stopped_out,
            "fees_paid": round(total_fees, 2),
            "win_rate_pct": win_rate,
            "profit_factor": profit_factor,
            "total_realized_pnl_usd": round(sum(t.realized_pnl_usd for t in closed_trades), 2),
        }

    def get_chart_markers(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Converts trade history into TradingView Lightweight Charts marker objects, filtered by symbol."""
        markers = []
        target_sym = symbol or self.symbol
        is_btc = target_sym in ("BTCUSDT", "BTC-PERPETUAL")

        used_times: dict = {}

        def unique_time(ts: int, position: str) -> int:
            key = (ts, position)
            count = used_times.get(key, 0)
            used_times[key] = count + 1
            return ts + count  # offset duplicates by 1 second each

        for trade in self.trade_history:
            if symbol:
                trade_is_btc = trade.symbol in ("BTCUSDT", "BTC-PERPETUAL")
                if is_btc:
                    if not trade_is_btc:
                        continue
                else:
                    if trade.symbol != symbol:
                        continue

            try:
                dt = datetime.fromisoformat(trade.timestamp.replace("Z", "+00:00"))
                ts_sec = int(dt.timestamp())
            except Exception:
                ts_sec = int(datetime.now(timezone.utc).timestamp())

            price = trade.effective_price
            # Compact price: use integer for large prices, 4dp for micro prices
            if price >= 100:
                price_str = f"${int(price):,}"
            elif price >= 1:
                price_str = f"${price:.2f}"
            else:
                price_str = f"${price:.5f}"

            if trade.side in ("LONG", "BUY"):
                pos = "belowBar"
                t = unique_time(ts_sec, pos)
                markers.append({"time": t, "position": pos, "color": "#22c55e",
                    "shape": "arrowUp", "text": f"L {price_str}"})
            elif trade.side in ("SHORT", "SELL"):
                pos = "aboveBar"
                t = unique_time(ts_sec, pos)
                markers.append({"time": t, "position": pos, "color": "#f43f5e",
                    "shape": "arrowDown", "text": f"S {price_str}"})
            elif trade.side == "EXIT_LONG":
                pnl = trade.realized_pnl_usd
                color = "#10b981" if pnl >= 0 else "#ef4444"
                sign = "+" if pnl >= 0 else ""
                pos = "aboveBar"
                t = unique_time(ts_sec, pos)
                markers.append({"time": t, "position": pos, "color": color,
                    "shape": "circle", "text": f"{sign}${int(pnl)}"})
            elif trade.side == "EXIT_SHORT":
                pnl = trade.realized_pnl_usd
                color = "#10b981" if pnl >= 0 else "#ef4444"
                sign = "+" if pnl >= 0 else ""
                pos = "belowBar"
                t = unique_time(ts_sec, pos)
                markers.append({"time": t, "position": pos, "color": color,
                    "shape": "circle", "text": f"{sign}${int(pnl)}"})

        return markers

    def reset(self):
        """Resets all market accounts, positions, and history to clean slate with 0 tracked contracts."""
        self.cash_usdt = self.initial_balance
        self.market_accounts = {}
        self.positions = {}
        self.trade_history = []
        self.position_side = "NONE"
        self.position_btc = 0.0
        self.position_eth = 0.0
        self.margin_usdt = 0.0
        self.entry_price = 0.0
        self.entry_timestamp = None
        self.stop_loss_price = None
        self.take_profit_price = None
        self.liquidation_price = None
        self.tracked_symbols = []
        self._save_state()
