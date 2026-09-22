"""Configuration manager for the Jev ETH Trading Bot.

Loads environment variables, validates parameters, and provides strong typing
for all trading, risk, and model configuration options.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Base directory for the project
BASE_DIR = Path(__file__).resolve().parent

# Load .env file
load_dotenv(dotenv_path=BASE_DIR / ".env")


@dataclass(frozen=True)
class RiskConfig:
    min_confidence_threshold: float
    max_risk_score: float
    min_entry_conviction: float
    exit_urgency_threshold: float
    position_size_pct: float
    max_open_positions: int
    stop_loss_pct: float
    take_profit_pct: float
    use_atr_stop: bool
    atr_multiplier: float
    max_daily_drawdown_pct: float


@dataclass(frozen=True)
class SimulationConfig:
    initial_balance_usdt: float
    slippage_bps: float
    fee_bps: float
    is_futures: bool = True
    leverage: int = 2
    maintenance_margin_pct: float = 0.005


@dataclass(frozen=True)
class BotConfig:
    typesafe_api_key: str
    typesafe_model: str
    typesafe_endpoint: str
    trading_mode: str
    trading_pair: str
    timeframe: str
    cycle_interval_seconds: int
    risk: RiskConfig
    simulation: SimulationConfig

    @property
    def has_valid_api_key(self) -> bool:
        return bool(self.typesafe_api_key and not self.typesafe_api_key.startswith("your_"))


def load_config() -> BotConfig:
    """Loads configuration from environment variables with sensible defaults."""
    typesafe_key = os.getenv("TYPESAFE_API_KEY", "").strip()
    model = os.getenv("TYPESAFE_MODEL", "jev-latest").strip()
    endpoint = os.getenv("TYPESAFE_ENDPOINT", "https://api.typesafe.ai/v1/systemone").strip()

    trading_mode = os.getenv("TRADING_MODE", "paper").lower().strip()
    trading_pair = os.getenv("TRADING_PAIR", "ETH/USDT").strip()
    timeframe = os.getenv("TIMEFRAME", "5m").strip()
    cycle_interval = int(os.getenv("CYCLE_INTERVAL_SECONDS", "60"))

    risk = RiskConfig(
        min_confidence_threshold=float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.70")),
        max_risk_score=float(os.getenv("MAX_RISK_SCORE", "1.5")),
        min_entry_conviction=float(os.getenv("MIN_ENTRY_CONVICTION", "0.65")),
        exit_urgency_threshold=float(os.getenv("EXIT_URGENCY_THRESHOLD", "0.75")),
        position_size_pct=float(os.getenv("POSITION_SIZE_PCT", "0.10")),
        max_open_positions=int(os.getenv("MAX_OPEN_POSITIONS", "1")),
        stop_loss_pct=float(os.getenv("STOP_LOSS_PCT", "0.025")),
        take_profit_pct=float(os.getenv("TAKE_PROFIT_PCT", "0.050")),
        use_atr_stop=os.getenv("USE_ATR_STOP", "True").lower() in ("true", "1", "yes"),
        atr_multiplier=float(os.getenv("ATR_MULTIPLIER", "1.8")),
        max_daily_drawdown_pct=float(os.getenv("MAX_DAILY_DRAWDOWN_PCT", "0.06")),
    )

    simulation = SimulationConfig(
        initial_balance_usdt=float(os.getenv("INITIAL_PAPER_BALANCE_USDT", "10000.0")),
        slippage_bps=float(os.getenv("SLIPPAGE_BPS", "5.0")),
        fee_bps=float(os.getenv("FEE_BPS", "5.0")),
        is_futures=os.getenv("IS_FUTURES", "True").lower() in ("true", "1", "yes"),
        leverage=int(os.getenv("LEVERAGE", "2")),
        maintenance_margin_pct=float(os.getenv("MAINTENANCE_MARGIN_PCT", "0.005")),
    )

    return BotConfig(
        typesafe_api_key=typesafe_key,
        typesafe_model=model,
        typesafe_endpoint=endpoint,
        trading_mode=trading_mode,
        trading_pair=trading_pair,
        timeframe=timeframe,
        cycle_interval_seconds=cycle_interval,
        risk=risk,
        simulation=simulation,
    )
