# TypeSafe Jev ETH Trading System 🤖

An autonomous, confidence-gated Ethereum (ETH) trading bot powered by TypeSafe AI's flagship System One model, **Jev**.

Unlike standard generative language models that produce unstructured text, **Jev** makes fast, structured, calibrated decisions with exact probabilities and confidence metrics (`Choice`, `Score`, and `Noul` primitives). This bot feeds live ETH market state and quantitative indicators into Jev, routes decisions through strict confidence and risk gates, and executes trades with simulated balance and accounting.

---

## Architecture Overview

```
                  ┌───────────────────────────────┐
                  │      Live Market Data         │
                  │  (Binance / Coinbase REST)    │
                  └──────────────┬────────────────┘
                                 │ OHLCV & Ticker
                                 ▼
                  ┌───────────────────────────────┐
                  │ Technical Indicator Engine    │
                  │ (RSI, EMA 9/21/50, MACD, ATR) │
                  └──────────────┬────────────────┘
                                 │ Structured State JSON
                                 ▼
                  ┌───────────────────────────────┐
                  │  TypeSafe Jev (System One)    │
                  │  • Action (Choice)            │
                  │  • Regime (Choice)            │
                  │  • Risk Score (Score: 0-3)    │
                  │  • Conviction / Urgency (Noul)│
                  └──────────────┬────────────────┘
                                 │ Answers + Confidence
                                 ▼
                  ┌───────────────────────────────┐
                  │ Risk Manager & Confidence Gate│
                  │  • Conf >= 70%                │
                  │  • Risk Score <= 1.5          │
                  │  • Dynamic ATR Stop Loss      │
                  └──────────────┬────────────────┘
                                 │ Approved Order Plan
                                 ▼
                  ┌───────────────────────────────┐
                  │ Paper Trading Execution Engine│
                  │ ($10k USDT, Slippage, Fees)   │
                  └───────────────────────────────┘
```

---

## Installation & Setup

1. **Navigate to the project directory**:
   ```bash
   cd C:\Users\Admin\.gemini\antigravity\scratch\jev_eth_trader
   ```

2. **Activate the Virtual Environment**:
   - Windows PowerShell:
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - Windows Command Prompt:
     ```cmd
     .\.venv\Scripts\activate.bat
     ```

3. **Configure your TypeSafe API Key**:
   Open `.env` and add your Jev API key:
   ```ini
   TYPESAFE_API_KEY=ts-live-your-api-key-here
   TYPESAFE_MODEL=jev-latest
   ```

---

## How to Run

### 1. Single-Cycle Evaluation (Telemetry Inspection)
Run a single cycle to inspect real-time ETH market conditions, Jev's evaluation, confidence scores, and risk routing:
```bash
python bot.py --once
```

### 2. Dry-Run / Offline Test (Mock Evaluator)
Test the entire pipeline without consuming TypeSafe API tokens:
```bash
python bot.py --once --mock
```

### 3. Continuous Trading Daemon
Run the autonomous loop every 60 seconds (customizable in `.env`):
```bash
python bot.py
```

### 4. Run Verification Suite
Run unit and integration tests across data fetching, indicator math, confidence gating, and execution:
```bash
python test_pipeline.py
```

---

## Confidence-Gated Decision Matrix

TypeSafe's core philosophy is: **The answer tells you what; confidence tells you whether to act.**

| Primitive | Metric | Target | Action Rule |
| :--- | :--- | :--- | :--- |
| **Trade Action** | `Choice` | `BUY` / `SELL` / `HOLD` | Requires confidence $\ge 70\%$ |
| **Downside Risk** | `Score` | 0.0 (Low) to 3.0 (Extreme) | Trades rejected if risk $> 1.5$ |
| **Entry Conviction** | `Noul` | 0.0 to 1.0 probability | Entry requires probability $\ge 0.65$ |
| **Exit Urgency** | `Noul` | 0.0 to 1.0 probability | Liquidates position immediately if $\ge 0.75$ |

---

## Risk Controls & Position Sizing

- **Position Sizing**: Allocates 10% of equity per trade (`POSITION_SIZE_PCT=0.10`).
- **Dynamic Volatility Stop Loss**: Calculated dynamically as $\text{Entry Price} - (1.8 \times \text{ATR}_{14})$.
- **Take Profit Target**: Calculated as $\text{Entry Price} + (3.6 \times \text{ATR}_{14})$ (1:2 Risk/Reward ratio).
- **Daily Drawdown Circuit Breaker**: Trading is halted automatically if daily drawdown reaches $6\%$.
- **Slippage & Fees**: Models 5 bps slippage ($0.05\%$) and 5 bps taker fees per simulated trade.
