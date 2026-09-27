# 🚀 Auto-Trade: Institutional Tokyo Breakout & ML Gatekeeper Engine

An institutional-grade, fully automated algorithmic trading system engineered for **BTC/USDT** on Binance. The architecture combines **ICT / RTM Asian Range Expansion**, a **3-Candle Confirmation Protocol**, a **15-Dimensional Machine Learning Gatekeeper** with conservative learning rate, and an **Autonomous Gemini AI Meta-Reviewer**.

---

## 📌 1. Core Trading Strategy

### 🏦 The Asian Range Edge (ICT / RTM)
During the Tokyo Session (00:00 to 09:00 UTC), low institutional volume creates a consolidation bracket where retail stop orders accumulate above the **Tokyo High (TH)** and below the **Tokyo Low (TL)**.
Subsequent sessions (London and New York) frequently drive price toward these liquidity pools to fuel trend expansion.

### 📐 Multi-Timeframe (MTF) Trend Filter
* **1-Hour (1H) & 4-Hour (4H) Alignment:**  
  * **LONG Trades:** Only permitted if 1H structure is `BULLISH` (trading above 50-EMA with bullish swing BOS) and 4H is not bearish.
  * **SHORT Trades:** Only permitted if 1H structure is `BEARISH` (trading below 50-EMA with bearish swing BOS) and 4H is not bullish.

### 🛡️ The 3-Candle Confirmation Protocol
Before a breakout is considered actionable, it must satisfy three structural criteria:
1. **Candle 0 (Initiation):** Must close strictly beyond the session level with volume $\ge 1.3\times$ the 20-period Volume SMA. Wick-only sweeps are filtered out instantly.
2. **Candle 1 (Momentum & FVG):** Must form a Fair Value Gap (FVG: $\text{Low}[1] > \text{High}[-1]$ for bullish; $\text{High}[1] < \text{Low}[-1]$ for bearish). A close back inside the session range flags an immediate 2-candle trap and aborts the signal.
3. **Candle 2 (Acceptance):** Confirms sustained acceptance beyond the broken level.

---

## 🧠 2. Quantitative ML Gatekeeper & Self-Improvement

Financial markets exhibit high stochastic noise where losses are an inherent characteristic. High learning rates cause models to rapidly overfit on transient noise. 

### ⚙️ Anti-Overfitting Regularization
* **Conservative Learning Rate:** $\eta = 0.015$ (prevents rapid weight destabilization).
* **L2 Regularization ($\lambda = 1.0$):** Suppresses large weight values, forcing the model to depend only on reliable macro features.
* **15 Quantitative Features:**
  1. `vol_ratio`: Volume of breakout candle relative to 20-period SMA.
  2. `fvg_size_pct`: Fair Value Gap size as a percentage of price.
  3. `has_fvg`: Binary indicator for FVG presence.
  4. `candle0_body_ratio`: Ratio of candle body to total range.
  5. `risk_atr_ratio`: Proposed Stop Loss distance normalized by ATR-14.
  6. `tokyo_range_pct`: Size of Tokyo consolidation relative to price.
  7. `hour_utc`: Normalized time of day (identifying London/NY overlap).
  8. `day_of_week`: Normalized day-of-week seasonality.
  9. `htf_alignment_strength`: Correlation score between 1H and 4H structure.
  10. `retest_depth_pct`: Depth of retest relative to entry price.
  11. `volatility_ratio`: Relative volatility expansion.
  12. `dist_from_tokyo_open_pct`: Displacement from Asian session open.
  13. `momentum_3c_pct`: 3-candle rate of change.
  14. `consecutive_run`: Number of consecutive candles in direction of break.
  15. `range_expansion_ratio`: Candle range relative to recent volatility.

---

## 🔬 3. Gemini Post-Mortem AI Meta-Reviewer

Every losing trade is logged to SQLite (`trade_history.db`). Periodically, the **Gemini Meta-Reviewer** (`ag/gemini-3.8-flash-high`) conducts automated post-mortems:
* Identifies failure mechanics (e.g. high-impact macro news, false expansion, session timing exhaustion).
* Records critique records directly into SQLite for post-analysis.

---

## 🏗️ 4. System Architecture

```text
Auto-trade/
├── config.py                 # Global parameters (risk 1.0%, R:R 1:2.5, endpoints)
├── data_engine.py            # Binance kline ingestion & Tokyo range calculation
├── trend_filter.py           # 1H and 4H MTF bias calculator
├── breakout_validator.py     # 3-Candle + FVG confirmation protocol
├── ml_gatekeeper.py          # 15-feature regularized ML gatekeeper
├── trade_logger.py           # SQLite database telemetry & state tracker
├── execution_risk.py         # Position sizing & +1.5R breakeven trigger
├── gemini_reviewer.py        # Automated LLM post-mortem module
├── train_and_backtest.py     # Backtesting & online training simulator
├── main.py                   # Real-time live / paper trading loop
└── requirements.txt          # Environment dependencies
```

---

## ⚡ 5. Quick Start

### 1. Requirements
The entire core engine is built strictly on Python 3.10+ standard libraries (`urllib`, `sqlite3`, `json`, `math`). No heavy external packages are mandatory.

### 2. Run the Bot (Paper Trading Mode)
```bash
python3 main.py
```

### 3. Run Historical Pre-Training / Backtest
```bash
python3 train_and_backtest.py
```

---

## 🛡️ Risk Management Parameters
* **Risk per Trade:** Strictly $1.0\%$ of account equity.
* **Risk-to-Reward:** Minimum $1:2.5$.
* **Breakeven Automation:** Automatically trails Stop Loss to entry upon reaching $+1.5\text{R}$.
