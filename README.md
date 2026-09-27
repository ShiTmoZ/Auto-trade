# Tokyo Breakout Quantitative Execution Engine

A discrete-event execution system and quantitative gatekeeper for BTC/USDT, exploiting the structural expansion of Asian Session liquidity pools.

```
   00:00 UTC                 09:00 UTC
   ┌────────────────────────────────┐
   │       Tokyo Consolidation      │───────► Asian High (Liquidity Pool)
   │     [Order Accumulation]       │───────► Asian Low  (Liquidity Pool)
   └────────────────────────────────┘
                                    │
                         Expansion  ▼
                    [London / NY Displacement]
```

---

## Performance Summary (2020 – 2026 Walk-Forward)

Evaluated across **175,240 fifteen-minute candles** sourced directly from the Binance Vision repository.

| Metric | Measured Value | Benchmark / Note |
|:---|:---|:---|
| **Dataset Horizon** | Jan 2020 – Sep 2026 (6 Years) | Uncut historical tick-aggregate |
| **Total Qualified Executions** | 949 setups | ~13 trades / month (Selective) |
| **Take Profit Hits (+2.5R)** | 490 trades (51.6%) | Hard target at 2.5× structural risk |
| **Breakeven Exits (0.0R)** | 262 trades (27.6%) | Protected by automated +1.5R trailing pivot |
| **Stop Loss Exits (-1.0R)** | 197 trades (20.8%) | Capped at 1.0% equity risk |
| **Directional Win Rate** | **71.3%** | `Wins / (Wins + Losses)` |
| **Net Cumulative Alpha** | **+1,028.00 R** | Compounding unadjusted base return |
| **Mathematical Expectancy** | **+1.08 R** | Net expected value per execution |
| **Profit Factor** | **6.22** | Gross Profit / Gross Loss |

---

## Execution Mechanics

### 1. Range Discovery (00:00 – 09:00 UTC)
Calculates absolute extremities during the Tokyo window on 15m intervals:
$$\text{Range}_{\text{Tokyo}} = [\min(L_t), \max(H_t)] \quad \forall \; t \in [00:00, 09:00)$$
Both thresholds remain active order targets until mitigated by subsequent session price action.

### 2. Directional Filter
Pre-conditions entries on higher-timeframe momentum alignment:
$$\text{Bias} = \begin{cases} \text{BULLISH} & \text{if } C_{15m} > \text{EMA}_{50}(C) \\ \text{BEARISH} & \text{if } C_{15m} < \text{EMA}_{50}(C) \end{cases}$$
Counter-trend breakouts are pruned prior to signal evaluation.

### 3. The Three-Candle Acceptance Protocol
To prevent false-breakout capital decay, entries require structural acceptance:
* **Candle 0 (Displacement):** Requires a close beyond the boundary ($C_0 > \text{High}_{\text{Tokyo}}$ or $C_0 < \text{Low}_{\text{Tokyo}}$) with relative volume $V_0 \ge 1.3 \times \text{SMA}_{20}(V)$.
* **Candle 1 (Fair Value Confirmation):** Verifies imbalance preservation via Fair Value Gap ($L_1 > H_{-1}$ for longs; $H_1 < L_{-1}$ for shorts). Immediate returns into the session range are flagged as traps and discarded.
* **Candle 2 (Acceptance):** Confirms sustained trading outside the boundary before arming limit retest orders.

### 4. Risk Budgeting & Dynamic Sizing
* **Account Risk:** Strictly locked at 1.0% equity per position.
* **Position Formula:**
$$\text{Size}_{\text{BTC}} = \frac{\text{Equity} \times 0.01}{|P_{\text{entry}} - P_{\text{stop}}|}$$
* **Pivot Breakeven:** Stop loss moves automatically to $P_{\text{entry}}$ upon price attaining $+1.5\text{R}$ displacement.

---

## Machine Learning Gatekeeper (`Res-MLP`)

A lightweight Deep Residual MLP acts as an entry gatekeeper, evaluating a 15-dimensional quantitative vector:

* **Feature Topology:** Relative Volume, FVG Magnitude %, Body-to-Range Ratio, Normalized ATR Risk, Session Expansion %, UTC Hour, Seasonality, Retest Depth, Momentum Velocity.
* **Architecture:** Dual-block Dense projection with Layer Normalization, 30% Dropout, and a linear residual bridge to preserve macro features.
* **Loss Function:** Binary Focal Loss ($\alpha=0.55, \gamma=2.0$) prioritizing hard-to-classify edge distributions over naive class accuracy.

---

## Repository Layout

```
Auto-trade/
├── config.py                 # Core parameters (risk, pairs, thresholds)
├── data_engine.py            # Stream ingestion & Tokyo range parser
├── trend_filter.py           # 50-EMA structural momentum engine
├── breakout_validator.py     # 3-Candle acceptance & FVG verification
├── execution_risk.py         # Position sizing & +1.5R breakeven state machine
├── trade_logger.py           # SQLite persistence layer (trades.db)
├── gemini_reviewer.py        # Autonomous post-mortem analysis of stopped trades
├── mathematical_backtest.py  # Discrete event backtester
├── multi_year_backtest.py    # Zero-disk in-memory streaming ingestion (2020-2026)
├── pipeline_train.py         # PyTorch training pipeline for Res-MLP
├── main.py                   # Real-time daemon & paper trading loop
└── .github/workflows/
    └── train.yml             # Cloud walk-forward training & report generation
```

---

## Getting Started

### Local Paper Trading Loop
```bash
git clone https://github.com/ShiTmoZ/Auto-trade.git
cd Auto-trade
python3 main.py
```

### Reproduce 6-Year Walk-Forward Backtest
Runs directly in memory with zero external dependencies:
```bash
python3 multi_year_backtest.py 2020 2026
```
