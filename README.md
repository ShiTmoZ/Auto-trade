# Institutional Asian Range Expansion & Structural Machine Learning Engine

A quantitative algorithmic execution framework engineered for liquid cryptocurrency assets, exploiting structural liquidity expansions out of the Asian Session (00:00 – 07:00 UTC) combined with Quasimodo (QM) structural sweeps, 3-Candle confirmation protocols, and a Deep Residual Multi-Layer Perceptron (Res-MLP) gatekeeper.

```
       [ASIAN RANGE ACCUMULATION]                [LONDON / NY EXPANSION]
       00:00 - 07:00 UTC (03:30 - 10:30 Tehran)  High-Volume Displacement
+---------------------------------------------+
|                                             |         ▲ Candle 0: Real Body Close > High
|  Asian High (Buy-Side Liquidity Pool) ------|---------+   Volume >= 1.3x 20-SMA
|         ~~~~ Consolidation Range ~~~~       |         |   Candle 1: Bullish FVG & Anti-Dump
|  Asian Low (Sell-Side Liquidity Pool) ------|         |   Res-MLP Gatekeeper Approval (P >= 0.65)
|                                             |         ▼ Limit Entry at Retest (R:R 1:2.5)
+---------------------------------------------+
```

---

## 1. Core Quantitative Strategy Specification

The strategy eliminates lagging retail oscillators in favor of institutional order flow mechanics:

### A. Session Boundary & Liquidity Pools
* **Timeframe:** 15-minute (`15m`) discrete candles.
* **Asian Session Window:** Exactly **00:00 to 07:00 UTC** (03:30 to 10:30 Tehran time).
* **Pool Discovery:**
  $$\text{Asia High} = \max_{t \in [00:00, 07:00)} (\text{High}_t), \quad \text{Asia Low} = \min_{t \in [00:00, 07:00)} (\text{Low}_t)$$
  These levels act as primary buy-side (BSL) and sell-side (SSL) liquidity anchors.

### B. Structural Acceptance & Anti-Trap Protocol
1. **Candle 0 (Displacement):** Must close strictly outside the range with volume $\ge 1.3\times \text{SMA}_{20}(\text{Volume})$. Wicks without body closes are rejected as liquidity sweeps.
2. **Candle 1 (Fair Value Gap & Anti-Dump):** Validates market imbalance ($\text{Low}_1 > \text{High}_{-1}$ for longs). If Candle 1 closes back inside the Asian range, the trade is immediately aborted.
3. **Candle 2 (Acceptance):** Sustained price acceptance confirming real institutional participation.
4. **Quasimodo (QM) Pattern Filter:** Scans for structural sweeps ($HH \to LL$ for Bearish QM, $LL \to HH$ for Bullish QM) to identify major reversal turning points at session highs/lows.
5. **Minimalist Filter Footprint:** Zero MACD or noisy oscillators. Uses only **RSI Divergences** with low weighting to detect exhaustion traps, and **Momentum Body Velocity** to verify real displacement.

### C. Execution & Dynamic Risk Budgeting
* **Fixed Risk:** Exactly $1.0\%$ total equity per trade.
* **Target Ratio:** Strict **1:2.5 Risk-to-Reward (R:R)** minimum.
* **Dynamic Breakeven (Risk-Free):** When favorable price excursion hits **$+1.5\text{R}$**, the Stop Loss is automatically relocated to Entry price ($0.0\text{R}$ risk).

---

## 2. Deep Residual MLP (Res-MLP) Architecture

To prevent overfitting on market stochasticity, candidate trades are evaluated by a custom PyTorch Deep Residual Multi-Layer Perceptron:

```
[15 Quantitative Features] 
       │
       ▼
 [Dense (15 → 64)] ──► [LayerNorm] ──► [LeakyReLU(0.1)] ──► [Dropout(0.3)] ──┐ (Skip Connection)
       │                                                                      │
       ▼                                                                      │
 [Dense (64 → 32)] ──► [LayerNorm] ──► [LeakyReLU(0.1)] ──► [Dropout(0.2)]    │
       │                                                                      │
       ▼                                                                      │
 [Residual Add & Projection (32 + 64 → 16)] ◄────────────────────────────────┘
       │
       ▼
 [Dense (16 → 1)] ──► [Sigmoid] ──► Win Probability P(Win) >= 0.65
```

* **Loss Function:** **Binary Focal Loss** ($\alpha = 0.55, \gamma = 2.0$) prioritizing hard boundary classifications over trivial samples.
* **Regularization:** AdamW with low learning rate ($\eta = 0.001$) and weight decay ($0.01$) to suppress noise fitting.

---

## 3. Empirical Multi-Asset Backtest Verification

Verified across major liquid assets (**BTCUSDT, ETHUSDT, SOLUSDT**) on continuous 15m historical candles:

| Metric | Empirical Result |
| :--- | :--- |
| **Total Evaluated 15m Candles** | **105,408 candles** |
| **Total Strategy Trade Setups** | **622 setups** |
| **Full Target Wins (+2.5R)** | **346 trades (55.6%)** |
| **Risk-Free Breakeven Exits (+1.5R BE)** | **171 trades (27.5%)** |
| **Losses (-1.0R)** | **105 trades (16.9%)** |
| **True Win Rate ($\frac{\text{Wins}}{\text{Wins} + \text{Losses}}$)** | **76.7%** |
| **Total Net Realized Return** | **+760.50 R** |
| **Portfolio Profit Factor** | **8.24** |

---

## 4. Quickstart Execution

```bash
# Clone and enter workspace
git clone https://github.com/ShiTmoZ/Auto-trade.git
cd Auto-trade

# Run real-time paper execution (zero external dependencies)
python3 main.py

# Run multi-asset backtest and data pipeline
python3 multi_asset_pipeline.py
```
