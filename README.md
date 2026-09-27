<p align="center">
  <img src="assets/banner.svg" alt="Institutional Tokyo Breakout" width="100%">
</p>

<p align="center">
  <a href="https://github.com/ShiTmoZ/Auto-trade/actions"><img src="https://img.shields.io/badge/GitHub_Actions-Multi--Asset_Pipeline-22c55e?style=for-the-badge&logo=githubactions&logoColor=white" alt="CI"></a>
  <a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-Residual_MLP-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <img src="https://img.shields.io/badge/AI_Win_Rate-49.5%25-00FFA3?style=for-the-badge" alt="AI Win Rate">
  <img src="https://img.shields.io/badge/Profit_Factor-1.79-FBBF24?style=for-the-badge" alt="Profit Factor">
  <img src="https://img.shields.io/badge/Net_Return-+1,204R-00F0FF?style=for-the-badge" alt="Net Return">
  <img src="https://img.shields.io/badge/Candles_Tested-4.75M-10B981?style=for-the-badge" alt="Candles">
</p>

---

## 🌟 Release v2.0: Quant-Hardened Reality Check

Following an in-depth quantitative methodology audit, version 2.0 strips away theoretical backtest inflation to model realistic institutional execution:

| Flaw / Dimension | v1.0 Legacy Architecture | v2.0 Quant-Hardened Reality |
| :--- | :--- | :--- |
| **Intra-bar Execution** | Optimistic: if candle touched +1.5R and SL, marked BE. | **Pessimistic Worst-Case:** assumes SL was triggered first. |
| **Limit Order Fill** | Phantom fill: assumed 100% fill at FVG median. | **Verified Fill:** requires subsequent bar low/high to touch entry. |
| **Exchange Friction** | Zero fees, zero slippage modeled. | **Deducted:** $-0.18\text{R}$ fixed taker fee per trade (Binance VIP0). |
| **Asset Universe** | 10 Cryptocurrencies (2021 – 2024). | **20 Top Cryptos + SPY, QQQ, USDJPY (2018 – 2026).** |
| **Candles Evaluated** | 1,401,860 candles (15m). | **4,753,163 candles (15m).** |
| **AI Feature Matrix** | Partial synthetic feature training. | **100% empirical feature extraction from real trade history.** |
| **Feature 14 Operator** | `high - low / atr` (division precedence bug). | Fixed: `(high - low) / atr`. |
| **True Performance** | Theoretical 74.4% WR / 7.26 PF (unrealistic). | **Sustainable 49.5% WR / 1.79 PF / +1,204R Net After Fees.** |

---

## 1. Executive Strategy Overview

The **Tokyo Breakout Quantitative Engine** is an institutional-grade algorithmic execution system. It rejects lagging retail indicators (MACD, Stochastics) in favor of **structural order flow liquidity**, capitalizing on the volatility expansion out of the **Asian Session (00:00 – 07:00 UTC / 03:30 – 10:30 Tehran)**.

<p align="center">
  <img src="assets/strategy_architecture.svg" alt="Strategy Pipeline Architecture" width="100%">
</p>

---

## 2. Quantitative Mechanics & Entry Protocols

### A. Asian Liquidity Discovery Window
* **Timeframe:** 15-minute (`15m`) continuous discrete candles.
* **Asian Session Accumulation:** Strictly bounded between **00:00 to 07:00 UTC** (03:30 to 10:30 Tehran time).
* **Liquidity Anchors:**
  $$\text{Asia High} = \max_{t \in [00:00, 07:00)} (\text{High}_t), \quad \text{Asia Low} = \min_{t \in [00:00, 07:00)} (\text{Low}_t)$$
  These levels act as primary buy-side (BSL) and sell-side (SSL) liquidity magnets during London and New York sessions.

### B. 3-Candle Confirmation & Anti-Trap Protocol
1. **Candle 0 (Real Displacement):** Must close strictly outside the range with volume $\ge 1.3\times \text{SMA}_{20}(\text{Volume})$. Pure wicks without body closes are rejected as liquidity sweeps.
2. **Candle 1 (Fair Value Gap & Anti-Dump):** Validates market imbalance ($\text{Low}_1 > \text{High}_{-1}$ for longs). If Candle 1 closes back inside the Asian range, the trade is aborted immediately.
3. **Candle 2 (Acceptance):** Sustained price acceptance confirming real institutional participation.
4. **Quasimodo (QM) Structural Sweeps (`qm_detector.py`):** Scans for institutional reversal turning points ($HH \to LL$ for Bearish QM, $LL \to HH$ for Bullish QM).
5. **Minimalist Technical Filter (`technical_indicators.py`):** Zero MACD or noisy oscillators. Employs only **RSI Divergences** with low weighting to detect exhaustion traps, and **Momentum Body Velocity** to verify real displacement.

### C. Execution & Dynamic Risk Budgeting
* **Fixed Capital Risk:** Exactly $1.0\%$ total equity per trade.
* **Target Ratio:** Strict **1:2.5 Risk-to-Reward (R:R)** minimum.
* **Dynamic Breakeven:** When favorable excursion hits **$+1.5\text{R}$**, the Stop Loss is automatically relocated to Entry price ($0.0\text{R}$ risk).

---

## 3. Deep Residual MLP (Res-MLP) Architecture & Self-Improvement

<p align="center">
  <img src="assets/neural_net_arch.svg" alt="Deep Residual MLP Architecture" width="100%">
</p>

### A. Dataset & Training Scope
* **Training Corpus:** **4,753,163 real 15-minute candles** across 2018 through 2026.
* **Multi-Asset Universe:** Top 20 high-liquidity cryptocurrency pairs (`BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `BNBUSDT`, `XRPUSDT`, `DOGEUSDT`, `ADAUSDT`, `AVAXUSDT`, `LINKUSDT`, `LTCUSDT`, `NEARUSDT`, `DOTUSDT`, `MATICUSDT`, `ATOMUSDT`, `UNIUSDT`, `ICPUSDT`, `FILUSDT`, `ETCUSDT`, `APTUSDT`, `ARBUSDT`) + US Equities (`SPY`, `QQQ`) & Forex (`USDJPY`).
* **Feature Vector:** 15 normalized quantitative metrics (Volume ratios, FVG depth, HTF directional alignment, ATR-relative distance to Asian liquidity, body velocity, RSI momentum, and QM sweep indicators).

### B. Mathematical Specifications
* **Residual Skip Connection:** A linear projection connects the 1st dense representation directly to the 3rd layer, ensuring critical price and volume gradients remain uncorrupted across deep transformations.
* **Binary Focal Loss:** Penalizes difficult boundary cases and suppresses trivial market noise:
  $$\mathcal{L}_{\text{Focal}} = -\alpha (1 - p_t)^\gamma \log(p_t), \quad (\alpha=0.55, \gamma=2.0)$$
* **Regularization:** AdamW with low learning rate ($\eta = 0.001$) and weight decay ($0.01$) to prevent memorization of historical stochasticity.

### C. Continuous Self-Improvement Loop
1. **Online Micro-Adaptation (`ml_gatekeeper.update_online`):** As live paper or real trades conclude, verified outcomes execute a micro-step gradient adjustment $(\eta_{\text{online}} = 0.0005)$ to adjust confidence scores to prevailing volatility regimes without catastrophic forgetting.
2. **Scheduled Automated Retraining:** Continuous integration pipelines regularly ingest the latest market months via GitHub Actions to refresh baseline model weights.

---

## 4. Multi-Asset Empirical Backtest Verification (2018 – 2026)

### Quant-Hardened Methodology:
* **Worst-Case Intra-Bar Execution:** If a 15-minute candle penetrates both Stop Loss and Breakeven/Take Profit, the engine pessimistically marks Stop Loss hit first.
* **Phantom Limit Fill Elimination:** Trades only trigger if post-breakout candles empirically pullback to the FVG median within 3 bars.
* **Realistic Fee & Slippage Friction:** Fixed **$-0.18\text{R}$ transaction fee** deducted from every trade ($0.08\%$ round-trip Binance VIP0 taker commission).

Evaluated across **20 liquid cryptocurrency assets** over **4,753,163 candles (15m)**:

| Metric | Raw Price Action Setup | Res-MLP AI Gatekeeper Filtered |
| :--- | :---: | :---: |
| **Total Candidates Evaluated** | 12,567 Setups | **2,960 High-Conviction Setups** |
| **Full Wins (+2.32R Net)** | 3,723 (29.6%) | **1,172 (39.6%)** |
| **Breakeven (-0.18R Net Fee)** | 2,123 (16.9%) | **594 (20.1%)** |
| **Losses (-1.18R Net)** | 6,721 (53.5%) | **1,194 (40.3%)** |
| **True Win Rate ($\frac{\text{Wins}}{\text{Wins} + \text{Losses}}$)** | 35.65% | **49.54% (~50%)** |
| **Net Realized Return (R)** | +331.44 R | **+1,204.20 R (Net After Fees)** |
| **Portfolio Profit Factor** | 1.04 | **1.79** |

### Key Institutional Takeaway:
* Without machine learning filtering, mechanical breakouts hover near break-even (Profit Factor 1.04).
* The **Res-MLP Gatekeeper** filters out 76% of false breakouts, elevating the net win rate to **49.54%** which, combined with a **1:2.5 Risk-to-Reward ratio**, generates a healthy and institutionally sustainable **+1,204.20 R net return**.

---

## 5. Quickstart & Deployment

```bash
# Clone repository
git clone https://github.com/ShiTmoZ/Auto-trade.git
cd Auto-trade

# Run real-time paper execution (zero external dependencies)
python3 main.py

# Run multi-asset backtest and data pipeline
python3 multi_asset_pipeline.py
```

---

## 6. Risk Disclaimer

> ⚠️ **IMPORTANT NOTICE:** All financial and cryptocurrency trading involves substantial risk of capital loss. This quantitative framework, backtest statistics, and machine learning models are provided strictly for educational, scientific, and research purposes. Historical performance and simulated backtest results are no guarantee of future returns. The entire financial risk of deployment, execution, and capital allocation rests solely and exclusively with the user.
