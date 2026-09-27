<p align="center">
  <img src="assets/banner.svg" alt="Institutional Tokyo Breakout" width="100%">
</p>

<p align="center">
  <a href="https://github.com/ShiTmoZ/Auto-trade/actions"><img src="https://img.shields.io/badge/GitHub_Actions-Multi--Asset_Pipeline-22c55e?style=for-the-badge&logo=githubactions&logoColor=white" alt="CI"></a>
  <a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-Residual_MLP_(v3.1)-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <img src="https://img.shields.io/badge/Base_Win_Rate-35.6%25-blue?style=for-the-badge" alt="Base Win Rate">
  <img src="https://img.shields.io/badge/Calibrated_WR-41%25--44%25-00FFA3?style=for-the-badge" alt="AI Win Rate">
  <img src="https://img.shields.io/badge/Net_Return-+1,177R-00F0FF?style=for-the-badge" alt="Net Return">
  <img src="https://img.shields.io/badge/Candles_Tested-4.75M-10B981?style=for-the-badge" alt="Candles">
</p>

---

## 🌟 Release v3.1: Genuine Res-MLP, Calibration Hardening & Zero-PyTorch Runtime

Following a rigorous quantitative audit and multi-asset cloud pipeline execution:
* **Genuine Deep Residual MLP:** Implemented mathematical feedforward matching PyTorch's `ResidualMLP` directly inside `ml_gatekeeper.py` using pure standard Python/NumPy (4,881 trainable parameters). Live production environments run with zero heavy deep-learning runtime dependencies.
* **Best-Val Loss Checkpointing:** Retraining exports the exact snapshot of minimal validation loss (`best_val_loss = 0.0760` at epoch 10) instead of the final overfitted epoch.
* **Persistent Z-Score Normalization:** Full preservation of 15-dimensional empirical feature mean ($\mu$) and standard deviation ($\sigma$) across server restarts, preventing unscaled feature leakage into non-linear activations.
* **Online Degradation Protection:** Disabled destructive online SGD updates on fallback logistic weights while Res-MLP acts as the authoritative gatekeeper.
* **Branch-Aware CI/CD Pipeline:** GitHub Actions automatically commits training weights and backtest artifacts directly to the active working branch (`github.ref_name`).

---

## 🌟 Release v2.0 & v2.1: Institutional Execution Engine

* **Worst-Case Intra-Bar Execution:** If a 15-minute bar touches both SL and +1.5R BE / 2.5R TP, pessimistic SL execution is enforced.
* **Verified Limit Fill:** Post-breakout trades require subsequent bar range penetration into the FVG median within 3 bars.
* **Friction & Exchange Fees:** $-0.18\text{R}$ fixed taker fee per round-trip deducted from every trade (Binance VIP0 fee modeling).
* **Parallel Universe Ingestion:** Concurrent 15m candle polling across 20 Binance pairs using `ThreadPoolExecutor` (reduced cycle time from 75s to ~3.2s).
* **Sub-Cent Floating Precision:** 5-decimal precision for low-priced assets (`ARBUSDT`, `ADAUSDT`).
* **Dynamic Level Registry:** Thread-safe level tracking preventing duplicate intraday triggers.

---

## 1. Executive Strategy Overview

The **Tokyo Breakout Quantitative Engine** is an institutional-grade algorithmic execution system. It rejects lagging retail indicators (MACD, Stochastics) in favor of **structural order flow liquidity**, capitalizing on volatility expansion out of the **Asian Session (00:00 – 07:00 UTC / 03:30 – 10:30 Tehran)**.

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
  These levels act as primary buy-side (BSL) and sell-side (SSL) liquidity pools during London and New York sessions.

### B. 3-Candle Confirmation & Anti-Trap Protocol
1. **Candle 0 (Real Displacement):** Must close strictly outside the range with volume $\ge 1.3\times \text{SMA}_{20}(\text{Volume})$. Pure wicks without body closes are rejected as liquidity sweeps.
2. **Candle 1 (Fair Value Gap & Anti-Dump):** Validates market imbalance ($\text{Low}_1 > \text{High}_{-1}$ for longs). If Candle 1 closes back inside the Asian range, the trade is aborted immediately.
3. **Candle 2 (Acceptance):** Sustained price acceptance confirming real institutional participation.
4. **Quasimodo (QM) Structural Sweeps (`qm_detector.py`):** Scans for institutional reversal turning points ($HH \to LL$ for Bearish QM, $LL \to HH$ for Bullish QM).
5. **Minimalist Technical Filter (`technical_indicators.py`):** Employs **RSI Divergences** with low weighting to detect exhaustion traps, and **Momentum Body Velocity** to verify real displacement.

### C. Execution & Dynamic Risk Budgeting
* **Fixed Capital Risk:** Exactly $1.0\%$ total equity per trade.
* **Target Ratio:** Strict **1:2.5 Risk-to-Reward (R:R)** minimum (Net $+2.32\text{R}$ after fees).
* **Dynamic Breakeven:** When favorable excursion hits **$+1.5\text{R}$**, the Stop Loss is automatically relocated to Entry price ($-0.18\text{R}$ fee risk).

---

## 3. Deep Residual MLP (Res-MLP) Architecture

<p align="center">
  <img src="assets/neural_net_arch.svg" alt="Deep Residual MLP Architecture" width="100%">
</p>

### A. Mathematical Topology (4,881 Parameters)
$$\mathbf{h}_1 = \text{LeakyReLU}_{0.1}\Big(\mathbf{W}_1 \mathbf{x}_{\text{norm}} + \mathbf{b}_1\Big) \quad [15 \to 64]$$
$$\mathbf{a}_1 = \text{LayerNorm}(\mathbf{h}_1)$$
$$\mathbf{h}_2 = \text{LeakyReLU}_{0.1}\Big(\text{LayerNorm}\big(\mathbf{W}_2 \mathbf{a}_1 + \mathbf{b}_2\big)\Big) \quad [64 \to 32]$$
$$\mathbf{s} = \mathbf{W}_{\text{skip}} \mathbf{a}_1 + \mathbf{b}_{\text{skip}} \quad [64 \to 16]$$
$$\mathbf{h}_3 = \text{LeakyReLU}_{0.1}\Big(\mathbf{W}_3 \mathbf{h}_2 + \mathbf{b}_3\Big) + \mathbf{s} \quad [32 \to 16 + \text{skip}]$$
$$\hat{y} = \sigma\Big(\mathbf{W}_{\text{head}} \mathbf{h}_3 + b_{\text{head}}\Big) \quad [16 \to 1]$$

* **Zero-PyTorch Inference:** Forward pass computed in pure NumPy / native arrays via `ml_gatekeeper.py` reading calibrated layer matrices from `res_mlp_weights.json`.
* **Objective:** Binary Focal Loss $(\alpha=0.55, \gamma=2.0)$ targeting profitable trade outcomes ($pnl\_r > 0$).
* **Optimizer:** AdamW ($\text{LR}=0.001$, Weight Decay $= 0.01$).

### B. 15-Feature Input Vector
1. `vol_ratio`: Volume expansion relative to 20-period SMA
2. `has_fvg`: Presence of Fair Value Gap on breakout
3. `fvg_size_pct`: Depth of Fair Value Gap imbalance
4. `trend_alignment`: Higher timeframe directional trend bias
5. `risk_to_atr`: Stop distance relative to 14-period ATR
6. `body_to_range`: Breakout candle body displacement ratio
7. `session_time`: Temporal progress through trading session
8. `asian_range_pct`: Asian consolidation width as % of price
9. `breakout_speed`: Velocity of range penetration
10. `retest_depth`: Pullback ratio relative to breakout level
11. `body_velocity`: Momentum body ratio of breakout bar
12. `adx_strength`: Directional trend strength
13. `atr_ratio`: Current ATR relative to baseline volatility
14. `qm_sweep`: Quasimodo structural liquidity sweep flag
15. `bias_const`: Calibrated constant baseline input

---

## 4. Multi-Asset Empirical Backtest Verification (2018 – 2026)

Evaluated across **4,753,163 candles (15m)** and **20 liquid crypto assets** over 8 years:

### A. Baseline vs. Calibrated Res-MLP Performance

| Metric | Raw ICT Rule Set (No ML) | Res-MLP Gatekeeper (Thr $\ge 0.45$) | Res-MLP Gatekeeper (Thr $\ge 0.50$) |
| :--- | :---: | :---: | :---: |
| **Total Setups Evaluated** | 12,567 | **2,877 (22.9%)** | **1,488 (11.8%)** |
| **Winning Trades (+2.32R)** | 3,723 (29.6%) | **1,142 (39.7%)** | **654 (44.0%)** |
| **Breakeven Trades (-0.18R)** | 2,123 (16.9%) | 557 (19.4%) | 291 (19.6%) |
| **Losing Trades (-1.18R)** | 6,721 (53.5%) | 1,178 (40.9%) | 543 (36.5%) |
| **True Win Rate ($\frac{\text{W}}{\text{W}+\text{L}}$)** | **35.65%** | **49.2%** | **54.6%** |
| **Nominal Win Rate ($\frac{\text{W}}{\text{Total}}$)** | 29.6% | **39.7%** | **44.0%** |
| **Total Realized Return (R)** | +331.44 R | **+1,177.60 R** | **+828.20 R** |
| **Avg Return Per Setup** | +0.026 R | **+0.409 R** | **+0.557 R** |
| **Profit Factor** | 1.04 | **1.62** | **1.78** |

### B. Out-of-Sample Symbol Validation (2,514 Setups)
Evaluated on out-of-sample assets (`APT`, `ARB`, `ETC`, `FIL`, `ICP`, `UNI`):
* **Baseline (Unfiltered):** 29.3% Win Rate | $+0.011\text{R}$ per trade
* **Gatekeeper Threshold 0.45:** **40.7% Win Rate** | **$+0.455\text{R}$ per trade** (+260.9R net across 573 setups)
* **Gatekeeper Threshold 0.50:** **43.5% Win Rate** | **$+0.559\text{R}$ per trade** (+154.3R net across 276 setups)
* **Model AUC:** **0.574** (Monotonically rising positive expectancy across confidence thresholds)

---

## 5. Quickstart & Deployment

```bash
# Clone repository
git clone https://github.com/ShiTmoZ/Auto-trade.git
cd Auto-trade

# Switch to the hardened v3.1 branch
git checkout fix/v3.1-mlp-checkpoint-and-calibration

# Run real-time execution engine
python3 main.py

# Run cloud multi-asset training & backtest locally
python3 multi_asset_pipeline.py
python3 pipeline_train.py
```

---

## 6. Risk Disclaimer

> ⚠️ **IMPORTANT NOTICE:** All financial and cryptocurrency trading involves substantial risk of capital loss. This quantitative framework, backtest statistics, and machine learning models are provided strictly for educational, scientific, and research purposes. Historical performance and simulated backtest results are no guarantee of future returns. The entire financial risk of deployment, execution, and capital allocation rests solely and exclusively with the user.
