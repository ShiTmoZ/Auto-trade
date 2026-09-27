# Complete Quantitative System Prompt: Institutional Tokyo Breakout Bot

You can use the following comprehensive specification prompt with advanced coding models (such as Claude Sonnet 4.6 or DeepSeek V4.1 Flash via Cline) to reproduce, extend, or benchmark this quantitative trading system.

---

```markdown
Role: Quantitative Software Architect & Quantitative Trader
Task: Build an institutional-grade, zero-dependency algorithmic execution and machine learning validation system for BTC/USDT futures on Binance.

================================================================================
CORE STRATEGY SPECIFICATION (ICT / RTM ASIAN RANGE EXPANSION)
================================================================================
1. Session Timing & Liquidity Discovery:
   - Timeframe: 15-minute (15m) execution candles.
   - Asian / Tokyo Session Window: Exactly 00:00 UTC to 09:00 UTC daily.
   - Extract Tokyo High (TH = max(High[t])) and Tokyo Low (TL = min(Low[t])) for t in [00:00, 09:00).
   - Dynamic Mitigation State: TH and TL remain active liquidity boundaries until mitigated (touched/swept) by subsequent post-09:00 UTC candles.

2. Higher Timeframe (HTF) Trend Gatekeeper:
   - Calculate 50-period Exponential Moving Average (EMA) on 15m/1H data.
   - Directional Bias:
     * BULLISH if Current Close > 50-EMA. Allow LONG signals only.
     * BEARISH if Current Close < 50-EMA. Allow SHORT signals only.
   - Prune any counter-trend breakout immediately before validation.

3. The Strict Three-Candle Structural Acceptance Protocol:
   - Candle 0 (Displacement / Penetration):
     * Long: High[0] > TH and Close[0] > TH (Real body close, not a wick sweep).
     * Short: Low[0] < TL and Close[0] < TL.
     * Volume Filter: Volume[0] >= 1.3 * SMA_20(Volume).
   - Candle 1 (Fair Value Gap & Anti-Trap):
     * Imbalance Detection (Bullish FVG: Low[1] > High[-1]; Bearish FVG: High[1] < Low[-1]).
     * 2-Candle Trap Guard: If Candle 1 dumps back inside the Tokyo range (Close[1] <= TH for long or Close[1] >= TL for short), invalidate immediately.
   - Candle 2 (Acceptance Confirmation):
     * Must maintain sustained price acceptance outside the range.

4. Execution Mechanics & Dynamic Risk Budgeting:
   - Fixed Risk Model: Exactly 1.0% of total equity at risk per execution.
   - Order Entry: Limit retest at TH/TL boundary or 50% midpoint of the FVG.
   - Structural Stop Loss: Below Low[0] - (0.15 * ATR14) for Long; Above High[0] + (0.15 * ATR14) for Short.
   - Target Profit (TP): Fixed 1:2.5 Risk-to-Reward ratio minimum.
   - Dynamic Breakeven Trigger: Once price reaches +1.5R favorable displacement, automatically move Stop Loss to Entry price (0.0R risk).

5. Machine Learning Entry Gatekeeper (Res-MLP):
   - Extract a 15-dimensional quantitative feature vector per candidate trade:
     [vol_ratio, fvg_size_pct, has_fvg, body_ratio, risk_atr_ratio, tokyo_range_pct,
      utc_hour, day_of_week, htf_alignment, retest_depth, volatility, dist_from_open,
      momentum_3c, consecutive_run, expansion_ratio]
   - Neural Architecture: Deep Residual MLP with LayerNorm, Dropout(0.3/0.2), and a linear skip connection from Layer 1 to Layer 3.
   - Objective: Binary Focal Loss (alpha=0.55, gamma=2.0) to prioritize hard-to-predict tails over naive noise.
   - Optimization: AdamW with low learning rate (0.001) and strict weight decay (0.01) to resist overfitting.
   - Approval Gate: Approve execution only if Win Probability >= 0.65.

6. Autonomous Post-Mortem AI Reviewer:
   - Query recent stopped-out trades from SQLite database.
   - Submit failure telemetry to Gemini Flash (low reasoning) to identify macro regime changes or news spikes.
   - Store feedback in the database to adjust feature weighting.

7. Architecture & Code Constraints:
   - Self-contained, modular Python code (Standard Library first: urllib, json, sqlite3, math).
   - Network resilience: Automatic fallback across Binance endpoints (data-api.binance.vision, api1..api3) and configurable SOCKS5/HTTP proxy support.
   - Persistent SQLite trade journal (`trades.db`).
================================================================================
```
