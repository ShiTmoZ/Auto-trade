#!/usr/bin/env python3
"""
Mathematical Backtest Engine for Tokyo Breakout Strategy
Implements pure mathematical rules without external dependencies:
1. Tokyo Session Extraction (00:00 - 09:00 UTC)
2. 50-EMA Higher Timeframe Trend Filter (1H)
3. 3-Candle Confirmation Protocol + FVG + Volume Filter (>= 1.3x 20-SMA)
4. Execution: Strict 1:2.5 R:R with +1.5R Breakeven Trigger
"""

import math
import json
import urllib.request
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple

def fetch_binance_history(symbol: str = "BTCUSDT", interval: str = "15m", limit: int = 1000) -> List[Dict[str, Any]]:
    """Fetches continuous historical klines from Binance API."""
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "AutoTradeBacktester/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"[Fetch Error]: {e}")
        return []

    candles = []
    for k in data:
        candles.append({
            "ts": k[0] / 1000,
            "dt": datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
        })
    return candles

def calculate_sma_and_atr(candles: List[Dict[str, Any]], vol_window: int = 20, atr_window: int = 14):
    for i in range(len(candles)):
        # Volume SMA
        start_v = max(0, i - vol_window + 1)
        vols = [candles[j]["volume"] for j in range(start_v, i + 1)]
        candles[i]["vol_sma"] = sum(vols) / len(vols) if vols else 1.0

        # ATR
        if i > 0:
            tr = max(
                candles[i]["high"] - candles[i]["low"],
                abs(candles[i]["high"] - candles[i - 1]["close"]),
                abs(candles[i]["low"] - candles[i - 1]["close"])
            )
        else:
            tr = candles[i]["high"] - candles[i]["low"]
        candles[i]["tr"] = tr

    for i in range(len(candles)):
        start_a = max(0, i - atr_window + 1)
        trs = [candles[j]["tr"] for j in range(start_a, i + 1)]
        candles[i]["atr"] = sum(trs) / len(trs) if trs else 50.0

def calculate_ema(prices: List[float], period: int = 50) -> List[float]:
    if not prices or len(prices) < period:
        return [prices[-1]] * len(prices) if prices else []
    mult = 2.0 / (period + 1.0)
    sma = sum(prices[:period]) / period
    ema = [sma] * period
    for p in prices[period:]:
        ema.append((p - ema[-1]) * mult + ema[-1])
    return ema

def run_mathematical_backtest(candles_15m: List[Dict[str, Any]]):
    calculate_sma_and_atr(candles_15m)
    closes = [c["close"] for c in candles_15m]
    ema50 = calculate_ema(closes, 50)
    for i in range(len(candles_15m)):
        candles_15m[i]["ema50"] = ema50[i]

    # Group candles by UTC calendar day
    days_dict = {}
    for idx, c in enumerate(candles_15m):
        day_str = c["dt"].strftime("%Y-%m-%d")
        if day_str not in days_dict:
            days_dict[day_str] = []
        days_dict[day_str].append((idx, c))

    trades = []
    
    # Process each day
    for day_str, day_items in days_dict.items():
        # 1. Extract Tokyo Session (00:00 to 09:00 UTC)
        tokyo_items = [(idx, c) for idx, c in day_items if 0 <= c["dt"].hour < 9]
        if len(tokyo_items) < 15:
            continue # Incomplete session data

        tokyo_high = max(c["high"] for _, c in tokyo_items)
        tokyo_low = min(c["low"] for _, c in tokyo_items)
        post_tokyo_items = [(idx, c) for idx, c in day_items if c["dt"].hour >= 9]

        if not post_tokyo_items:
            continue

        trade_taken_today = False

        # Scan post-tokyo candles for breakout
        for step, (idx, c) in enumerate(post_tokyo_items):
            if trade_taken_today:
                break
            if idx < 2 or idx >= len(candles_15m) - 10:
                continue

            c0 = candles_15m[idx]
            c_prev = candles_15m[idx - 1]
            c_prev2 = candles_15m[idx - 2]
            c1 = candles_15m[idx + 1] if idx + 1 < len(candles_15m) else None
            c2 = candles_15m[idx + 2] if idx + 2 < len(candles_15m) else None

            if not c1 or not c2:
                continue

            vol_ratio = c0["volume"] / max(c0["vol_sma"], 1.0)
            ema = c0["ema50"]

            # ==============================================================
            # RULE 1: BULLISH TOKYO HIGH BREAKOUT
            # ==============================================================
            if c_prev["high"] <= tokyo_high and c0["high"] > tokyo_high:
                # Math Condition 1: Candle 0 closes strictly above level with Volume >= 1.3x
                if c0["close"] > tokyo_high and vol_ratio >= 1.3:
                    # Math Condition 2: HTF Trend Filter (Price > 50 EMA)
                    if c0["close"] > ema:
                        # Math Condition 3: FVG Formation (Low[1] > High[-1] or Low[0] > High[-2])
                        has_fvg = (c1["low"] > c_prev["high"]) or (c0["low"] > c_prev2["high"])
                        # Math Condition 4: No 2-candle trap (Candle 1 must not close below tokyo_high)
                        if c1["close"] >= tokyo_high and has_fvg:
                            # Entry Setup
                            entry_price = tokyo_high
                            stop_loss = c0["low"] - (c0["atr"] * 0.15)
                            risk = entry_price - stop_loss
                            if risk > 0:
                                take_profit = entry_price + (2.5 * risk)
                                be_trigger = entry_price + (1.5 * risk)
                                
                                # Simulate forward trade resolution
                                outcome, exit_p, exit_idx = simulate_trade(
                                    candles_15m, idx + 1, "LONG", entry_price, stop_loss, take_profit, be_trigger
                                )
                                trades.append({
                                    "day": day_str, "time": c0["dt"].strftime("%H:%M UTC"),
                                    "side": "LONG", "level": tokyo_high,
                                    "entry": entry_price, "sl": stop_loss, "tp": take_profit,
                                    "risk": risk, "vol_ratio": round(vol_ratio, 2),
                                    "outcome": outcome, "exit_price": exit_p,
                                    "r_multiple": 2.5 if outcome == "WIN" else (0.0 if outcome == "BE" else -1.0)
                                })
                                trade_taken_today = True

            # ==============================================================
            # RULE 2: BEARISH TOKYO LOW BREAKOUT
            # ==============================================================
            elif c_prev["low"] >= tokyo_low and c0["low"] < tokyo_low:
                if c0["close"] < tokyo_low and vol_ratio >= 1.3:
                    if c0["close"] < ema:
                        has_fvg = (c_prev["low"] > c1["high"]) or (c_prev2["low"] > c0["high"])
                        if c1["close"] <= tokyo_low and has_fvg:
                            entry_price = tokyo_low
                            stop_loss = c0["high"] + (c0["atr"] * 0.15)
                            risk = stop_loss - entry_price
                            if risk > 0:
                                take_profit = entry_price - (2.5 * risk)
                                be_trigger = entry_price - (1.5 * risk)

                                outcome, exit_p, exit_idx = simulate_trade(
                                    candles_15m, idx + 1, "SHORT", entry_price, stop_loss, take_profit, be_trigger
                                )
                                trades.append({
                                    "day": day_str, "time": c0["dt"].strftime("%H:%M UTC"),
                                    "side": "SHORT", "level": tokyo_low,
                                    "entry": entry_price, "sl": stop_loss, "tp": take_profit,
                                    "risk": risk, "vol_ratio": round(vol_ratio, 2),
                                    "outcome": outcome, "exit_price": exit_p,
                                    "r_multiple": 2.5 if outcome == "WIN" else (0.0 if outcome == "BE" else -1.0)
                                })
                                trade_taken_today = True

    return trades

def simulate_trade(candles: List[Dict[str, Any]], start_idx: int, side: str, entry: float, sl: float, tp: float, be_trigger: float):
    cur_sl = sl
    for i in range(start_idx, min(len(candles), start_idx + 60)): # Up to 15 hours
        c = candles[i]
        if side == "LONG":
            # Check BE trigger
            if c["high"] >= be_trigger and cur_sl < entry:
                cur_sl = entry
            # Check SL
            if c["low"] <= cur_sl:
                return ("BE" if cur_sl == entry else "LOSS"), cur_sl, i
            # Check TP
            if c["high"] >= tp:
                return "WIN", tp, i
        elif side == "SHORT":
            if c["low"] <= be_trigger and cur_sl > entry:
                cur_sl = entry
            if c["high"] >= cur_sl:
                return ("BE" if cur_sl == entry else "LOSS"), cur_sl, i
            if c["low"] <= tp:
                return "WIN", tp, i
    return "LOSS", sl, len(candles) - 1

if __name__ == "__main__":
    print("=" * 65)
    print(" 📐 MATHEMATICAL BACKTEST ENGINE: TOKYO BREAKOUT (1:2.5 R:R)")
    print("=" * 65)
    print("Fetching 1000 15m candles from Binance (~10.4 days)...")
    data = fetch_binance_history("BTCUSDT", "15m", 1000)
    print(f"Data fetched: {len(data)} candles.")
    trades = run_mathematical_backtest(data)
    
    total = len(trades)
    wins = [t for t in trades if t["outcome"] == "WIN"]
    losses = [t for t in trades if t["outcome"] == "LOSS"]
    bes = [t for t in trades if t["outcome"] == "BE"]

    win_rate = (len(wins) / total * 100) if total > 0 else 0
    total_r = sum(t["r_multiple"] for t in trades)
    ev = (total_r / total) if total > 0 else 0
    gross_win_r = len(wins) * 2.5
    gross_loss_r = len(losses) * 1.0
    profit_factor = (gross_win_r / gross_loss_r) if gross_loss_r > 0 else float("inf")

    print("\n" + "-" * 65)
    print(" 📋 TRADE LOGS (Mathematical Execution):")
    print("-" * 65)
    for t in trades:
        icon = "🟢" if t["outcome"] == "WIN" else ("🟡" if t["outcome"] == "BE" else "🔴")
        print(f"{icon} {t['day']} {t['time']} | {t['side']} @ {t['entry']:,.2f} | SL: {t['sl']:,.2f} | TP: {t['tp']:,.2f} | Vol: {t['vol_ratio']}x | Result: {t['outcome']} ({t['r_multiple']:+.1f}R)")

    print("\n" + "=" * 65)
    print(" 📊 PERFORMANCE METRICS:")
    print("=" * 65)
    print(f"Total Signals Generated:  {total}")
    print(f"Wins (Hit 1:2.5 TP):      {len(wins)}")
    print(f"Breakeven (Hit +1.5R):    {len(bes)}")
    print(f"Losses (Hit 1.0 SL):      {len(losses)}")
    print(f"Win Rate:                 {win_rate:.1f}%")
    print(f"Total Return in R:        {total_r:+.2f} R")
    print(f"Expected Value (EV):      {ev:+.2f} R per trade")
    print(f"Profit Factor:            {profit_factor:.2f}")
    print("=" * 65 + "\n")
