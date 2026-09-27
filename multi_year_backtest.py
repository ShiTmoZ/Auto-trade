#!/usr/bin/env python3
"""
Multi-Year Historical Ingestion & Mathematical Backtest Engine (Binance Vision)
Streams raw monthly archives in-memory without saving heavy files to disk.
Evaluates Tokyo Breakout across 2020-2026.
"""

import sys
import io
import csv
import json
import zipfile
import urllib.request
from datetime import datetime, timezone
from typing import List, Dict, Any
from mathematical_backtest import calculate_sma_and_atr, calculate_ema, simulate_trade

def download_and_parse_month(year: int, month: int, symbol: str = "BTCUSDT") -> List[Dict[str, Any]]:
    month_str = f"{year}-{month:02d}"
    url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/15m/{symbol}-15m-{month_str}.zip"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    candles = []
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            z = zipfile.ZipFile(io.BytesIO(resp.read()))
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                reader = csv.reader(io.TextIOWrapper(f))
                for row in reader:
                    if not row[0].isdigit():
                        continue
                    candles.append({
                        "ts": int(row[0]) / 1000,
                        "dt": datetime.fromtimestamp(int(row[0]) / 1000, tz=timezone.utc),
                        "open": float(row[1]),
                        "high": float(row[2]),
                        "low": float(row[3]),
                        "close": float(row[4]),
                        "volume": float(row[5]),
                    })
    except Exception as e:
        # Month might not be published yet or network glitch
        pass
    return candles

def run_multi_year_backtest(start_year: int = 2023, end_year: int = 2026, symbol: str = "BTCUSDT"):
    print("=" * 70)
    print(f" 🌐 MULTI-YEAR STREAMING BACKTEST ({start_year} - {end_year}) | {symbol}")
    print("=" * 70)

    all_candles = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            # Skip future months in current year
            now = datetime.now(timezone.utc)
            if y == now.year and m > now.month:
                continue
            candles = download_and_parse_month(y, m, symbol)
            if candles:
                all_candles.extend(candles)
                print(f"  ▪ Loaded {y}-{m:02d}: {len(candles)} candles | Cumulative: {len(all_candles):,}", end="\r")

    print(f"\n✅ Total historical dataset loaded: {len(all_candles):,} 15m candles.")
    if not all_candles:
        print("No candles loaded.")
        return

    calculate_sma_and_atr(all_candles)
    closes = [c["close"] for c in all_candles]
    ema50 = calculate_ema(closes, 50)
    for i in range(len(all_candles)):
        all_candles[i]["ema50"] = ema50[i]

    # Group by calendar day
    days_dict = {}
    for idx, c in enumerate(all_candles):
        d_str = c["dt"].strftime("%Y-%m-%d")
        if d_str not in days_dict:
            days_dict[d_str] = []
        days_dict[d_str].append((idx, c))

    trades = []
    
    for day_str, day_items in days_dict.items():
        tokyo_items = [(idx, c) for idx, c in day_items if 0 <= c["dt"].hour < 9]
        if len(tokyo_items) < 15:
            continue

        tokyo_high = max(c["high"] for _, c in tokyo_items)
        tokyo_low = min(c["low"] for _, c in tokyo_items)
        post_tokyo_items = [(idx, c) for idx, c in day_items if c["dt"].hour >= 9]

        if not post_tokyo_items:
            continue

        trade_taken = False
        for step, (idx, c) in enumerate(post_tokyo_items):
            if trade_taken:
                break
            if idx < 2 or idx >= len(all_candles) - 10:
                continue

            c0 = all_candles[idx]
            c_prev = all_candles[idx - 1]
            c_prev2 = all_candles[idx - 2]
            c1 = all_candles[idx + 1] if idx + 1 < len(all_candles) else None
            c2 = all_candles[idx + 2] if idx + 2 < len(all_candles) else None

            if not c1 or not c2:
                continue

            vol_ratio = c0["volume"] / max(c0["vol_sma"], 1.0)
            ema = c0["ema50"]

            # Bullish Breakout
            if c_prev["high"] <= tokyo_high and c0["high"] > tokyo_high:
                if c0["close"] > tokyo_high and vol_ratio >= 1.3 and c0["close"] > ema:
                    has_fvg = (c1["low"] > c_prev["high"]) or (c0["low"] > c_prev2["high"])
                    if c1["close"] >= tokyo_high and has_fvg:
                        entry = tokyo_high
                        sl = c0["low"] - (c0["atr"] * 0.15)
                        risk = entry - sl
                        if risk > 0:
                            tp = entry + (2.5 * risk)
                            be = entry + (1.5 * risk)
                            outcome, exit_p, _ = simulate_trade(all_candles, idx + 1, "LONG", entry, sl, tp, be)
                            r_mult = 2.5 if outcome == "WIN" else (0.0 if outcome == "BE" else -1.0)
                            trades.append({
                                "date": day_str, "time": c0["dt"].strftime("%H:%M UTC"),
                                "side": "LONG", "entry": entry, "sl": sl, "tp": tp,
                                "vol_ratio": round(vol_ratio, 2), "outcome": outcome, "r": r_mult
                            })
                            trade_taken = True

            # Bearish Breakout
            elif c_prev["low"] >= tokyo_low and c0["low"] < tokyo_low:
                if c0["close"] < tokyo_low and vol_ratio >= 1.3 and c0["close"] < ema:
                    has_fvg = (c_prev["low"] > c1["high"]) or (c_prev2["low"] > c0["high"])
                    if c1["close"] <= tokyo_low and has_fvg:
                        entry = tokyo_low
                        sl = c0["high"] + (c0["atr"] * 0.15)
                        risk = sl - entry
                        if risk > 0:
                            tp = entry - (2.5 * risk)
                            be = entry - (1.5 * risk)
                            outcome, exit_p, _ = simulate_trade(all_candles, idx + 1, "SHORT", entry, sl, tp, be)
                            r_mult = 2.5 if outcome == "WIN" else (0.0 if outcome == "BE" else -1.0)
                            trades.append({
                                "date": day_str, "time": c0["dt"].strftime("%H:%M UTC"),
                                "side": "SHORT", "entry": entry, "sl": sl, "tp": tp,
                                "vol_ratio": round(vol_ratio, 2), "outcome": outcome, "r": r_mult
                            })
                            trade_taken = True

    # Performance Analysis
    total = len(trades)
    wins = [t for t in trades if t["outcome"] == "WIN"]
    losses = [t for t in trades if t["outcome"] == "LOSS"]
    bes = [t for t in trades if t["outcome"] == "BE"]
    win_rate = (len(wins) / total * 100) if total > 0 else 0.0
    total_r = sum(t["r"] for t in trades)
    profit_factor = (len(wins) * 2.5) / (len(losses) * 1.0) if len(losses) > 0 else float("inf")
    ev = total_r / total if total > 0 else 0.0

    report = f"""
# 📊 Tokyo Breakout Multi-Year Backtest Report ({start_year} - {end_year})

| Metric | Result |
| :--- | :--- |
| **Total Candles Analyzed** | {len(all_candles):,} |
| **Total Setups Taken** | {total} trades |
| **Winning Trades (+2.5R)** | {len(wins)} |
| **Breakeven Trades (0.0R)** | {len(bes)} |
| **Losing Trades (-1.0R)** | {len(losses)} |
| **Overall Win Rate** | **{win_rate:.1f}%** |
| **True Win Rate (Wins / (Wins+Losses))** | **{(len(wins)/(len(wins)+len(losses))*100 if len(wins)+len(losses)>0 else 0):.1f}%** |
| **Net Cumulative Return** | **{total_r:+.2f} R** |
| **Expected Value (EV)** | **{ev:+.2f} R per trade** |
| **Profit Factor** | **{profit_factor:.2f}** |

---
*Generated mathematically using strict 3-Candle Confirmation + FVG + Volume (1.3x) + 50-EMA Trend Filter.*
"""
    print("\n" + "=" * 70)
    print(report)
    print("=" * 70)

    with open("BACKTEST_REPORT.md", "w", encoding="utf-8") as f:
        f.write(report)
    with open("trades_backtest.json", "w", encoding="utf-8") as f:
        json.dump(trades, f, indent=2)

if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 2023
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 2026
    run_multi_year_backtest(start, end)
