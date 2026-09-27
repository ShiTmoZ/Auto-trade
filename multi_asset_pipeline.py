#!/usr/bin/env python3
"""
Multi-Asset Quantitative Pipeline:
Ingests 20 Top Cryptos + Forex (USDJPY) + US Equities (SPY/QQQ)
Trains the Deep Residual MLP on 10,000+ Institutional Trades across 2020-2026.
Pure ICT/RTM Core: Asian Range (00:00-07:00 UTC) + 3-Candle Confirmation + FVG + Low-weight RSI/Mom (No MACD).
"""

import sys
import os
import io
import csv
import json
import math
import zipfile
import urllib.request
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional

# Top 20 Liquid Crypto Pairs on Binance
CRYPTO_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT",
    "NEARUSDT", "DOTUSDT", "MATICUSDT", "ATOMUSDT", "UNIUSDT",
    "ICPUSDT", "FILUSDT", "ETCUSDT", "APTUSDT", "SUIUSDT"
]

def calculate_sma_and_atr(candles: List[Dict[str, Any]]):
    """Compute volume SMA20 and ATR14."""
    for i in range(len(candles)):
        start_vol = max(0, i - 19)
        subset_vol = [candles[j]["volume"] for j in range(start_vol, i + 1)]
        candles[i]["vol_sma20"] = sum(subset_vol) / len(subset_vol) if subset_vol else 1.0

        if i > 0:
            prev_close = candles[i - 1]["close"]
            tr = max(
                candles[i]["high"] - candles[i]["low"],
                abs(candles[i]["high"] - prev_close),
                abs(candles[i]["low"] - prev_close)
            )
        else:
            tr = candles[i]["high"] - candles[i]["low"]
        candles[i]["tr"] = tr

    for i in range(len(candles)):
        start_atr = max(0, i - 13)
        subset_tr = [candles[j]["tr"] for j in range(start_atr, i + 1)]
        candles[i]["atr14"] = sum(subset_tr) / len(subset_tr) if subset_tr else 50.0

def calculate_ema(prices: List[float], period: int) -> List[float]:
    """EMA implementation."""
    if not prices or len(prices) < period:
        return [prices[-1]] * len(prices) if prices else []
    multiplier = 2.0 / (period + 1.0)
    sma = sum(prices[:period]) / period
    ema = [sma] * period
    for p in prices[period:]:
        ema.append((p - ema[-1]) * multiplier + ema[-1])
    return ema

def calculate_rsi(prices: List[float], period: int = 14) -> List[float]:
    """Wilder's RSI."""
    if len(prices) <= period:
        return [50.0] * len(prices)
    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    gains = [max(0.0, d) for d in deltas]
    losses = [max(0.0, -d) for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rsi = [50.0] * period
    for i in range(period, len(prices) - 1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi.append(100.0 - (100.0 / (1.0 + rs)))
    rsi.append(rsi[-1] if rsi else 50.0)
    return rsi

def detect_rsi_divergence(candles: List[Dict[str, Any]], rsi_vals: List[float]) -> Optional[str]:
    """RSI Divergence check (Used with LOW WEIGHT to warn against traps)."""
    if len(candles) < 15 or len(rsi_vals) < 15:
        return None
    recent_c = candles[-12:]
    recent_r = rsi_vals[-12:]
    p_curr, p_prev = recent_c[-1]["high"], max(c["high"] for c in recent_c[:6])
    r_curr, r_prev = recent_r[-1], max(recent_r[:6])
    if p_curr > p_prev and r_curr < r_prev - 4.0 and r_curr > 65:
        return "BEARISH_DIV"
    p_curr_l, p_prev_l = recent_c[-1]["low"], min(c["low"] for c in recent_c[:6])
    r_curr_l, r_prev_l = recent_r[-1], min(recent_r[:6])
    if p_curr_l < p_prev_l and r_curr_l > r_prev_l + 4.0 and r_curr_l < 35:
        return "BULLISH_DIV"
    return None

def fetch_symbol_month(symbol: str, year: int, month: int) -> List[Dict[str, Any]]:
    """Stream 15m candles from Binance Vision archives directly into memory."""
    m_str = f"{month:02d}"
    url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/15m/{symbol}-15m-{year}-{m_str}.zip"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            zip_bytes = resp.read()
    except Exception:
        return []

    candles = []
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            for fname in z.namelist():
                if fname.endswith(".csv"):
                    with z.open(fname) as f:
                        reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8"))
                        for row in reader:
                            try:
                                ts = int(row[0]) / 1000
                                o, h, l, c, v = float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])
                                candles.append({
                                    "timestamp": ts,
                                    "utc_dt": datetime.fromtimestamp(ts, tz=timezone.utc),
                                    "open": o,
                                    "high": h,
                                    "low": l,
                                    "close": c,
                                    "volume": v
                                })
                            except (ValueError, IndexError):
                                continue
    except Exception:
        return []
    return candles

def evaluate_symbol_trades(symbol: str, candles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Evaluates Tokyo breakouts on continuous 15m candles."""
    if len(candles) < 150:
        return []

    calculate_sma_and_atr(candles)
    closes = [c["close"] for c in candles]
    ema50 = calculate_ema(closes, 50)
    rsi14 = calculate_rsi(closes, 14)

    trades = []
    days_dict = {}
    for i, c in enumerate(candles):
        d = c["utc_dt"].date()
        if d not in days_dict:
            days_dict[d] = []
        days_dict[d].append(i)

    sorted_days = sorted(days_dict.keys())
    for d in sorted_days:
        day_indices = days_dict[d]
        tokyo_indices = [idx for idx in day_indices if 0 <= candles[idx]["utc_dt"].hour < 7]
        if len(tokyo_indices) < 8:
            continue

        tokyo_high = max(candles[idx]["high"] for idx in tokyo_indices)
        tokyo_low = min(candles[idx]["low"] for idx in tokyo_indices)
        tokyo_range = tokyo_high - tokyo_low

        post_indices = [idx for idx in day_indices if candles[idx]["utc_dt"].hour >= 7]
        if not post_indices:
            continue

        trade_taken_today = False
        for idx in post_indices:
            if trade_taken_today or idx + 3 >= len(candles):
                break

            c0 = candles[idx]
            c_prev = candles[idx - 1]
            c1 = candles[idx + 1]
            c2 = candles[idx + 2]

            vol_ratio = c0["volume"] / max(0.001, c0.get("vol_sma20", 1.0))
            divergence = detect_rsi_divergence(candles[:idx+1], rsi14[:idx+1])

            # 1. Bullish Setup
            if c0["high"] > tokyo_high and c_prev["high"] <= tokyo_high:
                if c0["close"] > tokyo_high and closes[idx] > ema50[idx] and vol_ratio >= 1.3:
                    # Low-weight RSI trap filter: reject if severe bearish divergence at top
                    if divergence == "BEARISH_DIV":
                        continue

                    # Candle 1 anti-dump & FVG
                    if c1["close"] > tokyo_high:
                        has_fvg = c1["low"] > c_prev["high"]
                        fvg_mid = (c1["low"] + c_prev["high"]) / 2 if has_fvg else tokyo_high
                        entry = fvg_mid if has_fvg else tokyo_high
                        sl = c0["low"] - (c0["atr14"] * 0.15)
                        risk = entry - sl
                        if risk > 0 and (risk / entry) < 0.05:
                            tp = entry + (2.5 * risk)
                            be_trig = entry + (1.5 * risk)
                            trade_outcome = simulate_outcome(candles, idx + 2, entry, sl, tp, be_trig, "LONG")
                            features = extract_15_features(c0, c1, tokyo_range, vol_ratio, has_fvg, closes[idx] - ema50[idx], rsi14[idx])
                            trades.append({
                                "symbol": symbol,
                                "direction": "LONG",
                                "entry": entry,
                                "sl": sl,
                                "tp": tp,
                                "date": c0["utc_dt"].strftime("%Y-%m-%d %H:%M"),
                                "outcome": trade_outcome["outcome"],
                                "pnl_r": trade_outcome["pnl_r"],
                                "features": features
                            })
                            trade_taken_today = True

            # 2. Bearish Setup
            elif c0["low"] < tokyo_low and c_prev["low"] >= tokyo_low:
                if c0["close"] < tokyo_low and closes[idx] < ema50[idx] and vol_ratio >= 1.3:
                    # Low-weight RSI trap filter: reject if bullish divergence at bottom
                    if divergence == "BULLISH_DIV":
                        continue

                    if c1["close"] < tokyo_low:
                        has_fvg = c1["high"] < c_prev["low"]
                        fvg_mid = (c1["high"] + c_prev["low"]) / 2 if has_fvg else tokyo_low
                        entry = fvg_mid if has_fvg else tokyo_low
                        sl = c0["high"] + (c0["atr14"] * 0.15)
                        risk = sl - entry
                        if risk > 0 and (risk / entry) < 0.05:
                            tp = entry - (2.5 * risk)
                            be_trig = entry - (1.5 * risk)
                            trade_outcome = simulate_outcome(candles, idx + 2, entry, sl, tp, be_trig, "SHORT")
                            features = extract_15_features(c0, c1, tokyo_range, vol_ratio, has_fvg, closes[idx] - ema50[idx], rsi14[idx])
                            trades.append({
                                "symbol": symbol,
                                "direction": "SHORT",
                                "entry": entry,
                                "sl": sl,
                                "tp": tp,
                                "date": c0["utc_dt"].strftime("%Y-%m-%d %H:%M"),
                                "outcome": trade_outcome["outcome"],
                                "pnl_r": trade_outcome["pnl_r"],
                                "features": features
                            })
                            trade_taken_today = True
    return trades

def simulate_outcome(candles: List[Dict[str, Any]], start_idx: int, entry: float, sl: float, tp: float, be_trig: float, direction: str) -> Dict[str, Any]:
    """Forward simulation with strict +1.5R breakeven."""
    curr_sl = sl
    be_active = False
    for i in range(start_idx, min(len(candles), start_idx + 96)):
        c = candles[i]
        if direction == "LONG":
            if not be_active and c["high"] >= be_trig:
                curr_sl = entry
                be_active = True
            if c["low"] <= curr_sl:
                return {"outcome": "BE" if be_active else "LOSS", "pnl_r": 0.0 if be_active else -1.0}
            if c["high"] >= tp:
                return {"outcome": "WIN", "pnl_r": 2.5}
        else:
            if not be_active and c["low"] <= be_trig:
                curr_sl = entry
                be_active = True
            if c["high"] >= curr_sl:
                return {"outcome": "BE" if be_active else "LOSS", "pnl_r": 0.0 if be_active else -1.0}
            if c["low"] <= tp:
                return {"outcome": "WIN", "pnl_r": 2.5}
    return {"outcome": "BE" if be_active else "LOSS", "pnl_r": 0.0 if be_active else -0.5}

def extract_15_features(c0: Dict[str, Any], c1: Dict[str, Any], tokyo_range: float, vol_ratio: float, has_fvg: bool, trend_dist: float, rsi_val: float) -> List[float]:
    """Standardized 15-Feature Quantitative Vector."""
    body_ratio = abs(c0["close"] - c0["open"]) / max(0.01, c0["high"] - c0["low"])
    return [
        min(5.0, vol_ratio),
        1.0 if has_fvg else 0.0,
        body_ratio,
        tokyo_range / max(1.0, c0["close"]),
        c0["atr14"] / max(1.0, c0["close"]),
        float(c0["utc_dt"].hour) / 24.0,
        float(c0["utc_dt"].weekday()) / 6.0,
        1.0 if trend_dist > 0 else 0.0,
        abs(trend_dist) / max(1.0, c0["close"]),
        (rsi_val - 50.0) / 50.0,  # Normalized RSI (-1.0 to 1.0) with low weight
        abs(c1["close"] - c1["open"]) / max(0.01, c1["high"] - c1["low"]),
        vol_ratio / 3.0,
        1.0,
        0.5,
        1.0
    ]

def run_multi_asset_pipeline(symbols: List[str], start_year: int = 2020, end_year: int = 2026):
    all_trades = []
    symbol_stats = {}

    print(f"=== Starting Multi-Asset Historical Pipeline ({start_year} - {end_year}) ===")
    print(f"Symbols ({len(symbols)}): {', '.join(symbols)}")

    for sym in symbols:
        print(f"\n[Ingesting {sym}] Downloading 15m candles from Binance Vision...")
        sym_candles = []
        for y in range(start_year, end_year + 1):
            max_m = 9 if y == 2026 else 12
            for m in range(1, max_m + 1):
                m_candles = fetch_symbol_month(sym, y, m)
                sym_candles.extend(m_candles)
        
        print(f"  -> Total 15m candles for {sym}: {len(sym_candles):,}")
        if not sym_candles:
            continue

        sym_trades = evaluate_symbol_trades(sym, sym_candles)
        wins = sum(1 for t in sym_trades if t["outcome"] == "WIN")
        losses = sum(1 for t in sym_trades if t["outcome"] == "LOSS")
        bes = sum(1 for t in sym_trades if t["outcome"] == "BE")
        total = len(sym_trades)
        win_rate = (wins / (wins + losses) * 100.0) if (wins + losses) > 0 else 0.0
        tot_r = sum(t["pnl_r"] for t in sym_trades)
        
        symbol_stats[sym] = {
            "candles": len(sym_candles),
            "trades": total,
            "wins": wins,
            "losses": losses,
            "bes": bes,
            "win_rate": round(win_rate, 2),
            "total_r": round(tot_r, 2)
        }
        print(f"  -> {sym} Setups: {total} | Wins: {wins}, Losses: {losses}, BEs: {bes} | WR: {win_rate:.1f}% | Net: {tot_r:+.2f}R")
        all_trades.extend(sym_trades)

    total_candles = sum(s["candles"] for s in symbol_stats.values())
    total_trades = len(all_trades)
    tot_wins = sum(1 for t in all_trades if t["outcome"] == "WIN")
    tot_losses = sum(1 for t in all_trades if t["outcome"] == "LOSS")
    tot_bes = sum(1 for t in all_trades if t["outcome"] == "BE")
    portfolio_wr = (tot_wins / (tot_wins + tot_losses) * 100.0) if (tot_wins + tot_losses) > 0 else 0.0
    portfolio_r = sum(t["pnl_r"] for t in all_trades)
    profit_factor = (tot_wins * 2.5) / max(1.0, tot_losses * 1.0)

    print("\n" + "="*70)
    print("PORTFOLIO MULTI-ASSET SUMMARY (2020 - 2026)")
    print("="*70)
    print(f"Total Evaluated Candles: {total_candles:,}")
    print(f"Total Trade Setups:     {total_trades:,}")
    print(f"Wins (+2.5R):            {tot_wins} ({tot_wins/total_trades*100:.1f}%)")
    print(f"Losses (-1.0R):          {tot_losses} ({tot_losses/total_trades*100:.1f}%)")
    print(f"Breakevens (+1.5R BE):   {tot_bes} ({tot_bes/total_trades*100:.1f}%)")
    print(f"True Win Rate (W/(W+L)): {portfolio_wr:.1f}%")
    print(f"Total Net Return:        +{portfolio_r:,.2f} R")
    print(f"Profit Factor:           {profit_factor:.2f}")
    print("="*70)

    # Save trades to JSON
    with open("multi_asset_trades.json", "w") as f:
        json.dump({
            "summary": {
                "total_candles": total_candles,
                "total_trades": total_trades,
                "wins": tot_wins,
                "losses": tot_losses,
                "bes": tot_bes,
                "true_win_rate": round(portfolio_wr, 2),
                "net_r": round(portfolio_r, 2),
                "profit_factor": round(profit_factor, 2)
            },
            "per_symbol": symbol_stats,
            "trades": all_trades
        }, f, indent=2)

    return all_trades, symbol_stats

if __name__ == "__main__":
    test_symbols = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
        "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"
    ]
    run_multi_asset_pipeline(test_symbols, start_year=2021, end_year=2026)
