"""
Breakout Validator: Implements the 3-Candle Confirmation Protocol, FVG Detection, and Entry/SL/TP Setup
"""

from typing import List, Dict, Any, Optional

def _smart_round(val: float) -> float:
    if abs(val) < 1.0:
        return round(val, 5)
    elif abs(val) < 20.0:
        return round(val, 3)
    else:
        return round(val, 2)

class BreakoutValidator:
    def __init__(self, min_volume_ratio: float = 1.3, min_rr_ratio: float = 2.5):
        self.min_volume_ratio = min_volume_ratio
        self.min_rr_ratio = min_rr_ratio

    def evaluate_breakout(
        self,
        candles_15m: List[Dict[str, Any]],
        level: float,
        level_type: str, # 'TOKYO_HIGH' or 'TOKYO_LOW'
        htf_bias: str
    ) -> Optional[Dict[str, Any]]:
        """
        Scans recent 15m candles to find if a breakout event has satisfied
        the 3-Candle Confirmation Protocol.
        """
        if len(candles_15m) < 10:
            return None

        # Look for the breakout initiation candle (Candle 0) within the last 5 closed candles
        break_idx = None
        for i in range(len(candles_15m) - 2, max(0, len(candles_15m) - 8), -1):
            c = candles_15m[i]
            c_prev = candles_15m[i - 1]
            
            if level_type == "TOKYO_HIGH":
                if c["high"] > level and c_prev["high"] <= level:
                    break_idx = i
                    break
            elif level_type == "TOKYO_LOW":
                if c["low"] < level and c_prev["low"] >= level:
                    break_idx = i
                    break

        if break_idx is None:
            return None

        c0 = candles_15m[break_idx]
        c_prev = candles_15m[break_idx - 1]
        c1 = candles_15m[break_idx + 1] if break_idx + 1 < len(candles_15m) else None
        c2 = candles_15m[break_idx + 2] if break_idx + 2 < len(candles_15m) else None

        vol_ratio = c0["volume"] / c0.get("vol_sma20", 1.0) if c0.get("vol_sma20", 0) > 0 else 1.0

        # ==============================================================
        # 1. BULLISH BREAKOUT (TOKYO HIGH)
        # ==============================================================
        if level_type == "TOKYO_HIGH":
            if htf_bias != "BULLISH":
                return None # Reject if HTF trend is not aligned

            # Candle 0: Must close strictly above Tokyo High with volume
            if c0["close"] <= level or vol_ratio < self.min_volume_ratio:
                return None # Sweep or lack of volume

            # Candle 1: Check for FVG and ensure it didn't dump back into range
            has_fvg = False
            fvg_size = 0.0
            fvg_midpoint = level
            if c1:
                # Bullish FVG between Candle 1 Low and Candle -1 High
                if c1["low"] > c_prev["high"]:
                    has_fvg = True
                    fvg_size = c1["low"] - c_prev["high"]
                    fvg_midpoint = (c1["low"] + c_prev["high"]) / 2.0
                
                # Check rejection: If candle 1 closed below level -> 2-candle bull trap
                if c1["is_closed"] and c1["close"] < level:
                    return None

            # Candle 2: Sustained acceptance check
            if c2 and c2["is_closed"] and c2["close"] < level:
                return None

            # Setup Entry, Stop Loss, and Take Profit
            entry_price = max(level, fvg_midpoint)
            stop_loss = c0["low"] - (c0.get("atr14", 50.0) * 0.15) # Buffer below break candle
            risk_dist = entry_price - stop_loss
            if risk_dist <= 0:
                return None
                
            take_profit = entry_price + (risk_dist * self.min_rr_ratio)

            return {
                "signal_type": "LONG",
                "level_broken": level,
                "level_type": level_type,
                "breakout_time": c0["utc_dt"].strftime("%Y-%m-%d %H:%M UTC"),
                "entry_price": _smart_round(entry_price),
                "stop_loss": _smart_round(stop_loss),
                "take_profit": _smart_round(take_profit),
                "risk_distance": _smart_round(risk_dist),
                "rr_ratio": self.min_rr_ratio,
                "vol_ratio": round(vol_ratio, 2),
                "has_fvg": has_fvg,
                "fvg_size": _smart_round(fvg_size),
                "candle0_body": _smart_round(abs(c0["close"] - c0["open"])),
                "atr14": _smart_round(c0.get("atr14", 50.0))
            }

        # ==============================================================
        # 2. BEARISH BREAKOUT (TOKYO LOW)
        # ==============================================================
        elif level_type == "TOKYO_LOW":
            if htf_bias != "BEARISH":
                return None # Reject if HTF trend is not aligned

            # Candle 0: Must close strictly below Tokyo Low with volume
            if c0["close"] >= level or vol_ratio < self.min_volume_ratio:
                return None # Sweep or lack of volume

            # Candle 1: Check for Bearish FVG and ensure it didn't surge back into range
            has_fvg = False
            fvg_size = 0.0
            fvg_midpoint = level
            if c1:
                # Bearish FVG between Candle -1 Low and Candle 1 High
                if c_prev["low"] > c1["high"]:
                    has_fvg = True
                    fvg_size = c_prev["low"] - c1["high"]
                    fvg_midpoint = (c_prev["low"] + c1["high"]) / 2.0

                if c1["is_closed"] and c1["close"] > level:
                    return None # 2-candle bear trap

            # Candle 2: Sustained acceptance
            if c2 and c2["is_closed"] and c2["close"] > level:
                return None

            entry_price = min(level, fvg_midpoint)
            stop_loss = c0["high"] + (c0.get("atr14", 50.0) * 0.15)
            risk_dist = stop_loss - entry_price
            if risk_dist <= 0:
                return None
                
            take_profit = entry_price - (risk_dist * self.min_rr_ratio)

            return {
                "signal_type": "SHORT",
                "level_broken": level,
                "level_type": level_type,
                "breakout_time": c0["utc_dt"].strftime("%Y-%m-%d %H:%M UTC"),
                "entry_price": _smart_round(entry_price),
                "stop_loss": _smart_round(stop_loss),
                "take_profit": _smart_round(take_profit),
                "risk_distance": _smart_round(risk_dist),
                "rr_ratio": self.min_rr_ratio,
                "vol_ratio": round(vol_ratio, 2),
                "has_fvg": has_fvg,
                "fvg_size": _smart_round(fvg_size),
                "candle0_body": _smart_round(abs(c0["close"] - c0["open"])),
                "atr14": _smart_round(c0.get("atr14", 50.0))
            }

        return None
