"""
Quasimodo (QM) Pattern Detector (RTM - Read The Market)
Mathematically identifies Bullish and Bearish QM formations:
- Bearish QM: High -> Low -> Higher High (Liquidity Sweep) -> Lower Low (BOS) -> Retest Left Shoulder
- Bullish QM: Low -> High -> Lower Low (Liquidity Sweep) -> Higher High (BOS) -> Retest Left Shoulder
"""

from typing import List, Dict, Any, Optional

class QMDetector:
    def __init__(self, swing_window: int = 3):
        self.swing_window = swing_window

    def find_swings(self, candles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify fractal swing highs and swing lows."""
        swings = []
        w = self.swing_window
        n = len(candles)
        
        for i in range(w, n - w):
            c = candles[i]
            # Swing High
            is_high = all(candles[i]["high"] >= candles[i - j]["high"] for j in range(1, w + 1)) and \
                      all(candles[i]["high"] >= candles[i + j]["high"] for j in range(1, w + 1))
            if is_high:
                swings.append({
                    "type": "HIGH",
                    "idx": i,
                    "price": c["high"],
                    "dt": c["utc_dt"]
                })

            # Swing Low
            is_low = all(candles[i]["low"] <= candles[i - j]["low"] for j in range(1, w + 1)) and \
                     all(candles[i]["low"] <= candles[i + j]["low"] for j in range(1, w + 1))
            if is_low:
                swings.append({
                    "type": "LOW",
                    "idx": i,
                    "price": c["low"],
                    "dt": c["utc_dt"]
                })
        return swings

    def detect_qm_pattern(self, candles: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Scans recent swings to detect an active QM setup near the current price.
        """
        swings = self.find_swings(candles)
        if len(swings) < 4:
            return None

        recent_swings = swings[-6:]
        curr_price = candles[-1]["close"]

        # Check for Bearish QM: High1 -> Low1 -> HigherHigh -> LowerLow
        for i in range(len(recent_swings) - 3):
            s0, s1, s2, s3 = recent_swings[i], recent_swings[i+1], recent_swings[i+2], recent_swings[i+3]
            
            # Bearish QM
            if s0["type"] == "HIGH" and s1["type"] == "LOW" and s2["type"] == "HIGH" and s3["type"] == "LOW":
                h1, l1, hh, ll = s0["price"], s1["price"], s2["price"], s3["price"]
                if hh > h1 and ll < l1:
                    left_shoulder = h1
                    # Active setup if price is pulling back toward the left shoulder
                    if curr_price >= l1 and curr_price <= hh:
                        risk = hh - left_shoulder
                        if risk > 0:
                            return {
                                "pattern": "BEARISH_QM",
                                "left_shoulder": left_shoulder,
                                "higher_high": hh,
                                "lower_low": ll,
                                "entry": left_shoulder,
                                "sl": hh + (candles[-1]["atr14"] * 0.1),
                                "tp": ll,
                                "swings": [s0, s1, s2, s3]
                            }

            # Bullish QM: Low1 -> High1 -> LowerLow -> HigherHigh
            if s0["type"] == "LOW" and s1["type"] == "HIGH" and s2["type"] == "LOW" and s3["type"] == "HIGH":
                l1, h1, ll, hh = s0["price"], s1["price"], s2["price"], s3["price"]
                if ll < l1 and hh > h1:
                    left_shoulder = l1
                    # Active setup if price is pulling back toward the left shoulder
                    if curr_price <= h1 and curr_price >= ll:
                        risk = left_shoulder - ll
                        if risk > 0:
                            return {
                                "pattern": "BULLISH_QM",
                                "left_shoulder": left_shoulder,
                                "lower_low": ll,
                                "higher_high": hh,
                                "entry": left_shoulder,
                                "sl": ll - (candles[-1]["atr14"] * 0.1),
                                "tp": hh,
                                "swings": [s0, s1, s2, s3]
                            }

        return None
