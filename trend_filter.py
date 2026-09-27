"""
Higher Timeframe (HTF) Trend Filter: Evaluates 1H and 4H Market Structure & EMAs
"""

from typing import List, Dict, Any, Tuple

class TrendFilter:
    def __init__(self):
        pass

    def calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """Calculates Exponential Moving Average without external dependencies."""
        if not prices or len(prices) < period:
            return [prices[-1]] * len(prices) if prices else []
        
        multiplier = 2.0 / (period + 1.0)
        # Seed with SMA
        sma = sum(prices[:period]) / period
        ema = [sma] * period
        
        for price in prices[period:]:
            new_ema = (price - ema[-1]) * multiplier + ema[-1]
            ema.append(new_ema)
        return ema

    def analyze_structure(self, candles: List[Dict[str, Any]]) -> str:
        """
        Analyzes market structure:
        - Price vs EMA 50
        - Swing Highs / Swing Lows alignment (BOS)
        Returns: 'BULLISH', 'BEARISH', or 'NEUTRAL'
        """
        if len(candles) < 30:
            return "NEUTRAL"

        closes = [c["close"] for c in candles]
        ema50 = self.calculate_ema(closes, 50)
        
        current_close = closes[-1]
        current_ema = ema50[-1]
        
        # Check last 3 swing points
        highs = [c["high"] for c in candles[-20:]]
        lows = [c["low"] for c in candles[-20:]]
        
        higher_highs = highs[-1] > max(highs[:10])
        lower_lows = lows[-1] < min(lows[:10])
        
        if current_close > current_ema and not lower_lows:
            return "BULLISH"
        elif current_close < current_ema and not higher_highs:
            return "BEARISH"
        return "NEUTRAL"

    def get_market_bias(self, klines_1h: List[Dict[str, Any]], klines_4h: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        """
        Synthesizes 1H and 4H timeframes into an actionable bias.
        Rule:
        - LONG allowed only if 1H is BULLISH and 4H is not BEARISH.
        - SHORT allowed only if 1H is BEARISH and 4H is not BULLISH.
        """
        trend_1h = self.analyze_structure(klines_1h)
        trend_4h = self.analyze_structure(klines_4h)
        
        if trend_1h == "BULLISH" and trend_4h in ["BULLISH", "NEUTRAL"]:
            bias = "BULLISH"
        elif trend_1h == "BEARISH" and trend_4h in ["BEARISH", "NEUTRAL"]:
            bias = "BEARISH"
        else:
            bias = "NEUTRAL"

        metadata = {
            "bias": bias,
            "trend_1h": trend_1h,
            "trend_4h": trend_4h,
            "1h_price": klines_1h[-1]["close"] if klines_1h else 0,
            "4h_price": klines_4h[-1]["close"] if klines_4h else 0
        }
        return bias, metadata
