"""
Technical Indicators Engine:
Pure Python implementation of Momentum, RSI Divergences, MACD, and EMA 50/200 Ribbon
without heavy external dependencies.
"""

from typing import List, Dict, Any, Tuple, Optional

class TechnicalIndicators:
    @staticmethod
    def calculate_ema(prices: List[float], period: int) -> List[float]:
        """Exponential Moving Average."""
        if not prices or len(prices) < period:
            return [prices[-1]] * len(prices) if prices else []
        multiplier = 2.0 / (period + 1.0)
        sma = sum(prices[:period]) / period
        ema = [sma] * period
        for price in prices[period:]:
            new_val = (price - ema[-1]) * multiplier + ema[-1]
            ema.append(new_val)
        return ema

    @staticmethod
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

        # Append last
        rsi.append(rsi[-1] if rsi else 50.0)
        return rsi

    @staticmethod
    def calculate_macd(
        prices: List[float],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Dict[str, List[float]]:
        """MACD Line, Signal Line, and Histogram."""
        ema_fast = TechnicalIndicators.calculate_ema(prices, fast)
        ema_slow = TechnicalIndicators.calculate_ema(prices, slow)
        macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
        signal_line = TechnicalIndicators.calculate_ema(macd_line, signal)
        histogram = [m - s for m, s in zip(macd_line, signal_line)]

        return {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram
        }

    @staticmethod
    def detect_rsi_divergence(
        candles: List[Dict[str, Any]],
        rsi_values: List[float]
    ) -> Optional[str]:
        """
        Detects Regular Divergence between Price and RSI in the last 15 candles.
        Returns 'BEARISH_DIV', 'BULLISH_DIV', or None.
        """
        if len(candles) < 20 or len(rsi_values) < 20:
            return None

        # Look at last 15 candles
        recent_c = candles[-15:]
        recent_rsi = rsi_values[-15:]

        p_curr, p_prev = recent_c[-1]["high"], max(c["high"] for c in recent_c[:8])
        r_curr, r_prev = recent_rsi[-1], max(recent_rsi[:8])

        # Bearish: Price Higher High, RSI Lower High
        if p_curr > p_prev and r_curr < r_prev - 3.0 and r_curr > 60:
            return "BEARISH_DIV"

        p_curr_l, p_prev_l = recent_c[-1]["low"], min(c["low"] for c in recent_c[:8])
        r_curr_l, r_prev_l = recent_rsi[-1], min(recent_rsi[:8])

        # Bullish: Price Lower Low, RSI Higher Low
        if p_curr_l < p_prev_l and r_curr_l > r_prev_l + 3.0 and r_curr_l < 40:
            return "BULLISH_DIV"

        return None

    @staticmethod
    def calculate_momentum_score(candles: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculates pure momentum displacement velocity across the last 3 candles:
        - Body to Range ratio (conviction)
        - Velocity vs ATR
        """
        if len(candles) < 3:
            return {"body_ratio": 0.5, "velocity_atr": 1.0}

        c0 = candles[-1]
        c_range = max(0.01, c0["high"] - c0["low"])
        body = abs(c0["close"] - c0["open"])
        body_ratio = body / c_range

        atr = c0.get("atr14", 50.0)
        disp_3c = abs(candles[-1]["close"] - candles[-3]["open"])
        velocity_atr = disp_3c / max(1.0, atr)

        return {
            "body_ratio": round(body_ratio, 3),
            "velocity_atr": round(velocity_atr, 3)
        }
