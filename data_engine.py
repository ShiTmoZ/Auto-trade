"""
Data Engine: Binance API integration, Candle Parsing, and Tokyo Session Range Extraction
"""

import json
import urllib.request
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

class DataEngine:
    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self.base_url = "https://api.binance.com/api/v3/klines"

    def fetch_klines(self, interval: str = "15m", limit: int = 150) -> List[Dict[str, Any]]:
        """
        Fetch OHLCV candles from Binance public API.
        Pure Python standard library implementation for maximum portability.
        """
        url = f"{self.base_url}?symbol={self.symbol}&interval={interval}&limit={limit}"
        req = urllib.request.Request(url, headers={"User-Agent": "AutoTradeBot/1.0"})
        
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw_data = json.loads(resp.read().decode())
        except Exception as e:
            print(f"[DataEngine Error] Failed to fetch {interval} klines: {e}")
            return []

        candles = []
        for k in raw_data:
            candles.append({
                "timestamp": k[0] / 1000,
                "utc_dt": datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "close_time": k[6] / 1000,
                "is_closed": False
            })
            
        # The last candle in Binance response is currently forming
        for c in candles[:-1]:
            c["is_closed"] = True
            
        self._calculate_indicators(candles)
        return candles

    def _calculate_indicators(self, candles: List[Dict[str, Any]]):
        """Calculate Volume SMA 20 and ATR for volatility benchmarking."""
        for i in range(len(candles)):
            # Volume SMA (20 period)
            start_vol = max(0, i - 19)
            subset_vol = [candles[j]["volume"] for j in range(start_vol, i + 1)]
            candles[i]["vol_sma20"] = sum(subset_vol) / len(subset_vol) if subset_vol else 1.0
            
            # Simple True Range (TR)
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

        # ATR 14
        for i in range(len(candles)):
            start_atr = max(0, i - 13)
            trs = [candles[j]["tr"] for j in range(start_atr, i + 1)]
            candles[i]["atr14"] = sum(trs) / len(trs) if trs else 100.0

    def extract_latest_tokyo_session(self, candles: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Extract the most recent completed Tokyo Session (00:00 to 09:00 UTC).
        Tracks High, Low, and unmitigated state.
        """
        tokyo_candles = []
        for c in candles:
            utc_hour = c["utc_dt"].hour
            # Tokyo session: 00:00 to 09:00 UTC
            if 0 <= utc_hour < 9:
                tokyo_candles.append(c)

        if not tokyo_candles:
            return None

        # Group by UTC date to find the latest session
        latest_date = tokyo_candles[-1]["utc_dt"].date()
        current_session_candles = [c for c in tokyo_candles if c["utc_dt"].date() == latest_date]

        if not current_session_candles:
            return None

        session_high = max(c["high"] for c in current_session_candles)
        session_low = min(c["low"] for c in current_session_candles)
        session_open = current_session_candles[0]["open"]
        session_close = current_session_candles[-1]["close"]

        # Find post-session candles to determine mitigation status
        last_tokyo_ts = current_session_candles[-1]["timestamp"]
        post_candles = [c for c in candles if c["timestamp"] > last_tokyo_ts]

        high_mitigated = any(c["high"] >= session_high for c in post_candles)
        low_mitigated = any(c["low"] <= session_low for c in post_candles)

        return {
            "date": str(latest_date),
            "high": session_high,
            "low": session_low,
            "open": session_open,
            "close": session_close,
            "range_usd": session_high - session_low,
            "high_mitigated": high_mitigated,
            "low_mitigated": low_mitigated,
            "candle_count": len(current_session_candles)
        }
