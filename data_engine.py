"""
Data Engine: Binance API integration, Candle Parsing, and Tokyo Session Range Extraction
Includes automatic censorship-bypass mirrors and optional proxy support for restricted networks.
"""

import os
import json
import urllib.request
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from config import CONFIG

class DataEngine:
    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        net_cfg = CONFIG.get("network", {})
        self.mirrors = net_cfg.get("mirrors", [
            "https://data-api.binance.vision",
            "https://api1.binance.com",
            "https://api2.binance.com",
            "https://api3.binance.com",
            "https://api.binance.com"
        ])
        self.proxy = net_cfg.get("proxy", "") or os.environ.get("HTTPS_PROXY", "") or os.environ.get("HTTP_PROXY", "")
        
        # Configure urllib opener if proxy is specified
        if self.proxy:
            proxy_handler = urllib.request.ProxyHandler({"http": self.proxy, "https": self.proxy})
            self.opener = urllib.request.build_opener(proxy_handler)
        else:
            self.opener = urllib.request.build_opener()

    def fetch_klines(self, interval: str = "15m", limit: int = 150) -> List[Dict[str, Any]]:
        """
        Fetch OHLCV candles from Binance, cycling through mirrors on network blockades.
        """
        raw_data = None
        last_error = None

        for base_url in self.mirrors:
            url = f"{base_url}/api/v3/klines?symbol={self.symbol}&interval={interval}&limit={limit}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            try:
                with self.opener.open(req, timeout=8) as resp:
                    if resp.status == 200:
                        raw_data = json.loads(resp.read().decode())
                        break
            except Exception as e:
                last_error = e
                continue

        if raw_data is None:
            print(f"[DataEngine Error] Failed to fetch {interval} klines across all mirrors. Last error: {last_error}")
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
            
        # The last candle in response is currently forming
        for c in candles[:-1]:
            c["is_closed"] = True
            
        self._calculate_indicators(candles)
        return candles

    def _calculate_indicators(self, candles: List[Dict[str, Any]]):
        """Calculate Volume SMA 20 and ATR for volatility benchmarking."""
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

        # ATR 14
        for i in range(len(candles)):
            start_atr = max(0, i - 13)
            subset_tr = [candles[j]["tr"] for j in range(start_atr, i + 1)]
            candles[i]["atr14"] = sum(subset_tr) / len(subset_tr) if subset_tr else 50.0

    def extract_latest_tokyo_session(self, candles_15m: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Identify Asian / Tokyo Session (00:00 - 07:00 UTC / 03:30 - 10:30 Tehran) range.
        Extends High and Low forward until mitigated by London/NY price action.
        """
        end_hour = CONFIG.get("tokyo_session", {}).get("end_utc_hour", 7)
        tokyo_candles = [c for c in candles_15m if 0 <= c["utc_dt"].hour < end_hour]
        if not tokyo_candles:
            return None

        latest_date = tokyo_candles[-1]["utc_dt"].date()
        current_tokyo = [c for c in tokyo_candles if c["utc_dt"].date() == latest_date]
        
        if len(current_tokyo) < 4:
            return None

        tokyo_high = max(c["high"] for c in current_tokyo)
        tokyo_low = min(c["low"] for c in current_tokyo)
        
        # Check if mitigated by candles after end_hour (London/NY)
        post_tokyo = [c for c in candles_15m if c["utc_dt"].date() == latest_date and c["utc_dt"].hour >= end_hour]
        
        high_mitigated = any(c["high"] >= tokyo_high for c in post_tokyo)
        low_mitigated = any(c["low"] <= tokyo_low for c in post_tokyo)

        return {
            "date": latest_date.isoformat(),
            "high": tokyo_high,
            "low": tokyo_low,
            "high_mitigated": high_mitigated,
            "low_mitigated": low_mitigated,
            "candle_count": len(current_tokyo)
        }
