"""
Gemini Meta-Reviewer:
- Periodic post-mortem analysis of losing trades
- Connects via 9Router (ag/gemini-3.8-flash-high) or compatible endpoint
- Identifies macro failure patterns and records feedback into SQLite
"""

import json
import urllib.request
from typing import Dict, Any, List
from trade_logger import TradeLogger

class GeminiReviewer:
    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:20128/v1/chat/completions",
        model: str = "ag/gemini-3.8-flash-high",
        timeout: int = 30
    ):
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout

    def review_losing_trades(self, logger: TradeLogger) -> List[Dict[str, Any]]:
        """
        Pulls recent unreviewed losing trades and sends them for AI root cause analysis.
        """
        losses = logger.get_recent_losses_without_critique(limit=3)
        if not losses:
            return []

        reviews = []
        for trade in losses:
            features = json.loads(trade["features_json"])
            prompt = f"""
You are a Quantitative Risk Auditor reviewing a losing algorithmic trade on BTC/USDT.
Trade Metadata:
- Trade ID: {trade['trade_id']}
- Side: {trade['side']}
- Entry Price: {trade['entry_price']} USD
- Stop Loss: {trade['stop_loss']} USD
- Exit Price: {trade['exit_price']} USD
- Net Loss: {trade['pnl_usd']:.2f} USD ({trade['pnl_r']:.2f}R)
- ML Model Confidence: {trade['ml_confidence']:.2%}

Key Signal Features:
- Volume Ratio vs 20-SMA: {features[0]:.2f}x
- Fair Value Gap Size %: {features[1]:.4f}%
- Has FVG: {'Yes' if features[2] == 1.0 else 'No'}
- Candle Body Ratio: {features[3]:.2f}
- Tokyo Session Range %: {features[5]:.2f}%
- UTC Hour: {int(features[6] * 24)}:00 UTC

Please provide a concise (3-4 sentences) root-cause post-mortem:
1. What market micro-structure factor likely caused this failure (e.g. liquidity sweep, false expansion, session timing)?
2. What concrete adjustment should be made to prevent similar losses?
Keep your response analytical and direct.
"""
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are an institutional trading systems auditor."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2
            }

            try:
                data_bytes = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    self.endpoint,
                    data=data_bytes,
                    headers={"Content-Type": "application/json", "Authorization": "Bearer dummy"}
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    res_json = json.loads(resp.read().decode())
                    critique = res_json["choices"][0]["message"]["content"].strip()
                    logger.attach_gemini_review(trade["trade_id"], critique)
                    reviews.append({"trade_id": trade["trade_id"], "critique": critique})
                    print(f"[GeminiReviewer] 🧠 Reviewed Trade #{trade['trade_id']}:\n{critique}\n")
            except Exception as e:
                print(f"[GeminiReviewer] Could not review Trade #{trade['trade_id']}: {e}")

        return reviews
