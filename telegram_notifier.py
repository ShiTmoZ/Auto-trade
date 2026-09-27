"""
Telegram Live Alert Dispatcher for Auto-Trade Bot
Sends immediate notifications for valid breakouts, AI decisions, entries, breakevens, and exits.
"""

import json
import urllib.request
import os
from typing import Dict, Any, Optional

class TelegramNotifier:
    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        self.bot_token = bot_token or self._load_token()
        self.chat_id = chat_id or "7441068375"

    def _load_token(self) -> str:
        env_paths = ["/root/.hermes/.env", "/root/.env"]
        for p in env_paths:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("TELEGRAM_BOT_TOKEN="):
                            return line.split("=", 1)[1].strip()
        return ""

    def send_message(self, text: str) -> bool:
        if not self.bot_token or not self.chat_id:
            return False
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except Exception as e:
            print(f"[TelegramNotifier Error]: {e}")
            return False

    def notify_trade_signal(self, symbol: str, signal: Dict[str, Any], score: float, approved: bool):
        icon = "🟢" if signal["signal_type"] == "LONG" else "🔴"
        status_txt = "✅ *AI APPROVED (Trade Executed)*" if approved else "❌ *AI REJECTED (Filtered)*"
        
        msg = (
            f"⚡ *New Signal Detected: {symbol}*\n"
            f"{icon} *Direction:* {signal['signal_type']}\n"
            f"📍 *Broken Level:* `{signal['level_broken']:,.2f} USD`\n"
            f"🎯 *Entry:* `{signal['entry_price']:,.2f}`\n"
            f"🛑 *Stop Loss:* `{signal['stop_loss']:,.2f}`\n"
            f"🏁 *Take Profit:* `{signal['take_profit']:,.2f}` (R:R: 1:{signal['rr_ratio']})\n"
            f"📊 *Vol Ratio:* `{signal['vol_ratio']}x` | *FVG:* `{signal['has_fvg']}`\n"
            f"🧠 *AI Gatekeeper Score:* `{score:.1%}`\n"
            f"📌 *Status:* {status_txt}"
        )
        self.send_message(msg)

    def notify_trade_closed(self, trade_id: int, symbol: str, side: str, status: str,
                            exit_price: float, pnl_usd: float, pnl_r: float):
        icon = "🎯" if "WIN" in status else ("🟡" if "BREAKEVEN" in status else "🛑")
        msg = (
            f"{icon} *Trade Closed #{trade_id} on {symbol} ({side})*\n"
            f"📌 *Result:* `{status}`\n"
            f"🚪 *Exit Price:* `{exit_price:,.2f} USD`\n"
            f"💰 *PnL:* `{pnl_usd:+,.2f} USD` (`{pnl_r:+.2f}R`)"
        )
        self.send_message(msg)

    def notify_breakeven(self, trade_id: int, symbol: str, entry_price: float):
        msg = (
            f"🛡️ *Breakeven Protected: Trade #{trade_id} on {symbol}*\n"
            f"Favorable excursion exceeded `+1.5R`.\n"
            f"Stop Loss moved to Entry (`{entry_price:,.2f}`). Capital is now 100% risk-free!"
        )
        self.send_message(msg)
