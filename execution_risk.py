"""
Execution & Risk Management Engine:
- Exact 1.0% Account Equity Risk Model
- Dynamic BTC Position Sizing
- Breakeven Trigger (+1.5R) and Trade Lifecycle Management
"""

from typing import Dict, Any, Tuple

class ExecutionRiskManager:
    def __init__(
        self,
        equity_usd: float = 10000.0,
        risk_pct: float = 0.01,
        breakeven_r: float = 1.5
    ):
        self.equity_usd = equity_usd
        self.risk_pct = risk_pct
        self.breakeven_r = breakeven_r

    def calculate_position_size(self, entry: float, stop_loss: float) -> Tuple[float, float]:
        """
        Calculates position size in BTC based on exact 1.0% risk.
        Returns: (position_size_btc, risk_usd)
        """
        risk_usd = self.equity_usd * self.risk_pct
        price_risk = abs(entry - stop_loss)
        
        if price_risk <= 0:
            return 0.0, 0.0

        position_size_btc = risk_usd / price_risk
        return round(position_size_btc, 6), round(risk_usd, 2)

    def check_trade_update(
        self,
        trade: Dict[str, Any],
        current_candle: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Monitors an open trade against high/low/close of incoming candles:
        - Breakeven check at +1.5R
        - Stop Loss trigger
        - Take Profit trigger
        """
        high = current_candle["high"]
        low = current_candle["low"]
        side = trade["side"]
        entry = trade["entry_price"]
        sl = trade["stop_loss"]
        tp = trade["take_profit"]
        risk_dist = abs(entry - sl)

        if side == "LONG":
            # 1. Breakeven check
            be_price = entry + (risk_dist * self.breakeven_r)
            if high >= be_price and sl < entry:
                trade["stop_loss"] = entry
                print(f"[RiskManager] 🛡️ Trade #{trade['trade_id']} reached +{self.breakeven_r}R! SL moved to Breakeven ({entry:,.2f})")

            # 2. Check Stop Loss
            if low <= trade["stop_loss"]:
                exit_price = trade["stop_loss"]
                pnl_usd = (exit_price - entry) * trade["position_size_btc"]
                pnl_r = (exit_price - entry) / risk_dist
                status = "CLOSED_BREAKEVEN" if exit_price == entry else "CLOSED_LOSS"
                return {"is_closed": True, "exit_price": exit_price, "status": status, "pnl_usd": pnl_usd, "pnl_r": pnl_r}

            # 3. Check Take Profit
            if high >= tp:
                exit_price = tp
                pnl_usd = (exit_price - entry) * trade["position_size_btc"]
                pnl_r = (exit_price - entry) / risk_dist
                return {"is_closed": True, "exit_price": exit_price, "status": "CLOSED_WIN", "pnl_usd": pnl_usd, "pnl_r": pnl_r}

        elif side == "SHORT":
            # 1. Breakeven check
            be_price = entry - (risk_dist * self.breakeven_r)
            if low <= be_price and sl > entry:
                trade["stop_loss"] = entry
                print(f"[RiskManager] 🛡️ Trade #{trade['trade_id']} reached +{self.breakeven_r}R! SL moved to Breakeven ({entry:,.2f})")

            # 2. Check Stop Loss
            if high >= trade["stop_loss"]:
                exit_price = trade["stop_loss"]
                pnl_usd = (entry - exit_price) * trade["position_size_btc"]
                pnl_r = (entry - exit_price) / risk_dist
                status = "CLOSED_BREAKEVEN" if exit_price == entry else "CLOSED_LOSS"
                return {"is_closed": True, "exit_price": exit_price, "status": status, "pnl_usd": pnl_usd, "pnl_r": pnl_r}

            # 3. Check Take Profit
            if low <= tp:
                exit_price = tp
                pnl_usd = (entry - exit_price) * trade["position_size_btc"]
                pnl_r = (entry - exit_price) / risk_dist
                return {"is_closed": True, "exit_price": exit_price, "status": "CLOSED_WIN", "pnl_usd": pnl_usd, "pnl_r": pnl_r}

        return {"is_closed": False}
