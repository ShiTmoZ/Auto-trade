"""
Backtesting & Model Pre-Training Engine:
Simulates Tokyo Breakout Strategy on Historical Binance Data,
Extracts 15-Feature Vectors, and Trains the ML Gatekeeper with Conservative Regularization.
"""

from data_engine import DataEngine
from trend_filter import TrendFilter
from breakout_validator import BreakoutValidator
from ml_gatekeeper import MLGatekeeper
from execution_risk import ExecutionRiskManager

def run_backtest_and_train():
    print("=" * 65)
    print(" 📊 Historical Backtesting & Machine Learning Training Engine")
    print("=" * 65)

    data_engine = DataEngine("BTCUSDT")
    trend_filter = TrendFilter()
    validator = BreakoutValidator(min_volume_ratio=1.3, min_rr_ratio=2.5)
    ml_gatekeeper = MLGatekeeper(
        weights_file="ml_weights.json",
        learning_rate=0.015,
        l2_lambda=1.0,
        confidence_threshold=0.60
    )
    risk_manager = ExecutionRiskManager(equity_usd=10000.0, risk_pct=0.01, breakeven_r=1.5)

    print("Fetching historical candles from Binance API...")
    candles_15m = data_engine.fetch_klines("15m", limit=800)
    candles_1h = data_engine.fetch_klines("1h", limit=200)
    candles_4h = data_engine.fetch_klines("4h", limit=100)

    if not candles_15m:
        print("[Error] Failed to fetch candles.")
        return

    print(f"Loaded {len(candles_15m)} 15m candles (~8.3 days of historical data). Starting simulation...\n")

    trades_history = []
    active_trade = None

    # Step through historical candles
    for i in range(120, len(candles_15m) - 5):
        current_slice_15m = candles_15m[:i]
        current_candle = current_slice_15m[-1]

        # 1. Update open trade if active
        if active_trade:
            update = risk_manager.check_trade_update(active_trade, current_candle)
            if update["is_closed"]:
                active_trade["status"] = update["status"]
                active_trade["exit_price"] = update["exit_price"]
                active_trade["pnl_usd"] = update["pnl_usd"]
                active_trade["pnl_r"] = update["pnl_r"]
                trades_history.append(active_trade)

                # Online learning step
                was_win = (update["status"] == "CLOSED_WIN")
                ml_gatekeeper.update_model(active_trade["features"], was_win)
                active_trade = None
            continue

        # 2. Check for new setup
        tokyo_data = data_engine.extract_latest_tokyo_session(current_slice_15m)
        bias, htf_meta = trend_filter.get_market_bias(candles_1h, candles_4h)

        if not tokyo_data:
            continue

        signal = None
        if not tokyo_data["high_mitigated"] and bias == "BULLISH":
            signal = validator.evaluate_breakout(current_slice_15m, tokyo_data["high"], "TOKYO_HIGH", bias)
        elif not tokyo_data["low_mitigated"] and bias == "BEARISH":
            signal = validator.evaluate_breakout(current_slice_15m, tokyo_data["low"], "TOKYO_LOW", bias)

        if signal:
            features = ml_gatekeeper.extract_features(signal, tokyo_data, htf_meta, current_slice_15m)
            approved, score, _ = ml_gatekeeper.evaluate_signal(features)
            
            if approved:
                pos_btc, risk_usd = risk_manager.calculate_position_size(signal["entry_price"], signal["stop_loss"])
                active_trade = {
                    "trade_id": len(trades_history) + 1,
                    "side": signal["signal_type"],
                    "entry_price": signal["entry_price"],
                    "stop_loss": signal["stop_loss"],
                    "take_profit": signal["take_profit"],
                    "position_size_btc": pos_btc,
                    "features": features,
                    "ml_confidence": score
                }

    # Summary Statistics
    total_trades = len(trades_history)
    wins = [t for t in trades_history if t["status"] == "CLOSED_WIN"]
    losses = [t for t in trades_history if t["status"] == "CLOSED_LOSS"]
    be_trades = [t for t in trades_history if t["status"] == "CLOSED_BREAKEVEN"]

    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0.0
    net_pnl_usd = sum(t["pnl_usd"] for t in trades_history)
    net_pnl_r = sum(t["pnl_r"] for t in trades_history)

    print("\n" + "=" * 65)
    print(" 📈 BACKTEST & REINFORCEMENT RESULTS")
    print("=" * 65)
    print(f"Total Completed Trades:  {total_trades}")
    print(f"Winning Trades (TP Hit): {len(wins)}")
    print(f"Losing Trades (SL Hit):  {len(losses)}")
    print(f"Breakeven Trades:        {len(be_trades)}")
    print(f"Overall Win Rate:        {win_rate:.1f}%")
    print(f"Net PnL in Dollars:      ${net_pnl_usd:+,.2f} USD")
    print(f"Net Risk Multiple:       {net_pnl_r:+.2f} R")
    print(f"Updated Trained Samples: {ml_gatekeeper.total_trained_samples}")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    run_backtest_and_train()
