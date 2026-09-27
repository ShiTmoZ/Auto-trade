#!/usr/bin/env python3
"""
Main Execution Engine:
Orchestrates Tokyo Session Breakout Trading Bot with ML Gatekeeper & Gemini Meta-Reviewer
"""

import time
import sys
from datetime import datetime, timezone
from config import CONFIG
from data_engine import DataEngine
from trend_filter import TrendFilter
from breakout_validator import BreakoutValidator
from ml_gatekeeper import MLGatekeeper
from trade_logger import TradeLogger
from execution_risk import ExecutionRiskManager
from gemini_reviewer import GeminiReviewer

def run_bot(paper_mode: bool = True):
    print("=" * 70)
    print(" 🚀 Institutional Tokyo Breakout Auto-Trade Bot (ICT/RTM + ML)")
    print(f"    Symbol: {CONFIG['symbol']} | Mode: {'PAPER TRADING' if paper_mode else 'LIVE'}")
    print(f"    Risk per Trade: {CONFIG['risk']['risk_per_trade_pct'] * 100}% | Min R:R: 1:{CONFIG['risk']['min_rr_ratio']}")
    print(f"    ML Learning Rate: {CONFIG['ml_gatekeeper']['learning_rate']} (Conservative Anti-Overfitting)")
    print("=" * 70)

    # Initialize modules
    data_engine = DataEngine(symbol=CONFIG["symbol"])
    trend_filter = TrendFilter()
    validator = BreakoutValidator(
        min_volume_ratio=CONFIG["strategy"]["min_volume_ratio"],
        min_rr_ratio=CONFIG["risk"]["min_rr_ratio"]
    )
    ml_gatekeeper = MLGatekeeper(
        weights_file=CONFIG["ml_gatekeeper"]["model_file"],
        learning_rate=CONFIG["ml_gatekeeper"]["learning_rate"],
        l2_lambda=CONFIG["ml_gatekeeper"]["l2_regularization"],
        confidence_threshold=CONFIG["ml_gatekeeper"]["confidence_threshold"]
    )
    trade_logger = TradeLogger(db_path=CONFIG["database_path"])
    risk_manager = ExecutionRiskManager(
        equity_usd=CONFIG["risk"]["default_equity_usd"],
        risk_pct=CONFIG["risk"]["risk_per_trade_pct"],
        breakeven_r=CONFIG["risk"]["breakeven_trigger_r"]
    )
    ai_reviewer = GeminiReviewer(
        endpoint=CONFIG["ai_reviewer"]["endpoint"],
        model=CONFIG["ai_reviewer"]["model"]
    )

    tick_count = 0

    while True:
        try:
            tick_count += 1
            now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

            # 1. Fetch Market Data
            candles_15m = data_engine.fetch_klines("15m", limit=120)
            candles_1h = data_engine.fetch_klines("1h", limit=80)
            candles_4h = data_engine.fetch_klines("4h", limit=50)

            if not candles_15m or not candles_1h:
                time.sleep(15)
                continue

            current_candle = candles_15m[-1]
            bias, htf_meta = trend_filter.get_market_bias(candles_1h, candles_4h)

            # 2. Monitor & Update Existing Open Trades
            open_trades = trade_logger.get_open_trades()
            for trade in open_trades:
                update_res = risk_manager.check_trade_update(trade, current_candle)
                if update_res["is_closed"]:
                    trade_logger.update_trade_exit(
                        trade_id=trade["trade_id"],
                        exit_price=update_res["exit_price"],
                        status=update_res["status"],
                        pnl_usd=update_res["pnl_usd"],
                        pnl_r=update_res["pnl_r"]
                    )
                    print("\n" + "#" * 65)
                    print(f"🏁 TRADE CLOSED #{trade['trade_id']} ({trade['side']}) | Result: {update_res['status']}")
                    print(f"   Exit Price: {update_res['exit_price']:,.2f} USD | Net PnL: {update_res['pnl_usd']:+,.2f} USD ({update_res['pnl_r']:+.2f}R)")
                    print("#" * 65 + "\n")

                    # Online ML update step
                    features = eval(trade["features_json"])
                    was_win = (update_res["status"] == "CLOSED_WIN")
                    ml_gatekeeper.update_model(features, was_win)

            # Periodically review losing trades via Gemini
            if tick_count % 20 == 0 and CONFIG["ai_reviewer"]["enabled"]:
                ai_reviewer.review_losing_trades(trade_logger)

            # 3. Check for New Trading Opportunities
            # Only evaluate entry if no trade is currently open
            if not open_trades:
                tokyo_data = data_engine.extract_latest_tokyo_session(candles_15m)
                bias, htf_meta = trend_filter.get_market_bias(candles_1h, candles_4h)

                if tokyo_data:
                    # Check Tokyo High Breakout (Long setup)
                    if not tokyo_data["high_mitigated"] and bias == "BULLISH":
                        signal = validator.evaluate_breakout(
                            candles_15m, tokyo_data["high"], "TOKYO_HIGH", bias
                        )
                        if signal:
                            _process_potential_signal(
                                signal, tokyo_data, htf_meta, candles_15m,
                                ml_gatekeeper, risk_manager, trade_logger
                            )

                    # Check Tokyo Low Breakout (Short setup)
                    if not tokyo_data["low_mitigated"] and bias == "BEARISH":
                        signal = validator.evaluate_breakout(
                            candles_15m, tokyo_data["low"], "TOKYO_LOW", bias
                        )
                        if signal:
                            _process_potential_signal(
                                signal, tokyo_data, htf_meta, candles_15m,
                                ml_gatekeeper, risk_manager, trade_logger
                            )

            # Live CLI Heartbeat
            status_text = f"[{now_str}] Live: {current_candle['close']:,.2f} $ | HTF: {htf_meta['bias']} | Open Trades: {len(open_trades)}"
            print(status_text, end="\r")

            time.sleep(15)

        except KeyboardInterrupt:
            print("\n[INFO] Bot stopped safely by user.")
            break
        except Exception as e:
            print(f"\n[Engine Loop Error]: {e}")
            time.sleep(10)

def _process_potential_signal(signal, tokyo_data, htf_meta, candles_15m, ml_gatekeeper, risk_manager, trade_logger):
    # Extract 15 features
    features = ml_gatekeeper.extract_features(signal, tokyo_data, htf_meta, candles_15m)
    approved, score, details = ml_gatekeeper.evaluate_signal(features)

    print("\n\n" + "=" * 65)
    print(f"🚨 VALID BREAKOUT DETECTED: {signal['signal_type']} @ {signal['level_broken']:,.2f} USD")
    print(f"   Entry: {signal['entry_price']:,.2f} | SL: {signal['stop_loss']:,.2f} | TP: {signal['take_profit']:,.2f} (R:R: 1:{signal['rr_ratio']})")
    print(f"   Vol Ratio: {signal['vol_ratio']}x | Has FVG: {signal['has_fvg']} ({signal['fvg_size']:.2f} USD)")
    print(f"🧠 AI Gatekeeper Confidence Score: {score:.1%} (Threshold: {ml_gatekeeper.confidence_threshold:.0%})")

    if approved:
        pos_size, risk_usd = risk_manager.calculate_position_size(signal["entry_price"], signal["stop_loss"])
        trade_id = trade_logger.log_new_trade(
            symbol=CONFIG["symbol"],
            side=signal["signal_type"],
            entry_price=signal["entry_price"],
            stop_loss=signal["stop_loss"],
            take_profit=signal["take_profit"],
            risk_usd=risk_usd,
            position_size_btc=pos_size,
            ml_confidence=score,
            features=features
        )
        print(f"   ✅ AI APPROVED TRADE! Executed Trade #{trade_id}")
        print(f"   Position Size: {pos_size:.4f} BTC | Dollar Risk: ${risk_usd:,.2f}")
    else:
        print(f"   ❌ AI REJECTED TRADE (Score {score:.1%} below safety threshold). Filtered noise successfully.")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    run_bot(paper_mode=True)
