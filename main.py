#!/usr/bin/env python3
"""
Main Execution Engine:
Multi-Asset Institutional Tokyo Breakout Auto-Trade Bot (ICT/RTM + ML)
Monitors Top 10 Liquid Cryptos with Res-MLP Gatekeeper & Zero External Dependencies
"""

import time
import sys
import concurrent.futures
from datetime import datetime, timezone
from typing import Dict, Any, List
from config import CONFIG
from data_engine import DataEngine
from trend_filter import TrendFilter
from breakout_validator import BreakoutValidator
from ml_gatekeeper import MLGatekeeper
from trade_logger import TradeLogger
from execution_risk import ExecutionRiskManager
from gemini_reviewer import GeminiReviewer
from telegram_notifier import TelegramNotifier

def run_bot(paper_mode: bool = True):
    symbols = CONFIG["symbols"] if CONFIG.get("multi_asset_mode") else [CONFIG["symbol"]]
    
    print("=" * 75)
    print(" 🚀 Institutional Tokyo Breakout Auto-Trade Bot (ICT/RTM + ML)")
    print(f"    Mode: {'MULTI-ASSET (' + str(len(symbols)) + ' Pairs)' if len(symbols) > 1 else symbols[0]} | {'PAPER TRADING' if paper_mode else 'LIVE'}")
    print(f"    Assets: {', '.join(symbols[:5])}...")
    print(f"    Risk per Trade: {CONFIG['risk']['risk_per_trade_pct'] * 100}% | Min R:R: 1:{CONFIG['risk']['min_rr_ratio']} | BE: +{CONFIG['risk']['breakeven_trigger_r']}R")
    print(f"    ML Gatekeeper: Res-MLP Gatekeeper (Threshold: {CONFIG['ml_gatekeeper']['confidence_threshold']:.0%})")
    print("=" * 75)

    # Initialize per-symbol Data Engines
    data_engines = {sym: DataEngine(symbol=sym) for sym in symbols}
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
    notifier = TelegramNotifier()

    tick_count = 0
    max_total_trades = CONFIG["risk"].get("max_total_open_trades", 3)
    htf_cache: Dict[str, Any] = {}
    executed_levels = set()  # (symbol, tokyo_date, level_type)

    def fetch_symbol_data(sym: str):
        de = data_engines[sym]
        candles_15m = de.fetch_klines("15m", limit=80)
        now_ts = time.time()
        cached = htf_cache.get(sym)
        if not cached or (now_ts - cached["time"] > 180):
            c1h = de.fetch_klines("1h", limit=50)
            c4h = de.fetch_klines("4h", limit=30)
            if c1h and c4h:
                b, meta = trend_filter.get_market_bias(c1h, c4h)
                htf_cache[sym] = {"bias": b, "htf_meta": meta, "time": now_ts}
            elif cached:
                b, meta = cached["bias"], cached["htf_meta"]
            else:
                b, meta = "NEUTRAL", {}
        else:
            b, meta = cached["bias"], cached["htf_meta"]
        return sym, candles_15m, b, meta

    while True:
        try:
            tick_count += 1
            now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            open_trades = trade_logger.get_open_trades()
            open_symbols = {t["symbol"] for t in open_trades}

            # Fast concurrent ingestion (all 20 pairs in ~3-4 seconds)
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                batch_data = list(pool.map(fetch_symbol_data, symbols))

            status_items = []

            for sym, candles_15m, bias, htf_meta in batch_data:
                if not candles_15m:
                    continue

                de = data_engines[sym]
                current_candle = candles_15m[-1]

                # Monitor open trades for this specific symbol
                sym_open = [t for t in open_trades if t["symbol"] == sym]
                for trade in sym_open:
                    update_res = risk_manager.check_trade_update(trade, current_candle)
                    if update_res.get("be_triggered"):
                        notifier.notify_breakeven(trade["trade_id"], sym, trade["entry_price"])

                    if update_res["is_closed"]:
                        trade_logger.update_trade_exit(
                            trade_id=trade["trade_id"],
                            exit_price=update_res["exit_price"],
                            status=update_res["status"],
                            pnl_usd=update_res["pnl_usd"],
                            pnl_r=update_res["pnl_r"]
                        )
                        notifier.notify_trade_closed(
                            trade["trade_id"], sym, trade["side"], update_res["status"],
                            update_res["exit_price"], update_res["pnl_usd"], update_res["pnl_r"]
                        )
                        print(f"\n🏁 TRADE CLOSED #{trade['trade_id']} on {sym} ({trade['side']}) | Result: {update_res['status']}")
                        print(f"   Exit: {update_res['exit_price']:,.2f} | PnL: {update_res['pnl_usd']:+,.2f} USD ({update_res['pnl_r']:+.2f}R)\n")

                # Check new breakout opportunity if not already exposed
                if sym not in open_symbols and len(open_trades) < max_total_trades:
                    tokyo_data = de.extract_latest_tokyo_session(candles_15m)
                    if tokyo_data:
                        # Long setup: Tokyo High breakout
                        high_key = (sym, tokyo_data["date"], "TOKYO_HIGH")
                        if high_key not in executed_levels and bias == "BULLISH":
                            signal = validator.evaluate_breakout(
                                candles_15m, tokyo_data["high"], "TOKYO_HIGH", bias
                            )
                            if signal:
                                executed_levels.add(high_key)
                                _process_potential_signal(
                                    sym, signal, tokyo_data, htf_meta, candles_15m,
                                    ml_gatekeeper, risk_manager, trade_logger, notifier
                                )

                        # Short setup: Tokyo Low breakout
                        low_key = (sym, tokyo_data["date"], "TOKYO_LOW")
                        if low_key not in executed_levels and bias == "BEARISH":
                            signal = validator.evaluate_breakout(
                                candles_15m, tokyo_data["low"], "TOKYO_LOW", bias
                            )
                            if signal:
                                executed_levels.add(low_key)
                                _process_potential_signal(
                                    sym, signal, tokyo_data, htf_meta, candles_15m,
                                    ml_gatekeeper, risk_manager, trade_logger, notifier
                                )

                # Status snippet for top symbols
                clean_sym = sym.replace("USDT", "")
                status_items.append(f"{clean_sym}:{current_candle['close']:,.1f}")

            # Heartbeat display
            sample_status = " | ".join(status_items[:5])
            print(f"[{now_str}] Portfolio ({len(symbols)} Assets) | Open: {len(open_trades)}/{max_total_trades} | {sample_status}", flush=True)

            # Review losing trades if enabled
            if tick_count % 30 == 0 and CONFIG["ai_reviewer"]["enabled"]:
                ai_reviewer.review_losing_trades(trade_logger)

            time.sleep(15)

        except KeyboardInterrupt:
            print("\n[INFO] Multi-Asset Bot stopped safely by user.")
            break
        except Exception as e:
            print(f"\n[Engine Loop Error]: {e}")
            time.sleep(10)

def _process_potential_signal(symbol: str, signal: Dict[str, Any], tokyo_data: Dict[str, Any],
                              htf_meta: Dict[str, Any], candles_15m: List[Dict[str, Any]],
                              ml_gatekeeper: MLGatekeeper, risk_manager: ExecutionRiskManager,
                              trade_logger: TradeLogger, notifier: TelegramNotifier):
    features = ml_gatekeeper.extract_features(signal, tokyo_data, htf_meta, candles_15m)
    approved, score, details = ml_gatekeeper.evaluate_signal(features)

    # Immediately push signal alert to Telegram
    try:
        notifier.notify_trade_signal(symbol, signal, score, approved)
    except Exception as e:
        print(f"[Notifier Warning]: {e}")

    print("\n\n" + "=" * 70)
    print(f"🚨 VALID BREAKOUT DETECTED: {symbol} {signal['signal_type']} @ {signal['level_broken']:,.2f} USD")
    print(f"   Entry: {signal['entry_price']:,.2f} | SL: {signal['stop_loss']:,.2f} | TP: {signal['take_profit']:,.2f} (R:R: 1:{signal['rr_ratio']})")
    print(f"   Volume Ratio: {signal['vol_ratio']}x | FVG: {signal['has_fvg']} ({signal['fvg_size']:.2f} USD)")
    print(f"🧠 AI Gatekeeper Confidence Score: {score:.1%} (Threshold: {ml_gatekeeper.confidence_threshold:.0%})")

    if approved:
        pos_size, risk_usd = risk_manager.calculate_position_size(signal["entry_price"], signal["stop_loss"])
        trade_id = trade_logger.log_new_trade(
            symbol=symbol,
            side=signal["signal_type"],
            entry_price=signal["entry_price"],
            stop_loss=signal["stop_loss"],
            take_profit=signal["take_profit"],
            risk_usd=risk_usd,
            position_size_btc=pos_size,
            ml_confidence=score,
            features=features
        )
        print(f"   ✅ AI APPROVED TRADE! Executed Trade #{trade_id} on {symbol}")
        print(f"   Position Units: {pos_size:.4f} | Risk Budget: ${risk_usd:,.2f}")
    else:
        print(f"   ❌ AI REJECTED TRADE (Score {score:.1%} below safety threshold). Signal filtered.")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    run_bot(paper_mode=True)
