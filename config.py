"""
Global Configuration for Tokyo Session Breakout Trading Bot
"""

from typing import Dict, Any

CONFIG: Dict[str, Any] = {
    # Market & Symbols
    "symbol": "BTCUSDT",
    "symbols": [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
        "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"
    ],
    "multi_asset_mode": True,  # True = Monitor all 10 trained liquid symbols simultaneously
    "execution_timeframe": "15m",
    "htf_timeframes": ["1h", "4h"],
    
    # Tokyo / Asian Session in UTC (ICT Asian Range: 03:30 to 10:30 Tehran)
    "tokyo_session": {
        "start_utc_hour": 0,   # 00:00 UTC (03:30 Tehran)
        "end_utc_hour": 7,     # 07:00 UTC (10:30 Tehran - London Open Killzone begins)
    },
    
    # Risk Management
    "risk": {
        "risk_per_trade_pct": 0.01,    # Exactly 1.0% of total equity
        "min_rr_ratio": 2.5,           # Minimum 1:2.5 Risk-to-Reward ratio
        "breakeven_trigger_r": 1.5,    # Move SL to Entry at +1.5R
        "max_open_trades_per_symbol": 1,
        "max_total_open_trades": 3,    # Maximum concurrent portfolio exposure
        "default_equity_usd": 10000.0, # Default initial balance for paper trading
    },
    
    # 3-Candle Confirmation Protocol
    "strategy": {
        "min_volume_ratio": 1.3,       # Candle 0 volume must be >= 1.3x 20-period SMA
        "require_fvg": True,           # Require Fair Value Gap on Candle 1
        "fvg_min_dollar_size": 25.0,   # Minimum size of FVG in USD
        "retest_tolerance_pct": 0.0015 # Retest entry window around broken level (0.15%)
    },
    
    # Machine Learning Gatekeeper (Conservative Anti-Overfitting Hyperparameters)
    "ml_gatekeeper": {
        "model_file": "ml_weights.json",
        "learning_rate": 0.015,        # Ultra-conservative learning rate to handle financial noise
        "l2_regularization": 1.0,      # Strong L2 penalty to prevent memorizing random noise
        "confidence_threshold": 0.65,  # Minimum AI score (65%) to approve trade execution
        "train_validation_split": 0.8,
    },
    
    # Post-Mortem AI Reviewer (Optional: Gemini via 9Router or OpenAI-compatible endpoint)
    "ai_reviewer": {
        "enabled": False,  # Optional: Keep False for zero external dependencies
        "endpoint": "http://127.0.0.1:20128/v1/chat/completions",
        "model": "ag/gemini-3.8-flash-low",
        "timeout_seconds": 30,
        "review_frequency_trades": 5,
    },
    
    # Database & Storage
    "database_path": "trade_history.db",

    # Network & Censorship Bypass (for environments with restricted Binance access)
    "network": {
        "proxy": "",  # e.g., "http://127.0.0.1:10809" or "socks5://127.0.0.1:10808"
        "mirrors": [
            "https://data-api.binance.vision",
            "https://api1.binance.com",
            "https://api2.binance.com",
            "https://api3.binance.com",
            "https://api.binance.com"
        ]
    }
}
