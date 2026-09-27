"""
Trade Logger & Database Manager: Persistent SQLite storage for trade telemetry, features, and AI reviews
"""

import sqlite3
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

class TradeLogger:
    def __init__(self, db_path: str = "trade_history.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                risk_usd REAL NOT NULL,
                position_size_btc REAL NOT NULL,
                ml_confidence REAL NOT NULL,
                features_json TEXT NOT NULL,
                status TEXT NOT NULL,
                exit_price REAL,
                exit_time TEXT,
                pnl_usd REAL,
                pnl_r REAL,
                gemini_critique TEXT
            );
            CREATE TABLE IF NOT EXISTS executed_levels (
                symbol TEXT NOT NULL,
                session_date TEXT NOT NULL,
                level_type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (symbol, session_date, level_type)
            );
            """)
            conn.commit()

    def log_new_trade(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        risk_usd: float,
        position_size_btc: float,
        ml_confidence: float,
        features: List[float]
    ) -> int:
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO trades (
                timestamp, symbol, side, entry_price, stop_loss, take_profit,
                risk_usd, position_size_btc, ml_confidence, features_json, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now_str, symbol, side, entry_price, stop_loss, take_profit,
                risk_usd, position_size_btc, ml_confidence, json.dumps(features), "OPEN"
            ))
            conn.commit()
            return cursor.lastrowid or 0

    def update_trade_exit(
        self,
        trade_id: int,
        exit_price: float,
        status: str, # 'CLOSED_WIN', 'CLOSED_LOSS', 'CLOSED_BREAKEVEN'
        pnl_usd: float,
        pnl_r: float
    ):
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
            UPDATE trades SET
                exit_price = ?,
                exit_time = ?,
                status = ?,
                pnl_usd = ?,
                pnl_r = ?
            WHERE trade_id = ?
            """, (exit_price, now_str, status, pnl_usd, pnl_r, trade_id))
            conn.commit()

    def get_open_trades(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trades WHERE status = 'OPEN'")
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def get_recent_losses_without_critique(self, limit: int = 5) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
            SELECT * FROM trades 
            WHERE status = 'CLOSED_LOSS' AND (gemini_critique IS NULL OR gemini_critique = '')
            ORDER BY trade_id DESC LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def attach_gemini_review(self, trade_id: int, critique: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE trades SET gemini_critique = ? WHERE trade_id = ?", (critique, trade_id))
            conn.commit()

    def is_level_executed(self, symbol: str, session_date: str, level_type: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT 1 FROM executed_levels 
            WHERE symbol = ? AND session_date = ? AND level_type = ?
            """, (symbol, session_date, level_type))
            return cursor.fetchone() is not None

    def record_executed_level(self, symbol: str, session_date: str, level_type: str):
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR IGNORE INTO executed_levels (symbol, session_date, level_type, created_at)
            VALUES (?, ?, ?, ?)
            """, (symbol, session_date, level_type, now_str))
            conn.commit()
