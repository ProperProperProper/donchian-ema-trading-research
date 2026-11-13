#!/usr/bin/env python3
"""SQLite persistence for Moomoo candles, research runs, and paper events."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DB_PATH = Path(__file__).with_name("quant_dashboard.sqlite3")


def _json(value: Any) -> str:
    if is_dataclass(value):
        value = asdict(value)
    return json.dumps(value, default=str, sort_keys=True)


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    db = sqlite3.connect(path, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA busy_timeout=30000")
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS candles (
            symbol TEXT NOT NULL,
            interval_minutes INTEGER NOT NULL,
            time_key TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL,
            turnover REAL,
            source TEXT NOT NULL DEFAULT 'moomoo_opend',
            PRIMARY KEY (symbol, interval_minutes, time_key)
        );
        CREATE INDEX IF NOT EXISTS candles_lookup ON candles(symbol, interval_minutes, time_key);
        CREATE TABLE IF NOT EXISTS backtest_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            symbol TEXT NOT NULL,
            interval_minutes INTEGER NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL,
            params_json TEXT NOT NULL,
            stats_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS backtest_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL REFERENCES backtest_runs(id) ON DELETE CASCADE,
            trade_index INTEGER NOT NULL,
            trade_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS optimizer_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL REFERENCES backtest_runs(id) ON DELETE CASCADE,
            rank INTEGER NOT NULL,
            params_json TEXT NOT NULL,
            stats_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS execution_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            symbol TEXT,
            event_type TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );
        """
    )
    return db


def save_candles(frame: pd.DataFrame, symbol: str, interval_minutes: int, source: str = "moomoo_opend") -> int:
    db = connect()
    count = 0
    try:
        rows = []
        for stamp, row in frame.iterrows():
            rows.append((
                symbol, interval_minutes, pd.Timestamp(stamp).isoformat(),
                float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]),
                float(row["volume"]) if "volume" in row and pd.notna(row["volume"]) else None,
                float(row["turnover"]) if "turnover" in row and pd.notna(row["turnover"]) else None,
                source,
            ))
        db.executemany(
            """INSERT OR REPLACE INTO candles
            (symbol, interval_minutes, time_key, open, high, low, close, volume, turnover, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", rows,
        )
        db.commit()
        count = len(rows)
    finally:
        db.close()
    return count


def load_candles(symbol: str, interval_minutes: int, start: str, end: str) -> pd.DataFrame:
    db = connect()
    try:
        frame = pd.read_sql_query(
            """SELECT time_key, open, high, low, close, volume, turnover
               FROM candles
               WHERE symbol = ? AND interval_minutes = ? AND time_key >= ? AND time_key <= ?
               ORDER BY time_key""",
            db, params=(symbol, interval_minutes, start, end), parse_dates=["time_key"],
        )
    finally:
        db.close()
    if frame.empty:
        return frame
    return frame.set_index("time_key").dropna(subset=["open", "high", "low", "close"])


def save_backtest(symbol: str, interval_minutes: int, start: str, end: str, params: dict, stats: dict, trades: list[Any], optimization: list[dict]) -> int:
    db = connect()
    try:
        cur = db.execute(
            "INSERT INTO backtest_runs (created_at, symbol, interval_minutes, start_time, end_time, params_json, stats_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), symbol, interval_minutes, start, end, _json(params), _json(stats)),
        )
        run_id = int(cur.lastrowid)
        db.executemany(
            "INSERT INTO backtest_trades (run_id, trade_index, trade_json) VALUES (?, ?, ?)",
            [(run_id, i, _json(trade)) for i, trade in enumerate(trades)],
        )
        db.executemany(
            "INSERT INTO optimizer_results (run_id, rank, params_json, stats_json) VALUES (?, ?, ?, ?)",
            [(run_id, i + 1, _json(row["params"]), _json(row["stats"])) for i, row in enumerate(optimization)],
        )
        db.commit()
        return run_id
    finally:
        db.close()


def save_execution_event(event_type: str, symbol: str | None, payload: Any) -> None:
    db = connect()
    try:
        db.execute(
            "INSERT INTO execution_events (created_at, symbol, event_type, payload_json) VALUES (?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), symbol, event_type, _json(payload)),
        )
        db.commit()
    finally:
        db.close()
