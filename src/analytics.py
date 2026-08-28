import sqlite3
from pathlib import Path
from typing import Dict
import pandas as pd


class AnalyticsStore:
    def __init__(self, db_path: str = "support_logs.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    category TEXT,
                    intent TEXT,
                    sentiment TEXT,
                    urgency TEXT,
                    needs_human INTEGER,
                    confidence REAL,
                    retrieval_score REAL,
                    retrieval_mode TEXT,
                    source_policy TEXT,
                    response_mode TEXT
                )
            """)
            existing = {r[1] for r in conn.execute("PRAGMA table_info(interactions)").fetchall()}
            additions = {
                "intent": "TEXT", "confidence": "REAL", "retrieval_mode": "TEXT", "source_policy": "TEXT"
            }
            for col, typ in additions.items():
                if col not in existing:
                    conn.execute(f"ALTER TABLE interactions ADD COLUMN {col} {typ}")

    def log(self, query: str, result: Dict, retrieved):
        top = retrieved[0] if retrieved else {}
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO interactions (
                    query, answer, category, intent, sentiment, urgency, needs_human,
                    confidence, retrieval_score, retrieval_mode, source_policy, response_mode
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                query, result.get("answer", ""), result.get("category", "Other"),
                result.get("intent", "unknown"), result.get("sentiment", "Neutral"),
                result.get("urgency", "Normal"), int(bool(result.get("needs_human", False))),
                float(result.get("confidence", top.get("score", 0.0)) or 0.0),
                float(top.get("score", 0.0) or 0.0), top.get("retrieval_mode", "Unknown"),
                top.get("policy_id", ""), result.get("mode", "Unknown"),
            ))

    def dataframe(self) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query("SELECT * FROM interactions ORDER BY id DESC", conn)

    def clear(self):
        with self._connect() as conn:
            conn.execute("DELETE FROM interactions")
