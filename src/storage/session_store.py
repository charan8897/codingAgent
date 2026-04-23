import sqlite3
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class AttemptRecord:
    attempt: int
    command: str
    stdout: str
    stderr: str
    returncode: int
    status: str
    eval_feedback: str = ""
    eval_reasoning: str = ""
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)


class SessionStore:
    def __init__(self, db_path: str):
        import os
        self.db_path = os.path.expanduser(db_path)
        self._init_db()

    def _init_db(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_query TEXT,
                intent TEXT,
                created_at REAL,
                updated_at REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                attempt INTEGER,
                command TEXT,
                stdout TEXT,
                stderr TEXT,
                returncode INTEGER,
                status TEXT,
                confidence REAL,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )
        """)
        conn.commit()
        conn.close()

    def save_session(self, session_id: str, data: dict):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO sessions (id, user_query, intent, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, data.get("user_query", ""), data.get("intent", ""), time.time(), time.time()))
        conn.commit()
        conn.close()

    def save_attempt(self, session_id: str, record: AttemptRecord):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO attempts (session_id, attempt, command, stdout, stderr, returncode, status, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (session_id, record.attempt, record.command, record.stdout[:3000], record.stderr[:500], record.returncode, record.status, record.confidence))
        conn.commit()
        conn.close()

    def get_recent_sessions(self, limit: int = 10) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.execute("""
            SELECT id, user_query, intent, created_at 
            FROM sessions 
            ORDER BY updated_at DESC 
            LIMIT ?
        """, (limit,))
        rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_session_attempts(self, session_id: str) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.execute("""
            SELECT * FROM attempts WHERE session_id = ? ORDER BY attempt
        """, (session_id,))
        rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_session(self, session_id: str) -> Optional[dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = cur.fetchone()
        conn.close()
        return dict(row) if row else None