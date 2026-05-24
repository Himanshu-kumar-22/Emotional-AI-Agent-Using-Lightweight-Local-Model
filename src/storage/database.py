"""
src/storage/database.py
========================
SQLite database connection and schema management.

Two operating modes:

1. PERSISTENT MODE (default):
   - SQLite database written to data/conversations.db
   - All messages encrypted with AES-256 before storage
   - Data survives application restarts
   - User can review conversation history across sessions

2. PRIVACY MODE:
   - SQLite database created in memory (:memory:)
   - No filesystem writes at any point during the session
   - All data lost when application closes
   - Verified by monitoring filesystem write operations

SQLite is ideal for this use case because:
   - Serverless: no background process required
   - Single file: easy to backup, move, or delete
   - Python stdlib: no additional installation
   - ACID compliant: no data corruption on unexpected shutdown
   - Cross-platform: identical behavior on Mac/Windows/Linux
"""

import sys
import sqlite3
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings

logger = logging.getLogger(__name__)

# ── Schema Definitions ────────────────────────────────────────────────────────
SCHEMA_SQL = """
-- Sessions table: one row per conversation session
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    -- AES-encrypted JSON: {"model": "mistral", "emotion_model": "distilbert", ...}
    metadata    TEXT NOT NULL DEFAULT '{}'
);

-- Messages table: one row per conversation turn
CREATE TABLE IF NOT EXISTS messages (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    role        TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    -- AES-encrypted message text
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    -- AES-encrypted JSON: {"primary_emotion": "sadness", "confidence": 0.87, ...}
    emotion_data TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

-- Emotion log table: detailed per-turn emotion vectors for analysis
CREATE TABLE IF NOT EXISTS emotion_log (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    turn_number     INTEGER NOT NULL,
    -- AES-encrypted JSON: {"joy": 0.02, "sadness": 0.87, ...}
    raw_vector      TEXT,
    -- AES-encrypted JSON: smoothed probability vector
    smoothed_vector TEXT,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

-- Indices for efficient session-based queries
CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages(session_id, turn_number);

CREATE INDEX IF NOT EXISTS idx_emotion_log_session
    ON emotion_log(session_id, turn_number);
"""


class DatabaseManager:
    """
    Manages the SQLite connection and provides low-level database operations.

    Handles both file-based and in-memory database modes.
    All SQL operations go through this class — no raw sqlite3 calls
    outside this module.

    Usage:
        db = DatabaseManager(privacy_mode=False)
        db.initialize()

        with db.get_connection() as conn:
            conn.execute("SELECT * FROM sessions")
    """

    def __init__(self, privacy_mode: Optional[bool] = None):
        self.privacy_mode = (
            privacy_mode if privacy_mode is not None else settings.privacy_mode_default
        )

        if self.privacy_mode:
            # In-memory database: no filesystem writes ever
            self._db_path = ":memory:"
            logger.info("Database: PRIVACY MODE (in-memory, no disk writes)")
        else:
            self._db_path = str(settings.database_path)
            logger.info(f"Database: persistent at {self._db_path}")

        # Persistent connection for in-memory mode
        # (in-memory DBs are lost if connection closes)
        self._memory_connection: Optional[sqlite3.Connection] = None
        self._initialized = False

    def initialize(self):
        """
        Create database schema if it does not exist.
        Must be called once before any other operations.
        """
        with self.get_connection() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()
        self._initialized = True
        mode = "privacy" if self.privacy_mode else "persistent"
        logger.info(f"Database initialized ({mode} mode)")

    @contextmanager
    def get_connection(self):
        """
        Context manager that provides a database connection.

        For in-memory mode: reuses the single persistent connection.
        For file mode: creates a new connection per operation
        (SQLite is thread-safe with check_same_thread=False).

        Usage:
            with db.get_connection() as conn:
                rows = conn.execute("SELECT * FROM sessions").fetchall()
        """
        if self.privacy_mode:
            # In-memory: create once and reuse
            if self._memory_connection is None:
                self._memory_connection = sqlite3.connect(":memory:")
                self._memory_connection.row_factory = sqlite3.Row
                # Enable WAL mode for better concurrent read performance
                self._memory_connection.execute("PRAGMA journal_mode=WAL")
                self._memory_connection.execute("PRAGMA foreign_keys=ON")
            yield self._memory_connection
        else:
            # File-based: new connection per operation
            conn = sqlite3.connect(
                self._db_path,
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def execute(self, sql: str, params: tuple = ()) -> list:
        """Execute a SQL query and return all rows as dicts."""
        with self.get_connection() as conn:
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def execute_write(self, sql: str, params: tuple = ()) -> int:
        """
        Execute a write SQL statement and return the number of affected rows.
        """
        with self.get_connection() as conn:
            cursor = conn.execute(sql, params)
            conn.commit()
            return cursor.rowcount

    def execute_many(self, sql: str, params_list: list[tuple]) -> int:
        """Execute a SQL statement for multiple parameter sets."""
        with self.get_connection() as conn:
            cursor = conn.executemany(sql, params_list)
            conn.commit()
            return cursor.rowcount

    def close(self):
        """Close the database connection and clear in-memory data."""
        if self._memory_connection:
            self._memory_connection.close()
            self._memory_connection = None
        logger.debug("Database connection closed")

    @property
    def is_privacy_mode(self) -> bool:
        return self.privacy_mode
