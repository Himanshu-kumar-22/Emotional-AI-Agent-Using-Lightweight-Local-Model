"""
src/storage/session_manager.py
================================
High-level session and message storage operations.

This is the ONLY storage interface that other modules use.
Nothing outside this file calls DatabaseManager or EncryptionManager directly.

Responsibilities:
    - Create and manage conversation sessions
    - Store and retrieve encrypted messages
    - Store emotion vectors for analysis
    - Provide conversation history for LLM context
    - Handle Privacy Mode (no-op storage when enabled)

Usage:
    manager = SessionManager()
    manager.initialize(password="user_password")

    session_id = manager.create_session()

    manager.save_message(
        session_id=session_id,
        role="user",
        content="I feel really sad today",
        emotion_data=smoothed_emotion.to_dict(),
    )

    history = manager.get_conversation_history(session_id)
"""

import sys
import uuid
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from src.storage.database import DatabaseManager
from src.storage.encryption import EncryptionManager

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    """Generate a new UUID string."""
    return str(uuid.uuid4())


class SessionManager:
    """
    Manages all conversation data storage with encryption.

    Combines DatabaseManager and EncryptionManager into a single
    clean interface for the rest of the application.

    One instance per application session. Initialize once at startup
    with the user's password, then use throughout the session.
    """

    def __init__(self, privacy_mode: Optional[bool] = None):
        self.privacy_mode = (
            privacy_mode if privacy_mode is not None else settings.privacy_mode_default
        )

        self._db = DatabaseManager(privacy_mode=self.privacy_mode)
        self._enc = EncryptionManager()
        self._initialized = False

        logger.info(f"SessionManager created | " f"privacy_mode={self.privacy_mode}")

    def initialize(self, password: str) -> bool:
        """
        Initialize storage with the user's password.

        Derives encryption key from password and sets up the database.
        Must be called before any other operations.

        Args:
            password: User's plaintext password for key derivation

        Returns:
            True if initialization succeeded
        """
        try:
            # Derive encryption key
            self._enc.initialize(password)

            # Initialize database schema
            self._db.initialize()

            self._initialized = True
            logger.info("SessionManager initialized successfully")
            return True

        except Exception as e:
            logger.error(f"SessionManager initialization failed: {e}")
            return False

    def initialize_no_password(self):
        """
        Initialize without encryption (Privacy Mode or development).
        Data is stored in plaintext but only in memory.
        Used when privacy_mode=True since in-memory data
        is already ephemeral.
        """
        # Use a fixed development key — acceptable for in-memory only
        self._enc.initialize("dev_ephemeral_key_memory_only")
        self._db.initialize()
        self._initialized = True
        logger.info("SessionManager initialized (no-password / privacy mode)")

    # ── Session Operations ────────────────────────────────────────────────────
    def create_session(
        self,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Create a new conversation session.

        Args:
            metadata: Optional dict with session info
                      (model names, settings, etc.)

        Returns:
            session_id: UUID string for this session
        """
        self._check_initialized()

        session_id = _new_id()
        now = _now_iso()

        session_metadata = metadata or {
            "llm_model": settings.llm_model_name,
            "emotion_model": settings.emotion_model_type,
            "privacy_mode": self.privacy_mode,
        }

        encrypted_metadata = self._enc.encrypt_dict(session_metadata)

        self._db.execute_write(
            """
            INSERT INTO sessions (id, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, now, now, encrypted_metadata),
        )

        logger.info(f"Session created: {session_id[:8]}...")
        return session_id

    def get_session(self, session_id: str) -> Optional[dict]:
        """Retrieve session metadata by ID."""
        self._check_initialized()

        rows = self._db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))

        if not rows:
            return None

        row = rows[0]
        try:
            metadata = self._enc.decrypt_dict(row["metadata"])
        except Exception:
            metadata = {}

        return {
            "id": row["id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "metadata": metadata,
        }

    def list_sessions(self, limit: int = 20) -> list[dict]:
        """Return list of recent sessions (newest first)."""
        self._check_initialized()

        rows = self._db.execute(
            """
            SELECT id, created_at, updated_at
            FROM sessions
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return rows

    def get_first_user_message(self, session_id: str) -> Optional[str]:
        """Return the first user message text for a session, decrypted."""
        self._check_initialized()
        rows = self._db.execute(
            """
            SELECT content FROM messages
            WHERE session_id = ? AND role = 'user'
            ORDER BY turn_number ASC, created_at ASC
            LIMIT 1
            """,
            (session_id,),
        )
        if not rows:
            return None
        try:
            return self._enc.decrypt(rows[0]["content"])
        except Exception:
            return None

    def update_session_timestamp(self, session_id: str):
        """Update the session's updated_at timestamp."""
        self._db.execute_write(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (_now_iso(), session_id),
        )

    # ── Message Operations ────────────────────────────────────────────────────
    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        turn_number: int,
        emotion_data: Optional[dict] = None,
    ) -> str:
        """
        Save a single conversation message with encryption.

        Args:
            session_id:   UUID of the parent session
            role:         "user" or "assistant"
            content:      The message text (will be encrypted)
            turn_number:  Position in conversation (1-indexed)
            emotion_data: Optional emotion detection result dict

        Returns:
            message_id: UUID of the saved message
        """
        self._check_initialized()

        message_id = _new_id()
        now = _now_iso()

        # Encrypt content — no plaintext ever hits the database
        encrypted_content = self._enc.encrypt(content)

        # Encrypt emotion data if provided
        encrypted_emotion = ""
        if emotion_data:
            encrypted_emotion = self._enc.encrypt_dict(emotion_data)

        self._db.execute_write(
            """
            INSERT INTO messages
                (id, session_id, turn_number, role, content, created_at, emotion_data)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                session_id,
                turn_number,
                role,
                encrypted_content,
                now,
                encrypted_emotion,
            ),
        )

        # Update session timestamp
        self.update_session_timestamp(session_id)

        logger.debug(
            f"Message saved | session={session_id[:8]} | "
            f"role={role} | turn={turn_number}"
        )
        return message_id

    def save_emotion_log(
        self,
        session_id: str,
        turn_number: int,
        raw_vector: dict,
        smoothed_vector: dict,
    ):
        """Save detailed emotion vectors for analysis and evaluation."""
        self._check_initialized()

        self._db.execute_write(
            """
            INSERT INTO emotion_log
                (id, session_id, turn_number, raw_vector, smoothed_vector, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                _new_id(),
                session_id,
                turn_number,
                self._enc.encrypt_dict(raw_vector),
                self._enc.encrypt_dict(smoothed_vector),
                _now_iso(),
            ),
        )

    def get_messages(self, session_id: str) -> list[dict]:
        """
        Retrieve and decrypt all messages for a session.

        Returns list of dicts with decrypted content, ordered by turn.
        """
        self._check_initialized()

        rows = self._db.execute(
            """
            SELECT id, session_id, turn_number, role, content, created_at, emotion_data
            FROM messages
            WHERE session_id = ?
            ORDER BY turn_number ASC, created_at ASC
            """,
            (session_id,),
        )

        messages = []
        for row in rows:
            try:
                content = self._enc.decrypt(row["content"])
                emotion = (
                    self._enc.decrypt_dict(row["emotion_data"])
                    if row["emotion_data"]
                    else None
                )
                messages.append(
                    {
                        "id": row["id"],
                        "session_id": row["session_id"],
                        "turn_number": row["turn_number"],
                        "role": row["role"],
                        "content": content,
                        "created_at": row["created_at"],
                        "emotion_data": emotion,
                    }
                )
            except Exception as e:
                logger.error(f"Failed to decrypt message {row['id']}: {e}")
                continue

        return messages

    def get_conversation_history(
        self,
        session_id: str,
        max_turns: int = 10,
    ) -> list[dict]:
        """
        Get conversation history formatted for OllamaClient.

        Returns list of {"role": str, "content": str} dicts,
        suitable for passing directly to OllamaClient.generate()
        as conversation_history.

        Args:
            session_id: UUID of the session
            max_turns:  Maximum number of complete turns to return
                        (one turn = one user + one assistant message)
        """
        messages = self.get_messages(session_id)

        # Format for Ollama — only role and content needed
        formatted = [
            {"role": msg["role"], "content": msg["content"]} for msg in messages
        ]

        # Limit to last max_turns * 2 messages (each turn = 2 messages)
        max_messages = max_turns * 2
        return formatted[-max_messages:]

    def get_session_stats(self, session_id: str) -> dict:
        """
        Return statistics about a session for the UI sidebar.
        """
        self._check_initialized()

        rows = self._db.execute(
            """
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN role='user' THEN 1 ELSE 0 END) as user_count
            FROM messages WHERE session_id = ?
            """,
            (session_id,),
        )

        stats = rows[0] if rows else {"total": 0, "user_count": 0}

        return {
            "total_messages": stats["total"],
            "user_messages": stats["user_count"],
            "turn_count": stats["user_count"],
        }

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and all its messages.
        CASCADE foreign key ensures messages and emotion_log are deleted too.
        """
        self._check_initialized()

        affected = self._db.execute_write(
            "DELETE FROM sessions WHERE id = ?", (session_id,)
        )
        if affected > 0:
            logger.info(f"Session deleted: {session_id[:8]}...")
            return True
        return False

    # ── User Profile ──────────────────────────────────────────────────────────
    def save_user_profile(self, name: str, ram_gb: int):
        """Save (or replace) the single-row user profile."""
        self._check_initialized()
        self._db.execute_write(
            """
            INSERT OR REPLACE INTO user_profile (id, name, ram_gb, created_at)
            VALUES ('profile', ?, ?, ?)
            """,
            (self._enc.encrypt(name), ram_gb, _now_iso()),
        )
        logger.info("User profile saved")

    def get_user_profile(self) -> Optional[dict]:
        """Return the user profile dict or None if not set up yet."""
        self._check_initialized()
        rows = self._db.execute(
            "SELECT name, ram_gb FROM user_profile WHERE id = 'profile'"
        )
        if not rows:
            return None
        row = rows[0]
        try:
            name = self._enc.decrypt(row["name"])
        except Exception:
            name = ""
        return {"name": name, "ram_gb": row["ram_gb"]}

    def close(self):
        """
        Clean shutdown: clear encryption key and close database.
        Call when the application exits.
        """
        self._enc.clear()
        self._db.close()
        logger.info("SessionManager closed")

    # ── Private Helpers ───────────────────────────────────────────────────────
    def _check_initialized(self):
        if not self._initialized:
            raise RuntimeError(
                "SessionManager not initialized. "
                "Call initialize() or initialize_no_password() first."
            )

    @property
    def is_privacy_mode(self) -> bool:
        return self.privacy_mode
