"""
tests/test_storage.py
======================
Tests for encryption, database, and session management.

Run with:
    pytest tests/test_storage.py -v
"""

import sys
import pytest
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.storage.encryption import EncryptionManager
from src.storage.database import DatabaseManager
from src.storage.session_manager import SessionManager


# ── Encryption Tests ──────────────────────────────────────────────────────────
class TestEncryptionManager:

    def test_initialize_with_password(self):
        enc = EncryptionManager()
        enc.initialize("test_password")
        assert enc.is_initialized

    def test_not_initialized_by_default(self):
        enc = EncryptionManager()
        assert not enc.is_initialized

    def test_encrypt_decrypt_roundtrip(self):
        enc = EncryptionManager()
        enc.initialize("my_password")
        original = "Hello, this is sensitive data!"
        encrypted = enc.encrypt(original)
        decrypted = enc.decrypt(encrypted)
        assert decrypted == original

    def test_encrypted_differs_from_plaintext(self):
        enc = EncryptionManager()
        enc.initialize("password")
        plaintext = "sensitive message"
        encrypted = enc.encrypt(plaintext)
        assert encrypted != plaintext
        assert plaintext not in encrypted

    def test_same_plaintext_different_ciphertext(self):
        """Each encryption of the same text should produce different output (random IV)."""
        enc = EncryptionManager()
        enc.initialize("password")
        text = "same message every time"
        enc1 = enc.encrypt(text)
        enc2 = enc.encrypt(text)
        assert enc1 != enc2  # Different IVs → different ciphertexts

    def test_wrong_key_raises_on_decrypt(self):
        enc1 = EncryptionManager()
        enc1.initialize("correct_password")
        encrypted = enc1.encrypt("secret data")

        enc2 = EncryptionManager()
        enc2.initialize("wrong_password")
        with pytest.raises(ValueError):
            enc2.decrypt(encrypted)

    def test_encrypt_empty_string(self):
        enc = EncryptionManager()
        enc.initialize("password")
        assert enc.encrypt("") == ""
        assert enc.decrypt("") == ""

    def test_encrypt_unicode(self):
        enc = EncryptionManager()
        enc.initialize("password")
        text = "I feel 😢 and 😔 today. 你好世界"
        assert enc.decrypt(enc.encrypt(text)) == text

    def test_encrypt_dict_roundtrip(self):
        enc = EncryptionManager()
        enc.initialize("password")
        data = {"emotion": "sadness", "confidence": 0.94, "turn": 3}
        assert enc.decrypt_dict(enc.encrypt_dict(data)) == data

    def test_clear_removes_key(self):
        enc = EncryptionManager()
        enc.initialize("password")
        assert enc.is_initialized
        enc.clear()
        assert not enc.is_initialized

    def test_encrypt_without_init_raises(self):
        enc = EncryptionManager()
        with pytest.raises(RuntimeError):
            enc.encrypt("something")

    def test_long_text_roundtrip(self):
        enc = EncryptionManager()
        enc.initialize("password")
        long_text = "I feel overwhelmed. " * 200
        assert enc.decrypt(enc.encrypt(long_text)) == long_text


# ── Database Tests ─────────────────────────────────────────────────────────────
class TestDatabaseManager:

    def test_in_memory_initialization(self):
        db = DatabaseManager(privacy_mode=True)
        db.initialize()
        assert db.is_privacy_mode

    def test_schema_creates_tables(self):
        db = DatabaseManager(privacy_mode=True)
        db.initialize()
        rows = db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        table_names = [r["name"] for r in rows]
        assert "sessions" in table_names
        assert "messages" in table_names
        assert "emotion_log" in table_names

    def test_insert_and_retrieve(self):
        db = DatabaseManager(privacy_mode=True)
        db.initialize()
        db.execute_write(
            "INSERT INTO sessions (id, created_at, updated_at, metadata) "
            "VALUES (?, ?, ?, ?)",
            ("test-id", "2024-01-01", "2024-01-01", "{}"),
        )
        rows = db.execute("SELECT * FROM sessions WHERE id = ?", ("test-id",))
        assert len(rows) == 1
        assert rows[0]["id"] == "test-id"


# ── SessionManager Tests ──────────────────────────────────────────────────────
class TestSessionManager:

    @pytest.fixture
    def manager(self):
        """Fresh in-memory session manager for each test."""
        m = SessionManager(privacy_mode=True)
        m.initialize_no_password()
        yield m
        m.close()

    def test_create_session_returns_id(self, manager):
        session_id = manager.create_session()
        assert isinstance(session_id, str)
        assert len(session_id) == 36  # UUID format

    def test_get_session_returns_metadata(self, manager):
        session_id = manager.create_session({"test_key": "test_value"})
        session = manager.get_session(session_id)
        assert session is not None
        assert session["id"] == session_id
        assert session["metadata"]["test_key"] == "test_value"

    def test_save_and_retrieve_message(self, manager):
        session_id = manager.create_session()
        manager.save_message(
            session_id=session_id,
            role="user",
            content="I feel really happy today!",
            turn_number=1,
        )
        messages = manager.get_messages(session_id)
        assert len(messages) == 1
        assert messages[0]["content"] == "I feel really happy today!"
        assert messages[0]["role"] == "user"

    def test_message_content_encrypted_in_db(self, manager):
        """Verify raw DB content is not plaintext."""
        session_id = manager.create_session()
        secret = "my deeply personal secret"
        manager.save_message(session_id, "user", secret, 1)

        # Query raw database — should NOT contain plaintext
        raw_rows = manager._db.execute(
            "SELECT content FROM messages WHERE session_id = ?", (session_id,)
        )
        raw_content = raw_rows[0]["content"]
        assert secret not in raw_content
        assert raw_content != secret

    def test_emotion_data_stored_and_retrieved(self, manager):
        session_id = manager.create_session()
        emotion = {"primary_emotion": "sadness", "confidence": 0.92}
        manager.save_message(session_id, "user", "I feel sad", 1, emotion_data=emotion)
        messages = manager.get_messages(session_id)
        assert messages[0]["emotion_data"]["primary_emotion"] == "sadness"
        assert messages[0]["emotion_data"]["confidence"] == 0.92

    def test_conversation_history_format(self, manager):
        session_id = manager.create_session()
        manager.save_message(session_id, "user", "Hello", 1)
        manager.save_message(session_id, "assistant", "Hi there!", 1)
        manager.save_message(session_id, "user", "How are you?", 2)

        history = manager.get_conversation_history(session_id)
        assert len(history) == 3
        assert all("role" in h and "content" in h for h in history)
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_history_limited_by_max_turns(self, manager):
        session_id = manager.create_session()
        for i in range(1, 16):
            manager.save_message(session_id, "user", f"message {i}", i)
            manager.save_message(session_id, "assistant", f"reply {i}", i)

        history = manager.get_conversation_history(session_id, max_turns=5)
        assert len(history) <= 10  # 5 turns × 2 messages

    def test_multiple_sessions_independent(self, manager):
        sid1 = manager.create_session()
        sid2 = manager.create_session()

        manager.save_message(sid1, "user", "Session 1 message", 1)
        manager.save_message(sid2, "user", "Session 2 message", 1)

        msgs1 = manager.get_messages(sid1)
        msgs2 = manager.get_messages(sid2)

        assert len(msgs1) == 1
        assert len(msgs2) == 1
        assert msgs1[0]["content"] == "Session 1 message"
        assert msgs2[0]["content"] == "Session 2 message"

    def test_session_stats(self, manager):
        session_id = manager.create_session()
        for i in range(1, 4):
            manager.save_message(session_id, "user", f"user msg {i}", i)
            manager.save_message(session_id, "assistant", f"reply {i}", i)

        stats = manager.get_session_stats(session_id)
        assert stats["total_messages"] == 6
        assert stats["user_messages"] == 3
        assert stats["turn_count"] == 3

    def test_delete_session(self, manager):
        session_id = manager.create_session()
        manager.save_message(session_id, "user", "test", 1)
        assert manager.get_session(session_id) is not None

        manager.delete_session(session_id)
        assert manager.get_session(session_id) is None
        # Messages should be cascade deleted
        assert len(manager.get_messages(session_id)) == 0

    def test_uninitialized_raises(self):
        m = SessionManager(privacy_mode=True)
        with pytest.raises(RuntimeError):
            m.create_session()

    def test_unicode_content_preserved(self, manager):
        session_id = manager.create_session()
        text = "I feel 😢 今日は悲しい nadzieja"
        manager.save_message(session_id, "user", text, 1)
        messages = manager.get_messages(session_id)
        assert messages[0]["content"] == text
