"""
src/storage/encryption.py
==========================
AES-256 encryption and PBKDF2 key derivation for local data storage.

Security design:
    - AES-256-CBC: symmetric encryption, NIST-endorsed, fast for local use
    - PBKDF2-HMAC-SHA256: password-based key derivation
      * 390,000 iterations (NIST 2023 recommendation for SHA-256)
      * Random 16-byte salt per key derivation
      * Makes brute-force attacks computationally infeasible
    - Random 16-byte IV per encryption operation
      * Ensures identical plaintexts produce different ciphertexts
      * Prevents pattern analysis on encrypted data
    - PKCS7 padding: standard padding for block cipher alignment

Storage format for each encrypted value:
    base64( salt[16] + iv[16] + ciphertext[variable] )

    The salt and IV are stored alongside the ciphertext because:
    - Salt: needed to re-derive the same key from the password
    - IV: needed to decrypt the specific ciphertext block
    Neither the salt nor the IV needs to be secret —
    their purpose is randomness, not secrecy.

Key storage:
    The derived key is cached in memory for the session duration.
    It is stored in the OS keychain (via keyring library) between
    sessions so the user does not need to re-enter their password.
    The raw key is NEVER written to disk in plaintext form.
"""

import os
import sys
import base64
import hashlib
import logging
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings

logger = logging.getLogger(__name__)

# AES block size is always 128 bits (16 bytes) regardless of key size
AES_BLOCK_SIZE = 16
# AES-256 key size: 256 bits = 32 bytes
KEY_SIZE = 32
# Salt size for PBKDF2
SALT_SIZE = 16
# IV size for AES-CBC
IV_SIZE = 16


class EncryptionManager:
    """
    Handles all cryptographic operations for the storage layer.

    One instance per application session. Holds the derived key
    in memory so encryption/decryption operations do not require
    re-deriving the key on every call.

    Usage:
        # Initialize with user password
        enc = EncryptionManager()
        enc.initialize("user_password")

        # Encrypt data
        encrypted = enc.encrypt("sensitive message")

        # Decrypt data
        original = enc.decrypt(encrypted)

        # Check if initialized
        if enc.is_initialized:
            ...
    """

    def __init__(self):
        # The derived AES key — held in memory only
        self._key: Optional[bytes] = None
        self._backend = default_backend()
        logger.debug("EncryptionManager created (not yet initialized)")

    @property
    def is_initialized(self) -> bool:
        """True if a key has been derived and is ready for use."""
        return self._key is not None

    def initialize(self, password: str, salt: Optional[bytes] = None) -> bytes:
        """
        Derive an AES-256 key from the user's password using PBKDF2.

        This is called once at session start. The derived key is cached
        in self._key for the session duration.

        Args:
            password: The user's plaintext password
            salt:     Optional existing salt (for re-deriving same key).
                      If None, generates a new random salt.
                      Pass existing salt when loading an existing session.

        Returns:
            The salt used for key derivation (must be stored to re-derive
            the same key in future sessions).
        """
        if not password:
            raise ValueError("Password cannot be empty")

        # Generate new salt if not provided
        if salt is None:
            salt = os.urandom(SALT_SIZE)

        # PBKDF2-HMAC-SHA256 key derivation
        # iterations=390_000 is NIST's 2023 recommendation for SHA-256
        # This means each password guess takes ~0.5s on modern hardware
        # making brute-force attacks infeasible for even weak passwords
        self._key = hashlib.pbkdf2_hmac(
            hash_name="sha256",
            password=password.encode("utf-8"),
            salt=salt,
            iterations=settings.pbkdf2_iterations,
            dklen=KEY_SIZE,
        )

        logger.info(
            f"Encryption key derived | "
            f"iterations={settings.pbkdf2_iterations} | "
            f"key_size={KEY_SIZE * 8}bit"
        )
        return salt

    def initialize_from_key(self, key: bytes):
        """
        Initialize directly from a pre-derived key.
        Used when the key is retrieved from the OS keychain.
        """
        if len(key) != KEY_SIZE:
            raise ValueError(f"Key must be {KEY_SIZE} bytes, got {len(key)}")
        self._key = key

    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt a string using AES-256-CBC.

        Args:
            plaintext: The string to encrypt

        Returns:
            Base64-encoded string containing: salt + iv + ciphertext
            This single string contains everything needed for decryption.

        Raises:
            RuntimeError: If encryption manager is not initialized
        """
        if not self.is_initialized:
            raise RuntimeError(
                "EncryptionManager not initialized. Call initialize() first."
            )

        if not plaintext:
            return ""

        # Generate a fresh random IV for each encryption operation
        # This ensures identical plaintexts produce different ciphertexts
        iv = os.urandom(IV_SIZE)

        # PKCS7 padding to align plaintext to AES block boundary
        padder = padding.PKCS7(AES_BLOCK_SIZE * 8).padder()
        padded_data = padder.update(plaintext.encode("utf-8")) + padder.finalize()

        # AES-256-CBC encryption
        cipher = Cipher(
            algorithms.AES(self._key),
            modes.CBC(iv),
            backend=self._backend,
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()

        # Pack: iv + ciphertext → base64
        # Note: we don't pack the salt here because the salt is stored
        # once per session in the sessions table, not per-message
        packed = iv + ciphertext
        return base64.b64encode(packed).decode("utf-8")

    def decrypt(self, encrypted_b64: str) -> str:
        """
        Decrypt a base64-encoded AES-256-CBC ciphertext.

        Args:
            encrypted_b64: Base64 string from encrypt()

        Returns:
            Original plaintext string

        Raises:
            RuntimeError: If not initialized
            ValueError: If ciphertext is malformed
        """
        if not self.is_initialized:
            raise RuntimeError(
                "EncryptionManager not initialized. Call initialize() first."
            )

        if not encrypted_b64:
            return ""

        try:
            # Decode base64
            packed = base64.b64decode(encrypted_b64.encode("utf-8"))

            # Unpack: first IV_SIZE bytes are the IV, rest is ciphertext
            if len(packed) < IV_SIZE + AES_BLOCK_SIZE:
                raise ValueError("Ciphertext too short — likely corrupted")

            iv = packed[:IV_SIZE]
            ciphertext = packed[IV_SIZE:]

            # AES-256-CBC decryption
            cipher = Cipher(
                algorithms.AES(self._key),
                modes.CBC(iv),
                backend=self._backend,
            )
            decryptor = cipher.decryptor()
            padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()

            # Remove PKCS7 padding
            unpadder = padding.PKCS7(AES_BLOCK_SIZE * 8).unpadder()
            plaintext_bytes = unpadder.update(padded_plaintext) + unpadder.finalize()

            return plaintext_bytes.decode("utf-8")

        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise ValueError(
                f"Decryption failed — wrong password or corrupted data: {e}"
            )

    def encrypt_dict(self, data: dict) -> str:
        """Serialize a dict to JSON and encrypt it."""
        import json

        return self.encrypt(json.dumps(data, ensure_ascii=False))

    def decrypt_dict(self, encrypted_b64: str) -> dict:
        """Decrypt and deserialize a JSON dict."""
        import json

        if not encrypted_b64:
            return {}
        decrypted = self.decrypt(encrypted_b64)
        return json.loads(decrypted)

    def clear(self):
        """
        Clear the key from memory.
        Call when the session ends to minimize key exposure time.
        """
        self._key = None
        logger.debug("Encryption key cleared from memory")


# ── Module-level default instance ─────────────────────────────────────────────
# Shared across the storage layer. Initialized by SessionManager at startup.
encryption_manager = EncryptionManager()
