"""
config/settings.py
==================
Centralized configuration management for the Emotional AI Agent.

This module is the single source of truth for all configuration values.
Every other module imports from here — never reads .env directly.

Design Pattern: Configuration as a typed dataclass loaded once at startup.
This means:
  - Type safety: wrong config types fail loudly at startup, not silently at runtime
  - Single import: `from config.settings import settings` works everywhere
  - Testability: tests can override settings without touching .env files
  - Documentation: the dataclass definition IS the documentation

Cross-Platform Notes:
  - All paths use pathlib.Path for OS-agnostic path handling
  - Device detection handles MPS (Mac), CUDA (Linux/Windows), and CPU fallbacks
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal
from dotenv import load_dotenv

# ── Resolve project root ──────────────────────────────────────────────────────
# __file__ is config/settings.py
# .parent is config/
# .parent.parent is the project root (emotional-ai-agent/)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Load .env from project root
# override=False means existing environment variables take priority over .env
# This allows CI/CD systems to inject values without a .env file
load_dotenv(PROJECT_ROOT / ".env", override=False)


# ── Helper: read env with type coercion ──────────────────────────────────────
def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key, str(default)).lower()
    return val in ("true", "1", "yes", "on")


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


# ── Device Detection ─────────────────────────────────────────────────────────
def _detect_device() -> str:
    """
    Automatically detect the best available compute device.

    Priority order:
      1. CUDA  — NVIDIA GPU (Linux/Windows machines)
      2. MPS   — Apple Silicon GPU (M1/M2/M3 Macs)
      3. CPU   — Universal fallback

    This function is called once at startup. The result is stored in
    AppConfig.device and used everywhere that needs a torch device.

    Cross-platform: this function works identically on all platforms.
    The torch library handles the platform-specific implementation.
    """
    device_override = _env_str("DEVICE", "auto").lower()

    if device_override != "auto":
        return device_override

    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            logging.info(f"[Device] CUDA GPU detected: {gpu_name}")
            return device

        # MPS check: Apple Silicon Metal Performance Shaders
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logging.info("[Device] Apple Silicon MPS detected — using Metal GPU")
            return "mps"

        logging.info("[Device] No GPU detected — using CPU")
        return "cpu"

    except ImportError:
        logging.warning("[Device] PyTorch not found — defaulting to cpu")
        return "cpu"


# ── Application Configuration Dataclass ──────────────────────────────────────
@dataclass
class AppConfig:
    """
    Complete application configuration.

    All values come from environment variables with sensible defaults.
    Instantiated once as a module-level singleton (see bottom of file).

    Usage anywhere in the codebase:
        from config.settings import settings
        model_path = settings.distilbert_model_path
        device = settings.device
    """

    # ── Application ────────────────────────────────────────────────────────
    app_name: str = field(
        default_factory=lambda: _env_str("APP_NAME", "Emotional AI Agent")
    )
    app_version: str = field(
        default_factory=lambda: _env_str("APP_VERSION", "1.0.0")
    )
    debug: bool = field(
        default_factory=lambda: _env_bool("DEBUG", False)
    )
    log_level: str = field(
        default_factory=lambda: _env_str("LOG_LEVEL", "INFO")
    )

    # ── Paths (all resolved relative to PROJECT_ROOT) ──────────────────────
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)

    data_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data"
    )
    raw_data_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "raw"
    )
    processed_data_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "processed"
    )
    models_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "models"
    )
    distilbert_model_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "models" / "emotion" / "distilbert-emotion"
    )
    minilm_model_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "models" / "emotion" / "minilm-emotion"
    )

    # Database path: supports both absolute paths and relative-to-root paths
    database_path: Path = field(
        default_factory=lambda: PROJECT_ROOT
        / _env_str("DATABASE_PATH", "data/conversations.db")
    )

    # ── Emotion Detection ──────────────────────────────────────────────────
    emotion_model_type: Literal["distilbert", "minilm"] = field(
        default_factory=lambda: _env_str("EMOTION_MODEL_TYPE", "distilbert")
    )
    emotion_confidence_threshold: float = field(
        default_factory=lambda: _env_float("EMOTION_CONFIDENCE_THRESHOLD", 0.3)
    )
    emotion_smoothing_window: int = field(
        default_factory=lambda: _env_int("EMOTION_SMOOTHING_WINDOW", 4)
    )
    num_emotion_classes: int = field(
        default_factory=lambda: _env_int("NUM_EMOTION_CLASSES", 8)
    )

    # The 8 coarse emotion labels (27 GoEmotions labels merged into these)
    emotion_labels: list = field(
        default_factory=lambda: [
            "joy",
            "sadness",
            "anger",
            "fear",
            "surprise",
            "disgust",
            "trust",
            "neutral",
        ]
    )

    # Emoji and color mapping for the Streamlit UI emotion badges
    emotion_display: dict = field(
        default_factory=lambda: {
            "joy":      {"emoji": "😊", "color": "#FFD700", "label": "Joy"},
            "sadness":  {"emoji": "😢", "color": "#4169E1", "label": "Sadness"},
            "anger":    {"emoji": "😠", "color": "#DC143C", "label": "Anger"},
            "fear":     {"emoji": "😨", "color": "#8B008B", "label": "Fear"},
            "surprise": {"emoji": "😮", "color": "#FF8C00", "label": "Surprise"},
            "disgust":  {"emoji": "🤢", "color": "#556B2F", "label": "Disgust"},
            "trust":    {"emoji": "🤝", "color": "#20B2AA", "label": "Trust"},
            "neutral":  {"emoji": "😐", "color": "#708090", "label": "Neutral"},
        }
    )

    # ── LLM / Ollama ───────────────────────────────────────────────────────
    llm_model_name: str = field(
        default_factory=lambda: _env_str("LLM_MODEL_NAME", "phi3:mini")
    )
    ollama_base_url: str = field(
        default_factory=lambda: _env_str("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    ollama_timeout: int = field(
        default_factory=lambda: _env_int("OLLAMA_TIMEOUT", 120)
    )
    llm_max_tokens: int = field(
        default_factory=lambda: _env_int("LLM_MAX_TOKENS", 300)
    )
    llm_temperature: float = field(
        default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.7)
    )

    # ── Storage ────────────────────────────────────────────────────────────
    privacy_mode_default: bool = field(
        default_factory=lambda: _env_bool("PRIVACY_MODE_DEFAULT", False)
    )
    # PBKDF2 iterations — NIST recommends minimum 310,000 for SHA-256 as of 2023
    pbkdf2_iterations: int = field(
        default_factory=lambda: _env_int("PBKDF2_ITERATIONS", 390_000)
    )

    # ── Training ───────────────────────────────────────────────────────────
    train_batch_size: int = field(
        default_factory=lambda: _env_int("TRAIN_BATCH_SIZE", 32)
    )
    train_epochs: int = field(
        default_factory=lambda: _env_int("TRAIN_EPOCHS", 5)
    )
    train_learning_rate: float = field(
        default_factory=lambda: _env_float("TRAIN_LEARNING_RATE", 2e-5)
    )
    train_max_length: int = field(
        default_factory=lambda: _env_int("TRAIN_MAX_LENGTH", 128)
    )
    train_weight_decay: float = field(
        default_factory=lambda: _env_float("TRAIN_WEIGHT_DECAY", 0.01)
    )
    validation_split: float = field(
        default_factory=lambda: _env_float("VALIDATION_SPLIT", 0.1)
    )
    early_stopping_patience: int = field(
        default_factory=lambda: _env_int("EARLY_STOPPING_PATIENCE", 2)
    )

    # ── Hardware ───────────────────────────────────────────────────────────
    device: str = field(default_factory=_detect_device)

    def __post_init__(self):
        """
        Post-initialization validation and directory creation.

        Called automatically by Python's dataclass machinery after __init__.
        Ensures all required directories exist so modules never need to
        create directories themselves.
        """
        # Ensure all required directories exist
        # exist_ok=True means no error if directory already exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.distilbert_model_path.mkdir(parents=True, exist_ok=True)
        self.minilm_model_path.mkdir(parents=True, exist_ok=True)
        # Ensure database directory exists
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        # Configure logging based on settings
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper(), logging.INFO),
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def get_emotion_model_path(self) -> Path:
        """Return the correct model path based on configured emotion_model_type."""
        if self.emotion_model_type == "distilbert":
            return self.distilbert_model_path
        return self.minilm_model_path

    def is_model_trained(self) -> bool:
        """
        Check if the emotion model has been trained and saved locally.
        Looks for the HuggingFace model config file as the presence indicator.
        """
        model_path = self.get_emotion_model_path()
        return (model_path / "config.json").exists()

    def __repr__(self) -> str:
        """Clean string representation for logging/debugging."""
        return (
            f"AppConfig(\n"
            f"  app_name={self.app_name!r},\n"
            f"  device={self.device!r},\n"
            f"  emotion_model={self.emotion_model_type!r},\n"
            f"  llm_model={self.llm_model_name!r},\n"
            f"  privacy_mode={self.privacy_mode_default},\n"
            f"  project_root={self.project_root}\n"
            f")"
        )


# ── Module-level singleton ────────────────────────────────────────────────────
# Instantiated once when the module is first imported.
# All other modules do: from config.settings import settings
# This is the standard Python singleton pattern for configuration.
settings = AppConfig()