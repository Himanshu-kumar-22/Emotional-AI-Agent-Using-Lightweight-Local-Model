"""
src/emotion/detector.py
=======================
Production inference class for emotion detection.

This is the ONLY class that the agent pipeline and Streamlit UI interact with.
Everything else in src/emotion/ is training infrastructure.

Design principles:
  - Single responsibility: detect emotion from text, return structured result
  - Lazy loading: model loads on first call, not at import time
  - Stateless: no conversation state stored here (that's ContextSmoother's job)
  - Graceful degradation: if model not trained, returns neutral with low confidence

Usage:
    from src.emotion.detector import EmotionDetector

    detector = EmotionDetector()
    result = detector.detect("I'm really struggling today and feeling overwhelmed")

    print(result.primary_emotion)     # "sadness"
    print(result.confidence)          # 0.87
    print(result.probabilities)       # {"joy": 0.02, "sadness": 0.87, ...}
    print(result.all_emotions)        # ["sadness"]  (emotions above threshold)
"""

import sys
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings

logger = logging.getLogger(__name__)


# ── Result Dataclass ──────────────────────────────────────────────────────────
@dataclass
class EmotionResult:
    """
    Structured result from emotion detection.

    Using a dataclass rather than a plain dict provides:
      - Type hints and IDE autocomplete
      - Immutability option (frozen=True)
      - Clean repr for logging/debugging
      - Easy serialization to dict for storage

    Fields:
        primary_emotion: The highest-probability coarse emotion label
        confidence:      Probability of the primary emotion (0.0 - 1.0)
        probabilities:   Full probability distribution over all 8 classes
        all_emotions:    List of emotions above the confidence threshold
        raw_text:        The input text (for context in logs/storage)
        inference_ms:    How long the inference took (for benchmarking)
        model_type:      Which model was used (distilbert or minilm)
    """

    primary_emotion: str
    confidence: float
    probabilities: dict[str, float]
    all_emotions: list[str]
    raw_text: str
    inference_ms: float
    model_type: str

    def to_dict(self) -> dict:
        """Serialize to dictionary for database storage."""
        return {
            "primary_emotion": self.primary_emotion,
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "all_emotions": self.all_emotions,
            "inference_ms": round(self.inference_ms, 2),
            "model_type": self.model_type,
        }

    def get_display_info(self) -> dict:
        """Get emotion display metadata (emoji, color) for UI rendering."""
        return settings.emotion_display.get(
            self.primary_emotion,
            {"emoji": "😐", "color": "#708090", "label": self.primary_emotion.title()},
        )

    @classmethod
    def neutral_fallback(cls, text: str = "", reason: str = "") -> "EmotionResult":
        """
        Create a neutral EmotionResult for fallback scenarios.
        Used when the model isn't loaded or inference fails.
        """
        if reason:
            logger.warning(f"Using neutral fallback: {reason}")
        return cls(
            primary_emotion="neutral",
            confidence=0.5,
            probabilities={
                label: (0.5 if label == "neutral" else 0.0)
                for label in settings.emotion_labels
            },
            all_emotions=["neutral"],
            raw_text=text,
            inference_ms=0.0,
            model_type="fallback",
        )


# ── Detector ──────────────────────────────────────────────────────────────────
class EmotionDetector:
    """
    Loads a fine-tuned transformer model and performs emotion inference.

    Handles:
      - Model loading from local checkpoint
      - Text preprocessing (same pipeline as training)
      - Inference with sigmoid thresholding
      - Graceful fallback if model not available
      - Inference latency tracking

    The model is loaded lazily on first call to detect() to avoid
    slowing down application startup.
    """

    def __init__(
        self,
        model_type: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
    ):
        self.model_type = model_type or settings.emotion_model_type
        self.confidence_threshold = (
            confidence_threshold or settings.emotion_confidence_threshold
        )
        self.device = torch.device(settings.device)

        # Lazy-loaded components (None until first detect() call)
        self._model = None
        self._tokenizer = None
        self._emotion_labels = settings.emotion_labels
        self._is_loaded = False

        logger.info(
            f"EmotionDetector initialized | "
            f"model={self.model_type} | "
            f"device={self.device} | "
            f"threshold={self.confidence_threshold}"
        )

    def _get_model_path(self) -> Path:
        """Get the correct model path based on configured type."""
        if self.model_type == "distilbert":
            return settings.distilbert_model_path
        return settings.minilm_model_path

    def _get_base_model_id(self) -> str:
        """Get the HuggingFace model ID for the tokenizer."""
        if self.model_type == "distilbert":
            return "distilbert-base-uncased"
        return "sentence-transformers/all-MiniLM-L6-v2"

    def _ensure_model_downloaded(self):
        """
        Check if the model exists locally. If not, attempt to download
        from HuggingFace Hub. Runs transparently on first launch.

        This is the mechanism that allows users to run the app without
        training anything themselves. You (the developer) train once,
        upload to HuggingFace, and every user downloads automatically.

        No HuggingFace token required — public repos are freely readable.
        """
        model_path = self._get_model_path()

        # Model already exists locally — nothing to do
        if (model_path / "config.json").exists():
            return

        # Determine which HF repo to pull from
        if self.model_type == "distilbert":
            repo_id = settings.hf_distilbert_repo
        else:
            repo_id = settings.hf_minilm_repo

        if not repo_id:
            # No repo configured — user must train manually
            logger.warning(
                f"Model '{self.model_type}' not found locally and no "
                f"HuggingFace repo configured in .env.\n"
                f"Option 1 — Train locally:\n"
                f"  python3 scripts/train_emotion_model.py --model {self.model_type}\n"
                f"Option 2 — Set HF repo in .env:\n"
                f"  HF_DISTILBERT_REPO=username/repo-name"
            )
            return

        logger.info(
            f"Model '{self.model_type}' not found locally.\n"
            f"Downloading from HuggingFace: {repo_id}\n"
            f"This is a one-time download. Future runs use the local copy."
        )

        try:
            from huggingface_hub import snapshot_download

            model_path.mkdir(parents=True, exist_ok=True)

            snapshot_download(
                repo_id=repo_id,
                local_dir=str(model_path),
                # Skip non-PyTorch weight formats to save bandwidth
                ignore_patterns=[
                    "*.msgpack",
                    "flax_model*",
                    "tf_model*",
                    "rust_model*",
                ],
            )
            logger.info(f"✓ Model downloaded successfully to {model_path}")

        except Exception as e:
            logger.error(
                f"Auto-download failed for '{self.model_type}': {e}\n"
                f"Train manually: "
                f"python3 scripts/train_emotion_model.py --model {self.model_type}"
            )

    def load_model(self) -> bool:
        """
        Load the fine-tuned model from disk.
        Returns True if loaded successfully, False otherwise.
        """
        from transformers import AutoTokenizer, AutoConfig
        from src.emotion.trainer import EmotionClassificationModel

        # ← This line is the only change to the existing load_model method
        self._ensure_model_downloaded()

        model_path = self._get_model_path()

        if not (model_path / "config.json").exists():
            logger.warning(
                f"No trained model found at {model_path}\n"
                f"Run: python3 scripts/train_emotion_model.py"
            )
            return False

        try:
            logger.info(f"Loading emotion model from {model_path}")
            start = time.time()

            self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))

            self._model = EmotionClassificationModel(
                model_id=self._get_base_model_id(),
                num_labels=len(self._emotion_labels),
            )

            state_dict = torch.load(
                model_path / "pytorch_model.bin",
                map_location=self.device,
                weights_only=True,
            )
            self._model.load_state_dict(state_dict)
            self._model.to(self.device)
            self._model.eval()

            elapsed = (time.time() - start) * 1000
            self._is_loaded = True
            logger.info(f"Model loaded in {elapsed:.0f}ms")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def detect(self, text: str) -> EmotionResult:
        """
        Detect emotions in the given text.

        This is the main public API. All other methods are implementation details.

        Args:
            text: Raw user input text (not pre-cleaned)

        Returns:
            EmotionResult with primary emotion, confidence, and full distribution
        """
        if not text or not text.strip():
            return EmotionResult.neutral_fallback(text, "empty input")

        # Lazy load model on first call
        if not self._is_loaded:
            success = self.load_model()
            if not success:
                return EmotionResult.neutral_fallback(text, "model not loaded")

        start_time = time.time()

        try:
            # Preprocess text (same operations as training preprocessor)
            clean_text = self._preprocess(text)

            # Tokenize
            inputs = self._tokenizer(
                clean_text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=settings.train_max_length,
            )

            # Move to device
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            # Inference (no gradient computation needed)
            with torch.no_grad():
                logits = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

            # Convert logits to probabilities via sigmoid
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]

            inference_ms = (time.time() - start_time) * 1000

            return self._build_result(text, probabilities, inference_ms)

        except Exception as e:
            logger.error(f"Inference error: {e}")
            return EmotionResult.neutral_fallback(text, f"inference error: {e}")

    def _preprocess(self, text: str) -> str:
        """Apply the same text cleaning used during training."""
        import re

        text = text.lower()
        text = re.sub(r"http\S+|www\.\S+", "", text)
        text = re.sub(r"\[name\]", "person", text)
        text = re.sub(r"\ufffd", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _build_result(
        self,
        original_text: str,
        probabilities: np.ndarray,
        inference_ms: float,
    ) -> EmotionResult:
        """Build an EmotionResult from raw probability array."""

        # Build probability dictionary
        prob_dict = {
            label: float(prob)
            for label, prob in zip(self._emotion_labels, probabilities)
        }

        # Primary emotion: highest probability
        primary_emotion = max(prob_dict, key=prob_dict.get)
        primary_confidence = prob_dict[primary_emotion]

        # All active emotions: above threshold
        active_emotions = [
            label
            for label, prob in prob_dict.items()
            if prob >= self.confidence_threshold
        ]

        # Always include at least the primary emotion
        if not active_emotions:
            active_emotions = [primary_emotion]

        return EmotionResult(
            primary_emotion=primary_emotion,
            confidence=primary_confidence,
            probabilities=prob_dict,
            all_emotions=active_emotions,
            raw_text=original_text,
            inference_ms=inference_ms,
            model_type=self.model_type,
        )

    @property
    def is_ready(self) -> bool:
        """Check if the model is loaded and ready for inference."""
        return self._is_loaded
