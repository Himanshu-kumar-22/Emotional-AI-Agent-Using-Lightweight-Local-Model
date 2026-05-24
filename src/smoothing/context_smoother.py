"""
src/smoothing/context_smoother.py
==================================
Sliding window context smoother for emotion probability vectors.

Problem solved:
    Raw per-turn emotion predictions are noisy. A single sarcastic
    sentence, an off-topic comment, or an ambiguous phrasing can
    cause the classifier to output a completely different emotion
    even when the user's underlying emotional state hasn't changed.

Solution:
    Maintain a fixed-size window of the last N emotion probability
    vectors. Compute a weighted average where more recent turns
    carry higher weight. Use this smoothed vector as the true
    emotional signal passed to the LLM.

Design decisions:
    - Window size 4: validated in your report as optimal balance
      between responsiveness and stability
    - Exponential-style weights [0.1, 0.2, 0.3, 0.4]: recent turns
      matter more but older context still contributes
    - One smoother instance per session: state resets between sessions
    - Stateless between detect() calls: smoother only updates when
      explicitly told to via update()

Usage:
    smoother = ContextSmoother()

    # After each user message:
    raw_result = detector.detect(user_text)
    smoothed = smoother.update(raw_result)

    print(smoothed.primary_emotion)   # stable label
    print(smoothed.confidence)        # weighted confidence
    print(smoothed.window_size)       # how many turns in window
    print(smoothed.was_smoothed)      # True if smoothing changed label
"""

import sys
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from src.emotion.detector import EmotionResult

logger = logging.getLogger(__name__)


# ── Result Dataclass ──────────────────────────────────────────────────────────
@dataclass
class SmoothedEmotion:
    """
    Result from the context smoothing process.

    Contains both the smoothed output AND the raw input for comparison.
    This allows the UI to show users when smoothing changed the label,
    which contributes to the system's explainability.

    Fields:
        primary_emotion:    Smoothed dominant emotion label
        confidence:         Smoothed confidence for primary emotion
        probabilities:      Full smoothed probability distribution
        raw_emotion:        What the detector said before smoothing
        raw_confidence:     Raw detector confidence before smoothing
        was_smoothed:       True if smoothing changed the primary label
        window_size:        Number of turns currently in the window
        turn_number:        Which conversation turn this is (1-indexed)
    """

    primary_emotion: str
    confidence: float
    probabilities: dict[str, float]
    raw_emotion: str
    raw_confidence: float
    was_smoothed: bool
    window_size: int
    turn_number: int

    def to_dict(self) -> dict:
        """Serialize for database storage."""
        return {
            "primary_emotion": self.primary_emotion,
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "raw_emotion": self.raw_emotion,
            "raw_confidence": round(self.raw_confidence, 4),
            "was_smoothed": self.was_smoothed,
            "window_size": self.window_size,
            "turn_number": self.turn_number,
        }

    def get_display_info(self) -> dict:
        """Get UI display metadata (emoji, color) for the smoothed emotion."""
        return settings.emotion_display.get(
            self.primary_emotion,
            {
                "emoji": "😐",
                "color": "#708090",
                "label": self.primary_emotion.title(),
            },
        )

    def get_stability_description(self) -> str:
        """
        Human-readable description of smoothing effect.
        Used in the Streamlit UI tooltip and debug logs.
        """
        if self.window_size == 1:
            return "First message — no smoothing context yet"
        if not self.was_smoothed:
            return f"Consistent across last {self.window_size} turns"
        return (
            f"Smoothed: raw={self.raw_emotion} → "
            f"context={self.primary_emotion} "
            f"(window={self.window_size} turns)"
        )


# ── Core Smoother ─────────────────────────────────────────────────────────────
class ContextSmoother:
    """
    Maintains a sliding window of emotion probability vectors and
    computes weighted averages to produce stable emotion signals.

    One instance per conversation session. Reset between sessions
    by calling reset() or creating a new instance.

    Thread safety: not thread-safe. Each session should have its
    own ContextSmoother instance (which is the natural usage pattern
    since sessions are independent).
    """

    # Default weights for window positions (oldest → newest)
    # Must sum to 1.0 and length must equal window_size
    DEFAULT_WEIGHTS = [0.1, 0.2, 0.3, 0.4]

    def __init__(
        self,
        window_size: Optional[int] = None,
        weights: Optional[list[float]] = None,
    ):
        """
        Initialize the context smoother.

        Args:
            window_size: Number of turns to keep in the window.
                        Defaults to settings.emotion_smoothing_window (4).
            weights:    Per-turn weights from oldest to newest.
                        Must have length == window_size and sum to 1.0.
                        Defaults to [0.1, 0.2, 0.3, 0.4].
        """
        self.window_size = window_size or settings.emotion_smoothing_window
        self.emotion_labels = settings.emotion_labels
        self.num_labels = len(self.emotion_labels)

        # Validate and set weights
        if weights is not None:
            self._validate_weights(weights, self.window_size)
            self.weights = weights
        else:
            # Use default weights, truncated or extended to match window_size
            self.weights = self._build_default_weights(self.window_size)

        # The sliding window: stores numpy arrays of shape [num_labels]
        # deque with maxlen automatically drops oldest when full
        self._window: deque[np.ndarray] = deque(maxlen=self.window_size)

        # Turn counter for this session
        self._turn_number = 0

        logger.debug(
            f"ContextSmoother initialized | "
            f"window={self.window_size} | "
            f"weights={self.weights} | "
            f"labels={self.emotion_labels}"
        )

    @staticmethod
    def _validate_weights(weights: list[float], window_size: int):
        """Validate weight list is correct length and sums to 1.0."""
        if len(weights) != window_size:
            raise ValueError(
                f"weights length ({len(weights)}) must equal "
                f"window_size ({window_size})"
            )
        weight_sum = sum(weights)
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1.0, got {weight_sum:.6f}")
        if any(w < 0 for w in weights):
            raise ValueError("All weights must be non-negative")

    @staticmethod
    def _build_default_weights(window_size: int) -> list[float]:
        """
        Build default weights for any window size.

        Uses linear spacing: each position gets proportionally more
        weight than the previous. For window_size=4: [0.1, 0.2, 0.3, 0.4].
        For window_size=3: [0.167, 0.333, 0.5] (approximately).

        Always normalized to sum to 1.0.
        """
        if window_size == 4:
            return [0.1, 0.2, 0.3, 0.4]

        # Generate linear weights: position 1 gets weight 1, position 2
        # gets weight 2, etc. Then normalize.
        raw = list(range(1, window_size + 1))
        total = sum(raw)
        return [w / total for w in raw]

    def _extract_probability_vector(self, emotion_result: EmotionResult) -> np.ndarray:
        """
        Extract the probability vector from an EmotionResult as a numpy array.

        Returns array of shape [num_labels] with values in [0, 1].
        Labels are ordered according to settings.emotion_labels.
        """
        vector = np.array(
            [
                emotion_result.probabilities.get(label, 0.0)
                for label in self.emotion_labels
            ],
            dtype=np.float32,
        )
        return vector

    def _compute_weighted_average(self) -> np.ndarray:
        """
        Compute the weighted average of all vectors in the current window.

        If window has fewer turns than window_size (early in conversation),
        we use only the available turns with renormalized weights.

        Returns:
            Smoothed probability vector of shape [num_labels]
        """
        current_window = list(self._window)
        n_available = len(current_window)

        if n_available == 0:
            # Should never happen since we add before smoothing, but be safe
            return np.ones(self.num_labels) / self.num_labels

        if n_available == 1:
            # Only one turn available — return it directly
            return current_window[0].copy()

        # Use the last n_available weights and renormalize
        # If window has 4 slots but only 2 turns, use weights[-2:] normalized
        raw_weights = self.weights[-n_available:]
        weight_sum = sum(raw_weights)
        normalized_weights = [w / weight_sum for w in raw_weights]

        # Compute weighted average
        smoothed = np.zeros(self.num_labels, dtype=np.float32)
        for turn_vector, weight in zip(current_window, normalized_weights):
            smoothed += turn_vector * weight

        return smoothed

    def update(self, emotion_result: EmotionResult) -> SmoothedEmotion:
        """
        Add a new emotion result to the window and return smoothed output.

        This is the main public API. Call once per conversation turn,
        immediately after calling EmotionDetector.detect().

        Args:
            emotion_result: Raw EmotionResult from EmotionDetector

        Returns:
            SmoothedEmotion with weighted-average probabilities
        """
        self._turn_number += 1

        # Extract probability vector from result
        raw_vector = self._extract_probability_vector(emotion_result)

        # Add to window (deque automatically removes oldest if full)
        self._window.append(raw_vector)

        # Compute smoothed vector
        smoothed_vector = self._compute_weighted_average()

        # Build smoothed probability dict
        smoothed_probs = {
            label: float(smoothed_vector[i])
            for i, label in enumerate(self.emotion_labels)
        }

        # Determine smoothed primary emotion
        smoothed_primary = max(smoothed_probs, key=smoothed_probs.get)
        smoothed_confidence = smoothed_probs[smoothed_primary]

        # Detect if smoothing changed the label
        was_smoothed = smoothed_primary != emotion_result.primary_emotion

        if was_smoothed:
            logger.debug(
                f"Turn {self._turn_number}: smoothing changed label "
                f"{emotion_result.primary_emotion} → {smoothed_primary} "
                f"(raw_conf={emotion_result.confidence:.2f}, "
                f"smoothed_conf={smoothed_confidence:.2f})"
            )

        result = SmoothedEmotion(
            primary_emotion=smoothed_primary,
            confidence=smoothed_confidence,
            probabilities=smoothed_probs,
            raw_emotion=emotion_result.primary_emotion,
            raw_confidence=emotion_result.confidence,
            was_smoothed=was_smoothed,
            window_size=len(self._window),
            turn_number=self._turn_number,
        )

        return result

    def reset(self):
        """
        Clear the window and reset turn counter.
        Call at the start of each new conversation session.
        """
        self._window.clear()
        self._turn_number = 0
        logger.debug("ContextSmoother reset — window cleared")

    @property
    def current_window_size(self) -> int:
        """Number of turns currently in the window."""
        return len(self._window)

    @property
    def turn_number(self) -> int:
        """Current turn number in this session."""
        return self._turn_number

    @property
    def is_warmed_up(self) -> bool:
        """
        True when the window is full (has seen at least window_size turns).
        Before warm-up, smoothing uses fewer data points and is less reliable.
        """
        return len(self._window) >= self.window_size

    def get_window_snapshot(self) -> list[dict]:
        """
        Return current window contents as a list of dicts.
        Used for debugging, logging, and the Streamlit debug panel.
        """
        snapshot = []
        for i, vector in enumerate(self._window):
            turn_age = len(self._window) - i  # 1 = most recent
            primary = self.emotion_labels[np.argmax(vector)]
            confidence = float(np.max(vector))
            snapshot.append(
                {
                    "position": i + 1,
                    "turns_ago": turn_age,
                    "primary_emotion": primary,
                    "confidence": round(confidence, 4),
                    "weight": self.weights[-len(self._window) :][i],
                }
            )
        return snapshot

    def get_dominant_emotion_trend(self) -> str:
        """
        Analyze the window to describe the emotional trend.
        Returns a human-readable trend description for the UI.

        Examples:
            "Consistently sad across 4 turns"
            "Shifting from anger to neutral"
            "Mixed emotions: joy and sadness"
        """
        if len(self._window) == 0:
            return "No data yet"

        # Get primary emotion for each turn in window
        turn_emotions = []
        for vector in self._window:
            primary_idx = np.argmax(vector)
            turn_emotions.append(self.emotion_labels[primary_idx])

        unique_emotions = list(dict.fromkeys(turn_emotions))  # Preserve order

        if len(unique_emotions) == 1:
            count = len(turn_emotions)
            return f"Consistently {unique_emotions[0]} across {count} turn{'s' if count > 1 else ''}"

        if len(unique_emotions) == 2:
            return f"Shifting from {unique_emotions[0]} to {unique_emotions[-1]}"

        emotions_str = ", ".join(unique_emotions[:-1]) + f" and {unique_emotions[-1]}"
        return f"Mixed emotions: {emotions_str}"
