"""
tests/test_context_smoother.py
================================
Unit tests for the ContextSmoother module.

Run with:
    pytest tests/test_context_smoother.py -v
"""

import sys
import pytest
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.smoothing.context_smoother import ContextSmoother, SmoothedEmotion
from src.emotion.detector import EmotionResult
from config.settings import settings


# ── Helpers ───────────────────────────────────────────────────────────────────
def make_emotion_result(primary: str, confidence: float = 0.9) -> EmotionResult:
    """
    Create a mock EmotionResult for testing.
    Sets primary emotion to confidence, all others to near-zero.
    """
    probs = {label: 0.01 for label in settings.emotion_labels}
    probs[primary] = confidence
    # Normalize remaining probability
    remaining = 1.0 - confidence
    other_labels = [l for l in settings.emotion_labels if l != primary]
    for label in other_labels:
        probs[label] = remaining / len(other_labels)

    return EmotionResult(
        primary_emotion=primary,
        confidence=confidence,
        probabilities=probs,
        all_emotions=[primary],
        raw_text="test input",
        inference_ms=10.0,
        model_type="distilbert",
    )


# ── Tests ─────────────────────────────────────────────────────────────────────
class TestContextSmootherInitialization:

    def test_default_initialization(self):
        smoother = ContextSmoother()
        assert smoother.window_size == settings.emotion_smoothing_window
        assert smoother.current_window_size == 0
        assert smoother.turn_number == 0
        assert not smoother.is_warmed_up

    def test_custom_window_size(self):
        smoother = ContextSmoother(window_size=3)
        assert smoother.window_size == 3
        assert len(smoother.weights) == 3
        assert abs(sum(smoother.weights) - 1.0) < 1e-6

    def test_custom_weights_valid(self):
        smoother = ContextSmoother(window_size=4, weights=[0.1, 0.2, 0.3, 0.4])
        assert smoother.weights == [0.1, 0.2, 0.3, 0.4]

    def test_custom_weights_wrong_length_raises(self):
        with pytest.raises(ValueError, match="weights length"):
            ContextSmoother(window_size=4, weights=[0.5, 0.5])

    def test_custom_weights_wrong_sum_raises(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            ContextSmoother(window_size=4, weights=[0.1, 0.2, 0.3, 0.5])

    def test_negative_weights_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            ContextSmoother(window_size=3, weights=[-0.1, 0.5, 0.6])


class TestSingleTurnBehavior:

    def test_first_turn_returns_raw_emotion(self):
        """With only one turn, smoothed output should match raw input."""
        smoother = ContextSmoother()
        result = make_emotion_result("joy", 0.9)
        smoothed = smoother.update(result)

        assert smoothed.primary_emotion == "joy"
        assert smoothed.window_size == 1
        assert smoothed.turn_number == 1
        assert not smoothed.was_smoothed

    def test_first_turn_confidence_preserved(self):
        smoother = ContextSmoother()
        result = make_emotion_result("sadness", 0.85)
        smoothed = smoother.update(result)

        # With single turn, smoothed confidence should closely match raw
        assert abs(smoothed.confidence - smoothed.probabilities["sadness"]) < 1e-4


class TestWindowGrowth:

    def test_window_grows_to_max(self):
        smoother = ContextSmoother(window_size=4)
        for i in range(6):
            result = make_emotion_result("joy")
            smoothed = smoother.update(result)

        # Window should never exceed max size
        assert smoother.current_window_size == 4
        assert smoother.turn_number == 6

    def test_warmup_detection(self):
        smoother = ContextSmoother(window_size=4)

        for i in range(3):
            smoother.update(make_emotion_result("joy"))
            assert not smoother.is_warmed_up

        smoother.update(make_emotion_result("joy"))
        assert smoother.is_warmed_up


class TestSmoothing:

    def test_neutral_spike_dampened(self):
        """
        Core test: a single neutral turn surrounded by sadness turns
        should still produce sadness as the smoothed output.
        This directly validates the project's core hypothesis.
        """
        smoother = ContextSmoother(window_size=4, weights=[0.1, 0.2, 0.3, 0.4])

        # Three sadness turns
        smoother.update(make_emotion_result("sadness", 0.90))
        smoother.update(make_emotion_result("sadness", 0.88))
        smoother.update(make_emotion_result("sadness", 0.85))

        # One neutral spike (sarcasm/deflection)
        smoothed = smoother.update(make_emotion_result("neutral", 0.91))

        # Smoothed output should still be sadness
        assert smoothed.primary_emotion == "sadness", (
            f"Expected sadness but got {smoothed.primary_emotion}. "
            f"Smoothing failed to dampen neutral spike."
        )
        assert smoothed.was_smoothed, "Expected was_smoothed=True"

    def test_genuine_emotion_change_detected(self):
        """
        If emotion genuinely changes across multiple turns, the smoother
        should eventually reflect that change.
        """
        smoother = ContextSmoother(window_size=4, weights=[0.1, 0.2, 0.3, 0.4])

        # Fill window with sadness
        for _ in range(4):
            smoother.update(make_emotion_result("sadness", 0.90))

        # Now fill with joy
        for _ in range(4):
            smoothed = smoother.update(make_emotion_result("joy", 0.95))

        # After 4 joy turns, joy should dominate
        assert smoothed.primary_emotion == "joy", (
            f"Expected joy after 4 consistent joy turns, "
            f"got {smoothed.primary_emotion}"
        )

    def test_probabilities_sum_approximately_one(self):
        """
        Smoothed probabilities should sum to approximately 1.0
        since they are averaged sigmoid outputs.
        """
        smoother = ContextSmoother()
        for emotion in ["joy", "sadness", "anger", "fear"]:
            smoother.update(make_emotion_result(emotion, 0.85))

        result = make_emotion_result("neutral", 0.7)
        smoothed = smoother.update(result)

        prob_sum = sum(smoothed.probabilities.values())
        # Sigmoid outputs don't sum to exactly 1, but should be reasonable
        assert 0.5 <= prob_sum <= 2.5, f"Probability sum out of range: {prob_sum}"

    def test_consistent_emotion_not_marked_as_smoothed(self):
        """If emotion stays the same, was_smoothed should be False."""
        smoother = ContextSmoother()
        for _ in range(4):
            smoothed = smoother.update(make_emotion_result("joy", 0.9))

        assert not smoothed.was_smoothed


class TestReset:

    def test_reset_clears_window(self):
        smoother = ContextSmoother()
        for _ in range(4):
            smoother.update(make_emotion_result("joy"))

        assert smoother.current_window_size == 4

        smoother.reset()

        assert smoother.current_window_size == 0
        assert smoother.turn_number == 0
        assert not smoother.is_warmed_up

    def test_after_reset_behaves_like_new(self):
        smoother = ContextSmoother()

        # Session 1: sadness
        for _ in range(4):
            smoother.update(make_emotion_result("sadness", 0.9))

        # Reset (new session)
        smoother.reset()

        # Session 2: joy — should not be influenced by session 1
        smoothed = smoother.update(make_emotion_result("joy", 0.9))
        assert smoothed.primary_emotion == "joy"
        assert smoothed.turn_number == 1


class TestWindowSnapshot:

    def test_snapshot_structure(self):
        smoother = ContextSmoother(window_size=4)
        smoother.update(make_emotion_result("joy", 0.9))
        smoother.update(make_emotion_result("sadness", 0.85))

        snapshot = smoother.get_window_snapshot()

        assert len(snapshot) == 2
        assert snapshot[0]["position"] == 1
        assert snapshot[0]["primary_emotion"] == "joy"
        assert snapshot[1]["primary_emotion"] == "sadness"
        assert "weight" in snapshot[0]
        assert "turns_ago" in snapshot[0]


class TestTrendDescription:

    def test_consistent_trend(self):
        smoother = ContextSmoother()
        for _ in range(4):
            smoother.update(make_emotion_result("sadness"))
        trend = smoother.get_dominant_emotion_trend()
        assert "sadness" in trend.lower()
        assert "consistent" in trend.lower()

    def test_shifting_trend(self):
        smoother = ContextSmoother(window_size=4)
        smoother.update(make_emotion_result("anger"))
        smoother.update(make_emotion_result("anger"))
        smoother.update(make_emotion_result("neutral"))
        smoother.update(make_emotion_result("neutral"))
        trend = smoother.get_dominant_emotion_trend()
        assert "anger" in trend.lower() or "neutral" in trend.lower()

    def test_empty_trend(self):
        smoother = ContextSmoother()
        trend = smoother.get_dominant_emotion_trend()
        assert "no data" in trend.lower()
