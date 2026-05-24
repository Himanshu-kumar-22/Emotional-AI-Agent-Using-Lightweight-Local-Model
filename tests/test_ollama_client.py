"""
tests/test_ollama_client.py
============================
Tests for OllamaClient and PromptBuilder.

Run with:
    pytest tests/test_ollama_client.py -v

Note: Tests that actually call Ollama are marked with @pytest.mark.integration
and require Ollama to be running. Pure unit tests run without Ollama.
"""

import sys
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.ollama_client import OllamaClient, LLMResponse
from src.llm.prompt_builder import PromptBuilder, BuiltPrompt
from src.smoothing.context_smoother import SmoothedEmotion
from config.settings import settings


# ── LLMResponse Tests ─────────────────────────────────────────────────────────
class TestLLMResponse:

    def test_error_response_structure(self):
        resp = LLMResponse.error_response("test error", "phi3:mini")
        assert resp.success is False
        assert resp.error == "test error"
        assert resp.model == "phi3:mini"
        assert len(resp.text) > 0  # Should have fallback message

    def test_to_dict_keys(self):
        resp = LLMResponse(text="Hello", model="phi3:mini", total_ms=500, success=True)
        d = resp.to_dict()
        assert "text" in d
        assert "model" in d
        assert "total_ms" in d
        assert "success" in d


# ── OllamaClient Unit Tests ───────────────────────────────────────────────────
class TestOllamaClientUnit:
    """Tests that do not require Ollama to be running."""

    def test_initialization_defaults(self):
        client = OllamaClient()
        assert client.model == settings.llm_model_name
        assert client.base_url == settings.ollama_base_url

    def test_custom_model(self):
        client = OllamaClient(model="phi3:mini")
        assert client.model == "phi3:mini"

    def test_clean_response_strips_whitespace(self):
        client = OllamaClient()
        assert client._clean_response("  hello  ") == "hello"

    def test_clean_response_removes_prefix(self):
        client = OllamaClient()
        result = client._clean_response("Assistant: I understand you")
        assert not result.startswith("Assistant:")
        assert "I understand you" in result

    def test_clean_response_empty_string(self):
        client = OllamaClient()
        assert client._clean_response("") == ""

    def test_build_messages_with_system(self):
        client = OllamaClient()
        msgs = client._build_messages(
            prompt="Hello",
            system_prompt="You are helpful",
            conversation_history=None,
        )
        assert msgs[0]["role"] == "system"
        assert msgs[-1]["role"] == "user"
        assert msgs[-1]["content"] == "Hello"

    def test_build_messages_history_limited_to_6(self):
        client = OllamaClient()
        long_history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"message {i}"}
            for i in range(20)
        ]
        msgs = client._build_messages(
            prompt="Current message",
            system_prompt=None,
            conversation_history=long_history,
        )
        # Should have at most 6 history + 1 current = 7
        non_system = [m for m in msgs if m["role"] != "system"]
        assert len(non_system) <= 7

    def test_empty_prompt_returns_error(self):
        client = OllamaClient()
        resp = client.generate("")
        assert resp.success is False


# ── PromptBuilder Tests ───────────────────────────────────────────────────────
class TestPromptBuilder:

    def _make_smoothed(self, emotion: str, confidence: float = 0.85) -> SmoothedEmotion:
        return SmoothedEmotion(
            primary_emotion=emotion,
            confidence=confidence,
            probabilities={
                e: (confidence if e == emotion else 0.02)
                for e in settings.emotion_labels
            },
            raw_emotion=emotion,
            raw_confidence=confidence,
            was_smoothed=False,
            window_size=2,
            turn_number=2,
        )

    def test_build_returns_built_prompt(self):
        builder = PromptBuilder()
        smoothed = self._make_smoothed("sadness")
        result = builder.build("I feel sad", smoothed)
        assert isinstance(result, BuiltPrompt)

    def test_system_prompt_contains_emotion(self):
        builder = PromptBuilder()
        for emotion in settings.emotion_labels:
            smoothed = self._make_smoothed(emotion)
            prompt = builder.build("test message", smoothed)
            assert (
                emotion in prompt.system_prompt.lower()
                or emotion.title() in prompt.system_prompt
            )

    def test_high_distress_adds_extra_guidance(self):
        builder = PromptBuilder()
        # High confidence sadness should add extra care instructions
        smoothed = self._make_smoothed("sadness", confidence=0.95)
        prompt = builder.build("I feel terrible", smoothed)
        assert (
            "distress" in prompt.system_prompt.lower()
            or "acknowledgment" in prompt.system_prompt.lower()
        )

    def test_low_distress_no_extra_guidance(self):
        builder = PromptBuilder()
        smoothed = self._make_smoothed("joy", confidence=0.9)
        prompt = builder.build("I feel great!", smoothed)
        assert "distress" not in prompt.system_prompt.lower()

    def test_smoothed_emotion_noted_in_user_prompt(self):
        builder = PromptBuilder()
        smoothed = SmoothedEmotion(
            primary_emotion="sadness",
            confidence=0.85,
            probabilities={e: 0.02 for e in settings.emotion_labels},
            raw_emotion="neutral",
            raw_confidence=0.7,
            was_smoothed=True,  # <-- smoothing changed the label
            window_size=4,
            turn_number=4,
        )
        prompt = builder.build("lol whatever", smoothed)
        # User prompt should note the smoothing correction
        assert "sadness" in prompt.user_prompt.lower()

    def test_conversation_history_summarized(self):
        builder = PromptBuilder()
        smoothed = self._make_smoothed("neutral")
        history = [
            {"role": "user", "content": "I have been feeling down"},
            {"role": "assistant", "content": "I hear you"},
            {"role": "user", "content": "Work has been hard"},
            {"role": "assistant", "content": "That sounds tough"},
        ]
        prompt = builder.build("Yes exactly", smoothed, history)
        # History summary should appear in user prompt
        assert len(prompt.user_prompt) > len("Yes exactly")

    def test_build_simple_works(self):
        builder = PromptBuilder()
        prompt = builder.build_simple("Hello there", emotion="joy")
        assert isinstance(prompt, BuiltPrompt)
        assert prompt.emotion_label == "joy"

    def test_to_messages_returns_tuple(self):
        builder = PromptBuilder()
        smoothed = self._make_smoothed("neutral")
        prompt = builder.build("hello", smoothed)
        system, user = prompt.to_messages()
        assert isinstance(system, str)
        assert isinstance(user, str)
        assert "hello" in user


# ── Integration Tests (require Ollama running) ────────────────────────────────
@pytest.mark.integration
class TestOllamaIntegration:
    """
    These tests actually call Ollama and require it to be running.

    Run only these with:
        pytest tests/test_ollama_client.py -v -m integration
    """

    @pytest.fixture(scope="class")
    def client(self):
        c = OllamaClient()
        if not c.is_running():
            pytest.skip("Ollama not running — skipping integration tests")
        return c

    def test_is_running(self, client):
        assert client.is_running()

    def test_model_available(self, client):
        assert client.is_model_available()

    def test_basic_generation(self, client):
        response = client.generate(
            prompt="Say exactly the word: HELLO",
            system_prompt="You are a test assistant. Follow instructions exactly.",
        )
        assert response.success
        assert len(response.text) > 0
        assert response.total_ms > 0

    def test_empathetic_response_for_sadness(self, client):
        builder = PromptBuilder()
        prompt = builder.build_simple(
            "I feel really sad and alone today", emotion="sadness"
        )
        response = client.generate(
            prompt=prompt.user_prompt,
            system_prompt=prompt.system_prompt,
        )
        assert response.success
        assert len(response.text) > 20
        print(f"\nSadness response ({response.total_ms:.0f}ms):")
        print(f"  {response.text}")

    def test_response_does_not_start_with_assistant(self, client):
        response = client.generate("Hello, how are you?")
        assert not response.text.startswith("Assistant:")
        assert not response.text.startswith("AI:")

    def test_streaming_yields_tokens(self, client):
        tokens = list(
            client.generate_stream(
                "Count to 3: one, two, three",
                system_prompt="You are helpful.",
            )
        )
        assert len(tokens) > 0
        full_text = "".join(tokens)
        assert len(full_text) > 0
