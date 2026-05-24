"""
src/llm/ollama_client.py
=========================
HTTP client wrapper for the Ollama local LLM runtime.

Ollama exposes a REST API at http://localhost:11434 with two
endpoints we care about:

  POST /api/generate  — single-turn completion
  POST /api/chat      — multi-turn chat with message history
  GET  /api/tags      — list available models
  GET  /api/version   — Ollama version info

We use /api/chat because it natively handles the system prompt +
conversation history structure we need. Each request includes
the full conversation history, which Ollama passes to the model
as properly formatted context.

Why not use the official ollama-python library?
  The official library is a thin wrapper around these same endpoints.
  Building our own gives us:
    - Explicit control over timeout and retry behavior
    - Streaming support with clean token-by-token callback
    - Structured error types for graceful degradation
    - Zero extra dependencies beyond `requests`

Cross-platform note:
  Ollama's API is identical on Mac, Windows, and Linux.
  Only the base URL might differ if using a remote Ollama instance,
  which is handled via settings.ollama_base_url.
"""

import sys
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Iterator, Callable

import requests

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings

logger = logging.getLogger(__name__)


# ── Response Dataclass ────────────────────────────────────────────────────────
@dataclass
class LLMResponse:
    """
    Structured response from the local LLM.

    Fields:
        text:           The generated response text (cleaned)
        model:          Which Ollama model was used
        total_ms:       Total time from request to complete response
        prompt_tokens:  Approximate token count of the input prompt
        response_tokens: Approximate token count of the generated response
        success:        Whether generation completed without error
        error:          Error message if success=False
    """

    text: str
    model: str
    total_ms: float
    prompt_tokens: int = 0
    response_tokens: int = 0
    success: bool = True
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "model": self.model,
            "total_ms": round(self.total_ms, 2),
            "prompt_tokens": self.prompt_tokens,
            "response_tokens": self.response_tokens,
            "success": self.success,
        }

    @classmethod
    def error_response(cls, error: str, model: str = "") -> "LLMResponse":
        """Create a failed response for error handling."""
        return cls(
            text="I'm having trouble responding right now. Please try again.",
            model=model,
            total_ms=0.0,
            success=False,
            error=error,
        )


# ── Custom Exceptions ─────────────────────────────────────────────────────────
class OllamaNotRunningError(Exception):
    """Raised when Ollama server is not reachable."""

    pass


class OllamaModelNotFoundError(Exception):
    """Raised when the requested model is not pulled."""

    pass


class OllamaTimeoutError(Exception):
    """Raised when generation exceeds timeout threshold."""

    pass


# ── Client ────────────────────────────────────────────────────────────────────
class OllamaClient:
    """
    Clean interface to the Ollama local LLM runtime.

    Usage:
        client = OllamaClient()

        # Simple generation
        response = client.generate("Tell me something kind")

        # With system prompt
        response = client.generate(
            prompt="I feel really sad today",
            system_prompt="You are a compassionate counselor. Be empathetic."
        )

        # Streaming (token by token)
        for token in client.generate_stream("Tell me a story"):
            print(token, end="", flush=True)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        self.model = model or settings.llm_model_name
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.timeout = timeout or settings.ollama_timeout

        # Reuse HTTP session for connection pooling
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

        logger.info(
            f"OllamaClient initialized | "
            f"model={self.model} | "
            f"base_url={self.base_url}"
        )

    # ── Health and Model Checks ───────────────────────────────────────────────
    def is_running(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            resp = self._session.get(
                f"{self.base_url}/api/tags",
                timeout=5,
            )
            return resp.status_code == 200
        except requests.exceptions.ConnectionError:
            return False
        except Exception:
            return False

    def is_model_available(self, model: Optional[str] = None) -> bool:
        """Check if the specified model is pulled and available."""
        target_model = model or self.model
        try:
            resp = self._session.get(
                f"{self.base_url}/api/tags",
                timeout=5,
            )
            if resp.status_code != 200:
                return False
            models = resp.json().get("models", [])
            available_names = [m["name"] for m in models]
            # Check exact match or base name match
            target_base = target_model.split(":")[0]
            return any(
                target_model == name or target_base == name.split(":")[0]
                for name in available_names
            )
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """Return list of all available model names."""
        try:
            resp = self._session.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            pass
        return []

    def check_ready(self):
        """
        Verify Ollama is running and model is available.
        Raises descriptive exceptions if not ready.
        Call this at application startup.
        """
        if not self.is_running():
            raise OllamaNotRunningError(
                f"Ollama is not running at {self.base_url}\n"
                f"Start it with: ollama serve"
            )
        if not self.is_model_available():
            available = self.list_models()
            raise OllamaModelNotFoundError(
                f"Model '{self.model}' not found in Ollama.\n"
                f"Pull it with: ollama pull {self.model}\n"
                f"Available models: {available}"
            )
        logger.info(f"Ollama ready: model={self.model}")

    # ── Core Generation ───────────────────────────────────────────────────────
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[list[dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Uses /api/chat endpoint with full message history for proper
        multi-turn context management.

        Args:
            prompt:               The user's current message / final prompt
            system_prompt:        Instructions that define model behavior
            conversation_history: Previous turns as list of
                                  {"role": "user"/"assistant", "content": "..."}
            temperature:          Sampling temperature (0=deterministic, 1=creative)
            max_tokens:           Maximum response length in tokens

        Returns:
            LLMResponse with generated text and metadata
        """
        if not prompt or not prompt.strip():
            return LLMResponse.error_response("Empty prompt provided")

        # Build message list for /api/chat
        messages = self._build_messages(prompt, system_prompt, conversation_history)

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature or settings.llm_temperature,
                "num_predict": max_tokens or settings.llm_max_tokens,
                # Disable mirostat for more consistent empathetic tone
                "mirostat": 0,
                # Top-p sampling for quality/diversity balance
                "top_p": 0.9,
                # Repeat penalty prevents repetitive consolation phrases
                "repeat_penalty": 1.1,
            },
        }

        start_time = time.time()

        try:
            response = self._session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            total_ms = (time.time() - start_time) * 1000

            data = response.json()

            # Extract response text
            raw_text = data.get("message", {}).get("content", "")
            clean_text = self._clean_response(raw_text)

            # Extract token counts if available
            prompt_tokens = data.get("prompt_eval_count", 0)
            response_tokens = data.get("eval_count", 0)

            logger.debug(
                f"Generated response | "
                f"model={self.model} | "
                f"tokens={response_tokens} | "
                f"latency={total_ms:.0f}ms"
            )

            return LLMResponse(
                text=clean_text,
                model=self.model,
                total_ms=total_ms,
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
                success=True,
            )

        except requests.exceptions.Timeout:
            elapsed = (time.time() - start_time) * 1000
            logger.error(f"Ollama timeout after {elapsed:.0f}ms")
            return LLMResponse.error_response(
                f"Response timed out after {self.timeout}s",
                model=self.model,
            )

        except requests.exceptions.ConnectionError:
            logger.error("Ollama connection lost during generation")
            return LLMResponse.error_response(
                "Lost connection to Ollama — is it still running?",
                model=self.model,
            )

        except requests.exceptions.HTTPError as e:
            logger.error(f"Ollama HTTP error: {e}")
            return LLMResponse.error_response(str(e), model=self.model)

        except Exception as e:
            logger.error(f"Unexpected Ollama error: {e}")
            return LLMResponse.error_response(str(e), model=self.model)

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[list[dict]] = None,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Iterator[str]:
        """
        Generate a streaming response, yielding one token at a time.

        This is used by the Streamlit UI to show responses as they
        are generated rather than waiting for the complete response.
        The visual effect is similar to how ChatGPT shows responses.

        Args:
            prompt:               User message
            system_prompt:        System instructions
            conversation_history: Previous conversation turns
            on_token:             Optional callback called for each token

        Yields:
            Individual tokens (strings) as they are generated
        """
        messages = self._build_messages(prompt, system_prompt, conversation_history)

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": settings.llm_temperature,
                "num_predict": settings.llm_max_tokens,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
            },
        }

        try:
            with self._session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=self.timeout,
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line.decode("utf-8"))
                        token = data.get("message", {}).get("content", "")
                        if token:
                            if on_token:
                                on_token(token)
                            yield token
                        # Stop if done flag received
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

        except requests.exceptions.ConnectionError:
            yield "\n[Connection to Ollama lost]"
        except requests.exceptions.Timeout:
            yield "\n[Response timed out]"
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"\n[Error: {e}]"

    # ── Private Helpers ───────────────────────────────────────────────────────
    def _build_messages(
        self,
        prompt: str,
        system_prompt: Optional[str],
        conversation_history: Optional[list[dict]],
    ) -> list[dict]:
        """
        Build the messages array for /api/chat.

        Ollama's chat endpoint expects:
        [
            {"role": "system",    "content": "..."},  # optional
            {"role": "user",      "content": "..."},  # turn 1
            {"role": "assistant", "content": "..."},  # response 1
            {"role": "user",      "content": "..."},  # turn 2
            ...
            {"role": "user",      "content": "..."},  # current turn
        ]
        """
        messages = []

        # System prompt first (defines model persona and behavior)
        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        # Previous conversation turns (context for multi-turn coherence)
        if conversation_history:
            # Limit history to last 6 turns to avoid context overflow
            # 6 turns = 3 user + 3 assistant = reasonable context window
            recent_history = conversation_history[-6:]
            messages.extend(recent_history)

        # Current user message
        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        return messages

    def _clean_response(self, text: str) -> str:
        """
        Clean LLM output for display.

        Removes common artifacts:
        - Leading/trailing whitespace
        - Repeated newlines (models sometimes add excessive spacing)
        - Occasional prefix artifacts like "Assistant:" that some
          models add even when instructed not to
        """
        if not text:
            return ""

        # Strip whitespace
        text = text.strip()

        # Remove common model self-identification prefixes
        prefixes_to_remove = [
            "Assistant:",
            "assistant:",
            "AI:",
            "Counselor:",
            "Response:",
            "Reply:",
        ]
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()

        # Collapse excessive newlines (more than 2 consecutive)
        import re

        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def get_model_info(self) -> dict:
        """Get metadata about the current model from Ollama."""
        try:
            resp = self._session.post(
                f"{self.base_url}/api/show",
                json={"name": self.model},
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return {}
