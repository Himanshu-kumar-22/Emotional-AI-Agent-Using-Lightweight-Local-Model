"""
src/pipeline/agent.py
======================
EmotionalAgent: the central orchestrator of the entire pipeline.

This class is the single entry point for the Streamlit UI.
It owns and manages all pipeline components:
  - EmotionDetector
  - ContextSmoother
  - PromptBuilder
  - OllamaClient
  - SessionManager

Design pattern: Facade
  Hides the complexity of four separate subsystems behind
  one clean chat() method. The UI never imports any other
  pipeline component directly.

Lifecycle:
  1. EmotionalAgent() — creates instance, no heavy operations
  2. agent.initialize() — loads model, connects to Ollama, sets up DB
  3. agent.start_session() — begins a new conversation session
  4. agent.chat(text) — processes one message, returns AgentResponse
  5. agent.end_session() — closes session gracefully
  6. agent.shutdown() — cleans up all resources
"""

import sys
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Iterator

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from src.emotion.detector import EmotionDetector, EmotionResult
from src.smoothing.context_smoother import ContextSmoother, SmoothedEmotion
from src.llm.ollama_client import OllamaClient, LLMResponse
from src.llm.prompt_builder import PromptBuilder
from src.storage.session_manager import SessionManager

logger = logging.getLogger(__name__)


# ── Response Dataclass ────────────────────────────────────────────────────────
@dataclass
class AgentResponse:
    """
    Complete response from one pipeline cycle.

    Contains everything the UI needs to render:
    - The generated text
    - Emotion information for the badge
    - Timing data for the sidebar
    - Session metadata
    """

    text: str
    primary_emotion: str
    emotion_confidence: float
    emotion_emoji: str
    emotion_color: str
    raw_emotion: str
    was_smoothed: bool
    emotion_trend: str
    total_ms: float
    emotion_ms: float
    llm_ms: float
    session_id: str
    turn_number: int
    success: bool = True
    error: str = ""

    def get_emotion_display(self) -> dict:
        return settings.emotion_display.get(
            self.primary_emotion,
            {"emoji": "😐", "color": "#708090", "label": "Neutral"},
        )

    @classmethod
    def error_response(cls, error: str, session_id: str = "") -> "AgentResponse":
        return cls(
            text="I'm having trouble right now. Please try again.",
            primary_emotion="neutral",
            emotion_confidence=0.0,
            emotion_emoji="😐",
            emotion_color="#708090",
            raw_emotion="neutral",
            was_smoothed=False,
            emotion_trend="",
            total_ms=0.0,
            emotion_ms=0.0,
            llm_ms=0.0,
            session_id=session_id,
            turn_number=0,
            success=False,
            error=error,
        )


# ── Agent ─────────────────────────────────────────────────────────────────────
class EmotionalAgent:
    """
    Orchestrates the complete emotional AI pipeline.

    One instance per Streamlit session (stored in st.session_state).
    """

    def __init__(
        self,
        privacy_mode: Optional[bool] = None,
        llm_model: Optional[str] = None,
        emotion_model: Optional[str] = None,
    ):
        self.privacy_mode = (
            privacy_mode if privacy_mode is not None else settings.privacy_mode_default
        )
        self.llm_model = llm_model or settings.llm_model_name
        self.emotion_model = emotion_model or settings.emotion_model_type

        # Pipeline components — initialized lazily
        self._detector: Optional[EmotionDetector] = None
        self._smoother: Optional[ContextSmoother] = None
        self._prompt_builder: Optional[PromptBuilder] = None
        self._llm_client: Optional[OllamaClient] = None
        self._storage: Optional[SessionManager] = None

        # Session state
        self._current_session_id: Optional[str] = None
        self._turn_number: int = 0
        self._is_initialized: bool = False
        self._conversation_history: list[dict] = []
        self._user_name: Optional[str] = None

        logger.info(
            f"EmotionalAgent created | "
            f"privacy_mode={self.privacy_mode} | "
            f"llm={self.llm_model}"
        )

    def initialize(self, password: Optional[str] = None) -> bool:
        """
        Initialize all pipeline components.

        Args:
            password: User password for storage encryption.
                      Pass None for Privacy Mode (no encryption needed).

        Returns:
            True if initialization succeeded.
        """
        try:
            logger.info("Initializing EmotionalAgent...")

            # 1. Emotion detector
            logger.info("Loading emotion model...")
            self._detector = EmotionDetector(model_type=self.emotion_model)
            self._detector.load_model()
            if not self._detector.is_ready:
                raise RuntimeError(
                    "Emotion model failed to load. "
                    "Run: python3 scripts/train_emotion_model.py"
                )

            # 2. Context smoother (stateless, always ready)
            self._smoother = ContextSmoother()

            # 3. Prompt builder (stateless, always ready)
            self._prompt_builder = PromptBuilder()

            # 4. LLM client — verify Ollama is running
            logger.info("Connecting to Ollama...")
            self._llm_client = OllamaClient(model=self.llm_model)
            self._llm_client.check_ready()

            # 5. Storage
            logger.info("Initializing storage...")
            self._storage = SessionManager(privacy_mode=self.privacy_mode)
            if self.privacy_mode or not password:
                self._storage.initialize_no_password()
            else:
                self._storage.initialize(password)

            self._is_initialized = True
            logger.info("EmotionalAgent initialized successfully")
            return True

        except Exception as e:
            logger.error(f"EmotionalAgent initialization failed: {e}")
            self._is_initialized = False
            raise

    def start_session(self) -> str:
        """
        Begin a new conversation session.

        Returns:
            session_id for the new session.
        """
        self._check_initialized()

        # Reset per-session state
        self._smoother.reset()
        self._turn_number = 0
        self._conversation_history = []

        # Create database session
        self._current_session_id = self._storage.create_session(
            {
                "llm_model": self.llm_model,
                "emotion_model": self.emotion_model,
                "privacy_mode": self.privacy_mode,
            }
        )

        logger.info(f"Session started: {self._current_session_id[:8]}...")
        return self._current_session_id

    def chat(self, user_message: str) -> AgentResponse:
        """
        Process one user message through the complete pipeline.

        Pipeline steps:
          1. Detect emotion from user message
          2. Smooth emotion across conversation window
          3. Build emotion-conditioned prompt
          4. Generate response via Ollama
          5. Save both messages to encrypted storage
          6. Update conversation history

        Args:
            user_message: Raw text from the user

        Returns:
            AgentResponse with text, emotion data, and timing
        """
        self._check_initialized()

        if not self._current_session_id:
            self.start_session()

        if not user_message or not user_message.strip():
            return AgentResponse.error_response(
                "Empty message", self._current_session_id
            )

        self._turn_number += 1
        pipeline_start = time.time()

        try:
            # ── Step 1: Emotion Detection ──────────────────────────────────
            emotion_start = time.time()
            raw_emotion: EmotionResult = self._detector.detect(user_message)
            emotion_ms = (time.time() - emotion_start) * 1000

            # ── Step 2: Context Smoothing ──────────────────────────────────
            smoothed: SmoothedEmotion = self._smoother.update(raw_emotion)

            # ── Step 3: Prompt Building ────────────────────────────────────
            built_prompt = self._prompt_builder.build(
                user_message=user_message,
                smoothed_emotion=smoothed,
                conversation_history=self._conversation_history,
                user_name=self._user_name,
            )

            # ── Step 4: LLM Generation ─────────────────────────────────────
            llm_start = time.time()
            llm_response: LLMResponse = self._llm_client.generate(
                prompt=built_prompt.user_prompt,
                system_prompt=built_prompt.system_prompt,
                conversation_history=self._conversation_history,
            )
            llm_ms = (time.time() - llm_start) * 1000

            if not llm_response.success:
                logger.warning(f"LLM generation failed: {llm_response.error}")

            # ── Step 5: Save to Storage ────────────────────────────────────
            self._storage.save_message(
                session_id=self._current_session_id,
                role="user",
                content=user_message,
                turn_number=self._turn_number,
                emotion_data=smoothed.to_dict(),
            )
            self._storage.save_message(
                session_id=self._current_session_id,
                role="assistant",
                content=llm_response.text,
                turn_number=self._turn_number,
            )

            # Save detailed emotion log for evaluation
            self._storage.save_emotion_log(
                session_id=self._current_session_id,
                turn_number=self._turn_number,
                raw_vector=raw_emotion.probabilities,
                smoothed_vector=smoothed.probabilities,
            )

            # ── Step 6: Update History ─────────────────────────────────────
            self._conversation_history.append(
                {
                    "role": "user",
                    "content": user_message,
                }
            )
            self._conversation_history.append(
                {
                    "role": "assistant",
                    "content": llm_response.text,
                }
            )
            # Keep history bounded to last 10 turns
            if len(self._conversation_history) > 20:
                self._conversation_history = self._conversation_history[-20:]

            total_ms = (time.time() - pipeline_start) * 1000

            # ── Build Response ─────────────────────────────────────────────
            display = smoothed.get_display_info()

            response = AgentResponse(
                text=llm_response.text,
                primary_emotion=smoothed.primary_emotion,
                emotion_confidence=smoothed.confidence,
                emotion_emoji=display["emoji"],
                emotion_color=display["color"],
                raw_emotion=smoothed.raw_emotion,
                was_smoothed=smoothed.was_smoothed,
                emotion_trend=self._smoother.get_dominant_emotion_trend(),
                total_ms=total_ms,
                emotion_ms=emotion_ms,
                llm_ms=llm_ms,
                session_id=self._current_session_id,
                turn_number=self._turn_number,
                success=llm_response.success,
            )

            logger.info(
                f"Turn {self._turn_number} complete | "
                f"emotion={smoothed.primary_emotion} ({smoothed.confidence:.0%}) | "
                f"total={total_ms:.0f}ms "
                f"(detect={emotion_ms:.0f}ms, llm={llm_ms:.0f}ms)"
            )

            return response

        except Exception as e:
            logger.error(f"Pipeline error on turn {self._turn_number}: {e}")
            return AgentResponse.error_response(str(e), self._current_session_id)

    def chat_stream(
        self,
        user_message: str,
    ) -> tuple[Iterator[str], SmoothedEmotion]:
        """
        Streaming version of chat() — yields tokens as they are generated.

        Returns a tuple of:
          - Iterator[str]: token stream from Ollama
          - SmoothedEmotion: emotion result (available immediately,
            before streaming completes)

        Usage in Streamlit:
            token_stream, emotion = agent.chat_stream(user_input)
            for token in token_stream:
                response_placeholder.write(collected_text + token)
        """
        self._check_initialized()

        if not self._current_session_id:
            self.start_session()

        self._turn_number += 1

        # Run emotion detection synchronously (fast, ~12ms on your M3)
        raw_emotion = self._detector.detect(user_message)
        smoothed = self._smoother.update(raw_emotion)

        built_prompt = self._prompt_builder.build(
            user_message=user_message,
            smoothed_emotion=smoothed,
            conversation_history=self._conversation_history,
            user_name=self._user_name,
        )

        # Return stream iterator and emotion immediately
        token_stream = self._llm_client.generate_stream(
            prompt=built_prompt.user_prompt,
            system_prompt=built_prompt.system_prompt,
            conversation_history=self._conversation_history,
        )

        return token_stream, smoothed

    def save_streamed_response(
        self,
        user_message: str,
        assistant_response: str,
        smoothed: SmoothedEmotion,
        raw_emotion: Optional[EmotionResult] = None,
    ):
        """
        Save a completed streamed response to storage.
        Called after streaming finishes and full text is collected.
        """
        self._storage.save_message(
            session_id=self._current_session_id,
            role="user",
            content=user_message,
            turn_number=self._turn_number,
            emotion_data=smoothed.to_dict(),
        )
        self._storage.save_message(
            session_id=self._current_session_id,
            role="assistant",
            content=assistant_response,
            turn_number=self._turn_number,
        )
        self._conversation_history.extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_response},
            ]
        )
        if len(self._conversation_history) > 20:
            self._conversation_history = self._conversation_history[-20:]

    def get_session_history(self) -> list[dict]:
        """Get full message history for current session."""
        if not self._current_session_id:
            return []
        return self._storage.get_messages(self._current_session_id)

    def get_session_stats(self) -> dict:
        """Get statistics for the current session."""
        if not self._current_session_id:
            return {"total_messages": 0, "turn_count": 0}
        return self._storage.get_session_stats(self._current_session_id)

    def new_session(self) -> str:
        """Start a fresh conversation (new session)."""
        return self.start_session()

    def list_sessions(self, limit: int = 30) -> list[dict]:
        """Return recent sessions for the chat history sidebar."""
        if not self._is_initialized:
            return []
        try:
            return self._storage.list_sessions(limit=limit)
        except Exception:
            return []

    def load_session(self, session_id: str) -> list[dict]:
        """Load an existing session and restore its conversation context."""
        self._check_initialized()
        messages = self._storage.get_messages(session_id)
        self._conversation_history = [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]
        if len(self._conversation_history) > 20:
            self._conversation_history = self._conversation_history[-20:]
        self._current_session_id = session_id
        self._turn_number = sum(1 for m in messages if m["role"] == "user")
        self._smoother.reset()
        return messages

    def list_sessions_with_titles(self, limit: int = 30) -> list[dict]:
        """Return recent sessions with their first user message as the display title."""
        if not self._is_initialized:
            return []
        try:
            sessions = self._storage.list_sessions(limit=limit)
            result = []
            for session in sessions:
                first_msg = self._storage.get_first_user_message(session["id"])
                if first_msg:
                    text = first_msg.strip()
                    title = text[:42] + "…" if len(text) > 42 else text
                else:
                    title = "New conversation"
                result.append({**session, "title": title})
            return result
        except Exception:
            return []

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages."""
        if not self._is_initialized:
            return False
        try:
            return self._storage.delete_session(session_id)
        except Exception:
            return False

    def set_user_name(self, name: str):
        """Set the user's display name for personalised LLM responses."""
        self._user_name = name

    def save_user_profile(self, name: str, ram_gb: int):
        """Persist user profile (name + RAM) to the database."""
        self._check_initialized()
        self._storage.save_user_profile(name, ram_gb)
        self._user_name = name

    def get_user_profile(self) -> Optional[dict]:
        """Return stored user profile or None if first run."""
        if not self._is_initialized:
            return None
        try:
            return self._storage.get_user_profile()
        except Exception:
            return None

    def shutdown(self):
        """Clean shutdown of all components."""
        if self._storage:
            self._storage.close()
        logger.info("EmotionalAgent shut down")

    def _check_initialized(self):
        if not self._is_initialized:
            raise RuntimeError(
                "EmotionalAgent not initialized. Call initialize() first."
            )

    @property
    def is_ready(self) -> bool:
        return self._is_initialized

    @property
    def current_session_id(self) -> Optional[str]:
        return self._current_session_id

    @property
    def turn_number(self) -> int:
        return self._turn_number
