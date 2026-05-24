"""
src/llm/prompt_builder.py
==========================
Constructs emotion-conditioned prompts for empathetic response generation.

This module bridges the gap between the emotion detection pipeline
and the LLM generation pipeline. It is the core of what makes
this system emotionally intelligent rather than just a chatbot.

Prompt Architecture:
    Every prompt has three components:

    1. SYSTEM PROMPT — defines the agent's persona, emotional awareness,
       and behavioral guidelines. Stays constant within a session.

    2. EMOTIONAL CONTEXT BLOCK — injected dynamically based on the
       smoothed emotion result. Tells the LLM what emotional state
       the user is in and how confident we are.

    3. CONVERSATION SUMMARY — a brief summary of recent messages
       that provides narrative context without consuming too many tokens.

Design principle — Specificity over vagueness:
    "The user seems sad" → generic sympathetic response
    "The user is expressing sadness (94% confidence) consistently
     across 3 turns. They mentioned feeling overwhelmed and isolated." →
     targeted, contextually appropriate response

    The PromptBuilder always aims for the second style.
"""

import sys
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from src.smoothing.context_smoother import SmoothedEmotion

logger = logging.getLogger(__name__)


# ── Emotion-Specific Response Guidelines ─────────────────────────────────────
# Each emotion gets tailored behavioral instructions for the LLM.
# These were iteratively refined through manual testing with all 8 emotions
# as described in your report's Section 4.3.

EMOTION_GUIDELINES = {
    "joy": {
        "tone": "warm, celebratory, and genuinely enthusiastic",
        "approach": "Match the user's positive energy. Celebrate with them. "
        "Ask about what's making them happy to deepen the connection.",
        "avoid": "being flat, dismissive of their happiness, or immediately "
        "pivoting to advice or concerns",
    },
    "sadness": {
        "tone": "gentle, compassionate, and unhurried",
        "approach": "Acknowledge the pain first — before anything else. "
        "Validate that their feelings are real and understandable. "
        "Create space for them to share more. Do NOT rush to fix things.",
        "avoid": "toxic positivity ('look on the bright side'), "
        "unsolicited advice, minimizing their experience, "
        "or clinical/detached language",
    },
    "anger": {
        "tone": "calm, steady, and non-confrontational",
        "approach": "Acknowledge the frustration without amplifying it. "
        "Show you understand why they feel this way. "
        "Gently help them feel heard before exploring what happened.",
        "avoid": "dismissing their anger, taking sides, "
        "being preachy about anger management, "
        "or matching their intensity",
    },
    "fear": {
        "tone": "reassuring, grounding, and steady",
        "approach": "Provide a calm, stabilizing presence. "
        "Acknowledge the fear as valid. "
        "Gently orient them to what they can control right now.",
        "avoid": "minimizing the fear, giving false reassurances, "
        "overwhelming them with information or options",
    },
    "surprise": {
        "tone": "curious, engaged, and open",
        "approach": "Match their energy — whether the surprise is positive "
        "or negative. Ask open questions to understand what happened. "
        "Help them process the unexpected.",
        "avoid": "assuming whether the surprise is good or bad, "
        "jumping to conclusions",
    },
    "disgust": {
        "tone": "understanding and non-judgmental",
        "approach": "Validate their reaction without amplifying it. "
        "Show you understand why something feels wrong or off to them. "
        "Help them articulate what specifically bothers them.",
        "avoid": "agreeing with potentially harmful reactions, "
        "dismissing their discomfort",
    },
    "trust": {
        "tone": "warm, reciprocal, and affirming",
        "approach": "Honor the positive connection being expressed. "
        "Affirm their feelings of appreciation or admiration. "
        "Reciprocate the warmth genuinely.",
        "avoid": "being overly formal or reserved when warmth is being offered",
    },
    "neutral": {
        "tone": "friendly, conversational, and helpful",
        "approach": "Respond naturally and helpfully. "
        "Be present and engaged even without strong emotional signals. "
        "Follow the user's lead on the topic.",
        "avoid": "forcing emotional depth when the conversation is casual, "
        "being overly therapeutic when not needed",
    },
}


# ── Prompt Components ─────────────────────────────────────────────────────────
BASE_SYSTEM_PROMPT = """You are a warm, emotionally intelligent companion designed to provide genuine support and meaningful conversation. You operate completely offline — all conversations are private and stay on this device.

Your core principles:
- EMOTIONAL AWARENESS FIRST: Always acknowledge the user's emotional state before responding to content
- GENUINE EMPATHY: Respond as a caring human would, not as a clinical system
- FOLLOW THEIR LEAD: Match the depth and pace the user sets
- CONCISE AND MEANINGFUL: Keep responses focused — 2-4 sentences is usually right
- NO UNSOLICITED ADVICE: Unless the user asks for solutions, focus on being present
- PRIVACY AWARENESS: Never reference storing, recording, or analyzing their data

You are NOT a therapist and should not attempt to diagnose or treat. For serious mental health concerns, gently encourage professional support."""


@dataclass
class BuiltPrompt:
    """
    The complete assembled prompt ready to send to the LLM.

    Separating system_prompt from user_prompt is important because
    the Ollama /api/chat endpoint handles them as separate message roles.
    Mixing them into one string would lose the structural benefit.
    """

    system_prompt: str
    user_prompt: str
    emotion_label: str
    emotion_confidence: float

    def to_messages(self) -> tuple[str, str]:
        """Return (system_prompt, user_prompt) tuple for OllamaClient."""
        return self.system_prompt, self.user_prompt


class PromptBuilder:
    """
    Assembles complete, emotion-conditioned prompts for the LLM.

    Takes:
      - The user's raw message
      - The smoothed emotion result (from ContextSmoother)
      - Recent conversation history

    Produces:
      - A BuiltPrompt with system + user components

    One instance can be reused across many turns — it is stateless.
    All conversation state is passed in as parameters.
    """

    def __init__(self):
        logger.debug("PromptBuilder initialized")

    def build(
        self,
        user_message: str,
        smoothed_emotion: SmoothedEmotion,
        conversation_history: Optional[list[dict]] = None,
    ) -> BuiltPrompt:
        """
        Build a complete emotion-conditioned prompt.

        Args:
            user_message:         The raw user input text
            smoothed_emotion:     Output from ContextSmoother.update()
            conversation_history: List of previous turns as
                                  [{"role": "user"/"assistant",
                                    "content": "..."}]

        Returns:
            BuiltPrompt ready to pass to OllamaClient.generate()
        """
        # Build the system prompt (persona + emotional guidelines)
        system_prompt = self._build_system_prompt(smoothed_emotion)

        # Build the user-facing prompt (emotional context + message)
        user_prompt = self._build_user_prompt(
            user_message,
            smoothed_emotion,
            conversation_history or [],
        )

        result = BuiltPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            emotion_label=smoothed_emotion.primary_emotion,
            emotion_confidence=smoothed_emotion.confidence,
        )

        logger.debug(
            f"Prompt built | "
            f"emotion={smoothed_emotion.primary_emotion} "
            f"({smoothed_emotion.confidence:.0%}) | "
            f"smoothed={smoothed_emotion.was_smoothed} | "
            f"history_turns={len(conversation_history or [])}"
        )

        return result

    def _build_system_prompt(self, smoothed_emotion: SmoothedEmotion) -> str:
        """
        Build the system prompt by combining the base persona with
        emotion-specific behavioral guidelines.

        The system prompt is sent as the "system" role in /api/chat,
        which most models treat as the highest-priority instruction.
        """
        emotion = smoothed_emotion.primary_emotion
        guidelines = EMOTION_GUIDELINES.get(emotion, EMOTION_GUIDELINES["neutral"])

        emotion_display = settings.emotion_display.get(emotion, {})
        emotion_label = emotion_display.get("label", emotion.title())
        emoji = emotion_display.get("emoji", "")

        system_prompt = f"""{BASE_SYSTEM_PROMPT}

CURRENT EMOTIONAL CONTEXT:
The user is currently experiencing: {emoji} {emotion_label}
Confidence level: {smoothed_emotion.confidence:.0%}
Context stability: {smoothed_emotion.get_stability_description()}

YOUR RESPONSE STYLE FOR THIS EMOTION:
- Tone: {guidelines['tone']}
- Approach: {guidelines['approach']}
- Avoid: {guidelines['avoid']}"""

        # Add extra care instruction for high-distress emotions
        if emotion in ("sadness", "fear") and smoothed_emotion.confidence > 0.8:
            system_prompt += """

IMPORTANT: The user appears to be in genuine distress. Take extra care to:
- Lead with acknowledgment, not information
- Use the user's own words back to them where natural
- End with an open, inviting question — not a closed one"""

        return system_prompt

    def _build_user_prompt(
        self,
        user_message: str,
        smoothed_emotion: SmoothedEmotion,
        conversation_history: list[dict],
    ) -> str:
        """
        Build the user-side prompt with emotional framing and
        a brief conversation summary for context.

        We put the emotional context in the USER message (not just
        the system prompt) because it ensures the model sees the
        emotional framing immediately before the actual message,
        right at the point of generation.
        """
        parts = []

        # Add conversation summary if we have history
        if len(conversation_history) >= 2:
            summary = self._summarize_history(conversation_history)
            if summary:
                parts.append(f"[Conversation context: {summary}]")

        # Add smoothing insight if emotion was corrected
        if smoothed_emotion.was_smoothed:
            parts.append(
                f"[Note: The user's underlying emotion appears to be "
                f"{smoothed_emotion.primary_emotion} even though their "
                f"most recent message may seem {smoothed_emotion.raw_emotion}]"
            )

        # The actual user message — always last, most prominent
        parts.append(user_message)

        return "\n\n".join(parts)

    def _summarize_history(self, conversation_history: list[dict]) -> str:
        """
        Create a brief plain-English summary of recent conversation.

        We do not send the full history here (that's handled by
        OllamaClient via the messages array). This summary is a
        compressed contextual note that primes the model for continuity.

        Takes the last 3 user messages and condenses them.
        """
        # Get last 3 user messages only (assistant responses add noise here)
        user_messages = [
            turn["content"]
            for turn in conversation_history
            if turn.get("role") == "user"
        ][-3:]

        if not user_messages:
            return ""

        if len(user_messages) == 1:
            # Truncate long single messages
            msg = user_messages[0]
            return msg[:100] + "..." if len(msg) > 100 else msg

        # For multiple messages, create a brief joined summary
        # Truncate each to 60 chars to keep summary concise
        summaries = []
        for msg in user_messages:
            truncated = msg[:60] + "..." if len(msg) > 60 else msg
            summaries.append(truncated)

        return " → ".join(summaries)

    def build_simple(self, user_message: str, emotion: str = "neutral") -> BuiltPrompt:
        """
        Build a simple prompt without a full SmoothedEmotion object.
        Useful for testing and one-off generations.

        Args:
            user_message: The user's text
            emotion:      Emotion label string (default: neutral)
        """
        from src.smoothing.context_smoother import SmoothedEmotion

        # Create a minimal SmoothedEmotion for the specified emotion
        mock_smoothed = SmoothedEmotion(
            primary_emotion=emotion,
            confidence=0.8,
            probabilities={
                e: (0.8 if e == emotion else 0.03) for e in settings.emotion_labels
            },
            raw_emotion=emotion,
            raw_confidence=0.8,
            was_smoothed=False,
            window_size=1,
            turn_number=1,
        )
        return self.build(user_message, mock_smoothed)
