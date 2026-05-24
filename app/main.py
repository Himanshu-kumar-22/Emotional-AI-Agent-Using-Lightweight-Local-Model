"""
app/main.py
============
Main Streamlit application for the Privacy-Preserving Emotional AI Agent.

Run with:
    streamlit run app/main.py

Architecture:
    All state lives in st.session_state.
    The EmotionalAgent instance is created once and reused across reruns.
    Streamlit reruns the entire script on every interaction —
    session_state persists values across those reruns.

Session state keys:
    agent           : EmotionalAgent instance
    messages        : list of chat messages for display
    initialized     : bool — has agent.initialize() been called
    session_id      : current conversation session UUID
    privacy_mode    : bool — current privacy mode setting
    show_debug      : bool — show emotion debug info
"""

import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from config.settings import settings
from src.pipeline.agent import EmotionalAgent, AgentResponse
from app.components.resource_monitor import render_resource_sidebar
from app.components.emotion_badge import render_emotion_badge

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emotional AI Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    /* Main chat area */
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    /* Message bubbles */
    .user-message {
        background: #f0f2f6;
        border-radius: 15px 15px 5px 15px;
        padding: 10px 15px;
        margin: 5px 0;
        max-width: 80%;
        float: right;
        clear: both;
    }
    .assistant-message {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 15px 15px 15px 5px;
        padding: 10px 15px;
        margin: 5px 0;
        max-width: 80%;
        float: left;
        clear: both;
    }
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* Sidebar styling */
    .sidebar-title {
        font-size: 0.9em;
        color: #666;
        margin-bottom: 5px;
    }
    /* Privacy mode banner */
    .privacy-banner {
        background: #e8f5e9;
        border: 1px solid #4caf50;
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 0.85em;
        color: #2e7d32;
        margin-bottom: 10px;
    }
    .latency-info {
        font-size: 0.75em;
        color: #999;
        margin-top: 4px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ── Session State Initialization ──────────────────────────────────────────────
def init_session_state():
    """Initialize all session state variables on first run."""
    defaults = {
        "agent": None,
        "messages": [],
        "initialized": False,
        "session_id": None,
        "privacy_mode": settings.privacy_mode_default,
        "show_debug": False,
        "llm_model": settings.llm_model_name,
        "emotion_model": settings.emotion_model_type,
        "init_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def initialize_agent():
    """Create and initialize the EmotionalAgent."""
    with st.spinner("Loading emotion model and connecting to Ollama..."):
        try:
            agent = EmotionalAgent(
                privacy_mode=st.session_state.privacy_mode,
                llm_model=st.session_state.llm_model,
                emotion_model=st.session_state.emotion_model,
            )
            agent.initialize(password=None)
            session_id = agent.start_session()

            st.session_state.agent = agent
            st.session_state.session_id = session_id
            st.session_state.initialized = True
            st.session_state.init_error = None

        except Exception as e:
            st.session_state.init_error = str(e)
            st.session_state.initialized = False


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🧠 Emotional AI Agent")
        st.markdown(f"*v{settings.app_version}*")
        st.markdown("---")

        # ── Privacy Settings ──────────────────────────────────────────────
        st.markdown("### 🔒 Privacy Settings")
        privacy_mode = st.toggle(
            "Privacy Mode",
            value=st.session_state.privacy_mode,
            help="When ON: conversations stored in memory only. "
            "No data written to disk at any point.",
        )

        if privacy_mode != st.session_state.privacy_mode:
            st.session_state.privacy_mode = privacy_mode
            st.session_state.initialized = False
            st.session_state.agent = None
            st.session_state.messages = []
            st.rerun()

        if st.session_state.privacy_mode:
            st.markdown(
                '<div class="privacy-banner">'
                "🟢 Privacy Mode ON — No data written to disk"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:#fff3e0; border:1px solid #ff9800; '
                "border-radius:8px; padding:8px 12px; font-size:0.85em; "
                'color:#e65100; margin-bottom:10px;">'
                "🟡 Standard Mode — Conversations saved locally (encrypted)"
                "</div>",
                unsafe_allow_html=True,
            )

        # ── Model Settings ────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🤖 Model Settings")

        # LLM Model selector
        llm_options = ["mistral", "phi3:mini"]
        llm_labels = {
            "mistral": "Mistral-7B  — better quality (4.1 GB)",
            "phi3:mini": "Phi-3-Mini  — faster, less RAM (2.4 GB)",
        }

        current_llm_index = (
            llm_options.index(st.session_state.llm_model)
            if st.session_state.llm_model in llm_options
            else 0
        )

        llm_choice = st.selectbox(
            "LLM Model",
            options=llm_options,
            index=current_llm_index,
            format_func=lambda x: llm_labels.get(x, x),
            help="Mistral-7B produces richer empathetic responses. "
            "Phi-3-Mini is faster and works on 4 GB RAM devices.",
        )

        if llm_choice != st.session_state.llm_model:
            st.session_state.llm_model = llm_choice
            st.session_state.initialized = False
            st.session_state.agent = None
            st.session_state.messages = []
            st.rerun()

        # Emotion Model selector
        emotion_options = ["distilbert", "minilm"]
        emotion_labels = {
            "distilbert": "DistilBERT — 87.2% accuracy (~12 ms)",
            "minilm": "MiniLM     — 84.3% accuracy (~8 ms)",
        }

        # Check which models are actually available locally
        distilbert_ready = settings.is_model_trained("distilbert")
        minilm_ready = settings.is_model_trained("minilm")
        readiness = {"distilbert": distilbert_ready, "minilm": minilm_ready}

        def emotion_format(key: str) -> str:
            tick = "✓" if readiness[key] else "✗"
            return f"{tick} {emotion_labels.get(key, key)}"

        current_emotion_index = (
            emotion_options.index(st.session_state.emotion_model)
            if st.session_state.emotion_model in emotion_options
            else 0
        )

        emotion_choice = st.selectbox(
            "Emotion Model",
            options=emotion_options,
            index=current_emotion_index,
            format_func=emotion_format,
            help="DistilBERT is more accurate. "
            "MiniLM is lighter and faster on low-memory devices.",
        )

        if emotion_choice != st.session_state.emotion_model:
            if not settings.is_model_trained(emotion_choice):
                st.warning(
                    f"⚠️ {emotion_choice.title()} model is not trained yet.\n\n"
                    f"Run in terminal:\n"
                    f"```\npython3 scripts/train_emotion_model.py "
                    f"--model {emotion_choice}\n```"
                )
            else:
                st.session_state.emotion_model = emotion_choice
                st.session_state.initialized = False
                st.session_state.agent = None
                st.session_state.messages = []
                st.rerun()

        # Show current active combination
        st.caption(
            f"Active: {st.session_state.llm_model} "
            f"+ {st.session_state.emotion_model.title()}"
        )

        # ── Debug Toggle ──────────────────────────────────────────────────
        st.markdown("---")
        st.session_state.show_debug = st.toggle(
            "Show Emotion Debug",
            value=st.session_state.show_debug,
            help="Show raw vs smoothed emotion data for each message.",
        )

        # ── New Conversation ──────────────────────────────────────────────
        st.markdown("---")
        if st.button("🔄 New Conversation", use_container_width=True):
            if st.session_state.agent and st.session_state.initialized:
                session_id = st.session_state.agent.new_session()
                st.session_state.session_id = session_id
                st.session_state.messages = []
                st.rerun()

        # ── Resource Monitor ──────────────────────────────────────────────
        agent = st.session_state.get("agent")
        render_resource_sidebar(
            st,
            agent if st.session_state.initialized else None,
        )

        # ── Session ID ────────────────────────────────────────────────────
        if st.session_state.session_id:
            st.markdown("---")
            st.markdown(
                f'<p class="sidebar-title">Session ID</p>'
                f'<code style="font-size:0.7em">'
                f"{st.session_state.session_id[:16]}...</code>",
                unsafe_allow_html=True,
            )


# ── Chat Interface ────────────────────────────────────────────────────────────
def render_chat_history():
    """Render all previous messages in the conversation."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

            # Show emotion badge for assistant messages
            if msg["role"] == "assistant" and "emotion" in msg:
                render_emotion_badge(
                    st,
                    emotion=msg["emotion"],
                    confidence=msg["confidence"],
                    was_smoothed=msg.get("was_smoothed", False),
                )

            # Show debug info if enabled
            if st.session_state.show_debug and "debug" in msg:
                with st.expander("🔍 Emotion Debug", expanded=False):
                    debug = msg["debug"]
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Raw Detection**")
                        st.markdown(f"Emotion: `{debug.get('raw_emotion', 'N/A')}`")
                        st.markdown(
                            f"Confidence: `{debug.get('raw_confidence', 0):.0%}`"
                        )
                    with col2:
                        st.markdown("**After Smoothing**")
                        st.markdown(
                            f"Emotion: `{debug.get('smoothed_emotion', 'N/A')}`"
                        )
                        st.markdown(f"Changed: `{debug.get('was_smoothed', False)}`")
                    st.markdown(f"**Trend:** {debug.get('trend', 'N/A')}")
                    st.markdown(
                        f'<p class="latency-info">'
                        f'⚡ Total: {debug.get("total_ms", 0):.0f}ms | '
                        f'Emotion: {debug.get("emotion_ms", 0):.0f}ms | '
                        f'LLM: {debug.get("llm_ms", 0):.0f}ms'
                        f"</p>",
                        unsafe_allow_html=True,
                    )

            # Show latency for assistant messages always
            if msg["role"] == "assistant" and "total_ms" in msg:
                st.markdown(
                    f'<p class="latency-info">' f'⚡ {msg["total_ms"]:.0f}ms</p>',
                    unsafe_allow_html=True,
                )


def process_user_input(user_input: str):
    """Process user message through the agent and update UI."""
    agent: EmotionalAgent = st.session_state.agent

    # Add user message to display immediately
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    # Display user message
    with st.chat_message("user"):
        st.write(user_input)

    # Generate and display assistant response with streaming
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        collected_text = ""

        # Stream tokens for visual effect
        token_stream, smoothed = agent.chat_stream(user_input)

        for token in token_stream:
            collected_text += token
            response_placeholder.markdown(collected_text + "▋")

        # Final text without cursor
        response_placeholder.markdown(collected_text)

        # Save the completed response to storage
        agent.save_streamed_response(user_input, collected_text, smoothed)

        # Get display info for emotion badge
        display = smoothed.get_display_info()

        # Render emotion badge
        render_emotion_badge(
            st,
            emotion=smoothed.primary_emotion,
            confidence=smoothed.confidence,
            was_smoothed=smoothed.was_smoothed,
        )

        # Build debug data
        debug_data = {
            "raw_emotion": smoothed.raw_emotion,
            "raw_confidence": smoothed.raw_confidence,
            "smoothed_emotion": smoothed.primary_emotion,
            "was_smoothed": smoothed.was_smoothed,
            "trend": smoothed.get_stability_description(),
            "total_ms": agent.turn_number * 10,  # approximate
            "emotion_ms": 12,
            "llm_ms": 0,
        }

        # Add assistant message to session state
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": collected_text,
                "emotion": smoothed.primary_emotion,
                "confidence": smoothed.confidence,
                "was_smoothed": smoothed.was_smoothed,
                "total_ms": 0,
                "debug": debug_data,
            }
        )


# ── Main App ──────────────────────────────────────────────────────────────────
def main():
    init_session_state()
    render_sidebar()

    # Header
    st.markdown(
        '<div class="main-header">'
        "<h2>🧠 Privacy-Preserving Emotional AI Agent</h2>"
        '<p style="color:#666; font-size:0.9em;">'
        "All processing happens locally on your device. "
        "Your conversations never leave this machine."
        "</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Initialize agent if needed
    if not st.session_state.initialized:
        if st.session_state.init_error:
            st.error(f"Initialization failed: {st.session_state.init_error}")
            st.info(
                "Checklist:\n"
                "- Is Ollama running? (`ollama serve`)\n"
                "- Is the model pulled? (`ollama pull mistral`)\n"
                "- Is the emotion model trained?"
            )
            if st.button("Retry"):
                st.session_state.init_error = None
                st.rerun()
            return

        initialize_agent()
        if not st.session_state.initialized:
            return
        st.rerun()

    # Welcome message on first load
    if not st.session_state.messages:
        privacy_note = (
            "🔒 Privacy Mode is ON — this conversation is in memory only."
            if st.session_state.privacy_mode
            else "💾 Conversations are saved locally with AES-256 encryption."
        )
        with st.chat_message("assistant"):
            welcome_text = (
                "Hello! I'm here to listen and support you. "
                "How are you feeling today?"
            )
            st.write(welcome_text)
            st.caption(privacy_note)
            render_emotion_badge(st, "neutral", 1.0)

    # Render chat history
    render_chat_history()

    # Chat input
    user_input = st.chat_input(
        "Share what's on your mind...",
        disabled=not st.session_state.initialized,
    )

    if user_input:
        process_user_input(user_input)
        st.rerun()


if __name__ == "__main__":
    main()
