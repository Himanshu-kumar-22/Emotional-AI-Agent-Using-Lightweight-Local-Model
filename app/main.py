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
    show_debug      : bool — show emotion debug info per message
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from config.settings import settings
from src.pipeline.agent import EmotionalAgent
from app.components.emotion_badge import render_emotion_badge

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emotional AI Agent",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Hide top-right deploy button / toolbar */
        [data-testid="stHeader"] {
        display: none !important;
    }

    /* Kill Streamlit bottom/background layers completely */
    [data-testid="stBottomBlockContainer"],
    [data-testid="stBottomBlockContainer"] > div,
    [data-testid="stDecoration"],
    [data-testid="stToolbar"] {
        background: #0d0d0d !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* ── Hide ALL sidebar collapse / expand controls ── */
    [data-testid="collapsedControl"],
    button[data-testid="baseButton-headerNoPadding"],
    section[data-testid="stSidebar"] button[kind="header"],
    button[aria-label="Close sidebar"],
    button[aria-label="Collapse sidebar"],
    button[title="Close sidebar"],
    .st-emotion-cache-1cypcdb,
    [data-testid="stSidebarCollapseButton"] {
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
    }

    /* ── Force sidebar always expanded ── */
    section[data-testid="stSidebar"] {
        transform: none !important;
        transition: none !important;
        left: 0 !important;
        visibility: visible !important;
        display: block !important;
    }
    section[data-testid="stSidebar"][aria-expanded="false"] {
        transform: none !important;
        width: 20vw !important;
        min-width: 200px !important;
        max-width: 300px !important;
        visibility: visible !important;
        display: block !important;
        margin-left: 0 !important;
    }

    /* ── Pure black everywhere ── */
    .stApp {
        background: #0d0d0d !important;
        min-height: 100vh !important;
    }

    /* ── Layout container — edge-to-edge, no padding/gaps ── */
    [data-testid="stAppViewContainer"] {
        padding: 0 !important;
        gap: 0 !important;
        align-items: stretch !important;
        height: 100vh !important;
        background: #0d0d0d !important;
    }

    /* ── Floating sidebar ───────────────────────────── */

    section[data-testid="stSidebar"] {
        width: 20vw !important;
        min-width: 200px !important;
        max-width: 300px !important;
        background: #0d0d0d !important;
        margin: 10px 0 10px 10px !important;
        height: calc(100vh - 30px) !important;
        border-radius: 18px !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        overflow: hidden !important;
        box-shadow:
            0 10px 30px rgba(0,0,0,0.35),
            0 0 0 1px rgba(255,255,255,0.02) inset !important;
        flex-shrink: 0 !important;
        backdrop-filter: blur(14px) !important;
    }

    /* ── Main panel — same flat dark, no decoration ── */
    [data-testid="stAppViewContainer"] .main {
        flex: 1 !important;
        margin: 0 !important;
        border-radius: 0 !important;
        border: none !important;
        overflow: visible !important;
        position: relative !important;
        box-shadow: none !important;
        background: #0d0d0d !important;
    }

    /* Blur disabled until gradient is correct */
    /* Subtle ambient blur/glow */

    [data-testid="stAppViewContainer"] .main::before {
        content: "";
        position: absolute;
        inset: 0;
        pointer-events: none;
        z-index: 0;
        background:
            radial-gradient(
                circle at 50% 20%,
                rgba(80, 90, 140, 0.10),
                transparent 45%
            ),
            radial-gradient(
                circle at 80% 80%,
                rgba(60, 70, 110, 0.08),
                transparent 40%
            );
        filter: blur(70px);
        opacity: 0.7;
    }

    /* ── Center chat content (works open AND collapsed) ── */
    [data-testid="stMainBlockContainer"] {
        max-width: 820px !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }

    /* ─────────────────────────────────────────────
   CHAT INPUT — TRUE CHATGPT STYLE
───────────────────────────────────────────── */

    /* Bottom dock — match main background exactly */
    section[data-testid="stBottom"],

    section[data-testid="stBottom"] {
        padding: 0 calc(50% - 410px) 18px !important;
    }

    /* Wipe any Streamlit wrapper backgrounds */
    section[data-testid="stBottom"] > div,
    section[data-testid="stBottom"] > div > div,
    section[data-testid="stBottom"] > div > div > div {
        background: #0d0d0d !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* Actual floating chat input pill */
    .stChatInputContainer {
        background: #0d0d0d  !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 22px !important;
        box-shadow:
            0 8px 30px rgba(0,0,0,0.35),
            0 0 0 1px rgba(255,255,255,0.02) inset !important;
        padding: 0.2rem 0.45rem !important;
    }

    /* Force ALL bottom layers to same black */
        section[data-testid="stBottom"],
        section[data-testid="stBottom"] > div,
        section[data-testid="stBottom"] > div > div,
        section[data-testid="stBottom"] > div > div > div {
        background: #0d0d0d !important;
    }

    /* Inner area */
    .stChatInputContainer > div {
        background: transparent !important;
        border: none !important;
    }

    /* Text area */
    .stChatInputContainer textarea {
        background: transparent !important;
        border: none !important;
        color: white !important;
        font-size: 0.96rem !important;
        box-shadow: none !important;
    }

    /* Placeholder */
    .stChatInputContainer textarea::placeholder {
        color: #7d7d86 !important;
    }

    /* Remove focus glow */
    .stChatInputContainer textarea:focus {
        box-shadow: none !important;
    }

    /* Send button */
    .stChatInputContainer button {
        border-radius: 12px !important;
    }

    /* ── Orbs disabled while tuning ── */
    .bg-orb { display: none !important; }

    /* ── Component styles ── */
    .privacy-banner {
        background: rgba(46, 125, 50, 0.15);
        border: 1px solid rgba(76, 175, 80, 0.4);
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 0.85em;
        color: #81c784;
        margin-bottom: 10px;
    }
    .latency-info {
        font-size: 0.75em;
        color: #999;
        margin-top: 4px;
    }
    .session-active {
        background: rgba(100, 110, 230, 0.18);
        border-radius: 6px;
        padding: 6px 10px;
        font-size: 0.85em;
        color: #a0aaff;
        font-weight: 500;
        margin: 2px 0;
    }
    /* Session history item buttons */
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child button {
        background: transparent !important;
        border: none !important;
        text-align: left !important;
        font-size: 0.85em !important;
        color: #ccc !important;
        padding: 4px 8px !important;
        border-radius: 6px !important;
        min-height: 0 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child button:hover {
        background: rgba(255,255,255,0.08) !important;
        color: #fff !important;
    }
    /* Delete (×) buttons */
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > [data-testid="column"]:last-child button {
        background: transparent !important;
        border: none !important;
        color: #444 !important;
        font-size: 0.9em !important;
        padding: 4px 6px !important;
        min-height: 0 !important;
        border-radius: 4px !important;
    }
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > [data-testid="column"]:last-child button:hover {
        color: #ff4d4d !important;
        background: rgba(255, 77, 77, 0.12) !important;
    }
    /* Neon green New Chat button */
    section[data-testid="stSidebar"] [data-testid="baseButton-primary"] {
        background-color: #39ff14 !important;
        border-color: #39ff14 !important;
        color: #000 !important;
        font-weight: 600 !important;
        box-shadow: 0 0 10px rgba(57, 255, 20, 0.45) !important;
    }
    section[data-testid="stSidebar"] [data-testid="baseButton-primary"]:hover {
        background-color: #2ee00f !important;
        box-shadow: 0 0 20px rgba(57, 255, 20, 0.75) !important;
    }
    /* ── App hero heading (empty state) ── */
    .app-hero {
        text-align: center;
        padding: 3.5rem 1rem 1.5rem;
    }
    .app-hero h1 {
        font-size: 2.5em;
        font-weight: 700;
        letter-spacing: -0.03em;
        margin-bottom: 0.45em;
        line-height: 1.1;
    }
    .app-hero .tagline {
        color: #777;
        font-size: 0.88em;
        letter-spacing: 0.02em;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ── Background decorative orbs ────────────────────────────────────────────────
st.markdown(
    """
<div class="bg-orb" style="top:-15%;left:30%;width:50vw;height:50vw;
    background:radial-gradient(circle,rgba(80,90,130,0.12) 0%,transparent 65%);"></div>
<div class="bg-orb" style="bottom:-12%;right:8%;width:40vw;height:40vw;
    background:radial-gradient(circle,rgba(60,70,110,0.10) 0%,transparent 65%);"></div>
""",
    unsafe_allow_html=True,
)


# ── Session State ─────────────────────────────────────────────────────────────
_RAM_OPTIONS = {
    "4 GB": 4,
    "8 GB": 8,
    "16 GB": 16,
    "32 GB or more": 32,
}

# Maps each RAM tier to the best-fit emotion model + LLM combo.
_RAM_TO_MODELS = {
    "4 GB": {"emotion_model": "minilm", "llm_model": "gemma2:2b-instruct-q4_K_M"},
    "8 GB": {"emotion_model": "distilbert", "llm_model": "gemma2:2b-instruct-q4_K_M"},
    "16 GB": {"emotion_model": "distilbert", "llm_model": "mistral"},
    "32 GB or more": {"emotion_model": "distilbert", "llm_model": "qwen2.5:14b"},
}


def init_session_state():
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
        "user_profile": None,
        "user_profile_loaded": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def initialize_agent():
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


# ── Helpers ───────────────────────────────────────────────────────────────────
def _storage_messages_to_display(messages: list[dict]) -> list[dict]:
    return [{"role": m["role"], "content": m["content"]} for m in messages]


# ── First-Run Setup Screen ────────────────────────────────────────────────────
def render_setup_screen():
    st.markdown(
        """
<div style="max-width:480px; margin:6rem auto 0; text-align:center;">
    <h1 style="font-size:2rem; font-weight:700; margin-bottom:0.3em;">
        Welcome
    </h1>
    <p style="color:#777; font-size:0.9em; margin-bottom:2rem;">
        Quick setup so I can address you properly and suggest suitable models.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )

    _, col_c, _ = st.columns([1, 2, 1])
    with col_c:
        with st.form("first_run_setup", clear_on_submit=False):
            name = st.text_input(
                "Your name",
                placeholder="What should I call you?",
                max_chars=64,
            )
            ram_choice = st.selectbox(
                "How much RAM does your machine have?",
                options=list(_RAM_OPTIONS.keys()),
                index=1,
            )
            submitted = st.form_submit_button(
                "Get Started", use_container_width=True, type="primary"
            )

        if submitted:
            if not name.strip():
                st.error("Please enter your name so I know what to call you.")
            else:
                ram_gb = _RAM_OPTIONS[ram_choice]
                models = _RAM_TO_MODELS[ram_choice]
                agent: EmotionalAgent = st.session_state.agent
                agent.save_user_profile(name.strip(), ram_gb)
                st.session_state.user_profile = {"name": name.strip(), "ram_gb": ram_gb}
                st.session_state.user_profile_loaded = True
                # Apply RAM-matched models and force agent re-init
                st.session_state.llm_model = models["llm_model"]
                st.session_state.emotion_model = models["emotion_model"]
                st.session_state.initialized = False
                st.session_state.agent = None
                st.session_state.messages = []
                st.rerun()


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown(
            f"## Emotional AI Agent "
            f'<span style="font-size:0.45em; color:#888; font-weight:400; '
            f'vertical-align:middle;">v{settings.app_version}</span>',
            unsafe_allow_html=True,
        )

        # New Chat button — right at top
        if st.button("New Chat", use_container_width=True, type="primary"):
            if st.session_state.agent and st.session_state.initialized:
                session_id = st.session_state.agent.new_session()
                st.session_state.session_id = session_id
                st.session_state.messages = []
                st.rerun()

        st.markdown("---")

        # ── Chat History ──────────────────────────────────────────────────
        agent = st.session_state.get("agent")
        if agent and st.session_state.initialized:
            sessions = agent.list_sessions_with_titles(limit=30)
            if sessions:
                for session in sessions:
                    title = session["title"]
                    is_active = session["id"] == st.session_state.session_id
                    col_sess, col_del = st.columns([5, 1])
                    with col_sess:
                        if is_active:
                            st.markdown(
                                f'<div class="session-active">{title}</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            if st.button(
                                title,
                                key=f"sess_{session['id']}",
                                use_container_width=True,
                            ):
                                msgs = agent.load_session(session["id"])
                                st.session_state.session_id = session["id"]
                                st.session_state.messages = (
                                    _storage_messages_to_display(msgs)
                                )
                                st.rerun()
                    with col_del:
                        if st.button("×", key=f"del_{session['id']}"):
                            agent.delete_session(session["id"])
                            if session["id"] == st.session_state.session_id:
                                new_id = agent.new_session()
                                st.session_state.session_id = new_id
                                st.session_state.messages = []
                            st.rerun()
            else:
                st.caption("No past conversations yet.")
        else:
            st.caption("Conversations will appear here.")


# ── Chat Interface ────────────────────────────────────────────────────────────
def render_chat_history():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

            if msg["role"] == "assistant" and "emotion" in msg:
                render_emotion_badge(
                    st,
                    emotion=msg["emotion"],
                    confidence=msg["confidence"],
                    was_smoothed=msg.get("was_smoothed", False),
                )

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

            if msg["role"] == "assistant" and "total_ms" in msg:
                st.markdown(
                    f'<p class="latency-info">⚡ {msg["total_ms"]:.0f}ms</p>',
                    unsafe_allow_html=True,
                )


def process_user_input(user_input: str):
    agent: EmotionalAgent = st.session_state.agent

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        collected_text = ""

        token_stream, smoothed = agent.chat_stream(user_input)
        for token in token_stream:
            collected_text += token
            response_placeholder.markdown(collected_text + "▋")

        response_placeholder.markdown(collected_text)
        agent.save_streamed_response(user_input, collected_text, smoothed)

        render_emotion_badge(
            st,
            emotion=smoothed.primary_emotion,
            confidence=smoothed.confidence,
            was_smoothed=smoothed.was_smoothed,
        )

        debug_data = {
            "raw_emotion": smoothed.raw_emotion,
            "raw_confidence": smoothed.raw_confidence,
            "smoothed_emotion": smoothed.primary_emotion,
            "was_smoothed": smoothed.was_smoothed,
            "trend": smoothed.get_stability_description(),
            "total_ms": agent.turn_number * 10,
            "emotion_ms": 12,
            "llm_ms": 0,
        }

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

    # ── Top bar: brand (left) · model selector · settings (right) ──────
    llm_options = ["mistral", "phi3:mini", "gemma2:2b-instruct-q4_K_M", "qwen2.5:14b"]
    llm_labels = {
        "mistral": "Mistral-7B",
        "phi3:mini": "Phi-3-Mini",
        "gemma2:2b-instruct-q4_K_M": "Gemma 2 2B",
        "qwen2.5:14b": "Qwen 2.5 14B",
    }
    current_llm_index = (
        llm_options.index(st.session_state.llm_model)
        if st.session_state.llm_model in llm_options
        else 0
    )

    col_brand, _, col_model, col_settings = st.columns([3, 4, 2, 1])
    with col_brand:
        st.markdown(
            '<p style="font-size:0.72em; color:#555; margin:0; padding-top:10px; '
            'letter-spacing:0.06em; text-transform:uppercase;">'
            "Emotional AI &nbsp;·&nbsp; Local LLM</p>",
            unsafe_allow_html=True,
        )
    with col_model:
        llm_choice = st.selectbox(
            "Model",
            options=llm_options,
            index=current_llm_index,
            format_func=lambda x: llm_labels.get(x, x),
            label_visibility="collapsed",
            disabled=not st.session_state.initialized,
        )
        if llm_choice != st.session_state.llm_model:
            st.session_state.llm_model = llm_choice
            st.session_state.initialized = False
            st.session_state.agent = None
            st.session_state.messages = []
            st.session_state.user_profile_loaded = False
            st.rerun()
    with col_settings:
        with st.popover("Settings"):
            st.session_state.show_debug = st.toggle(
                "Show Emotion Debug",
                value=st.session_state.show_debug,
                help="Show raw vs smoothed emotion data for each message.",
            )
            st.markdown("---")
            st.markdown("**Privacy**")
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
                st.session_state.user_profile_loaded = False
                st.rerun()
            if st.session_state.privacy_mode:
                st.caption("Privacy Mode ON — memory only.")
            else:
                st.caption("Standard Mode — saved locally (encrypted).")

    # ── Initialize agent if needed ────────────────────────────────────────
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

    # ── Load user profile once after agent is ready ───────────────────────
    if not st.session_state.user_profile_loaded:
        profile = st.session_state.agent.get_user_profile()
        st.session_state.user_profile = profile
        st.session_state.user_profile_loaded = True

    # Always sync name to the (possibly freshly re-initialised) agent
    if st.session_state.user_profile:
        st.session_state.agent.set_user_name(st.session_state.user_profile["name"])

    # ── First-run setup gate ──────────────────────────────────────────────
    if st.session_state.user_profile is None:
        render_setup_screen()
        return

    # ── Welcome / empty state ─────────────────────────────────────────────
    if not st.session_state.messages:
        st.markdown(
            '<div class="app-hero">'
            "<h1>Emotional AI Agent</h1>"
            '<p class="tagline">'
            "Privacy-preserving &nbsp;·&nbsp; Powered by local LLM"
            "&nbsp;·&nbsp; Runs entirely on your device"
            "</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        privacy_note = (
            "Privacy Mode is ON — this conversation is in memory only."
            if st.session_state.privacy_mode
            else "Conversations are saved locally with AES-256 encryption."
        )
        with st.chat_message("assistant"):
            user_name = st.session_state.user_profile.get("name", "")
            greeting = (
                f"Hello, {user_name}! I'm here to listen and support you. "
                "How are you feeling today?"
                if user_name
                else "Hello! I'm here to listen and support you. "
                "How are you feeling today?"
            )
            st.write(greeting)
            st.caption(privacy_note)
            render_emotion_badge(st, "neutral", 1.0)

    render_chat_history()

    # ── Chat input ────────────────────────────────────────────────────────
    user_input = st.chat_input(
        "Share what's on your mind...",
        disabled=not st.session_state.initialized,
    )

    if user_input:
        process_user_input(user_input)
        st.rerun()


if __name__ == "__main__":
    main()
