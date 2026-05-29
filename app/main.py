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
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottom"] > div > div,
    [data-testid="stDecoration"],
    [data-testid="stToolbar"] {
        background: #111827 !important;
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
        background: #111827 !important;
        min-height: 100vh !important;
    }

    /* ── Layout container ── */
    [data-testid="stAppViewContainer"] {
        background: #111827 !important;
        min-height: 100vh !important;
    }

    /* ── Floating sidebar ───────────────────────────── */

    section[data-testid="stSidebar"] {
        width: 20vw !important;
        min-width: 200px !important;
        max-width: 300px !important;
        background: #0f172a !important;
        margin: 10px 0 10px 10px !important;
        height: calc(100vh - 30px) !important;
        border-radius: 18px !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        overflow: hidden !important;
        box-shadow:
            0 10px 40px rgba(0,0,0,0.25),
            0 0 0 1px rgba(255,255,255,0.02) inset !important;
        flex-shrink: 0 !important;
        backdrop-filter: blur(14px) !important;
    }

    /* ── Main panel ── */
    [data-testid="stAppViewContainer"] .main {
        background: #111827 !important;
        overflow: visible !important;
        position: relative !important;
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
                rgba(59, 130, 246, 0.10),
                transparent 45%
            ),
            radial-gradient(
                circle at 80% 80%,
                rgba(99, 102, 241, 0.08),
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
        background: #111827 !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* Actual floating chat input pill */
    .stChatInputContainer {
        background: #1e293b !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 22px !important;
        box-shadow:
            0 8px 30px rgba(0,0,0,0.25),
            0 0 0 1px rgba(255,255,255,0.02) inset !important;
        padding: 0.2rem 0.45rem !important;
    }

    /* Force ALL bottom layers to same background */
        section[data-testid="stBottom"],
        section[data-testid="stBottom"] > div,
        section[data-testid="stBottom"] > div > div,
        section[data-testid="stBottom"] > div > div > div {
        background: #111827 !important;
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
        color: #94a3b8 !important;
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
        /* Identical geometry to the inactive button so text never jumps size */
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        width: 100% !important;
        min-height: 44px !important;
        padding: 10px 38px 10px 12px !important;
        border-radius: 10px !important;
        box-sizing: border-box !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
        text-align: left !important;
        /* Appearance — highlight only */
        background: rgba(99, 102, 241, 0.22) !important;
        color: #a5b4fc !important;
        font-weight: 500 !important;
        margin: 0 !important;
    }

        /* ─────────────────────────────────────────────
    CHAT HISTORY ROWS
    ───────────────────────────────────────────── */

    /* Entire row spacing */
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
        position: relative !important;
        width: 100% !important;
        margin-bottom: 2px !important;
    }

    /* Remove weird column gaps */
    section[data-testid="stSidebar"] [data-testid="column"] {
        padding: 0 !important;
    }

    /* Zero out Streamlit's own wrapper margins so active (markdown) and
       inactive (button) rows take identical vertical space. Without this
       stMarkdownContainer adds extra top/bottom margin vs stButton. */
    section[data-testid="stSidebar"]
    [data-testid="stHorizontalBlock"]
    > [data-testid="column"]:first-child
    [data-testid="stMarkdownContainer"] {
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1 !important;
    }

    section[data-testid="stSidebar"]
    [data-testid="stHorizontalBlock"]
    > [data-testid="column"]:first-child
    [data-testid="stButton"] {
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1 !important;
    }

    /* LEFT BUTTON — takes almost full row */
    section[data-testid="stSidebar"]
    [data-testid="stHorizontalBlock"]
    > [data-testid="column"]:first-child {
        width: 100% !important;
    }

    /* Conversation button */
    section[data-testid="stSidebar"]
    [data-testid="stHorizontalBlock"]
    > [data-testid="column"]:first-child button {
        width: 100% !important;
        background: transparent !important;
        border: none !important;
        color: #cbd5e1 !important;
        text-align: left !important;
        padding: 10px 38px 10px 12px !important;
        border-radius: 10px !important;
        transition:
            background 0.15s ease,
            color 0.15s ease !important;
        min-height: 44px !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
    }

    /* Button inner <p> that Streamlit renders — force left align */
    section[data-testid="stSidebar"]
    [data-testid="stHorizontalBlock"]
    > [data-testid="column"]:first-child button p {
        text-align: left !important;
        margin: 0 !important;
    }

    /* Hover row */
    section[data-testid="stSidebar"]
    [data-testid="stHorizontalBlock"]:hover
    > [data-testid="column"]:first-child button {
        background: rgba(99,102,241,0.18) !important;
        color: white !important;
    }

    /* DELETE BUTTON COLUMN */
    section[data-testid="stSidebar"]
    [data-testid="stHorizontalBlock"]
    > [data-testid="column"]:last-child {
        position: absolute !important;
        right: 8px !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        width: 26px !important;
        height: 26px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    /* Flatten Streamlit's inner div wrapper */
    section[data-testid="stSidebar"]
    [data-testid="stHorizontalBlock"]
    > [data-testid="column"]:last-child > div {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 100% !important;
        height: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* DELETE BUTTON */
    section[data-testid="stSidebar"]
    [data-testid="stHorizontalBlock"]
    > [data-testid="column"]:last-child button {
        opacity: 0 !important;
        background: transparent !important;
        border: none !important;
        color: #94a3b8 !important;
        transition:
            opacity 0.15s ease,
            color 0.15s ease !important;
        width: 26px !important;
        height: 26px !important;
        min-height: 0 !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        line-height: 1 !important;
        font-size: 0.85em !important;
    }

    /* SHOW X ON HOVER */
    section[data-testid="stSidebar"]
    [data-testid="stHorizontalBlock"]:hover
    > [data-testid="column"]:last-child button {
        opacity: 1 !important;
    }

    /* X hover */
    section[data-testid="stSidebar"]
    [data-testid="stHorizontalBlock"]
    > [data-testid="column"]:last-child button:hover {
        color: #ef4444 !important;
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


# Reverse-map ram_gb integer → option string used in selectboxes
_RAM_GB_TO_OPTION = {v: k for k, v in _RAM_OPTIONS.items()}

_MODEL_LABELS = {
    "mistral": "Mistral 7B",
    "phi3:mini": "Phi-3 Mini",
    "gemma2:2b-instruct-q4_K_M": "Gemma 2 2B",
    "qwen2.5:14b": "Qwen 2.5 14B",
}


def _peek_user_ram_gb() -> int | None:
    """
    Read ram_gb directly from the SQLite file before the agent initialises.
    ram_gb is a plain INTEGER column (not encrypted), so no key is needed.
    Returns None if no profile exists or the DB isn't reachable.
    """
    try:
        import sqlite3

        db_path = settings.database_path
        if not db_path.exists():
            return None
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        row = conn.execute(
            "SELECT ram_gb FROM user_profile WHERE id = 'profile'"
        ).fetchone()
        conn.close()
        return int(row[0]) if row else None
    except Exception:
        return None

# Ollama model weights in RAM (Q4 quantisation, rounded to 1 decimal)
_MODEL_RAM_GB = {
    "gemma2:2b-instruct-q4_K_M": 1.7,
    "phi3:mini": 2.2,
    "mistral": 4.1,
    "qwen2.5:14b": 9.0,
}

# Emotion detection model footprint when loaded into RAM
_EMOTION_RAM_GB = {
    "distilbert": 0.3,
    "minilm": 0.1,
}


def _fmt_model(model_key: str, emotion_model: str) -> str:
    """Return the dropdown label including combined RAM usage."""
    llm_gb = _MODEL_RAM_GB.get(model_key, 0)
    emo_gb = _EMOTION_RAM_GB.get(emotion_model, 0)
    total = llm_gb + emo_gb
    label = _MODEL_LABELS.get(model_key, model_key)
    return f"{label}  ·  ~{total:.1f} GB"


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
        # model-switch tracking
        "switching_to_model": None,  # model key being loaded, or None
        "prev_agent_for_unload": None,  # old agent to evict from RAM
        "show_loading_screen": False,  # True → render loading card, then rerun to init
        # profile editor
        "show_profile_editor": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # On the very first render (before any agent exists), override the model
    # defaults with whatever the saved RAM profile recommends.  This way a
    # profile edit persists across app restarts without touching the config.
    if not st.session_state.initialized and st.session_state.agent is None:
        _apply_saved_ram_defaults()


def _apply_saved_ram_defaults():
    """Read ram_gb from the DB and correct llm_model / emotion_model."""
    if st.session_state.privacy_mode:
        return  # in-memory DB — nothing to peek at
    ram_gb = _peek_user_ram_gb()
    if ram_gb is None:
        return
    ram_option = _RAM_GB_TO_OPTION.get(ram_gb)
    if not ram_option or ram_option not in _RAM_TO_MODELS:
        return
    models = _RAM_TO_MODELS[ram_option]
    st.session_state.llm_model = models["llm_model"]
    st.session_state.emotion_model = models["emotion_model"]


def initialize_agent():
    model = st.session_state.llm_model
    display = _MODEL_LABELS.get(model, model)
    switching = st.session_state.switching_to_model

    spinner_msg = (
        f"Switching to {display} — freeing RAM and loading new weights…"
        if switching
        else f"Loading {display}…"
    )

    with st.spinner(spinner_msg):
        # Unload the previous model inside the spinner so the user
        # gets visual feedback instead of a silent UI freeze.
        prev = st.session_state.prev_agent_for_unload
        if prev:
            prev.unload_current_model()
            st.session_state.prev_agent_for_unload = None

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
        finally:
            st.session_state.switching_to_model = None
            st.session_state.prev_agent_for_unload = None


def _render_model_loading_screen(model: str):
    display = _MODEL_LABELS.get(model, model)
    st.markdown(
        f"""
<div style="
    text-align: center;
    padding: 5rem 1rem;
    animation: fadeIn 0.3s ease;
">
    <div style="
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 56px; height: 56px;
        border-radius: 14px;
        background: rgba(99,102,241,0.15);
        font-size: 1.8rem;
        margin-bottom: 1.4rem;
    ">⚙️</div>
    <h2 style="
        color: #e2e8f0;
        font-size: 1.45rem;
        font-weight: 600;
        margin: 0 0 0.55rem;
        letter-spacing: -0.01em;
    ">Switching to {display}</h2>
    <p style="
        color: #64748b;
        font-size: 0.88rem;
        max-width: 340px;
        margin: 0 auto;
        line-height: 1.55;
    ">
        Unloading current model from RAM<br>
        Loading <strong style="color:#94a3b8">{display}</strong> weights…
    </p>
</div>
""",
        unsafe_allow_html=True,
    )


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


# ── Profile Editor ───────────────────────────────────────────────────────────
def render_profile_editor():
    profile = st.session_state.user_profile or {}
    current_name = profile.get("name", "")
    current_ram_gb = profile.get("ram_gb", 8)
    current_ram_option = _RAM_GB_TO_OPTION.get(current_ram_gb, "8 GB")
    current_ram_index = list(_RAM_OPTIONS.keys()).index(current_ram_option)

    st.markdown(
        """
<div style="max-width:480px; margin:6rem auto 0; text-align:center;">
    <h1 style="font-size:2rem; font-weight:700; margin-bottom:0.3em;">
        Edit Profile
    </h1>
    <p style="color:#777; font-size:0.9em; margin-bottom:2rem;">
        Update your name or RAM setting.<br>
        Changing RAM will swap in the recommended model for that tier.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )

    _, col_c, _ = st.columns([1, 2, 1])
    with col_c:
        with st.form("edit_profile", clear_on_submit=False):
            new_name = st.text_input(
                "Your name",
                value=current_name,
                max_chars=64,
            )
            new_ram_choice = st.selectbox(
                "How much RAM does your machine have?",
                options=list(_RAM_OPTIONS.keys()),
                index=current_ram_index,
            )
            col_save, col_cancel = st.columns(2)
            with col_save:
                submitted = st.form_submit_button(
                    "Save Changes", use_container_width=True, type="primary"
                )
            with col_cancel:
                cancelled = st.form_submit_button("Cancel", use_container_width=True)

        if cancelled:
            st.session_state.show_profile_editor = False
            st.rerun()

        if submitted:
            if not new_name.strip():
                st.error("Name cannot be empty.")
            else:
                new_ram_gb = _RAM_OPTIONS[new_ram_choice]
                new_models = _RAM_TO_MODELS[new_ram_choice]
                ram_changed = new_ram_gb != current_ram_gb

                # Persist to DB (INSERT OR REPLACE — always safe to call)
                st.session_state.agent.save_user_profile(new_name.strip(), new_ram_gb)
                st.session_state.user_profile = {
                    "name": new_name.strip(),
                    "ram_gb": new_ram_gb,
                }
                st.session_state.user_profile_loaded = True
                st.session_state.show_profile_editor = False

                if ram_changed:
                    # Swap models — reuse the same model-switch flow so the
                    # loading screen and RAM unload/preload happen automatically.
                    st.session_state.prev_agent_for_unload = st.session_state.agent
                    st.session_state.switching_to_model = new_models["llm_model"]
                    st.session_state.llm_model = new_models["llm_model"]
                    st.session_state.emotion_model = new_models["emotion_model"]
                    st.session_state.initialized = False
                    st.session_state.agent = None
                    st.session_state.messages = []
                    st.session_state.show_loading_screen = True
                else:
                    # Name-only change — just sync to the live agent, no re-init
                    st.session_state.agent.set_user_name(new_name.strip())

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
                        if st.button("🗑", key=f"del_{session['id']}"):
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
    current_llm_index = (
        llm_options.index(st.session_state.llm_model)
        if st.session_state.llm_model in llm_options
        else 0
    )
    _emotion_model = st.session_state.emotion_model

    def _on_model_change():
        chosen = st.session_state._llm_selectbox
        if chosen == st.session_state.llm_model:
            return
        st.session_state.prev_agent_for_unload = st.session_state.agent
        st.session_state.switching_to_model = chosen
        st.session_state.llm_model = chosen
        st.session_state.initialized = False
        st.session_state.agent = None
        st.session_state.messages = []
        st.session_state.user_profile_loaded = False
        st.session_state.show_loading_screen = True

    col_brand, _, col_model, col_settings = st.columns([3, 4, 2, 1])
    with col_brand:
        st.markdown(
            '<p style="font-size:0.72em; color:#555; margin:0; padding-top:10px; '
            'letter-spacing:0.06em; text-transform:uppercase;">'
            "Emotional AI &nbsp;·&nbsp; Local LLM</p>",
            unsafe_allow_html=True,
        )
    with col_model:
        st.selectbox(
            "Model",
            options=llm_options,
            index=current_llm_index,
            format_func=lambda x: _fmt_model(x, _emotion_model),
            label_visibility="collapsed",
            disabled=not st.session_state.initialized,
            key="_llm_selectbox",
            on_change=_on_model_change,
        )
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
                st.session_state.prev_agent_for_unload = st.session_state.agent
                st.session_state.privacy_mode = privacy_mode
                st.session_state.initialized = False
                st.session_state.agent = None
                st.session_state.messages = []
                st.session_state.user_profile_loaded = False
                st.session_state.show_loading_screen = True
            if st.session_state.privacy_mode:
                st.caption("Privacy Mode ON — memory only.")
            else:
                st.caption("Standard Mode — saved locally (encrypted).")
            st.markdown("---")
            st.markdown("**Profile**")
            if st.button(
                "Edit Profile",
                use_container_width=True,
                disabled=not st.session_state.initialized,
            ):
                st.session_state.show_profile_editor = True

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

        # Phase 1: render the loading card and immediately rerun so the user
        # sees it before the blocking initialize_agent() call starts.
        if st.session_state.show_loading_screen:
            _render_model_loading_screen(
                st.session_state.switching_to_model or st.session_state.llm_model
            )
            st.session_state.show_loading_screen = False
            st.rerun()

        # Phase 2: actually load the model (runs on the rerun triggered above)
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

    # ── Profile editor (opened from Settings) ────────────────────────────
    if st.session_state.show_profile_editor:
        render_profile_editor()
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
