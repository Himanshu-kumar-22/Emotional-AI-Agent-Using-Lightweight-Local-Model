"""
app/components/emotion_badge.py
================================
Renders emotion badges in the chat interface.
"""


def render_emotion_badge(
    st, emotion: str, confidence: float, was_smoothed: bool = False
):
    """
    Render an inline emotion badge with color and emoji.

    Args:
        st:           Streamlit module
        emotion:      Emotion label string
        confidence:   Confidence value 0.0-1.0
        was_smoothed: Whether context smoothing changed the label
    """
    from config.settings import settings

    display = settings.emotion_display.get(
        emotion, {"emoji": "😐", "color": "#708090", "label": emotion.title()}
    )

    smoothed_indicator = " ✨" if was_smoothed else ""
    tooltip = "Context smoothing adjusted this emotion" if was_smoothed else ""

    badge_html = f"""
    <span style="
        background-color: {display['color']}22;
        border: 1.5px solid {display['color']};
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.8em;
        color: {display['color']};
        font-weight: 600;
        title='{tooltip}'
    ">
        {display['emoji']} {display['label']} {confidence:.0%}{smoothed_indicator}
    </span>
    """
    st.markdown(badge_html, unsafe_allow_html=True)
