"""
app/components/resource_monitor.py
====================================
Live CPU, RAM, and GPU resource monitoring for the sidebar.
"""

import psutil
import platform


def get_resource_stats() -> dict:
    """Collect current system resource usage."""
    stats = {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "ram_used_gb": psutil.virtual_memory().used / (1024**3),
        "ram_total_gb": psutil.virtual_memory().total / (1024**3),
        "ram_percent": psutil.virtual_memory().percent,
        "platform": platform.system(),
        "machine": platform.machine(),
    }

    # Try to get GPU stats for Apple Silicon
    try:
        import subprocess

        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if "Apple" in result.stdout and "M" in result.stdout:
            stats["gpu_info"] = "Apple Silicon (Unified Memory)"
    except Exception:
        stats["gpu_info"] = None

    return stats


def render_resource_sidebar(st, agent=None):
    """
    Render the resource monitoring sidebar section.

    Args:
        st:    The Streamlit module
        agent: Optional EmotionalAgent for session stats
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 System Resources")

    stats = get_resource_stats()

    # CPU
    cpu_color = (
        "🟢"
        if stats["cpu_percent"] < 70
        else "🟡" if stats["cpu_percent"] < 90 else "🔴"
    )
    st.sidebar.metric(f"{cpu_color} CPU Usage", f"{stats['cpu_percent']:.0f}%")

    # RAM
    ram_color = (
        "🟢"
        if stats["ram_percent"] < 70
        else "🟡" if stats["ram_percent"] < 85 else "🔴"
    )
    st.sidebar.metric(
        f"{ram_color} RAM",
        f"{stats['ram_used_gb']:.1f} / {stats['ram_total_gb']:.0f} GB",
        f"{stats['ram_percent']:.0f}% used",
    )

    # GPU info
    if stats.get("gpu_info"):
        st.sidebar.markdown(f"🖥️ **GPU:** {stats['gpu_info']}")

    # Session stats
    if agent and agent.is_ready:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 💬 Session Stats")
        session_stats = agent.get_session_stats()
        st.sidebar.metric("Turns", session_stats.get("turn_count", 0))
