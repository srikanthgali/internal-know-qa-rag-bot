"""
Sidebar components for the Streamlit UI.

Configuration, settings, and controls.
"""

import streamlit as st
import requests


def render_api_status(api_url: str) -> None:
    """
    Display API connection status indicator.

    Args:
        api_url: API endpoint URL
    """
    try:
        response = requests.get(f"{api_url}/health", timeout=3)
        if response.status_code == 200:
            st.success("✅ API Online")
        else:
            st.warning(f"⚠️ API Status: {response.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API")
    except requests.exceptions.Timeout:
        st.error("❌ API timeout")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")


def render_conversation_controls() -> bool:
    """
    Render conversation management controls.

    Returns:
        True if conversation should be cleared
    """
    st.subheader("💬 Conversation")

    col1, col2 = st.columns(2)

    with col1:
        clear = st.button("🗑️ Clear", use_container_width=True)

    with col2:
        # Future: export conversation
        st.button("💾 Export", use_container_width=True, disabled=True)

    return clear


def render_settings() -> Dict:
    """
    Render application settings.

    Returns:
        Dict of current settings
    """
    st.subheader("⚙️ Settings")

    settings = {}

    # Temperature setting (for future LLM control)
    settings["temperature"] = st.slider(
        "Response Creativity",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Higher values make responses more creative",
    )

    # Max sources
    settings["max_sources"] = st.number_input(
        "Max Sources",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of source documents to display",
    )

    return settings
