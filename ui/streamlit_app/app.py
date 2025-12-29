"""
Main Streamlit application for GitLab Knowledge Q&A Bot.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ui.streamlit_app.components.sidebar import render_sidebar
from ui.streamlit_app.components.chat import render_chat

# Page configuration
st.set_page_config(
    page_title="GitLab Knowledge Q&A Bot",
    page_icon="🦊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject comprehensive CSS for fixed header and styling
st.markdown(
    """
    <style>
    /* Remove all default margins and padding */
    .stApp {
        margin-top: 56px;
        padding: 0;
    }

    /* Force header to be full width at document level */
    .fixed-header {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        width: 100vw !important;
        background: linear-gradient(90deg, #FC6D26 0%, #FCA326 100%);
        padding: 15px 20px 15px 60px;
        z-index: 999999 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
        display: flex;
        align-items: center;
        gap: 12px;
        height: 56px;
        margin: 0 !important;
    }

    .fixed-header .header-icon {
        font-size: 24px;
        line-height: 1;
    }

    .fixed-header .header-title {
        color: white;
        font-size: 20px;
        font-weight: 600;
        margin: 0;
        line-height: 1;
    }

    /* Remove ALL default padding from main container */
    .main .block-container {
        padding-top: 0.1rem !important;
        padding-bottom: 5rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
        margin-top: 0 !important;
    }

    /* Adjust sidebar to start RIGHT below header with NO gap */
    section[data-testid="stSidebar"] {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }

    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
        margin-top: 0 !important;
    }

    /* Sidebar toggle buttons */
    button[kind="header"],
    [data-testid="collapsedControl"] {
        top: 8px !important;
        left: 8px !important;
        z-index: 9999999 !important;
    }

    /* Remove top spacing from chat area */
    .main {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }

    /* Remove spacing from first element in chat */
    .main > div:first-child {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }

    /* Remove vertical block spacing */
    div[data-testid="stVerticalBlock"] {
        padding-top: 0 !important;
        gap: 0.5rem !important;
    }

    div[data-testid="stVerticalBlock"] > div:first-child {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }

    /* Chat messages - reduce top spacing */
    .stChatMessage:first-of-type {
        margin-top: 0 !important;
    }

    .stChatMessage {
        margin-bottom: 1rem;
        margin-top: 0.5rem;
    }

    /* Chat input - add more space at bottom */
    .stChatInput {
        padding: 1.5rem 0 2rem 0 !important;
        background: white;
        margin-bottom: 0 !important;
        margin-top: 1rem !important;
    }

    /* Add padding to chat input container */
    .stChatInput > div {
        padding-bottom: 1rem !important;
    }

    /* Custom scrollbar */
    .main::-webkit-scrollbar {
        width: 8px;
    }

    .main::-webkit-scrollbar-track {
        background: #f1f1f1;
    }

    .main::-webkit-scrollbar-thumb {
        background: #FC6D26;
        border-radius: 4px;
    }

    .main::-webkit-scrollbar-thumb:hover {
        background: #e55a1a;
    }
    .st-emotion-cache-liupih {
        width: 100%;
        padding: 4rem 1rem 1rem;
        max-width: initial;
        min-width: auto;
    }

    /* Chat input wrapper */
    .stChatInput {
        padding: 1.5rem 0 2rem 0 !important;
        background: white;
        margin-bottom: 0 !important;
        margin-top: 1rem !important;
    }

    /* Force flexbox layout for input container */
    .stChatInput > div > div {
        display: flex !important;
        align-items: center !important;
        gap: 0.75rem !important;
    }

    /* Textarea container */
    .stChatInput [data-testid="stChatInputTextArea"] {
        flex: 1;
        display: flex;
        align-items: center;
    }

    /* Textarea itself */
    .stChatInput textarea {
        min-height: 48px !important;
        max-height: 120px !important;
        padding: 12px 16px !important;
        line-height: 24px !important;
        border-radius: 24px !important;
        font-size: 16px !important;
    }

    .stChatInput textarea:focus {
        border-color: #FC6D26 !important;
        box-shadow: 0 0 0 3px rgba(252, 109, 38, 0.1) !important;
    }

    /* Send button container */
    .stChatInput button[kind="primary"] {
        height: 48px !important;
        min-height: 48px !important;
        width: 48px !important;
        min-width: 48px !important;
        border-radius: 24px !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin: 0 !important;
        align-self: flex-end !important;
    }

    /* Send button icon */
    .stChatInput button[kind="primary"] svg {
        width: 20px !important;
        height: 20px !important;
    }

    /* Remove default margins */
    .stChatInput form {
        margin: 0 !important;
    }

    .stChatInput > div {
        padding-bottom: 0 !important;
        margin-bottom: 0 !important;
    }
    </style>

    <div class="fixed-header">
        <span class="header-icon">🦊</span>
        <span class="header-title">GitLab Knowledge Q&A</span>
    </div>
    """,
    unsafe_allow_html=True,
)


def main():
    """Main application entry point."""
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "api_url" not in st.session_state:
        st.session_state.api_url = "http://localhost:8000"
    if "max_sources" not in st.session_state:
        st.session_state.max_sources = 5
    if "show_debug" not in st.session_state:
        st.session_state.show_debug = False

    # Render sidebar and get config
    config = render_sidebar()

    # Update session state with config values
    st.session_state.max_sources = config.get("max_sources", 5)
    st.session_state.show_debug = config.get("show_debug", False)

    # Render chat interface
    render_chat(config)


if __name__ == "__main__":
    main()
