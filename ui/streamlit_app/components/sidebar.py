"""
Sidebar component for the Streamlit app.
"""

import streamlit as st
import requests
from typing import Dict, Any


def render_sidebar() -> Dict[str, Any]:
    """
    Render the sidebar with configuration options.

    Returns:
        Dictionary with configuration settings
    """
    with st.sidebar:
        st.title("⚙️ Settings")

        # API Connection Status (compact)
        api_url = st.session_state.get("api_url", "http://localhost:8000")

        # Check API health
        try:
            response = requests.get(f"{api_url}/health", timeout=2)
            if response.status_code == 200:
                st.success("✅ Connected to API")
            else:
                st.error("⚠️ API Error")
        except requests.exceptions.RequestException:
            st.error("❌ API Unavailable")

        st.divider()

        # Query Settings
        st.subheader("🔍 Query Settings")

        max_sources = st.slider(
            "Maximum Sources",
            min_value=1,
            max_value=10,
            value=st.session_state.get("max_sources", 5),
            help="Number of source documents to retrieve",
        )

        st.divider()

        # Model Information
        st.subheader("🤖 Model Info")
        st.info(
            """
            **LLM**: GPT-4.1 mini
            **Embeddings**: Ada-002
            **Vector Store**: FAISS
            """,
            icon="ℹ️",
        )

        st.divider()

        # About Section
        st.subheader("ℹ️ About")
        st.markdown(
            """
            **GitLab Knowledge Q&A Bot**

            Ask questions about GitLab company, values, policies,
            processes, and best practices.

            Built with:
            - 🦙 LlamaIndex
            - 🤖 OpenAI GPT-4.1 mini
            - 🔍 FAISS Vector Search
            - 🎈 Streamlit
            """
        )

        # Advanced Settings (collapsible)
        with st.expander("🔧 Advanced"):
            show_debug = st.checkbox(
                "Show Debug Info",
                value=st.session_state.get("show_debug", False),
                help="Display additional debugging information",
            )

            # Only show API URL in advanced/debug mode
            if show_debug:
                st.text_input(
                    "API URL",
                    value=api_url,
                    disabled=True,
                    help="API endpoint (read-only)",
                )

        # Clear Chat Button
        st.divider()
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        return {
            "max_sources": max_sources,
            "show_debug": show_debug,
        }
