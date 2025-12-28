"""
Chat components for the Streamlit UI.

Reusable chat-related UI components.
"""

import streamlit as st
from typing import List, Dict


def render_source_card(source: Dict, index: int) -> None:
    """
    Render a single source card with document info.

    Args:
        source: Source document dict with content, score, metadata
        index: Source number for display
    """
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(f"**📄 Source {index}**")
        st.caption(source.get("source", "Unknown source"))

    with col2:
        score = source.get("score", 0)
        st.metric("Relevance", f"{score:.0%}")

    # Content preview
    content = source.get("content", "No content available")
    st.text_area(
        "Preview",
        value=content[:400] + ("..." if len(content) > 400 else ""),
        height=100,
        disabled=True,
        label_visibility="collapsed",
    )


def render_message_with_sources(content: str, sources: List[Dict]) -> None:
    """
    Render assistant message with expandable sources.

    Args:
        content: Message content
        sources: List of source documents
    """
    st.markdown(content)

    if sources:
        st.markdown("---")
        with st.expander(f"📚 View {len(sources)} source(s)", expanded=False):
            for idx, source in enumerate(sources, 1):
                render_source_card(source, idx)
                if idx < len(sources):
                    st.divider()
