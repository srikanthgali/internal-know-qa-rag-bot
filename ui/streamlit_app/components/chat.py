"""
Chat interface component for the Streamlit app.
"""

import streamlit as st
import requests
from typing import Dict, List, Any


def display_message(
    role: str, content: str, sources: List[Dict] = None, show_debug: bool = False
):
    """Display a chat message with optional sources."""
    with st.chat_message(role):
        st.markdown(content)

        # Display sources if available (not for greetings)
        if sources and role == "assistant":
            if len(sources) > 0:
                with st.expander(f"📚 Sources ({len(sources)})"):
                    for idx, source in enumerate(sources, 1):
                        score = source.get("score", 0)
                        st.markdown(
                            f"""
                        **Source {idx}**: `{source.get('source', 'Unknown')}`
                        **Relevance**: {score:.2%}
                        """
                        )
                        # Show debug info if enabled
                        if show_debug:
                            st.json(source.get("metadata", {}))
            elif content.startswith("I don't have enough information"):
                # Out-of-scope query
                st.info("ℹ️ No relevant sources found in the knowledge base.")
            # ADDED: Don't show "no sources" for greetings
            elif not any(
                greeting in content.lower()
                for greeting in ["hello", "hi", "i'm", "i can help"]
            ):
                st.info("ℹ️ No relevant sources found in the knowledge base.")


def render_chat(config: Dict[str, Any]):
    """
    Render the chat interface.

    Args:
        config: Configuration dictionary from sidebar
    """
    # Get API URL from session state
    api_url = st.session_state.get("api_url", "http://localhost:8000")
    max_sources = config.get("max_sources", 5)
    show_debug = config.get("show_debug", False)

    # Display chat messages in scrollable container
    for message in st.session_state.messages:
        display_message(
            message["role"],
            message["content"],
            message.get("sources"),
            show_debug=show_debug,
        )

    # Chat input (will be fixed at bottom via CSS)
    if prompt := st.chat_input("Ask a question about GitLab..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)

        # Query the API
        with st.spinner("🤔 Thinking..."):
            try:
                response = requests.post(
                    f"{api_url}/api/query",
                    json={"question": prompt, "max_sources": max_sources},
                    timeout=30,
                )

                if response.status_code == 200:
                    result = response.json()
                    answer = result.get("answer", "No answer generated")
                    sources = result.get("sources", [])

                    # Add assistant message
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )

                    # Display assistant response immediately
                    with st.chat_message("assistant"):
                        st.markdown(answer)

                        # Display sources if available
                        if sources:
                            if len(sources) > 0:
                                with st.expander(f"📚 Sources ({len(sources)})"):
                                    for idx, source in enumerate(sources, 1):
                                        score = source.get("score", 0)
                                        st.markdown(
                                            f"""
                                        **Source {idx}**: `{source.get('source', 'Unknown')}`
                                        **Relevance**: {score:.2%}
                                        """
                                        )
                                        if show_debug:
                                            st.json(source.get("metadata", {}))
                            elif answer.startswith("I don't have enough information"):
                                st.info(
                                    "ℹ️ No relevant sources found in the knowledge base."
                                )
                            elif not any(
                                greeting in answer.lower()
                                for greeting in ["hello", "hi", "i'm", "i can help"]
                            ):
                                st.info(
                                    "ℹ️ No relevant sources found in the knowledge base."
                                )

                    # Show debug info if enabled
                    if show_debug:
                        with st.expander("🔍 Debug Info"):
                            st.json(
                                {
                                    "query_time": result.get("query_time"),
                                    "model": result.get("model", "unknown"),
                                    "num_sources": len(sources),
                                    "max_score": (
                                        max([s.get("score", 0) for s in sources])
                                        if sources
                                        else 0
                                    ),
                                }
                            )

                    # Force rerun to update the full chat history
                    st.rerun()

                else:
                    st.error(f"❌ API Error: {response.status_code}")
                    st.json(response.json())

            except requests.exceptions.ConnectionError:
                st.error(
                    """
                    ❌ **Could not connect to API**

                    Please ensure the API is running:
                    ```bash
                    python run_api.py
                    ```
                    """
                )
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. Please try again.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                if show_debug:
                    st.exception(e)
