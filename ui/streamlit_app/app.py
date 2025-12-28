"""
Streamlit Chat Interface for Internal Knowledge Base RAG Chatbot

A clean, professional chat interface with source citations and conversation history.
"""

import streamlit as st
import requests
from typing import List, Dict
import time

# Page configuration
st.set_page_config(
    page_title="Internal Knowledge Base QA",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Load custom CSS
def load_css():
    """Load custom CSS for better styling"""
    with open("ui/streamlit_app/styles/custom.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_url" not in st.session_state:
    st.session_state.api_url = "http://localhost:8000"


def query_api(question: str, api_url: str) -> Dict:
    """
    Send question to FastAPI backend and get response with sources.

    Args:
        question: User's question
        api_url: API endpoint URL

    Returns:
        Response dict with answer and sources
    """
    try:
        response = requests.post(
            f"{api_url}/api/query", json={"question": question}, timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {
            "answer": f"Error connecting to API: {str(e)}",
            "sources": [],
            "error": True,
        }


def display_message(role: str, content: str, sources: List[Dict] = None):
    """
    Display a chat message with optional sources.

    Args:
        role: 'user' or 'assistant'
        content: Message content
        sources: List of source documents (for assistant messages)
    """
    with st.chat_message(role):
        st.markdown(content)

        # Display sources if available
        if sources and role == "assistant":
            with st.expander(f"📚 Sources ({len(sources)})"):
                for idx, source in enumerate(sources, 1):
                    st.markdown(
                        f"""
                    **Source {idx}**: `{source.get('source', 'Unknown')}`
                    **Relevance**: {source.get('score', 0):.2%}

                    {source.get('content', 'No content available')[:300]}...
                    """
                    )
                    st.divider()


def main():
    """Main application logic"""

    # Load custom styles
    try:
        load_css()
    except FileNotFoundError:
        pass  # CSS file optional

    # Sidebar
    with st.sidebar:
        st.title("⚙️ Configuration")

        # API URL configuration
        api_url = st.text_input(
            "API URL", value=st.session_state.api_url, help="Backend API endpoint"
        )
        st.session_state.api_url = api_url

        # Connection status
        try:
            health_check = requests.get(f"{api_url}/health", timeout=3)
            if health_check.status_code == 200:
                st.success("✅ Connected to API")
            else:
                st.error("❌ API connection failed")
        except:
            st.error("❌ Cannot reach API")

        st.divider()

        # Conversation controls
        st.subheader("💬 Conversation")

        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        # Display message count
        st.metric("Messages", len(st.session_state.messages))

        st.divider()

        # Info section
        with st.expander("ℹ️ About"):
            st.markdown(
                """
            **Internal Knowledge Base QA**

            Ask questions about your internal documents and get accurate answers with source citations.

            **Features:**
            - Context-aware answers
            - Source verification
            - Conversation history
            - Fast retrieval
            """
            )

    # Main chat interface
    st.title("🤖 Internal Knowledge Base QA")
    st.markdown("Ask questions about your internal documentation")

    # Display chat history
    for message in st.session_state.messages:
        display_message(message["role"], message["content"], message.get("sources"))

    # Chat input
    if prompt := st.chat_input("Ask a question about your knowledge base..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        display_message("user", prompt)

        # Get AI response with loading indicator
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                response = query_api(prompt, st.session_state.api_url)

            # Display response
            answer = response.get("answer", "No answer received")
            sources = response.get("sources", [])

            st.markdown(answer)

            # Display sources
            if sources:
                with st.expander(f"📚 Sources ({len(sources)})"):
                    for idx, source in enumerate(sources, 1):
                        st.markdown(
                            f"""
                        **Source {idx}**: `{source.get('source', 'Unknown')}`
                        **Relevance**: {source.get('score', 0):.2%}

                        {source.get('content', 'No content available')[:300]}...
                        """
                        )
                        st.divider()

        # Add assistant message to history
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )


if __name__ == "__main__":
    main()
