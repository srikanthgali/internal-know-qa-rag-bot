"""Prompt templates for RAG chatbot."""

SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on internal company documentation.

Your role is to:
1. Provide accurate, concise answers based ONLY on the provided context
2. Cite specific sources when answering
3. Admit when you don't have enough information to answer
4. Be professional and helpful

Guidelines:
- Always base your answers on the provided context
- If the context doesn't contain the answer, say so clearly
- Include relevant source references in your answer
- Be conversational but maintain professionalism
- Keep answers focused and to the point
"""

QUERY_PROMPT_TEMPLATE = """Context information is below:
---------------------
{context}
---------------------

Given the context information above, please answer the following question.
If you cannot answer the question based on the context, say "I don't have enough information to answer this question."

Always cite the sources you used (by filename) when providing an answer.

Question: {question}

Answer: """

CHAT_PROMPT_TEMPLATE = """Context information is below:
---------------------
{context}
---------------------

Chat History:
{chat_history}

Given the context information and chat history above, please answer the following question.
If you cannot answer based on the context, say so clearly.

Question: {question}

Answer: """

CONDENSED_QUESTION_TEMPLATE = """Given the following conversation history and a follow-up question,
rephrase the follow-up question to be a standalone question that captures all relevant context.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question: """


def format_context(retrieved_docs: list) -> str:
    """
    Format retrieved documents into context string.

    Args:
        retrieved_docs: List of retrieved document chunks

    Returns:
        Formatted context string
    """
    context_parts = []

    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.get("metadata", {}).get("filename", "Unknown")
        content = doc.get("content", "")
        score = doc.get("score", 0.0)

        context_parts.append(
            f"[Source {i}: {source} (relevance: {score:.2f})]\n{content}\n"
        )

    return "\n".join(context_parts)


def format_chat_history(chat_history: list) -> str:
    """
    Format chat history into string.

    Args:
        chat_history: List of chat messages

    Returns:
        Formatted chat history string
    """
    if not chat_history:
        return "No previous conversation."

    history_parts = []
    for msg in chat_history[-5:]:  # Last 5 messages
        role = msg.get("role", "user")
        content = msg.get("content", "")
        history_parts.append(f"{role.capitalize()}: {content}")

    return "\n".join(history_parts)
