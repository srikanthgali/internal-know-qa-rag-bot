from typing import List, Dict, Optional
from openai import OpenAI

from src.config.prompts import SYSTEM_PROMPT
from src.retrieval.retriever import Retriever
from src.generation.prompt_builder import PromptBuilder
from src.utils.logger import get_logger
from src.config.settings import settings

logger = get_logger(__name__)


class RAGPipeline:
    """End-to-end RAG pipeline for question answering."""

    def __init__(self):
        """Initialize RAG pipeline components."""
        self.retriever = Retriever()
        self.prompt_builder = PromptBuilder()
        self.client = OpenAI(api_key=settings.openai.api_key)
        self.model = settings.openai.model
        self.temperature = settings.openai.temperature
        self.max_tokens = settings.openai.max_tokens
        self.edge_case_min_score = settings.retrieval.edge_case_min_score

        # Greeting patterns
        self.greeting_patterns = [
            "hi",
            "hello",
            "hey",
            "greetings",
            "good morning",
            "good afternoon",
            "good evening",
            "howdy",
            "what's up",
            "whats up",
            "sup",
            "yo",
        ]

        self.intro_patterns = [
            "who are you",
            "what are you",
            "what can you do",
            "what do you do",
            "tell me about yourself",
            "introduce yourself",
            "your capabilities",
            "help me",
            "what is this",
            "how does this work",
        ]

        logger.info("Initialized RAG Pipeline")

    def _is_greeting(self, question: str) -> bool:
        """Check if question is a greeting."""
        question_lower = question.lower().strip()

        # Exact match or starts with greeting
        return question_lower in self.greeting_patterns or any(
            question_lower.startswith(g) for g in self.greeting_patterns
        )

    def _is_intro_request(self, question: str) -> bool:
        """Check if question is asking for introduction/help."""
        question_lower = question.lower().strip()
        return any(pattern in question_lower for pattern in self.intro_patterns)

    def _generate_greeting_response(self) -> str:
        """Generate friendly greeting response."""
        return """Hello! 👋 I'm the GitLab Knowledge Q&A Assistant.

                I can help you find information about:
                • GitLab's mission, values, and culture
                • Company policies and procedures
                • Time off, leave types, and benefits
                • Remote work practices
                • Training and professional development
                • And much more from the GitLab handbook!

                What would you like to know about GitLab today?"""

    def _generate_intro_response(self) -> str:
        """Generate introduction/capabilities response."""
        return """I'm an AI assistant specialized in GitLab's internal knowledge base! 🦊

                **What I can do:**
                • Answer questions about GitLab policies, procedures, and culture
                • Explain company values and operating principles
                • Guide you through HR processes (PTO, benefits, etc.)
                • Share information about remote work best practices
                • Provide training and development resources

                **How to use me:**
                • Ask specific questions like "How do I request time off?"
                • Inquire about policies like "What is GitLab's mission?"
                • Seek guidance on processes like "How does code review work?"

                **What I can't do:**
                • Answer questions outside GitLab's knowledge base
                • Provide personal advice or opinions
                • Access external information or real-time data

                Try asking me something about GitLab! For example: "What is GitLab's approach to remote work?" """

    def query(
        self,
        question: str,
        chat_history: Optional[List[Dict]] = None,
        top_k: Optional[int] = None,
    ) -> Dict:
        """
        Process a question through the RAG pipeline.

        Args:
            question: User question
            chat_history: Previous chat messages
            top_k: Number of documents to retrieve

        Returns:
            Dictionary containing answer, sources, and metadata
        """
        logger.info(f"Processing query: {question}")

        try:
            # ADDED: Handle greetings
            if self._is_greeting(question):
                logger.info("Detected greeting, returning friendly response")
                return {
                    "answer": self._generate_greeting_response(),
                    "sources": [],
                    "retrieved_docs": [],
                    "model": self.model,
                    "is_greeting": True,
                }

            # Handle introduction requests
            if self._is_intro_request(question):
                logger.info("Detected intro request, returning capabilities")
                return {
                    "answer": self._generate_intro_response(),
                    "sources": [],
                    "retrieved_docs": [],
                    "model": self.model,
                    "is_intro": True,
                }

            # Step 1: Retrieve relevant documents
            retrieved_docs = self.retriever.retrieve(question, top_k=top_k)

            # IMPROVED: Check if retrieval quality is too low
            if not retrieved_docs:
                logger.warning("No relevant documents found")
                return {
                    "answer": "I don't have enough information in the knowledge base to answer this question. Could you try rephrasing or ask something else about GitLab?",
                    "sources": [],
                    "retrieved_docs": [],
                    "model": self.model,
                }

            # Additional check: if best score is below threshold
            best_score = max(doc.get("score", 0.0) for doc in retrieved_docs)
            if best_score < self.edge_case_min_score:
                logger.warning(
                    f"Retrieved docs have low relevance (best: {best_score:.2%})"
                )
                return {
                    "answer": "I don't have enough information in the knowledge base to answer this question. Could you try rephrasing or ask something else about GitLab?",
                    "sources": [],
                    "retrieved_docs": [],
                    "model": self.model,
                }

            logger.info(f"Retrieved {len(retrieved_docs)} documents")

            # Step 2: Build prompt
            prompt = self.prompt_builder.build_prompt(
                question=question,
                retrieved_docs=retrieved_docs,
                chat_history=chat_history,
            )

            # Step 3: Generate answer using LLM
            logger.info("Generating answer...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": settings.get("generation.system_prompt"),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            answer = response.choices[0].message.content

            # Step 4: Extract sources
            sources = self._extract_sources(retrieved_docs)

            # Step 5: Check faithfulness
            if self._answer_seems_unfaithful(answer, retrieved_docs):
                logger.warning("Generated answer may be unfaithful")
                logger.info("Note: Answer synthesis detected - monitoring for quality")

            logger.info("Query processed successfully")

            return {
                "answer": answer,
                "sources": sources,
                "retrieved_docs": retrieved_docs,
                "model": self.model,
            }

        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            raise

    def _extract_sources(self, retrieved_docs: List[Dict]) -> List[Dict]:
        """
        Extract and format source information.

        Args:
            retrieved_docs: List of retrieved documents

        Returns:
            List of source information dictionaries
        """
        sources = []
        seen_sources = set()

        for doc in retrieved_docs:
            metadata = doc["metadata"]
            source_id = metadata.get("filename", "Unknown")

            if source_id not in seen_sources:
                sources.append(
                    {
                        "filename": metadata.get("filename", "Unknown"),
                        "source": metadata.get("source", ""),
                        "file_type": metadata.get("file_type", ""),
                        "relevance_score": doc["score"],
                    }
                )
                seen_sources.add(source_id)

        return sources

    def stream_query(
        self,
        question: str,
        chat_history: Optional[List[Dict]] = None,
        top_k: Optional[int] = None,
    ):
        """
        Process query with streaming response.

        Args:
            question: User question
            chat_history: Previous chat messages
            top_k: Number of documents to retrieve

        Yields:
            Chunks of generated text
        """
        logger.info(f"Processing streaming query: {question}")

        try:
            # Retrieve documents
            retrieved_docs = self.retriever.retrieve(question, top_k=top_k)

            if not retrieved_docs:
                yield "I'm sorry, I couldn't find any relevant information to answer your question."
                return

            # Build prompt
            prompt = self.prompt_builder.build_prompt(
                question=question,
                retrieved_docs=retrieved_docs,
                chat_history=chat_history,
            )

            # Stream response
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": settings.get("generation.system_prompt"),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Error in streaming query: {e}")
            yield f"Error: {str(e)}"

    def _answer_seems_unfaithful(self, answer: str, retrieved_docs: List[Dict]) -> bool:
        """
        Check if answer contains information not in context.

        Simple heuristic: Check for common hallucination phrases.
        """
        # CRITICAL HALLUCINATION INDICATORS ONLY
        hallucination_indicators = [
            "it is widely known",
            "according to common knowledge",
            "in my experience",
            "as we all know",
            "everyone knows",
            "i think",
            "i believe",
            "in my opinion",
        ]

        answer_lower = answer.lower()

        # Check for clear hallucination phrases
        for indicator in hallucination_indicators:
            if indicator in answer_lower:
                logger.warning(f"Hallucination phrase detected: '{indicator}'")
                return True

        return False

    def _build_strict_prompt(self, question: str, retrieved_docs: List[Dict]) -> str:
        """Build extremely strict prompt for regeneration."""
        context = self.prompt_builder._format_context_enhanced(retrieved_docs)

        strict_template = """Context (USE ONLY THIS):
{context}

Question: {question}

CRITICAL: Your answer MUST:
1. Use ONLY information from the context above
2. Quote directly from context
3. Be shorter than the context
4. Say "I don't have enough information" if answer isn't clearly in context

Answer:"""

        return strict_template.format(context=context, question=question)

    def _generate_with_retry(self, prompt: str) -> str:
        """Generate answer with lower temperature."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,  # Most deterministic
            max_tokens=500,  # Shorter
        )

        return response.choices[0].message.content.strip()
